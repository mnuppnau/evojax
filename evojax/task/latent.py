# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import numpy as np
from typing import List, Tuple

import jax
import optax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.struct import dataclass

from functools import partial
from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

import torch
import torchvision
import torchvision.transforms as T

@dataclass
class State(TaskState):
    obs: jnp.ndarray
    #latent_input: jnp.ndarray
    noise: jnp.ndarray
    shift_x: jnp.ndarray
    shift_y: jnp.ndarray
    cat_codes: jnp.ndarray
    codes60: jnp.ndarray
    real_centroids: jnp.ndarray
    con_codes: jnp.ndarray
    batch_stats_gen: any
    batch_stats_disc: any
    batch_stats_q: any

def rff_init(
    key: jax.random.PRNGKey,
    d_in: int,
    D_total: int = 2048,
    sigmas: Tuple[float, ...] = (1.5, 3.0, 6.0),
):
    """
    Initialize multi-bandwidth RFF parameters.
    """
    sigmas = jnp.array(sigmas, dtype=jnp.float32)
    n_groups = sigmas.shape[0]

    sizes = jnp.full((n_groups,), D_total // n_groups)
    sizes = sizes.at[0].add(D_total - sizes.sum())

    keys = jax.random.split(key, 2 * n_groups)
    Ws, bs = [], []

    for i, (sigma, Dg) in enumerate(zip(sigmas, sizes)):
        Wg = jax.random.normal(keys[2 * i], (Dg, d_in)) / (sigma + 1e-8)
        bg = jax.random.uniform(
            keys[2 * i + 1], (Dg,), minval=0.0, maxval=2 * jnp.pi
        )
        Ws.append(Wg)
        bs.append(bg)

    W = jnp.concatenate(Ws, axis=0)
    b = jnp.concatenate(bs, axis=0)
    scale = jnp.sqrt(2.0 / D_total)

    return {"W": W, "b": b, "scale": scale}


@jax.jit
def rff_embed(params, X):
    """
    X: [N, d_in]
    returns: [N, D]
    """
    return params["scale"] * jnp.cos(X @ params["W"].T + params["b"])


@jax.jit
def mean_embedding(params, X):
    """
    Mean RFF embedding μ = E[ψ(X)]
    """
    return rff_embed(params, X).mean(axis=0)

def compute_real_class_means(
    rff_params,
    real_by_class: List[jnp.ndarray],
):
    """
    real_by_class: list of 10 arrays, each [Nc,28,28] or [Nc,784]
    returns: [10, D] array of class mean embeddings
    """
    mus = []

    for Xc in real_by_class:
        if Xc.ndim == 3:  # [Nc,28,28]
            Xc = Xc.reshape((Xc.shape[0], -1))
        Xc = Xc.astype(jnp.float32)
        mus.append(mean_embedding(rff_params, Xc))

    return jnp.stack(mus, axis=0)

@partial(jax.jit, static_argnums=(1,))
def compute_fake_centroids(fake_features: jnp.ndarray, n_codes: int = 10) -> jnp.ndarray:
    """
    Compute centroids from interleaved fake features.
    
    Features are interleaved by code:
        - Indices 0, 10, 20, 30, ... -> code 0
        - Indices 1, 11, 21, 31, ... -> code 1
        - etc.
    
    Args:
        fake_features: (batch_size, feature_dim) e.g. (128, 256)
        n_codes: Number of codes (default 10)
    
    Returns:
        (n_codes, feature_dim) e.g. (10, 256)
    """
    batch_size = fake_features.shape[0]
    indices = jnp.arange(batch_size)
    
    def compute_centroid_for_code(code_idx: int) -> jnp.ndarray:
        mask = (indices % n_codes) == code_idx
        masked_sum = jnp.sum(jnp.where(mask[:, None], fake_features, 0.0), axis=0)
        count = jnp.sum(mask)
        return masked_sum / count
    
    return jax.vmap(compute_centroid_for_code)(jnp.arange(n_codes))

@partial(jax.jit, static_argnums=(2,))
def compute_centroid_loss(
    fake_features: jnp.ndarray,
    target_centroids: jnp.ndarray,
    n_codes: int = 10
) -> jnp.ndarray:
    # Truncate to largest multiple of n_codes (64 -> 60)
    batch_size = fake_features.shape[0]
    usable_size = (batch_size // n_codes) * n_codes
    fake_features = fake_features[:usable_size]
    
    # Compute fake centroids: (n_codes, feature_dim)
    fake_centroids = compute_fake_centroids(fake_features, n_codes)
    
    # MSE across all elements (codes and dimensions)
    diff = fake_centroids - target_centroids
    return jnp.mean(diff ** 2)  # scalar MSE

@partial(jax.jit, static_argnums=(2,))
def compute_hybrid_anchor_loss(
    fake_features: jnp.ndarray,
    target_centroids: jnp.ndarray,
    n_codes: int = 10
) -> jnp.ndarray:
    
    # ... (slicing and centroid computation as before) ...
    # Truncate to largest multiple of n_codes (64 -> 60)
    batch_size = fake_features.shape[0]
    usable_size = (batch_size // n_codes) * n_codes
    fake_features = fake_features[:usable_size]
    
    # Compute fake centroids: (n_codes, feature_dim)
    fake_centroids = compute_fake_centroids(fake_features, n_codes)
    # 1. Compute Cosine Distance for EACH code separately
    # Result shape: (n_codes,)
    fake_norm = fake_centroids / (jnp.linalg.norm(fake_centroids, axis=1, keepdims=True) + 1e-8)
    target_norm = target_centroids / (jnp.linalg.norm(target_centroids, axis=1, keepdims=True) + 1e-8)
    
    # Similarity per code
    similarity = jnp.sum(fake_norm * target_norm, axis=1)
    individual_losses = 1.0 - similarity
    
    # 2. Aggregation: Mean + Max
    # "avg_loss" keeps the general population moving
    # "max_loss" screams if one code is left behind
    avg_loss = jnp.mean(individual_losses)
    max_loss = jnp.max(individual_losses)
    
    # Weigh the max loss heavily (e.g., 50/50 split influence)
    return avg_loss + max_loss

@partial(jax.jit, static_argnums=(2,))
def compute_cosine_anchor_loss(fake_features, target_centroids, n_codes=10):
    # 1. Compute Centroids as before
    batch_size = fake_features.shape[0]
    usable_size = (batch_size // n_codes) * n_codes
    fake_features = fake_features[:usable_size]
    fake_centroids = compute_fake_centroids(fake_features, n_codes)
    
    # 2. Normalize both sets of vectors to unit length
    # epsilon prevents division by zero
    fake_norm = fake_centroids / (jnp.linalg.norm(fake_centroids, axis=1, keepdims=True) + 1e-8)
    target_norm = target_centroids / (jnp.linalg.norm(target_centroids, axis=1, keepdims=True) + 1e-8)
    
    # 3. Compute Cosine Similarity (Dot product of normalized vectors)
    # Result is between -1 (opposite) and 1 (identical)
    similarity = jnp.sum(fake_norm * target_norm, axis=1)
    
    # 4. Convert to Loss (We want to minimize this)
    # Loss = 1 - similarity. Range: [0, 2]
    return jnp.mean(1.0 - similarity)

@jax.jit
def class_similarity_weights(
    mu_fake: jnp.ndarray,     # [D]
    mu_classes: jnp.ndarray,  # [10, D]
):
    """
    Returns:
      d2      : [10]   MMD^2-like distances
      weights : [10]   soft similarity weights
      entropy : scalar
    """
    diffs = mu_classes - mu_fake[None, :]
    d2 = jnp.sum(diffs * diffs, axis=1)

    # temperature scaling (relative, not absolute)
    alpha = 10.0 / jnp.clip(jnp.mean(d2), 1e-8, None)

    weights = jax.nn.softmax(-alpha * d2)
    entropy = -jnp.sum(weights * jnp.log(jnp.clip(weights, 1e-12, 1.0)))

    return d2, weights, entropy

@jax.jit
def evaluate_fake_batch(
    fake_images: jnp.ndarray,   # [N,28,28] or [N,784]
    mu_classes: jnp.ndarray,    # [10,D]
    rff_params,
):
    """
    Runs the metric on a batch of fake images.
    """
    fake_images = fake_images.squeeze()  # (64, 28, 28, 1) -> (64, 28, 28)
    fake_images = fake_images.reshape((fake_images.shape[0], -1))  # -> (64, 784)

    fake_images = fake_images.astype(jnp.float32)

    mu_fake = mean_embedding(rff_params, fake_images)
    d2, weights, entropy = class_similarity_weights(mu_fake, mu_classes)

    pred_class = jnp.argmin(d2).astype(jnp.int32)
    margin = (jnp.sort(d2)[1] - jnp.sort(d2)[0]).astype(jnp.float32)

    return {
        "mmd2_per_class": d2,
        "weights": weights,
        "entropy": entropy.astype(jnp.float32),
        "predicted_class": pred_class,
        "margin": margin,
        "num_images": fake_images.shape[0],
    }

def build_real_by_class_mnist(
    root="./data",
    train=True,
    download=True,
):
    """
    Returns:
        real_by_class: list of 10 JAX arrays
                       real_by_class[c].shape == (Nc, 28, 28)
    """
    transform = T.Compose([
        T.ToTensor(),  # -> [1,28,28], float32 in [0,1]
    ])

    dataset = torchvision.datasets.MNIST(
        root=root,
        train=train,
        transform=transform,
        download=download,
    )

    # Buckets for each class
    buckets = [[] for _ in range(10)]

    for img, label in dataset:
        # img: torch tensor [1,28,28]
        img_np = img.squeeze(0).numpy()  # -> [28,28]
        buckets[label].append(img_np)

    # Stack and convert to JAX arrays
    real_by_class = [
        jnp.asarray(np.stack(b, axis=0), dtype=jnp.float32)
        for b in buckets
    ]

    return real_by_class

def sample_batch(key: jnp.ndarray,
                 latent_inputs: jnp.ndarray,
                 cat_codes: jnp.ndarray,
                 batch_size: int) -> tuple:
    ix = random.choice(
        key=key, a=latent_inputs.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(latent_inputs, indices=ix, axis=0),
            jnp.take(cat_codes, indices=ix, axis=0))

def cpc_mi_loss(code_cat, q_cat, negative_samples=10):
    """Better signal for PGPE by using contrastive learning"""
    batch_size = code_cat.shape[0]
    
    # Positive pairs (matching code and q)
    pos_scores = jnp.sum(code_cat * q_cat, axis=-1)
    
    # Generate negative samples by shuffling
    neg_indices = jax.random.permutation(jax.random.PRNGKey(0), batch_size)
    neg_q = q_cat[neg_indices]
    neg_scores = jnp.sum(code_cat[:, None, :] * neg_q[None, :, :], axis=-1)
    
    # InfoNCE loss
    logits = jnp.concatenate([pos_scores[:, None], neg_scores], axis=1)
    labels = jnp.zeros(batch_size, dtype=jnp.int32)
    
    return -jnp.mean(nn.log_softmax(logits, axis=1)[jnp.arange(batch_size), labels])

def loss_mutual_information(code_cat, q_cat):
    cat_loss = -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))
    mi_loss = -cat_loss
    return mi_loss

def loss_mutual_information_ce(code_cat, q_cat_logits):
    # code_cat is one-hot, q_cat_logits are raw outputs
    return -jnp.mean(optax.softmax_cross_entropy(logits=q_cat_logits, labels=code_cat))

def bce_logits(logit, label):
    neg_abs = -jnp.abs(logit)
    batch_bce = jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(neg_abs))
    return jnp.mean(batch_bce)

def continuous_loss(x, mu, var):
    # Simple MSE for mean prediction
    mse = jnp.mean((x - mu) ** 2)
    
    # Regularize variance to stay near 1.0
    var_reg = jnp.mean((var - 1.0) ** 2) * 0.1
    
    return mse + var_reg

#def neg_log_likelihood_normal(x, mean, logvar):
#    return 0.5 * jnp.mean(jnp.sum(logvar + jnp.exp(-logvar) * (x - mean) ** 2, axis=-1))

def normal_nll_loss(x, mu, var):
    """
    Calculate the negative log likelihood of a normal distribution
    (treating Q(c_j | x) as a factored Gaussian).
    """
    # log-likelihood term
    logli = -0.5 * jnp.log(var * 2.0 * jnp.pi + 1e-6) \
            - ((x - mu) ** 2) / (2.0 * var + 1e-6)
    
    # negative log-likelihood (to be minimized)
    nll = -jnp.mean(jnp.sum(logli, axis=1))
    return nll

class Latent_Points(VectorizedTask):
    """Latent point task for InfoGAN Generator."""

    def __init__(self,
                 batch_size: int = 1024,
                 dataset_size: int = 800,  # Similar to MNIST
                 latent_dim: int = 64,
                 n_classes: int = 10,
                 n_con: int = 2,
                 test: bool = False):
        self.max_steps = 1
        self.obs_shape = (latent_dim + n_classes,)

        self.batch_stats_gen = None
        self.batch_stats_disc = None
        self.batch_stats_q = None 

        self.mean_mi = jnp.array([0.0])
        self.mean_g = jnp.array([0.0])
        self.mean_con = jnp.array([0.0])

        self.var_mi = jnp.array([0.0000001])
        self.var_g = jnp.array([0.0000001])
        self.var_con = jnp.array([0.0000001])
        
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.n_con = n_con

        self.noise_dim = latent_dim - n_con

        key = jax.random.PRNGKey(0)
        self.rff_params = rff_init(key, d_in=28*28, D_total=2048)

        real_by_class = build_real_by_class_mnist(train=True)
        # Precompute once
        self.mu_classes = compute_real_class_means(self.rff_params, real_by_class)
        

        def reset_fn(noise_key, cat_key, con_key):
            if test:
                batch_latent = random.normal(noise_key, (self.batch_size, self.noise_dim))
                
                #c = jnp.tile(jnp.arange(10),52)
                # remove the last 4 elements to make it 256
                #c = c[:self.batch_size]
                #batch_cat_one_hot = jax.nn.one_hot(c, 10)
                
                #batch_cat = random.randint(cat_key, (self.batch_size,), 0, self.n_classes)

                #batch_cat_one_hot = jax.nn.one_hot(batch_cat, self.n_classes)

                c = jnp.tile(jnp.arange(10),52)
                # remove the last 4 elements to make it 256
                c = c[:self.batch_size]
                batch_cat_one_hot = jax.nn.one_hot(c, 10)

                batch_con = random.uniform(con_key, (self.batch_size, self.n_con), minval=-1.0, maxval=1.0)

                batch_latent_concat = jnp.concatenate([batch_latent, batch_cat_one_hot, batch_con], axis=-1)

            else:
                
                batch_latent = random.normal(noise_key, (self.batch_size, self.noise_dim))

                #batch_cat = random.randint(cat_key, (self.batch_size,), 0, self.n_classes)
                
                #c = jnp.tile(jnp.arange(10),7)
                # remove the last 4 elements to make it 256
                #c = c[:self.batch_size]
                #batch_cat_one_hot = jax.nn.one_hot(c, 10)
                
                #batch_cat_one_hot = jax.nn.one_hot(batch_cat, self.n_classes)
                # FIXED categorical code (same for all 64 images in this evaluation)
                c_cat_idx = random.randint(cat_key, (), 0, self.n_classes)
                batch_cat_one_hot = jax.nn.one_hot(jnp.full((self.batch_size,), c_cat_idx), self.n_classes)
                

                c_cont_value = random.uniform(con_key, (2,), minval=-1.0, maxval=1.0)
                batch_con = jnp.tile(c_cont_value, (self.batch_size,1))
                #c1 = jnp.tile(jnp.arange(10),6)
                #c2 = jax.random.randint(cat_key, (4,), 0, 10)
                #c = jnp.concatenate([c1, c2])
                #c = jax.random.permutation(cat_key, c)  # Shuffle the array
                # remove the last 4 elements to make it 256
                #c = c[:self.batch_size]
                #batch_cat_one_hot = jax.nn.one_hot(c, 10)
                #batch_con = random.uniform(con_key, (self.batch_size, self.n_con), minval=-1.0, maxval=1.0)
                
                batch_latent_concat = jnp.concatenate([batch_latent, batch_cat_one_hot, batch_con], axis=-1)
                #batch_latent_concat = jnp.concatenate([batch_latent, c_cat_batch, c_cont_batch], axis=-1)
            
            return State(obs=batch_latent_concat, noise=batch_latent_concat, shift_x=batch_latent_concat, shift_y=batch_latent_concat, cat_codes=batch_cat_one_hot, codes60=batch_cat_one_hot, real_centroids=batch_cat_one_hot, con_codes=batch_con, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc, batch_stats_q=self.batch_stats_q)
        
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action, q, mu, var, q_flat, topographic_ks, fake_images):
          

            target_centroids = state.real_centroids
            
            centroid_loss = compute_hybrid_anchor_loss(
                fake_features=q_flat,
                target_centroids=target_centroids,
                n_codes=self.n_classes
            )

            #centroid_loss = compute_centroid_loss(
            #    fake_features=q_flat,
            #    target_centroids=target_centroids,
            #    n_codes=self.n_classes
            #)


            mmd_out = evaluate_fake_batch(
                fake_images,
                self.mu_classes,
                self.rff_params,
            )

            entropy = mmd_out['entropy'].astype(jnp.float32)
            #jax.debug.print('entropy: {ent}', ent=mmd_out['entropy'])
            
            # normalize q_flat and topographic_ks
            q_flat_norm = jnp.linalg.norm(q_flat, axis=-1, keepdims=True)
            q_flat = q_flat / jnp.maximum(q_flat_norm, 1e-8)

            topographic_ks_norm = jnp.linalg.norm(topographic_ks, axis=-1, keepdims=True)
            topographic_mu = topographic_ks / jnp.maximum(topographic_ks_norm, 1e-8)

            B, F = q_flat.shape # B=batch size, F=features
            K = topographic_mu.shape[0]  # K=number of topographic codes

            #c = jnp.tile(jnp.arange(self.n_classes), (64 + 10 - 1) // 10 )[ :64]
            q_flat60 = q_flat[:60]
            codes_full = jnp.concatenate([state.codes60, state.codes60[:4]], axis=0)
            topo_for_sample = topographic_mu[state.codes60]  # (B, F)

            #sq = jnp.sum((q_flat60 - topo_for_sample) ** 2, axis=-1)
            cos_sim = jnp.sum(q_flat60 * topo_for_sample, axis=-1)
            #compact = jnp.mean(1.0 - cos_sim)
            #S = topographic_ks @ topographic_ks.T  # [K, K]
            r_cons = jnp.mean(1.0 - cos_sim)
            #cos_dist = 1.0 - S
            #mask = jnp.triu(jnp.ones((K, K), dtype=cos_dist.dtype), k=1)
            #sep = jnp.sum(cos_dist * mask) / jnp.maximum(jnp.sum(mask), 1.0)


            # Aggregate per code (handles any imbalance safely)
            #sums_per_code   = jnp.bincount(state.codes60, weights=sq, length=K)         # [K]
            #counts_per_code = jnp.bincount(state.codes60, length=K).astype(q_flat.dtype)  # [K]

            #means_per_code = jnp.where(counts_per_code > 0,
            #                           sums_per_code / (counts_per_code + 1e-8),
            #                           0.0)                                     # [K]
            #num_present = jnp.maximum(1.0, jnp.sum((counts_per_code > 0).astype(q_flat.dtype)))
            #compact = jnp.sum(means_per_code) / num_present   

            #diffs = topographic_ks[:, None, :] - topographic_ks[None, :, :]  # [K, K, F] 

            #dists = jnp.linalg.norm(diffs, axis=-1)  # [K, K]
            #iu = jnp.triu_indices(K, k=1)
            #sep = jnp.mean(jnp.exp(-dists[iu] / 2.0))  # scalar
            
            # JIT-safe sep (pick one)
            #mask = jnp.triu(jnp.ones((K, K), dtype=dists.dtype), k=1)
            #sep = jnp.sum(dists * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            
            #r_cons = (-compact) + sep * 0.5
       
            # r_sense, keep first 60 samples from q_flat batch
            #q_flat60 = q_flat[:60]

            # q_flat60: (60, F)
            grouped = q_flat60.reshape((6, 10, F))        # (instances, codes, F)
            
            # Optional: normalize individual features first (like your original code)
            n_sense = jnp.linalg.norm(grouped, axis=-1, keepdims=True)
            grouped = grouped / jnp.maximum(n_sense, 1e-8)   # (6, 10, F)
            
            # ---- per-code centroids ----
            # Average over the 6 instances for each of the 10 codes
            centroids = jnp.mean(grouped, axis=0)            # (10, F)
            
            # Normalize centroids so cosine is well-behaved
            c_norm = jnp.linalg.norm(centroids, axis=-1, keepdims=True)
            centroids = centroids / jnp.maximum(c_norm, 1e-8)   # (10, F)
            
            # ---- all-pairs cosine distances between codes ----
            # Gram matrix of cosine similarities between centroids
            cos_sim_mat = centroids @ centroids.T             # (10, 10)
            
            S = cos_sim_mat - jnp.eye(cos_sim_mat.shape[0]) * 2.0  # zero out diagonal
            nearest_sim = jnp.max(S, axis=1)  # (10,)
            nearest_dist = 1.0 - nearest_sim
            r_sense = jnp.mean(nearest_dist)
            # Convert to cosine distance
            ##cos_dist_mat = 1.0 - cos_sim_mat                  # (10, 10)
            
            # Take only the 45 unique pairs (upper triangle, no diagonal)
            ##num_codes = centroids.shape[0]                    # 10
            ##i, j = jnp.triu_indices(num_codes, k=1)
            ##cos_dist = cos_dist_mat[i, j]                     # (45,)
            
            # Clip like before
            #cos_dist = jnp.minimum(cos_dist, 0.5)
            # take the lowest of the 45 distances
            ##cos_dist = jnp.min(cos_dist)
            # Final scalar regularizer
            #r_sense = jnp.mean(cos_dist)
            ##r_sense = cos_dist
            #jax.debug.print('r_cons: {r}', r=r_cons)
            grouped_intra = q_flat60.reshape(6, 10, F).transpose(1,0,2)  # (10, 6, F)

            n_intra = jnp.linalg.norm(grouped_intra, axis=-1, keepdims=True)
            grouped_intra_cosine = grouped_intra / jnp.maximum(n_intra, 1e-8)

            mean_k = jnp.mean(grouped_intra_cosine, axis=1, keepdims=True)  # (10, 1, F)
            sq_dev = jnp.sum((grouped_intra_cosine - mean_k) ** 2, axis=-1)  # (10, 6)

            spread_k = jnp.mean(sq_dev, axis=1)  # (10,)

            below = jnp.clip((spread_k / jnp.maximum(0.05, 1e-8)),0.0,1.0)
            above = 1 - jnp.clip((spread_k - 0.2) / jnp.maximum(0.2, 1e-8),0.0,1.0)

            reward_k = jnp.minimum(below, above)
            r_intra = jnp.mean(reward_k)

            sum_per_cat_code = state.cat_codes.T @ q_flat

            count_per_code = state.cat_codes.sum(axis=0)
            
            q_cat = jax.nn.log_softmax(q, axis=-1)

            loss_q_disc = loss_mutual_information(state.cat_codes, q_cat)
            #loss_q_disc = -loss_q_disc
            #loss_q_disc = cpc_mi_loss(state.cat_codes, q_cat, negative_samples=10)

            #loss_g = bce_logits(action, jnp.ones((self.batch_size,), dtype=jnp.int32))
            loss_g = optax.sigmoid_binary_cross_entropy(action, jnp.ones((self.batch_size,))).mean()
            #loss_g = -loss_g
            #loss_con = neg_log_likelihood_normal(state.con_codes, action, jnp.zeros_like(action))
            
            #loss_con = normal_nll_loss(state.con_codes, mu, var)*0.1

            #loss_q_disc = -loss_q_disc
            loss_con = continuous_loss(state.con_codes, mu, var)
            #loss_con = -loss_con
            loss_g = -loss_g#*0.1 + loss_q_disc# + loss_q_cont*0.005
            
            return state, loss_q_disc, loss_g, loss_con, sum_per_cat_code, count_per_code, r_cons, r_sense, r_intra, entropy, centroid_loss, jnp.ones(())
        
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key1: jnp.ndarray, key2: jnp.ndarray, key3: jnp.ndarray) -> State:
        return self._reset_fn(key1, key2, key3)

    def step(self,
             state: TaskState,
             action: jnp.ndarray,
             disc_logits: jnp.ndarray,
             mu: jnp.ndarray,
             var: jnp.ndarray,
             q_flat: jnp.ndarray,
             topographic_ks: jnp.ndarray,
             fake_imgs: jnp.ndarray) -> tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action, disc_logits, mu, var, q_flat,topographic_ks, fake_imgs)

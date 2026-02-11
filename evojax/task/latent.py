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
    cat_codes: jnp.ndarray
    codes60: jnp.ndarray
    con_codes: jnp.ndarray
    # hist_centroids is of shape (10,256)
    hist_centroids: jnp.ndarray
    hist_velocity: jnp.ndarray
    pop_avg_spread: jnp.ndarray
    pop_min_safety: jnp.ndarray


    batch_stats_disc: any
   
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

#def continuous_loss(x, mu, var):
#    # Simple MSE for mean prediction
#    mse = jnp.mean((x - mu) ** 2)
#    
#    # Regularize variance to stay near 1.0
#    var_reg = jnp.mean((var - 1.0) ** 2) * 0.1
#    
#    return mse + var_reg

def continuous_loss(c_true, mu, logsigma):
    """
    Negative log-likelihood of c_true under N(mu, sigma^2).
    
    Args:
        c_true: (B, q_cont) - the actual continuous codes used to generate
        mu: (B, q_cont) - predicted mean from Q network
        logsigma: (B, q_cont) - predicted log(std) from Q network
    """
    # Clamp logsigma for numerical stability
    logsigma = jnp.clip(logsigma, -2.0, 2.0)
    
    # NLL of Gaussian: 0.5 * log(2π) + logsigma + 0.5 * ((x - mu) / sigma)^2
    # We can drop the constant 0.5 * log(2π)
    nll = logsigma + 0.5 * ((c_true - mu) / jnp.exp(logsigma)) ** 2
    
    return jnp.mean(nll)

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

        self.batch_stats_disc = None

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
                # Standard random sampling for testing
                batch_latent = random.normal(noise_key, (self.batch_size, self.noise_dim))
                batch_cat = random.randint(cat_key, (self.batch_size,), 0, self.n_classes)
                batch_cat_one_hot = jax.nn.one_hot(batch_cat, self.n_classes)
                batch_con = random.uniform(con_key, (self.batch_size, self.n_con), minval=-0.5, maxval=0.5)
                batch_latent_concat = jnp.concatenate([batch_latent, batch_cat_one_hot, batch_con], axis=-1)
                
                # For test mode, codes60 isn't strictly needed in the same way, but we construct dummies to match shape
                codes60 = jnp.zeros((60,), dtype=jnp.int32) 
            else:
                # --- Controlled Experiment (Latent Vector Design) ---
                # Generate 7 base vectors, repeat 10 times -> 70 items
                z_base_fixed = jax.random.normal(noise_key, (7, 62)) 
                z_base_70 = jnp.repeat(z_base_fixed, 10, axis=0) # (70, 62)
                
                # Slice to batch size (e.g. 64)
                z_base = z_base_70[:self.batch_size] 
        
                # Construct Discrete Codes: 6 full sets of 0-9 (60 total)
                codes60 = jnp.tile(jnp.arange(10), 6)  # (60,)
                onehot60 = jax.nn.one_hot(codes60, self.n_classes) # (60, 10)
        
                # Fill remaining spots (e.g. 4 spots for batch 64)
                remainder = self.batch_size - 60
                codes_rem = jnp.tile(jnp.arange(10), (remainder // 10) + 1)[:remainder]
                onehot_rem = jax.nn.one_hot(codes_rem, self.n_classes)
        
                batch_cat_one_hot = jnp.concatenate([onehot60, onehot_rem], axis=0)
                
                # Continuous codes
                batch_con = jax.random.uniform(con_key, (self.batch_size, 2), minval=-0.5, maxval=0.5)
        
                batch_latent_concat = jnp.concatenate([z_base, batch_cat_one_hot, batch_con], axis=-1)
        
                # Instance Noise for Discriminator stability
                noise = jax.random.normal(cat_key, (self.batch_size, 28, 28, 1)) * 0.1
            
            return State(
                obs=batch_latent_concat, 
                noise=noise, 
                cat_codes=batch_cat_one_hot, 
                codes60=codes60, 
                con_codes=batch_con, 
                batch_stats_disc=self.batch_stats_disc,
                hist_centroids=jnp.zeros((10, self.rff_params["W"].shape[0])),  # Placeholder, will be updated in step
                hist_velocity=jnp.zeros((10, self.rff_params["W"].shape[0])),   # Placeholder, will be updated in step
                pop_avg_spread=0.0,  # Placeholder, will be updated in step
                pop_min_safety=1.1   # Placeholder, will be updated in step
            )
        
        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        
        def step_fn(state, action, q, mu, var, q_flat, lookahead_factor=5.0):
            """
            topographic_ks: (hist_centroids, hist_velocity)
            normative_ks:   (pop_avg_spread, pop_min_safety) 
                            - derived from the Elite History in update_normative_ks
            """
            
            # --- 1. PREP & STANDARDIZATION ---
            q_flat_norm = jnp.linalg.norm(q_flat, axis=-1, keepdims=True)
            q_flat = q_flat / jnp.maximum(q_flat_norm, 1e-8)
            
            # Unpack Knowledge Sources
            #hist_centroids, hist_velocity = topographic_ks
            #pop_avg_spread, pop_min_safety = normative_ks
            hist_centroids = state.hist_centroids.reshape(10,256) # Ensure correct shape
            hist_velocity = state.hist_velocity.reshape(10,256)   # Ensure correct shape

            pop_avg_spread = state.pop_avg_spread
            pop_min_safety = state.pop_min_safety

            # Normalize History for consistency
            h_norm = jnp.linalg.norm(hist_centroids, axis=-1, keepdims=True)
            hist_centroids_norm = hist_centroids / jnp.maximum(h_norm, 1e-8)
        
            B, F = q_flat.shape
            
            # --- 2. CALCULATE CURRENT BATCH STATISTICS ---
            sum_per_cat_code = state.cat_codes.T @ q_flat 
            count_per_code = state.cat_codes.sum(axis=0)[:, None]
            
            current_centroids = sum_per_cat_code / jnp.maximum(count_per_code, 1e-5)
            c_norm = jnp.linalg.norm(current_centroids, axis=-1, keepdims=True)
            current_centroids = current_centroids / jnp.maximum(c_norm, 1e-8)
        
            # --- 3. CALCULATE PREDICTED STATISTICS (The "Drift" Check) ---
            # We project the CURRENT centroids forward using the HISTORICAL velocity.
            # Logic: "If I keep moving like the population has been moving, where do I end up?"
            predicted_centroids = current_centroids + (hist_velocity * lookahead_factor)
            
            # Re-normalize predicted centroids to keep them on the hypersphere
            p_norm = jnp.linalg.norm(predicted_centroids, axis=-1, keepdims=True)
            predicted_centroids = predicted_centroids / jnp.maximum(p_norm, 1e-8)
        
            # --- 4. CALCULATE SPREADS (Current State) ---
            # Spread is a property of the current batch's tightness.
            assigned_centroids = state.cat_codes @ current_centroids 
            dists = 1.0 - jnp.sum(q_flat * assigned_centroids, axis=-1)
            spreads = (state.cat_codes.T @ dists[:, None]) / jnp.maximum(count_per_code, 1e-5)
           
            # --- 5. CALCULATE SAFETY RATIOS (Future State) ---
            # We use PREDICTED centroids to catch collisions before they happen.
            
            # Pairwise distance of FUTURE positions
            pred_sim_mat = predicted_centroids @ predicted_centroids.T
            pred_separation_mat = 1.0 - pred_sim_mat
            
            # Sum of CURRENT spreads (Assuming spread stays roughly constant)
            sum_spreads = spreads + spreads.T
            
            # Safety Ratio = Predicted_Separation / Current_Spread
            safety_ratios = pred_separation_mat / jnp.maximum(sum_spreads, 1e-6)
            
            # Mask diagonal
            safety_ratios = safety_ratios + jnp.eye(10) * 100.0
            # --- 6. METRICS & REWARDS (Current State) ---
            
            # r_cons: Consistency with History (Anchor)
            # Compare current batch samples to historical centroids
            q_flat60 = q_flat[:60] # Use controlled samples
            topo_for_sample = hist_centroids_norm[state.codes60]
            cos_sim_hist = jnp.sum(q_flat60 * topo_for_sample, axis=-1)
            r_cons = jnp.mean(1.0 - cos_sim_hist)
            
            # r_sense: Current Separation (Reward for existing distinctness)
            curr_sim_mat = current_centroids @ current_centroids.T
            curr_sep_mat = 1.0 - curr_sim_mat + jnp.eye(10) * 100.0
            nearest_dist = jnp.min(curr_sep_mat, axis=1)
            r_sense = jnp.mean(nearest_dist)
            
            # r_intra: Cluster Tightness Reward
            # (Keeping original scaling logic)
            spread_flat = spreads.flatten()
            below = jnp.clip((spread_flat / 0.05), 0.0, 1.0)
            above = 1.0 - jnp.clip((spread_flat - 0.2) / 0.2, 0.0, 1.0)
            reward_k = jnp.minimum(below, above)
            r_intra = jnp.mean(reward_k)
        
            # --- 7. NORMATIVE PENALTIES (Auto-Calibrated) ---
            
            # A. Safety Violation (Topographic Warning)
            # Using the POPULATION AVERAGE safety (pop_min_safety) as the baseline.
            # If this individual's predicted safety is worse than the population norm, penalize.
            min_safety_per_code = jnp.min(safety_ratios, axis=1)
            # Allow a small buffer (e.g. 1.2 hard limit, or relative to pop)
            # Let's use hard limit 1.2 as the "Safety Envelope" based on clustering theory
            raw_violation = jnp.mean(jnp.maximum(0.0, pop_min_safety - min_safety_per_code))
            safety_violation = jnp.mean(jnp.minimum(raw_violation, 2.0))
            # B. Spread Violation (Normative Check)
            # Use the Auto-Calibrated Threshold: 0.75 * Population Average Spread
            # Logic: "You must be tighter than the average historical individual."
            tightness_threshold = 0.75 * pop_avg_spread
            # Relaxing it slightly to avoid collapse: max(tightness, 0.05)
            target_spread = jnp.maximum(tightness_threshold, 0.05)
            
            spread_violation = jnp.mean(jnp.maximum(0.0, spreads - target_spread))
            
            target_min_spread = 0.015
            violation_tight = jnp.mean(jnp.maximum(0.0, target_min_spread - spreads))
           
            normative_penalty = spread_violation + violation_tight
            #normative_penalty = safety_violation + spread_violation

            count_per_code = count_per_code.flatten()
            #spreads = spreads.flatten()
            # --- 8. LOSSES ---
            q_cat = jax.nn.log_softmax(q, axis=-1)
            loss_q_disc = loss_mutual_information(state.cat_codes, q_cat)
            loss_g = -optax.sigmoid_binary_cross_entropy(action, jnp.ones((self.batch_size,))).mean()
            loss_con = continuous_loss(state.con_codes, mu, var)
            
            return (
                state, 
                loss_q_disc, 
                loss_g, 
                loss_con, 
                sum_per_cat_code, 
                count_per_code, 
                r_cons, 
                r_sense, 
                r_intra, 
                normative_penalty, 
                safety_ratios,     
                spreads,
                jnp.ones(())
            )
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key1: jnp.ndarray, key2: jnp.ndarray, key3: jnp.ndarray) -> State:
        return self._reset_fn(key1, key2, key3)

    def step(self,
             state: TaskState,
             action: jnp.ndarray,
             disc_logits: jnp.ndarray,
             mu: jnp.ndarray,
             var: jnp.ndarray,
             q_flat: jnp.ndarray) -> tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action, disc_logits, mu, var, q_flat)

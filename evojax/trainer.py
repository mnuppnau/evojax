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

import logging
import os
import time
from typing import Optional, Callable
from jax import tree_util

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax import lax
from functools import partial
from evojax.task.base import VectorizedTask
from evojax.policy import PolicyNetwork
from evojax.algo import NEAlgorithm
from evojax.algo import QualityDiversityMethod
from evojax.algo.cultural.helper_functions import kmeans
from evojax.sim_mgr import SimManager
from evojax.obs_norm import ObsNormalizer
from evojax.util import create_logger
from evojax.util import load_model_gen, load_model_disc
from evojax.util import save_model
from evojax.util import save_lattices
from evojax.util import save_checkpoint, load_checkpoint
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import he_normal
from flax import linen as nn
from medmnist import OrganSMNIST
from optax.assignment import hungarian_algorithm
# import Tuple
from typing import Tuple


#class Generator(nn.Module):
#    """ Generator CNN for MNIST """
#
#    features: int = 64
#    training: bool = True
#
#    @nn.compact
#    def __call__(self, z):
#        z = z.reshape((z.shape[0], 1, 1, z.shape[1]))
#        
#        # Add an extra upsampling block
#        x = nn.ConvTranspose(self.features*8, [3, 3], [2, 2], 'VALID')(z)  # New layer
#        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#        x = nn.relu(x)
#
#        x = nn.ConvTranspose(self.features*4, [3, 3], [2, 2], 'VALID', kernel_init=he_normal())(x)
#        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#        x = nn.relu(x)
#        x = nn.ConvTranspose(self.features*2, [4, 4], [1, 1], 'VALID', kernel_init=he_normal())(x)
#        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#        x = nn.relu(x)
#        x = nn.ConvTranspose(self.features, [4, 4], [1, 1], 'VALID', kernel_init=he_normal())(x)
#        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#        x = nn.relu(x)
#        x = nn.ConvTranspose(1, [4, 4], [2, 2], 'VALID', kernel_init=he_normal())(x)
#        x = jnp.tanh(x)
#        return x

# Assuming you have something like:
# normal_init = nn.initializers.normal

class Generator(nn.Module):
    features: int = 64
    training: bool = True

    @nn.compact
    def __call__(self, z):

        # Project z -> 7*7*8 (small depth) -> Conv to features
        x = nn.Dense(7 * 7 * 8)(z)
        x = x.reshape((x.shape[0], 7, 7, 8))

        # Expand depth with 3x3 conv (clean edges, no spatial bias)
        x = nn.Conv(self.features, kernel_size=(3,3), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = jnp.tanh(x)

        # UPSAMPLE BLOCK 1 (7x7 -> 14x14)
        x = jax.image.resize(x, shape=(x.shape[0], 14, 14, x.shape[3]), method='nearest')

        # 3x3 kernels: eliminate edge bias from old 5x5 kernels
        x = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02)
        )(x)
        x = nn.GroupNorm(num_groups=32, epsilon=1e-5)(x)
        x = jnp.tanh(x)

        # Extra depth at 14x14: richer feature composition before final upsample
        x = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02)
        )(x)
        x = nn.GroupNorm(num_groups=32, epsilon=1e-5)(x)
        x = jnp.tanh(x)

        # UPSAMPLE BLOCK 2 (14x14 -> 28x28)
        x = jax.image.resize(x, shape=(x.shape[0], 28, 28, x.shape[3]), method='nearest')

        x = nn.Conv(
            self.features // 2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02)
        )(x)
        x = nn.GroupNorm(num_groups=16, epsilon=1e-5)(x)
        x = jnp.tanh(x)

        # OUTPUT BLOCK (28x28 -> 28x28)
        x = nn.Conv(
            1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02)
        )(x)
        x = jnp.tanh(x)

        return x

class Discriminator(nn.Module):
    """Discriminator with attached Q-network (SpectralNorm, no BatchNorm)."""
    features: int = 64
    q_cat: int = 11
    q_cont: int = 0  # continuous codes removed

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, 28, 28, 1) in [-1, 1]
            train: bool (True during training, False during eval)

        Returns:
            d_logits:        (B, 1)
            q_cat_logits:    (B, q_cat)
            q_cont_mu:       (B, q_cont) or None
            q_cont_logsigma: (B, q_cont) or None
            q_feat_avg:      (B, features*4)
        """

        def SN(layer):
            # Wrapper constructor (names auto-assigned unless you pass name=...)
            return nn.SpectralNorm(layer)

        train = True
        # ----- shared backbone -----
        h = SN(nn.Conv(
            self.features,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=normal_init(0.02),
        ))(x, update_stats=train)
        h = nn.leaky_relu(h, 0.2)

        h = SN(nn.Conv(
            self.features * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=normal_init(0.02),
        ))(h, update_stats=train)
        h = nn.leaky_relu(h, 0.2)

        # ----- D head -----
        d = SN(nn.Conv(
            1,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=normal_init(0.02),
        ))(h, update_stats=train)  # -> (B, 1, 1, 1)
        d_logits = d.reshape((d.shape[0], -1))  # (B, 1)

        # ----- Q trunk -----
        q = SN(nn.Conv(
            self.features * 4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=normal_init(0.02),
        ))(h, update_stats=train)  # -> typically (B, 1, 1, features*4)
        q = nn.leaky_relu(q, 0.2)

        q_feat_avg = jnp.mean(q, axis=(1, 2))      # (B, features*4)
        q_flat = q.reshape((q.shape[0], -1))       # (B, features*4)

        # ----- Q categorical head -----
        q_cat_logits = SN(nn.Dense(
            self.q_cat,
            kernel_init=normal_init(0.02),
        ))(q_flat, update_stats=train)

        return d_logits, q_cat_logits, q_feat_avg

# --- 1. The HyperNetwork (Linearized with Multiplicative Embeddings) ---
# Replaces one-hot + GELU non-linear HN with learned embeddings + linear projection.
# Rationale: GELU activations created a rugged fitness landscape where PGPE's Gaussian
# perturbations couldn't reliably estimate gradients at scale. The linear mapping
# preserves the coordinate-based structural inductive bias while giving PGPE a smooth
# landscape. Multiplicative layer*chunk interaction lets each (layer, chunk) pair
# have a unique representation without non-linear entanglement.
class HyperNetwork(nn.Module):
    chunk_size: int = 256
    n_layers: int = 20
    n_chunks: int = 200
    embed_dim: int = 32

    @nn.compact
    def __call__(self, layer_ids, chunk_ids, depths, scales):
        """
        Input:  layer_ids (N,) int, chunk_ids (N,) int,
                depths (N,) float, scales (N,) float
        Output: (N, chunk_size) generated weight chunks
        """
        # Learned embeddings replace sparse one-hot vectors
        layer_emb = nn.Embed(self.n_layers, self.embed_dim)(layer_ids)   # (N, 32)
        chunk_emb = nn.Embed(self.n_chunks, self.embed_dim)(chunk_ids)   # (N, 32)

        # Multiplicative interaction: layer-chunk specific representation
        interaction = layer_emb * chunk_emb  # (N, 32)

        # Concatenate with geometric features
        coords = jnp.concatenate([
            interaction,
            depths[:, None],
            scales[:, None]
        ], axis=-1)  # (N, 34)

        # Single linear projection to chunk weights — no non-linearity
        weights = nn.Dense(
            self.chunk_size,
            kernel_init=jax.nn.initializers.normal(stddev=0.025)
        )(coords)

        return weights

# --- 2. The Adapter (The "Context" Builder) ---
class ParameterAdapter:
    def __init__(self, target_init_params, chunk_size=256):
        self.chunk_size = chunk_size
        self.target_tree = tree_util.tree_structure(target_init_params)

        # --- A. Flatten and Map Shapes ---
        flat_params, _ = tree_util.tree_flatten(target_init_params)
        self.param_sizes = [np.prod(p.shape) for p in flat_params]
        self.param_shapes = [p.shape for p in flat_params]

        # --- B. Assign Layer IDs and Chunk IDs ---
        layer_ids_list = []
        chunk_ids_list = []

        for layer_idx, size in enumerate(self.param_sizes):
            n_chunks = (size + chunk_size - 1) // chunk_size
            layer_ids_list.append(np.full(n_chunks, layer_idx))
            chunk_ids_list.append(np.arange(n_chunks))

        self.layer_ids = jnp.array(np.concatenate(layer_ids_list), dtype=jnp.int32)
        self.chunk_ids = jnp.array(np.concatenate(chunk_ids_list), dtype=jnp.int32)
        self.total_chunks = len(self.layer_ids)

        # --- C. Define Geometric Context ---
        total_layers = len(self.param_sizes)
        self.depth_map = np.linspace(0.0, 1.0, total_layers)
        self.scale_map = np.linspace(0.25, 1.0, total_layers)
        self.depths = jnp.array(self.depth_map)[self.layer_ids]
        self.scales = jnp.array(self.scale_map)[self.layer_ids]

        self.split_indices = np.cumsum(self.param_sizes)[:-1]

        # --- D. Embedding dimensions (computed from actual Generator) ---
        self.N_LAYERS = int(np.max(np.concatenate(layer_ids_list))) + 1
        self.N_CHUNKS = int(np.max(np.concatenate(chunk_ids_list))) + 1

    def init_hypernet(self, rng):
        return HyperNetwork(
            self.chunk_size, self.N_LAYERS, self.N_CHUNKS
        ).init(
            rng,
            jnp.zeros(self.total_chunks, dtype=jnp.int32),
            jnp.zeros(self.total_chunks, dtype=jnp.int32),
            jnp.zeros(self.total_chunks),
            jnp.zeros(self.total_chunks)
        )

    def generate_params(self, hypernet_params):
        flat_chunks = HyperNetwork(
            self.chunk_size, self.N_LAYERS, self.N_CHUNKS
        ).apply(
            hypernet_params,
            self.layer_ids,
            self.chunk_ids,
            self.depths,
            self.scales
        )

        # Reconstruct Generator param tree from flat chunks
        raw_stream = flat_chunks.reshape(-1)
        total_gen_params = self.split_indices[-1] + self.param_sizes[-1]
        valid_stream = raw_stream[:total_gen_params]
        param_list = jnp.split(valid_stream, self.split_indices)

        reshaped_params = [
            p.reshape(s) for p, s in zip(param_list, self.param_shapes)
        ]
        return tree_util.tree_unflatten(self.target_tree, reshaped_params)

#class Generator(nn.Module):
#    """
#    Flax Generator for InfoGAN MNIST.
#    Flow: (z_dim,1,1) -> (256,1,1) -> (128,7,7) -> (64,14,14) -> (1,28,28)
#    """
#    z_dim: int = 74      # Total latent dimension (noise + latent codes)
#    training: bool = True
#    
#    @nn.compact
#    def __call__(self, z):
#        """Forward pass. Returns images in [0, 1]."""
#        # Reshape from (batch, z_dim) to (batch, 1, 1, z_dim)
#        x = z.reshape((z.shape[0], 1, 1, z.shape[1]))
#        
#        # 1) ConvTranspose: z_dim -> 256
#        x = nn.ConvTranspose(
#            features=256,
#            kernel_size=(1,1),
#            strides=(1,1),
#            padding='VALID',
#            kernel_init=normal_init(0.02),  # GAN standard initialization
#            use_bias=False,
#        )(x)
#        x = nn.BatchNorm(use_running_average=not self.training)(x)
#        x = nn.relu(x)
#        
#        # 2) ConvTranspose: 256 -> 128, (1,1) -> (7,7)
#        x = nn.ConvTranspose(
#            features=128,
#            kernel_size=(7,7),
#            strides=(1,1),
#            padding='VALID',
#            kernel_init=normal_init(0.02),
#            use_bias=False,
#        )(x)
#        x = nn.BatchNorm(use_running_average=not self.training)(x)
#        x = nn.relu(x)
#        
#        # 3) ConvTranspose: 128 -> 64, (7,7) -> (14,14)
#        x = nn.ConvTranspose(
#            features=64,
#            kernel_size=(4,4),
#            strides=(2,2),
#            padding='SAME',
#            kernel_init=normal_init(0.02),
#            use_bias=False,
#        )(x)
#        x = nn.BatchNorm(use_running_average=not self.training)(x)
#        x = nn.relu(x)
#        
#        # 4) ConvTranspose: 64 -> 1, (14,14) -> (28,28)
#        x = nn.ConvTranspose(
#            features=1,
#            kernel_size=(4,4),
#            strides=(2,2),
#            padding='SAME',
#            kernel_init=normal_init(0.02),
#            use_bias=True,
#        )(x)
#        
#        # Output in [0,1] for MNIST
#        x = nn.sigmoid(x)
#        
#        return x  # Output shape: (batch_size, 28, 28, 1)
#
#class Discriminator(nn.Module):
#    features: int = 64
#    training: bool = True
#    q_cat: int = 10
#    q_cont: int = 2
#    
#    @nn.compact
#    def __call__(self, x):
#        # Shared feature extraction
#        x = nn.Conv(self.features, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
#        x = nn.leaky_relu(x, 0.2)
#        
#        x = nn.Conv(self.features*2, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02), use_bias=False)(x)
#        x = nn.BatchNorm(use_running_average=not self.training)(x)
#        x = nn.leaky_relu(x, 0.2)
#        
#        # Shared features
#        features = x
#        
#        # Discriminator output path - properly flatten the tensor
#        d = nn.Conv(1, [4, 4], [1, 1], 'VALID', kernel_init=normal_init(0.02))(features)
#        # Flatten all dimensions except batch
#        d = d.reshape((d.shape[0], -1))  # Shape becomes (batch_size, 2*2*1)
#        # Then use Dense layer to get to a single value per batch item
#        d = nn.Dense(1, kernel_init=normal_init(0.02))(d)
#        d = d.squeeze(-1)  # Shape becomes (batch_size,)
#        
#        # Q-network path for mutual information maximization
#        q = nn.Conv(self.features*4, [1, 1], [1, 1], 'VALID', kernel_init=normal_init(0.02), use_bias=False)(features)
#        q = nn.BatchNorm(use_running_average=not self.training)(q)
#        q = nn.leaky_relu(q, 0.2)
#        q = q.reshape((q.shape[0], -1))  # Flattens all dimensions except batch
#        
#        # Latent feature space
#        q_latent = nn.Dense(128)(q)
#        q_latent = nn.leaky_relu(q_latent, 0.2)
#        
#        # Categorical distribution for digit class
#        q_logits_cat = nn.Dense(self.q_cat)(q_latent)
#        
#        # Continuous distribution parameters
#        mu = nn.Dense(features=self.q_cont)(q_latent)
#        logvar = nn.Dense(features=self.q_cont)(q_latent)
#        var = jnp.exp(logvar)
#        
#        return d, q_logits_cat, mu, var

#class Generator(nn.Module):
#    """
#    Flax Generator for MNIST with fewer channels (approx ~1.7M params).
#    Matches a DCGAN-like flow:
#      (z_dim,1,1) -> (256,1,1) -> (128,7,7) -> (64,14,14) -> (1,28,28)
#    """
#    z_dim: int = 74      # Typically noise + InfoGAN code dimension
#    training: bool = True
#    
#    @nn.compact
#    def __call__(self, z):
#        """Forward pass. Returns images in [0, 1]."""
#        # Reshape from (batch, z_dim) to (batch, 1,1, z_dim)
#        x = z.reshape((z.shape[0], 1, 1, z.shape[1]))
#        
#        # 1) ConvTranspose: 74 -> 256, kernel=1, stride=1 => still (1,1)
#        x = nn.ConvTranspose(
#            features=256,
#            kernel_size=(1,1),
#            strides=(1,1),
#            padding='VALID',
#            kernel_init=he_normal(),  # He normal initialization
#            use_bias=False,
#            #kernel_init=normal_init(0.02),
#        )(x)
#        x = nn.BatchNorm(use_running_average=not self.training)(x)
#        x = nn.relu(x)
#        
#        # 2) ConvTranspose: 256 -> 128, kernel=7, stride=1 => goes (1,1) -> (7,7)
#        x = nn.ConvTranspose(
#            features=128,
#            kernel_size=(7,7),
#            strides=(1,1),
#            padding='VALID',
#            kernel_init=he_normal(),  # He normal initialization
#            use_bias=False,
#            #kernel_init=normal_init(0.02),
#        )(x)
#        x = nn.BatchNorm(use_running_average=not self.training)(x)
#        x = nn.relu(x)
#        
#        # 3) ConvTranspose: 128 -> 64, kernel=4, stride=2 => goes (7,7) -> (14,14)
#        x = nn.ConvTranspose(
#            features=64,
#            kernel_size=(4,4),
#            strides=(2,2),
#            padding='SAME',  # 'SAME' with stride=2 ~ padding=1 in PyTorch
#            kernel_init=he_normal(),  # He normal initialization
#            use_bias=False,
#            #kernel_init=normal_init(0.02),
#        )(x)
#        x = nn.BatchNorm(use_running_average=not self.training)(x)
#        x = nn.relu(x)
#        
#        # 4) ConvTranspose: 64 -> 1, kernel=4, stride=2 => goes (14,14) -> (28,28)
#        x = nn.ConvTranspose(
#            features=1,
#            kernel_size=(4,4),
#            strides=(2,2),
#            padding='SAME',
#            kernel_init=he_normal(),  # He normal initialization
#            use_bias=False,
#            #kernel_init=normal_init(0.02),
#        )(x)
#
#        # Output in [0,1] for MNIST
#        x = nn.sigmoid(x)
#        
#        return x
#
#class Discriminator(nn.Module):
#    features: int = 32
#    training: bool = True
#
#    #q_cat: int = 10
#
#    @nn.compact
#    def __call__(self, x):
#        x = nn.Conv(self.features, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
#        #x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#        x = nn.leaky_relu(x, 0.1)
#        x = nn.Conv(self.features*4, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02), use_bias=False)(x)
#        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#        x = nn.leaky_relu(x, 0.1)
#        
#        x = nn.Conv(self.features*16, [3, 3], [1, 1], 'VALID', kernel_init=normal_init(0.02), use_bias=False)(x)
#        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#        x = nn.leaky_relu(x, 0.1)
#        
#        # Discriminator output
#        d = nn.Conv(1, [3, 3], [1, 1], 'VALID', kernel_init=normal_init(0.02))(x)
#        d = d.reshape((d.shape[0], -1))
#
#        # Q Network
#        q = nn.Conv(self.features*2, [1, 1], [1, 1], 'VALID', kernel_init=normal_init(0.02), use_bias=False)(x)
#        q = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(q)
#        q = nn.leaky_relu(q, 0.1)
#
#        q = q.reshape((q.shape[0], -1))
#
#        q_latent = nn.Dense(self.features*2)(q)
#        q_latent = nn.leaky_relu(q_latent, 0.1)
#
#        q_logits_cat = nn.Dense(10)(q_latent)
#        #q_logits_cat = q_logits_cat.reshape((q_logits_cat.shape[0], -1))
#
#        mu = nn.Dense(features=2)(q_latent)
#        log_var = nn.Dense(features=2)(q_latent)
#        var = jnp.square(log_var)
#        return d,q_logits_cat, mu.squeeze(), jnp.exp(var)

#@jax.jit
#def train_step_gen(params_g, batch_stats_g, latent):
#    (fake_images), vars_g = Generator().apply({'params': params_g, 'batch_stats': batch_stats_g},latent, mutable=['batch_stats'])
#    batch_stats_g = vars_g['batch_stats']
#    return fake_images, batch_stats_g

# =============================================================================
# Sampling
# =============================================================================

@partial(jax.jit, static_argnums=(2,))
def sample_from_clusters(
    clusters: jnp.ndarray,
    key: jax.Array,
    n_samples: int = 128
) -> jnp.ndarray:
    """
    Sample n_samples images from each cluster.
    
    Args:
        clusters: (n_clusters, n_images_per_cluster, image_dim) e.g. (10, 800, 784)
        key: PRNG key
        n_samples: Number of samples per cluster
    
    Returns:
        (n_clusters, n_samples, image_dim) e.g. (10, 128, 784)
    """
    n_clusters = clusters.shape[0]
    n_images = clusters.shape[1]
    keys = jax.random.split(key, n_clusters)
    
    def sample_single(cluster: jnp.ndarray, subkey: jax.Array) -> jnp.ndarray:
        indices = jax.random.choice(subkey, n_images, shape=(n_samples,), replace=False)
        return cluster[indices]
    
    return jax.vmap(sample_single)(clusters, keys)


# =============================================================================
# Centroid Computation
# =============================================================================

@jax.jit
def compute_real_centroids(real_features: jnp.ndarray) -> jnp.ndarray:
    """
    Compute centroids by averaging across samples for each cluster.
    
    Args:
        real_features: (n_clusters, n_samples, feature_dim) e.g. (10, 128, 256)
    
    Returns:
        (n_clusters, feature_dim) e.g. (10, 256)
    """
    return jnp.mean(real_features, axis=1)


@partial(jax.jit, static_argnums=(1,))
def compute_fake_centroids(fake_features: jnp.ndarray, n_codes: int = 11) -> jnp.ndarray:
    """
    Compute centroids from interleaved fake features.

    Features are interleaved by code:
        - Indices 0, n_codes, 2*n_codes, ... -> code 0
        - Indices 1, n_codes+1, 2*n_codes+1, ... -> code 1
        - etc.

    Args:
        fake_features: (batch_size, feature_dim) e.g. (128, 256)
        n_codes: Number of codes (default 11)

    Returns:
        (n_codes, feature_dim) e.g. (11, 256)
    """
    batch_size = fake_features.shape[0]
    indices = jnp.arange(batch_size)
    
    def compute_centroid_for_code(code_idx: int) -> jnp.ndarray:
        mask = (indices % n_codes) == code_idx
        masked_sum = jnp.sum(jnp.where(mask[:, None], fake_features, 0.0), axis=0)
        count = jnp.sum(mask)
        jax.debug.print("Code index: {}, Count: {}", code_idx, count)
        return masked_sum / count
    
    return jax.vmap(compute_centroid_for_code)(jnp.arange(n_codes))


# =============================================================================
# Cost Matrix & Assignment
# =============================================================================

@jax.jit
def compute_cost_matrix(
    real_centroids: jnp.ndarray,
    fake_centroids: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute squared L2 distance cost matrix between centroids.
    
    Args:
        real_centroids: (n_real, feature_dim) e.g. (10, 256)
        fake_centroids: (n_fake, feature_dim) e.g. (10, 256)
    
    Returns:
        (n_fake, n_real) cost matrix
    """
    diff = fake_centroids[:, None, :] - real_centroids[None, :, :]
    return jnp.sum(diff ** 2, axis=-1)


@jax.jit
def optimal_assignment(cost_matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find optimal assignment using Hungarian algorithm.
    
    Args:
        cost_matrix: (n_fake, n_real) cost matrix
    
    Returns:
        (row_indices, col_indices) where:
            fake_code[row_indices[i]] is assigned to real_cluster[col_indices[i]]
    """
    return hungarian_algorithm(cost_matrix)


# =============================================================================
# Combined Pipeline
# =============================================================================

@jax.jit
def assign_fake_to_real(
    real_features: jnp.ndarray,
    fake_features: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Complete assignment pipeline (after discriminator features are extracted).
    
    Args:
        real_features: (n_clusters, n_samples, feature_dim) e.g. (10, 128, 256)
        fake_features: (batch_size, feature_dim) e.g. (128, 256), interleaved by code
    
    Returns:
        (row_indices, col_indices, cost_matrix) where:
            fake_code[row_indices[i]] -> real_cluster[col_indices[i]]
    """
    real_centroids = compute_real_centroids(real_features)
    fake_centroids = compute_fake_centroids(fake_features, n_codes=11)
    cost_matrix = compute_cost_matrix(real_centroids, fake_centroids)
    row_idx, col_idx = optimal_assignment(cost_matrix)
    return row_idx, col_idx, cost_matrix


# =============================================================================
# Utility
# =============================================================================

def assignment_to_mapping(row_indices: jnp.ndarray, col_indices: jnp.ndarray) -> dict:
    """
    Convert assignment indices to a dictionary mapping.
    
    Returns:
        Dict mapping fake_code -> real_cluster
    """
    return {int(row_indices[y]): int(col_indices[y]) for y in range(len(row_indices))}

@jax.jit
def reorder_real_centroids(
    real_centroids: jnp.ndarray,  # (10, 256)
    col_indices: jnp.ndarray       # (10,) from optimal_assignment
) -> jnp.ndarray:                  # (10, 256) reordered
    return real_centroids[col_indices]

@partial(jax.jit, static_argnames=['solver'])
def train_step_disc(state, data, noise_shift_tup, fake_imgs, fake_cat_input, solver):
       
        noise, shift_x, shift_y = noise_shift_tup
        params_d, batch_stats_d, opt_disc = state
        #def bce_logits(logit, label):
        #          """
        #          Implements the BCE with logits loss, as described:
        #          https://github.com/pytorch/pytorch/issues/751
        #          """
        #          neg_abs = -jnp.abs(logit)
        #          batch_bce = jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(neg_abs))
        #          return jnp.mean(batch_bce)

        def loss_mutual_information(code_cat, q_cat):
                  return -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))
           
        def loss_mutual_information_ce(code_cat, q_cat_logits):
            # code_cat is one-hot, q_cat_logits are raw outputs
            return jnp.mean(optax.softmax_cross_entropy(logits=q_cat_logits, labels=code_cat))

        #def continuous_loss(x, mu, var):
        #    # Simple MSE for mean prediction
        #    mse = jnp.mean((x - mu) ** 2)
        #    
        #    # Regularize variance to stay near 1.0
        #    var_reg = jnp.mean((var - 1.0) ** 2) * 0.1
        #    
        #    return mse + var_reg

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
        
        def loss_discriminator(params_d, vars_d_batch_stats):
                
                  #(fake_imgs, vars_g) = Generator().apply(
                  #    {'params': params_g, 'batch_stats': batch_stats_g},
                  #    latent, mutable=['batch_stats']
                  #)
                  

                  #X = jnp.concatenate([fake_imgs, data], axis=0)
                  #idx = jax.random.permutation(jax.random.PRNGKey(0), X.shape[0])

                  #X = X[idx]

                  #Bf = fake_imgs.shape[0]


                  #(preds, q, mu, var), vars_d = Discriminator().apply(
                  #    {'params': params_d, 'batch_stats': vars_d_batch_stats},
                  #    X, mutable=['batch_stats']
                  #  )

                  #fake_preds = preds[:Bf]
                  #real_preds = preds[Bf:]

                  #q_fake = q[:Bf]
                  #mu_fake = mu[:Bf]
                  #var_fake = jnp.exp(var[:Bf])
                  
                  fake_images_with_noise = fake_imgs + noise
                  real_images_with_noise = data + noise

                  #fake_images_with_noise_perturbed = jnp.roll(fake_images_with_noise, shift=(shift_x, shift_y), axis=(1,2))
                  #real_images_with_noise_perturbed = jnp.roll(real_images_with_noise, shift=(shift_x, shift_y), axis=(1,2))
                  
                  (fake_preds, q_fake, _), vars_d = Discriminator().apply(
                      {'params': params_d, 'batch_stats': vars_d_batch_stats},
                      fake_images_with_noise, mutable=['batch_stats']
                  )
                  
                  (real_preds, _, _), vars_d = Discriminator().apply(
                      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
                      real_images_with_noise, mutable=['batch_stats']
                  )
                
                  # use q_logits and labels to calculate q accuracy
                  #q_preds = q_logits.argmax(axis=-1)
                  #q_acc = jnp.mean(q_preds == labels)
                  logit_penalty = 1e-2 * jnp.mean(fake_preds ** 2)
                  # Calculate Mutual Information loss
                  q_cat = nn.log_softmax(q_fake, axis=-1)
                  loss_mi = loss_mutual_information(fake_cat_input, q_cat)

                  # use 0.9 as the label for real images instead of 1.0
                  real_loss = optax.sigmoid_binary_cross_entropy(real_preds, jnp.ones_like(real_preds)*0.9)
                  # use 0.1 as the label for fake images instead of 0.0
                  fake_loss = optax.sigmoid_binary_cross_entropy(fake_preds, jnp.zeros_like(fake_preds)+0.05)

                  real_loss = jnp.mean(real_loss)
                  fake_loss = jnp.mean(fake_loss)

                  real_fake_loss = (real_loss + fake_loss) / 2.0
                  loss = real_fake_loss + loss_mi*0.1
                
                  return loss, (real_fake_loss, vars_d)

        grad_fn_disc = jax.value_and_grad(loss_discriminator, has_aux=True)
        (loss, (real_fake_loss, vars_d)), grads = grad_fn_disc(params_d, batch_stats_d)

        # Smooth D regulation: scale gradients based on real_fake_loss.
        # D gets full learning at real_fake_loss >= 0.5 (equilibrium),
        # zero learning at <= 0.35 (D dominant), linear ramp between.
        # This replaces the binary freeze threshold.
        d_scale = jnp.clip((real_fake_loss - 0.35) / 0.15, 0.0, 1.0)
        grads = jax.tree_util.tree_map(lambda g: g * d_scale, grads)

        # apply gradients
        updates, new_opt_state = solver.update(grads, opt_disc, params_d)
        params_d = optax.apply_updates(params_d, updates)
        #batch_stats_g = vars_g['batch_stats']
        # update batch stats
        batch_stats_d = vars_d['batch_stats']
        return (params_d, batch_stats_d, new_opt_state), loss, real_fake_loss

#class QNetwork(nn.Module):
#    features: int = 64
#    training: bool = True
#
#    q_cat: int = 10
#
#    @nn.compact
#    def __call__(self, x):
#        q = nn.Conv(self.features, [3, 3], [2, 2], 'VALID', kernel_init=he_normal())(x) 
#        q = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(q)
#        q = nn.leaky_relu(q, 0.1)
#       
#        disc_logits = nn.Conv(self.q_cat, [1, 1], [2, 2], 'VALID', kernel_init=he_normal())(q)
#        disc_logits = disc_logits.reshape((disc_logits.shape[0], -1)) 
#
#        return disc_logits
def build_big_latents(key, total_size, z_dim, n_disc):
    k_z, k_c, k_perm = jax.random.split(key, 3)
    z = jax.random.normal(k_z, (total_size, z_dim))
    reps = (total_size + n_disc - 1) // n_disc
    codes = jnp.tile(jnp.arange(n_disc), reps)[:total_size]
    c_onehot = jax.nn.one_hot(codes, n_disc)
    latent = jnp.concatenate([z, c_onehot], axis=-1)
    return latent[jax.random.permutation(k_perm, total_size)]

def build_recal_latents(key, batch_size, z_dim, n_disc):
    k_z, k_perm = jax.random.split(key, 2)
    # z: standard normal (or whatever you use at train time)
    z = jax.random.normal(k_z, (batch_size, z_dim))
    # categorical: balanced 0..n_disc-1 repeated
    reps = (batch_size + n_disc - 1) // n_disc
    c = jnp.tile(jnp.arange(n_disc), reps)[:batch_size]
    c_onehot = jax.nn.one_hot(c, n_disc)
    # concatenate
    latent = jnp.concatenate([z, c_onehot], axis=-1)
    # (optional) small permutation to avoid repeating same order every batch
    perm = jax.random.permutation(k_perm, batch_size)
    return latent[perm]

# --- BN standing-stats pass ---
def recalibrate_bn_stats(gen_module, params, batch_stats, key,
                         steps=12, batch_size=64,
                         z_dim=63, n_disc=11):
    """
    gen_module: e.g., Generator(training=True) or Generator(use_running_average=False)
    params: generator params pytree
    batch_stats: current running stats collection
    key: PRNGKey
    """
    def body(i, carry):
        bs, k = carry
        k, k_lat = jax.random.split(k)
        lat = build_recal_latents(k_lat, batch_size, z_dim, n_disc)
        # train-mode apply: mutate batch_stats only
        _, vars_out = gen_module.apply({'params': params, 'batch_stats': bs},
                                       lat,
                                       mutable=['batch_stats'])
        return (vars_out['batch_stats'], k)

    # run steps times; jit the whole loop
    (batch_stats_new, key_new) = lax.fori_loop(0, steps, body, (batch_stats, key))
    return batch_stats_new, key_new

def sample_latent(key, shape_noise, shape_cat):
  noise_key, cat_key = jax.random.split(key, 2)

  # Sample irreducible noise
  noise = jax.random.normal(noise_key, shape_noise)

  # Sample categorical latent code
  code_cat = jax.random.randint(cat_key, shape_cat, 0, 11)
  code_cat = jax.nn.one_hot(code_cat, 11)

  latent = jnp.concatenate([noise, code_cat], axis=-1)

  return latent, code_cat

def sample_batch(key: jnp.ndarray,
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 batch_size: int) -> Tuple:
    ix = jax.random.choice(
        key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(data, indices=ix, axis=0),
            jnp.take(labels, indices=ix, axis=0))

class Trainer(object):
    """A trainer that organizes the training logistics."""

    def __init__(self,
                 policy_gen: PolicyNetwork,
                 solver_hn: NEAlgorithm,
                 #solver_disc: NEAlgorithm,
                 #solver_q: NEAlgorithm,
                 train_task_gen: VectorizedTask,
                 test_task_disc: VectorizedTask,
                 train_task_disc: VectorizedTask,
                 test_task_gen: VectorizedTask,
                 max_iter: int = 1000,
                 log_interval: int = 20,
                 test_interval: int = 100,
                 n_repeats: int = 1,
                 test_n_repeats: int = 2,
                 n_evaluations: int = 100,
                 seed: int = 42,
                 debug: bool = False,
                 use_for_loop: bool = False,
                 normalize_obs: bool = False,
                 model_dir: str = None,
                 batch_size: int = 32,
                 log_dir: str = None,
                 checkpoint_dir: str = None,
                 checkpoint_interval: int = 0,
                 resume_from: str = None,
                 logger: logging.Logger = None,
                 log_scores_fn: Optional[Callable[[int, jnp.ndarray, str], None]] = None):
        """Initialization.

        Args:
            policy - The policy network to use.
            solver - The ES algorithm for optimization.
            train_task - The task for training.
            test_task - The task for evaluation.
            max_iter - Maximum number of training iterations.
            log_interval - Interval for logging.
            test_interval - Interval for tests.
            n_repeats - Number of rollout repetitions.
            n_evaluations - Number of tests to conduct.
            seed - Random seed to use.
            debug - Whether to turn on the debug flag.
            use_for_loop - Use for loop for rollouts.
            normalize_obs - Whether to use an observation normalizer.
            model_dir - Directory to save/load model.
            log_dir - Directory to dump logs.
            logger - Logger.
            log_scores_fn - custom function to log the scores array. Expects input:
                `current_iter`: int, `scores`: jnp.ndarray, 'stage': str = "train" | "test"
        """

        if logger is None:
            self._logger = create_logger(
                name='Trainer', log_dir=log_dir, debug=debug)
        else:
            self._logger = logger

        self.batch_stats_disc = policy_gen.flat_batch_stats_disc

        self.adapter = policy_gen.adapter

        self.batch_size = batch_size
        self.mini_batch_size = 64
        self.num_mini_batches = 1
       
        self.avg_mi_loss = -2.30
        self.fake_imgs = None
        self.cat_codes = None

        self.latent_dim = 63
        self.n_classes = 11

        self.decay_factor = 0.9

        self.noise_dim = self.latent_dim
        self.policy_gen = policy_gen

        self._key = jax.random.PRNGKey(44)

        self._log_interval = log_interval
        self._test_interval = test_interval
        self._max_iter = max_iter
        self.model_dir = model_dir
        self._log_dir = log_dir
        self._checkpoint_dir = checkpoint_dir or (os.path.join(log_dir, 'checkpoints') if log_dir else None)
        self._checkpoint_interval = checkpoint_interval
        self._resume_from = resume_from

        self._log_scores_fn = log_scores_fn or (lambda x, y, z: None)

        self._obs_normalizer = ObsNormalizer(
            obs_shape=train_task_gen.obs_shape,
            dummy=not normalize_obs,
        )

        self.solver_hn = solver_hn
        #self.solver_disc = solver_disc
        #self.solver_q = solver_q

        self.sim_mgr_gen = SimManager(
            n_repeats=n_repeats,
            test_n_repeats=test_n_repeats,
            pop_size=solver_hn.pop_size,
            n_evaluations=n_evaluations,
            policy_net=policy_gen,
            train_vec_task=train_task_gen,
            valid_vec_task=test_task_gen,
            seed=seed,
            obs_normalizer=self._obs_normalizer,
            use_for_loop=use_for_loop,
            logger=self._logger,
        )

        self._key, subkey = jax.random.split(self._key)
        
        dataset = OrganSMNIST(split='train', download=True, root='./data')
        data_raw = np.array(dataset.imgs, dtype=np.float32)
        if data_raw.ndim == 3:  # (N, 28, 28) -> (N, 28, 28, 1)
            data_raw = np.expand_dims(data_raw, axis=-1)
        self.data = data_raw / 127.5 - 1.0

        self._key, subkey = jax.random.split(self._key)

        self.labels = dataset.labels.flatten()

        # initialize the discriminator
        variables_disc = Discriminator().init(subkey, jnp.ones((self.batch_size, 28, 28, 1), dtype=jnp.float32))
        self.params_disc, self.batch_stats_disc = variables_disc['params'], variables_disc['batch_stats']

        self.solver_disc = optax.adam(learning_rate=0.00008, b1=0.5, b2=0.999)

    def run(self, demo_mode: bool = False) -> float:

        """Start the training / test process."""

        if self.model_dir is not None:
            params_hn = load_model_gen(model_dir=self.model_dir)
            params_disc, self.batch_stats_disc = load_model_disc(model_dir=self.model_dir)
            #self.sim_mgr.obs_params = obs_params
            self._logger.info(
                'Loaded model parameters from {}.'.format(self.model_dir))
        else:
            params_hn, params_disc, params_q = None, None, None

        if demo_mode:
            if params is None:
                raise ValueError('No policy parameters to evaluate.')
            self._logger.info('Start to test the parameters.')
            scores = np.array(
                self.sim_mgr.eval_params(params=params, test=True)[0])
            self._logger.info(
                '[TEST] #tests={0}, max={1:.4f}, avg={2:.4f}, min={3:.4f}, '
                'std={4:.4f}'.format(scores.size, scores.max(), scores.mean(),
                                     scores.min(), scores.std()))
            return scores.mean()
        else:
            solver_disc = self.solver_disc
            opt_disc = solver_disc.init(self.params_disc)


            self._logger.info(
                'Start to train for {} iterations.'.format(self._max_iter))

            if params_hn is not None and params_disc is not None and params_q is not None:
                # Continue training from the breakpoint.
                self.solver_hn.best_params = params_hn

            best_score_gen, best_score_disc, best_score_q = -float('Inf'), -float('Inf'), -float('Inf')

            params_disc = self.params_disc

            # --- Checkpoint resume ---
            start_iter = 0
            if self._resume_from is not None:
                ckpt_path = self._resume_from
                if os.path.isdir(ckpt_path):
                    ckpt_path = os.path.join(ckpt_path, 'checkpoint_latest.pkl')
                start_iter, params_disc, self.batch_stats_disc, opt_disc, self._key = load_checkpoint(
                    checkpoint_path=ckpt_path,
                    solver_hn=self.solver_hn,
                    disc_params_ref=params_disc,
                    disc_batch_stats_ref=self.batch_stats_disc,
                    opt_disc_ref=opt_disc,
                    logger=self._logger,
                )
                start_iter += 1  # Resume from the next iteration
           
            num_mini_batches = self.num_mini_batches
           
            self._key, noise_key = jax.random.split(self._key, 2)

            fixed_batch_latent = jax.random.normal(noise_key, (self.batch_size, self.latent_dim))
            fixed_c = jnp.tile(jnp.arange(self.n_classes), 7)
            fixed_c = fixed_c[:self.batch_size]

            fixed_latent = jnp.concatenate([fixed_batch_latent, jax.nn.one_hot(fixed_c, self.n_classes)], axis=-1)

            self.ordered_centroids = jnp.zeros((self.n_classes, 256))

            real_fake_loss = 0.0  # Default for logging when D is not trained

            # --- Set up metric and KS weight log files ---
            metrics_log_path = os.path.join(self._log_dir, 'metrics.tsv') if self._log_dir else 'metrics.tsv'
            with open(metrics_log_path, 'w') as f:
                f.write('\t'.join([
                    'iter',
                    'adv_max', 'adv_avg', 'adv_min', 'adv_std',
                    'mi_max', 'mi_avg', 'mi_min', 'mi_std',
                    'r_cons_avg', 'r_sense_avg', 'r_intra_avg',
                    'norm_pen_avg', 'safety_avg', 'spread_avg',
                    'real_fake_loss',
                ]) + '\n')

            ks_log_path = os.path.join(self._log_dir, 'ks_weights.tsv') if self._log_dir else 'ks_weights.tsv'
            self.solver_hn.set_ks_log_path(ks_log_path)

            for i in range(start_iter, self._max_iter):

                shape_noise = (self.mini_batch_size, self.latent_dim)
                shape_cat = (self.mini_batch_size,)
                
                # D-step frequency: reduce D updates early so G can establish
                # diversity before D crushes it. Every 3rd iter for first 1.5k,
                # then every iteration after.
                d_freq = 3 if i < 1500 else (2 if i < 3000 else 1)
                if i % d_freq == 0:
                    for mini_batch in range(num_mini_batches):
                        # Sample batch of data.

                        self._key, subkey_latent, subkey_mnist, subkey_noise, subkey_shift_x, subkey_shift_y = jax.random.split(self._key,6)


                        data, labels = sample_batch(subkey_mnist, self.data, self.labels, self.mini_batch_size)
                        #data = np.expand_dims(data / 255.0, axis=-1)

                        latent, cat_codes = sample_latent(subkey_latent, shape_noise, shape_cat)

                        noise = jax.random.normal(subkey_noise, (self.mini_batch_size, 28, 28, 1)) * 0.1

                        shift_x = jax.random.randint(subkey_shift_x, shape=(), minval=-1, maxval=2)
                        shift_y = jax.random.randint(subkey_shift_y, shape=(), minval=-1, maxval=2)

                        #if i < 2:
                        params_hn = self.solver_hn.best_params
                        best_params_hn_formatted = self.policy_gen._format_single_params_hypernet_fn(params_hn)
                        #else:

                        params_g = self.adapter.generate_params(best_params_hn_formatted)

                        (fake_images) = Generator(training=False).apply({'params': params_g},latent)

                        # reshape fake_images to (64, 28, 28, 1) from [1,1,1,64, 28, 28, 1]
                        fake_images = fake_images.reshape((self.mini_batch_size, 28, 28, 1))

                        state = (params_disc, self.batch_stats_disc, opt_disc)

                        state, d_loss, real_fake_loss = train_step_disc(
                            state,
                            data,
                            (noise,shift_x, shift_y),
                            fake_images,
                            cat_codes,
                            solver_disc,
                        )
#jax.debug.print('loss: {} ', loss)
                        params_disc, self.batch_stats_disc, opt_disc = state

                leaves_params, _ = jax.tree_flatten(params_disc) 
                flat_params_disc = jnp.concatenate([p.flatten() for p in leaves_params])

                leaves_batch_stats_disc, _ = jax.tree_flatten(self.batch_stats_disc)
                flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

                params_hn, belief_space = self.solver_hn.ask()
                
                topographic_ks = belief_space[4]
                normative_ks = belief_space[5]

                #jax.debug.print('topographic_ks shape: {} ', topographic_ks.shape)
                #avg_per_code = topographic_ks[0]
                
                scores_gen_adv, scores_gen_mi, disc_logits, bds_gen, _, mean_var_fake, avg_per_code_current, r_cons, r_sense, r_intra, norm_pen, safety_ratios, spreads = self.sim_mgr_gen.eval_params(
                params_gen=params_hn, params_disc=flat_params_disc, batch_stats_disc=flat_batch_stats_disc, topographic_ks=topographic_ks, normative_ks=normative_ks, generator=True, test=False
                )

                #jax.debug.print('fake_imgs shape: {} ', fake_imgs.shape)
                if isinstance(self.solver_hn, QualityDiversityMethod):
                    self.solver_hn.observe_bd(bds_gen)
                
                self.solver_hn.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, disc_logits=disc_logits, pop_var=mean_var_fake, avg_per_code=avg_per_code_current, r_cons=r_cons, r_sense=r_sense, r_intra=r_intra, normative_penalty=norm_pen, safety_ratios=safety_ratios, spreads=spreads, adv=False)

                
                self.avg_mi_loss = jnp.mean(scores_gen_mi)

                if i > 0 and i % self._log_interval == 0:
                    scores_gen_adv = np.array(scores_gen_adv)
                    self._logger.info('Generator:')
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores_gen_adv.size, scores_gen_adv.max(), scores_gen_adv.mean(),
                            scores_gen_adv.min(), scores_gen_adv.std()))
                    #scores_disc = np.array(scores_real+scores_fake)
                    #self._logger.info('Discriminator:')
                    #self._logger.info(
                    #    'Iter={0}, size={1}, max={2:.4f}, '
                    #    'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                    #        i, scores_disc.size, scores_disc.max(), scores_disc.mean(),
                    #        scores_disc.min(), scores_disc.std()))
                    scores_mi = np.array(scores_gen_mi)
                    #self._logger.info('Mutual Information:')
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores_mi.size, scores_mi.max(), scores_mi.mean(),
                            scores_mi.min(), scores_mi.std()))

                    r_cons = np.array(r_cons)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, r_cons.size, r_cons.max(), r_cons.mean(),
                            r_cons.min(), r_cons.std()))
                    
                    r_sense = np.array(r_sense)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, r_sense.size, r_sense.max(), r_sense.mean(),
                            r_sense.min(), r_sense.std()))
                    
                    r_intra = np.array(r_intra)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, r_intra.size, r_intra.max(), r_intra.mean(),
                            r_intra.min(), r_intra.std()))

                    norm_pen = np.array(norm_pen)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, norm_pen.size, norm_pen.max(), norm_pen.mean(),
                            norm_pen.min(), norm_pen.std()))

                    safety_ratios = np.array(safety_ratios)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, safety_ratios.size, safety_ratios.max(), safety_ratios.mean(),
                            safety_ratios.min(), safety_ratios.std()))

                    spreads = np.array(spreads)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, spreads.size, spreads.max(), spreads.mean(),
                            spreads.min(), spreads.std()))

                    self._logger.info(
                        'Iter={0}, real_fake_loss={1:.4f}'.format(
                            i, real_fake_loss))

                    # Write metrics to TSV file
                    with open(metrics_log_path, 'a') as f:
                        f.write('\t'.join([
                            str(i),
                            f'{scores_gen_adv.max():.6f}', f'{scores_gen_adv.mean():.6f}',
                            f'{scores_gen_adv.min():.6f}', f'{scores_gen_adv.std():.6f}',
                            f'{scores_mi.max():.6f}', f'{scores_mi.mean():.6f}',
                            f'{scores_mi.min():.6f}', f'{scores_mi.std():.6f}',
                            f'{r_cons.mean():.6f}', f'{r_sense.mean():.6f}',
                            f'{r_intra.mean():.6f}', f'{norm_pen.mean():.6f}',
                            f'{safety_ratios.mean():.6f}', f'{spreads.mean():.6f}',
                            f'{float(real_fake_loss):.6f}',
                        ]) + '\n')

                if i > 0 and i % self._test_interval == 0:
                    best_params_hn = self.solver_hn.best_params
                    best_params_hn_formatted = self.policy_gen._format_single_params_hypernet_fn(best_params_hn)

                    self._key, noise_key, con_key = jax.random.split(self._key, 3)

                    params_g = self.adapter.generate_params(best_params_hn_formatted)

                    (fake_imgs) = Generator(training=False).apply({'params': params_g},fixed_latent)
                    
                    filename = f"iteration-{i}.npy"
                    np.save(filename, fake_imgs[:, :, :, :])

                if self._checkpoint_interval > 0 and i > 0 and i % self._checkpoint_interval == 0:
                    save_checkpoint(
                        checkpoint_dir=self._checkpoint_dir,
                        iteration=i,
                        solver_hn=self.solver_hn,
                        params_disc=params_disc,
                        batch_stats_disc=self.batch_stats_disc,
                        opt_disc=opt_disc,
                        prng_key=self._key,
                        logger=self._logger,
                    )

            # Test and save the final model.
            best_params_hn = self.solver_hn.best_params
            #best_params_disc = self.solver_disc.best_params
            #test_scores, _ = self.sim_mgr.eval_params(
            #    params=best_params, test=True)
            #self._logger.info(
            #    '[TEST] Iter={0}, #tests={1}, max={2:.4f}, avg={3:.4f}, '
            #    'min={4:.4f}, std={5:.4f}'.format(
            #        self._max_iter, test_scores.size, test_scores.max(),
            #        test_scores.mean(), test_scores.min(), test_scores.std()))
            #mean_test_score = test_scores.mean()
            save_model(
                model_dir=self._log_dir,
                model_name='final_model_gen',
                params=best_params_hn,
                obs_params=self.sim_mgr_gen.obs_params,
                batch_stats=self.batch_stats_disc,
                #best=mean_test_score > best_score,
            )
            #save_model(
            #    model_dir=self._log_dir,
            #    model_name='final_model_disc',
            #    params=best_params_disc,
            #    obs_params=self.sim_mgr_disc.obs_params,
            #    batch_stats=self.batch_stats_disc,
            #    #best=mean_test_score > best_score,
            #)
            #best_score = max(best_score, mean_test_score)
            #if isinstance(self.solver, QualityDiversityMethod):
            #    save_lattices(
            #        log_dir=self._log_dir,
            #        file_name='qd_lattices',
            #        fitness_lattice=self.solver.fitness_lattice,
            #        params_lattice=self.solver.params_lattice,
            #        occupancy_lattice=self.solver.occupancy_lattice,
            #    )
            self._logger.info(
                'Training done, best_score={0:.4f}'.format(best_score))

            return best_score

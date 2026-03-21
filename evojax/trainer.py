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
import torchvision
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
    n_codes: int = -1   # Must be set explicitly (e.g. 8 for BloodMNIST)
    noise_dim: int = -1  # Must be set explicitly (e.g. 62)
    code_seed_scale: float = 3.0
    cont_seed_scale: float = 1.0

    @nn.compact
    def __call__(self, z):
        # Split latent paths: noise and codes are handled by separate projections
        # to avoid code leakage into the noise seed path.
        noise_only = z[:, :self.noise_dim]
        code_onehot = z[:, self.noise_dim:self.noise_dim + self.n_codes]
        cont_codes = z[:, self.noise_dim + self.n_codes:]

        # Strengthen code influence with a dedicated seed path.
        seed_base = nn.Dense(7 * 7 * 8, name='seed_dense')(noise_only)
        seed_code = nn.Dense(
            7 * 7 * 8,
            use_bias=False,
            name='seed_code_dense',
        )(code_onehot)
        x = seed_base + (jnp.asarray(self.code_seed_scale, dtype=seed_base.dtype) * seed_code)
        if cont_codes.shape[-1] > 0:
            seed_cont = nn.Dense(
                7 * 7 * 8,
                use_bias=False,
                name='seed_cont_dense',
            )(cont_codes)
            x = x + (jnp.asarray(self.cont_seed_scale, dtype=seed_base.dtype) * seed_cont)
        x = x.reshape((x.shape[0], 7, 7, 8))

        # Now use a Conv to expand depth (standard HyperNet texture generation)
        x = nn.Conv(self.features, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = jnp.tanh(x)

        # 2. UPSAMPLE BLOCK 1 (7x7 -> 14x14)
        x = jax.image.resize(x, shape=(x.shape[0], 14, 14, x.shape[3]), method='linear')

        x = nn.Conv(
            self.features,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02),
        )(x)
        x = nn.GroupNorm(num_groups=32, epsilon=1e-5)(x)
        x = jnp.tanh(x)

        # Extra 14x14 refinement block
        x = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02),
        )(x)
        x = nn.GroupNorm(num_groups=32, epsilon=1e-5)(x)
        x = jnp.tanh(x)

        # 3. UPSAMPLE BLOCK 2 (14x14 -> 28x28)
        x = jax.image.resize(x, shape=(x.shape[0], 28, 28, x.shape[3]), method='linear')

        x = nn.Conv(
            self.features // 2,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02),
        )(x)
        x = nn.GroupNorm(num_groups=16, epsilon=1e-5)(x)
        x = jnp.tanh(x)

        # 4. OUTPUT BLOCK (28x28 -> 28x28, 1 channel grayscale)
        x = nn.Conv(
            1,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02),
        )(x)
        #x = jnp.tanh(x)
        x = jax.nn.sigmoid(x)

        return x

class Discriminator(nn.Module):
    """Discriminator with attached Q-network (SpectralNorm, no BatchNorm)."""
    features: int = 64
    q_cat: int = 10
    q_cont: int = 2

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, 28, 28, 1) in [0, 1]
            train: bool (True during training, False during eval)

        Returns:
            d_logits:        (B, 1)
            q_cat_logits:    (B, q_cat)
            q_cont_mu:       (B, q_cont) or None
            q_cont_logsigma: (B, q_cont) or None
            q_feat_avg:      (B, features*2)
        """

        def SN(layer):
            # Wrapper constructor (names auto-assigned unless you pass name=...)
            return nn.SpectralNorm(layer)

        train = True
        # ----- shared backbone (SAME padding for full spatial coverage) -----
        h = SN(nn.Conv(
            self.features,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            kernel_init=normal_init(0.02),
        ))(x, update_stats=train)  # 28x28 -> 14x14
        h = nn.leaky_relu(h, 0.2)

        h = SN(nn.Conv(
            self.features * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            kernel_init=normal_init(0.02),
        ))(h, update_stats=train)  # 14x14 -> 7x7
        h = nn.leaky_relu(h, 0.2)

        # ----- D head (spatial conv reduction, SAME for full coverage) -----
        d = SN(nn.Conv(
            self.features * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            kernel_init=normal_init(0.02),
        ))(h, update_stats=train)  # 7x7 -> 4x4
        d = nn.leaky_relu(d, 0.2)

        d = SN(nn.Conv(
            1,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="VALID",
            kernel_init=normal_init(0.02),
        ))(d, update_stats=train)  # 4x4 -> 1x1
        d_logits = d.reshape((d.shape[0], -1))  # (B, 1)

        # ----- Q trunk -----
        # Brightness guard: remove global spatial bias before Q to reduce
        # steganographic "global intensity" shortcuts.
        h_q = h - jnp.mean(h, axis=(1, 2), keepdims=True)

        # Spatial Q head (no GAP): keeps topology-sensitive code prediction
        # while using a bottleneck to control parameter count.
        q = SN(nn.Conv(
            self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            kernel_init=normal_init(0.02),
        ))(h_q, update_stats=train)
        q = nn.leaky_relu(q, 0.2)

        q = SN(nn.Conv(
            self.features * 2,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding="VALID",
            kernel_init=normal_init(0.02),
        ))(q, update_stats=train)
        q = nn.leaky_relu(q, 0.2)
        q_flat = q.reshape((q.shape[0], -1))  # (B, features*2)
        q_feat_avg = q_flat

        # ----- Q categorical head -----
        q_cat_logits = SN(nn.Dense(
            self.q_cat,
            kernel_init=normal_init(0.02),
        ))(q_flat, update_stats=train)
        q_cont_mu = jnp.zeros((q_flat.shape[0], 0), dtype=q_flat.dtype)
        q_cont_logsigma = jnp.zeros((q_flat.shape[0], 0), dtype=q_flat.dtype)
        if self.q_cont > 0:
            q_cont_mu = SN(nn.Dense(
                self.q_cont,
                kernel_init=normal_init(0.02),
            ))(q_flat, update_stats=train)
            q_cont_logsigma = SN(nn.Dense(
                self.q_cont,
                kernel_init=normal_init(0.02),
            ))(q_flat, update_stats=train)

        return d_logits, q_cat_logits, q_cont_mu, q_cont_logsigma, q_feat_avg

# --- 1. The HyperNetwork (Learned Chunk Embeddings) ---
class HyperNetwork(nn.Module):
    chunk_size: int = 256
    n_chunks: int = 200
    chunk_embed_dim: int = 16

    @nn.compact
    def __call__(self, chunk_ids, context):
        """
        Input:  chunk_ids (N,) int  — chunk indices for learned embedding
                context   (N, C)    — layer one-hot + depth + scale
        Output: (N, chunk_size)     — generated weight chunks
        """
        chunk_emb = nn.Embed(self.n_chunks, self.chunk_embed_dim)(chunk_ids)
        x = jnp.concatenate([chunk_emb, context], axis=-1)

        x = nn.Dense(48)(x)
        x = nn.gelu(x)

        x = nn.Dense(48)(x)
        x = nn.gelu(x)

        weights = nn.Dense(
            self.chunk_size,
            kernel_init=jax.nn.initializers.normal(stddev=0.01)
        )(x)

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

        layer_ids_np = np.concatenate(layer_ids_list).astype(np.int32)
        chunk_ids_np = np.concatenate(chunk_ids_list).astype(np.int32)

        self.layer_ids = jnp.array(layer_ids_np)
        self.chunk_ids = jnp.array(chunk_ids_np)
        self.total_chunks = len(self.layer_ids)

        # --- C. Geometric Context ---
        total_layers = len(self.param_sizes)
        self.depth_map = np.linspace(0.0, 1.0, total_layers)
        self.scale_map = np.linspace(0.25, 1.0, total_layers)

        self.depths = jnp.array(self.depth_map)[self.layer_ids]
        self.scales = jnp.array(self.scale_map)[self.layer_ids]

        self.split_indices = np.cumsum(self.param_sizes)[:-1]

        # Per-layer chunk counts for proper reconstruction (avoids padding misalignment)
        self.layer_chunks_split = [(int(s) + chunk_size - 1) // chunk_size for s in self.param_sizes]
        self.layer_chunks_split_indices = np.cumsum(self.layer_chunks_split)[:-1].tolist()

        # --- D. Input Dimensions ---
        self.N_LAYERS = int(layer_ids_np.max()) + 1
        self.N_CHUNKS = int(chunk_ids_np.max()) + 1
        self.CHUNK_EMBED_DIM = 16
        self.CONTEXT_DIM = self.N_LAYERS + 2

        # --- PRE-CALCULATE STATIC CONTEXT (no chunk encoding) ---
        l_oh = jax.nn.one_hot(self.layer_ids, self.N_LAYERS)
        d_feat = self.depths[:, None]
        s_feat = self.scales[:, None]

        self.static_context = jnp.concatenate([l_oh, d_feat, s_feat], axis=-1)

    def _make_hn(self):
        return HyperNetwork(
            self.chunk_size, self.N_CHUNKS, self.CHUNK_EMBED_DIM)

    def init_hypernet(self, rng):
        dummy_chunk_ids = jnp.zeros((self.total_chunks,), dtype=jnp.int32)
        dummy_context = jnp.zeros((self.total_chunks, self.CONTEXT_DIM))
        return self._make_hn().init(rng, dummy_chunk_ids, dummy_context)

    def generate_params(self, hypernet_params):

        # 4. Run HyperNet
        flat_chunks = self._make_hn().apply(
            hypernet_params, self.chunk_ids, self.static_context)

        # 5. Reconstruct — split by layer first, then truncate padding per-layer
        chunks_per_layer = jnp.split(flat_chunks, self.layer_chunks_split_indices)

        reshaped_params = []
        for i, chunks in enumerate(chunks_per_layer):
            flat = chunks.reshape(-1)
            reshaped = flat[:self.param_sizes[i]].reshape(self.param_shapes[i])
            reshaped_params.append(reshaped)

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
def compute_fake_centroids(fake_features: jnp.ndarray, n_codes: int = 10) -> jnp.ndarray:
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
    fake_centroids = compute_fake_centroids(fake_features, n_codes=10)
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


def continuous_loss(c_true, mu, logsigma):
    """Gaussian NLL for InfoGAN continuous codes."""
    logsigma = jnp.clip(logsigma, -2.0, 2.0)
    nll = logsigma + 0.5 * ((c_true - mu) / jnp.exp(logsigma)) ** 2
    # Cap per-element NLL to prevent gradient explosion when mu is far from
    # truth.  Without this the cont term can dominate D-step loss by 100x,
    # destabilising backbone training and poisoning the G-step MI signal.
    nll = jnp.minimum(nll, 10.0)
    return jnp.mean(nll)

@partial(jax.jit, static_argnames=['solver', 'n_codes', 'n_cont', 'disc_features'])
def train_step_disc(
        state,
        data,
        noise_shift_tup,
        fake_imgs,
        fake_cat_input,
        fake_cont_input,
        solver,
        n_codes: int = 10,
        n_cont: int = 2,
        disc_features: int = 64):

        noise, _, _ = noise_shift_tup
        params_d, batch_stats_d, opt_disc = state

        def loss_mutual_information(code_cat, q_cat):
            return -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))

        def loss_discriminator(params_d, vars_d_batch_stats):
            fake_images_with_noise = fake_imgs + noise
            real_images_with_noise = data + noise

            (fake_preds, q_fake_cat, q_fake_mu, q_fake_logsigma, _), vars_d = Discriminator(
                features=disc_features,
                q_cat=n_codes,
                q_cont=n_cont,
            ).apply(
                {'params': params_d, 'batch_stats': vars_d_batch_stats},
                fake_images_with_noise,
                mutable=['batch_stats'],
            )

            (real_preds, _, _, _, _), vars_d = Discriminator(
                features=disc_features,
                q_cat=n_codes,
                q_cont=n_cont,
            ).apply(
                {'params': params_d, 'batch_stats': vars_d['batch_stats']},
                real_images_with_noise,
                mutable=['batch_stats'],
            )

            q_cat = nn.log_softmax(q_fake_cat, axis=-1)
            loss_mi_cat = loss_mutual_information(fake_cat_input, q_cat)
            if n_cont > 0:
                q_fake_mu = jnp.nan_to_num(q_fake_mu, nan=0.0, posinf=0.0, neginf=0.0)
                q_fake_logsigma = jnp.nan_to_num(q_fake_logsigma, nan=0.0, posinf=0.0, neginf=0.0)
                loss_mi_cont = continuous_loss(fake_cont_input, q_fake_mu, q_fake_logsigma)
            else:
                loss_mi_cont = 0.0
            #loss_mi = loss_mi_cat + loss_mi_cont

            # use 0.9 as the label for real images instead of 1.0
            real_loss = optax.sigmoid_binary_cross_entropy(
                real_preds, jnp.ones_like(real_preds) * 0.9)
            # use 0.05 as the label for fake images instead of 0.0
            fake_loss = optax.sigmoid_binary_cross_entropy(
                fake_preds, jnp.zeros_like(fake_preds) + 0.05)

            real_loss = jnp.mean(real_loss)
            fake_loss = jnp.mean(fake_loss)

            real_fake_loss = (real_loss + fake_loss) / 2.0
            loss = real_fake_loss + loss_mi_cat*0.4 + loss_mi_cont * 0.1

            return loss, (real_fake_loss, vars_d)

        grad_fn_disc = jax.value_and_grad(loss_discriminator, has_aux=True)
        (loss, (real_fake_loss, vars_d)), grads = grad_fn_disc(params_d, batch_stats_d)

        updates, new_opt_state = solver.update(grads, opt_disc, params_d)
        params_d = optax.apply_updates(params_d, updates)
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
                         z_dim=62, n_disc=10):
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

def sample_latent(key, shape_noise, shape_cat, n_disc: int = 10, n_cont: int = 2):
  noise_key, cat_key, cont_key = jax.random.split(key, 3)

  # Sample irreducible noise
  noise = jax.random.normal(noise_key, shape_noise)

  # Sample categorical latent code
  code_cat = jax.random.randint(cat_key, shape_cat, 0, n_disc)
  code_cat = jax.nn.one_hot(code_cat, n_disc)

  if n_cont > 0:
      code_cont = jax.random.uniform(
          cont_key, (shape_noise[0], n_cont), minval=-1.0, maxval=1.0)
      latent = jnp.concatenate([noise, code_cat, code_cont], axis=-1)
  else:
      code_cont = jnp.zeros((shape_noise[0], 0), dtype=noise.dtype)
      latent = jnp.concatenate([noise, code_cat], axis=-1)

  return latent, code_cat, code_cont

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

        self.latent_dim = int(getattr(train_task_gen, 'latent_dim', 62))
        self.n_classes = int(getattr(train_task_gen, 'n_classes', 10))
        self.n_cont = int(getattr(train_task_gen, 'n_cont', 2))
        self.disc_features = int(getattr(policy_gen.model_disc, 'features', 64))
        self.disc_feature_dim = int(getattr(policy_gen, 'disc_feature_dim', self.disc_features * 2))

        self.decay_factor = 0.9

        self.noise_dim = self.latent_dim
        # latent_dim from task is noise-only (62), not total (72).
        # Generator code-path conditioning uses the noise_dim offset to
        # locate the discrete one-hot code slice in z.
        self.actual_noise_dim = self.latent_dim
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
        self._metrics_tsv_path = os.path.join(log_dir, 'metrics.tsv') if log_dir else None
        self._ks_tsv_path = os.path.join(log_dir, 'ks_weights.tsv') if log_dir else None
        self._metrics_tsv_header = [
            'iter',
            'adv_max', 'adv_avg', 'adv_min', 'adv_std',
            'mi_max', 'mi_avg', 'mi_min', 'mi_std',
            'r_cons_avg', 'r_sense_avg', 'r_intra_avg', 'r_shape_div_avg', 'r_shape_div_min_avg',
            'morph_dark_range_avg', 'morph_center_edge_range_avg', 'edge_dark_frac_avg', 'code_proto_corr_avg',
            'proto_angle_spread_avg', 'nuc_cell_ratio_range_avg', 'nuc_eccentricity_range_avg',
            'cell_circularity_avg', 'cell_area_var_avg', 'nucleus_offset_avg',
            'norm_pen_avg', 'safety_avg', 'spread_avg',
            'real_fake_loss',
            'ca_blend', 'ca_rfl_gate', 'ca_has_data',
            'd_health', 'shape_health', 'objective_health', 'static_fitness_weights',
            'ks_winner',
            # Exploration & LR metrics
            'stdev_mean', 'stdev_std', 'stdev_min', 'stdev_max',
            'effective_center_lr',
        ]
        self._ks_tsv_header = [
            'iter',
            'adv_short', 'mi_short', 'adv_med', 'mi_med', 'ent_long',
            'sense_short', 'intra_short', 'adv_avg_short', 'sense_med', 'shape_short', 'shape_med',
            'spread_short', 'spread_med',
            'dark_range_short', 'code_corr_short', 'angle_spread_short',
            'dark_range_latest', 'code_corr_latest', 'angle_spread_latest',
            'cell_circularity_latest', 'cell_area_var_latest', 'nucleus_offset_latest',
            'nuc_ratio_latest', 'nuc_ecc_latest', 'bio_score_latest',
            'norm_cell_circularity_floor', 'norm_cell_area_var_ceiling',
            'norm_nucleus_offset_low', 'norm_nucleus_offset_high',
            'norm_circ_violation', 'norm_area_violation', 'norm_offset_violation',
            'norm_morph_violation', 'norm_active',
            'norm_dom_penalty', 'norm_hist_boost', 'norm_sit_boost', 'norm_stdev_scale',
            'shape_div_avg', 'shape_div_min_avg', 'shape_div_score_avg',
            'w_adv', 'w_mi', 'w_div', 'w_sense', 'w_intra', 'w_shape', 'mi_guard', 'w_cons_floor', 'w_norm',
            'w_dark_range', 'w_center_edge_range', 'w_edge_dark_penalty', 'w_code_corr',
            'd_health', 'shape_health', 'objective_health',
            'ks_dom_score', 'ks_sit_score', 'ks_hist_score', 'ks_topo_score',
            'ks_dom_weight', 'ks_sit_weight', 'ks_hist_weight', 'ks_topo_weight',
            'semantic_trap_score', 'semantic_trap_active',
            'semantic_dom_penalty', 'semantic_hist_boost', 'semantic_topo_boost', 'semantic_stdev_boost',
            'ks_winner',
            'ks_win_dom', 'ks_win_sit', 'ks_win_hist', 'ks_win_topo',
            # CA gradient & exploration insight
            'reinforce_grad_norm', 'ca_grad_norm',
            'reinforce_stdev_grad_norm', 'ca_stdev_grad_norm',
            'ca_stdev_direction', 'ca_stdev_rel_delta',
            'reinforce_stdev_direction',
            'archive_stdev_diversity', 'archive_stdev_range', 'archive_occupancy',
        ]

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
        
        if hasattr(train_task_disc, 'data') and hasattr(train_task_disc, 'labels'):
            data_raw = np.array(train_task_disc.data, dtype=np.float32)
            if data_raw.ndim == 3:
                data_raw = np.expand_dims(data_raw, axis=-1)
            self.data = data_raw if data_raw.max() <= 1.0 else data_raw / 255.0
            self.labels = np.array(train_task_disc.labels, dtype=np.int32).flatten()
        else:
            dataset = torchvision.datasets.MNIST(
                './data', train=True, download=True)
            data_raw = np.array(dataset.data, dtype=np.float32)
            if data_raw.ndim == 3:
                data_raw = np.expand_dims(data_raw, axis=-1)
            self.data = data_raw / 255.0
            self.labels = np.array(dataset.targets, dtype=np.int32).flatten()

        self._key, subkey = jax.random.split(self._key)

        # initialize the discriminator
        variables_disc = Discriminator(
            features=self.disc_features,
            q_cat=self.n_classes,
            q_cont=self.n_cont,
        ).init(subkey, jnp.ones((self.batch_size, 28, 28, 1), dtype=jnp.float32))
        self.params_disc, self.batch_stats_disc = variables_disc['params'], variables_disc['batch_stats']

        self.solver_disc = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.999)

    def _init_structured_logs(self, resume_mode: bool) -> None:
        if self._log_dir is None:
            return
        os.makedirs(self._log_dir, exist_ok=True)

        def _ensure_file(path, header, append):
            if path is None:
                return
            file_exists = os.path.exists(path)
            mode = 'a' if append else 'w'
            if (not file_exists) and append:
                mode = 'w'
            with open(path, mode, encoding='utf-8') as fp:
                if (not file_exists) or mode == 'w':
                    fp.write('\t'.join(header) + '\n')

        _ensure_file(self._metrics_tsv_path, self._metrics_tsv_header, append=resume_mode)
        _ensure_file(self._ks_tsv_path, self._ks_tsv_header, append=resume_mode)

    @staticmethod
    def _to_scalar(value, default=np.nan) -> float:
        if value is None:
            return float(default)
        arr = np.asarray(value)
        if arr.size == 0:
            return float(default)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.reshape(-1)[0])

    @staticmethod
    def _nan_stat(arr, fn_name: str) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            return float(np.nan)
        fn = getattr(np, fn_name)
        return float(fn(arr))

    @staticmethod
    def _format_tsv_value(value) -> str:
        arr = np.asarray(value)
        if arr.ndim == 0:
            v = float(arr)
            if np.isnan(v):
                return 'nan'
            if np.isinf(v):
                return 'inf' if v > 0 else '-inf'
            return f'{v:.6f}'
        return str(arr.tolist())

    def _append_tsv_row(self, path: Optional[str], row_values) -> None:
        if path is None:
            return
        with open(path, 'a', encoding='utf-8') as fp:
            fp.write('\t'.join(self._format_tsv_value(v) for v in row_values) + '\n')

    def _write_structured_rows(
            self,
            iteration: int,
            scores_gen_adv: np.ndarray,
            scores_gen_mi: np.ndarray,
            r_cons: np.ndarray,
            r_sense: np.ndarray,
            r_intra: np.ndarray,
            r_shape_div: np.ndarray,
            r_shape_div_min: np.ndarray,
            morph_dark_range: np.ndarray,
            morph_center_edge_range: np.ndarray,
            edge_dark_frac: np.ndarray,
            code_proto_corr: np.ndarray,
            proto_angle_spread: np.ndarray,
            nuc_cell_ratio_range: np.ndarray,
            nuc_eccentricity_range: np.ndarray,
            cell_circularity: np.ndarray,
            cell_area_var: np.ndarray,
            nucleus_offset: np.ndarray,
            norm_pen: np.ndarray,
            safety_ratios: np.ndarray,
            spreads: np.ndarray,
            real_fake_loss: float) -> None:
        if self._metrics_tsv_path is None or self._ks_tsv_path is None:
            return

        diagnostics = {}
        if hasattr(self.solver_hn, 'get_diagnostics'):
            try:
                diagnostics = self.solver_hn.get_diagnostics() or {}
            except Exception as err:
                self._logger.warning('Structured diagnostics unavailable at iter %d: %s', iteration, err)

        metrics_row = [
            iteration,
            self._nan_stat(scores_gen_adv, 'nanmax'),
            self._nan_stat(scores_gen_adv, 'nanmean'),
            self._nan_stat(scores_gen_adv, 'nanmin'),
            self._nan_stat(scores_gen_adv, 'nanstd'),
            self._nan_stat(scores_gen_mi, 'nanmax'),
            self._nan_stat(scores_gen_mi, 'nanmean'),
            self._nan_stat(scores_gen_mi, 'nanmin'),
            self._nan_stat(scores_gen_mi, 'nanstd'),
            self._nan_stat(r_cons, 'nanmean'),
            self._nan_stat(r_sense, 'nanmean'),
            self._nan_stat(r_intra, 'nanmean'),
            self._nan_stat(r_shape_div, 'nanmean'),
            self._nan_stat(r_shape_div_min, 'nanmean'),
            self._nan_stat(morph_dark_range, 'nanmean'),
            self._nan_stat(morph_center_edge_range, 'nanmean'),
            self._nan_stat(edge_dark_frac, 'nanmean'),
            self._nan_stat(code_proto_corr, 'nanmean'),
            self._nan_stat(proto_angle_spread, 'nanmean'),
            self._nan_stat(nuc_cell_ratio_range, 'nanmean'),
            self._nan_stat(nuc_eccentricity_range, 'nanmean'),
            self._nan_stat(cell_circularity, 'nanmean'),
            self._nan_stat(cell_area_var, 'nanmean'),
            self._nan_stat(nucleus_offset, 'nanmean'),
            self._nan_stat(norm_pen, 'nanmean'),
            self._nan_stat(safety_ratios, 'nanmean'),
            self._nan_stat(spreads, 'nanmean'),
            float(real_fake_loss),
            self._to_scalar(diagnostics.get('ca_blend')),
            self._to_scalar(diagnostics.get('ca_rfl_gate')),
            self._to_scalar(diagnostics.get('ca_has_data')),
            self._to_scalar(diagnostics.get('d_health')),
            self._to_scalar(diagnostics.get('shape_health')),
            self._to_scalar(diagnostics.get('objective_health')),
            self._to_scalar(diagnostics.get('static_fitness_weights')),
            self._to_scalar(diagnostics.get('ks_winner')),
            # Exploration & LR metrics
            self._to_scalar(diagnostics.get('stdev_mean')),
            self._to_scalar(diagnostics.get('stdev_std')),
            self._to_scalar(diagnostics.get('stdev_min')),
            self._to_scalar(diagnostics.get('stdev_max')),
            self._to_scalar(diagnostics.get('effective_center_lr')),
        ]
        self._append_tsv_row(self._metrics_tsv_path, metrics_row)

        ks_winner_counts = diagnostics.get('ks_winner_counts')
        if ks_winner_counts is None:
            ks_winner_counts = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        else:
            ks_winner_counts = np.asarray(ks_winner_counts, dtype=np.float64).reshape(-1)
            if ks_winner_counts.size < 4:
                ks_winner_counts = np.pad(
                    ks_winner_counts,
                    (0, 4 - ks_winner_counts.size),
                    mode='constant',
                    constant_values=np.nan
                )

        ks_row = [
            iteration,
            self._to_scalar(diagnostics.get('adv_short')),
            self._to_scalar(diagnostics.get('mi_short')),
            self._to_scalar(diagnostics.get('adv_med')),
            self._to_scalar(diagnostics.get('mi_med')),
            self._to_scalar(diagnostics.get('ent_long')),
            self._to_scalar(diagnostics.get('sense_short')),
            self._to_scalar(diagnostics.get('intra_short')),
            self._to_scalar(diagnostics.get('adv_avg_short')),
            self._to_scalar(diagnostics.get('sense_med')),
            self._to_scalar(diagnostics.get('shape_short')),
            self._to_scalar(diagnostics.get('shape_med')),
            self._to_scalar(diagnostics.get('spread_short')),
            self._to_scalar(diagnostics.get('spread_med')),
            self._to_scalar(diagnostics.get('dark_range_short')),
            self._to_scalar(diagnostics.get('code_corr_short')),
            self._to_scalar(diagnostics.get('angle_spread_short')),
            self._to_scalar(diagnostics.get('dark_range_latest')),
            self._to_scalar(diagnostics.get('code_corr_latest')),
            self._to_scalar(diagnostics.get('angle_spread_latest')),
            self._to_scalar(diagnostics.get('cell_circularity_latest')),
            self._to_scalar(diagnostics.get('cell_area_var_latest')),
            self._to_scalar(diagnostics.get('nucleus_offset_latest')),
            self._to_scalar(diagnostics.get('nuc_ratio_latest')),
            self._to_scalar(diagnostics.get('nuc_ecc_latest')),
            self._to_scalar(diagnostics.get('bio_score_latest')),
            self._to_scalar(diagnostics.get('norm_cell_circularity_floor')),
            self._to_scalar(diagnostics.get('norm_cell_area_var_ceiling')),
            self._to_scalar(diagnostics.get('norm_nucleus_offset_low')),
            self._to_scalar(diagnostics.get('norm_nucleus_offset_high')),
            self._to_scalar(diagnostics.get('norm_circ_violation')),
            self._to_scalar(diagnostics.get('norm_area_violation')),
            self._to_scalar(diagnostics.get('norm_offset_violation')),
            self._to_scalar(diagnostics.get('norm_morph_violation')),
            self._to_scalar(diagnostics.get('norm_active')),
            self._to_scalar(diagnostics.get('norm_dom_penalty')),
            self._to_scalar(diagnostics.get('norm_hist_boost')),
            self._to_scalar(diagnostics.get('norm_sit_boost')),
            self._to_scalar(diagnostics.get('norm_stdev_scale')),
            self._to_scalar(diagnostics.get('shape_div_avg')),
            self._to_scalar(diagnostics.get('shape_div_min_avg')),
            self._to_scalar(diagnostics.get('shape_div_score_avg')),
            self._to_scalar(diagnostics.get('w_adv')),
            self._to_scalar(diagnostics.get('w_mi')),
            self._to_scalar(diagnostics.get('w_div')),
            self._to_scalar(diagnostics.get('w_sense')),
            self._to_scalar(diagnostics.get('w_intra')),
            self._to_scalar(diagnostics.get('w_shape')),
            self._to_scalar(diagnostics.get('mi_guard')),
            self._to_scalar(diagnostics.get('w_cons_floor')),
            self._to_scalar(diagnostics.get('w_norm')),
            self._to_scalar(diagnostics.get('w_dark_range')),
            self._to_scalar(diagnostics.get('w_center_edge_range')),
            self._to_scalar(diagnostics.get('w_edge_dark_penalty')),
            self._to_scalar(diagnostics.get('w_code_corr')),
            self._to_scalar(diagnostics.get('d_health')),
            self._to_scalar(diagnostics.get('shape_health')),
            self._to_scalar(diagnostics.get('objective_health')),
            self._to_scalar(diagnostics.get('ks_dom_score')),
            self._to_scalar(diagnostics.get('ks_sit_score')),
            self._to_scalar(diagnostics.get('ks_hist_score')),
            self._to_scalar(diagnostics.get('ks_topo_score')),
            self._to_scalar(diagnostics.get('ks_dom_weight')),
            self._to_scalar(diagnostics.get('ks_sit_weight')),
            self._to_scalar(diagnostics.get('ks_hist_weight')),
            self._to_scalar(diagnostics.get('ks_topo_weight')),
            self._to_scalar(diagnostics.get('semantic_trap_score')),
            self._to_scalar(diagnostics.get('semantic_trap_active')),
            self._to_scalar(diagnostics.get('semantic_dom_penalty')),
            self._to_scalar(diagnostics.get('semantic_hist_boost')),
            self._to_scalar(diagnostics.get('semantic_topo_boost')),
            self._to_scalar(diagnostics.get('semantic_stdev_boost')),
            self._to_scalar(diagnostics.get('ks_winner')),
            self._to_scalar(ks_winner_counts[0]),
            self._to_scalar(ks_winner_counts[1]),
            self._to_scalar(ks_winner_counts[2]),
            self._to_scalar(ks_winner_counts[3]),
            # CA gradient & exploration insight
            self._to_scalar(diagnostics.get('reinforce_grad_norm')),
            self._to_scalar(diagnostics.get('ca_grad_norm')),
            self._to_scalar(diagnostics.get('reinforce_stdev_grad_norm')),
            self._to_scalar(diagnostics.get('ca_stdev_grad_norm')),
            self._to_scalar(diagnostics.get('ca_stdev_direction')),
            self._to_scalar(diagnostics.get('ca_stdev_rel_delta')),
            self._to_scalar(diagnostics.get('reinforce_stdev_direction')),
            self._to_scalar(diagnostics.get('archive_stdev_diversity')),
            self._to_scalar(diagnostics.get('archive_stdev_range')),
            self._to_scalar(diagnostics.get('archive_occupancy')),
        ]
        self._append_tsv_row(self._ks_tsv_path, ks_row)

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

            self._init_structured_logs(resume_mode=(start_iter > 0))
           
            num_mini_batches = self.num_mini_batches
           
            self._key, noise_key = jax.random.split(self._key, 2)

            # Build a balanced preview panel for saved iterations so each code
            # has identical sample count (avoids remainder-code bias in columns).
            viz_repeats = max(1, self.batch_size // self.n_classes)
            viz_batch_size = viz_repeats * self.n_classes
            fixed_batch_latent = jax.random.normal(noise_key, (viz_batch_size, self.latent_dim))
            fixed_c = jnp.tile(jnp.arange(self.n_classes), viz_repeats)
            fixed_parts = [
                fixed_batch_latent,
                jax.nn.one_hot(fixed_c, self.n_classes),
            ]
            if self.n_cont > 0:
                self._key, fixed_cont_key = jax.random.split(self._key)
                fixed_cont = jax.random.uniform(
                    fixed_cont_key, (viz_batch_size, self.n_cont), minval=-1.0, maxval=1.0)
                fixed_parts.append(fixed_cont)
            fixed_latent = jnp.concatenate(fixed_parts, axis=-1)

            self.ordered_centroids = jnp.zeros((self.n_classes, self.disc_feature_dim))

            real_fake_loss = 0.0  # Default for logging when D is not trained

            for i in range(start_iter, self._max_iter):

                shape_noise = (self.mini_batch_size, self.latent_dim)
                shape_cat = (self.mini_batch_size,)
                
                # D-step frequency: reduce D updates early so G can establish
                # diversity before D crushes it. Every 3rd iter for first 1.5k,
                # then every iteration after.
                d_freq = 2 if i < 1000 else 1
                if i % d_freq == 0:
                    for mini_batch in range(num_mini_batches):
                        # Sample batch of data.

                        self._key, subkey_latent, subkey_mnist, subkey_noise, subkey_shift_x, subkey_shift_y = jax.random.split(self._key,6)


                        data, labels = sample_batch(subkey_mnist, self.data, self.labels, self.mini_batch_size)
                        #data = np.expand_dims(data / 255.0, axis=-1)

                        latent, cat_codes, cont_codes = sample_latent(
                            subkey_latent,
                            shape_noise,
                            shape_cat,
                            n_disc=self.n_classes,
                            n_cont=self.n_cont,
                        )

                        noise = jax.random.normal(subkey_noise, (self.mini_batch_size, 28, 28, 1)) * 0.1

                        shift_x = jax.random.randint(subkey_shift_x, shape=(), minval=-1, maxval=2)
                        shift_y = jax.random.randint(subkey_shift_y, shape=(), minval=-1, maxval=2)

                        #if i < 2:
                        params_hn = self.solver_hn.best_params
                        best_params_hn_formatted = self.policy_gen._format_single_params_hypernet_fn(params_hn)
                        #else:

                        params_g = self.adapter.generate_params(best_params_hn_formatted)

                        (fake_images) = Generator(training=False, n_codes=self.n_classes, noise_dim=self.actual_noise_dim).apply({'params': params_g},latent)

                        # reshape fake_images to (64, 28, 28, 1) from [1,1,1,64, 28, 28, 1]
                        fake_images = fake_images.reshape((self.mini_batch_size, 28, 28, 1))

                        state = (params_disc, self.batch_stats_disc, opt_disc)

                        state, d_loss, real_fake_loss = train_step_disc(
                            state,
                            data,
                            (noise,shift_x, shift_y),
                            fake_images,
                            cat_codes,
                            cont_codes,
                            solver_disc,
                            self.n_classes,
                            self.n_cont,
                            self.disc_features,
                        )
#jax.debug.print('loss: {} ', loss)
                        if real_fake_loss > 0.3:
                            params_disc, self.batch_stats_disc, opt_disc = state

                leaves_params, _ = jax.tree_flatten(params_disc) 
                flat_params_disc = jnp.concatenate([p.flatten() for p in leaves_params])

                leaves_batch_stats_disc, _ = jax.tree_flatten(self.batch_stats_disc)
                flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

                # Feed discriminator health into CA blend gating (if supported).
                if hasattr(self.solver_hn, 'set_runtime_metrics'):
                    self.solver_hn.set_runtime_metrics(real_fake_loss=float(real_fake_loss))

                params_hn, belief_space = self.solver_hn.ask()
                
                topographic_ks = belief_space[4]
                normative_ks = belief_space[5]

                #jax.debug.print('topographic_ks shape: {} ', topographic_ks.shape)
                #avg_per_code = topographic_ks[0]
                
                scores_gen_adv, scores_gen_mi, disc_logits, bds_gen, _, mean_var_fake, avg_per_code_current, r_cons, r_sense, r_intra, r_shape_div, r_shape_div_min, morph_dark_range, morph_center_edge_range, edge_dark_frac, code_proto_corr, proto_angle_spread, nuc_cell_ratio_range, nuc_eccentricity_range, cell_circularity, cell_area_var, nucleus_offset, norm_pen, safety_ratios, spreads = self.sim_mgr_gen.eval_params(
                params_gen=params_hn, params_disc=flat_params_disc, batch_stats_disc=flat_batch_stats_disc, topographic_ks=topographic_ks, normative_ks=normative_ks, generator=True, test=False
                )

                #jax.debug.print('fake_imgs shape: {} ', fake_imgs.shape)
                if isinstance(self.solver_hn, QualityDiversityMethod):
                    self.solver_hn.observe_bd(bds_gen)
                
                self.solver_hn.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, disc_logits=disc_logits, pop_var=mean_var_fake, avg_per_code=avg_per_code_current, r_cons=r_cons, r_sense=r_sense, r_intra=r_intra, r_shape_div=r_shape_div, r_shape_div_min=r_shape_div_min, morph_dark_range=morph_dark_range, morph_center_edge_range=morph_center_edge_range, edge_dark_frac=edge_dark_frac, code_proto_corr=code_proto_corr, proto_angle_spread=proto_angle_spread, nuc_cell_ratio_range=nuc_cell_ratio_range, nuc_eccentricity_range=nuc_eccentricity_range, cell_circularity=cell_circularity, cell_area_var=cell_area_var, nucleus_offset=nucleus_offset, normative_penalty=norm_pen, safety_ratios=safety_ratios, spreads=spreads, adv=False)

                
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

                    r_shape_div = np.array(r_shape_div)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, r_shape_div.size, r_shape_div.max(), r_shape_div.mean(),
                            r_shape_div.min(), r_shape_div.std()))

                    r_shape_div_min = np.array(r_shape_div_min)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, r_shape_div_min.size, r_shape_div_min.max(), r_shape_div_min.mean(),
                            r_shape_div_min.min(), r_shape_div_min.std()))

                    morph_dark_range = np.array(morph_dark_range)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, morph_dark_range.size, morph_dark_range.max(), morph_dark_range.mean(),
                            morph_dark_range.min(), morph_dark_range.std()))

                    morph_center_edge_range = np.array(morph_center_edge_range)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, morph_center_edge_range.size, morph_center_edge_range.max(), morph_center_edge_range.mean(),
                            morph_center_edge_range.min(), morph_center_edge_range.std()))

                    edge_dark_frac = np.array(edge_dark_frac)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, edge_dark_frac.size, edge_dark_frac.max(), edge_dark_frac.mean(),
                            edge_dark_frac.min(), edge_dark_frac.std()))

                    code_proto_corr = np.array(code_proto_corr)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, code_proto_corr.size, code_proto_corr.max(), code_proto_corr.mean(),
                            code_proto_corr.min(), code_proto_corr.std()))

                    proto_angle_spread = np.array(proto_angle_spread)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, proto_angle_spread.size, proto_angle_spread.max(), proto_angle_spread.mean(),
                            proto_angle_spread.min(), proto_angle_spread.std()))

                    nuc_cell_ratio_range = np.array(nuc_cell_ratio_range)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, nuc_cell_ratio_range.size, nuc_cell_ratio_range.max(), nuc_cell_ratio_range.mean(),
                            nuc_cell_ratio_range.min(), nuc_cell_ratio_range.std()))

                    nuc_eccentricity_range = np.array(nuc_eccentricity_range)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, nuc_eccentricity_range.size, nuc_eccentricity_range.max(), nuc_eccentricity_range.mean(),
                            nuc_eccentricity_range.min(), nuc_eccentricity_range.std()))

                    cell_circularity = np.array(cell_circularity)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, cell_circularity.size, cell_circularity.max(), cell_circularity.mean(),
                            cell_circularity.min(), cell_circularity.std()))

                    cell_area_var = np.array(cell_area_var)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, cell_area_var.size, cell_area_var.max(), cell_area_var.mean(),
                            cell_area_var.min(), cell_area_var.std()))

                    nucleus_offset = np.array(nucleus_offset)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, nucleus_offset.size, nucleus_offset.max(), nucleus_offset.mean(),
                            nucleus_offset.min(), nucleus_offset.std()))

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

                    self._write_structured_rows(
                        iteration=i,
                        scores_gen_adv=scores_gen_adv,
                        scores_gen_mi=scores_mi,
                        r_cons=r_cons,
                        r_sense=r_sense,
                        r_intra=r_intra,
                        r_shape_div=r_shape_div,
                        r_shape_div_min=r_shape_div_min,
                        morph_dark_range=morph_dark_range,
                        morph_center_edge_range=morph_center_edge_range,
                        edge_dark_frac=edge_dark_frac,
                        code_proto_corr=code_proto_corr,
                        proto_angle_spread=proto_angle_spread,
                        nuc_cell_ratio_range=nuc_cell_ratio_range,
                        nuc_eccentricity_range=nuc_eccentricity_range,
                        cell_circularity=cell_circularity,
                        cell_area_var=cell_area_var,
                        nucleus_offset=nucleus_offset,
                        norm_pen=norm_pen,
                        safety_ratios=safety_ratios,
                        spreads=spreads,
                        real_fake_loss=float(real_fake_loss),
                    )
                    
                    #with open('/home/gh0st/Downloads/pgpe_main.csv', 'a') as file:
                        #file.write(f'Iter: {i}, Max: {scores.max()}, Mean: {scores.mean()}, Std: {scores.std()}, Min: {scores.min()}\n')
                    #self._log_scores_fn(i, scores, "train")

                if i > 0 and i % self._test_interval == 0:
                    best_params_hn = self.solver_hn.best_params
                    best_params_hn_formatted = self.policy_gen._format_single_params_hypernet_fn(best_params_hn)

                    self._key, noise_key, con_key = jax.random.split(self._key, 3)

                    params_g = self.adapter.generate_params(best_params_hn_formatted)

                    (fake_imgs) = Generator(training=False, n_codes=self.n_classes, noise_dim=self.actual_noise_dim).apply({'params': params_g},fixed_latent)
                    
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
                'Training done, best_score_gen={0:.4f}'.format(best_score_gen))

            return best_score_gen

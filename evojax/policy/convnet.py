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
from typing import Tuple
from jax import tree_util

import orbax.checkpoint as orbax_cp
import optax
import jax
import numpy as np
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state

from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import he_normal
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn, get_single_params_format_fn, get_params_format_disc_fn


class CNN(nn.Module):
    """CNN for MNIST."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=8, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x

def generate_latent_points(rng, latent_dim, n_samples):
      rng, latent_rng, cat_rng = jrandom.split(rng, num=3)
      latent_inputs = jrandom.normal(latent_rng, (n_samples, latent_dim))
      cat_codes = jrandom.randint(cat_rng, (n_samples,), 0, 11)
      cat_codes = nn.one_hot(cat_codes, 11) 
      z_input = jnp.concatenate((latent_inputs, cat_codes), axis=1)
    
      return z_input, cat_codes

def create_train_state(rng, learning_rate=1e-3):
    model = BinaryMNISTClassifier() 
    dummy_input = jnp.ones((1, 784), jnp.float32)
    params = model.init(rng, dummy_input)['params']

    tx = optax.adam(learning_rate)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return state, model

def load_model(state, path):
    checkpointer = orbax_cp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(path, item=state)
    return restored_state

# --- 1. The HyperNetwork (Now with Geometric Input) ---
class HyperNetwork(nn.Module):
    chunk_size: int = 256
    
    @nn.compact
    def __call__(self, inputs):
        """
        Input:  (Batch, Input_Dim) 
                Input_Dim = Layer_OneHot + Chunk_OneHot + 2 (Depth, Scale)
        Output: (Batch, Chunk_Size)
        """
        # We start with a slightly wider first layer to handle the mixed inputs
        x = nn.Dense(48)(inputs) 
        x = nn.gelu(x)
        
        x = nn.Dense(48)(x)
        x = nn.gelu(x)
        
        # Initialize output with higher variance as discussed to ensure signal strength
        weights = nn.Dense(
            self.chunk_size, 
            kernel_init=jax.nn.initializers.normal(stddev=0.025) 
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
        
        # --- C. Define Geometric Context (The New Part) ---
        # We manually map each layer index to a "Depth" (0-1) and "Scale" (0-1)
        # Assuming the Generator order: [Dense(Start), GroupNorm, Conv(7x7), GN, Conv(14x14), GN, Conv(28x28)]
        # You can adjust these based on your exact parameter list order.
        # This is a heuristic: Start=0.0, End=1.0. 
        total_layers = len(self.param_sizes)
        self.depth_map = np.linspace(0.0, 1.0, total_layers)
        
        # For Scale, we map based on expected resolution.
        # We create an array matching 'param_sizes' length.
        # 0.25 = 7x7, 0.5 = 14x14, 1.0 = 28x28
        # (Simplified: Just using increasing scale for deeper layers)
        self.scale_map = np.linspace(0.25, 1.0, total_layers)

        # Convert to JAX arrays for the GPU
        self.depths = jnp.array(self.depth_map)[self.layer_ids] 
        self.scales = jnp.array(self.scale_map)[self.layer_ids]

        self.split_indices = np.cumsum(self.param_sizes)[:-1]

        # --- D. Input Dimensions ---
        # Derive embedding sizes from the actual generator parameter layout.
        # This avoids silent truncation when capacity changes increase chunks.
        self.N_LAYERS = int(layer_ids_np.max()) + 1
        self.N_CHUNKS = int(chunk_ids_np.max()) + 1
        # +2 comes from the new Depth and Scale features
        self.INPUT_DIM = self.N_LAYERS + self.N_CHUNKS + 2 

        # --- PRE-CALCULATE EMBEDDINGS ONCE ---
        l_oh = jax.nn.one_hot(self.layer_ids, self.N_LAYERS)      
        c_oh = jax.nn.one_hot(self.chunk_ids, self.N_CHUNKS)     
        d_feat = self.depths[:, None]
        s_feat = self.scales[:, None]

        self.static_embeddings = jnp.concatenate([l_oh, c_oh, d_feat, s_feat], axis=-1)

    def init_hypernet(self, rng):
        dummy_input = jnp.zeros((self.total_chunks, self.INPUT_DIM))
        return HyperNetwork(self.chunk_size).init(rng, dummy_input)

    def generate_params(self, hypernet_params):

        # 4. Run HyperNet
        flat_chunks = HyperNetwork(self.chunk_size).apply(hypernet_params, self.static_embeddings)
        
        # 5. Reconstruct
        raw_stream = flat_chunks.reshape(-1)
        total_gen_params = self.split_indices[-1] + self.param_sizes[-1]
        valid_stream = raw_stream[:total_gen_params]
        param_list = jnp.split(valid_stream, self.split_indices)
        
        reshaped_params = [
            p.reshape(s) for p, s in zip(param_list, self.param_shapes)
        ]
        return tree_util.tree_unflatten(self.target_tree, reshaped_params)

class Generator(nn.Module):
    features: int = 64
    training: bool = True
    
    @nn.compact
    def __call__(self, z):
                
        # BETTER: Project z -> 7*7*8 (small depth) -> Conv to 64
        x = nn.Dense(7 * 7 * 8)(z) # 74 -> 392 outputs = 29k params. Very manageable.
        x = x.reshape((x.shape[0], 7, 7, 8))
        
        # Now use a Conv to expand depth (standard HyperNet texture generation)
        x = nn.Conv(self.features, kernel_size=(3,3), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = jnp.tanh(x)
        
        # ... Rest of the Resize-Conv network ...
        # 2. UPSAMPLE BLOCK 1 (7x7 -> 14x14)
        # Resize: Nearest Neighbor is clean and sharp (no ringing).
        x = jax.image.resize(x, shape=(x.shape[0], 14, 14, x.shape[3]), method='nearest')
        
        # Convolve: Process the upsampled features
        # We maintain 'features' depth (64) to keep capacity high
        x = nn.Conv(
            self.features,
            kernel_size=(5, 5),  # 5x5 kernel helps smooth the nearest-neighbor edges
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02)
        )(x)
        x = nn.GroupNorm(num_groups=32, epsilon=1e-5)(x)
        x = jnp.tanh(x)

        # 3. UPSAMPLE BLOCK 2 (14x14 -> 28x28)
        x = jax.image.resize(x, shape=(x.shape[0], 28, 28, x.shape[3]), method='nearest')
        
        # Convolve
        x = nn.Conv(
            self.features // 2,  # Reduce depth to 32
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02)
        )(x)
        x = nn.GroupNorm(num_groups=16, epsilon=1e-5)(x) # Adjusted groups for smaller depth
        x = jnp.tanh(x)

        # 4. OUTPUT BLOCK (28x28 -> 28x28)
        # Collapse to 1 channel (Grayscale)
        x = nn.Conv(
            1,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='SAME',
            kernel_init=normal_init(0.02)
        )(x)
        x = jnp.tanh(x)
        
        return x

# assumes you already have: normal_init

class Discriminator(nn.Module):
    """Discriminator with attached Q-network (SpectralNorm, no BatchNorm)."""
    features: int = 64
    q_cat: int = 10
    train: bool = False
    q_cont: int = 2  # set to 0 if you only want categorical codes

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

        train = False
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

#class Generator(nn.Module):
#    """ Generator CNN for MNIST """
#
#    features: int = 64
#    training: bool = True
#
#    @nn.compact
#    def __call__(self, z):
#        #jax.debug.print('z shape : {} ', z.shape)
#        #if len(z.shape) == 3:
#        #    z = z.reshape((z.shape[0], z.shape[1], 1, 1, z.shape[2]))
#        #else:
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
#        #activations1 = x
#        x = nn.ConvTranspose(self.features*2, [4, 4], [1, 1], 'VALID', kernel_init=he_normal())(x)
#        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#        x = nn.relu(x)
#        #activations2 = x
#        x = nn.ConvTranspose(self.features, [4, 4], [1, 1], 'VALID', kernel_init=he_normal())(x)
#        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#        x = nn.relu(x)
#        #activations3 = x
#        x = nn.ConvTranspose(1, [4, 4], [2, 2], 'VALID', kernel_init=he_normal())(x)
#        x = jnp.tanh(x)
#        # use sigmoid
#        #x = nn.sigmoid(x)
#        return x#, activations1, activations2, activations3

class BinaryMNISTClassifier(nn.Module):
    """CNN for MNIST."""

    @nn.compact
    def __call__(self, x, training: bool = True): 
        x = x.reshape((x.shape[0], 28, 28, 1))

        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        x = nn.Dense(features=1)(x)
        return jnp.squeeze(x)

#class Discriminator(nn.Module):
#    features: int = 32
#    training: bool = True
#
#    q_cat: int = 10
#
#    @nn.compact
#    def __call__(self, x):
#        x = nn.Conv(self.features, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
#        #x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#        x = nn.leaky_relu(x, 0.1)
#        
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
#        #d = nn.sigmoid(d)
#        d = d.reshape((d.shape[0], -1))
#        #d = d.reshape((d.shape[0], -1))
#        #d = nn.sigmoid(d)
#        # Q outpiut
#        q = nn.Conv(self.features*2, [1, 1], [1, 1], 'VALID', kernel_init=normal_init(0.02), use_bias=False)(x)
#        q = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(q)
#        q = nn.leaky_relu(q, 0.1)
#       
#        q = q.reshape((q.shape[0], -1))
#
#        q_latent = nn.Dense(self.features*2)(q)
#        q_latent = nn.leaky_relu(q_latent, 0.1)
#
#        q_logits_cat = nn.Dense(self.q_cat)(q_latent)
#        #q = nn.Conv(self.q_cat, [1, 1], [2, 2], 'VALID', kernel_init=normal_init(0.02))(q)
#        #q = q.reshape((q.shape[0], -1))
#               
#        mu = nn.Dense(features=2)(q_latent)
#        log_var = nn.Dense(features=2)(q_latent)
#        var = jnp.squeeze(log_var)
#    
#        return d, q_logits_cat, mu.squeeze(), jnp.exp(var)

class QNetwork(nn.Module):
    features: int = 64
    training: bool = True

    q_cat: int = 10

    @nn.compact
    def __call__(self, x):
        #x = nn.Conv(self.features, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
        #x = nn.leaky_relu(x, 0.2)
        #x = nn.Conv(self.features*2, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
        #x = nn.leaky_relu(x, 0.2)

        #q = nn.Conv(self.features*2, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
        q = nn.Conv(self.features, [3, 3], [2, 2], 'VALID', kernel_init=he_normal())(x) 
        q = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(q)
        q = nn.leaky_relu(q, 0.1)
       
        disc_logits = nn.Conv(self.q_cat, [1, 1], [2, 2], 'VALID', kernel_init=he_normal())(q)
        disc_logits = disc_logits.reshape((disc_logits.shape[0], -1)) 

        #mu = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1))(q)
        #mu = jnp.squeeze(mu)

        #log_var = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1))(q)
        #var = jnp.squeeze(log_var)
        #var = jnp.exp(var)

        return disc_logits#, mu, var

class GenPolicy(PolicyNetwork):
    """A convolutional neural network for the MNIST classification task."""

    def __init__(
            self,
            logger: logging.Logger = None,
            noise_dim: int = 62,
            n_discrete_codes: int = 10,
            n_continuous_codes: int = 2):
        if logger is None:
            self._logger = create_logger('ConvNetPolicy')
        else:
            self._logger = logger

        self.noise_dim = int(noise_dim)
        self.n_classes = int(n_discrete_codes)
        self.n_cont = int(n_continuous_codes)
        self.total_latent_dim = self.noise_dim + self.n_classes + self.n_cont

        self.model_gen = Generator(training=False)
       
        self.model_disc = Discriminator(
            train=False,
            q_cat=self.n_classes,
            q_cont=self.n_cont,
        )

        self.model_q = Discriminator(
            q_cat=self.n_classes,
            q_cont=self.n_cont,
        )
        
        key = random.PRNGKey(122)

        key, key_gen, key_disc, key_bin = random.split(key, 4)

        variables_gen = self.model_gen.init(
            key_gen, jnp.ones([64, self.total_latent_dim], jnp.float32))
        variables_disc = self.model_disc.init(key_disc, jnp.ones([1,28,28,1], jnp.float32))
        
        variables_q = self.model_q.init(key_bin, jnp.ones([64,5,5,128], jnp.float32))
        
        self.init_params_gen = variables_gen['params']
        self.init_params_disc, self.init_batch_stats_disc = variables_disc['params'], variables_disc['batch_stats']

        self.adapter = ParameterAdapter(self.init_params_gen, chunk_size=512)
        self.init_params_hypernet = self.adapter.init_hypernet(random.PRNGKey(11))

        #jax.debug.print('batch stats gen shape : {}', self.init_batch_stats_gen.shape)
        self.latent_dim = self.total_latent_dim
  
        self.num_params, format_params_gen_fn = get_params_format_fn(self.init_params_gen)
        
        format_single_params_gen_fn = get_single_params_format_fn(self.init_params_gen)
        self._format_single_params_gen_fn = format_single_params_gen_fn
        
        format_single_params_hypernet_fn = get_single_params_format_fn(self.init_params_hypernet)
        self._format_single_params_hypernet_fn = format_single_params_hypernet_fn

        self._logger.info(
            'GenPolicy.num_params = {}'.format(self.num_params))
        self._format_params_gen_fn = jax.vmap(format_params_gen_fn)

        self.num_params_hypernet, format_params_hn_fn = get_params_format_fn(self.init_params_hypernet)
        self._logger.info(
            'GenPolicy.num_params_hypernet = {}'.format(self.num_params_hypernet))
        self._format_params_hn_fn = jax.vmap(format_params_hn_fn)

        leaves_params, _ = jax.tree_util.tree_flatten(self.init_params_gen)
        self.flat_params_gen = jnp.concatenate([p.flatten() for p in leaves_params])

        leaves_params_hypernet, _ = jax.tree_util.tree_flatten(self.init_params_hypernet)
        self.flat_params_hypernet = jnp.concatenate([p.flatten() for p in leaves_params_hypernet])

        self.num_params_disc, format_params_disc_fn = get_params_format_fn(self.init_params_disc)
        self._logger.info(
            'DiscPolicy.num_params = {}'.format(self.num_params_disc))
        self._format_params_disc_fn = jax.vmap(format_params_disc_fn)
        
        self.num_batch_stats_disc, format_batch_stats_disc_fn = get_params_format_disc_fn(self.init_batch_stats_disc)
        self._logger.info(
            'DiscPolicy.num_batch_stats = {}'.format(self.num_batch_stats_disc))
        self._format_batch_stats_disc_fn = jax.vmap(format_batch_stats_disc_fn)

        leaves_batch_stats_disc, _ = jax.tree_util.tree_flatten(self.init_batch_stats_disc)
        self.flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

        def forward_fn_gen(params_hn, params_d, vars_d_batch_stats, latent_input, noise):
          
            params_g = self.adapter.generate_params(params_hn)
       
            (fake_data) = self.model_gen.apply({'params': params_g}, latent_input) 
           
            fake_data_with_noise = fake_data + noise
          
            
            (preds, q_cat, q_cont_mu, q_cont_logsigma, q_flat) = self.model_disc.apply(
                {'params': params_d, 'batch_stats': vars_d_batch_stats},
                fake_data_with_noise,
                mutable=False,
            )

            # Code pixel diversity: variance across codes sharing the same z.
            # Latent design: groups of n_classes consecutive samples share same z,
            # differ only by code. Any pixel difference = code influence.
            # Brightness-normalized: subtract per-image mean so the generator
            # cannot cheat by encoding codes as bright vs dark. Forces
            # structural/textural diversity instead.
            n_classes = self.n_classes
            n_ctrl = (fake_data.shape[0] // n_classes) * n_classes  # 55 for batch=64
            grouped = fake_data[:n_ctrl].reshape(-1, n_classes, 28, 28, 1)
            grouped_centered = grouped - grouped.mean(axis=(2, 3), keepdims=True)
            code_pixel_div = jnp.mean(jnp.var(grouped_centered, axis=1))

            return preds, q_cat, code_pixel_div, q_flat, q_cont_mu, q_cont_logsigma

        self._forward_fn_gen = jax.vmap(forward_fn_gen)

    def set_format_params_disc_fn(self, format_params_disc_fn):
        self._format_params_disc_fn = format_params_disc_fn

    def set_format_batch_stats_disc_fn(self, format_batch_stats_disc_fn):
        self._format_batch_stats_disc_fn = format_batch_stats_disc_fn

    def get_actions(self,
                    t_states: TaskState,
                    params_hn: jnp.ndarray,
                    params_disc: jnp.ndarray,
                    #params_q: jnp.ndarray,
                    p_states: PolicyState):
        
        params_hn = self._format_params_hn_fn(params_hn)
        params_disc = self._format_params_disc_fn(params_disc)
        #params_q = self._format_params_q_fn(params_q)

        batch_stats_disc = self._format_batch_stats_disc_fn(t_states.batch_stats_disc)

        #batch_stats_q = self._format_batch_stats_q_fn(t_states.batch_stats_q) 

        #jax.debug.print('params gen : {} ', params_gen)

        preds, disc_logits, mean_var_fake, q_flat, q_cont_mu, q_cont_logsigma = self._forward_fn_gen(
            params_hn, params_disc, batch_stats_disc, t_states.obs, t_states.noise)
        
        return preds, disc_logits, mean_var_fake, q_flat, q_cont_mu, q_cont_logsigma, p_states
        #return self._forward_fn(params, t_states.obs), p_states

class DiscPolicy(PolicyNetwork):
    """A convolutional neural network for the MNIST classification task."""

    def __init__(self, gen_policy: GenPolicy, logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger('ConvNetPolicy')
        else:
            self._logger = logger

        self.model_disc = Discriminator()
        
        self.model_q = QNetwork()

        self.model_gen = gen_policy.model_gen

        key = random.PRNGKey(59)

        key, key_gen, key_disc, key_q = random.split(key, 4)

        image_shape = (64, 28, 28, 1)
        q_shape = [1,5,5,128]

        variables_disc = self.model_disc.init(key_disc, jnp.ones(image_shape, jnp.float32))
        variables_q = self.model_q.init(key_q, jnp.ones(q_shape, jnp.float32))

        self.init_params_disc, self.init_batch_stats_disc = variables_disc['params'], variables_disc['batch_stats']
        self.init_params_q, self.init_batch_stats_q = variables_q['params'], variables_q['batch_stats']

        self.num_params, format_params_disc_fn = get_params_format_fn(self.init_params_disc)
        self._logger.info(
            'DiscPolicy.num_params = {}'.format(self.num_params))
        self._format_params_disc_fn = jax.vmap(format_params_disc_fn)

        self.num_batch_stats, format_batch_stats_disc_fn = get_params_format_fn(self.init_batch_stats_disc)
        self._logger.info(
            'DiscPolicy.num_batch_stats = {}'.format(self.num_batch_stats))
        self._format_batch_stats_disc_fn = jax.vmap(format_batch_stats_disc_fn)

        self.num_params_q, format_params_q_fn = get_params_format_fn(self.init_params_q)
        self._logger.info(
            'QPolicy.num_params = {}'.format(self.num_params_q))
        self._format_params_q_fn = jax.vmap(format_params_q_fn)

        self.num_batch_stats_q, format_batch_stats_q_fn = get_params_format_fn(self.init_batch_stats_q)
        self._logger.info(
            'QPolicy.num_batch_stats = {}'.format(self.num_batch_stats_q))
        self._format_batch_stats_q_fn = jax.vmap(format_batch_stats_q_fn)

        self._format_params_gen_fn = gen_policy._format_params_gen_fn
        self._format_batch_stats_gen_fn = gen_policy._format_batch_stats_gen_fn

        gen_policy.set_model_disc(self.model_disc)
        gen_policy.set_format_params_disc_fn(self._format_params_disc_fn)
        gen_policy.set_format_batch_stats_disc_fn(self._format_batch_stats_disc_fn)
        gen_policy.set_model_q(self.model_q)
        gen_policy.set_format_params_q_fn(self._format_params_q_fn)
        gen_policy.set_format_batch_stats_q_fn(self._format_batch_stats_q_fn)
        
        leaves_params, _ = jax.tree_util.tree_flatten(self.init_params_disc)
        self.flat_params_disc = jnp.concatenate([p.flatten() for p in leaves_params])
        leaves_batch_stats_disc, _ = jax.tree_util.tree_flatten(self.init_batch_stats_disc)
        self.flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

        leaves_params_q, _ = jax.tree_util.tree_flatten(self.init_params_q)
        self.flat_params_q = jnp.concatenate([p.flatten() for p in leaves_params_q])
        leaves_batch_stats_q, _ = jax.tree_util.tree_flatten(self.init_batch_stats_q)
        self.flat_batch_stats_q = jnp.concatenate([p.flatten() for p in leaves_batch_stats_q])

        def forward_fn_disc(params_g, vars_g_batch_stats, params_d, vars_d_batch_stats, params_q, vars_q_batch_stats, real_data, latent_input, cat_codes):
            
            (fake_data), vars_g = self.model_gen.apply({'params': params_g, 'batch_stats': vars_g_batch_stats}, latent_input, mutable=['batch_stats'])

            (real_preds, _, _, _, _), vars_d = self.model_disc.apply(
                {'params': params_d, 'batch_stats': vars_d_batch_stats},
                real_data,
                mutable=['batch_stats'],
            )
            
            (fake_preds, _, _, _, q), vars_d = self.model_disc.apply(
                {'params': params_d, 'batch_stats': vars_d['batch_stats']},
                fake_data,
                mutable=['batch_stats'],
            )

            (disc_logits), vars_q = self.model_q.apply({'params': params_q, 'batch_stats': vars_q_batch_stats}, q, mutable=['batch_stats'])

            leaves_batch_stats_disc, _ = jax.tree_util.tree_flatten(vars_d['batch_stats'])
            flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])
 
            leaves_batch_stats_gen, _ = jax.tree_util.tree_flatten(vars_g['batch_stats'])
            flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

            leaves_batch_stats_q, _ = jax.tree_util.tree_flatten(vars_q['batch_stats'])
            flat_batch_stats_q = jnp.concatenate([p.flatten() for p in leaves_batch_stats_q])
            
            return real_preds, fake_preds, disc_logits, flat_batch_stats_disc, flat_batch_stats_gen, flat_batch_stats_q

        self._forward_fn_disc = jax.vmap(forward_fn_disc)

    def get_actions(self,
                    t_states: TaskState,
                    params_gen: jnp.ndarray,
                    params_disc: jnp.ndarray,
                    params_q: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        
        params_gen = self._format_params_gen_fn(params_gen)
        params_disc = self._format_params_disc_fn(params_disc)
        params_q = self._format_params_q_fn(params_q)

        batch_stats_gen = self._format_batch_stats_gen_fn(t_states.batch_stats_gen)
        batch_stats_disc = self._format_batch_stats_disc_fn(t_states.batch_stats_disc)
        batch_stats_q = self._format_batch_stats_q_fn(t_states.batch_stats_q) 
        
        #jax.debug.print('params gen : {} ', params_gen)

        real_preds, fake_preds, disc_logits, batch_stats_d, batch_stats_g, batch_stats_q = self._forward_fn_disc(params_gen, batch_stats_gen, params_disc, batch_stats_disc, params_q, batch_stats_q, t_states.obs, t_states.latent, t_states.cat_codes)
                
        return real_preds, fake_preds, disc_logits, batch_stats_d, batch_stats_g, batch_stats_q, p_states

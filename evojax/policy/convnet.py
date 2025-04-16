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

import orbax.checkpoint as orbax_cp
import optax
import jax
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
from evojax.util import get_params_format_fn, get_single_params_format_fn


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
      cat_codes = jrandom.randint(cat_rng, (n_samples,), 0, 10)
      cat_codes = nn.one_hot(cat_codes, 10) 
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

class Generator(nn.Module):
  features: int = 64
  training: bool = True

  @nn.compact
  def __call__(self, z):
    z = z.reshape((z.shape[0], 1, 1, z.shape[1]))
    x = nn.ConvTranspose(self.features*4, [3, 3], [2, 2], 'VALID', kernel_init=normal_init(0.02))(z)
    x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
    x = nn.relu(x)
    x = nn.ConvTranspose(self.features*2, [4, 4], [1, 1], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
    x = nn.relu(x)
    x = nn.ConvTranspose(self.features, [3, 3], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
    x = nn.relu(x)
    x = nn.ConvTranspose(1, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.sigmoid(x)
    return x


class Discriminator(nn.Module):
  features: int = 64
  training: bool = True

  q_cat: int = 10

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(self.features, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
    x = nn.leaky_relu(x, 0.2)
    x = nn.Conv(self.features*2, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
    x = nn.leaky_relu(x, 0.2)
    
    # Discriminator output
    d = nn.Conv(1, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    d = d.reshape((d.shape[0], -1))

    # Q outpiut
    q = nn.Conv(self.features*2, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    q = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(q)
    q = nn.leaky_relu(q, 0.2)

    disc_logits = nn.Conv(self.q_cat, [1, 1], [2, 2], 'VALID', kernel_init=normal_init(0.02))(q)
    disc_logits = disc_logits.reshape((disc_logits.shape[0], -1))
      
    mu = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1))(q)
    #print('mu shape : ', mu.shape)
    log_var = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1))(q)
    #print('log var shape : ', log_var.shape)
    var = jnp.squeeze(log_var)
    
    return d, disc_logits, mu.squeeze(), jnp.exp(var)

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

    def __init__(self, logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger('ConvNetPolicy')
        else:
            self._logger = logger

        self.model_gen = Generator()
       
        self.model_disc = Discriminator()

        key = random.PRNGKey(122)

        key, key_gen, key_disc, key_bin = random.split(key, 4)

        variables_gen = self.model_gen.init(key_gen, jnp.ones([64,74], jnp.float32))
        variables_disc = self.model_disc.init(key_disc, jnp.ones([64,28,28,1], jnp.float32))
        
        self.init_params_gen, self.init_batch_stats_gen = variables_gen['params'], variables_gen['batch_stats']
        self.init_params_disc, self.init_batch_stats_disc = variables_disc['params'], variables_disc['batch_stats']

        #jax.debug.print('batch stats gen shape : {}', self.init_batch_stats_gen.shape)
        self.latent_dim = 64
  
        self.num_params, format_params_gen_fn = get_params_format_fn(self.init_params_gen)
        
        format_single_params_gen_fn = get_single_params_format_fn(self.init_params_gen)
        self._format_single_params_gen_fn = format_single_params_gen_fn
        
        format_single_params_disc_fn = get_single_params_format_fn(self.init_params_disc)
        self._format_single_params_disc_fn = format_single_params_disc_fn

        self._logger.info(
            'GenPolicy.num_params = {}'.format(self.num_params))
        self._format_params_gen_fn = jax.vmap(format_params_gen_fn)

        self.num_batch_stats, format_batch_stats_gen_fn = get_params_format_fn(self.init_batch_stats_gen)
        self._logger.info(
            'GenPolicy.num_batch_stats = {}'.format(self.num_batch_stats))
        self._format_batch_stats_gen_fn = jax.vmap(format_batch_stats_gen_fn)

        leaves_params, _ = jax.tree_util.tree_flatten(self.init_params_gen)
        
        self.flat_params_gen = jnp.concatenate([p.flatten() for p in leaves_params])

        leaves_batch_stats_gen, _ = jax.tree_util.tree_flatten(self.init_batch_stats_gen)

        self.flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

        self.num_params_disc, format_params_disc_fn = get_params_format_fn(self.init_params_disc)
        self._logger.info(
            'DiscPolicy.num_params = {}'.format(self.num_params_disc))
        self._format_params_disc_fn = jax.vmap(format_params_disc_fn)
        self.num_batch_stats_disc, format_batch_stats_disc_fn = get_params_format_fn(self.init_batch_stats_disc)
        self._logger.info(
            'DiscPolicy.num_batch_stats = {}'.format(self.num_batch_stats_disc))
        self._format_batch_stats_disc_fn = jax.vmap(format_batch_stats_disc_fn)

        leaves_batch_stats_disc, _ = jax.tree_util.tree_flatten(self.init_batch_stats_disc)
        self.flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

        def forward_fn_gen(params_g, vars_g_batch_stats, params_d, vars_d_batch_stats, latent_input):
          
            #(fake_data) = self.model_gen.apply({'params': params_g, 'batch_stats': vars_g_batch_stats}, latent_input)
       
            #(preds, q), vars_d = self.model_disc.apply({'params': params_d, 'batch_stats': vars_d_batch_stats}, fake_data, mutable=['batch_stats'])

            (fake_data), vars_g = self.model_gen.apply({'params': params_g, 'batch_stats': vars_g_batch_stats}, latent_input, mutable=['batch_stats'])
            (preds, q, mu, var), vars_d = self.model_disc.apply({'params': params_d, 'batch_stats': vars_d_batch_stats}, fake_data, mutable=['batch_stats'])
            #(disc_logits), vars_q = self.model_q.apply({'params': params_q, 'batch_stats': vars_q_batch_stats}, q, mutable=['batch_stats'])

            leaves_batch_stats_gen, _ = jax.tree_util.tree_flatten(vars_g['batch_stats'])
            flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

            leaves_batch_stats_disc, _ = jax.tree_util.tree_flatten(vars_d['batch_stats'])
            flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

            #leaves_batch_stats_q, _ = jax.tree_util.tree_flatten(vars_q['batch_stats'])
            #flat_batch_stats_q = jnp.concatenate([p.flatten() for p in leaves_batch_stats_q])
            
            return fake_data, flat_batch_stats_gen, preds, q, flat_batch_stats_disc, mu, var

        self._forward_fn_gen = jax.vmap(forward_fn_gen)

    def set_format_params_disc_fn(self, format_params_disc_fn):
        self._format_params_disc_fn = format_params_disc_fn

    def set_format_batch_stats_disc_fn(self, format_batch_stats_disc_fn):
        self._format_batch_stats_disc_fn = format_batch_stats_disc_fn

    def get_actions(self,
                    t_states: TaskState,
                    params_gen: jnp.ndarray,
                    params_disc: jnp.ndarray,
                    #params_q: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        
        params_gen = self._format_params_gen_fn(params_gen)
        params_disc = self._format_params_disc_fn(params_disc)
        #params_q = self._format_params_q_fn(params_q)

        batch_stats_gen = self._format_batch_stats_gen_fn(t_states.batch_stats_gen)
        batch_stats_disc = self._format_batch_stats_disc_fn(t_states.batch_stats_disc)
        #batch_stats_q = self._format_batch_stats_q_fn(t_states.batch_stats_q) 

        #jax.debug.print('params gen : {} ', params_gen)

        fake_data, batch_stats_g, preds, disc_logits, batch_stats_d, mu, var = self._forward_fn_gen(params_gen, batch_stats_gen, params_disc, batch_stats_disc, t_states.obs)
        
        return fake_data, preds, disc_logits, batch_stats_g, batch_stats_d, mu, var, p_states
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

        format_single_params_disc_fn = get_single_params_format_fn(self.init_params_disc)
        self._format_single_params_disc_fn = format_single_params_disc_fn

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
        gen_policy.set_format_params_disc_fn(self._format_single_params_disc_fn)
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

            (real_preds,_), vars_d = self.model_disc.apply({'params': params_d, 'batch_stats': vars_d_batch_stats}, real_data, mutable=['batch_stats'])
            
            (fake_preds,q), vars_d = self.model_disc.apply({'params': params_d, 'batch_stats': vars_d['batch_stats']}, fake_data, mutable=['batch_stats'])

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

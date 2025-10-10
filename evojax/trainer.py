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
import time
from typing import Optional, Callable

import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.serialization as serialization

from functools import partial
from evojax.task.base import VectorizedTask
from evojax.policy import PolicyNetwork
from evojax.algo import NEAlgorithm
from evojax.algo import QualityDiversityMethod
from evojax.sim_mgr import SimManager
from evojax.obs_norm import ObsNormalizer
from evojax.util import create_logger
from evojax.util import load_model_gen, load_model_disc
from evojax.util import save_model
from evojax.util import save_lattices
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import he_normal
from flax import linen as nn
from torchvision import datasets

# import Tuple
from typing import Tuple

def save_optimizer_state(opt_state, file_path: str):
    """
    Saves the optimizer state to disk using Flax serialization.
    """
    # Convert the pytree to raw bytes
    bytes_output = serialization.to_bytes(opt_state)
    # Write to file
    with open(file_path, 'wb') as f:
        f.write(bytes_output)

def load_optimizer_state(file_path: str, optimizer_state_structure):
    """
    Loads the optimizer state from disk. You must provide a "template"
    structure (optimizer_state_structure) that has the same structure 
    (PyTree) as what was originally saved.
    
    Typically, you can pass in an uninitialized or dummy version 
    of the optimizer state.
    """
    with open(file_path, 'rb') as f:
        bytes_input = f.read()
    # 'optimizer_state_structure' is a placeholder with the same structure.
    loaded_optimizer_state = serialization.from_bytes(
        optimizer_state_structure,
        bytes_input
    )
    return loaded_optimizer_state

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

class PixelNorm(nn.Module):
    eps: float = 1e-8
    @nn.compact
    def __call__(self, x):
        return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)


class GeneratorOld(nn.Module):
  features: int = 64
  training: bool = True
  
  @nn.compact
  def __call__(self, z):
    z = z.reshape((z.shape[0], 1, 1, z.shape[1]))
    x = nn.ConvTranspose(self.features*4, [3, 3], [2, 2], 'VALID', kernel_init=normal_init(0.02))(z)
    x = PixelNorm()(x)
    x = nn.relu(x)
    x = nn.ConvTranspose(self.features*2, [4, 4], [1, 1], 'VALID', kernel_init=normal_init(0.02))(x)
    x = PixelNorm()(x)
    x = nn.relu(x)
    x = nn.ConvTranspose(self.features, [3, 3], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    x = PixelNorm()(x)
    x = nn.relu(x)
    x = nn.ConvTranspose(1, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.sigmoid(x)
    return x


class Generator(nn.Module):
  features: int = 64
  training: bool = True
  
  @nn.compact
  def __call__(self, z):
    z = z.reshape((z.shape[0], 1, 1, z.shape[1]))
    x = nn.ConvTranspose(self.features*4, [3, 3], [2, 2], 'VALID', kernel_init=normal_init(0.02))(z)
    x = nn.GroupNorm(num_groups=16)(x)
    x = nn.silu(x)
    x = nn.ConvTranspose(self.features*2, [4, 4], [1, 1], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.GroupNorm(num_groups=16)(x)
    x = nn.silu(x)
    x = nn.ConvTranspose(self.features, [3, 3], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.GroupNorm(num_groups=16)(x)
    x = nn.silu(x)
    x = nn.ConvTranspose(1, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.sigmoid(x)
    return x

#class Generator(nn.Module):
#  features: int = 64
#  training: bool = True
#
#  @nn.compact
#  def __call__(self, z):
#    z = z.reshape((z.shape[0], 1, 1, z.shape[1]))
#    x = nn.ConvTranspose(self.features*4, [3, 3], [2, 2], 'VALID', kernel_init=normal_init(0.02),use_bias=False)(z)
#    x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#    x = nn.relu(x)
#    x = nn.ConvTranspose(self.features*2, [4, 4], [1, 1], 'VALID', kernel_init=normal_init(0.02),use_bias=False)(x)
#    x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#    x = nn.relu(x)
#    x = nn.ConvTranspose(self.features, [3, 3], [2, 2], 'VALID', kernel_init=normal_init(0.02),use_bias=False)(x)
#    x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
#    x = nn.relu(x)
#    x = nn.ConvTranspose(1, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02),use_bias=False)(x)
#    x = nn.sigmoid(x)
#    return x

class SharedEncoder(nn.Module):
  features: int = 64
  training: bool = True

  q_cat: int = 10

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(self.features, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.GroupNorm(num_groups=16)(x)
    x = nn.leaky_relu(x, 0.2)
    x = nn.Conv(self.features*2, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    x = nn.GroupNorm(num_groups=16)(x)
    x = nn.leaky_relu(x, 0.2)
    
    return x

class Discriminator(nn.Module):
  features: int = 64
  training: bool = True

  q_cat: int = 10

  @nn.compact
  def __call__(self, features):
    # Discriminator output
    d = nn.Conv(1, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(features)
    d = d.reshape((d.shape[0], -1))

    # Q outpiut
    #q = nn.Conv(self.features*2, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
    #q = nn.GroupNorm(num_groups=16)(q)
    #q = nn.leaky_relu(q, 0.2)

    # ADD BOTTLENECK: Force through narrow layer
    #q = nn.Conv(32, [1, 1], [1, 1], 'SAME')(q)  # Squeeze to 8 channels
    #q = nn.leaky_relu(q, 0.2)
    #q = nn.Conv(self.features, [1, 1], [1, 1], 'SAME')(q)  # Expand back

    #disc_logits = nn.Conv(self.q_cat, [1, 1], [1, 1], 'VALID', kernel_init=normal_init(0.02))(q)
    #disc_logits = disc_logits.reshape((disc_logits.shape[0], -1))
      
    #mu = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1))(q)
    #print('mu shape : ', mu.shape)
    #log_var = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1))(q)
    #print('log var shape : ', log_var.shape)
    #var = jnp.squeeze(log_var)
    
    return d#, disc_logits, mu.squeeze(), jnp.exp(var)


class Q(nn.Module):
  features: int = 64
  training: bool = True

  q_cat: int = 10

  @nn.compact
  def __call__(self, features):
    # Q outpiut
    q = nn.Conv(self.features*2, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(features)
    q = nn.GroupNorm(num_groups=16)(q)
    q = nn.leaky_relu(q, 0.2)

    disc_logits = nn.Conv(self.q_cat, [1, 1], [1, 1], 'VALID', kernel_init=normal_init(0.02))(q)
    disc_logits = disc_logits.reshape((disc_logits.shape[0], -1))
      
    mu = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1))(q)
    #print('mu shape : ', mu.shape)
    log_var = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1))(q)
    #print('log var shape : ', log_var.shape)
    var = jnp.squeeze(log_var)
    
    return disc_logits, mu.squeeze(), jnp.exp(var)

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

#@partial(jax.jit, static_argnames=['solver_enc', 'solver_disc'])
def train_step_disc(state, data, fake_imgs, solver_enc, solver_disc, key, iteration):
       
        params_e, params_d, opt_enc, opt_disc = state
                
        def loss_discriminator(params_e, params_d):
                
                  real_features = SharedEncoder().apply(
                      {'params': params_e}, data
                  )

                  real_preds = Discriminator().apply(
                      {'params': params_d}, real_features
                  )
                  
                  fake_features = SharedEncoder().apply( 
                      {'params': params_e}, fake_imgs
                  )

                  fake_preds = Discriminator().apply(
                      {'params': params_d}, fake_features
                  )

                  # use 0.9 as the label for real images instead of 1.0
                  real_loss = optax.sigmoid_binary_cross_entropy(real_preds, jnp.ones_like(real_preds))
                  # use 0.1 as the label for fake images instead of 0.0
                  fake_loss = optax.sigmoid_binary_cross_entropy(fake_preds, jnp.zeros_like(fake_preds))

                  real_loss = jnp.mean(real_loss)
                  fake_loss = jnp.mean(fake_loss)

                  loss = (real_loss + fake_loss) / 2.0 #+ loss_mi*0.4 + loss_con*0.06
                
                  return loss

           
        grad_fn_disc = jax.value_and_grad(loss_discriminator, argnums=(0,1))
        loss, (grads_e, grads_d) = grad_fn_disc(params_e, params_d)

        # apply enc gradients
        updates_enc, new_opt_state_enc = solver_enc.update(grads_e, opt_enc, params_d)
        params_e = optax.apply_updates(params_e, updates_enc)
       
        # apply disc gradients
        updates_d, new_opt_state_disc = solver_disc.update(grads_d, opt_disc, params_d)
        params_d = optax.apply_updates(params_d, updates_d)

        return (params_e, params_d, new_opt_state_enc, new_opt_state_disc), loss



@partial(jax.jit, static_argnames=['solver'])
def train_step_q(state, fake_imgs, fake_cat_input, con_codes, solver, key, iteration):
       
        params_e, params_q, opt_q = state

        def loss_mutual_information(code_cat, q_cat):
                  return -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))
           
        def continuous_loss(x, mu, var):
            # Simple MSE for mean prediction
            mse = jnp.mean((x - mu) ** 2)
            
            # Regularize variance to stay near 1.0
            var_reg = jnp.mean((var - 1.0) ** 2) * 0.1
            
            return mse + var_reg

        def loss_q(params_q):
                
                  fake_features = SharedEncoder(training=False).apply( 
                      {'params': params_e}, fake_imgs
                  )
                  
                  q, mu, var = Q().apply(
                      {'params': params_q}, fake_features
                  )

                  
                  # Calculate Mutual Information loss
                  q_cat = nn.log_softmax(q, axis=-1)
                  loss_mi = loss_mutual_information(fake_cat_input, q_cat)
               
                  #loss_con = normal_nll_loss(con_codes, mu, var)
                  loss_con = continuous_loss(con_codes, mu, var)
                                
                  #jax.debug.print('con loss: {} ', loss_con)
                  loss = loss_mi + loss_con*0.1
                
                  return loss

        grad_fn_q = jax.value_and_grad(loss_q)
        loss_q_val, grads_q = grad_fn_q(params_q)

        updates_q, new_opt_state_q = solver.update(grads_q, opt_q, params_q)
        params_q = optax.apply_updates(params_q, updates_q)

        return (params_q, new_opt_state_q), loss_q_val

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

def sample_latent(key, shape_noise, shape_cat):
  noise_key, cat_key, con_key = jax.random.split(key, 3)
  
  # Sample irreducible noise
  noise = jax.random.normal(noise_key, shape_noise)

  # Sample categorical latent code
  code_cat = jax.random.randint(cat_key, shape_cat, 0, 10)
  code_cat = jax.nn.one_hot(code_cat, 10)

  #c1 = jnp.tile(jnp.arange(10), 6)
  #c2 = jax.random.randint(cat_key, (4,), 0, 10)  # Randomly sample some indices
  #c = c[:64]
  #c = jnp.concatenate([c1, c2])  # Combine the two arrays
  #c = jax.random.permutation(cat_key, c)  # Shuffle to randomize
  #code_cat = jax.nn.one_hot(c, 10)  # One-hot encoding for categorical code
  #code_cat = jax.nn.one_hot(c, 10)

  con = jax.random.uniform(con_key, (64, 2), minval=-1, maxval=1)
  
  latent = jnp.concatenate([noise, code_cat, con], axis=-1)

  return latent, code_cat, con

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
                 solver_gen: NEAlgorithm,
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

        #self.batch_stats_gen = policy_gen.flat_batch_stats_gen
        #self.batch_stats_disc = policy_gen.flat_batch_stats_disc

        self.batch_size = batch_size
        self.mini_batch_size = 64
        self.num_mini_batches = 1
       
        #self.avg_mi_loss = -2.30
        self.fake_imgs = None
        self.cat_codes = None

        self.latent_dim = 64
        self.n_classes = 10
        self.n_con = 2

        self.decay_factor = 0.9

        self.noise_dim = self.latent_dim - self.n_con
        self.policy_gen = policy_gen

        self._key = jax.random.PRNGKey(44)

        self._cat_code_noise_key = jax.random.PRNGKey(56)
        
        self._log_interval = log_interval
        self._test_interval = test_interval
        self._max_iter = max_iter
        self.model_dir = model_dir
        self._log_dir = log_dir

        self._log_scores_fn = log_scores_fn or (lambda x, y, z: None)

        self._obs_normalizer = ObsNormalizer(
            obs_shape=train_task_gen.obs_shape,
            dummy=not normalize_obs,
        )

        self.solver_gen = solver_gen
        #self.solver_disc = solver_disc
        #self.solver_q = solver_q

        self.sim_mgr_gen = SimManager(
            n_repeats=n_repeats,
            test_n_repeats=test_n_repeats,
            pop_size=solver_gen.pop_size,
            n_evaluations=n_evaluations,
            policy_net=policy_gen,
            train_vec_task=train_task_gen,
            valid_vec_task=test_task_gen,
            seed=seed,
            obs_normalizer=self._obs_normalizer,
            use_for_loop=use_for_loop,
            logger=self._logger,
        )

        self._key, subkey_enc, subkey_disc, subkey_q = jax.random.split(self._key, 4)
        
        dataset = datasets.MNIST('./data', train=True, download=True)
        self.data = np.expand_dims(dataset.data.numpy() / 255.0, axis=-1)
        self.labels = dataset.targets.numpy()

        # initialize the shared encoder
        variables_enc = SharedEncoder().init(subkey_enc, jnp.ones((self.batch_size, 28, 28, 1), dtype=jnp.float32))
        self.params_enc = variables_enc['params']

        self.solver_enc = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.999)

        # initialize the discriminator
        variables_disc = Discriminator().init(subkey_disc, jnp.ones((self.batch_size, 5, 5, 128), dtype=jnp.float32))
        self.params_disc = variables_disc['params']

        self.solver_disc = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.999)

        # initialize the Q network
        variables_q = Q().init(subkey_q, jnp.ones((self.batch_size, 5, 5, 128), dtype=jnp.float32))
        self.params_q = variables_q['params']

        self.solver_q = optax.adam(learning_rate=0.00004, b1=0.5, b2=0.999)
        
    def run(self, demo_mode: bool = False) -> float:

        def gather_pop_stats(belief_space):

            mean_mi = belief_space[5][9]
            mean_g = belief_space[5][9]
            mean_cond = belief_space[5][9]

            var_mi = belief_space[5][9]
            var_g = belief_space[5][9]
            var_cond = belief_space[5][9]
        
            return jnp.array([mean_mi, mean_g, mean_cond, var_mi, var_g, var_cond])

        """Start the training / test process."""

        solver_enc = self.solver_enc
        solver_disc = self.solver_disc
        solver_q = self.solver_q

        #if self.model_dir is not None:
        #    params_gen, batch_stats_gen, params_disc, self.batch_stats_disc, obs_params = load_model_gen(model_dir=self.model_dir)
        #    
        #    #jax.debug.print('params gen shape: {}', params_gen.shape) 
        #    #jax.debug.print('batch stats gen shape: {}', batch_stats_gen.shape)
        #    self.params_disc = self.policy_gen._format_single_params_disc_fn(params_disc)
        #    
        #    batch_stats_disc = jnp.expand_dims(self.batch_stats_disc, axis=0)
        #    
        #    self.batch_stats_disc = self.policy_gen._format_batch_stats_disc_fn(batch_stats_disc)
        #  
        #    init_opt_state = solver_disc.init(params_disc)
        #    
        #    opt_disc = load_optimizer_state(os.path.join(self._log_dir, 'disc_opt_state.msgpack'), init_opt_state)
        #
        #    self.sim_mgr_gen.obs_params = obs_params
        #    self._logger.info(
        #        'Loaded model parameters from {}.'.format(self.model_dir))
        #else:
        params_gen, params_disc = None, None

        opt_enc = solver_enc.init(self.params_enc)
        opt_disc = solver_disc.init(self.params_disc)
        opt_q = solver_q.init(self.params_q)

        if demo_mode:
            if params_gen is None:
                raise ValueError('No policy parameters to evaluate.')
            self._logger.info('Start to test the parameters.')
            scores = np.array(
                self.sim_mgr_gen.eval_params(params=params_gen, test=True)[0])
            self._logger.info(
                '[TEST] #tests={0}, max={1:.4f}, avg={2:.4f}, min={3:.4f}, '
                'std={4:.4f}'.format(scores.size, scores.max(), scores.mean(),
                                     scores.min(), scores.std()))
            return scores.mean()
        else:

            self._logger.info(
                'Start to train for {} iterations.'.format(self._max_iter))

            #if params_gen is not None and self.params_disc is not None:# and params_q is not None:
            #

            #    jax.debug.print('Continuing training from the last checkpoint.')
            #    # Continue training from the breakpoint.
            #    self.solver_gen.best_params = params_gen

            best_score_gen, best_score_disc, best_score_q = -float('Inf'), -float('Inf'), -float('Inf')

            params_enc = self.params_enc
            params_disc = self.params_disc
            params_q = self.params_q 

            num_mini_batches = self.num_mini_batches
           
            self._key, noise_key, con_key = jax.random.split(self._key, 3)

            fixed_batch_latent = jax.random.normal(noise_key, (self.batch_size, self.latent_dim-self.n_con))
            fixed_c = jnp.tile(jnp.arange(10), 7)
            fixed_c = fixed_c[:self.batch_size]
            fixed_con = jax.random.uniform(con_key, (self.batch_size, 2), minval=-1, maxval=1) 
                    
            fixed_latent = jnp.concatenate([fixed_batch_latent, jax.nn.one_hot(fixed_c, 10), fixed_con], axis=-1)

            self._cat_code_noise_key, cat_code_subkey = jax.random.split(self._cat_code_noise_key)
 
            for i in range(self._max_iter):
                
                shape_noise = (self.mini_batch_size, self.latent_dim-self.n_con)
                shape_cat = (self.mini_batch_size,)

                #if i < 1:
                    #if len(self.batch_stats_gen.shape) == 1:
                    #    self.batch_stats_gen = jnp.expand_dims(self.batch_stats_gen, axis=0)
                
                    #batch_stats_gen = self.policy_gen._format_batch_stats_gen_fn(self.batch_stats_gen)

                for mini_batch in range(num_mini_batches):
                    # Sample batch of data.

                    self._key, subkey_latent, subkey_mnist = jax.random.split(self._key, 3)
                    

                    data, labels = sample_batch(subkey_mnist, self.data, self.labels, self.mini_batch_size)

                    #data = np.expand_dims(data / 255.0, axis=-1)

                    latent, cat_codes, con_codes = sample_latent(subkey_latent, shape_noise, shape_cat)
                  
                    #if i > 600:
                    params_gen = self.solver_gen.best_params
                    params_gen_formatted = self.policy_gen._format_single_params_gen_fn(params_gen)
                    #else:
                    
                    #fake_images = Generator(training=False).apply({'params': params_gen_formatted, 'batch_stats': batch_stats_gen},latent, mutable=False)
                    fake_images = Generator(training=False).apply({'params': params_gen_formatted}, latent)
                    #batch_stats_gen = vars_g['batch_stats']
                    # reshape fake_images to (64, 28, 28, 1) from [1,1,1,64, 28, 28, 1]
                    fake_images = fake_images.reshape((self.mini_batch_size, 28, 28, 1))
                    
                    state_d = (params_enc, params_disc, opt_enc, opt_disc)

                    state_d, d_loss = train_step_disc(
                        state_d,
                        data,
                        fake_images,
                        solver_enc,
                        solver_disc,
                        cat_code_subkey,
                        iteration=i
                    )

                    
                    #jax.debug.print('loss: {} ', loss)
                    
                    params_enc, params_disc, opt_enc, opt_disc = state_d

                    state_q = (params_enc, params_q, opt_q)

                    state_q, q_loss = train_step_q(
                        state_q,
                        fake_images,
                        cat_codes,
                        con_codes,
                        solver_q,
                        cat_code_subkey,
                        iteration=i
                    )

                    params_q, opt_q = state_q

                
                leaves_params, _ = jax.tree_flatten(params_enc)
                flat_params_enc = jnp.concatenate([p.flatten() for p in leaves_params])

                leaves_params, _ = jax.tree_flatten(params_disc) 
                flat_params_disc = jnp.concatenate([p.flatten() for p in leaves_params])

                leaves_params, _ = jax.tree_flatten(params_q)
                flat_params_q = jnp.concatenate([p.flatten() for p in leaves_params])
                                
                params_gen, belief_space = self.solver_gen.ask()
                
                scores_gen_adv, scores_gen_mi, scores_gen_con, disc_logits, bds_gen, _ = self.sim_mgr_gen.eval_params(
                params_gen=params_gen, params_enc=flat_params_enc, params_disc=flat_params_disc, params_q=flat_params_q, generator=True, test=False
                )

                if isinstance(self.solver_gen, QualityDiversityMethod):
                    self.solver_gen.observe_bd(bds_gen)
                
                self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_con, disc_logits=disc_logits, adv=False)

                params_gen, belief_space = self.solver_gen.ask()

                #scores_gen_adv, scores_gen_mi, scores_gen_con, disc_logits, bds_gen, BN_stats_gen, _, _ = self.sim_mgr_gen.eval_params(
                #params_gen=params_gen, params_disc=flat_params_disc, batch_stats_gen=flat_batch_stats_gen, batch_stats_disc=flat_batch_stats_disc,  generator=True, test=False
                #)

                #if isinstance(self.solver_gen, QualityDiversityMethod):
                #    self.solver_gen.observe_bd(bds_gen)
                #
                #self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_con, disc_logits=disc_logits, adv=True)

                #best_params_gen = self.solver_gen.best_params
                #best_params_gen_formatted = self.policy_gen._format_single_params_gen_fn(best_params_gen)

                #self._key, subkey = jax.random.split(self._key)
                #shape_noise = (self.batch_size, self.latent_dim-self.n_con)
                #shape_cat = (self.batch_size,)
                #latent, cat_codes, con_codes = sample_latent(subkey, shape_noise, shape_cat)

                #(fake_images), vars_g = Generator().apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},latent, mutable=['batch_stats'])
                #batch_stats_gen = vars_g['batch_stats'] 

                #(_, _, _, _), vars_d = Discriminator().apply({'params': params_disc, 'batch_stats': self.batch_stats_disc}, fake_images, mutable=['batch_stats'])
                #self.batch_stats_disc = vars_d['batch_stats']

                #leaves_batch_stats_gen, _ = jax.tree_flatten(batch_stats_gen)
                #flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

                #leaves_batch_stats_disc, _ = jax.tree_flatten(self.batch_stats_disc)
                #flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

                params_gen, belief_space = self.solver_gen.ask()

                scores_gen_adv, scores_gen_mi, scores_gen_con, disc_logits, bds_gen, _ = self.sim_mgr_gen.eval_params(
                params_gen=params_gen, params_enc=flat_params_enc, params_disc=flat_params_disc, params_q=flat_params_q,  generator=True, test=False
                )

                if isinstance(self.solver_gen, QualityDiversityMethod):
                    self.solver_gen.observe_bd(bds_gen)
                
                self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_con, disc_logits=disc_logits, adv=True)

                #best_params_gen = self.solver_gen.best_params
                #best_params_gen_formatted = self.policy_gen._format_single_params_gen_fn(best_params_gen)

                #self._key, subkey = jax.random.split(self._key)
                #shape_noise = (self.mini_batch_size, self.latent_dim-self.n_con)
                #shape_cat = (self.batch_size,)
                #latent, cat_codes, con_codes = sample_latent(subkey, shape_noise, shape_cat)

                #(fake_images), vars_g = Generator().apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},latent, mutable=['batch_stats'])
                #batch_stats_gen = vars_g['batch_stats'] 
                
                #(_, _, _, _), vars_d = Discriminator().apply({'params': params_disc, 'batch_stats': self.batch_stats_disc}, fake_images, mutable=['batch_stats'])
                #self.batch_stats_disc = vars_d['batch_stats']
                #self.batch_stats_disc = batch_stats_disc
                #self.batch_stats_gen = batch_stats_gen
                
                #self.avg_mi_loss = jnp.mean(scores_gen_mi)

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

                    scores_con = np.array(scores_gen_con)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores_con.size, scores_con.max(), scores_con.mean(),
                            scores_con.min(), scores_con.std()))

                    self._logger.info(
                        'Iter={0}, d_loss={1:.4f}'.format(
                            i, d_loss))
                    
                    #with open('/home/gh0st/Downloads/pgpe_main.csv', 'a') as file:
                        #file.write(f'Iter: {i}, Max: {scores.max()}, Mean: {scores.mean()}, Std: {scores.std()}, Min: {scores.min()}\n')
                    #self._log_scores_fn(i, scores, "train")

                if i > 0 and i % self._test_interval == 0:
                    best_params_gen = self.solver_gen.best_params
                    best_params_gen_formatted = self.policy_gen._format_single_params_gen_fn(best_params_gen)

                    self._key, noise_key, con_key = jax.random.split(self._key, 3)

                    fixed_batch_latent = jax.random.normal(noise_key, (self.batch_size, self.latent_dim-self.n_con))
                    fixed_c = jnp.tile(jnp.arange(10), 7)
                    fixed_c = fixed_c[:self.batch_size]
                    fixed_con = jax.random.uniform(con_key, (self.batch_size, 2), minval=-1, maxval=1) 
                            
                    fixed_latent = jnp.concatenate([fixed_batch_latent, jax.nn.one_hot(fixed_c, 10), fixed_con], axis=-1)


                    #batch_stats_gen_leaves, _ = jax.tree_flatten(batch_stats_gen)
                    #flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in batch_stats_gen_leaves])
                    
                    #(fake_imgs) = Generator(training=False).apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},fixed_latent)
                    fake_imgs = Generator(training=False).apply({'params': best_params_gen_formatted}, fixed_latent)
                    filename = f"iteration-{i}.npy"
                    np.save(filename, fake_imgs[:, :, :, :])

            # Test and save the final model.
            best_params_gen = self.solver_gen.best_params
            #best_params_disc = self.solver_disc.best_params
            #test_scores, _ = self.sim_mgr.eval_params(
            #    params=best_params, test=True)
            #self._logger.info(
            #    '[TEST] Iter={0}, #tests={1}, max={2:.4f}, avg={3:.4f}, '
            #    'min={4:.4f}, std={5:.4f}'.format(
            #        self._max_iter, test_scores.size, test_scores.max(),
            #        test_scores.mean(), test_scores.min(), test_scores.std()))
            #mean_test_score = test_scores.mean()
            #leaves_batch_stats_gen, _ = jax.tree_flatten(batch_stats_gen)
            #flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

            save_model(
                model_dir=self._log_dir,
                model_name='final_model_gen',
                params=best_params_gen,
                params_disc=flat_params_disc,
                obs_params=self.sim_mgr_gen.obs_params,
                #batch_stats=flat_batch_stats_gen,
                batch_stats_disc=flat_batch_stats_disc,
                #best=mean_test_score > best_score,
            )
            save_optimizer_state(
                opt_state=opt_disc,
                file_path=os.path.join(self._log_dir, 'disc_opt_state.msgpack'),
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
            best_score = scores_gen_adv.mean()
            self._logger.info(
                'Training done, best_score={0:.4f}'.format(best_score))

            return best_score

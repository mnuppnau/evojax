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
    """InfoGAN generator for MNIST, based on your simple ConvTranspose stack."""
    features: int = 64
    training: bool = True

    @nn.compact
    def __call__(self, z):
        """
        Args:
            z:      (B, z_dim)
            c_cat:  (B, n_cat)    one-hot
            c_cont: (B, n_cont)   e.g. 2 dims in [-1, 1]

        Returns:
            x: (B, 28, 28, 1) in [-1, 1]
        """

        z_full = z.reshape((z.shape[0], 1, 1, z.shape[1]))

        x = nn.ConvTranspose(
            self.features * 4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='VALID',
            kernel_init=normal_init(0.02),
        )(z_full)
        x = nn.BatchNorm(
            use_running_average=not self.training,
            axis=-1,
            momentum=0.1,
            scale_init=normal_init(0.02),
        )(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(
            self.features * 2,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding='VALID',
            kernel_init=normal_init(0.02),
        )(x)
        x = nn.BatchNorm(
            use_running_average=not self.training,
            axis=-1,
            momentum=0.1,
            scale_init=normal_init(0.02),
        )(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(
            self.features,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='VALID',
            kernel_init=normal_init(0.02),
        )(x)
        x = nn.BatchNorm(
            use_running_average=not self.training,
            axis=-1,
            momentum=0.1,
            scale_init=normal_init(0.02),
        )(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(
            1,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='VALID',
            kernel_init=normal_init(0.02),
        )(x)
        x = jnp.tanh(x)
        return x

class Discriminator(nn.Module):
    """Discriminator with attached Q-network, built on your architecture."""
    features: int = 64
    training: bool = True
    q_cat: int = 10
    q_cont: int = 2   # set to 0 if you only want categorical codes

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, 28, 28, 1) in [-1, 1]

        Returns:
            d_logits:        (B, 1)         real/fake logits
            q_cat_logits:    (B, q_cat)
            q_cont_mu:       (B, q_cont) or None
            q_cont_logsigma: (B, q_cont) or None
        """
        # ----- shared backbone -----
        h = nn.Conv(
            self.features,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='VALID',
            kernel_init=normal_init(0.02),
        )(x)
        h = nn.BatchNorm(
            use_running_average=not self.training,
            axis=-1,
            momentum=0.1,
            scale_init=normal_init(0.02),
        )(h)
        h = nn.leaky_relu(h, 0.2)

        h = nn.Conv(
            self.features * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='VALID',
            kernel_init=normal_init(0.02),
        )(h)
        h = nn.BatchNorm(
            use_running_average=not self.training,
            axis=-1,
            momentum=0.1,
            scale_init=normal_init(0.02),
        )(h)
        h = nn.leaky_relu(h, 0.2)

        # At this point the spatial size is 5x5; next heads go from there.

        # ----- D head -----
        d = nn.Conv(
            1,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='VALID',
            kernel_init=normal_init(0.02),
        )(h)        # -> (B, 1, 1, 1)
        d = d.reshape((d.shape[0], -1))  # (B, 1)
        d_logits = d  # treat as logits; apply sigmoid in loss if desired

        # ----- Q trunk -----
        q = nn.Conv(
            self.features * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='VALID',
            kernel_init=normal_init(0.02),
        )(h)        # -> (B, 1, 1, 2*features)
        q = nn.BatchNorm(
            use_running_average=not self.training,
            axis=-1,
            momentum=0.1,
            scale_init=normal_init(0.02),
        )(q)
        q = nn.leaky_relu(q, 0.2)
        q = q.reshape((q.shape[0], -1))  # (B, 2*features)

        q_flat = q.reshape((q.shape[0], -1))

        # ----- Q categorical head -----
        q_cat_logits = nn.Dense(
            self.q_cat,
            kernel_init=normal_init(0.02),
        )(q)

        # ----- Q continuous head -----
        q_cont_mu = nn.Dense(
                self.q_cont,
                kernel_init=normal_init(0.02),
                )(q)
        q_cont_logsigma = nn.Dense(
                self.q_cont,
                kernel_init=normal_init(0.02),
                )(q)

        return d_logits, q_cat_logits, q_cont_mu, q_cont_logsigma


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

@partial(jax.jit, static_argnames=['solver'])
def train_step_disc(state, data, labels, fake_imgs, fake_cat_input, con_codes, solver):
       
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
           
        def continuous_loss(x, mu, var):
            # Simple MSE for mean prediction
            mse = jnp.mean((x - mu) ** 2)
            
            # Regularize variance to stay near 1.0
            var_reg = jnp.mean((var - 1.0) ** 2) * 0.1
            
            return mse + var_reg

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
                  (fake_preds, q, mu, var), vars_d = Discriminator().apply(
                      {'params': params_d, 'batch_stats': vars_d_batch_stats},
                      fake_imgs, mutable=['batch_stats']
                  )
                  (real_preds, _, _, _), vars_d = Discriminator().apply(
                      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
                      data, mutable=['batch_stats']
                  )
                
                  # use q_logits and labels to calculate q accuracy
                  #q_preds = q_logits.argmax(axis=-1)
                  #q_acc = jnp.mean(q_preds == labels)
                  #jax.debug.print('Q accuracy: {} ', q_acc)
                  # Calculate Mutual Information loss
                  q_cat = nn.log_softmax(q, axis=-1)
                  loss_mi = loss_mutual_information(fake_cat_input, q_cat)
                  #loss_mi = cpc_mi_loss(fake_cat_input, q_cat, negative_samples=10)
                  #loss_con = normal_nll_loss(con_codes, mu, var)
                  loss_con = continuous_loss(con_codes, mu, var)
                  #predicted_cat = jnp.argmax(q, axis=-1)
                  
                  #true_cat = jnp.argmax(fake_cat_input, axis=-1)

                  #accuracy = jnp.mean(predicted_cat == true_cat)

                  #jax.debug.print('accuracy: {} ', accuracy)

                  # real_preds reshape array of shape (64, 0) (size 0) to (64,)
                  #real_preds = real_preds.reshape((real_preds.shape[0],))
                  #fake_preds = fake_preds.reshape((fake_preds.shape[0],))
                  #real_loss = bce_logits(real_preds, jnp.ones((32,), dtype=jnp.int32))
                  #fake_loss = bce_logits(fake_preds, jnp.zeros((32,), dtype=jnp.int32))
              
                  # use 0.9 as the label for real images instead of 1.0
                  real_loss = optax.sigmoid_binary_cross_entropy(real_preds, jnp.ones_like(real_preds)*0.95)
                  # use 0.1 as the label for fake images instead of 0.0
                  fake_loss = optax.sigmoid_binary_cross_entropy(fake_preds, jnp.zeros_like(fake_preds))

                  real_loss = jnp.mean(real_loss)
                  fake_loss = jnp.mean(fake_loss)

                  #jax.debug.print('real loss: {} ', real_loss)
                  #jax.debug.print('fake loss: {} ', fake_loss)

                  #jax.debug.print('mi loss: {} ', loss_mi)
                  #jax.debug.print('con loss: {} ', loss_con)
                  loss = (real_loss + fake_loss) / 2.0 + loss_mi*0.6 + loss_con*0.1
                
                  return loss, vars_d

        grad_fn_disc = jax.value_and_grad(loss_discriminator, has_aux=True)
        (loss, vars_d), grads = grad_fn_disc(params_d, batch_stats_d)
        
        # apply gradients
        updates, new_opt_state = solver.update(grads, opt_disc, params_d)
        params_d = optax.apply_updates(params_d, updates)
        #batch_stats_g = vars_g['batch_stats']
        # update batch stats
        batch_stats_d = vars_d['batch_stats']
        return (params_d, batch_stats_d, new_opt_state), loss

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
def build_big_latents(key, total_size, z_dim, n_disc, n_con):
    k_z, k_c, k_perm = jax.random.split(key, 3)
    z = jax.random.normal(k_z, (total_size, z_dim))
    reps = (total_size + n_disc - 1) // n_disc
    codes = jnp.tile(jnp.arange(n_disc), reps)[:total_size]
    c_onehot = jax.nn.one_hot(codes, n_disc)
    c_cont = jax.random.uniform(k_c, (total_size, n_con), minval=-1., maxval=1.)
    latent = jnp.concatenate([z, c_onehot, c_cont], axis=-1)
    return latent[jax.random.permutation(k_perm, total_size)]

def build_recal_latents(key, batch_size, z_dim, n_disc, n_con):
    k_z, k_con, k_perm = jax.random.split(key, 3)
    # z: standard normal (or whatever you use at train time)
    z = jax.random.normal(k_z, (batch_size, z_dim))
    # categorical: balanced 0..n_disc-1 repeated
    reps = (batch_size + n_disc - 1) // n_disc
    c = jnp.tile(jnp.arange(n_disc), reps)[:batch_size]
    c_onehot = jax.nn.one_hot(c, n_disc)
    # continuous: same range you use at train time
    c_cont = jax.random.uniform(k_con, (batch_size, n_con), minval=-1.0, maxval=1.0)
    # concatenate
    latent = jnp.concatenate([z, c_onehot, c_cont], axis=-1)
    # (optional) small permutation to avoid repeating same order every batch
    perm = jax.random.permutation(k_perm, batch_size)
    return latent[perm]

# --- BN standing-stats pass ---
def recalibrate_bn_stats(gen_module, params, batch_stats, key,
                         steps=12, batch_size=64,
                         z_dim=62, n_disc=10, n_con=2):
    """
    gen_module: e.g., Generator(training=True) or Generator(use_running_average=False)
    params: generator params pytree
    batch_stats: current running stats collection
    key: PRNGKey
    """
    def body(i, carry):
        bs, k = carry
        k, k_lat = jax.random.split(k)
        lat = build_recal_latents(k_lat, batch_size, z_dim, n_disc, n_con)
        # train-mode apply: mutate batch_stats only
        _, vars_out = gen_module.apply({'params': params, 'batch_stats': bs},
                                       lat,
                                       mutable=['batch_stats'])
        return (vars_out['batch_stats'], k)

    # run steps times; jit the whole loop
    (batch_stats_new, key_new) = lax.fori_loop(0, steps, body, (batch_stats, key))
    return batch_stats_new, key_new

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

  con = jax.random.uniform(con_key, (shape_cat[0], 2), minval=-1, maxval=1)
  
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

        self.batch_stats_gen = policy_gen.flat_batch_stats_gen
        self.batch_stats_disc = policy_gen.flat_batch_stats_disc

        self.batch_size = batch_size
        self.mini_batch_size = 64
        self.num_mini_batches = 1
       
        self.avg_mi_loss = -2.30
        self.fake_imgs = None
        self.cat_codes = None

        self.latent_dim = 64
        self.n_classes = 10
        self.n_con = 2

        self.decay_factor = 0.9

        self.noise_dim = self.latent_dim - self.n_con
        self.policy_gen = policy_gen

        self._key = jax.random.PRNGKey(44)

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

        self._key, subkey = jax.random.split(self._key)
        
        dataset = datasets.MNIST('./data', train=True, download=True)
        self.data = np.expand_dims(dataset.data.numpy() / 127.5 - 1.0, axis=-1)
        self.labels = dataset.targets.numpy()

        # initialize the discriminator
        variables_disc = Discriminator().init(subkey, jnp.ones((self.batch_size, 28, 28, 1), dtype=jnp.float32))
        self.params_disc, self.batch_stats_disc = variables_disc['params'], variables_disc['batch_stats']

        self.solver_disc = optax.adam(learning_rate=0.0001, b1=0.5, b2=0.999)

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

        if self.model_dir is not None:
            params_gen, self.batch_stats_gen = load_model_gen(model_dir=self.model_dir)
            params_disc, self.batch_stats_disc = load_model_disc(model_dir=self.model_dir)
            #self.sim_mgr.obs_params = obs_params
            self._logger.info(
                'Loaded model parameters from {}.'.format(self.model_dir))
        else:
            params_gen, params_disc, params_q = None, None, None

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

            if params_gen is not None and params_disc is not None and params_q is not None:
                # Continue training from the breakpoint.
                self.solver_gen.best_params = params_gen

            best_score_gen, best_score_disc, best_score_q = -float('Inf'), -float('Inf'), -float('Inf')

            params_disc = self.params_disc
           
            num_mini_batches = self.num_mini_batches
           
            self._key, noise_key, con_key = jax.random.split(self._key, 3)

            fixed_batch_latent = jax.random.normal(noise_key, (self.batch_size, self.latent_dim-self.n_con))
            fixed_c = jnp.tile(jnp.arange(10), 7)
            fixed_c = fixed_c[:self.batch_size]
            fixed_con = jax.random.uniform(con_key, (self.batch_size, 2), minval=-1, maxval=1) 
                    
            fixed_latent = jnp.concatenate([fixed_batch_latent, jax.nn.one_hot(fixed_c, 10), fixed_con], axis=-1)

            self._key, noise_key, con_key = jax.random.split(self._key, 3)

            for i in range(self._max_iter):
                
                shape_noise = (self.mini_batch_size, self.latent_dim-self.n_con)
                shape_cat = (self.mini_batch_size,)

                if i < 1:
                    if len(self.batch_stats_gen.shape) == 1:
                        batch_stats_gen = jnp.expand_dims(self.batch_stats_gen, axis=0)
                
                    batch_stats_gen = self.policy_gen._format_batch_stats_gen_fn(batch_stats_gen)

                if i % 1 == 0:
                    for mini_batch in range(num_mini_batches):
                        # Sample batch of data.

                        self._key, subkey_latent, subkey_mnist = jax.random.split(self._key, 3)
                        

                        data, labels = sample_batch(subkey_mnist, self.data, self.labels, self.mini_batch_size)
                        #data = np.expand_dims(data / 255.0, axis=-1)

                        latent, cat_codes, con_codes = sample_latent(subkey_latent, shape_noise, shape_cat)
                      
                        if i < 2:
                            params_gen = self.solver_gen.best_params
                            best_params_gen_formatted = self.policy_gen._format_single_params_gen_fn(params_gen)
                        #else:
                        
                        (fake_images) = Generator(training=False).apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},latent, mutable=False)
                        #batch_stats_gen = vars_g['batch_stats']
                        # reshape fake_images to (64, 28, 28, 1) from [1,1,1,64, 28, 28, 1]
                        fake_images = fake_images.reshape((self.mini_batch_size, 28, 28, 1))
                        
                        state = (params_disc, self.batch_stats_disc, opt_disc)

                        state, d_loss = train_step_disc(
                            state,
                            data,
                            labels,
                            fake_images,
                            cat_codes,
                            con_codes,
                            solver_disc,
                        )

                        #jax.debug.print('loss: {} ', loss)
                        
                        params_disc, self.batch_stats_disc, opt_disc = state 

                leaves_params, _ = jax.tree_flatten(params_disc) 
                flat_params_disc = jnp.concatenate([p.flatten() for p in leaves_params])

                leaves_batch_stats_disc, _ = jax.tree_flatten(self.batch_stats_disc)
                flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

                params_gen, belief_space = self.solver_gen.ask()
                
                self._key, subkey_recal = jax.random.split(self._key)
                # During recal:
                big_bs   = 64 * 8            # e.g. 512; use what fits memory
                lat_big  = build_big_latents(subkey_recal, big_bs, (self.latent_dim - self.n_con), 10, self.n_con)
                
                # IMPORTANT: construct the generator with BN in train-mode AND momentum=0.0 just for this call.
                gen_recal = Generator(training=True)  # add bn_momentum arg in your Module if needed
                
                # One forward that updates only batch_stats
                _, vars_out = gen_recal.apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},
                                              lat_big, mutable=['batch_stats'])
                batch_stats_gen = vars_out['batch_stats']  # <- frozen for next gen scoring               

                leaves_batch_stats_gen, _ = jax.tree_flatten(batch_stats_gen)
                flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])


                self._key, key_z_fixed, key_z, key_con_fixed, key_con = jax.random.split(self._key, 5)
                
                z_base_fixed = jax.random.normal(key_z_fixed, (3, 62))  # 6 different z vectors
                z_base = jax.random.normal(key_z, (30, 62)) 
                
                con_base_fixed = jax.random.uniform(key_con_fixed, (3, 2), minval=-1, maxval=1)  # 6 different continuous codes
                #z_base_concat = jnp.concatenate([z_base, fixed_z_subset], axis=0) 
                z_block = jnp.repeat(z_base_fixed, 10, axis=0)  # Repeat each z 10 times for each categorical code
                z_block = jnp.concatenate([z_base, z_block], axis=0)
                #con_base_concat = jnp.concatenate([con_base, fixed_con_subset], axis=0)
                #con_block = jnp.repeat(con_base_concat, 10, axis=0)  # Repeat each con code 10 times
                codes60 = jnp.tile(jnp.arange(10, dtype=jnp.int32), 6)  # Categorical codes from 0 to 9, repeated 6 times
                onehot60 = jax.nn.one_hot(codes60, 10)
                
                con_base = jax.random.uniform(key_con, (30, 2), minval=-1, maxval=1)
  
                con_block = jnp.repeat(con_base_fixed, 10, axis=0)  # Repeat each con code 10 times

                con_block = jnp.concatenate([con_base, con_block], axis=0)

                latent60 = jnp.concatenate([z_block, onehot60, con_block], axis=-1)

                self._key, key_z, key_con = jax.random.split(self._key, 3)

                z_block4 = jax.random.normal(key_z, (4, 62))  # 4 different z vectors
                con_block4 = jax.random.uniform(key_con, (4, 2), minval=-1, maxval=1)

                z_block_full = jnp.concatenate([z_block, z_block4], axis=0)
                con_block_full = jnp.concatenate([con_block, con_block4], axis=0)
                
                #latent = jnp.concat([latent60, latent60[:4]], axis=0)
                c_onehot = jnp.concat([onehot60, onehot60[:4]], axis=0)
                latent = jnp.concat([z_block_full, c_onehot, con_block_full], axis=1)
                #con_full_block = jnp.concat([con_block, con_block[:4]], axis=0)
                #con_full_block = jnp.concatenate([con_block, jax.random.uniform(key_con, (4, 2), minval=-1, maxval=1)], axis=0)

                topographic_ks = belief_space[4]

                #jax.debug.print('topographic_ks shape: {} ', topographic_ks.shape)
                avg_per_code = topographic_ks[0]
                
                scores_gen_adv, scores_gen_mi, scores_gen_con, disc_logits, bds_gen, BN_stats_gen, _, mean_var_fake, avg_per_code_current, r_cons, r_sense, r_intra = self.sim_mgr_gen.eval_params(
                params_gen=params_gen, params_disc=flat_params_disc, batch_stats_gen=flat_batch_stats_gen, batch_stats_disc=flat_batch_stats_disc, latent=latent, cat_codes=c_onehot, codes60=codes60, con_codes=con_block_full, features=avg_per_code, generator=True, test=False
                )

                #jax.debug.print('fake_imgs shape: {} ', fake_imgs.shape)
                if isinstance(self.solver_gen, QualityDiversityMethod):
                    self.solver_gen.observe_bd(bds_gen)
                
                self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_con, disc_logits=disc_logits, pop_var=mean_var_fake, avg_per_code=avg_per_code_current, r_cons=r_cons, r_sense=r_sense, r_intra=r_intra, adv=False)

                #params_gen, belief_space = self.solver_gen.ask()

                #scores_gen_adv, scores_gen_mi, scores_gen_con, disc_logits, bds_gen, BN_stats_gen, _, _ = self.sim_mgr_gen.eval_params(
                #params_gen=params_gen, params_disc=flat_params_disc, batch_stats_gen=flat_batch_stats_gen, batch_stats_disc=flat_batch_stats_disc,  generator=True, test=False
                #)

                #if isinstance(self.solver_gen, QualityDiversityMethod):
                #    self.solver_gen.observe_bd(bds_gen)
                #
                #self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_con, disc_logits=disc_logits, adv=True)

                best_params_gen = self.solver_gen.best_params
                best_params_gen_formatted = self.policy_gen._format_single_params_gen_fn(best_params_gen)

                #self._key, subkey = jax.random.split(self._key)
                #shape_noise = (self.batch_size, self.latent_dim-self.n_con)
                #shape_cat = (self.batch_size,)
                #latent, cat_codes, con_codes = sample_latent(subkey, shape_noise, shape_cat)

                #(fake_images), vars_g = Generator().apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},latent, mutable=['batch_stats'])
                #batch_stats_gen = vars_g['batch_stats'] 

                #gen_train = Generator(training=True)
                #batch_stats_gen, self._key = recalibrate_bn_stats(
                #    gen_train,
                #    best_params_gen_formatted,
                #    batch_stats_gen,
                #    self._key,
                #    steps=12,
                #    batch_size=self.batch_size,
                #    z_dim=self.latent_dim - self.n_con,
                #    n_disc=10,
                #    n_con=self.n_con,
                #)
              
                self._key, subkey_recal = jax.random.split(self._key)
                # During recal:
                big_bs   = 64 * 8            # e.g. 512; use what fits memory
                lat_big  = build_big_latents(subkey_recal, big_bs, (self.latent_dim - self.n_con), 10, self.n_con)
                
                # IMPORTANT: construct the generator with BN in train-mode AND momentum=0.0 just for this call.
                gen_recal = Generator(training=True)  # add bn_momentum arg in your Module if needed
                
                # One forward that updates only batch_stats
                _, vars_out = gen_recal.apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},
                                              lat_big, mutable=['batch_stats'])
                batch_stats_gen = vars_out['batch_stats']  # <- frozen for next gen scoring               

                #(_, _, _, _), vars_d = Discriminator().apply({'params': params_disc, 'batch_stats': self.batch_stats_disc}, fake_images, mutable=['batch_stats'])
                #self.batch_stats_disc = vars_d['batch_stats']

                #leaves_batch_stats_gen, _ = jax.tree_flatten(batch_stats_gen)
                #flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

                #leaves_batch_stats_disc, _ = jax.tree_flatten(self.batch_stats_disc)
                #flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

                #params_gen, belief_space = self.solver_gen.ask()

                #scores_gen_adv, scores_gen_mi, scores_gen_con, disc_logits, bds_gen, BN_stats_gen, _, _ = self.sim_mgr_gen.eval_params(
                #params_gen=params_gen, params_disc=flat_params_disc, batch_stats_gen=flat_batch_stats_gen, batch_stats_disc=flat_batch_stats_disc, latent=latent, cat_codes=cat_codes, con_codes=con_codes,  generator=True, test=False
                #)

                #if isinstance(self.solver_gen, QualityDiversityMethod):
                #    self.solver_gen.observe_bd(bds_gen)
                
                #self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_con, disc_logits=disc_logits, adv=True)

                #params_gen, belief_space = self.solver_gen.ask()
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

                    scores_con = np.array(scores_gen_con)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores_con.size, scores_con.max(), scores_con.mean(),
                            scores_con.min(), scores_con.std()))

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

                    #fixed_batch_latent = jax.random.normal(noise_key, (self.batch_size, self.latent_dim-self.n_con))
                    #fixed_c = jnp.tile(jnp.arange(10), 7)
                    #fixed_c = fixed_c[:self.batch_size]
                    #fixed_con = jax.random.uniform(con_key, (self.batch_size, 2), minval=-1, maxval=1) 
                            
                    #fixed_latent = jnp.concatenate([fixed_batch_latent, jax.nn.one_hot(fixed_c, 10), fixed_con], axis=-1)


                    #batch_stats_gen_leaves, _ = jax.tree_flatten(batch_stats_gen)
                    #flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in batch_stats_gen_leaves])
                    
                    (fake_imgs) = Generator(training=False).apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},fixed_latent)
                    
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
            save_model(
                model_dir=self._log_dir,
                model_name='final_model_gen',
                params=best_params_gen,
                obs_params=self.sim_mgr_gen.obs_params,
                batch_stats=batch_stats_gen,
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

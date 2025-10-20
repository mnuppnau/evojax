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
from typing import Tuple

import jax
import optax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

@dataclass
class State(TaskState):
    obs: jnp.ndarray
    #latent_input: jnp.ndarray
    cat_codes: jnp.ndarray
    con_codes: jnp.ndarray
    batch_stats_gen: any
    batch_stats_disc: any
    batch_stats_q: any

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
                
                c = jnp.tile(jnp.arange(10),7)
                # remove the last 4 elements to make it 256
                c = c[:self.batch_size]
                batch_cat_one_hot = jax.nn.one_hot(c, 10)
                
                #batch_cat_one_hot = jax.nn.one_hot(batch_cat, self.n_classes)
                # FIXED categorical code (same for all 64 images in this evaluation)
                #c_cat_idx = random.randint(cat_key, (), 0, self.n_classes)
                #batch_cat_one_hot = jax.nn.one_hot(jnp.full((self.batch_size,), c_cat_idx), self.n_classes)
                

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
            
            return State(obs=batch_latent_concat, cat_codes=batch_cat_one_hot, con_codes=batch_con, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc, batch_stats_q=self.batch_stats_q)
        
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action, q, mu, var):
           
            q_cat = jax.nn.log_softmax(q, axis=-1)
            
            #loss_q_disc = loss_mutual_information(state.cat_codes, q_cat)

            loss_q_disc = cpc_mi_loss(state.cat_codes, q_cat, negative_samples=10)

            loss_g = bce_logits(action, jnp.ones((self.batch_size,), dtype=jnp.int32))
           
            #loss_con = neg_log_likelihood_normal(state.con_codes, action, jnp.zeros_like(action))
            
            #loss_con = normal_nll_loss(state.con_codes, mu, var)*0.1

            loss_q_disc = -loss_q_disc
            loss_con = continuous_loss(state.con_codes, mu, var)
            #loss_con = -loss_con
            loss_g = -loss_g#*0.1 + loss_q_disc# + loss_q_cont*0.005
            
            return state, loss_q_disc, loss_g, loss_con, jnp.ones(())
        
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key1: jnp.ndarray, key2: jnp.ndarray, key3: jnp.ndarray) -> State:
        return self._reset_fn(key1, key2, key3)

    def step(self,
             state: TaskState,
             action: jnp.ndarray,
             disc_logits: jnp.ndarray,
             mu: jnp.ndarray,
             var: jnp.ndarray) -> tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action, disc_logits, mu, var)

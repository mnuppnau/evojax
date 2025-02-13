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
    batch_stats_gen: any
    batch_stats_disc: any

def sample_batch(key: jnp.ndarray,
                 latent_inputs: jnp.ndarray,
                 cat_codes: jnp.ndarray,
                 batch_size: int) -> tuple:
    ix = random.choice(
        key=key, a=latent_inputs.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(latent_inputs, indices=ix, axis=0),
            jnp.take(cat_codes, indices=ix, axis=0))

def loss_mutual_information(code_cat, q_cat):
    cat_loss = -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))
    mi_loss = -cat_loss
    return mi_loss

def bce_logits(logit, label):
    neg_abs = -jnp.abs(logit)
    batch_bce = jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(neg_abs))
    return jnp.mean(batch_bce)

def neg_log_likelihood_normal(x, mean, logvar):
    return 0.5 * jnp.mean(jnp.sum(logvar + jnp.exp(-logvar) * (x - mean) ** 2, axis=-1))

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

        # Generate the dataset
        key = random.PRNGKey(42)
        self.key = key
        latent_key, cat_key, con_key = random.split(key,3)
        
        noise = random.normal(latent_key, (dataset_size, self.noise_dim))
        cat_codes = random.randint(cat_key, (dataset_size,), 0, n_classes)
        self.cat_codes = jax.nn.one_hot(cat_codes, n_classes)
        self.con_codes = random.uniform(con_key, (dataset_size, n_con), minval=-1, maxval=1)

        #jax.debug.print('latent input shape : {} ', latent_inputs.shape)
        #jax.debug.print('cat codes shape : {} ', self.cat_codes.shape)

        self.latent_inputs = jnp.concatenate([noise, self.cat_codes, self.con_codes], axis=1)

        #jax.debug.print('latent input shape after concat : {} ', self.latent_inputs.shape)
        def reset_fn(noise_key, cat_key, con_key):
            if test:
                #batch_latent_concat, batch_cat_one_hot = sample_batch(
                #    key, self.latent_inputs, self.cat_codes, 10) 
                batch_latent = random.normal(noise_key, (self.batch_size, self.latent_dim))
                
                #structured_codes = jnp.tile(jnp.arange(10), 10)  # Shape: (100,)
                #structured_codes = nn.one_hot(structured_codes, 10)

                #jax.debug.print('structured codes : {}', structured_codes)
                #c = jnp.ones((30,)) + 2
                c = jnp.tile(jnp.arange(10),26)
                # remove the last 4 elements to make it 256
                c = c[:self.batch_size]
                batch_cat_one_hot = jax.nn.one_hot(c, 10)
                
                # Step 2: Generate the remaining random one-hot encoded vectors
                #num_random_samples = self.batch_size - 100
                
                #random_cat_codes = random.randint(cat_key, (num_random_samples,), 0, 10)
                #random_cat_codes = nn.one_hot(random_cat_codes, 10)

                #batch_cat = random.randint(cat_key, (self.batch_size,), 0, n_classes)
                #batch_cat_one_hot = jax.nn.one_hot(batch_cat, n_classes)
                # Step 3: Concatenate the structured and random codes
                #batch_cat_one_hot = jnp.concatenate([structured_codes, random_cat_codes], axis=0)
                #batch_con = random.uniform(con_key, (self.batch_size, self.n_con), minval=-1, maxval=1)

                batch_latent_concat = jnp.concatenate([batch_latent, batch_cat_one_hot], axis=-1)


                #batch_latent_concat = jnp.concatenate([batch_latent, batch_cat_one_hot], axis=1)
                #batch_cat = random.randint(key, (self.batch_size,), 0, self.n_classes)
                #batch_latent_concat = jnp.concatenate([batch_latent, jax.nn.one_hot(batch_cat, self.n_classes)], axis=-1)

                #batch_cat_one_hot = jax.nn.one_hot(batch_cat, self.n_classes)

            else:
                #batch_latent, batch_cat = sample_batch(
                #    key, self.latent_inputs, self.cat_codes, batch_size)
                #latent_key, cat_key = random.split(self.key)
                batch_latent = random.normal(noise_key, (self.batch_size, self.latent_dim))
                #jax.debug.print('batch latent : {}', batch_latent)

                batch_cat = random.randint(cat_key, (self.batch_size,), 0, self.n_classes)
                
                #batch_latent_concat = jnp.concatenate([batch_latent, jax.nn.one_hot(batch_cat, self.n_classes)], axis=-1)

                batch_cat_one_hot = jax.nn.one_hot(batch_cat, self.n_classes)

                #jax.debug.print('batch cat one hot : {}', batch_cat_one_hot)

                #batch_con = random.uniform(con_key, (self.batch_size, self.n_con), minval=-1, maxval=1)

                #jax.debug.print('batch con : {}', batch_con)
                batch_latent_concat = jnp.concatenate([batch_latent, batch_cat_one_hot], axis=-1)

            return State(obs=batch_latent_concat, cat_codes=batch_cat_one_hot, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc)
        
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action, q):
           
            #jax.debug.print('q shape : {} ', q.shape)
            q_cat = jax.nn.log_softmax(q, axis=-1)
            # print last 10 features of the latent input batch
            #jax.debug.print('latent input : {} ', state.obs[:,-10:]) 
            #jax.debug.print('cat codes : {} ', state.cat_codes)
            
            loss_q_disc = loss_mutual_information(state.cat_codes, q_cat)
            #loss_q_disc = optax.softmax_cross_entropy(state.cat_codes, q_cat).mean()
            #loss_q_cont = jnp.mean(jnp.sum(0.5 * jnp.log(2 * jnp.pi * logvar) + 0.5 * (state.con_codes - mu) ** 2 / logvar, axis=-1))
            
            #loss_q_disc = -loss_q_disc
            #loss_mi = loss_q_disc + loss_q_cont

            loss_g = bce_logits(action, jnp.ones((self.batch_size,), dtype=jnp.int32))
            #loss_g = optax.sigmoid_binary_cross_entropy(action, jnp.ones((self.batch_size,), dtype=jnp.int32)).mean()
            ##loss_con = neg_log_likelihood_normal(state.con_codes, mu, logvar)
            
            #loss_g = -jnp.mean(jnp.log(nn.sigmoid(action)))
            loss_g = -loss_g#*0.1 + loss_q_disc# + loss_q_cont*0.005
            #1jax.debug.print('loss g shape in latent: {} ', loss_g.shape)
            #jax.debug.print('loss mi gen : {} ', loss_mi)
            #jax.debug.print('loss con gen : {} ', loss_con)
            
            #loss_d = -jnp.mean(jnp.log(nn.sigmoid(action)))
            #loss_g = loss_g + loss_mi*0.2 + loss_con*0.05
            #jax.debug.print('loss g: {} ', loss_g)
            #Add weight to loss_mi
            #loss_mi = 1.8 * loss_mi
            #loss = loss_mi + loss_g + loss_con
            #reward = -loss # Minimize the loss
            return state, loss_q_disc, loss_g, jnp.ones(())
        
        self._step_fn = jax.jit(jax.vmap(step_fn))

    #def set_batch_stats(self, batch_stats_gen, batch_stats_disc):
    #    self.batch_stats_gen = batch_stats_gen
    #    self.batch_stats_disc = batch_stats_disc

    def reset(self, key1: jnp.ndarray, key2: jnp.ndarray, key3: jnp.ndarray) -> State:
        return self._reset_fn(key1, key2, key3)

    def step(self,
             state: TaskState,
             action: jnp.ndarray,
             disc_logits: jnp.ndarray) -> tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action, disc_logits)

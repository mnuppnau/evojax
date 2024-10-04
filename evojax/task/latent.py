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
    mi_loss = cat_loss
    return mi_loss

class Latent_Points(VectorizedTask):
    """Latent point task for InfoGAN Generator."""

    def __init__(self,
                 batch_size: int = 1024,
                 dataset_size: int = 800,  # Similar to MNIST
                 latent_dim: int = 64,
                 n_classes: int = 10,
                 testing: bool = False,
                 test: bool = False):
        self.max_steps = 1
        self.obs_shape = (latent_dim + n_classes,)

        self.batch_stats_gen = None
        self.batch_stats_disc = None
        
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Generate the dataset
        key = random.PRNGKey(42)
        self.key = key
        latent_key, cat_key = random.split(key)
        
        latent_inputs = random.normal(latent_key, (dataset_size, latent_dim))
        cat_codes = random.randint(cat_key, (dataset_size,), 0, n_classes)
        self.cat_codes = jax.nn.one_hot(cat_codes, n_classes)

        jax.debug.print('latent input shape : {} ', latent_inputs.shape)
        jax.debug.print('cat codes shape : {} ', self.cat_codes.shape)

        self.latent_inputs = jnp.concatenate([latent_inputs, self.cat_codes], axis=-1)

        jax.debug.print('latent input shape after concat : {} ', self.latent_inputs.shape)
        def reset_fn(noise_key, cat_key):
            if testing:
                #batch_latent_concat, batch_cat_one_hot = sample_batch(
                #    key, self.latent_inputs, self.cat_codes, 10) 
                batch_latent = random.normal(noise_key, (self.batch_size, self.latent_dim))
                
                #structured_codes = jnp.tile(jnp.arange(10), 10)  # Shape: (100,)
                #structured_codes = nn.one_hot(structured_codes, 10)

                #jax.debug.print('structured codes : {}', structured_codes)
                c = jnp.ones((128,)) + 5
                batch_cat_one_hot = jax.nn.one_hot(c, 10)
                
                # Step 2: Generate the remaining random one-hot encoded vectors
                #num_random_samples = self.batch_size - 100
                
                #random_cat_codes = random.randint(cat_key, (num_random_samples,), 0, 10)
                #random_cat_codes = nn.one_hot(random_cat_codes, 10)

                # Step 3: Concatenate the structured and random codes
                #batch_cat_one_hot = jnp.concatenate([structured_codes, random_cat_codes], axis=0)

                batch_latent_concat = jnp.concatenate([batch_latent, batch_cat_one_hot], axis=-1)
                #batch_cat = random.randint(key, (self.batch_size,), 0, self.n_classes)
                #batch_latent_concat = jnp.concatenate([batch_latent, jax.nn.one_hot(batch_cat, self.n_classes)], axis=-1)

                #batch_cat_one_hot = jax.nn.one_hot(batch_cat, self.n_classes)

            else:
                #batch_latent, batch_cat = sample_batch(
                #    key, self.latent_inputs, self.cat_codes, batch_size)
                #latent_key, cat_key = random.split(self.key)
                batch_latent = random.normal(noise_key, (self.batch_size, self.latent_dim))
                batch_cat = random.randint(cat_key, (self.batch_size,), 0, self.n_classes)
                batch_latent_concat = jnp.concatenate([batch_latent, jax.nn.one_hot(batch_cat, self.n_classes)], axis=-1)

                batch_cat_one_hot = jax.nn.one_hot(batch_cat, self.n_classes)

            return State(obs=batch_latent_concat, cat_codes=batch_cat_one_hot, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc)
        
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action, q):
           
            #jax.debug.print('q shape : {} ', q.shape)
            q_cat = jax.nn.log_softmax(q, axis=-1)
            #jax.debug.print('q cat : {} ', q_cat)
            #jax.debug.print('cat codes : {} ', state.cat_codes)
            loss_mi = loss_mutual_information(state.cat_codes, q_cat)
            loss_d = -jnp.mean(jnp.log(nn.sigmoid(action)))
            loss = loss_mi + loss_d
            #jax.debug.print('loss d: {} ', loss_d)
            #jax.debug.print('loss mi: {} ', loss_mi)
            reward = -loss # Minimize the loss
            return state, reward, jnp.ones(())
        
        self._step_fn = jax.jit(jax.vmap(step_fn))

    #def set_batch_stats(self, batch_stats_gen, batch_stats_disc):
    #    self.batch_stats_gen = batch_stats_gen
    #    self.batch_stats_disc = batch_stats_disc

    def reset(self, key1: jnp.ndarray, key2: jnp.ndarray) -> State:
        return self._reset_fn(key1, key2)

    def step(self,
             state: TaskState,
             action: jnp.ndarray,
             q: jnp.ndarray) -> tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action, q)

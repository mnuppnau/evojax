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
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

@dataclass
class State(TaskState):
    latent_input: jnp.ndarray
    cat_codes: jnp.ndarray

def sample_batch(key: jnp.ndarray,
                 latent_inputs: jnp.ndarray,
                 cat_codes: jnp.ndarray,
                 batch_size: int) -> tuple:
    ix = random.choice(
        key=key, a=latent_inputs.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(latent_inputs, indices=ix, axis=0),
            jnp.take(cat_codes, indices=ix, axis=0))

class Latent_Points(VectorizedTask):
    """Latent point task for InfoGAN Generator."""

    def __init__(self,
                 batch_size: int = 1024,
                 dataset_size: int = 60000,  # Similar to MNIST
                 latent_dim: int = 62,
                 n_classes: int = 10,
                 test: bool = False):
        self.max_steps = 1
        self.obs_shape = (latent_dim + n_classes,)
        self.act_shape = (28, 28, 1)  # Assuming MNIST-like output
        
        # Generate the dataset
        key = random.PRNGKey(0)
        latent_key, cat_key = random.split(key)
        
        self.latent_inputs = random.normal(latent_key, (dataset_size, latent_dim))
        cat_codes = random.randint(cat_key, (dataset_size,), 0, n_classes)
        self.cat_codes = jax.nn.one_hot(cat_codes, n_classes)

        def reset_fn(key):
            if test:
                batch_latent, batch_cat = self.latent_inputs, self.cat_codes
            else:
                batch_latent, batch_cat = sample_batch(
                    key, self.latent_inputs, self.cat_codes, batch_size)
            return State(latent_input=batch_latent, cat_codes=batch_cat)
        
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            # In a real scenario, you might want to implement a more meaningful
            # reward function based on your specific InfoGAN objectives.
            # For now, we'll use a placeholder reward.
            reward = jnp.zeros(())
            return state, reward, jnp.ones(())
        
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

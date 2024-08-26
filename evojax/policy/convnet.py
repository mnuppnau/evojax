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

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn


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

class Generator(nn.Module):
    """ Generator CNN for MNIST """

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
        x = jnp.tanh(x)
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
        q = nn.Conv(self.q_cat, [1, 1], [2, 2], 'VALID', kernel_init=normal_init(0.02))(q)
        q = q.reshape((q.shape[0], -1))
        return d, q


class GenPolicy(PolicyNetwork):
    """A convolutional neural network for the MNIST classification task."""

    def __init__(self, logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger('ConvNetPolicy')
        else:
            self._logger = logger

        self.model_gen = Generator()
        self.model_disc = Discriminator()
        
        key = random.PRNGKey(0)

        key, key_gen, key_disc, key_latent = random.split(key, 4)

        noise = random.normal(key_latent, (100, 64))
        c = jnp.tile(jnp.arrange(10), 10)
        c = jax.nn.one_hot(c, 10)

        latent = jnp.concatenate([noise, c], axis=-1)

        image_shape = (1, 28, 28, 1)

        variables_gen = self.model_gen.init(key_gen, jnp.ones(latent.shape, jnp.float32))
        variables_disc = self.model_disc.init(key_disc, jnp.ones(image_shape, jnp.float32))

        self.init_params_gen, self.init_batch_stats_gen = variables_gen['params'], variables_gen['batch_stats']
        self.init_params_disc, self.init_batch_stats_disc = variables_disc['params'], variables_disc['batch_stats']

        self.latent_dim = 64
        self.batch_size = 512
  
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info(
            'ConvNetPolicy.num_params = {}'.format(self.num_params))
        self._format_params_fn = jax.vmap(format_params_fn)

    def forward_fn_gen(self, params_g, vars_g_batch_stats, params_d, vars_d_batch_stats):
          
        z_input, cat_one_hot = generate_latent_points(key, self.latent_dim, self.batch_size)

        fake_data, vars_g = self.model_gen.apply({'params': params_g, 'batch_stats': vars_g_batch_stats}, z_input, mutable=['batch_stats'])
        (preds, q), vars_d = self.model_disc.apply({'params': params_d, 'batch_stats': vars_d_batch_stats}, fake_data, mutable=['batch_stats'])

        return fake_data, preds, q, vars_g, vars_d, q

    def forward_fn_disc(self, params_d, vars_d_batch_stats, real_data, fake_data):
        
        (real_preds, _), vars_d = self.model_disc.apply({'params': params_d, 'batch_stats': vars_d_batch_stats}, real_data, mutable=['batch_stats'])
        (fake_preds, q), vars_d = self.model_disc.apply({'params': params_d, 'batch_stats': vars_d_batch_stats}, fake_data, mutable=['batch_stats'])

        return real_preds, fake_preds, q, vars_d

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states

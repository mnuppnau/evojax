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
    """ Generator CNN for MNIST """

    features: int = 64
    training: bool = True

    @nn.compact
    def __call__(self, z):
        #jax.debug.print('z shape : {} ', z.shape)
        #if len(z.shape) == 3:
        #    z = z.reshape((z.shape[0], z.shape[1], 1, 1, z.shape[2]))
        #else:
        z = z.reshape((z.shape[0], 1, 1, z.shape[1]))
        x = nn.ConvTranspose(self.features*4, [3, 3], [2, 2], 'VALID', kernel_init=normal_init(0.02))(z)
        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
        x = nn.relu(x)
        activations1 = x
        x = nn.ConvTranspose(self.features*2, [4, 4], [1, 1], 'VALID', kernel_init=normal_init(0.02))(x)
        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
        x = nn.relu(x)
        activations2 = x
        x = nn.ConvTranspose(self.features, [3, 3], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
        x = nn.relu(x)
        activations3 = x
        x = nn.ConvTranspose(1, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
        x = jnp.tanh(x)
        return x, activations1, activations2, activations3

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
        #
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
        log_var = nn.Conv(features=2, kernel_size=(1, 1), strides=(1, 1))(q)
        var = jnp.squeeze(log_var)
    
        return d, disc_logits, mu.squeeze(), jnp.exp(var)

class GenPolicy(PolicyNetwork):
    """A convolutional neural network for the MNIST classification task."""

    def __init__(self, logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger('ConvNetPolicy')
        else:
            self._logger = logger

        self.model_gen = Generator()
       
        #self.model_bin_classifier = BinaryMNISTClassifier()
        
        key = random.PRNGKey(0)

        key, key_gen, key_disc, key_bin = random.split(key, 4)

        empty_state, self.model_bin_classifier = create_train_state(key_bin)
        
        loaded_state = load_model(empty_state, '/home/gh0st/projects/evojax/mnist-classification/models/state/')

        self.variables_bin_classifier = {'params': loaded_state.params}

        #noise = random.normal(key_latent, (100, 64))
        #c = jnp.tile(jnp.arange(10), 10)
        #c = jax.nn.one_hot(c, 10)

        #cat_codes = random.randint(key_latent, (64, 100), 0, 10)

        # Apply one-hot encoding
        #c = nn.one_hot(cat_codes, 10)
        
        #latent = jnp.concatenate([noise, c], axis=-1)

        #jax.debug.print('latent shape before init : {} ', latent.shape)

        variables_gen = self.model_gen.init(key_gen, jnp.ones([1,74], jnp.float32))

        self.init_params_gen, self.init_batch_stats_gen = variables_gen['params'], variables_gen['batch_stats']

        self.latent_dim = 64
  
        self.num_params, format_params_gen_fn = get_params_format_fn(self.init_params_gen)
        self._logger.info(
            'ConvNetPolicy.num_params = {}'.format(self.num_params))
        self._format_params_gen_fn = jax.vmap(format_params_gen_fn)

        self.num_batch_stats, format_batch_stats_gen_fn = get_params_format_fn(self.init_batch_stats_gen)
        self._logger.info(
            'ConvNetPolicy.num_batch_stats = {}'.format(self.num_batch_stats))
        self._format_batch_stats_gen_fn = jax.vmap(format_batch_stats_gen_fn)

        leaves_params, _ = jax.tree_util.tree_flatten(self.init_params_gen)
        
        self.flat_params_gen = jnp.concatenate([p.flatten() for p in leaves_params])

        leaves_batch_stats_gen, _ = jax.tree_util.tree_flatten(self.init_batch_stats_gen)

        self.flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

        def forward_fn_gen(params_g, vars_g_batch_stats, params_d, vars_d_batch_stats, latent_input):
          
            #z_input, cat_one_hot = generate_latent_points(key, self.latent_dim, self.batch_size)

            #jax.debug.print('latent input : {} ', latent_input[:, -10:])
            (fake_data, act1, act2, act3), vars_g = self.model_gen.apply({'params': params_g, 'batch_stats': vars_g_batch_stats}, latent_input, mutable=['batch_stats'])
       
            #act1, act2, act3 = activations

            bin_logits = self.model_bin_classifier.apply(self.variables_bin_classifier, fake_data)

            (preds, disc_logits, mu, var), vars_d = self.model_disc.apply({'params': params_d, 'batch_stats': vars_d_batch_stats}, fake_data, mutable=['batch_stats'])

            leaves_batch_stats_gen, _ = jax.tree_util.tree_flatten(vars_g['batch_stats'])

            flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

            leaves_batch_stats_disc, _ = jax.tree_util.tree_flatten(vars_d['batch_stats'])

            flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

            return fake_data, (act1,act2,act3), flat_batch_stats_gen, bin_logits, preds, disc_logits, mu, var, flat_batch_stats_disc

        self._forward_fn_gen = jax.vmap(forward_fn_gen)

    def set_format_params_disc_fn(self, format_params_disc_fn):
        self._format_params_disc_fn = format_params_disc_fn

    def set_format_batch_stats_disc_fn(self, format_batch_stats_disc_fn):
        self._format_batch_stats_disc_fn = format_batch_stats_disc_fn

    def set_model_disc(self, model_disc):
        self.model_disc = model_disc

    def get_actions(self,
                    t_states: TaskState,
                    params_gen: jnp.ndarray,
                    params_disc: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        
        params_gen = self._format_params_gen_fn(params_gen)
        params_disc = self._format_params_disc_fn(params_disc)

        batch_stats_gen = self._format_batch_stats_gen_fn(t_states.batch_stats_gen)
        batch_stats_disc = self._format_batch_stats_disc_fn(t_states.batch_stats_disc)
       
        #jax.debug.print('params gen : {} ', params_gen)

        fake_data, activations, batch_stats_g, bin_logits, preds, disc_logits, mu, var, batch_stats_d = self._forward_fn_gen(params_gen, batch_stats_gen, params_disc, batch_stats_disc, t_states.obs)
        
        return fake_data, activations, bin_logits, preds, disc_logits, mu, var, batch_stats_g, batch_stats_d, p_states
        #return self._forward_fn(params, t_states.obs), p_states

class DiscPolicy(PolicyNetwork):
    """A convolutional neural network for the MNIST classification task."""

    def __init__(self, gen_policy: GenPolicy, logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger('ConvNetPolicy')
        else:
            self._logger = logger

        self.model_disc = Discriminator()
        
        self.model_gen = gen_policy.model_gen

        key = random.PRNGKey(0)

        key, key_gen, key_disc, key_latent = random.split(key, 4)

        image_shape = (1, 28, 28, 1)

        variables_disc = self.model_disc.init(key_disc, jnp.ones(image_shape, jnp.float32))

        self.init_params_disc, self.init_batch_stats_disc = variables_disc['params'], variables_disc['batch_stats']

        self.num_params, format_params_disc_fn = get_params_format_fn(self.init_params_disc)
        self._logger.info(
            'ConvNetPolicy.num_params = {}'.format(self.num_params))
        self._format_params_disc_fn = jax.vmap(format_params_disc_fn)

        self.num_batch_stats, format_batch_stats_disc_fn = get_params_format_fn(self.init_batch_stats_disc)
        self._logger.info(
            'ConvNetPolicy.num_batch_stats = {}'.format(self.num_batch_stats))
        self._format_batch_stats_disc_fn = jax.vmap(format_batch_stats_disc_fn)

        self._format_params_gen_fn = gen_policy._format_params_gen_fn
        self._format_batch_stats_gen_fn = gen_policy._format_batch_stats_gen_fn

        gen_policy.set_model_disc(self.model_disc)

        gen_policy.set_format_params_disc_fn(self._format_params_disc_fn)

        gen_policy.set_format_batch_stats_disc_fn(self._format_batch_stats_disc_fn)

        leaves_params, _ = jax.tree_util.tree_flatten(self.init_params_disc)
        
        self.flat_params_disc = jnp.concatenate([p.flatten() for p in leaves_params])

        leaves_batch_stats_disc, _ = jax.tree_util.tree_flatten(self.init_batch_stats_disc)

        self.flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])


        def forward_fn_disc(params_g, vars_g_batch_stats, params_d, vars_d_batch_stats, real_data, latent_input, cat_codes):
            
            (fake_data, act1, act2, act3), vars_g = self.model_gen.apply({'params': params_g, 'batch_stats': vars_g_batch_stats}, latent_input, mutable=['batch_stats'])

            (real_preds, _, _, _), vars_d = self.model_disc.apply({'params': params_d, 'batch_stats': vars_d_batch_stats}, real_data, mutable=['batch_stats'])
            
            (fake_preds, disc_logits, mu, var), vars_d = self.model_disc.apply({'params': params_d, 'batch_stats': vars_d['batch_stats']}, fake_data, mutable=['batch_stats'])

            leaves_batch_stats_disc, _ = jax.tree_util.tree_flatten(vars_d['batch_stats'])

            flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

            leaves_batch_stats_gen, _ = jax.tree_util.tree_flatten(vars_g['batch_stats'])

            flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

            return real_preds, fake_preds, disc_logits, mu, var, flat_batch_stats_disc, flat_batch_stats_gen

        self._forward_fn_disc = jax.vmap(forward_fn_disc)

    def get_actions(self,
                    t_states: TaskState,
                    params_gen: jnp.ndarray,
                    params_disc: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        
        params_gen = self._format_params_gen_fn(params_gen)
        params_disc = self._format_params_disc_fn(params_disc)

        batch_stats_gen = self._format_batch_stats_gen_fn(t_states.batch_stats_gen)
        batch_stats_disc = self._format_batch_stats_disc_fn(t_states.batch_stats_disc)
       
        #jax.debug.print('params gen : {} ', params_gen)

        real_preds, fake_preds, disc_logits, mu, var, batch_stats_d, batch_stats_g = self._forward_fn_disc(params_gen, batch_stats_gen, params_disc, batch_stats_disc, t_states.obs, t_states.latent, t_states.cat_codes)
                
        return real_preds, fake_preds, disc_logits, mu, var, batch_stats_d, batch_stats_g, p_states

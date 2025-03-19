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
import optax
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass
from flax import linen as nn

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    latent: jnp.ndarray
    cat_codes: jnp.ndarray
    #con_codes: jnp.ndarray
    labels: jnp.ndarray
    cat_codes: jnp.ndarray
    #fake_imgs: jnp.ndarray
    batch_stats_gen: any = None
    batch_stats_disc: any = None
    batch_stats_q: any = None

def sample_batch(key: jnp.ndarray,
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 batch_size: int) -> Tuple:
    ix = random.choice(
        key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(data, indices=ix, axis=0),
            jnp.take(labels, indices=ix, axis=0))


def bce_logits(logit, label):
    """
    Implements the BCE with logits loss, as described:
    https://github.com/pytorch/pytorch/issues/751
    """
    neg_abs = -jnp.abs(logit)
    batch_bce = jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(neg_abs))
    return jnp.mean(batch_bce)

def loss_mutual_information(code_cat, q_cat):
    cat_loss = -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))
    mi_loss = -cat_loss
    return mi_loss

class MNIST(VectorizedTask):
    """MNIST classification task."""

    def __init__(self,
                 batch_stats_gen: dict = None,
                 batch_stats_disc: dict = None,
                 batch_stats_q: dict = None,
                 batch_size: int = 1024,
                 test: bool = False):

        self.max_steps = 1
        self.obs_shape = tuple([28, 28, 1])
        self.act_shape = tuple([10, ])

        self.batch_size = batch_size
        self.batch_stats_gen = batch_stats_gen 
        self.batch_stats_disc = batch_stats_disc
        self.batch_stats_q = batch_stats_q
        
        self.latent_dim = 64
        self.n_classes = 10
        self.n_con = 2

        self.noise_dim = self.latent_dim - self.n_con
        #self.fake_imgs = None
        #self.cat_codes = None
        # Delayed importing of torchvision

        try:
            from torchvision import datasets
        except ModuleNotFoundError:
            print('You need to install torchvision for this task.')
            print('  pip install torchvision')
            sys.exit(1)

        dataset = datasets.MNIST('./data', train=not test, download=True)

        data = np.expand_dims(dataset.data.numpy() / 255., axis=-1)
        labels = dataset.targets.numpy()

        def reset_fn(key, noise_key, cat_key, con_key):
            if test:
                batch_data, batch_labels = data, labels
            else:
                batch_data, batch_labels = sample_batch(
                    key, data, labels, self.batch_size)
                batch_latent = random.normal(noise_key, (self.batch_size, self.latent_dim))
                #batch_cat = random.randint(cat_key, (self.batch_size,), 0, 10)
                
                #batch_cat_one_hot = jax.nn.one_hot(batch_cat, 10)

                c = jnp.tile(jnp.arange(10),13)
                # remove the last 4 elements to make it 256
                c = c[:self.batch_size]
                batch_cat_one_hot = jax.nn.one_hot(c, 10)
                #batch_con = random.uniform(con_key, (self.batch_size, self.n_con), minval=-1.0, maxval=1.0)

                batch_latent_concat = jnp.concatenate([batch_latent, batch_cat_one_hot], axis=-1)
            
            return State(obs=batch_data, latent=batch_latent_concat, cat_codes=batch_cat_one_hot, labels=batch_labels, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc, batch_stats_q=self.batch_stats_q)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, real_preds, action, q):
            # Compute the loss
            q_cat = nn.log_softmax(q, axis=-1)
            # cross entropy loss for Discrete Codes
            loss_q_disc = loss_mutual_information(state.cat_codes, q_cat)
           
            real_loss = optax.sigmoid_binary_cross_entropy(real_preds, jnp.ones((batch_size,1), dtype=jnp.float32)).mean()
            fake_loss = optax.sigmoid_binary_cross_entropy(action, jnp.zeros((batch_size,1), dtype=jnp.float32)).mean()

            reward_fake = -fake_loss

            reward_real = -real_loss

            reward_mi = loss_q_disc

            return state, reward_real, reward_fake, reward_mi, jnp.ones(())
        
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray, noise_key: jnp.ndarray, cat_key: jnp.ndarray, con_key: jnp.ndarray) -> State:
        return self._reset_fn(key, noise_key, cat_key, con_key)

    def step(self,
             state: TaskState,
             real_preds: jnp.ndarray,
             action: jnp.ndarray,
             q: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, real_preds, action, q)

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
from flax import linen as nn

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    latent: jnp.ndarray
    cat_codes: jnp.ndarray
    labels: jnp.ndarray
    cat_codes: jnp.ndarray
    #fake_imgs: jnp.ndarray
    batch_stats_gen: any = None
    batch_stats_disc: any = None

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
    mi_loss = cat_loss
    return mi_loss

class MNIST(VectorizedTask):
    """MNIST classification task."""

    def __init__(self,
                 batch_stats_gen: dict = None,
                 batch_stats_disc: dict = None,
                 batch_size: int = 1024,
                 test: bool = False):

        self.max_steps = 1
        self.obs_shape = tuple([28, 28, 1])
        self.act_shape = tuple([10, ])

        self.batch_stats_gen = batch_stats_gen 
        self.batch_stats_disc = batch_stats_disc

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

        #if not test:
        #    data_full = dataset.data.numpy()
        #    labels_full = dataset.targets.numpy()
            
            # Calculate indices for 10% of the dataset
        #    indices = np.random.choice(len(data_full), int(len(data_full) * 0.02), replace=False)
            
            # Extract the data and labels for the subset
        #    data_subset = data_full[indices] / 255.  # Normalize the data
        #    data = np.expand_dims(data_subset, axis=-1)  # Add channel dimension
        #    labels = labels_full[indices]

        #else:
        data = np.expand_dims(dataset.data.numpy() / 255., axis=-1)
        labels = dataset.targets.numpy()

        def reset_fn(key, noise_key, cat_key):
            if test:
                batch_data, batch_labels = data, labels
            else:
                batch_data, batch_labels = sample_batch(
                    key, data, labels, batch_size)
                batch_latent = random.normal(noise_key, (batch_size, 64))
                batch_cat = random.randint(cat_key, (batch_size), 0, 10)
                batch_latent_concat = jnp.concatenate([batch_latent, jax.nn.one_hot(batch_cat, 10)], axis=-1)

                batch_cat_one_hot = jax.nn.one_hot(batch_cat, 10)

            return State(obs=batch_data, latent=batch_latent_concat, cat_codes=batch_cat_one_hot, labels=batch_labels, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, real_preds, action, q):
            # Compute the loss
            q_cat = nn.log_softmax(q, axis=-1)
            loss_mi = loss_mutual_information(state.cat_codes, q_cat)
            
            real_loss = bce_logits(real_preds, jnp.ones((), dtype=jnp.int32))
            fake_loss = bce_logits(action, jnp.zeros((), dtype=jnp.int32))
           
            # add weight to loss mi
            #loss_mi = 1.4 * loss_mi
            #jax.debug.print('loss mi disc : {}', loss_mi)
            loss = (real_loss + fake_loss) #/ 2 + loss_mi
            #loss = real_loss + fake_loss

            reward = -loss

            return state, reward, jnp.ones(())
        
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray, noise_key: jnp.ndarray, cat_key: jnp.ndarray) -> State:
        return self._reset_fn(key, noise_key, cat_key)

    def step(self,
             state: TaskState,
             real_preds: jnp.ndarray,
             action: jnp.ndarray,
             q: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, real_preds, action, q)

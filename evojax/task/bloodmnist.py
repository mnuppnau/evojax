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
    labels: jnp.ndarray
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


def loss_mutual_information(code_cat, q_cat):
    cat_loss = -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))
    mi_loss = -cat_loss
    return mi_loss


def _load_bloodmnist_split(test: bool = False, root: str = './data'):
    data = np.load(f'{root}/bloodmnist.npz')
    split_prefix = 'test' if test else 'train'
    images = data[f'{split_prefix}_images'].astype(np.float32)
    labels = data[f'{split_prefix}_labels'].astype(np.int32).reshape(-1)
    # Convert RGB stain image to grayscale to match the baseline pipeline.
    gray = (
        0.2989 * images[..., 0]
        + 0.5870 * images[..., 1]
        + 0.1140 * images[..., 2]
    ) / 255.0
    gray = np.expand_dims(gray.astype(np.float32), axis=-1)
    return gray, labels


class BloodMNIST(VectorizedTask):
    """BloodMNIST classification task (grayscale-converted baseline)."""

    def __init__(self,
                 batch_stats_gen: dict = None,
                 batch_stats_disc: dict = None,
                 batch_stats_q: dict = None,
                 batch_size: int = 1024,
                 test: bool = False):

        self.max_steps = 1
        self.obs_shape = tuple([28, 28, 1])
        self.act_shape = tuple([8, ])

        self.batch_size = batch_size
        self.batch_stats_gen = batch_stats_gen
        self.batch_stats_disc = batch_stats_disc
        self.batch_stats_q = batch_stats_q
        self.dataset_name = 'bloodmnist'

        self.latent_dim = 62
        self.n_classes = 8
        self.n_con = 2

        self.noise_dim = self.latent_dim - self.n_con
        data, labels = _load_bloodmnist_split(test=test)
        self.data = data
        self.labels = labels

        def reset_fn(key, noise_key, cat_key, con_key):
            batch_data, batch_labels = sample_batch(
                key, data, labels, self.batch_size)
            batch_latent = random.normal(
                noise_key, (self.batch_size, self.latent_dim))

            reps = (self.batch_size + self.n_classes - 1) // self.n_classes
            c = jnp.tile(jnp.arange(self.n_classes), reps)
            c = c[:self.batch_size]
            batch_cat_one_hot = jax.nn.one_hot(c, self.n_classes)
            batch_latent_concat = jnp.concatenate(
                [batch_latent, batch_cat_one_hot], axis=-1)

            return State(
                obs=batch_data,
                latent=batch_latent_concat,
                cat_codes=batch_cat_one_hot,
                labels=batch_labels,
                batch_stats_gen=self.batch_stats_gen,
                batch_stats_disc=self.batch_stats_disc,
                batch_stats_q=self.batch_stats_q,
            )

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, real_preds, action, q):
            q_cat = nn.log_softmax(q, axis=-1)
            loss_q_disc = loss_mutual_information(state.cat_codes, q_cat)

            real_loss = optax.sigmoid_binary_cross_entropy(
                real_preds, jnp.ones((batch_size, 1), dtype=jnp.float32)).mean()
            fake_loss = optax.sigmoid_binary_cross_entropy(
                action, jnp.zeros((batch_size, 1), dtype=jnp.float32)).mean()

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

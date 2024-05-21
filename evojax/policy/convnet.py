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
    """CNN updated for CIFAR-10 with 3-channel input."""

    @nn.compact
    def __call__(self, x):
        # Adjust the first Conv layer to accept 3 channels
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        act1 = x
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        act2 = x
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # flatten
        act3 = x
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x, (act1, act2, act3)

class ConvNetPolicy(PolicyNetwork):
    """A convolutional neural network for the CIFAR-10 classification task."""

    def __init__(self, logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger('ConvNetPolicy')
        else:
            self._logger = logger

        model = CNN()
        # Initialize the model with an example CIFAR-10 image shape: [batch_size, height, width, channels]
        params = model.init(random.PRNGKey(0), jnp.zeros([1, 32, 32, 3]))
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info(
            'ConvNetPolicy.num_params = {}'.format(self.num_params))
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        logits, activations = self._forward_fn(params, t_states.obs)
        return logits, p_states, activations 

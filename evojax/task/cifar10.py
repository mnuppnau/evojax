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
import torch

from typing import Tuple

import optax
import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray

def sample_batch(key: jnp.ndarray,
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 batch_size: int) -> Tuple:
    ix = random.choice(
        key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(data, indices=ix, axis=0),
            jnp.take(labels, indices=ix, axis=0))


def loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    return optax.softmax_cross_entropy_with_integer_labels(prediction, target).mean() 


def accuracy(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    return (prediction.argmax(axis=-1) == target).mean()

class CIFAR10(VectorizedTask):
    """CIFAR10 classification task."""

    def __init__(self,
                 batch_size: int = 1024,
                 test: bool = False):

        self.max_steps = 1
        self.obs_shape = tuple([32, 32, 1])
        self.act_shape = tuple([10, ])

        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img
        
        def numpy_collate(batch):
            if isinstance(batch[0], np.ndarray):
                return np.stack(batch)
            elif isinstance(batch[0], (tuple,list)):
                transposed = zip(*batch)
                return [numpy_collate(samples) for samples in transposed]
            else:
                return np.array(batch)


        # Delayed importing of torchvision

        try:
            from torchvision import datasets
            from torchvision import transforms
        except ModuleNotFoundError:
            print('You need to install torchvision for this task.')
            print('  pip install torchvision')
            sys.exit(1)

        train_dataset = datasets.CIFAR10('./data', train=True, download=True)
        DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
        DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))

        if test:
            transform = image_to_numpy
        else:
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      image_to_numpy
                                     ])
        
        dataset = datasets.CIFAR10('./data', train=not test, transform=transform, download=True)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data, labels = next(iter(dataloader))

        data = data.cpu().numpy()

        labels = labels.cpu().numpy()
        
        def reset_fn(key):
            if test:
                batch_data, batch_labels = sample_batch(key, data, labels, batch_size)
            else:
                batch_data, batch_labels = sample_batch(key, data, labels, batch_size)
            
            return State(obs=batch_data, labels=batch_labels)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            if test:
                reward = accuracy(action, state.labels)
            else:
                reward = -loss(action, state.labels)
            return state, reward, jnp.ones(())
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: TaskState,
             action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)

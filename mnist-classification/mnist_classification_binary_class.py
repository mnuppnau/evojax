import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from jax.scipy.special import logsumexp
import flax.linen as nn

import numpy as np
from jax.tree_util import tree_map
from torch.utils import data
from torchvision import datasets
from torchvision.datasets import MNIST

import time

class MNIST(data.Dataset):
    """`MNIST classification"""

    def __init__(self,
                 batch_size=64,
                 test: bool = False):

        self.batch_size = batch_size

        dataset = datasets.MNIST('./data', train=not test, download=True)
        data = np.expand_dims(np.array(dataset.data), axis=-1)
        labels = dataset.targets.numpy()

        def get_all_data():
            return data, labels
       

class CNN(nn.Module):

    # A helper function to randomly initialize weights and biases
    # for a dense neural network model
    def random_layer_params(self, m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    
    # Initialize all layers for a fully-connected neural network with sizes "sizes"
    def init_network_params(self, sizes, key):
        keys = random.split(key, len(sizes))
        return [self.random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    
    def relu(self, x):
        return jnp.maximum(0, x)

    def predict(self, params, image):
        # per-example predictions
        activations = image
        for w,b in params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = self.relu(outputs)
    
        final_w, final_b = params[:-1]
        logits = jnp.dot(final_w, activations) + final_b
        return logits - logsumexp(logits)

    self._predict = jit(vmap(self.predict, in_axes=(None, 0)))



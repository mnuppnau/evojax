import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from evojax.policy.convnet import Generator, ParameterAdapter

model = Generator(training=False, n_codes=8, noise_dim=62)
key = jax.random.PRNGKey(0)
vars = model.init(key, jnp.ones((1, 72)))
params = vars['params']
adapter = ParameterAdapter(params, chunk_size=512)

print("Layer sizes:", adapter.param_sizes)
print("Split indices:", adapter.split_indices)
print("Total chunks:", adapter.total_chunks)

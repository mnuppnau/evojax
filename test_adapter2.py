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

for i, (p_shape, p_size) in enumerate(zip(adapter.param_shapes, adapter.param_sizes)):
    pad = (512 - (p_size % 512)) % 512
    print(f"Layer {i}: size {p_size}, shape {p_shape}, padding {pad}")


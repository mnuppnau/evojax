import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from evojax.policy.convnet import Generator

model = Generator(training=False, n_codes=10, noise_dim=62)
key = jax.random.PRNGKey(0)
vars = model.init(key, jnp.ones((1, 74)))
params = unfreeze(vars['params'])

flat_params, _ = jax.tree_util.tree_flatten(params)
shapes = [p.shape for p in flat_params]
print("Flat shapes:", shapes)
print("Total elements:", sum([p.size for p in flat_params]))

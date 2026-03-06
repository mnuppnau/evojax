import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import numpy as np
from evojax.policy.convnet import Generator

model = Generator(training=False, n_codes=8, noise_dim=62)
key = jax.random.PRNGKey(0)
vars = model.init(key, jnp.ones((1, 72)))
params = unfreeze(vars['params'])
direct_keys = ['film_gamma_0', 'film_beta_0', 'film_gamma_1', 'film_beta_1', 'film_gamma_2', 'film_beta_2', 'film_gamma_3', 'film_beta_3', 'seed_code_dense']
params_direct = {k: v for k, v in params.items() if k in direct_keys}
params_hn = {k: v for k, v in params.items() if k not in direct_keys}

print("Direct keys:", params_direct.keys())
print("HN keys:", params_hn.keys())

flat_direct, _ = jax.tree_util.tree_flatten(params_direct)
print("Direct params count:", sum(x.size for x in flat_direct))

flat_hn, _ = jax.tree_util.tree_flatten(params_hn)
print("HN target params count:", sum(x.size for x in flat_hn))

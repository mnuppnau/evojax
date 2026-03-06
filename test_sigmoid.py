import jax
import jax.numpy as jnp
import optax
x = jnp.array([-10.0, 0.0, 10.0])
print(jax.nn.sigmoid(x))

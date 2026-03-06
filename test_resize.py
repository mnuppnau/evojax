import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
x = jnp.arange(7, dtype=jnp.float32)
y = jax.image.resize(x, (14,), method='linear')
print("7 to 14 linear:")
for i in range(14):
    print(f"{i}: {y[i]:.2f}")

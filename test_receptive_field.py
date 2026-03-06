import jax
import jax.numpy as jnp
from flax import linen as nn

class Disc(nn.Module):
    @nn.compact
    def __call__(self, x):
        h1 = nn.Conv(1, (4, 4), strides=(2, 2), padding="VALID")(x)
        h2 = nn.Conv(1, (4, 4), strides=(2, 2), padding="VALID")(h1)
        h3 = nn.Conv(1, (4, 4), strides=(2, 2), padding="VALID")(h2)
        return h3

x = jnp.zeros((1, 28, 28, 1))
x = x.at[0, 24:, :, 0].set(1.0) # Bottom 4 rows
x = x.at[0, :, 24:, 0].set(1.0) # Right 4 columns

model = Disc()
vars = model.init(jax.random.PRNGKey(0), x)

# evaluate Jacobian to see which pixels affect the output
def apply_fn(x_in):
    return jnp.sum(model.apply(vars, x_in))

grad = jax.grad(apply_fn)(x)
print("Gradient at bottom-right 4x4:")
print(grad[0, 24:, 24:, 0])

print("Max gradient in bottom 4 rows:", jnp.max(jnp.abs(grad[0, 24:, :, 0])))
print("Max gradient in right 4 cols:", jnp.max(jnp.abs(grad[0, :, 24:, 0])))
print("Max gradient overall:", jnp.max(jnp.abs(grad)))

import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from flax import linen as nn

class Discriminator(nn.Module):
    features: int = 64
    q_cat: int = 10
    q_cont: int = 2

    @nn.compact
    def __call__(self, x):
        def SN(layer):
            return layer # nn.SpectralNorm(layer) - simplifying for grad test

        train = False
        # 28x28 -> 14x14
        h = SN(nn.Conv(self.features, kernel_size=(4, 4), strides=(2, 2), padding="SAME"))(x)
        h = nn.leaky_relu(h, 0.2)

        # 14x14 -> 7x7
        h = SN(nn.Conv(self.features * 2, kernel_size=(4, 4), strides=(2, 2), padding="SAME"))(h)
        h = nn.leaky_relu(h, 0.2)

        # 7x7 -> 1x1
        q = SN(nn.Conv(self.features * 2, kernel_size=(7, 7), strides=(1, 1), padding="VALID"))(h)
        q = nn.leaky_relu(q, 0.2)

        q_flat = q.reshape((q.shape[0], -1))
        q_cat_logits = SN(nn.Dense(self.q_cat))(q_flat)

        return q_cat_logits

model = Discriminator()
x = jnp.ones((1, 28, 28, 1))
key = jax.random.PRNGKey(0)
vars = model.init(key, x)

def apply_q(x_in):
    return jnp.sum(model.apply(vars, x_in))

grad = jax.grad(apply_q)(x)
print("Max grad overall:", jnp.max(jnp.abs(grad)))
print("Max grad top-left 14x14:", jnp.max(jnp.abs(grad[0, :14, :14, 0])))
print("Max grad bottom-right 4x4:", jnp.max(jnp.abs(grad[0, -4:, -4:, 0])))
print("Max grad right 4 cols:", jnp.max(jnp.abs(grad[0, :, -4:, 0])))
print("Max grad bottom 4 rows:", jnp.max(jnp.abs(grad[0, -4:, :, 0])))

import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
grad_ds = np.abs(np.array(grad[0, :, :, 0])).reshape(7, 4, 7, 4).max(axis=(1, 3))
print("Gradient magnitude heatmap (7x7 pooled):")
print(grad_ds)

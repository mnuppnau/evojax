"""Population evaluation with common data and common random numbers."""

from functools import partial

import jax


@partial(jax.jit, static_argnames=("score_fn",))
def evaluate_population(score_fn, population, batch, key):
    """Map score_fn(params, batch, key) over population rows.

    The SAME batch and key reach every individual. Split or fold the key in
    the outer training loop to obtain fresh randomness each generation.
    The score function must be pure and JAX-transformable; its result can
    be a scalar or a pytree of raw metrics. This helper uses one device.
    """
    if population.ndim != 2 or min(population.shape) < 1:
        raise ValueError("population must be a nonempty parameter matrix")
    return jax.vmap(score_fn, in_axes=(0, None, None))(population, batch, key)

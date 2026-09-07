"""Population fitness shaping with explicit objective directions."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def rank_normalize(values):
    """Map ascending average ranks to [-1, 1], preserving ties.

    Constant finite inputs (including a singleton) receive zero. Nonfinite
    entries receive -1 and do not affect finite ranks. Training entry points
    reject nonfinite measurements; this kernel also has defined JIT behavior.
    """
    values = jnp.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("rank_normalize expects a nonempty vector")
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise ValueError("fitness components must be real-valued")
    finite = jnp.isfinite(values)
    # Integer inputs have no nonfinite values. Do not promote them to float
    # just to insert an infinity sentinel: adjacent large integers can merge.
    safe = (jnp.where(finite, values, jnp.inf)
            if jnp.issubdtype(values.dtype, jnp.floating) else values)
    ordered = jnp.sort(safe)
    lower = jnp.searchsorted(ordered, safe, side="left")
    upper = jnp.searchsorted(ordered, safe, side="right")
    ranks = (lower + upper - 1).astype(jnp.float32) / 2
    count = jnp.sum(finite)
    normalized = 2 * ranks / jnp.maximum(count - 1, 1) - 1
    normalized = jnp.where(count > 1, normalized, 0.0)
    return jnp.where(finite, normalized, -1.0)


def combine_fitness(components: Mapping, weights: Mapping, *, maximize=None):
    """Combine independently ranked components, without a second ranking.

    Weights are finite and nonnegative. Direction defaults to maximization;
    pass e.g. maximize={"loss": False, "q": True} for mixed objectives.
    Zero-weight diagnostics are excluded, including from finiteness checks.
    This host boundary deliberately fails on invalid active measurements.
    """
    if not weights or set(weights) - set(components):
        raise ValueError("weights must name existing fitness components")
    maximize = {} if maximize is None else dict(maximize)
    if set(maximize) - set(weights):
        raise ValueError("objective directions must name weighted components")
    if any(not isinstance(v, (bool, np.bool_)) for v in maximize.values()):
        raise ValueError("objective directions must be booleans")
    result = None
    shape = None
    for name, weight in weights.items():
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid weight for {name}")
        if weight == 0:
            continue
        values = np.asarray(components[name])
        if (not np.isrealobj(values) or values.ndim != 1
                or not values.size or not np.isfinite(values).all()):
            raise ValueError(f"{name} must be a finite, nonempty vector")
        if shape is not None and values.shape != shape:
            raise ValueError("fitness components must have equal shapes")
        shape = values.shape
        ranked = rank_normalize(values)
        # Reverse ranks, not raw unsigned integers (whose negation wraps).
        directed = ranked if maximize.get(name, True) else -ranked
        weighted = weight * directed
        result = weighted if result is None else result + weighted
    if result is None:
        raise ValueError("at least one fitness weight must be positive")
    if not np.isfinite(np.asarray(result)).all():
        raise ValueError("fitness weights overflowed the combined score")
    return result

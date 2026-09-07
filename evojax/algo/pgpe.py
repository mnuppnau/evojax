# Copyright 2022 The EvoJAX Authors. Licensed under Apache-2.0.
"""Antithetic PGPE with optimizer-owned exploration and resumable state.

The scaled score-function directions retain the EvoJAX PGPE formulation.
Cultural control belongs in scalar fitness, never in this update rule.
"""

from functools import partial

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .base import NEAlgorithm
from evojax.fitness import rank_normalize


@partial(jax.jit, static_argnums=(3, 4))
def ask_func(key, stdev, center, num_directions, solution_size):
    """Return [mu+eps_0, mu-eps_0, mu+eps_1, mu-eps_1, ...]."""
    next_key, sample_key = jax.random.split(key)
    noises = jax.random.normal(sample_key, (num_directions, solution_size)) * stdev
    population = jnp.stack((center + noises, center - noises), axis=1)
    return next_key, noises, population.reshape(2 * num_directions, solution_size)


@jax.jit
def compute_reinforce_update(fitness_scores, scaled_noises, stdev):
    """Compute variance-scaled PGPE directions from interleaved pairs."""
    pairs = fitness_scores.reshape((-1, 2))
    difference = pairs[:, 0] - pairs[:, 1]
    pair_mean = pairs.mean(axis=1)
    center_direction = jnp.mean(0.5 * difference[:, None] * scaled_noises, axis=0)
    stdev_direction = jnp.mean(
        (pair_mean - pairs.mean())[:, None]
        * (scaled_noises ** 2 - stdev ** 2) / stdev,
        axis=0,
    )
    return center_direction, stdev_direction


@jax.jit
def update_stdev(stdev, lr, grad, max_change, minimum, maximum):
    allowed = stdev * max_change
    delta = jnp.clip(lr * grad, -allowed, allowed)
    return jnp.clip(stdev + delta, minimum, maximum)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _vector(value, size, name):
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be real-valued")
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return jnp.asarray(value)


class PGPE(NEAlgorithm):
    """Maximizing PGPE optimizer, using Adam or SGD for its center.

    Set solution_ranking=False for an already rank-shaped composite fitness.
    Sigma limits are fixed optimizer configuration, with no CA override.
    Calls to ask/tell alternate; checkpoint only after tell.
    """

    def __init__(
        self, pop_size, param_size, init_params=None, *, optimizer="adam",
        optimizer_config=None, center_learning_rate=0.0048,
        stdev_learning_rate=0.062, init_stdev=0.032, stdev_max_change=0.1,
        stdev_min=0.005, stdev_max=10.0, solution_ranking=True, seed=0,
    ):
        self.pop_size = _positive_int(pop_size, "pop_size")
        self.param_size = _positive_int(param_size, "param_size")
        if self.pop_size % 2:
            raise ValueError("pop_size must be even for antithetic sampling")
        optimizer = "sgd" if optimizer is None else optimizer
        if optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        options = dict(optimizer_config or {})
        defaults = dict(beta1=0.9, beta2=0.999, epsilon=1e-8,
                        center_lr_decay_coef=1.0, center_lr_decay_steps=100000)
        if options.keys() - defaults.keys():
            raise ValueError(f"unknown optimizer options: {options.keys() - defaults.keys()}")
        defaults.update(options)
        interval = _positive_int(defaults["center_lr_decay_steps"], "center_lr_decay_steps")
        defaults["center_lr_decay_steps"] = interval
        for name in ("beta1", "beta2", "epsilon", "center_lr_decay_coef"):
            defaults[name] = float(defaults[name])
            if not np.isfinite(defaults[name]):
                raise ValueError(f"{name} must be finite")
        if not (0 <= defaults["beta1"] < 1 and 0 <= defaults["beta2"] < 1):
            raise ValueError("Adam beta values must be in [0, 1)")
        if defaults["epsilon"] <= 0 or not 0 < defaults["center_lr_decay_coef"] <= 1:
            raise ValueError("epsilon must be positive and decay coefficient in (0, 1]")
        scalars = dict(center_learning_rate=center_learning_rate,
                       stdev_learning_rate=stdev_learning_rate,
                       stdev_max_change=stdev_max_change,
                       stdev_min=stdev_min, stdev_max=stdev_max)
        scalars = {k: float(v) for k, v in scalars.items()}
        if not all(np.isfinite(v) for v in scalars.values()):
            raise ValueError("optimizer hyperparameters must be finite")
        if scalars["center_learning_rate"] < 0 or scalars["stdev_learning_rate"] < 0:
            raise ValueError("learning rates must be nonnegative")
        if not 0 <= scalars["stdev_max_change"] <= 1:
            raise ValueError("stdev_max_change must be in [0, 1]")
        if not 0 < scalars["stdev_min"] <= scalars["stdev_max"]:
            raise ValueError("sigma bounds must satisfy 0 < min <= max")
        if (not all(np.isfinite(np.float32(v)) for v in scalars.values())
                or np.float32(scalars["stdev_min"]) <= 0):
            raise ValueError("optimizer bounds/rates must be representable in float32")
        if (np.float32(defaults["beta1"]) >= 1 or np.float32(defaults["beta2"]) >= 1
                or not 0 < np.float32(defaults["epsilon"]) < np.inf):
            raise ValueError("Adam hyperparameters must be valid in float32")
        if not isinstance(solution_ranking, (bool, np.bool_)):
            raise ValueError("solution_ranking must be a boolean")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self._config = dict(pop_size=self.pop_size, param_size=self.param_size,
                            optimizer=optimizer, solution_ranking=bool(solution_ranking),
                            optimizer_config=defaults, **scalars)
        self._center = _vector(
            np.zeros(self.param_size) if init_params is None else init_params,
            self.param_size, "init_params",
        )
        if not np.isrealobj(init_stdev):
            raise ValueError("init_stdev must be real-valued")
        sigma = np.asarray(init_stdev, dtype=np.float32)
        if sigma.ndim == 0:
            sigma = np.full(self.param_size, sigma, dtype=np.float32)
        self._stdev = _vector(sigma, self.param_size, "init_stdev")
        self._validate_stdev(self._stdev)
        rate = scalars["center_learning_rate"]
        decay = defaults["center_lr_decay_coef"]
        # Only the learning-rate schedule uses interval division. Optax's
        # Adam counter advances on EVERY update for correct bias correction.
        schedule = lambda count: rate * decay ** (count // interval)
        self._optimizer = (
            optax.adam(schedule, b1=defaults["beta1"], b2=defaults["beta2"], eps=defaults["epsilon"])
            if optimizer == "adam" else optax.sgd(schedule)
        )
        self._opt_state = self._optimizer.init(self._center)
        self._update_center = jax.jit(self._optimizer.update)
        self._key = jax.random.PRNGKey(int(seed))
        self._t = 0
        self._solutions = None
        self._scaled_noises = None

    def _validate_stdev(self, value):
        value = np.asarray(value)
        lo = np.float32(self._config["stdev_min"])
        hi = np.float32(self._config["stdev_max"])
        if np.any(value < lo) or np.any(value > hi):
            raise ValueError("standard deviations must be within configured bounds")

    def ask(self):
        if self._solutions is not None:
            raise RuntimeError("tell must consume the pending population before another ask")
        key, noises, population = ask_func(
            self._key, self._stdev, self._center, self.pop_size // 2, self.param_size,
        )
        if not np.isfinite(np.asarray(population)).all():
            raise FloatingPointError("sampling overflowed; optimizer state was not changed")
        self._key, self._scaled_noises, self._solutions = key, noises, population
        return population

    def tell(self, fitness):
        if self._solutions is None:
            raise RuntimeError("ask must precede tell")
        scores = _vector(fitness, self.pop_size, "fitness")
        if self._config["solution_ranking"]:
            scores = 0.5 * rank_normalize(scores)
        center_grad, sigma_grad = compute_reinforce_update(scores, self._scaled_noises, self._stdev)
        updates, opt_state = self._update_center(-center_grad, self._opt_state, self._center)
        center = optax.apply_updates(self._center, updates)
        sigma = update_stdev(
            self._stdev, self._config["stdev_learning_rate"], sigma_grad,
            self._config["stdev_max_change"], self._config["stdev_min"], self._config["stdev_max"],
        )
        leaves = jax.tree_util.tree_leaves((center, sigma, opt_state))
        if not all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves):
            raise FloatingPointError("nonfinite PGPE update; pending population is unchanged")
        self._center, self._stdev, self._opt_state = center, sigma, opt_state
        self._t += 1
        self._solutions = self._scaled_noises = None

    @property
    def center(self):
        return self._center

    @property
    def stdev(self):
        return self._stdev

    @property
    def iteration(self):
        return self._t

    @property
    def best_params(self):
        """Compatibility name for the distribution center, not a best sample."""
        return self.center

    @best_params.setter
    def best_params(self, params):
        """Start a fresh center-optimizer run, retaining sigma and RNG state."""
        if self._solutions is not None:
            raise RuntimeError("cannot replace the center with a population pending")
        center = _vector(params, self.param_size, "params")
        self._center = center
        self._opt_state = self._optimizer.init(center)
        self._t = 0

    def save_state(self):
        if self._solutions is not None:
            raise RuntimeError("checkpoint only at a generation boundary, after tell")
        state = dict(version=1, config=self._config, center=self._center,
                     stdev=self._stdev, iteration=self._t, key=self._key,
                     opt_state=serialization.to_state_dict(self._opt_state))
        return jax.tree_util.tree_map(
            lambda x: np.array(x, copy=True) if isinstance(x, jax.Array) else x, state,
        )

    def load_state(self, state):
        if self._solutions is not None:
            raise RuntimeError("cannot restore with a population pending")
        expected = {"version", "config", "center", "stdev", "iteration", "key", "opt_state"}
        if set(state) != expected or state["version"] != 1 or state["config"] != self._config:
            raise ValueError("incompatible PGPE checkpoint schema or configuration")
        center = _vector(state["center"], self.param_size, "checkpoint center")
        sigma = _vector(state["stdev"], self.param_size, "checkpoint sigma")
        self._validate_stdev(sigma)
        iteration = state["iteration"]
        if isinstance(iteration, bool) or not isinstance(iteration, (int, np.integer)) or iteration < 0:
            raise ValueError("invalid checkpoint iteration")
        key = np.asarray(state["key"])
        if key.shape != (2,) or key.dtype != np.dtype("uint32"):
            raise ValueError("invalid checkpoint random key")
        reference = self._optimizer.init(center)
        opt_state = serialization.from_state_dict(reference, state["opt_state"])
        actual_leaves, actual_tree = jax.tree_util.tree_flatten(opt_state)
        reference_leaves, reference_tree = jax.tree_util.tree_flatten(reference)
        if actual_tree != reference_tree:
            raise ValueError("incompatible optimizer state structure")
        for actual, target in zip(actual_leaves, reference_leaves):
            actual = np.asarray(actual)
            if actual.shape != target.shape or actual.dtype != target.dtype or not np.isfinite(actual).all():
                raise ValueError("invalid optimizer state array")
            if np.issubdtype(actual.dtype, np.integer) and np.any(actual != iteration):
                raise ValueError("optimizer counter does not match checkpoint iteration")
        opt_state = jax.tree_util.tree_map(jnp.asarray, opt_state)
        # Commit only after every field has been validated.
        self._center, self._stdev, self._opt_state = center, sigma, opt_state
        self._key, self._t = jnp.asarray(key), int(iteration)

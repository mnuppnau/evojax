# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight PGPE solver for per-layer HyperNetwork evolution.

This is a stripped-down version of PGPE_CA that handles only the core
search distribution (center, stdev, optimizer, ask/tell) without any
Cultural Algorithm integration. The shared CA stays at the Trainer level
and passes pre-computed fitness weights to each solver's tell().

One instance per Generator layer group. Typically 6 instances total.
"""

import logging
from typing import Optional, Union, Tuple
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

try:
    from jax.example_libraries import optimizers
except ModuleNotFoundError:
    from jax.experimental import optimizers

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


# --- JIT-compiled helper functions ---

@partial(jax.jit, static_argnums=(3, 4))
def ask_func(
    key: jnp.ndarray,
    stdev: jnp.ndarray,
    center: jnp.ndarray,
    num_directions: int,
    solution_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample a population of parameters from Gaussian (mirrored)."""
    next_key, key = random.split(key)
    scaled_noises = random.normal(key, [num_directions, solution_size]) * stdev
    solutions = jnp.hstack(
        [center + scaled_noises, center - scaled_noises]
    ).reshape(-1, solution_size)
    return next_key, scaled_noises, solutions


@jax.jit
def compute_reinforce_update(
    fitness_scores: jnp.ndarray,
    scaled_noises: jnp.ndarray,
    stdev: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute REINFORCE updates for center and stdev."""
    fitness_scores = fitness_scores.reshape((-1, 2))
    baseline = jnp.mean(fitness_scores)
    all_scores = (fitness_scores[:, 0] - fitness_scores[:, 1]).squeeze()
    all_avg_scores = fitness_scores.sum(axis=-1) / 2
    stdev_sq = stdev ** 2.0
    total_mu = scaled_noises * jnp.expand_dims(all_scores, axis=1) * 0.5
    total_sigma = (
        (jnp.expand_dims(all_avg_scores, axis=1) - baseline)
        * (scaled_noises ** 2 - jnp.expand_dims(stdev_sq, axis=0))
        / stdev
    )
    return total_mu.mean(axis=0), total_sigma.mean(axis=0)


@jax.jit
def update_stdev(
    stdev: jnp.ndarray,
    lr: float,
    grad: jnp.ndarray,
    max_change: float,
) -> jnp.ndarray:
    """Update (and clip) the standard deviation."""
    allowed_delta = jnp.abs(stdev) * max_change
    min_allowed = stdev - allowed_delta
    max_allowed = stdev + allowed_delta
    new_stdev = jnp.clip(stdev + lr * grad, min_allowed, max_allowed)
    return jnp.clip(new_stdev, 1e-8, 1e1)


@jax.jit
def rank_normalize(x: jnp.ndarray) -> jnp.ndarray:
    """Rank-normalize fitness scores to [-1, 1]."""
    n = x.shape[0]
    ranks = jnp.argsort(jnp.argsort(x)).astype(jnp.float32)
    return 2.0 * ranks / jnp.maximum(n - 1, 1) - 1.0


# --- ClipUp optimizer (alternative to Adam) ---

@optimizers.optimizer
def clipup(
    step_size: float,
    momentum: float = 0.9,
    max_speed: float = 0.15,
    fix_gradient_size: bool = True,
):
    """ClipUp optimizer: momentum SGD with velocity clipping."""
    step_size = optimizers.make_schedule(step_size)

    def init(x0):
        v0 = jnp.zeros_like(x0)
        return x0, v0

    def update(i, g, state):
        x, v = state
        g = jax.lax.cond(
            fix_gradient_size,
            lambda p: p / jnp.sqrt(jnp.sum(p * p)),
            lambda p: p,
            g,
        )
        step = g * step_size(i)
        v = momentum * v + step
        length = jnp.sqrt(jnp.sum(v * v))
        v = jax.lax.cond(
            length > max_speed,
            lambda p: p * max_speed / length,
            lambda p: p,
            v,
        )
        return x - v, v

    def get_params(state):
        x, _ = state
        return x

    return init, update, get_params


class PGPE_Layer(NEAlgorithm):
    """Lightweight PGPE solver for per-layer HyperNetwork evolution.

    Core PGPE mechanics only — no Cultural Algorithm integration.
    Receives pre-weighted fitness from the Trainer's shared CA.

    Args:
        pop_size: Population size (will be rounded up to even).
        param_size: Number of parameters to evolve.
        init_params: Optional initial center vector.
        optimizer: "adam" (default), "clipup", or "sgd".
        optimizer_config: Dict of optimizer-specific config.
        center_learning_rate: Learning rate for center updates.
        stdev_learning_rate: Learning rate for stdev updates.
        init_stdev: Initial standard deviation.
        stdev_max_change: Max relative change in stdev per step.
        stdev_min: Floor for stdev to prevent exploration collapse.
        seed: Random seed.
        logger: Optional logger.
    """

    def __init__(
        self,
        pop_size: int,
        param_size: int,
        init_params: Optional[Union[jnp.ndarray, np.ndarray]] = None,
        optimizer: str = "adam",
        optimizer_config: Optional[dict] = None,
        center_learning_rate: float = 0.15,
        stdev_learning_rate: float = 0.1,
        init_stdev: Union[float, jnp.ndarray, np.ndarray] = 0.1,
        stdev_max_change: float = 0.1,
        stdev_min: float = 0.005,
        seed: int = 0,
        logger: logging.Logger = None,
    ):
        if logger is None:
            self._logger = create_logger(name='PGPE_Layer')
        else:
            self._logger = logger

        self.pop_size = abs(pop_size)
        if self.pop_size % 2 == 1:
            self.pop_size += 1
        self._num_directions = self.pop_size // 2

        # Center and stdev initialization
        if init_params is None:
            self._center = jnp.zeros(abs(param_size))
        else:
            self._center = jnp.array(init_params)

        if isinstance(init_stdev, (int, float)):
            self._stdev = jnp.ones(abs(param_size)) * abs(init_stdev)
        else:
            self._stdev = jnp.array(init_stdev)

        self._center_lr = abs(center_learning_rate)
        self._stdev_lr = abs(stdev_learning_rate)
        self._stdev_max_change = abs(stdev_max_change)
        self._stdev_min = abs(stdev_min)
        self._key = random.PRNGKey(seed)

        # Optimizer setup
        if optimizer_config is None:
            optimizer_config = {}

        if optimizer == "adam":
            opt_init, opt_update, get_params = optimizers.adam(
                step_size=self._center_lr,
                b1=optimizer_config.get("beta1", 0.9),
                b2=optimizer_config.get("beta2", 0.999),
                eps=optimizer_config.get("epsilon", 1e-8),
            )
        elif optimizer == "clipup":
            opt_init, opt_update, get_params = clipup(
                step_size=self._center_lr,
                momentum=optimizer_config.get("momentum", 0.9),
                max_speed=optimizer_config.get("max_speed", 0.08),
                fix_gradient_size=optimizer_config.get("fix_gradient_size", True),
            )
        else:
            opt_init, opt_update, get_params = optimizers.sgd(
                step_size=self._center_lr,
            )

        self._t = 0
        self._opt_state = jax.jit(opt_init)(self._center)
        self._opt_update = jax.jit(opt_update)
        self._get_params = jax.jit(get_params)

        # Internal state for tell()
        self._scaled_noises = None

        self._logger.info(
            f'PGPE_Layer: pop_size={self.pop_size}, param_size={abs(param_size)}, '
            f'optimizer={optimizer}, center_lr={self._center_lr}, '
            f'stdev_lr={self._stdev_lr}, init_stdev={float(self._stdev[0]):.4f}'
        )

    def ask(self) -> jnp.ndarray:
        """Sample population from Gaussian search distribution.

        Returns:
            (pop_size, param_size) array of candidate solutions.
        """
        self._key, self._scaled_noises, solutions = ask_func(
            self._key,
            self._stdev,
            self._center,
            self._num_directions,
            self._center.size,
        )
        return solutions

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        """Update search distribution based on fitness scores.

        Expects pre-combined fitness where individual components have
        already been rank-normalized and weighted by the Trainer's
        shared CA. No additional normalization is applied here.

        Args:
            fitness: (pop_size,) array of combined fitness scores.
        """
        fitness = jnp.array(fitness)

        grad_center, grad_stdev = compute_reinforce_update(
            fitness_scores=fitness,
            scaled_noises=self._scaled_noises,
            stdev=self._stdev,
        )

        # Update center via optimizer
        self._opt_state = self._opt_update(
            self._t, -grad_center, self._opt_state
        )
        self._t += 1
        self._center = self._get_params(self._opt_state)

        # Update stdev with clipped gradient
        self._stdev = update_stdev(
            stdev=self._stdev,
            lr=self._stdev_lr,
            max_change=self._stdev_max_change,
            grad=grad_stdev,
        )
        self._stdev = jnp.maximum(self._stdev, self._stdev_min)

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.array(self._center, copy=True)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self._center = jnp.array(params, copy=True)

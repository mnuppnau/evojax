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

"""Implementation of the PGPE algorithm in JAX.

Ref: https://github.com/nnaisense/pgpelib/blob/release/pgpelib/pgpe.py
"""

import numpy as np
import logging
from typing import Optional
from typing import Union
from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax import lax

try:
    from jax.example_libraries import optimizers
except ModuleNotFoundError:
    from jax.experimental import optimizers

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


@jax.jit
def non_dominated_sort_lax(objectives: jnp.ndarray) -> jnp.ndarray:
    """
    Perform non-dominated sorting on a set of points in multi-objective space,
    using jax.lax.while_loop for the iterative rank assignment.
    
    Args:
        objectives (jnp.ndarray): Array of shape (N, M), where
            N = number of points,
            M = number of objectives (assume minimization).
            
    Returns:
        jnp.ndarray of shape (N,):
            The integer Pareto rank of each point (0 = best/front, 1 = next front, etc.).
            Points that cannot be assigned (e.g., if a front is not found) remain at -1.
    """
    # --- Step A: Build the "dominates" matrix ---
    # dominates[i, j] = True if point i dominates point j (all dims <=, at least one dim <)
    less_equal = objectives[:, None, :] <= objectives[None, :, :]  # (N, N, M)
    strictly_less = objectives[:, None, :] < objectives[None, :, :]  # (N, N, M)
    all_le = jnp.all(less_equal, axis=-1)    # (N, N)
    any_lt = jnp.any(strictly_less, axis=-1) # (N, N)
    dominates = jnp.logical_and(all_le, any_lt)  # (N, N)
    
    # --- Step B: Iteratively identify Pareto layers using lax.while_loop ---
    N = objectives.shape[0]
    init_ranks = -1 * jnp.ones((N,), dtype=jnp.int32)  # -1 => unassigned
    init_rank_idx = jnp.int32(0)
    
    # A boolean mask of which points are still unranked:
    init_unranked = (init_ranks == -1)  # True/False array
    init_done = False  # Will indicate if we should stop

    # Pack into a "carry" tuple to pass between iterations
    carry_init = (init_ranks, init_rank_idx, init_unranked, init_done)

    def cond_fun(carry):
        """Return True if we should continue; False if done."""
        ranks, current_rank, unranked, done = carry
        return jnp.logical_not(done)

    def body_fun(carry):
        """One iteration of finding the next front and assigning ranks."""
        ranks, current_rank, unranked, done = carry
        
        # Check if there are still unranked points
        still_unranked = jnp.any(unranked)  # bool
        
        # For each j, check if it is dominated by any unranked i:
        # dominators[i, j] = (dominates[i, j] & unranked[i])
        dominators = jnp.logical_and(dominates, unranked[:, None])
        dominated_by_unranked = jnp.any(dominators, axis=0)
        
        # The next front = unranked points NOT dominated by any unranked
        front_mask = jnp.logical_and(unranked, jnp.logical_not(dominated_by_unranked))
        
        # If front_mask is empty, we can't assign a next layer.
        # So we set a "done" condition to break out of the loop.
        no_front = jnp.logical_not(jnp.any(front_mask))
        
        # We stop if EITHER we have no unranked points left OR no new front is found
        done_cond = jnp.logical_or(jnp.logical_not(still_unranked), no_front)
        
        # Tentative updates if we are NOT done:
        new_ranks = jnp.where(front_mask, current_rank, ranks)
        new_unranked = jnp.logical_and(unranked, jnp.logical_not(front_mask))
        new_current_rank = current_rank + 1
        new_done = jnp.logical_or(done, done_cond)  # once done => always done
        
        # If done_cond is True, keep the old values (no update):
        new_ranks = jnp.where(done_cond, ranks, new_ranks)
        new_unranked = jnp.where(done_cond, unranked, new_unranked)
        new_current_rank = jnp.where(done_cond, current_rank, new_current_rank)
        
        return (new_ranks, new_current_rank, new_unranked, new_done)

    # Run the while_loop
    final_ranks, _, _, _ = lax.while_loop(cond_fun, body_fun, carry_init)
    return final_ranks

@partial(jax.jit, static_argnums=(1,))
def process_scores(
    x: Union[np.ndarray, jnp.ndarray], use_ranking: bool
) -> jnp.ndarray:
    """Convert fitness scores to rank if necessary."""

    x = jnp.array(x)
    if use_ranking:
        ranks = jnp.zeros(x.size, dtype=int)
        ranks = ranks.at[x.argsort()].set(jnp.arange(x.size)).reshape(x.shape)
        return ranks / ranks.max() - 0.5
    else:
        return x


@jax.jit
def compute_reinforce_update(
    fitness_scores: jnp.ndarray, scaled_noises: jnp.ndarray, stdev: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the updates for the center and the standard deviation."""

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
    stdev: jnp.ndarray, lr: float, grad: jnp.ndarray, max_change: float
) -> jnp.ndarray:
    """Update (and clip) the standard deviation."""

    allowed_delta = jnp.abs(stdev) * max_change
    min_allowed = stdev - allowed_delta
    max_allowed = stdev + allowed_delta
    return jnp.clip(stdev + lr * grad, min_allowed, max_allowed)


@partial(jax.jit, static_argnums=(3, 4))
def ask_func(
    key: jnp.ndarray,
    stdev: jnp.ndarray,
    center: jnp.ndarray,
    num_directions: int,
    solution_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A function that samples a population of parameters from Gaussian."""

    next_key, key = random.split(key)
    scaled_noises = random.normal(key, [num_directions, solution_size]) * stdev
    solutions = jnp.hstack(
        [center + scaled_noises, center - scaled_noises]
    ).reshape(-1, solution_size)
    return next_key, scaled_noises, solutions


class PGPE_DISC(NEAlgorithm):
    """Policy Gradient with Parameter-based Exploration (PGPE) algorithm.

    Ref: https://people.idsia.ch/~juergen/icann2008sehnke.pdf
    """

    def __init__(
        self,
        pop_size: int,
        param_size: int,
        init_params: Optional[Union[jnp.ndarray, np.ndarray]] = None,
        optimizer: Optional[str] = None,
        optimizer_config: Optional[dict] = None,
        center_learning_rate: float = 0.15,
        stdev_learning_rate: float = 0.1,
        init_stdev: Union[float, jnp.ndarray, np.ndarray] = 0.1,
        stdev_max_change: float = 0.2,
        solution_ranking: bool = True,
        seed: int = 0,
        logger: logging.Logger = None,
    ):
        """Initialization function.

        Args:
            pop_size - Population size.
            param_size - Parameter size.
            init_params - Initial parameters, all zeros if not given.
            optimizer - Possible values are {None, 'adam', 'clipup'}.
            optimizer_config - Configurations specific to the optimizer.
                               For None: No configuration is required.
                               For Adam: {'epsilon', 'beta1', 'beta2'}.
                               For ClipUp: {'momentum', 'max_speed'}.
            center_learning_rate - Learning rate for the Gaussian mean.
            stdev_learning_rate - Learning rate for the Gaussian stdev.
            init_stdev - Initial stdev for the Gaussian distribution.
            stdev_max_change - Maximum allowed change for stdev in abs values.
            solution_ranking - Should we treat the fitness as rankings or not.
            seed - Random seed for parameters sampling.
        """

        if logger is None:
            self._logger = create_logger(name="PGPE")
        else:
            self._logger = logger

        self.pop_size = abs(pop_size)
        if self.pop_size % 2 == 1:
            self.pop_size += 1
            self._logger.info(
                "Population size should be an even number, set to {}".format(
                    self.pop_size
                )
            )
        self._num_directions = self.pop_size // 2

        jax.debug.print('init params in PGPE before array {} : ', init_params)
        if init_params is None:
            self._center = np.zeros(abs(param_size))
        else:
            self._center = init_params
        self._center = jnp.array(self._center)
        print('init params in PGPE after array {} : ', self._center)
        if isinstance(init_stdev, float):
            self._stdev = np.ones(abs(param_size)) * abs(init_stdev)
        self._stdev = jnp.array(self._stdev)

        self._center_lr = abs(center_learning_rate)
        self._stdev_lr = abs(stdev_learning_rate)
        self._stdev_max_change = abs(stdev_max_change)
        self._solution_ranking = solution_ranking

        if optimizer_config is None:
            optimizer_config = {}
        decay_coef = optimizer_config.get("center_lr_decay_coef", 1.0)
        self._lr_decay_steps = optimizer_config.get(
            "center_lr_decay_steps", 1000
        )

        if optimizer == "adam":
            opt_init, opt_update, get_params = optimizers.adam(
                step_size=lambda x: self._center_lr * jnp.power(decay_coef, x),
                b1=optimizer_config.get("beta1", 0.9),
                b2=optimizer_config.get("beta2", 0.999),
                eps=optimizer_config.get("epsilon", 1e-8),
            )
        elif optimizer == "clipup":
            opt_init, opt_update, get_params = clipup(
                step_size=lambda x: self._center_lr * jnp.power(decay_coef, x),
                momentum=optimizer_config.get("momentum", 0.99),
                max_speed=optimizer_config.get("max_speed", 0.55),
                fix_gradient_size=optimizer_config.get(
                    "fix_gradient_size", True
                ),
            )
        else:
            opt_init, opt_update, get_params = optimizers.sgd(
                step_size=lambda x: self._center_lr * jnp.power(decay_coef, x),
            )
        self._t = 0
        self._opt_state = jax.jit(opt_init)(self._center)
        self._opt_update = jax.jit(opt_update)
        self._get_params = jax.jit(get_params)

        self._best_real_window = jnp.ones(60)
        self._best_fake_window = jnp.ones(60)

        self._best_prev_fitness_real = jnp.array([0.0])
        self._best_prev_fitness_fake = jnp.array([0.0])

        self._key = random.PRNGKey(seed=seed)
        self._solutions = None
        self._scaled_noises = None

    def get_top_idx(self) -> jnp.ndarray:
        """Get the index of the best solution."""
        return self._top_idx

    def ask(self) -> jnp.ndarray:
        
        center, stdev = self._center, self._stdev

        self._key, self._scaled_noises, self._solutions = ask_func(
            self._key,
            stdev,
            center,
            self._num_directions,
            self._center.size,
        )

        return self._solutions


    def tell(self, fitness_real: Union[np.ndarray, jnp.ndarray], fitness_fake: Union[np.ndarray, jnp.ndarray]) -> None:
        #fitness_scores = process_scores(fitness, self._solution_ranking)

        fitness_real_one = fitness_real[:, None]
        fitness_fake_one = fitness_fake[:, None]

        objectives = jnp.hstack([abs(fitness_real_one), abs(fitness_fake_one)])
        ranks = non_dominated_sort_lax(objectives)

        order = jnp.lexsort((-fitness_real_one.flatten(), ranks))
        top_idx = order[:1]
        self._top_idx = top_idx
        #best_fitness_real = jnp.max(fitness_real)
        #best_fitness_fake = jnp.max(fitness_fake)

        #best_prev_fitness_real = self._best_prev_fitness_real
        #best_prev_fitness_fake = self._best_prev_fitness_fake

        #best_prev_fitness_real_item = best_prev_fitness_real.item()
        #best_prev_fitness_fake_item = best_prev_fitness_fake.item()

        #if self._t < 2:
        #    best_real = best_fitness_real
        #    best_fake = best_fitness_fake
        #else:
        #    best_real = jnp.max(jnp.array([best_fitness_real, best_prev_fitness_real_item]))
        #    best_fake = jnp.max(jnp.array([best_fitness_fake, best_prev_fitness_fake_item]))

        #rng_real = jnp.max(fitness_real) - jnp.min(fitness_real)
        #rng_fake = jnp.max(fitness_fake) - jnp.min(fitness_fake)

        #norm_fitness_real = (fitness_real - best_real) / rng_real
        #norm_fitness_fake = (fitness_fake - best_fake) / rng_fake

        #one_dim_best_fitness_real = jnp.array([best_fitness_real])
        #one_dim_best_fitness_fake = jnp.array([best_fitness_fake])

        #best_real_window = self._best_real_window
        #best_fake_window = self._best_fake_window

        #self._best_real_window = jnp.concatenate([best_real_window, one_dim_best_fitness_real], axis=0)[1:]
        #self._best_fake_window = jnp.concatenate([best_fake_window, one_dim_best_fitness_fake], axis=0)[1:] 

        #real_window_var = jnp.var(best_real_window)
        #fake_window_var = jnp.var(best_fake_window)

        #total_var = real_window_var + fake_window_var

        #if total_var < 1e-8:
        #    lambda_real = 0.5
        #    lambda_fake = 0.5
        #elif self._t < 100:
        #    lambda_real = 0.5
        #    lambda_fake = 0.5
        #else:
        #    lambda_real = real_window_var / total_var
        #    lambda_fake = fake_window_var / total_var

        #lambda_real = jnp.clip(lambda_real, 0.2, 0.8)
        #lambda_fake = 1 - lambda_real

        #if self._t < 60:
        #    tchebycheff_scores = fitness_real + fitness_fake
        #else:
        #    tchebycheff_scores = lambda_real * norm_fitness_real + lambda_fake * norm_fitness_fake

        ##ranks = 1 / (ranks + 1)
        #ranks = jnp.log10(ranks + 2)
        #
        #if self._t < 60:
        #    fitness_scores = -ranks
        #else:
        #    fitness_scores = tchebycheff_scores #* ranks

        fitness_real_avg = jnp.mean(fitness_real)
        fitness_fake_avg = jnp.mean(fitness_fake)

        w_real = fitness_real_avg / (fitness_real_avg + fitness_fake_avg)
        w_real = jnp.clip(w_real, 0.3, 0.7)
        w_fake = 1 - w_real

        fitness_scores = w_real * fitness_real + w_fake * fitness_fake
        
        fitness_scores = process_scores(fitness_scores, True)

        grad_center, grad_stdev = compute_reinforce_update(
            fitness_scores=fitness_scores,
            scaled_noises=self._scaled_noises,
            stdev=self._stdev,
        )
        
        self._opt_state = self._opt_update(
            self._t // self._lr_decay_steps, -grad_center, self._opt_state
        )
        
        self._t += 1
        
        #if (self._t % 100) < 90:  
        self._center = self._get_params(self._opt_state)
        self._stdev = update_stdev(
                stdev=self._stdev,
                lr=self._stdev_lr,
                max_change=self._stdev_max_change,
                grad=grad_stdev,
            )

    @property
    def best_params(self) -> jnp.ndarray:
        return jnp.array(self._center, copy=True)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        self._center = jnp.array(params, copy=True)


@optimizers.optimizer
def clipup(
    step_size: float,
    momentum: float = 0.9,
    max_speed: float = 0.15,
    fix_gradient_size: bool = True,
):
    """Construct optimizer triple for ClipUp."""

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
        # Clip.
        length = jnp.sqrt(jnp.sum(v * v))
        v = jax.lax.cond(
            length > max_speed, lambda p: p * max_speed / length, lambda p: p, v
        )
        return x - v, v

    def get_params(state):
        x, _ = state
        return x

    return init, update, get_params

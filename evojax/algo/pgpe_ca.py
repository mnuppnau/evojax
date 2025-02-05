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

from evojax.algo.cultural.belief_space import (
    initialize_belief_space,
    get_updated_params,
)

from evojax.algo.cultural.knowledge_sources import (
    update_knowledge_sources,
    update_topographic_ks,
    update_normative_ks,
    add_ind_topographic_ks,
)

from evojax.algo.cultural.population_space import update_population

try:
    from jax.example_libraries import optimizers
except ModuleNotFoundError:
    from jax.experimental import optimizers

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger
from evojax.algo.cultural.helper_functions import calculate_entropy_sampling

@jax.jit
def compute_weights_by_rank(rank: jnp.ndarray) -> jnp.ndarray:
    """
    Given rank[i] in {0, 1, 2, ..., R-1} for each solution i,
    return harmonic weights so that rank=0 => 1.0, rank=1 => 0.5, etc.

    :param rank:  shape (N,), integer array of ranks
                  with 0-based indexing (0 = best front).
    :return:      weights of shape (N,), float array
    """
    # Convert to float for division
    rank_f = rank.astype(jnp.float32)
    # Harmonic: 1/(rank+1)
    weights = 1.0 / (rank_f + 1.0)
    return weights

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

def compute_crowding_distance(objectives: jnp.ndarray,
                              ranks: jnp.ndarray) -> jnp.ndarray:
    """
    Compute crowding distance for each point, given its objectives and Pareto rank.

    Args:
        objectives: shape (N, M) array of objective values (we assume minimization).
        ranks: shape (N,) array of Pareto ranks, e.g. from `non_dominated_sort_lax`.

    Returns:
        cdist: shape (N,) array of crowding distances.
    """
    N, M = objectives.shape
    # Initialize distances to 0
    cdist = jnp.zeros((N,), dtype=jnp.float32)

    # Gather all unique ranks (excluding -1 if present)
    unique_ranks = jnp.unique(ranks[ranks >= 0])

    # Because we’re going to do a loop in Python, cdist won't be fully JIT-traceable.
    # This is typically acceptable for moderate N. For a purely JAX solution, you'd
    # use lax.fori_loop or other transforms, which is more advanced.

    for rank_val in unique_ranks:
        # Indices belonging to this front
        front_mask = (ranks == rank_val)
        front_idx = jnp.where(front_mask)[0]  # the actual integer indices in this front
        num_front = front_idx.size

        # If there's fewer than 2 solutions in the front, those points get "infinite" distance
        if num_front <= 2:
            # Because we do it with Python, we can’t do an in-place update of cdist as in NumPy.
            # We’ll just use `jnp.where` to assign inf to these points.
            cdist = cdist.at[front_idx].set(jnp.inf)
            continue

        # For each objective dimension, compute partial crowding distances
        front_obj = objectives[front_idx, :]  # shape (num_front, M)

        # For each objective, sort the front by that objective
        for m in range(M):
            sorted_idx_within_front = jnp.argsort(front_obj[:, m], axis=0)
            sorted_actual_idx = front_idx[sorted_idx_within_front]  # actual indices in original array

            # Mark boundary solutions as infinite
            cdist = cdist.at[sorted_actual_idx[0]].set(jnp.inf)
            cdist = cdist.at[sorted_actual_idx[-1]].set(jnp.inf)

            # If all boundary points are inf, only the interior points get calculations
            # max_obj and min_obj in this front for dimension m
            obj_min = jnp.min(front_obj[:, m])
            obj_max = jnp.max(front_obj[:, m])
            denom = obj_max - obj_min
            # If denom is zero (all points same in this objective), increments are 0

            def middle_update(i, cd):
                # i indexes from 1..(num_front-2)
                left_idx  = sorted_actual_idx[i-1]
                right_idx = sorted_actual_idx[i+1]
                mid_idx   = sorted_actual_idx[i]

                # difference in objectives
                diff = (objectives[right_idx, m] - objectives[left_idx, m]) / jnp.where(denom == 0., 1., denom)
                return cd.at[mid_idx].add(diff)

            # We can do a python loop or a small lax.fori_loop over i in [1..num_front-2]
            # Here is a simple Python approach:
            for i in range(1, num_front - 1):
                cdist_val = (objectives[sorted_actual_idx[i+1], m]
                             - objectives[sorted_actual_idx[i-1], m])
                # Normalize by the range (avoid div by zero)
                increment = cdist_val / (denom if denom != 0.0 else 1e-30)
                cdist = cdist.at[sorted_actual_idx[i]].add(increment)

    return cdist

@partial(jax.jit, static_argnums=(1,))
def process_scores(
    x: Union[np.ndarray, jnp.ndarray], use_ranking: bool
) -> jnp.ndarray:
    """Convert fitness scores to rank if necessary."""

    x = jnp.array(x)
    if use_ranking:
        ranks = jnp.zeros(x.size, dtype=int)
        ranks = ranks.at[x.argsort()].set(jnp.arange(x.size)).reshape(x.shape)
        return ranks / ranks.max() - 0.5, jnp.array(x).max(), jnp.array(x).mean()
    else:
        return x, jnp.array(x).max(), jnp.array(x).mean()

@jax.jit
def normalize_gradients(
    grad_center: jnp.ndarray, grad_stdev: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    grad_center = grad_center / jnp.linalg.norm(grad_center)
    grad_stdev = grad_stdev / jnp.linalg.norm(grad_stdev)
    return grad_center, grad_stdev

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


class PGPE(NEAlgorithm):
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
        belief_space: jnp.ndarray = None,
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

        self._key = random.PRNGKey(seed=seed)
        self._key, self._subkey = random.split(self._key)
        self._solutions = None
        self._scaled_noises = None

        self.belief_space = belief_space if belief_space is not None else initialize_belief_space(
            population_size=self.pop_size, param_size=abs(param_size), key=self._key)

    def ask(self) -> jnp.ndarray:
        #if self._t > 94000:
        #    center, stdev = get_updated_params(
        #        self.belief_space, self._center, self._stdev, self._t
        #    )
        #else:
        center, stdev = self._center, self._stdev

        self._key, self._scaled_noises, self._solutions = ask_func(
            self._key,
            stdev,
            center,
            self._num_directions,
            self._center.size,
        )

        return self._solutions, self.belief_space


    def tell(self, fitness_adv: Union[np.ndarray, jnp.ndarray],fitness_bin: Union[np.ndarray, jnp.ndarray],fitness_mi: Union[np.ndarray, jnp.ndarray],fitness_con: Union[np.ndarray, jnp.ndarray], pop_stats: jnp.ndarray) -> None:
        #fitness_scores, self._best_score, self._avg_score = process_scores(fitness_adv, self._solution_ranking)
        #fitness_scores_bin, best_score_bin, avg_score_bin = process_scores(fitness_bin, self._solution_ranking)
        #fitness_scores_mi, self._best_score_mi, self._avg_score_mi = process_scores(fitness_mi, self._solution_ranking)
        #fitness_scores_con, _, _ = process_scores(fitness_con, self._solution_ranking)

        # add a dimension to the fitness scores so that (256,) becomes (256, 1)
        fitness_adv = fitness_adv[:, None]
        fitness_mi = fitness_mi[:, None]

        objectives = jnp.hstack([abs(fitness_adv), abs(fitness_mi)])
        ranks = non_dominated_sort_lax(objectives)
        #jax.debug.print('ranks : {} ', ranks)
        #get unique ranks
        #unique_ranks = jnp.unique(ranks)
        #crowding_distances = compute_crowding_distance_lax(objectives, ranks, unique_ranks)

        # get indices of the top-ranked individuals, rank 0
        #top_ranked_indices = jnp.where(ranks == 0)[0]

        #cdist = compute_crowding_distance(objectives, ranks)

        #order = jnp.lexsort((-cdist, ranks))
        
        #top_four_indices = order[:4]

        #top_four_solutions = self._solutions[top_four_indices]
        #top_four_scaled_noises = self._scaled_noises[top_four_indices]

        #top_four_fitness_adv = fitness_adv[top_four_indices]
        #top_four_fitness_mi = fitness_mi[top_four_indices]

        best_fitness_adv = jnp.max(fitness_adv)
        best_fitness_mi = jnp.max(fitness_mi)

        best_prev_fitness_adv = self.belief_space[5][8]
        best_prev_fitness_mi = self.belief_space[5][9]

        best_prev_fitness_adv = best_prev_fitness_adv.item()
        best_prev_fitness_mi = best_prev_fitness_mi.item()

        #jax.debug.print('best fitness adv : {} ', best_fitness_adv)
        #jax.debug.print('best prev fitness adv : {} ', best_prev_fitness_adv)

        # take the max between the current and previous best
        if self._t < 2:
            best_adv = best_fitness_adv
            best_mi = best_fitness_mi
        else:
            best_adv = jnp.max(jnp.array([best_fitness_adv, best_prev_fitness_adv]))
            best_mi = jnp.max(jnp.array([best_fitness_mi, best_prev_fitness_mi]))

        #jax.debug.print('best adv : {} ', best_adv)
        #jax.debug.print('best mi : {} ', best_mi)

        avg_fitness_adv = jnp.mean(fitness_adv)
        avg_fitness_mi = jnp.mean(fitness_mi)

        rng_adv = jnp.max(fitness_adv) - jnp.min(fitness_adv)
        rng_mi = jnp.max(fitness_mi) - jnp.min(fitness_mi)

       
        #jax.debug.print('rng adv : {} ', rng_adv)
        #jax.debug.print('rng mi : {} ', rng_mi)

        norm_fitness_adv = (fitness_adv - best_adv) / rng_adv
        norm_fitness_mi = (fitness_mi - best_mi) / rng_mi


        #jax.debug.print('norm fitness adv avg : {} ', jnp.mean(norm_fitness_adv))
        #jax.debug.print('norm fitness mi avg : {} ', jnp.mean(norm_fitness_mi))
        if self._t < 20000:
            tchebycheff_scores = jnp.minimum(norm_fitness_adv*0.6, norm_fitness_mi*0.4)
        elif self._t >= 20000 and self._t < 21000:
            tchebycheff_scores = norm_fitness_adv*0.55 + norm_fitness_mi*0.45
        elif self._t >= 21000 and self._t < 30000:
            tchebycheff_scores = norm_fitness_adv*0.7 + norm_fitness_mi*0.3
        else:
            tchebycheff_scores = norm_fitness_adv*0.5 + norm_fitness_mi*0.5
        #tchebycheff_scores = norm_fitness_adv*0.3 + norm_fitness_mi*0.7
        
        best_tchebycheff_scores = jnp.max(tchebycheff_scores)
        avg_tchebycheff_scores = jnp.mean(tchebycheff_scores)

        self.belief_space = update_normative_ks(
            self.belief_space,
            best_fitness=best_fitness_adv,
            best_fitness_mi=best_fitness_mi,
            avg_fitness=avg_fitness_adv,
            avg_fitness_mi=avg_fitness_mi,
            best_adv=best_adv,
            best_mi=best_mi,
            rng_adv=rng_adv,
            rng_mi=rng_mi,
            best_tchebycheff_scores=best_tchebycheff_scores,
            avg_tchebycheff_scores=avg_tchebycheff_scores,
        )
        #weights = compute_weights_by_rank(ranks)
       
        #fitness_adv_std = jnp.std(fitness_adv)
        #fitness_adv_min = jnp.min(fitness_adv)

        #jax.debug.print('weights : {} ', weights)
        #weights = fitness_adv_std * weights + fitness_adv_min 

        #jax.debug.print('weights : {} ', weights)
        #weights = -weights
        fitness_scores, self._best_score, self._avg_score = process_scores(tchebycheff_scores, False)
        
        #if self._t % 5 == 0:
        #    grad_center, grad_stdev = compute_reinforce_update(
        #        fitness_scores=fitness_scores_mi,
        #        scaled_noises=self._scaled_noises,
        #       stdev=self._stdev,
        #    )
        #else:
        

        grad_center, grad_stdev = compute_reinforce_update(
                fitness_scores=fitness_scores,
                scaled_noises=self._scaled_noises,
                stdev=self._stdev,
            )
        
        ##grad_center_norm, grad_stdev_norm = normalize_gradients(grad_center, grad_stdev)

        #self.population, best_individual = update_population(
        #    fitness_scores=fitness_scores_mi,
        #    center=self._center,
        #    stdev=self._stdev,
        #)

        #jax.debug.print('solutions shape in tell : {} ', self._solutions.shape)
        #self._subkey, norm_entropy = calculate_entropy_sampling(self._subkey, self._solutions)

        ##norm_entropy = 0.0

        #best_score = jnp.array([self._best_score])
        #best_score_mi = jnp.array([self._best_score_mi])

        #if self._t > 90000 and self._t < 94000 and self._t % 20 == 0:
            #grad_center_topo = grad_center_mi*0.7 + grad_center_con*0.3
            #grad_stdev_topo = grad_stdev_mi*0.7 + grad_stdev_con*0.3

            #grad_center_topo_norm, grad_stdev_topo_norm = normalize_gradients(grad_center_topo, grad_stdev_topo)
        #    self.belief_space = add_ind_topographic_ks(
        #        self.belief_space, grad_center, grad_stdev, best_score_mi, max_individuals=20
        #    )
        #elif self._t >= 94000 and self._t % 2 == 0:
            #grad_center_topo = grad_center_mi*0.7 + grad_center_con*0.3
            #grad_stdev_topo = grad_stdev_mi*0.7 + grad_stdev_con*0.3

            #grad_center_topo_norm, grad_stdev_topo_norm = normalize_gradients(grad_center_topo, grad_stdev_topo)
       #     self.belief_space = update_topographic_ks(
       #         self.belief_space, grad_center, grad_stdev, best_score_mi, max_individuals=20
       #     )

        #min_index = 0

        #if self._t > 90000:
        #    self.belief_space, ks_weights = update_normative_ks(
        #        self.belief_space,
        #        best_fitness=self._best_score_mi,
        #        avg_fitness=self._avg_score_mi,
        #        norm_entropy=norm_entropy,
        #        pop_stats=pop_stats,
        #    )

        #    ##jax.debug.print('ks weights before update : {} ', ks_weights) 
        #    min_index = jnp.argmin(ks_weights)
        #    result = jnp.zeros(4)
        #    ks_weights = result.at[min_index].set(1.0)

        ###jax.debug.print('ks weights after update : {} ', ks_weights)
        #if self._t > 94000:
        #    if min_index== 3:
        #        updated_grad_center = self.belief_space[4][3]
        #        updated_grad_stdev = self.belief_space[4][4]

        #        cluster_weights_center = self.belief_space[4][7]
        #        cluster_weights_stdev = self.belief_space[4][8]

        #        weighted_sum_center = jnp.sum(
        #            cluster_weights_center[:, None] * updated_grad_center, axis=0
        #        )
        #        weighted_sum_stdev = jnp.sum(
        #            cluster_weights_stdev[:, None] * updated_grad_stdev, axis=0
        #        )
        #        grad_center = weighted_sum_center * 0.7 + grad_center * 0.3
        #        grad_stdev = weighted_sum_stdev * 0.7 + grad_stdev * 0.3
        
                #grad_center_norm, grad_stdev_norm = normalize_gradients(grad_center, grad_stdev)
            #elif min_index == 0 and self._t % 5 == 0:
            #     #    grad_center = processed_activation_grads * 0.32 + grad_center * 0.68
            #    grad_center = grad_center_bin
            #    grad_stdev = grad_stdev_bin
            #elif min_index == 1 and self._t % 5 == 0:
            #    grad_center = grad_center_bin
            #    grad_stdev = grad_stdev_bin

            #    #grad_center_norm, grad_stdev_norm = normalize_gradients(grad_center_sit, grad_stdev_sit)
            #elif min_index == 2:
            #    grad_center = grad_center*0.95 + grad_center_bin*0.05
            #    grad_stdev = grad_stdev*0.95 + grad_stdev_bin*0.05

            #    #grad_center_norm, grad_stdev_norm = normalize_gradients(grad_center_hist, grad_stdev_hist)
            #else:
            #    grad_center = grad_center
            #    grad_stdev = grad_stdev

        
        self._opt_state = self._opt_update(
                self._t // self._lr_decay_steps, -grad_center, self._opt_state
        )
        self._t += 1
       
        self._center = self._get_params(self._opt_state)
        
        self._stdev = update_stdev(
                stdev=self._stdev,
                lr=self._stdev_lr,
                max_change=self._stdev_max_change,
                grad=grad_stdev,
            )

        #self.belief_space = update_knowledge_sources(
        #    self.belief_space,
        #    (
        #        self._center,
        #        self._stdev,
        #        best_individual[2],
        #    ),
        #    pop_stats,
        #)

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

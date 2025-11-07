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
    update_topographic_ks,
    update_topographic_ks_idx_one,
    update_topographic_ks_idx_two,
    update_topographic_ks_idx_three,
    update_topographic_ks_idx_four,
    update_topographic_ks_idx_five,
    update_topographic_ks_idx_six,
    update_topographic_ks_idx_seven,
    update_topographic_ks_idx_eight,
    update_topographic_ks_idx_nine,
    update_domain_ks,
    update_situational_ks,
    update_history_ks,
    update_normative_ks,
)

from evojax.algo.cultural.helper_functions import non_dominated_sort_lax
from evojax.algo.cultural.population_space import update_population

try:
    from jax.example_libraries import optimizers
except ModuleNotFoundError:
    from jax.experimental import optimizers

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger

@jax.jit
def compute_closeness(objectives):
 
    normalized_objectives = jnp.divide(objectives, jnp.sqrt(jnp.sum(objectives**2, axis=0)))
        
    ideal_ind = jnp.min(normalized_objectives, axis=0)
    anti_ideal_ind = jnp.max(normalized_objectives, axis=0)

    distance_plus = jnp.sqrt(jnp.sum((normalized_objectives - ideal_ind)**2, axis=1))
    distance_minus = jnp.sqrt(jnp.sum((normalized_objectives - anti_ideal_ind)**2, axis=1))

    closeness_coef = distance_minus / (distance_plus + distance_minus)
    return closeness_coef

@jax.jit
def normalize_ranks(ranks):
    max_rank = jnp.max(ranks)
    return (max_rank - ranks) / max_rank  # Higher values for better ranks

@jax.jit
def compute_fitness(closeness, ranks):
    normalized_ranks = normalize_ranks(ranks)
    
    # Combine rank and closeness (you can adjust these weights)
    rank_weight = 0.9
    closeness_weight = 0.1
    
    fitness = rank_weight * normalized_ranks + closeness_weight * closeness
    
    return fitness

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
    new_stdev = jnp.clip(stdev + lr * grad, min_allowed, max_allowed)
    
    return jnp.clip(new_stdev, 1e-8, 1e1)

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

# create an ask_func that takes in two sets of parameters and concatenates the two solutions
@partial(jax.jit, static_argnums=(3, 4, 7))
def ask_func_concat(
    key: jnp.ndarray,
    stdev: jnp.ndarray,
    center: jnp.ndarray,
    num_directions: int,
    solution_size: int,
    stdev_ca: jnp.ndarray,
    center_ca: jnp.ndarray,
    num_directions_ca: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A function that samples a population of parameters from Gaussian."""

    next_key, key = random.split(key)
    scaled_noises = jnp.vstack([
        random.normal(key, [num_directions - num_directions_ca, solution_size]) * stdev, 
        random.normal(key, [num_directions_ca, solution_size]) * stdev_ca
    ])
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

        self.MIN_DIVERSITY = 0.005

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
        decay_coef = optimizer_config.get("center_lr_decay_coef", 0.99)
        self._lr_decay_steps = optimizer_config.get(
            "center_lr_decay_steps", 30000
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

        self._arr = jnp.arange(10)
        self._key = random.PRNGKey(seed=seed)
        self._key, self._subkey = random.split(self._key)
        self._solutions = None
        self._scaled_noises = None

        self.belief_space = belief_space if belief_space is not None else initialize_belief_space(
            population_size=self.pop_size, param_size=abs(param_size), key=self._key)

    def get_top_idx(self) -> jnp.ndarray:
        """Get the index of the top solution."""
        return self._top_indices

    def ask_ca(self) -> jnp.ndarray:
        center_ca, stdev_ca, min_index = get_updated_params(
            self.belief_space, self._center, self._stdev, self._t
        )
        #jax.debug.print('center ca shape {} : ', center_ca.shape)
        #jax.debug.print('min index {} : ', min_index)
        return center_ca.flatten()

    def ask(self) -> jnp.ndarray:
        #if self._t > 100000:
        #    center_ca, stdev_ca, min_index = get_updated_params(
        #        self.belief_space, self._center, self._stdev, self._t
        #    )
        #    #jax.debug.print('max center ca value {} : ', jnp.max(center_ca))
        #    #jax.debug.print('min center ca value {} : ', jnp.min(center_ca))
        #    #jax.debug.print('max stddev ca value {} : ', jnp.max(stdev_ca))
        #    #jax.debug.print('min stddev ca value {} : ', jnp.min(stdev_ca))
        #    #jax.debug.print('min index {} : ', min_index)
        #    #jax.debug.print('max center value {} : ', jnp.max(self._center))
        #    #jax.debug.print('min center value {} : ', jnp.min(self._center))
        #    #jax.debug.print('max stddev value {} : ', jnp.max(self._stdev))
        #    #jax.debug.print('min stddev value {} : ', jnp.min(self._stdev))
        #    if min_index == 0:
        #        stdev_ca = stdev_ca * 0.2
        #        num_directions_ca = 16
        #    elif min_index == 1:
        #        stdev_ca = stdev_ca * 0.1
        #        num_directions_ca = 16
        #    elif min_index == 2:
        #        stdev_ca = stdev_ca * 0.2
        #        num_directions_ca = 16
        #    elif min_index == 3:
        #        stdev_ca = stdev_ca * 0.3
        #        num_directions_ca = 16
        #    center, stdev = self._center, self._stdev
        #else:
        center, stdev = self._center, self._stdev

        
        #if self._t > 100000:
        #    # clip stdev_ca to be between 1e-4 and 1e1
        #    stdev_ca = jnp.clip(stdev_ca, 1e-4, 1e1)
        #    self._key, self._scaled_noises, self._solutions = ask_func_concat(
        #        self._key,
        #        stdev,
        #        center,
        #        self._num_directions,
        #        self._center.size,
        #        stdev_ca,
        #        center_ca,
        #        num_directions_ca // 2
        #    )
        #else:
        self._key, self._scaled_noises, self._solutions = ask_func(
            self._key,
            stdev,
            center,
            self._num_directions,
            self._center.size,
        )

        return self._solutions, self.belief_space


    def tell(self, fitness_adv: Union[np.ndarray, jnp.ndarray], fitness_mi: Union[np.ndarray, jnp.ndarray],fitness_con: Union[np.ndarray, jnp.ndarray], disc_logits: Union[np.ndarray, jnp.ndarray], pop_var: Union[np.ndarray, jnp.ndarray], avg_per_code: Union[np.ndarray, jnp.ndarray], r_cons: Union[np.ndarray, jnp.ndarray], r_sense: Union[np.ndarray, jnp.ndarray], r_intra: Union[np.ndarray, jnp.ndarray], adv: bool) -> None:

        
        
        # add a dimension to the fitness scores so that (256,) becomes (256, 1)
        fitness_adv = fitness_adv[:, None]
        fitness_mi = fitness_mi[:, None]
        fitness_con = fitness_con[:, None]

        r_cons2 = r_cons[:, None]
        r_sense2 = r_sense[:, None]
        r_intra2 = r_intra[:, None]
        pop_var2 = pop_var[:, None]

        penalty = jnp.where(
             pop_var < self.MIN_DIVERSITY,
             (self.MIN_DIVERSITY - pop_var) * 100,  # Heavy penalty if below threshold
             0.0  # No penalty if above threshold
        )
        #jax.debug.print('fitness adv scores {} : ', fitness_adv.flatten())
       
        if self._t < 20000:
            objectives = jnp.hstack([-fitness_adv, -fitness_mi, -fitness_con])
        elif self._t < 30000:
            objectives = jnp.hstack([-fitness_adv, -r_cons2,  -fitness_mi])
        elif self._t < 40000:
            objectives = jnp.hstack([-r_cons2, -r_sense2, -fitness_adv])
        else: 
            objectives = jnp.hstack([-r_cons2, -r_intra2, -fitness_adv])

        #objectives = jnp.hstack([-fitness_adv, -fitness_mi, -pop_var2 , r_cons2, -r_sense2, -r_intra2, -fitness_con])
        #jax.debug.print('objectives {} : ', objectives)
        #jax.debug.print('objectives shape {} : ', objectives.shape)
        ranks = non_dominated_sort_lax(objectives)
        #get unique ranks
        #unique_ranks = jnp.unique(ranks)
        #crowding_distances = compute_crowding_distance_lax(objectives, ranks, unique_ranks)

        # get indices of the top-ranked individuals, rank 0
        #top_ranked_indices = jnp.where(ranks == 0)[0]

        #cdist = compute_crowding_distance(objectives, ranks)

        #if adv:
        order = jnp.lexsort((-fitness_adv.flatten(), ranks))
        #else:
        #order = jnp.lexsort((-fitness_adv.flatten(), ranks))

        #top_index = order[:1]

        #self._top_idx = top_index
        
        #top_indices = order[:40]

        #self._top_indices = top_indices

        #top_solution = self._solutions[top_index]
        #top_scaled_noise = self._scaled_noises[top_index]

        #top_fitness_adv = fitness_adv[top_index]
        #top_fitness_mi = fitness_mi[top_index]

        #top_disc_logit = disc_logits[top_index]

        # apply softmax to the logits
        #softmax_logits = jax.nn.softmax(top_disc_logit, axis=-1)
        #max_disc_logit_idx = jnp.argmax(abs(top_disc_logit))

        best_fitness_adv = jnp.max(fitness_adv)
        best_fitness_mi = jnp.max(fitness_mi)
        best_fitness_con = jnp.max(fitness_con)

        #best_prev_fitness_adv = self.belief_space[5][8]
        #best_prev_fitness_mi = self.belief_space[5][9]

        #best_prev_fitness_adv = best_prev_fitness_adv.item()
        #best_prev_fitness_mi = best_prev_fitness_mi.item()

        # take the max between the current and previous best
        #if self._t < 2:
        #    best_adv = best_fitness_adv
        #    best_mi = best_fitness_mi
        #else:
        #    best_adv = jnp.max(jnp.array([best_fitness_adv, best_prev_fitness_adv]))
        #    best_mi = jnp.max(jnp.array([best_fitness_mi, best_prev_fitness_mi]))

        avg_fitness_adv = jnp.mean(fitness_adv)
        avg_fitness_mi = jnp.mean(fitness_mi)

        #rolling_window_avg_adv = self.belief_space[5][2]
        #rolling_window_avg_mi = self.belief_space[5][3]

        #mu_adv = jnp.mean(rolling_window_avg_adv)
        #mu_mi = jnp.mean(rolling_window_avg_mi)

        #sigma_adv = jnp.std(rolling_window_avg_adv) + 1e-8
        #sigma_mi = jnp.std(rolling_window_avg_mi) + 1e-8

        #L_adv_norm = (fitness_adv.flatten() - mu_adv) / sigma_adv
        #L_mi_norm = (fitness_mi.flatten() - mu_mi) / sigma_mi


        #rng_adv = jnp.max(fitness_adv) - jnp.min(fitness_adv)
        #rng_mi = jnp.max(fitness_mi) - jnp.min(fitness_mi)
        #rng_con = jnp.max(fitness_con) - jnp.min(fitness_con)

        #norm_fitness_adv = (fitness_adv - best_adv) / rng_adv
        #norm_fitness_mi = (fitness_mi - best_mi) / rng_mi
        #norm_fitness_con = (fitness_con - best_fitness_con) / rng_con

        w_adv = avg_fitness_adv / (avg_fitness_adv + avg_fitness_mi)
        w_adv = jnp.clip(w_adv, 0.2, 0.8)
        #w_adv = w_adv*3
        #w_adv = jnp.clip(w_adv, 0.5, 0.9)
        w_mi = 1 - w_adv

        #jax.debug.print('w_adv : {} ', w_adv)
        #jax.debug.print('w_con : {} ', w_con)
        #best_adv_window = self.belief_space[5][0]
        #best_mi_window = self.belief_space[5][1]

        #adv_window_var = jnp.var(best_adv_window)
        #mi_window_var = jnp.var(best_mi_window)
        #total_var = adv_window_var + mi_window_var

        #if self._t < 60000:
        #    lambda_mi = 0.1 + self._t // 100000
        #else: 
        #    lambda_adv = 0.3 + (self._t - 59000) // 50000
        #    lambda_mi = 1 - lambda_adv

        lambda_mi = 0.4

        #if self._t < 100:
        #tchebycheff_scores = norm_fitness_adv.flatten()*(1-w_mi) + norm_fitness_mi.flatten()*w_mi #+ norm_fitness_con.flatten()*0.01
        #elif self._t < 8000:
        #tchebycheff_scores = L_adv_norm + L_mi_norm 
        #elif self._t < 16000:
        #    tchebycheff_scores = L_adv_norm + L_mi_norm * 0.2
        #elif self._t < 24000:
        #    tchebycheff_scores = L_adv_norm + L_mi_norm * 0.5
        #else:
        #    tchebycheff_scores = L_adv_norm + L_mi_norm * lambda_mi

        #tchebycheff_scores = w_adv * fitness_adv.flatten() + w_mi * fitness_mi.flatten() 
        #tchebycheff_scores =  lambda_adv * norm_fitness_adv.flatten() + lambda_mi * norm_fitness_mi.flatten() 
        #top_tchebycheff = tchebycheff_scores[top_index]

        #best_tchebycheff_scores = jnp.max(tchebycheff_scores)
        #avg_tchebycheff_scores = jnp.mean(tchebycheff_scores)

        #self._subkey, norm_entropy = calculate_entropy_sampling(self._subkey, self._solutions)
        
        #self.belief_space = update_domain_ks(
        #    self.belief_space,
        #    top_solution,
        #    self._stdev,
        #    top_scaled_noise,
        #    top_fitness_adv,
        #    top_fitness_mi,
        #    top_tchebycheff,
        #    softmax_logits,
        #)

        #self.belief_space = update_situational_ks(
        #    self.belief_space, 
        #    top_solution, 
        #    self._stdev, 
        #    top_scaled_noise, 
        #    top_fitness_adv, 
        #    top_fitness_mi, 
        #    top_tchebycheff, 
        #    softmax_logits,
        #)

        #if self._t > 400 and self._t % 10 == 0:
        #    self.belief_space = update_history_ks(
        #        self.belief_space, 
        #        top_solution, 
        #        self._stdev, 
        #        top_scaled_noise, 
        #        top_fitness_adv, 
        #        top_fitness_mi, 
        #        top_tchebycheff, 
        #        softmax_logits, 
        #    )
        #elif self._t <= 400:
        #    self.belief_space = update_history_ks(
        #        self.belief_space, 
        #        top_solution, 
        #        self._stdev, 
        #        top_scaled_noise, 
        #        top_fitness_adv, 
        #        top_fitness_mi, 
        #        top_tchebycheff, 
        #        softmax_logits, 
        #    )


        # top_disc_logit is shape (1,128,10), take the average over the 10 logits and return shape (10,)
        #softmax_avg = jnp.mean(softmax_logits, axis=1).squeeze()

        #max_softmax_logits_idx = jnp.argmax(abs(softmax_avg))

        #self._arr = jnp.concatenate(([max_softmax_logits_idx], self._arr[self._arr != max_softmax_logits_idx]))
       
        #oldest_idx = self._arr[-1]

        self.belief_space = update_topographic_ks(
            self.belief_space, avg_per_code
        )
        #jax.debug.print('max softmax idx {} : ', max_softmax_logits_idx)
        #if max_softmax_logits_idx == 0:
        #    self.belief_space = update_topographic_ks_idx_zero(
        #        self.belief_space, top_solution, self._stdev, top_scaled_noise, top_fitness_adv, top_fitness_mi, softmax_logits
        #    )
        #elif max_softmax_logits_idx == 1:
        #    self.belief_space = update_topographic_ks_idx_one(
        #        self.belief_space, top_solution, self._stdev, top_scaled_noise, top_fitness_adv, top_fitness_mi, softmax_logits
        #    )
        #elif max_softmax_logits_idx == 2:
        #    self.belief_space = update_topographic_ks_idx_two(
        #        self.belief_space, top_solution, self._stdev, top_scaled_noise, top_fitness_adv, top_fitness_mi, softmax_logits
        #    )
        #elif max_softmax_logits_idx == 3:
        #    self.belief_space = update_topographic_ks_idx_three(
        #        self.belief_space, top_solution, self._stdev, top_scaled_noise, top_fitness_adv, top_fitness_mi, softmax_logits
        #    )
        #elif max_softmax_logits_idx == 4:
        #    self.belief_space = update_topographic_ks_idx_four(
        #        self.belief_space, top_solution, self._stdev, top_scaled_noise, top_fitness_adv, top_fitness_mi, softmax_logits
        #    )
        #elif max_softmax_logits_idx == 5:
        #    self.belief_space = update_topographic_ks_idx_five(
        #        self.belief_space, top_solution, self._stdev, top_scaled_noise, top_fitness_adv, top_fitness_mi, softmax_logits
        #    )
        #elif max_softmax_logits_idx == 6:
        #    self.belief_space = update_topographic_ks_idx_six(
        #        self.belief_space, top_solution, self._stdev, top_scaled_noise, top_fitness_adv, top_fitness_mi, softmax_logits
        #    )
        #elif max_softmax_logits_idx == 7:
        #    self.belief_space = update_topographic_ks_idx_seven(
        #        self.belief_space, top_solution, self._stdev, top_scaled_noise, top_fitness_adv, top_fitness_mi, softmax_logits
        #    )
        #elif max_softmax_logits_idx == 8:
        #    self.belief_space = update_topographic_ks_idx_eight(
        #        self.belief_space, top_solution, self._stdev, top_scaled_noise, top_fitness_adv, top_fitness_mi, softmax_logits
        #    )
        #elif max_softmax_logits_idx == 9:
        #    self.belief_space = update_topographic_ks_idx_nine(
        #        self.belief_space, top_solution, self._stdev, top_scaled_noise, top_fitness_adv, top_fitness_mi, softmax_logits
        #    )

        #rolling_digits = self.belief_space[5][5]
        #unique_digits = jnp.unique(rolling_digits)

        #missing_digits = jnp.setdiff1d(jnp.arange(10), unique_digits)
        
        #if len(missing_digits) > 0:
        #    first_missing_digit = missing_digits[0]
        #else:
        #    first_missing_digit = 1

        #topographic_center_idx = (first_missing_digit+1)*6-6
        #topographic_center = self.belief_space[4][topographic_center_idx][0]
        #topographic_stdev = self.belief_space[4][topographic_center_idx+1][0]
       
        ## create empty array
        #softmax_logits = jnp.zeros((10, 128))
        #self.belief_space = update_normative_ks(
        #    self.belief_space,
        #    best_fitness=best_fitness_adv,
        #    best_fitness_mi=best_fitness_mi,
        #    avg_fitness=avg_fitness_adv,
        #    avg_fitness_mi=avg_fitness_mi,
        #    best_adv=best_adv,
        #    best_mi=best_mi,
        #    rng_adv=rng_adv,
        #    digit=None,#max_softmax_logits_idx,
        #    best_tchebycheff_scores=best_tchebycheff_scores,
        #    softmax_logits=softmax_logits,
        #    missing_digit=first_missing_digit,
        #    topographic_center=topographic_center,
        #    topographic_stdev=topographic_stdev,
        #)

        #ranks = jnp.log10(ranks + 2)

        # multiply each value in tchhebycheff_scores (axis 0, which is shape (128,1)) by the value in the same index in ranks (which is a scalar) 
        #if adv:
            #fitness_scores = -jnp.argsort(order+1)
        #else:
        #if adv:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten()*(30*(w_mi)) + fitness_con.flatten() * 0.1
        #elif not adv and self._t > 6000:
         #   fitness_scores = tchebycheff_scores #* ranks.reshape(ranks.shape[0], 1)
            #fitness_scores = fitness_adv.flatten() + fitness_mi.flatten()# + fitness_con.flatten()
            #closeness = compute_closeness(objectives)
            #fitness_scores = compute_fitness(closeness, ranks)
        #else:
        # create a weight that increases linearly from 0.1 to 0.7 over 20000 timesteps
        #if self._t <= 20000:
        #w_mi = 0.01 + ((1 + (self._t / 10000))**2)
        if self._t < 5000:
            w_adv = 12.0  # Scale up to match MI's natural magnitude
            w_mi = 0.1   # Scale down MI heavily initially
            w_con = 1.0  # Boost continuous
        else:
            w_adv = 6.0
            w_mi = 0.1 + (self._t - 5000) / 5000  # Ramp gradually
            w_con = 2.0
        #elif self._t < 40000:
        #    w_mi = 0.3 + (0.4 * ((self._t - 20000) / 20000)) 
        #else:
        #    w_mi = 0.7
        
        #if self._t < 40000:
        #    w_con = 0.05
        #elif self._t < 80000:
        #    w_con = 0.1
        #else:
        #    w_con = 0.2
        #if self._t < 4000:
            #fitness_scores = fitness_adv.flatten()*(1-w_mi) + fitness_mi.flatten()*w_mi + fitness_con.flatten()*0.1
        #elif self._t < 8000:
        #fitness_scores = fitness_adv.flatten()*w_adv + fitness_mi.flatten()*w_mi + fitness_con.flatten()*w_con
        #elif self._t < 11000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten()*(60*(w_mi)) + fitness_con.flatten()*0.8
        #elif self._t < 16000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten()*(100*(w_mi)) + fitness_con.flatten()*1
        #elif self._t < 24000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten()*(160*(w_mi)) + fitness_con.flatten()*2
        #else:
            #fitness_scores = fitness_adv.flatten() + fitness_mi.flatten()*(1000*(w_mi)) + fitness_con.flatten()*8
            #fitness_scores = tchebycheff_scores
            #fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() + fitness_con.flatten() * 0.01
        #fitness_scores = norm_fitness_adv.flatten()*(1-w_mi) + norm_fitness_mi.flatten()*w_mi + norm_fitness_con.flatten()*w_con
        #fitness_scores = -jnp.argsort(order+1)
        #else:
            #fitness_scores = fitness_adv.flatten() + fitness_mi.flatten()*50
            #closeness = compute_closeness(objectives)
            #fitness_scores = compute_fitness(closeness, ranks)
        #fitness_scores = compute_fitness(closeness, ranks)
        #if self._t < 200:
        #    fitness_scores = fitness_adv.flatten()
        #elif self._t >= 200 and self._t % 2 == 0:
        #if adv:
            #fitness_scores = fitness_adv.flatten()
        #fitness_scores = -jnp.argsort(order+1)
        #fitness_scores = tchebycheff_scores
        #else:
            #fitness_scores = fitness_mi.flatten()
            #fitness_scores = fitness_mi
        #jax.debug.print('weights : {} ', weights)
        #weights = -weights
        avg_diversity = pop_var.mean()
        std_diversity = pop_var.std()
        avg_fitness_adv = fitness_adv.mean()
        
        ## Phase detection
        #if avg_diversity < self.MIN_DIVERSITY * 0.8:
        #    # Phase 1: Need diversity desperately
        #    w_diversity = 50.0
        #    w_adversarial = 1.0
        #    phase = "BOOTSTRAPPING_DIVERSITY"
        #    
        #elif avg_diversity >= self.MIN_DIVERSITY and std_diversity > 0.002:
        #    # Phase 2: Have diversity, need quality
        #    w_diversity = 5.0  # Maintain but don't dominate
        #    w_adversarial = 10.0  # Push for quality!
        #    phase = "IMPROVING_QUALITY"
        #    
        #else:
        #    # Phase 3: Balance both
        #    w_diversity = 10.0
        #    w_adversarial = 5.0
        #    phase = "BALANCED"
      
        # Phase detection
        if avg_diversity < self.MIN_DIVERSITY * 0.8:
            # Phase 1: Need diversity desperately
            w_diversity = 50.0
            w_adversarial = 1.0
            w_mi = 0.0  # Not yet
            phase = "BOOTSTRAPPING_DIVERSITY"
            
        elif avg_diversity >= self.MIN_DIVERSITY and std_diversity > 0.002 and avg_fitness_adv < -0.85:
            # Phase 2: Have diversity, need quality
            w_diversity = 5.0
            w_adversarial = 10.0
            w_mi = 0.0  # Still not yet
            phase = "IMPROVING_QUALITY"
        
        elif avg_diversity >= self.MIN_DIVERSITY and avg_fitness_adv >= -0.85:
            # Phase 2b: Quality good, now add structure  
            w_diversity = 5.0  # Maintain
            w_adversarial = 5.0  # Maintain
            w_mi = 2.0  # ADD MI NOW
            phase = "ADDING_STRUCTURE"
            
        else:
            # Phase 3: Refinement
            w_diversity = 5.0
            w_adversarial = 10.0
            w_mi = 3.0
            phase = "REFINEMENT"

        if self._t < 400:
            w_diversity = 2.0
            w_adversarial = 1.0
            w_mi = 1.0
            w_con = 1.0
            w_r_cons = 2.0
            w_r_sense = 1.0
            w_r_intra = 0.5
        elif self._t < 10000:
            w_diversity = 4.0
            w_adversarial = 10.0
            w_mi = 0.1
            w_con = 1.0
            w_r_cons = 2.0
            w_r_sense = 1.4
            w_r_intra = 0.6
        elif self._t < 18000:
            w_diversity = 3.0
            w_adversarial = 10.0
            w_mi = 0.1
            w_con = 2.0
            w_r_cons = 1.4
            w_r_sense = 2.0
            w_r_intra = 0.4
        elif self._t < 30000:
            w_diversity = 2.0
            w_adversarial = 10.0
            w_mi = 0.1
            w_con = 2.0
            w_r_cons = 1.0
            w_r_sense = 1.6
            w_r_intra = 1.0
        else:
            w_diversity = 1.0
            w_adversarial = 20.0
            w_mi = 1.0
            w_con = 6.0
            w_r_cons = 1.4
            w_r_sense = 2.0
            w_r_intra = 1.0
        # print('Current Phase: {} ', phase)
        #w_adv = 6.0
        #w_div = 1.0
        #w_mi = 30.0
        # Apply weights
        #fitness_scores = fitness_adv.flatten() * w_adv + pop_var * w_div + fitness_mi.flatten() * w_mi + fitness_con.flatten() 
        # Apply weights
        #if self._t < 200:
        #    fitness_scores = fitness_mi.flatten()
        #elif self._t < 10000
        #else:
        #if self._t < 30000:
        #    fitness_scores = fitness_adv.flatten() * w_adversarial + pop_var * w_diversity + fitness_mi.flatten() * w_mi + fitness_con.flatten()*w_con + r_cons*w_r_cons + r_sense*w_r_sense #+ r_intra*w_r_intra#- penalty
        #elif self._t < 40000:
        #    fitness_scores = norm_fitness_adv.flatten() + norm_fitness_mi.flatten() + norm_fitness_con.flatten()*0.2
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() + fitness_con.flatten()
        #else:
        #    fitness_scores = fitness_adv.flatten() * w_adversarial + pop_var * w_diversity + fitness_mi.flatten() * w_mi + fitness_con.flatten()*w_con + r_cons*w_r_cons + r_sense*w_r_sense

        #fitness_scores = fitness_adv.flatten() + pop_var * 20 - penalty
        if self._t > 18000:
            fitness_scores = -jnp.argsort(order+1)
        else:
            fitness_scores = fitness_adv.flatten() * w_adversarial + pop_var * w_diversity + fitness_mi.flatten() * w_mi + fitness_con.flatten()*w_con + r_cons*w_r_cons + r_sense*w_r_sense #+ r_intra*w_r_intra#- penalty
            

        fitness_scores, self._best_score, self._avg_score = process_scores(fitness_scores,True)

        grad_center, grad_stdev = compute_reinforce_update(
                fitness_scores=fitness_scores,
                scaled_noises=self._scaled_noises,
                stdev=self._stdev,
            )
        
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

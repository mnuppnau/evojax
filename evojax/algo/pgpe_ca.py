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
def mi_penalty(fitness_mi, fitness_adv, mi_thr=-0.01, v_ref=0.01, M=5.0, eps=1e-8):
    # violation
    v = jnp.maximum(0.0, mi_thr - fitness_mi)

    # scale from current population
    scale = jnp.std(fitness_adv) + eps

    # penalty weight
    lam = (M * scale) / (v_ref + eps)

    return lam * v

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
        decay_coef = optimizer_config.get("center_lr_decay_coef", 1.0)
        self._lr_decay_steps = optimizer_config.get(
            "center_lr_decay_steps", 90000
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
                momentum=optimizer_config.get("momentum", 0.9),
                max_speed=optimizer_config.get("max_speed", 0.08),
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

        self._key, subkey = random.split(self._key)
       
        self.belief_space = belief_space if belief_space is not None else initialize_belief_space(
            population_size=self.pop_size, param_size=abs(param_size), key=subkey)

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


    def tell(self, fitness_adv: Union[np.ndarray, jnp.ndarray], fitness_mi: Union[np.ndarray, jnp.ndarray],fitness_con: Union[np.ndarray, jnp.ndarray], disc_logits: Union[np.ndarray, jnp.ndarray], pop_var: Union[np.ndarray, jnp.ndarray], avg_per_code: Union[np.ndarray, jnp.ndarray], r_cons: Union[np.ndarray, jnp.ndarray], r_sense: Union[np.ndarray, jnp.ndarray], r_intra: Union[np.ndarray, jnp.ndarray], r_anchor: Union[np.ndarray, jnp.ndarray], mmd: Union[np.ndarray, jnp.ndarray], adv: bool) -> None:

        
        avg_r_anchor = jnp.mean(r_anchor)

        #if avg_r_anchor < 0.0009:
        #    w_r_anchor = 1000.0
        #elif avg_r_anchor < 0.009:
        #    w_r_anchor = 100.0
        #elif avg_r_anchor < 0.09:
        #    w_r_anchor = 10.0
        #else:
        #    w_r_anchor = 2.0
        
        # add a dimension to the fitness scores so that (256,) becomes (256, 1)
        fitness_adv = fitness_adv[:, None]
        fitness_mi = fitness_mi[:, None]
        fitness_con = fitness_con[:, None]

        r_cons2 = r_cons[:, None]
        r_sense2 = r_sense[:, None]
        r_intra2 = r_intra[:, None]
        pop_var2 = pop_var[:, None]

        mmd2 = mmd[:, None]
        mmd_max = jnp.max(mmd2)
        fitness_adv_max = jnp.max(fitness_adv)
        fitness_mi_max = jnp.max(fitness_mi)
        fitness_con_max = jnp.max(fitness_con)
        #penalty = jnp.where(
        #     pop_var < self.MIN_DIVERSITY,
        #     (self.MIN_DIVERSITY - pop_var) * 100,  # Heavy penalty if below threshold
        #     0.0  # No penalty if above threshold
        #)
        #jax.debug.print('fitness adv scores {} : ', fitness_adv.flatten())
        #objectives = jnp.hstack([-fitness_adv])
        if self._t < 70000:
            objectives = jnp.hstack([-fitness_adv])
        #elif self._t < 1000:
        #    objectives = jnp.hstack([-fitness_adv,])
        elif self._t < 80000:
            objectives = jnp.hstack([-fitness_adv, -fitness_con])
        elif self._t < 190000 and fitness_mi_max < -0.001 and fitness_con_max < -0.005:
            objectives = jnp.hstack([-fitness_adv, -fitness_mi, -fitness_con])
        elif self._t < 190000 and fitness_con_max < -0.005:
            objectives = jnp.hstack([-fitness_adv, -fitness_con])
        elif self._t < 190000 and fitness_mi_max < -0.001:
            objectives = jnp.hstack([-fitness_adv, -fitness_mi])
        else:
            objectives = jnp.hstack([-fitness_adv])
        ##elif mmd_max < 2.03 and fitness_mi_max < -0.02:
        ##    objectives = jnp.hstack([-fitness_adv, -fitness_mi, -mmd2])
        ##elif mmd_max < 2.03:
        ##    objectives = jnp.hstack([-fitness_adv, -fitness_con, -mmd2])
        #elif fitness_mi_max < -0.01 and fitness_con_max < -0.05:
        #    objectives = jnp.hstack([-fitness_adv, -fitness_mi, -fitness_con])
        #elif fitness_con_max < -0.03:
        #    objectives = jnp.hstack([-fitness_adv, -fitness_con])
        #elif fitness_mi_max < -0.005:
        #    objectives = jnp.hstack([-fitness_adv, -fitness_mi])
        #else:
        #    objectives = jnp.hstack([-fitness_adv])
        #elif self._t < 40000 and mmd_max < 2.03:
        #    objectives = jnp.hstack([-fitness_mi, -fitness_adv, -mmd2])
        #elif self._t < 40000:
        #    objectives = jnp.hstack([-fitness_mi, -fitness_adv])
        #elif self._t < 50000 and mmd_max < 2.03:
        #    objectives = jnp.hstack([-r_sense2, -fitness_adv, -mmd2])
        #elif self._t < 50000:
        #    objectives = jnp.hstack([-r_sense2, -fitness_adv, -fitness_mi])
        #elif self._t < 60000 and mmd_max < 2.03:
        #    objectives = jnp.hstack([r_cons2, -fitness_adv, -mmd2])
        #elif self._t < 60000:
        #    objectives = jnp.hstack([r_cons2, -fitness_adv, -fitness_mi])
        #elif self._t < 70000 and mmd_max < 2.03:
        #    objectives = jnp.hstack([-r_sense2, -mmd2, -fitness_adv])
        #elif self._t < 70000:
        #    objectives = jnp.hstack([-r_sense2, -fitness_mi, -fitness_adv])
        ##
        #elif mmd_max < 2.03: 
        #    objectives = jnp.hstack([-mmd2, -fitness_adv, -fitness_con])
        #else:
        #    objectives = jnp.hstack([-fitness_mi, -fitness_adv, -fitness_con])
            #    objectives = jnp.hstack([r_cons2, -r_intra2, -fitness_adv])

        ranks = non_dominated_sort_lax(objectives)
       
        order = jnp.lexsort((-fitness_adv.flatten(), ranks))


        #avg_fitness_adv = jnp.mean(fitness_adv)
        #avg_fitness_mi = jnp.mean(fitness_mi)

        #w_adv = avg_fitness_adv / (avg_fitness_adv + avg_fitness_mi)
        #w_adv = jnp.clip(w_adv, 0.3, 0.7)
        #w_mi = 1 - w_adv


        self.belief_space = update_topographic_ks(
            self.belief_space, avg_per_code
        )
        
        #    fitness_scores = fitness_adv.flatten() * w_adversarial + pop_var * w_diversity + fitness_mi.flatten() * w_mi + fitness_con.flatten()*w_con + r_cons*w_r_cons + r_sense*w_r_sense

        #fitness_scores = fitness_adv.flatten() + pop_var * 20 - penalty
        #if self._t < 800:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() * 100#-jnp.argsort(order+1)
        #elif self._t < 2000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() * 60
        #elif self._t < 7000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten()
        #elif self._t < 9000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() * 200 + r_sense * 10
        #elif self._t < 12000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() * 10
        #elif self._t < 16000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() * 400 + r_sense * 10 - r_cons * 8
        #elif self._t < 22000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() * 400 + r_sense * 20 - r_cons * 10
        #elif self._t < 44000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() * 1000 - r_cons * 20 + fitness_con.flatten()*10 
        #elif self._t < 60000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() * 1000 + r_sense * 20 + fitness_con.flatten()*20
        #elif self._t < 80000:
        #    fitness_scores = fitness_adv.flatten() + fitness_mi.flatten() * 1000 + r_sense * 40 + fitness_con.flatten()*100
        #else:
        #    fitness_scores = fitness_adv.flatten()*0.1 + fitness_mi.flatten() * 1000 - r_cons * 20 + r_sense * 10 + r_intra
        #    
        #if jnp.mean(fitness_mi) < -0.4:
        #    w_mi = 10
        #elif jnp.mean(fitness_mi) < -0.2:
        #    w_mi = 20
        #else:
        if self._t < 3000: 
            w_mi = 40
        elif self._t < 6000:
            w_mi = 30
        elif self._t < 9000:
            w_mi = 20
        elif self._t < 25000:
            w_mi = 10
        else:
            w_mi = 100
       
        if self._t < 6000:
            w_adv = 1.0
        elif self._t < 8000:
            w_adv = 0.95
        elif self._t < 14000:
            w_adv = 0.9
        elif self._t < 30000:
            w_adv = 0.85
        elif self._t < 50000:
            w_adv = 0.9
        elif self._t < 70000:
            w_adv = 0.95
        else:
            w_adv = 1.0

        if self._t < 30000:
            w_sense = 2.0
            w_intra = 4.0
            w_cons = 1.0
            w_con = 2.0
        elif self._t < 50000:
            w_sense = 2.0
            w_intra = 2.0
            w_cons = 1.6
            w_con = 2.0
        elif self._t < 80000:
            w_sense = 3.0
            w_intra = 1.0
            w_cons = 2.0
            w_con = 2.0
        elif self._t < 160000:
            w_sense = 4.0
            w_intra = 1.0
            w_cons = 2.0
            w_con = 2.0
        elif self._t < 200000:
            w_sense = 3.0
            w_intra = 1.0
            w_cons = 1.0
            w_con = 2.0
        elif self._t < 240000:
            w_sense = 4.0
            w_intra = 1.0
            w_cons = 2.0
            w_con = 2.5
        else:
            w_sense = 8.0
            w_intra = 4.0
            w_cons = 3.0
            w_con = 3.0


        if self._t < 10000:
            w_realism = 0.6
        elif self._t < 30000:
            w_realism = 1.2
        elif self._t < 60000:
            w_realism = 1.8
        else:
            w_realism = 2.6
        
        raw_fitness_adv = fitness_adv.flatten() #* w_adv
        fitness_adv = raw_fitness_adv # * w_adv
        
        std_adv = jnp.std(fitness_adv) + 1e-8
        std_mi = jnp.std(fitness_mi) + 1e-8


        max_w_mi = jnp.minimum(100, jnp.maximum((std_adv / std_mi)//2,6))

        realism_gate = jax.nn.sigmoid(raw_fitness_adv + w_realism)
        # 2. MI is a constraint. If MI loss is high, it dominates fitness.
        # If MI loss is low (good), its gradient contribution diminishes.
        #mi_target = -0.001
        #mi_gap = jnp.minimum(fitness_mi.flatten() - mi_target, 0.00)
        
        #fitness_mi = w_mi * mi_gap
        #fitness_mi = 5 * fitness_mi.flatten()

        penalty_mi = mi_penalty(fitness_mi.flatten(), fitness_adv.flatten(), mi_thr=-0.01, v_ref=0.01, M=5.0)
        r_sense = w_sense * jnp.minimum(r_sense, 0.68)

        r_cons_weight = jnp.clip( (self._t - 3000) / 3000 , 0.0, 1.0)

        #r_cons_penalty = 2 * jnp.maximum(r_cons, 0.1)
      
        r_cons = -1.0 * r_cons_weight * r_cons#r_cons_penalty
        r_cons = w_cons * r_cons

        r_intra = w_intra * r_intra

        fitness_con = w_con * fitness_con.flatten()

        w_r_anchor = 10.0

        if self._t < 140000:
            fitness_scores = -jnp.argsort(order)
            #fitness_scores = fitness_adv + fitness_mi.flatten() * max_w_mi + mmd2.flatten()*0.02 + fitness_con.flatten()*0.1
        else:#if self._t < 160000:
            cultural_score = r_sense + r_intra + r_cons + fitness_con
            fitness_scores = fitness_adv + realism_gate * cultural_score - penalty_mi
        #else:
        #    cultural_score = r_sense + r_intra + r_cons + fitness_con
        #    fitness_scores = fitness_adv + realism_gate * cultural_score - penalty_mi - (r_anchor * w_r_anchor)
        #fitness_scores = fitness_adv + fitness_mi + r_sense + fitness_con.flatten()*0.6 + r_intra + r_cons

        #fitness_scores = -jnp.argsort(order+1)
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

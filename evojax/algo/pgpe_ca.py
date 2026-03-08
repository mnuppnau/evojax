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
    update_domain_ks,
    update_situational_ks,
    update_history_ks,
    update_normative_ks,
    update_metric_history,
    compute_metric_slopes,
)

from evojax.algo.cultural.helper_functions import non_dominated_sort_lax
from evojax.algo.cultural.helper_functions import (
    situational_score,
    historical_score,
    topographic_score,
    domain_score,
)
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
def standardize(x):
    return (x - jnp.mean(x)) / jnp.maximum(jnp.std(x), 1e-6)

@jax.jit
def rank_normalize(x):
    """Rank-based fitness shaping: maps values to [-1, 1] by rank order.
    Robust to near-zero variance (unlike standardize which amplifies noise)."""
    n = x.shape[0]
    ranks = jnp.argsort(jnp.argsort(x)).astype(jnp.float32)
    return 2.0 * ranks / jnp.maximum(n - 1, 1) - 1.0

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
        ca_blend_coeff: float = 0.05,
        ca_blend_start_iter: int = 0,
        ca_blend_ramp_iters: int = 1,
        ca_blend_rfl_lo: float = -1.0,
        ca_blend_rfl_hi: float = -1.0,
        shape_div_weight: float = 0.12,
        static_fitness_weights: bool = False,
        static_mi_sense_ramp: bool = False,
        static_w_adv: Optional[float] = None,
        static_w_mi: Optional[float] = None,
        static_w_div: Optional[float] = None,
        static_w_sense: Optional[float] = None,
        static_w_intra: Optional[float] = None,
        static_div_ramp_target: Optional[float] = None,
        static_div_ramp_start_iter: int = -1,
        static_div_ramp_end_iter: int = -1,
        static_sense_ramp_target: Optional[float] = None,
        static_sense_ramp_start_iter: int = -1,
        static_sense_ramp_end_iter: int = -1,
        static_intra_ramp_target: Optional[float] = None,
        static_intra_ramp_start_iter: int = -1,
        static_intra_ramp_end_iter: int = -1,
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
            ca_blend_coeff - Maximum CA gradient blend ratio when CA data is valid.
            ca_blend_start_iter - Iteration to start ramping CA blend.
            ca_blend_ramp_iters - Ramp length for CA blend schedule.
            ca_blend_rfl_lo - Optional lower bound of real_fake_loss gate.
            ca_blend_rfl_hi - Optional upper bound of real_fake_loss gate.
            shape_div_weight - Weight for conditional shape diversity reward.
            static_fitness_weights - Disable slope/health-driven adaptive
                                     weighting and use fixed fitness weights.
            static_mi_sense_ramp - When static_fitness_weights=True, optionally
                                   keep the legacy time-ramp on MI/sense bases.
            static_w_* - Optional static fitness weights used when
                         static_fitness_weights=True.
            static_div_ramp_target - Optional late-training target for the
                                     static diversity weight.
            static_div_ramp_start_iter - Absolute iteration where the late
                                         static diversity ramp begins.
            static_div_ramp_end_iter - Absolute iteration where the late
                                       static diversity ramp reaches target.
            static_sense_ramp_target - Optional late-training target for the
                                       static sense weight. When set, this
                                       linearly interpolates from
                                       static_w_sense to this target.
            static_sense_ramp_start_iter - Absolute iteration where the late
                                           static sense ramp begins.
            static_sense_ramp_end_iter - Absolute iteration where the late
                                         static sense ramp reaches target.
            static_intra_ramp_target - Optional late-training target for the
                                       static intra-code weight.
            static_intra_ramp_start_iter - Absolute iteration where the late
                                           static intra ramp begins.
            static_intra_ramp_end_iter - Absolute iteration where the late
                                         static intra ramp reaches target.
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
        self._ca_blend_coeff = float(max(ca_blend_coeff, 0.0))
        self._ca_blend_start_iter = int(max(ca_blend_start_iter, 0))
        self._ca_blend_ramp_iters = int(max(ca_blend_ramp_iters, 1))
        self._ca_blend_rfl_lo = float(ca_blend_rfl_lo)
        self._ca_blend_rfl_hi = float(ca_blend_rfl_hi)
        self._shape_div_weight = float(max(shape_div_weight, 0.0))
        self._static_fitness_weights = bool(static_fitness_weights)
        self._static_mi_sense_ramp = bool(static_mi_sense_ramp)
        self._static_w_adv = None if static_w_adv is None else float(static_w_adv)
        self._static_w_mi = None if static_w_mi is None else float(static_w_mi)
        self._static_w_div = None if static_w_div is None else float(static_w_div)
        self._static_w_sense = None if static_w_sense is None else float(static_w_sense)
        self._static_w_intra = None if static_w_intra is None else float(static_w_intra)
        self._static_div_ramp_target = (
            None if static_div_ramp_target is None else float(static_div_ramp_target)
        )
        self._static_div_ramp_start_iter = int(static_div_ramp_start_iter)
        self._static_div_ramp_end_iter = int(static_div_ramp_end_iter)
        self._static_sense_ramp_target = (
            None if static_sense_ramp_target is None else float(static_sense_ramp_target)
        )
        self._static_sense_ramp_start_iter = int(static_sense_ramp_start_iter)
        self._static_sense_ramp_end_iter = int(static_sense_ramp_end_iter)
        self._static_intra_ramp_target = (
            None if static_intra_ramp_target is None else float(static_intra_ramp_target)
        )
        self._static_intra_ramp_start_iter = int(static_intra_ramp_start_iter)
        self._static_intra_ramp_end_iter = int(static_intra_ramp_end_iter)
        self._runtime_real_fake_loss = np.nan
        self._debug_last = {}
        self._ks_winner_counts = np.zeros((4,), dtype=np.int64)

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
        #self._stdev_max_change = abs(stdev_max_change)
        self._stdev_max_change = 0.1
        self._solution_ranking = solution_ranking

        if optimizer_config is None:
            optimizer_config = {}
        decay_coef = optimizer_config.get("center_lr_decay_coef", 1.0)
        self._lr_decay_coef = decay_coef
        self._lr_decay_steps = optimizer_config.get(
            "center_lr_decay_steps", 100000
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

        self._arr = jnp.arange(11)
        self._key = random.PRNGKey(seed=seed)
        self._key, self._subkey = random.split(self._key)
        self._solutions = None
        self._scaled_noises = None

        self._key, subkey = random.split(self._key)
       
        self.belief_space = belief_space if belief_space is not None else initialize_belief_space(
            population_size=self.pop_size, param_size=abs(param_size), key=subkey)

    def set_runtime_metrics(self, real_fake_loss: Optional[float] = None) -> None:
        """Pass training-loop runtime metrics into PGPE_CA adaptive controls."""
        if real_fake_loss is not None:
            self._runtime_real_fake_loss = float(real_fake_loss)

    def get_diagnostics(self) -> dict:
        """Return lightweight runtime diagnostics for structured logging."""
        diag = {}
        for key, value in self._debug_last.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                diag[key] = float(arr)
            else:
                diag[key] = arr.astype(np.float64)
        diag["ks_winner_counts"] = self._ks_winner_counts.copy()
        return diag

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
        center, stdev = self._center, self._stdev

        self._key, self._scaled_noises, self._solutions = ask_func(
            self._key,
            stdev,
            center,
            self._num_directions,
            self._center.size,
        )

        return self._solutions, self.belief_space


    def tell(self, fitness_adv: Union[np.ndarray, jnp.ndarray], fitness_mi: Union[np.ndarray, jnp.ndarray], disc_logits: Union[np.ndarray, jnp.ndarray], pop_var: Union[np.ndarray, jnp.ndarray], avg_per_code: Union[np.ndarray, jnp.ndarray], r_cons: Union[np.ndarray, jnp.ndarray], r_sense: Union[np.ndarray, jnp.ndarray], r_intra: Union[np.ndarray, jnp.ndarray], r_shape_div: Union[np.ndarray, jnp.ndarray], r_shape_div_min: Union[np.ndarray, jnp.ndarray], morph_dark_range: Union[np.ndarray, jnp.ndarray], morph_center_edge_range: Union[np.ndarray, jnp.ndarray], edge_dark_frac: Union[np.ndarray, jnp.ndarray], code_proto_corr: Union[np.ndarray, jnp.ndarray], normative_penalty: Union[np.ndarray, jnp.ndarray], safety_ratios: Union[np.ndarray, jnp.ndarray], spreads: Union[np.ndarray, jnp.ndarray], adv: bool) -> None:

       
        #if avg_r_anchor < 0.0009:
        #    w_r_anchor = 1000.0
        #elif avg_r_anchor < 0.009:
        #    w_r_anchor = 100.0
        #elif avg_r_anchor < 0.09:
        #    w_r_anchor = 10.0
        #else:
        #    w_r_anchor = 2.0
        
        # add a dimension to the fitness scores so that (256,) becomes (256, 1)
        #fitness_adv = fitness_adv[:, None]
        #fitness_mi = fitness_mi[:, None]
        #fitness_con = fitness_con[:, None]

        #r_cons2 = r_cons[:, None]
        #r_sense2 = r_sense[:, None]
        #r_intra2 = r_intra[:, None]
        #r_anchor2 = r_anchor[:, None]
        #pop_var2 = pop_var[:, None]

        #mmd2 = mmd[:, None]
        #mmd_max = jnp.max(mmd2)
        #fitness_adv_max = jnp.max(fitness_adv)
        #fitness_mi_max = jnp.max(fitness_mi)
        #fitness_con_max = jnp.max(fitness_con)
        #penalty = jnp.where(
        #     pop_var < self.MIN_DIVERSITY,
        #     (self.MIN_DIVERSITY - pop_var) * 100,  # Heavy penalty if below threshold
        #     0.0  # No penalty if above threshold
        #)
        #jax.debug.print('fitness adv scores {} : ', fitness_adv.flatten())
        #objectives = jnp.hstack([-fitness_adv])
        #if self._t < 40000:
        #objectives = jnp.hstack([-fitness_adv,-fitness_mi])
        ##elif self._t < 1000:
        ##    objectives = jnp.hstack([-fitness_adv,])
        ##elif self._t < 99000:
        ##    objectives = jnp.hstack([-fitness_adv, -fitness_con])
        #elif self._t < 190000 and fitness_mi_max < -0.0001 and fitness_con_max < -0.005:
        #    objectives = jnp.hstack([-fitness_adv, -fitness_mi, -fitness_con])
        #elif self._t < 190000 and fitness_con_max < -0.005:
        #    objectives = jnp.hstack([-fitness_adv, -r_sense2, -fitness_con])
        #elif self._t < 190000 and fitness_mi_max < -0.0001:
        #    objectives = jnp.hstack([-fitness_adv, -fitness_mi])
        #else:
        #    objectives = jnp.hstack([-fitness_adv,-r_sense2, -r_cons2])
        
        #if self._t % 10 == 0:
        #    objectives = jnp.hstack([-fitness_adv, -fitness_mi, -r_intra2])

        ##elif mmd_max < 2.03 and fitness_mi_max < -0.02:
        #if not adv:
        #    objectives = jnp.hstack([-fitness_adv])
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

        #ranks = non_dominated_sort_lax(objectives)
       
        #order = jnp.lexsort((-fitness_adv.flatten(), ranks))


        #avg_fitness_adv = jnp.mean(fitness_adv)
        #avg_fitness_mi = jnp.mean(fitness_mi)

        #w_adv = avg_fitness_adv / (avg_fitness_adv + avg_fitness_mi)
        #w_adv = jnp.clip(w_adv, 0.3, 0.7)
        #w_mi = 1 - w_adv


        # CA activation schedule: short warmup for KS to fill (500 iter),
        # then ramp to full over 2000 iter. Active early so CA can adapt
        # to the environment from the start rather than arriving late.
        ca_weight = jnp.clip((self._t - 500) / 2000, 0.0, 1.0)

        # Centroid momentum: when CA is active, nearly freeze topographic centroids
        # to prevent locked codes from drifting. 0.7 (early) → 0.97 (full CA).
        topo_momentum = 0.7 + 0.27 * ca_weight

        self.belief_space = update_topographic_ks(
            self.belief_space, avg_per_code, topo_momentum
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
        #if self._t < 1000: 
        #    w_mi = 100
        #elif self._t < 3000:
        #    w_mi = 20
        #elif self._t < 9000:
        #    w_mi = 4
        #elif self._t < 35000:
        #    w_mi = 2
        #elif self._t < 70000:
        #    w_mi = 10
        #else:
        #    w_mi = 100
      
        # linearly ramp w_mi from 1.0 to 1000.0 over 200000 timesteps
        #w_mi = jnp.clip((self._t / 200000) * 200.0, 0.1, 200.0)
        #if self._t < 6000:
        #    w_adv = 1.0
        #elif self._t < 8000:
        #    w_adv = 0.95
        #elif self._t < 14000:
        #    w_adv = 0.9
        #elif self._t < 30000:
        #    w_adv = 0.85
        #elif self._t < 50000:
        #    w_adv = 0.9
        #elif self._t < 70000:
        #    w_adv = 0.95
        #else:
        #    w_adv = 1.0

        #if self._t < 30000:
        #    w_sense = 4.0
        #    w_intra = 1.0
        #    w_cons = 0.2
        #    w_con = 3.0
        #elif self._t < 50000:
        #    w_sense = 2.0
        #    w_intra = 1.0
        #    w_cons = 0.8
        #    w_con = 2.0
        #elif self._t < 80000:
        #    w_sense = 3.0
        #    w_intra = 2.0
        #    w_cons = 1.6
        #    w_con = 2.0
        #elif self._t < 160000:
        #    w_sense = 4.0
        #    w_intra = 3.0
        #    w_cons = 1.0
        #    w_con = 3.0
        #elif self._t < 200000:
        #    w_sense = 3.0
        #    w_intra = 1.0
        #    w_cons = 1.0
        #    w_con = 2.0
        #elif self._t < 240000:
        #    w_sense = 4.0
        #    w_intra = 2.0
        #    w_cons = 2.0
        #    w_con = 2.5
        #else:
        #    w_sense = 8.0
        #    w_intra = 4.0
        #    w_cons = 3.0
        #    w_con = 3.0


        #if self._t < 20000:
        #    w_realism = -0.8
        #elif self._t < 40000:
        #    w_realism = 0.3
        #elif self._t < 100000:
        #    w_realism = 0.5
        #else:
        #    w_realism = 0.8
        
        #raw_fitness_adv = fitness_adv #* w_adv
        #fitness_adv = raw_fitness_adv # * w_adv
        
        #std_adv = jnp.std(fitness_adv) + 1e-8
        #std_mi = jnp.std(fitness_mi) + 1e-8


        #max_w_mi = jnp.minimum(2.0, jnp.maximum((std_adv / std_mi)//2,0.05))

        #realism_gate = jax.nn.sigmoid(raw_fitness_adv + w_realism)
        # 2. MI is a constraint. If MI loss is high, it dominates fitness.
        # If MI loss is low (good), its gradient contribution diminishes.
        #mi_target = -0.001
        #mi_gap = jnp.minimum(fitness_mi.flatten() - mi_target, 0.00)
        
        #fitness_mi = w_mi * mi_gap
        #fitness_mi = 5 * fitness_mi.flatten()

        #penalty_mi = mi_penalty(fitness_mi.flatten(), fitness_adv.flatten(), mi_thr=-0.001, v_ref=0.01, M=5.0)
        #r_sense = w_sense * jnp.minimum(r_sense, 0.68)

        #r_cons_weight = jnp.clip( (self._t - 3000) / 3000 , 0.0, 1.0)
        ##r_cons_penalty = 2 * jnp.maximum(r_cons, 0.1)
        #r_cons = -1.0 * r_cons_weight * r_cons#r_cons_penalty
        #r_cons = w_cons * r_cons
        #r_cons = -4.0 - r_cons

        #r_intra = w_intra * r_intra

        #r_intra = -4.4 + r_intra
        #r_intra = 0.1 * r_intra
        #r_sense = -4.0 + r_sense
         
        #fitness_con = w_con * fitness_con.flatten()

        #fitness_adv_mean = jnp.mean(fitness_adv)
        #fitness_adv_std = jnp.std(fitness_adv)
        #fitness_adv_norm = (fitness_adv - fitness_adv_mean) / (fitness_adv_std + 1e-8)

        #r_anchor_mean = jnp.mean(r_anchor)
        #r_anchor_std = jnp.std(r_anchor)
        #r_anchor_norm = (r_anchor - r_anchor_mean) / (r_anchor_std + 1e-8)

        #fitness_adv = 10.0 - raw_fitness_adv
        #fitness_con = -6.0 - fitness_con 
        #fitness_adv = fitness_adv_norm + (0.2 * r_anchor_norm) 
        #jax.debug.print('realism gate shape {} : ', realism_gate.shape)
        #jax.debug.print('r_sense shape {} : ', r_sense.shape)
        #jax.debug.print('r_intra shape {} : ', r_intra.shape)
        #jax.debug.print('r_cons shape {} : ', r_cons.shape)
        #jax.debug.print('fitness_con shape {} : ', fitness_con.shape)
        
        #cultural_score = (r_sense - r_cons)
        #cultural_score = cultural_score[:, None]
        #jax.debug.print('cultural score shape {} : ', cultural_score.shape)


        #diversity_score = fitness_adv + (realism_gate * cultural_score)
        #diversity_score = diversity_score[:, None] 

        #jax.debug.print('diversity score shape {} : ', diversity_score.shape)

        #fitness_adv = fitness_adv[:, None]
        
        #if self._t % 2 == 0:
        #    objectives_final = jnp.hstack([-fitness_adv, -fitness_mi, -cultural_score])
        #else:
        #    objectives_final = jnp.hstack([-fitness_adv, -fitness_mi])
        #jax.debug.print('objectives final shape {} : ', objectives_final.shape)

        #ranks_final = non_dominated_sort_lax(objectives_final)
        #w_mi = 100.0
        #w_adv = 10.0
        #order = jnp.lexsort((-fitness_adv.flatten(), ranks_final))
        #if self._t < 40000:
        #fitness_scores = -fitness_adv
        #elif self._t < 40000:
            #    
        # increase w_norm from 0.01 to 1.0 linearly over 20000 iterations
        #w_norm = jnp.clip((self._t / 200000) * 1.0, 0.01, 1.0)
        

        # 1. Define the Schedule
        # ramp_start: 140k. ramp_end: 160k.
        # We fade CA in over 20k iterations so we don't shock the population.
        # Inside your fitness function or step_fn
        # 1. Calculate raw components
        
        #ca_weight = jnp.clip((self._t - 140000) / 20000, 0.0, 1.0)
        
        # 2. Define the Metric Weights
        #w_adv = 0.2
        ##w_mi = 0.28
        #w_mi = 0.1
        #w_con = 0.008
        #w_sense = 1.2 
        # CA Weights (Only active after 140k)
        #w_sense = 0.1 * ca_weight       # Reward separation
        #w_cons = 0.05 * ca_weight       # Penalize drift
        #w_norm = 0.05 * ca_weight       # Penalize violation (Keep this small!)
        # gradually warm up w_sense from 0.0 to 1.2 over 1k iterations starting at 180k
        #phase2_start = 180000
        #w_sense = jnp.clip((self._t - 180000) / 1000, 0.0, 1.2) 
        #w_cons = 0.0
        #w_norm = 0.0
        # Updated Weight Schedule (Gentler)
        
        # 1. Repulsion (w_sense): Decay SLOWLY. 
        # Don't drop to 0.2 yet. The "8" needs the pressure from the "6" and "2" to stay an "8".
        # Hold at 1.0 for 2k steps, then decay.
        #w_sense = jnp.clip(1.2 - ((self._t - 183000) / 100) * 0.8, 0.4, 1.2)
        
        # 2. Anchor (w_cons): Cap at 2.0 (Not 20.0!)
        # We want to prevent drift, not freeze evolution.
        #w_cons = jnp.clip((self._t - 183000) / 100, 0.0, 1.0)
        #
        ## 3. Normative (w_norm): Keep low but active
        #w_norm = jnp.clip((self._t - 183000) / 100, 0.0, 2.0)
       ## 3. Calculate Fitness
        ## Note: Ensure signs are correct (Subtracting penalties)
        #fitness_scores = (
        #    (fitness_adv * w_adv)
        #    + (fitness_mi * w_mi)
        #    + (fitness_con * w_con)
        #    + (r_sense * w_sense)             # Stage 2: Push clusters apart
        #    #- (normative_penalty * w_norm)    # Stage 2: Enforce safety/spread limits
        #    #- (r_cons * w_cons)               # Stage 2: Anchor distinct digits
        #)
        # 2. Define the Metric Weights
        # Signal analysis at 183k (pop std): adv=0.0465, mi=0.0004, con=0.0198,
        # r_sense=0.0092, r_cons=0.0012, normative=0.0007
        # To give r_sense comparable gradient influence to fitness_adv:
        #   need w_sense * 0.0092 ≈ w_adv * 0.0465 → w_sense ≈ 5.0 * w_adv
        # With min-pair r_sense (higher variance ~0.015), w_sense ~3.0 suffices.
        # --- CA-driven adaptive weight modulation ---
        # Base weights (user-tuned sweet spot from the stable regime).
        # In static mode, these are fixed and do not ramp with iteration.
        w_adv_base = jnp.float32(self._static_w_adv if self._static_w_adv is not None else 0.53)
        # Health gates for MI/sense pressure:
        # - D health from real_fake_loss window
        # - Shape health from worst-code conditional shape diversity
        # When either degrades, pause/soften MI+separation escalation.
        rfl = jnp.array(self._runtime_real_fake_loss, dtype=jnp.float32)
        rfl_finite = jnp.isfinite(rfl)
        d_lo = jnp.float32(0.38)
        d_hi = jnp.float32(0.58)
        d_shoulder = jnp.float32(0.05)
        d_gate_lo = jnp.clip((rfl - (d_lo - d_shoulder)) / d_shoulder, 0.0, 1.0)
        d_gate_hi = jnp.clip(((d_hi + d_shoulder) - rfl) / d_shoulder, 0.0, 1.0)
        d_health = jnp.where(rfl_finite, jnp.minimum(d_gate_lo, d_gate_hi), 1.0)

        shape_min_pop = jnp.mean(r_shape_div_min)
        # Relaxed for current BloodMNIST regime so objective_health can open.
        shape_lo = jnp.float32(0.02)
        shape_hi = jnp.float32(0.05)
        shape_health = jnp.clip((shape_min_pop - shape_lo) / (shape_hi - shape_lo + 1e-8), 0.0, 1.0)

        # Combined health factor for MI/sense pressure.
        objective_health = d_health * shape_health
        ramp_gate = 0.25 + 0.75 * objective_health

        def _apply_static_ramp(anchor, target, start_iter, end_iter):
            if target is None:
                return anchor
            ramp_start = float(start_iter)
            ramp_end = float(end_iter)
            ramp_span = max(ramp_end - ramp_start, 1.0)
            phase = jnp.clip((self._t - ramp_start) / ramp_span, 0.0, 1.0)
            target_val = jnp.float32(target)
            return anchor + (target_val - anchor) * phase

        w_mi_base = jnp.float32(self._static_w_mi if self._static_w_mi is not None else 0.30)
        w_div_anchor = jnp.float32(self._static_w_div if self._static_w_div is not None else 0.16)
        w_sense_anchor = jnp.float32(self._static_w_sense if self._static_w_sense is not None else 0.14)
        w_intra_anchor = jnp.float32(self._static_w_intra if self._static_w_intra is not None else 0.05)
        w_sense_base = w_sense_anchor
        if (not self._static_fitness_weights) or self._static_mi_sense_ramp:
            # Ramp adaptive bases in adaptive mode, or optionally in static
            # mode to reproduce legacy behavior.
            w_mi_base = w_mi_base + jnp.clip((self._t - 10000) / 50000 * 0.20, 0.00, 0.20) * ramp_gate
            if self._static_sense_ramp_target is None:
                w_sense_base = w_sense_base + jnp.clip((self._t - 10000) / 50000 * 0.20, 0.00, 0.20) * ramp_gate
        w_div_base = _apply_static_ramp(
            w_div_anchor,
            self._static_div_ramp_target,
            self._static_div_ramp_start_iter,
            self._static_div_ramp_end_iter,
        )
        w_sense_base = _apply_static_ramp(
            w_sense_base,
            self._static_sense_ramp_target,
            self._static_sense_ramp_start_iter,
            self._static_sense_ramp_end_iter,
        )
        w_intra_base = _apply_static_ramp(
            w_intra_anchor,
            self._static_intra_ramp_target,
            self._static_intra_ramp_start_iter,
            self._static_intra_ramp_end_iter,
        )
        w_norm_base = 0.04

        # Compute metric slopes from CA belief space
        slopes = compute_metric_slopes(self.belief_space)
        (adv_short, mi_short, adv_med, mi_med, ent_long,
         sense_short, intra_short, adv_avg_short, sense_med,
         shape_short, shape_med, spread_short, spread_med) = slopes

        # Mirror KS scoring used by the guidance functions for structured logs.
        ks_dom_score = domain_score(adv_med, mi_med, ent_long, spread_med)
        ks_sit_score = situational_score(adv_short, mi_short)
        ks_hist_score = historical_score(ent_long, adv_short)
        ks_topo_score = topographic_score(ent_long, adv_med, spread_short)
        ks_scores = jnp.array(
            [ks_dom_score, ks_sit_score, ks_hist_score, ks_topo_score],
            dtype=jnp.float32,
        )
        ks_weights_dbg = jax.nn.softmax(ks_scores / 2.0)

        # CA adaptation: adjust weights based on detected trends.
        # Positive slope = metric improving, Negative = metric declining.
        # Each adjustment is clamped to prevent runaway weight changes.
        if self._static_fitness_weights:
            # Post-fix baseline mode: fixed weights, no adaptive modulation.
            w_adv = jnp.float32(w_adv_base)
            w_mi = jnp.float32(w_mi_base)
            w_div = jnp.float32(w_div_base)
            w_sense = jnp.float32(w_sense_base)
            w_intra = jnp.float32(w_intra_base)
            w_shape = jnp.clip(jnp.float32(self._shape_div_weight), 0.02, 0.20)
            mi_guard = jnp.float32(0.0)
            w_cons_floor = jnp.float32(0.1)
            # Keep these variables defined for diagnostics consistency.
            adv_distress = jnp.float32(0.0)
            mi_distress = jnp.float32(0.0)
        else:
            # 1. If fitness_adv is dropping (D winning), boost w_adv, reduce diversity pressure
            #    adv_avg_short < 0 means avg adversarial fitness is declining
            adv_distress = jnp.clip(-adv_avg_short * 50.0, 0.0, 0.2)
            w_adv = w_adv_base + adv_distress
            w_div = w_div_base - adv_distress * 0.5  # ease off diversity when D is crushing

            # 2. If MI quality is stalling/declining, temporarily increase MI pressure.
            #    This is intentionally conservative: we only add up to +0.20.
            mi_distress_short = jnp.clip(-mi_short * 30.0, 0.0, 0.15)
            mi_distress_med = jnp.clip(-mi_med * 20.0, 0.0, 0.10)
            mi_distress = jnp.clip(mi_distress_short + mi_distress_med, 0.0, 0.20) * objective_health
            w_mi = w_mi_base + mi_distress
            # When MI is in distress, ease off pixel diversity slightly so pressure
            # shifts toward informative code alignment instead of texture variance.
            w_div = w_div - mi_distress * 0.25

            # 3. If r_sense is spiking too fast (divergence precursor), reduce w_sense
            #    sense_short > 0 means separation is increasing (good, but too fast = bad)
            sense_overshoot = jnp.clip(sense_short * 100.0 - 0.5, 0.0, 0.08)
            # If separation stays weak, give a modest sense boost.
            sense_target = 0.03
            sense_mean = jnp.mean(r_sense)
            sense_deficit = jnp.clip((sense_target - sense_mean) * 2.5, 0.0, 0.08) * objective_health
            sense_decline = jnp.clip(-sense_med * 25.0, 0.0, 0.05) * objective_health
            w_sense = w_sense_base - sense_overshoot + sense_deficit + sense_decline

            # 4. If r_intra is dropping (within-code variation collapsing), boost w_intra
            #    so PGPE rewards members that maintain within-code variety (use z-noise)
            intra_distress = jnp.clip(-intra_short * 100.0, 0.0, 0.15)
            w_intra = w_intra_base + intra_distress  # CA boosts when codes are tightening
            # 4b. Conditional shape-diversity pressure: increase when shape-div
            # trend is declining; keep strict caps to avoid destabilizing adv.
            shape_distress = jnp.clip(-shape_short * 30.0, 0.0, 0.10)
            shape_relief = jnp.clip(shape_short * 15.0, 0.0, 0.04)
            w_shape = self._shape_div_weight + shape_distress - shape_relief
            w_shape = jnp.clip(w_shape, 0.02, 0.20)

            # 5. r_cons floor penalty: penalize only when centroid consistency drops
            #    below a minimum threshold. This prevents drift without over-anchoring.
            cons_floor = 0.05
            cons_shortfall = jnp.maximum(0.0, cons_floor - r_cons)  # per-member penalty
            w_cons_floor = 0.1

            if self._t < 10000:
                w_mi_clip_max = 0.35
                w_sense_clip_max = 0.1
            else:
                w_mi_clip_max = 0.45
                w_sense_clip_max = 0.16

            # Ensure no weight goes negative
            w_adv = jnp.maximum(w_adv, 0.1)
            w_div = jnp.maximum(w_div, 0.02)
            # Extra damping under poor D/shape health to avoid shortcut collapse.
            w_mi = w_mi - (1.0 - objective_health) * 0.12
            w_sense = w_sense - (1.0 - objective_health) * 0.06
            w_mi = jnp.clip(w_mi, 0.05, w_mi_clip_max)
            w_sense = jnp.clip(w_sense, 0.02, w_sense_clip_max)

            # MI guard: when worst-code shape diversity collapses, reduce MI
            # pressure so optimization cannot improve MI by prototype collapse.
            shape_min_target = 0.10
            mi_guard = jnp.clip((shape_min_target - shape_min_pop) * 3.0, 0.0, 0.12)
            w_mi = jnp.clip(w_mi - mi_guard, 0.05, w_mi_clip_max)
            w_shape = jnp.clip(w_shape + (mi_guard * 0.5), 0.02, 0.20)

        # Normative: keep low
        w_norm = w_norm_base * ca_weight

        # Collapse-sensitive conditional diversity score:
        # prioritize the worst code while retaining global mean.
        shape_score = 0.7 * r_shape_div_min + 0.3 * r_shape_div

        # Morphology-aware terms (BloodMNIST):
        # Zeroed for post-alignment-fix baseline. Metrics still logged.
        w_dark_range = 0.0
        w_center_edge_range = 0.0
        w_edge_dark_penalty = 0.0
        edge_dark_target = 0.08
        edge_dark_excess = jnp.maximum(0.0, edge_dark_frac - edge_dark_target)
        # Anti-collapse: zeroed for baseline. Re-enable after characterization.
        w_code_corr = 0.0
        code_corr_target = 0.82
        code_corr_excess = jnp.maximum(0.0, code_proto_corr - code_corr_target)
        code_corr_penalty = jnp.where(
            jnp.max(code_corr_excess) > 1e-6,
            rank_normalize(code_corr_excess),
            jnp.zeros_like(code_corr_excess),
        )

        # 3. Calculate Fitness (all rank_normalize for scale parity)
        fitness_scores = (
            (rank_normalize(fitness_adv) * w_adv)          # Quality (capped, can't dominate)
            + (rank_normalize(fitness_mi) * w_mi)          # MI signal
            + (rank_normalize(r_sense) * w_sense)          # Feature-space code separation
            + (rank_normalize(pop_var) * w_div)            # Pixel-space code diversity
            + (rank_normalize(r_intra) * w_intra)          # Within-code variation (use z-noise)
            #+ (rank_normalize(morph_dark_range) * w_dark_range)
            #+ (rank_normalize(morph_center_edge_range) * w_center_edge_range)
            # Use raw excess (not rank-normalized) so members at/under target
            # receive exactly zero penalty instead of tie-rank noise.
            #- (edge_dark_excess * w_edge_dark_penalty)
            #- (code_corr_penalty * w_code_corr)
            #+ (rank_normalize(shape_score) * w_shape)      # Conditional shape variation (min+mean)
            #- (rank_normalize(cons_shortfall) * w_cons_floor) # Penalize centroid drift below floor
            #- (rank_normalize(normative_penalty) * w_norm) # Safety/spread limits
        )
        #w_mi = jnp.clip((self._t / 10000) * 10.0, 0.1, 0.6)
        
        #w_mi = 1.0
        #fitness_scores = -jnp.argsort(order)
        #    fitness_scores = fitness_adv
        #fitness_scores = fitness_adv + fitness_mi*0.18 + fitness_con*0.013 # + r_sense - normative_penalty*w_norm - r_cons*0.1
        #else:#if self._t < 160000:
        #cultural_score = r_sense + r_intra + r_cons + fitness_con
        # Keep reshaping population metrics dynamic so experiments with
        # different population sizes / class counts do not require code edits.
        n_members = fitness_scores.shape[0]

        if spreads.ndim == 1:
            n_codes = max(spreads.shape[0] // n_members, 1)
            spreads = spreads.reshape((n_members, n_codes, 1))
        elif spreads.ndim == 2 and spreads.shape[0] == n_members:
            spreads = spreads[:, :, None]

        if safety_ratios.ndim == 1:
            n_codes = max(int(np.sqrt(safety_ratios.shape[0] // n_members)), 1)
            safety_ratios = safety_ratios.reshape((n_members, n_codes, n_codes))
        elif safety_ratios.ndim == 2 and safety_ratios.shape[0] == n_members:
            n_codes = max(int(np.sqrt(safety_ratios.shape[1])), 1)
            safety_ratios = safety_ratios.reshape((n_members, n_codes, n_codes))

        self.belief_space = update_normative_ks(
            self.belief_space,
            fitness_scores,
            spreads,
            safety_ratios
        )

        # --- Update Domain, Situational, and Historical KS ---
        # Extract the best individual from this generation for the KS archives.
        # PGPE layout: solutions[0..255] = center + noise, solutions[256..511] = center - noise
        best_idx = jnp.argmax(fitness_scores.flatten())
        noise_idx = best_idx % self._num_directions
        sign = jnp.where(best_idx < self._num_directions, 1.0, -1.0)
        best_noise = self._scaled_noises[noise_idx] * sign
        best_solution = (self._center + best_noise).reshape(1, -1)
        best_scaled_noise = best_noise.reshape(1, -1)

        # Per-individual fitness components for the best individual
        best_fitness_adv = jnp.array([fitness_adv.flatten()[best_idx]])
        best_fitness_mi = jnp.array([fitness_mi.flatten()[best_idx]])
        best_fitness_combined = jnp.array([fitness_scores.flatten()[best_idx]])
        best_r_sense = jnp.array([r_sense.flatten()[best_idx]])
        best_r_cons = jnp.array([r_cons.flatten()[best_idx]])
        best_r_shape_div = jnp.array([shape_score.flatten()[best_idx]])

        # Population-level entropy proxy from discriminator class scores.
        # Convert scores -> probabilities first to avoid log of negative values.
        mean_disc_scores = jnp.mean(disc_logits, axis=(0, 1))  # (n_codes,)
        mean_disc_probs = jax.nn.softmax(
            jnp.nan_to_num(mean_disc_scores, nan=0.0, posinf=0.0, neginf=0.0)
        )
        mean_disc_probs = jnp.clip(mean_disc_probs, 1e-8, 1.0)
        mean_disc_probs = mean_disc_probs / jnp.maximum(jnp.sum(mean_disc_probs), 1e-8)
        pop_entropy = -jnp.sum(mean_disc_probs * jnp.log(mean_disc_probs))
        default_entropy = jnp.log(jnp.array(mean_disc_probs.shape[0], dtype=mean_disc_probs.dtype))
        pop_entropy = jnp.where(jnp.isfinite(pop_entropy), pop_entropy, default_entropy)

        # Update metric history (rolling buffer for slope computation)
        self.belief_space = update_metric_history(
            self.belief_space,
            jnp.max(fitness_adv),
            jnp.max(fitness_mi),
            pop_entropy,
            jnp.max(r_sense),
            avg_r_intra=jnp.mean(r_intra),
            avg_fitness_adv=jnp.mean(fitness_adv),
            avg_r_shape_div=jnp.mean(shape_score),
            avg_code_spread=jnp.mean(pop_var),
        )

        # Domain KS: Pareto front with GAN diagnostic metadata (r_sense, r_cons)
        self.belief_space = update_domain_ks(
            self.belief_space, best_solution, self._stdev,
            best_scaled_noise, best_fitness_adv, best_fitness_mi,
            best_fitness_combined, mean_disc_scores, best_r_sense, best_r_cons, best_r_shape_div
        )

        # Situational KS: tracks the single best solution (most exploitative)
        self.belief_space = update_situational_ks(
            self.belief_space, best_solution, self._stdev,
            best_scaled_noise, best_fitness_adv, best_fitness_mi,
            best_fitness_combined, mean_disc_scores
        )

        # Historical KS: archive of best solutions across generations
        self.belief_space = update_history_ks(
            self.belief_space, best_solution, self._stdev,
            best_scaled_noise, best_fitness_adv, best_fitness_mi,
            best_fitness_combined, mean_disc_scores
        )

        fitness_scores, self._best_score, self._avg_score = process_scores(fitness_scores,False)

        grad_center, grad_stdev = compute_reinforce_update(
                fitness_scores=fitness_scores,
                scaled_noises=self._scaled_noises,
                stdev=self._stdev,
            )

        # --- CA gradient blending (belief space → gradient influence) ---
        # The CA influences training by nudging the REINFORCE gradient direction
        # toward KS-suggested targets, not by directly modifying parameters.
        # This preserves PGPE's update mechanics (ClipUp for center, clipped
        # stdev update) while letting accumulated domain knowledge steer
        # the optimization away from failure modes like mode collapse.
        ca_center_g, ca_stdev_g, ks_winner = get_updated_params(
            self.belief_space, self._center, self._stdev, self._t
        )
        ca_center_g = ca_center_g.flatten()
        ca_stdev_g = jnp.clip(ca_stdev_g.flatten(), 1e-4, 1e1)

        # Direction from current toward KS-suggested target
        ca_grad_center = ca_center_g - self._center
        ca_grad_stdev = ca_stdev_g - self._stdev

        # Sanitize: replace any NaN with zero so it can't propagate.
        # NaN can arise from entropy computations on early disc_logit values.
        ca_grad_center = jnp.nan_to_num(ca_grad_center, nan=0.0)
        ca_grad_stdev = jnp.nan_to_num(ca_grad_stdev, nan=0.0)

        # Only blend when archives have real, non-zero data.
        has_ca_data = jnp.any(ca_center_g != 0.0) & jnp.all(jnp.isfinite(ca_center_g))
        # Delay + ramp the CA blend so early adversarial alignment is learned
        # from REINFORCE first, then CA guidance phases in.
        blend_progress = jnp.clip(
            (self._t - self._ca_blend_start_iter) / float(self._ca_blend_ramp_iters),
            0.0,
            1.0,
        )
        blend_target = self._ca_blend_coeff * blend_progress

        # Optional runtime gate from discriminator health.
        # Gate is active only if both bounds are valid and ordered.
        rfl_gate = jnp.float32(1.0)
        if self._ca_blend_rfl_hi > self._ca_blend_rfl_lo >= 0.0:
            rfl = jnp.array(self._runtime_real_fake_loss, dtype=jnp.float32)
            finite = jnp.isfinite(rfl)
            # Soft shoulders around [lo, hi] to avoid on/off jitter.
            shoulder = jnp.float32(0.05)
            lo = jnp.float32(self._ca_blend_rfl_lo)
            hi = jnp.float32(self._ca_blend_rfl_hi)
            gate_low = jnp.clip((rfl - (lo - shoulder)) / shoulder, 0.0, 1.0)
            gate_high = jnp.clip(((hi + shoulder) - rfl) / shoulder, 0.0, 1.0)
            rfl_gate = jnp.where(finite, jnp.minimum(gate_low, gate_high), 0.0)

        ca_blend = blend_target * jnp.float32(has_ca_data) * rfl_gate

        # Scale CA direction to match REINFORCE gradient magnitude so the
        # blend ratio is meaningful.  ClipUp normalizes center grad anyway;
        # for stdev, update_stdev clips by max_change so oversized grads
        # are safe, but matching scale keeps the 5% ratio honest.
        r_center_scale = jnp.linalg.norm(grad_center) + 1e-12
        ca_center_scale = jnp.linalg.norm(ca_grad_center) + 1e-12
        ca_grad_center = ca_grad_center * (r_center_scale / ca_center_scale)

        r_stdev_scale = jnp.linalg.norm(grad_stdev) + 1e-12
        ca_stdev_scale = jnp.linalg.norm(ca_grad_stdev) + 1e-12
        ca_grad_stdev = ca_grad_stdev * (r_stdev_scale / ca_stdev_scale)

        # Capture REINFORCE stdev direction before CA blending overwrites grad_stdev
        reinforce_stdev_direction = jnp.mean(jnp.sign(grad_stdev))

        # Use jnp.where to guard the blend — prevents 0*NaN=NaN propagation.
        # Unlike arithmetic (0.0 * NaN = NaN), jnp.where truly selects one
        # branch without contamination from the other.
        grad_center = jnp.where(
            has_ca_data,
            (1.0 - ca_blend) * grad_center + ca_blend * ca_grad_center,
            grad_center
        )
        grad_stdev = jnp.where(
            has_ca_data,
            (1.0 - ca_blend) * grad_stdev + ca_blend * ca_grad_stdev,
            grad_stdev
        )

        # --- Exploration & CA insight metrics ---
        # Stdev distribution stats (tracks whether exploration is growing/shrinking)
        stdev_mean = jnp.mean(self._stdev)
        stdev_std = jnp.std(self._stdev)
        stdev_min = jnp.min(self._stdev)
        stdev_max = jnp.max(self._stdev)

        # REINFORCE vs CA gradient magnitudes (before blending)
        reinforce_grad_norm = r_center_scale  # already computed above
        ca_grad_norm = ca_center_scale        # raw CA gradient norm (pre-rescaling)
        reinforce_stdev_grad_norm = r_stdev_scale
        ca_stdev_grad_norm = ca_stdev_scale

        # CA stdev guidance direction: positive = CA wants wider exploration
        # Fraction of dims where CA suggests increasing stdev
        ca_stdev_direction = jnp.mean(jnp.sign(ca_stdev_g - self._stdev))
        # Mean relative change CA is suggesting for stdev
        ca_stdev_rel_delta = jnp.mean((ca_stdev_g - self._stdev) / (self._stdev + 1e-8))

        # Domain KS archive stdev diversity: how different are archived exploration profiles
        domain_ks = self.belief_space[1]
        archived_stdevs = domain_ks[1]  # (20, param_size)
        # Mean stdev per archived solution, then std of those means
        archive_stdev_means = jnp.mean(archived_stdevs, axis=1)  # (20,)
        archive_stdev_diversity = jnp.std(archive_stdev_means)
        # Range of archived stdev profiles
        archive_stdev_range = jnp.max(archive_stdev_means) - jnp.min(archive_stdev_means)
        # Number of non-zero archive entries (how full is the Pareto archive)
        archive_occupancy = jnp.sum(jnp.any(archived_stdevs != 0, axis=1))

        # Effective center LR after decay
        decay_step = self._t // self._lr_decay_steps
        effective_center_lr = self._center_lr * jnp.power(
            jnp.float32(self._lr_decay_coef), jnp.float32(decay_step))

        # Cache diagnostics for trainer-side structured TSV logging.
        ks_winner_idx = int(np.asarray(ks_winner))
        if 0 <= ks_winner_idx < self._ks_winner_counts.size:
            self._ks_winner_counts[ks_winner_idx] += 1
        self._debug_last = {
            "adv_short": adv_short,
            "mi_short": mi_short,
            "adv_med": adv_med,
            "mi_med": mi_med,
            "ent_long": ent_long,
            "sense_short": sense_short,
            "intra_short": intra_short,
            "adv_avg_short": adv_avg_short,
            "sense_med": sense_med,
            "shape_short": shape_short,
            "shape_med": shape_med,
            "spread_short": spread_short,
            "spread_med": spread_med,
            "w_adv": w_adv,
            "w_mi": w_mi,
            "w_div": w_div,
            "w_sense": w_sense,
            "w_intra": w_intra,
            "w_shape": w_shape,
            "w_cons_floor": w_cons_floor,
            "w_norm": w_norm,
            "mi_guard": mi_guard,
            "d_health": d_health,
            "shape_health": shape_health,
            "objective_health": objective_health,
            "static_fitness_weights": jnp.float32(1.0 if self._static_fitness_weights else 0.0),
            "ca_blend": ca_blend,
            "ca_has_data": jnp.float32(has_ca_data),
            "ca_rfl_gate": rfl_gate,
            "shape_div_avg": jnp.mean(r_shape_div),
            "shape_div_min_avg": jnp.mean(r_shape_div_min),
            "shape_div_score_avg": jnp.mean(shape_score),
            "morph_dark_range_avg": jnp.mean(morph_dark_range),
            "morph_center_edge_range_avg": jnp.mean(morph_center_edge_range),
            "edge_dark_frac_avg": jnp.mean(edge_dark_frac),
            "code_proto_corr_avg": jnp.mean(code_proto_corr),
            "w_dark_range": w_dark_range,
            "w_center_edge_range": w_center_edge_range,
            "w_edge_dark_penalty": w_edge_dark_penalty,
            "w_code_corr": w_code_corr,
            "ks_dom_score": ks_scores[0],
            "ks_sit_score": ks_scores[1],
            "ks_hist_score": ks_scores[2],
            "ks_topo_score": ks_scores[3],
            "ks_dom_weight": ks_weights_dbg[0],
            "ks_sit_weight": ks_weights_dbg[1],
            "ks_hist_weight": ks_weights_dbg[2],
            "ks_topo_weight": ks_weights_dbg[3],
            "ks_winner": jnp.array(ks_winner_idx, dtype=jnp.float32),
            # --- Exploration & CA insight metrics ---
            "stdev_mean": stdev_mean,
            "stdev_std": stdev_std,
            "stdev_min": stdev_min,
            "stdev_max": stdev_max,
            "reinforce_grad_norm": reinforce_grad_norm,
            "ca_grad_norm": ca_grad_norm,
            "reinforce_stdev_grad_norm": reinforce_stdev_grad_norm,
            "ca_stdev_grad_norm": ca_stdev_grad_norm,
            "ca_stdev_direction": ca_stdev_direction,
            "ca_stdev_rel_delta": ca_stdev_rel_delta,
            "archive_stdev_diversity": archive_stdev_diversity,
            "archive_stdev_range": archive_stdev_range,
            "archive_occupancy": archive_occupancy,
            "effective_center_lr": effective_center_lr,
            "reinforce_stdev_direction": reinforce_stdev_direction,
        }

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
        self._stdev = jnp.maximum(self._stdev, 0.005)  # Floor: prevent exploration collapse

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

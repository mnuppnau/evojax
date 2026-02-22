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
    initialize_control_ks,
    update_topographic_ks,
    update_domain_ks,
    update_situational_ks,
    update_history_ks,
    update_normative_ks,
    update_metric_history,
    compute_metric_slopes,
    update_control_ks,
    get_control_outputs,
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
        logger: logging.Logger = None,
        fitness_mode: str = 'dynamic',
        ca_activation_iter: int = 5000,
        static_weights: Optional[dict] = None,
        ca_blend_max: float = 0.025,
        ca_blend_start_iter: int = 500,
        ca_blend_ramp_iters: int = 2000,
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
            fitness_mode - 'static' (fixed weights always) or 'dynamic'
                           (fixed weights early, CA modulation ramps in).
            ca_activation_iter - Iteration at which CA modulation begins
                                 ramping in (only used in dynamic mode).
            static_weights - Dict of fixed fitness weights. Used as base
                             weights in both modes.
            ca_blend_max - Maximum blend ratio for CA gradient guidance.
            ca_blend_start_iter - Iteration where CA gradient blending starts.
            ca_blend_ramp_iters - Number of iterations to ramp to ca_blend_max.
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

        if init_params is None:
            self._center = np.zeros(abs(param_size))
        else:
            self._center = init_params
        self._center = jnp.array(self._center)
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
        self._lr_decay_steps = optimizer_config.get(
            "center_lr_decay_steps", 10000
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
        if len(self.belief_space) == 7:
            # Backward compatibility for checkpoints created before control KS existed.
            self.belief_space = self.belief_space + (initialize_control_ks(),)

        # Fitness mode: 'static' (fixed weights always) or 'dynamic'
        # (fixed weights early, CA-modulated weights after ca_activation_iter)
        self.fitness_mode = fitness_mode
        self.ca_activation_iter = ca_activation_iter
        self.ca_blend_max = max(0.0, float(ca_blend_max))
        self.ca_blend_start_iter = int(ca_blend_start_iter)
        self.ca_blend_ramp_iters = max(1, int(ca_blend_ramp_iters))
        self.static_weights = static_weights or {
            'w_adv': 0.53, 'w_mi': 0.1, 'w_div': 0.48,
            'w_sense': 0.1, 'w_intra': 0.08, 'w_cons_floor': 0.04,
            'w_norm': 0.02, 'w_adv_ceiling': 0.0,
            'adv_ceiling': -0.65, 'adv_ceiling_warmup': 2000,
        }
        self._base_weight_vec = jnp.array([
            self.static_weights['w_adv'],
            self.static_weights['w_mi'],
            self.static_weights['w_div'],
            self.static_weights['w_sense'],
            self.static_weights['w_intra'],
            self.static_weights['w_cons_floor'],
            self.static_weights['w_norm'],
            self.static_weights.get('w_adv_ceiling', 0.0),
        ], dtype=jnp.float32)
        self._base_adv_ceiling = float(self.static_weights.get('adv_ceiling', -0.65))
        self._adv_ceiling_warmup = int(self.static_weights.get('adv_ceiling_warmup', 2000))

        # Latest CA runtime controls (used by Trainer to modulate D schedule).
        self._control_signals = {
            'd_dominance': 0.0,
            'shortcut_risk': 0.0,
            'diversity_distress': 0.0,
            'stagnation': 0.0,
            'prototype_lock': 0.0,
            'd_update_rate': 0.5,
            'ca_blend': 0.0,
            'adv_ceiling': self._base_adv_ceiling,
        }

        # KS weight logging: buffer entries and flush to file every 100 iterations
        self._ks_log_buffer = []
        self._ks_log_path = None

    def set_ks_log_path(self, path):
        """Set the file path for KS weight logging and write header."""
        self._ks_log_path = path
        with open(path, 'w') as f:
            f.write('\t'.join([
                'iter', 'ks_winner',
                'w_adv', 'w_mi', 'w_sense', 'w_div', 'w_intra', 'w_cons_floor', 'w_norm',
                'w_adv_ceiling', 'adv_ceiling', 'shortcut_violation_avg',
                'd_dominance', 'shortcut_risk', 'diversity_distress', 'stagnation', 'prototype_lock',
                'ca_blend', 'd_update_rate',
                'adv_short', 'mi_short', 'adv_med', 'mi_med', 'ent_long',
                'sense_short', 'intra_short', 'adv_avg_short', 'sense_med',
            ]) + '\n')

    def _flush_ks_log(self):
        """Flush buffered KS weight entries to file."""
        if self._ks_log_path is None or not self._ks_log_buffer:
            return
        with open(self._ks_log_path, 'a') as f:
            for row in self._ks_log_buffer:
                f.write('\t'.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in row) + '\n')
        self._ks_log_buffer = []

    def get_top_idx(self) -> jnp.ndarray:
        """Get the index of the top solution."""
        return self._top_indices

    def get_runtime_controls(self):
        """Expose CA control signals so Trainer can modulate D scheduling."""
        return dict(self._control_signals)

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


    def tell(self, fitness_adv: Union[np.ndarray, jnp.ndarray], fitness_mi: Union[np.ndarray, jnp.ndarray], disc_logits: Union[np.ndarray, jnp.ndarray], pop_var: Union[np.ndarray, jnp.ndarray], avg_per_code: Union[np.ndarray, jnp.ndarray], r_cons: Union[np.ndarray, jnp.ndarray], r_sense: Union[np.ndarray, jnp.ndarray], r_intra: Union[np.ndarray, jnp.ndarray], normative_penalty: Union[np.ndarray, jnp.ndarray], safety_ratios: Union[np.ndarray, jnp.ndarray], spreads: Union[np.ndarray, jnp.ndarray], adv: bool) -> None:

       
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


        # 1) Update topographic KS.
        # Momentum ramps up with iteration so centroids stabilize as training matures.
        base_blend_ramp = jnp.clip(
            (self._t - self.ca_blend_start_iter) / self.ca_blend_ramp_iters,
            0.0, 1.0
        )
        topo_momentum = 0.7 + 0.27 * base_blend_ramp
        self.belief_space = update_topographic_ks(
            self.belief_space, avg_per_code, topo_momentum
        )

        # 2) Update metric history before computing adaptive controls.
        mean_disc_logit = jnp.mean(disc_logits, axis=(0, 1))  # (11,) raw logits
        mean_disc_prob = jax.nn.softmax(mean_disc_logit)
        pop_entropy = -jnp.sum(mean_disc_prob * jnp.log(mean_disc_prob + 1e-8))
        self.belief_space = update_metric_history(
            self.belief_space,
            jnp.max(fitness_adv),
            jnp.max(fitness_mi),
            pop_entropy,
            jnp.max(r_sense),
            avg_r_intra=jnp.mean(r_intra),
            avg_fitness_adv=jnp.mean(fitness_adv),
        )

        # 3) Compute adaptive controls from belief space.
        if self.fitness_mode == 'dynamic':
            self.belief_space = update_control_ks(
                self.belief_space,
                jnp.float32(self._t),
                fitness_adv,
                fitness_mi,
                pop_var,
                r_sense,
                r_intra,
                jnp.mean(spreads),
                self._base_weight_vec,
                jnp.float32(self._base_adv_ceiling),
                jnp.float32(self.ca_activation_iter),
                jnp.float32(self.ca_blend_start_iter),
                jnp.float32(self.ca_blend_ramp_iters),
                jnp.float32(self.ca_blend_max),
            )
            adaptive_weights, adv_ceiling, ca_blend_adapt, d_update_rate, control_signals = get_control_outputs(self.belief_space)
            w_adv = adaptive_weights[0]
            w_mi = adaptive_weights[1]
            w_div = adaptive_weights[2]
            w_sense = adaptive_weights[3]
            w_intra = adaptive_weights[4]
            w_cons_floor = adaptive_weights[5]
            w_norm = adaptive_weights[6]
            w_adv_ceiling = adaptive_weights[7]
            d_dominance = control_signals[0]
            shortcut_risk = control_signals[1]
            diversity_distress = control_signals[2]
            stagnation = control_signals[3]
            prototype_lock = control_signals[4]
        else:
            w_adv = self._base_weight_vec[0]
            w_mi = self._base_weight_vec[1]
            w_div = self._base_weight_vec[2]
            w_sense = self._base_weight_vec[3]
            w_intra = self._base_weight_vec[4]
            w_cons_floor = self._base_weight_vec[5]
            w_norm = self._base_weight_vec[6]
            w_adv_ceiling = self._base_weight_vec[7]
            adv_ceiling = jnp.float32(self._base_adv_ceiling)
            ca_blend_adapt = jnp.float32(self.ca_blend_max * base_blend_ramp)
            d_update_rate = jnp.float32(0.5)
            d_dominance = jnp.float32(0.0)
            shortcut_risk = jnp.float32(0.0)
            diversity_distress = jnp.float32(0.0)
            stagnation = jnp.float32(0.0)
            prototype_lock = jnp.float32(0.0)

        # Runtime control signals for Trainer (Python-side scheduling).
        self._control_signals = {
            'd_dominance': float(d_dominance),
            'shortcut_risk': float(shortcut_risk),
            'diversity_distress': float(diversity_distress),
            'stagnation': float(stagnation),
            'prototype_lock': float(prototype_lock),
            'd_update_rate': float(d_update_rate),
            'ca_blend': float(ca_blend_adapt),
            'adv_ceiling': float(adv_ceiling),
        }

        # Metric slopes for logging/diagnostics.
        slopes = compute_metric_slopes(self.belief_space)
        (adv_short, mi_short, adv_med, mi_med, ent_long,
         sense_short, intra_short, adv_avg_short, sense_med) = slopes

        # r_cons floor penalty
        cons_floor = 0.05
        cons_shortfall = jnp.maximum(0.0, cons_floor - r_cons)

        # Guardrail against shortcut collapse: once adversarial reward rises
        # above the ceiling (toward 0), penalize those individuals directly.
        # This keeps G in the stable band instead of chasing transient D holes.
        adv_ceiling_gate = jnp.float32(self._t >= self._adv_ceiling_warmup)
        adv_shortcut_violation = jnp.maximum(0.0, fitness_adv - adv_ceiling)

        # Buffer KS weights for logging (flushed every 100 iterations)
        if self._ks_log_path is not None:
            self._ks_log_buffer.append([
                self._t, 0,  # ks_winner filled later after get_updated_params
                float(w_adv), float(w_mi), float(w_sense), float(w_div),
                float(w_intra), float(w_cons_floor), float(w_norm),
                float(w_adv_ceiling), float(adv_ceiling), float(jnp.mean(adv_shortcut_violation)),
                float(d_dominance), float(shortcut_risk), float(diversity_distress), float(stagnation), float(prototype_lock),
                float(ca_blend_adapt), float(d_update_rate),
                float(adv_short), float(mi_short), float(adv_med), float(mi_med),
                float(ent_long), float(sense_short), float(intra_short),
                float(adv_avg_short), float(sense_med),
            ])

        # Calculate Fitness (all rank_normalize for scale parity)
        fitness_scores = (
            (rank_normalize(fitness_adv) * w_adv)          # Quality (capped, can't dominate)
            + (rank_normalize(fitness_mi) * w_mi)          # MI signal
            + (rank_normalize(r_sense) * w_sense)          # Feature-space code separation
            + (rank_normalize(pop_var) * w_div)            # Pixel-space code diversity
            + (rank_normalize(r_intra) * w_intra)          # Within-code variation (use z-noise)
            - (rank_normalize(cons_shortfall) * w_cons_floor) # Penalize centroid drift below floor
            - (rank_normalize(normative_penalty) * w_norm) # Safety/spread limits
            - (adv_shortcut_violation * w_adv_ceiling * adv_ceiling_gate)
        )
        spreads = spreads.reshape(self.pop_size, 11, 1)
        safety_ratios = safety_ratios.reshape(self.pop_size, 11, 11)

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

        # Domain KS: Pareto front with GAN diagnostic metadata (r_sense, r_cons)
        # Pass mean_disc_prob (softmax probabilities) not raw logits — KS entropy
        # computation does sum(-log(x) * x) which assumes valid probabilities.
        self.belief_space = update_domain_ks(
            self.belief_space, best_solution, self._stdev,
            best_scaled_noise, best_fitness_adv, best_fitness_mi,
            best_fitness_combined, mean_disc_prob, best_r_sense, best_r_cons
        )

        # Situational KS: tracks the single best solution (most exploitative)
        self.belief_space = update_situational_ks(
            self.belief_space, best_solution, self._stdev,
            best_scaled_noise, best_fitness_adv, best_fitness_mi,
            best_fitness_combined, mean_disc_prob
        )

        # Historical KS: archive of best solutions across generations
        self.belief_space = update_history_ks(
            self.belief_space, best_solution, self._stdev,
            best_scaled_noise, best_fitness_adv, best_fitness_mi,
            best_fitness_combined, mean_disc_prob
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

        # Only blend when CA is active and archives have real, non-zero data.
        # Blend magnitude comes from CA control state (belief-space driven).
        has_ca_data = jnp.any(ca_center_g != 0.0) & jnp.all(jnp.isfinite(ca_center_g))
        ca_blend = ca_blend_adapt * jnp.float32(has_ca_data)

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

        # Backfill ks_winner into the buffered log entry for this iteration
        if self._ks_log_path is not None and self._ks_log_buffer:
            self._ks_log_buffer[-1][1] = int(ks_winner)

        self._opt_state = self._opt_update(
                self._t // self._lr_decay_steps, -grad_center, self._opt_state
        )
        self._t += 1

        # Flush KS log buffer every 100 iterations
        if self._t % 100 == 0:
            self._flush_ks_log()
       
        self._center = self._get_params(self._opt_state)
        
        self._stdev = update_stdev(
                stdev=self._stdev,
                lr=self._stdev_lr,
                max_change=self._stdev_max_change,
                grad=grad_stdev,
            )
        self._stdev = jnp.maximum(self._stdev, 0.005)  # Floor: prevent exploration collapse

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

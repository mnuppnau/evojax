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
from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import random

try:
    from jax.example_libraries import optimizers
except ModuleNotFoundError:
    from jax.experimental import optimizers

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger

from evojax.algo.cultural.population_space import update_population

from evojax.algo.cultural.belief_space import initialize_belief_space, influence, get_updated_params

from evojax.algo.cultural.knowledge_sources import update_knowledge_sources, update_topographic_ks, update_normative_ks

from evojax.algo.cultural.helper_functions import calculate_entropy

@partial(jax.jit, static_argnums=(1,))
def process_scores(
    x: Union[np.ndarray, jnp.ndarray], use_ranking: bool
) -> jnp.ndarray:
    """Convert fitness scores to rank if necessary."""

    x = jnp.array(x)
    if use_ranking:
        ranks = jnp.zeros(x.size, dtype=int)
        ranks = ranks.at[x.argsort()].set(jnp.arange(x.size)).reshape(x.shape)
        return ranks / ranks.max() - 0.5, jnp.array(x).max()
    else:
        return x, jnp.array(x).max()


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


#def update_center_and_stdev(
#    center: jnp.ndarray,
#    stdev: jnp.ndarray, belief_space: dict,
#    previous_best_score: float,
#    best_score: float,
#    t: int
#) -> Tuple[jnp.ndarray, jnp.ndarray]:
#    def update():
#        print("In ask_func Updating center and stdev")
#        return get_updated_params(belief_space, center, stdev, t)
#
#    def no_update():
#        print("No update needed")
#        return center, stdev
#
#    update_needed = jnp.logical_and(
#        jnp.logical_and(previous_best_score is not None, best_score is not None),
#        best_score < previous_best_score
#    )
#
#    return jax.lax.cond(update_needed, update, no_update)

@partial(jax.jit, static_argnums=(3,4))
def ask_func_infl(
    key: jnp.ndarray,
    stdev: jnp.ndarray,
    center: jnp.ndarray,
    num_directions: int,
    solution_size: int,
    belief_space: dict,
) -> Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
    next_key, key = random.split(key)
    scaled_noises = random.normal(key, [num_directions, solution_size]) * stdev
    # Apply the influence adjustments to the scaled noises
    scaled_noises = influence(belief_space,scaled_noises)
        
    solutions = jnp.hstack([center + scaled_noises, center - scaled_noises]).reshape(-1, solution_size)    
 
    return next_key, scaled_noises, solutions

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
        max_iter: int = 1000,
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
                momentum=optimizer_config.get("momentum", 0.9),
                max_speed=optimizer_config.get("max_speed", 0.15),
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
        self._solutions = None
        self._scaled_noises = None

        self._max_iter = max_iter

        self._previous_best_score = -11.0
        self._best_score = -10.0

        self._scaled_noises_adjustment_rate = 0.1
        self._kmeans_rate = 0.02
        self._adjust_kmeans_rate = 0.001

        self._kmeans_iterations = self._max_iter * self._kmeans_rate
        self._adjust_kmeans_iterations = self._max_iter * self._adjust_kmeans_rate

        #self.population_space = initialize_population(self.pop_size, self._center, self._stdev)

        self.belief_space = initialize_belief_space(population_size=self.pop_size, key=self._key, scaled_noises_adjustment_rate=self._scaled_noises_adjustment_rate, param_size=param_size)
        #self.belief_space.assign_indexes_to_knowledge_sources()

        #self._get_updated_params = jax.jit(get_updated_params)
        #self._influence = jax.jit(influence)

    def ask(self) -> jnp.ndarray:
        if self._t > 100:
       
           #self._center, self._stdev = update_center_and_stdev(
           #     self._center, 
           #     self._stdev, 
           #     self.belief_space, 
           #     self._previous_best_score, 
           #     self._best_score,
           #     self._t
           # )
           #if self._previous_best_score is not None and self._best_score is not None:
           #    if self._best_score < self._previous_best_score:
                   #print("Updating center and stdev")
            center, stdev = get_updated_params(self.belief_space ,self._center, self._stdev, self._t)
           #        self._center = center
           #        self._stdev = stdev
           #    else:
           #        center, stdev = self._center, self._stdev
           #else:
           #    center, stdev = self._center, self._stdev
        else:
             center, stdev = self._center, self._stdev

        #print("Center:", center)
        print("Center shape:", center.shape)
        #print("stdev:", stdev)
        #print("stdev shape:", stdev.shape)
       
        #if self._t > 4:
        #    print("scaled noises: ", self._scaled_noises)
        #    print("scaled noises shape: ", self._scaled_noises.shape)
        
        # Retrieve updated center and stdev from the belief space if available
        #if self._previous_best_score is not None and self._best_score is not None:
        #    if self._best_score < self._previous_best_score:
        #        print("Updating center and stdev")
        #        center, stdev = get_updated_params(self.belief_space ,self._center, self._stdev)
        #        self._center = center
        #        self._stdev = stdev
        #    else:
        #        center, stdev = self._center, self._stdev
        #else:
        #    center, stdev = self._center, self._stdev
        
        #belief_space = self.belief_space
   
        self._key, self._scaled_noises, self._solutions = ask_func(
            self._key,
            stdev,
            center,
            self._num_directions,
            self._center.size,
            )

        #next_key, key = random.split(self._key)
        #scaled_noises = random.normal(key, [self._num_directions, self._center.size]) * stdev
        
                # Apply the influence adjustments to the scaled noises
       
        #self._solutions = jnp.hstack(
        #    [center + self._scaled_noises, center - self._scaled_noises]
        #).reshape(-1, self._center.size)
        
        # Calculate and store noise magnitudes in the population space
        #for i, noise in enumerate(self._scaled_noises):
        #    magnitude = jnp.linalg.norm(noise)
        #    self.population_space.individuals[i].noise_magnitude = magnitude
        # 
        #self._key = next_key
        return self._solutions

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        self._previous_best_score = self._best_score
        self.fitness_scores, self._best_score = process_scores(fitness, self._solution_ranking)
      
        self._avg_score = jnp.array(fitness).mean()

        norm_entropy = calculate_entropy(self._center, self._stdev)

        self.population, best_individual = update_population(
            fitness_scores=self.fitness_scores,
            center=self._center,
            stdev=self._stdev,
            scaled_noises=self._scaled_noises
        )

        #individuals_fitness_score = [{'individual': ind, 'fitness_score': score} 
        #                     for ind, score in zip(self.population_space, self.population_space['fitness_scores'])]
        

        #best_individual = min(individuals_fitness_score, key=lambda x: x['fitness_score'])

        # Assuming self.population_space is your list of dictionaries
        #population_space = self.population_space  # Example, replace with your actual list
        
        # Initialize a variable to hold the dict with the fitness value closest to zero
        
        
        # closest_to_zero now holds the dict with the fitness value closest to zero

        #self.belief_space.accept(self.population_space.individuals)
        
        #self.belief_space.update()

        self.belief_space = update_knowledge_sources(self.belief_space, best_individual)   
        
        if self._t == 10:
            self.belief_space = update_topographic_ks(self.belief_space)
        elif self._t % self._kmeans_iterations == 0:
            self.belief_space = update_topographic_ks(self.belief_space)

        self.belief_space, self._adjust_kmeans_iterations = update_normative_ks(self.belief_space, best_fitness=self._best_score, avg_fitness=self._avg_score, norm_entropy=norm_entropy, adjust_kmeans_iterations=self._adjust_kmeans_iterations)

        self._kmeans_iterations += self._adjust_kmeans_iterations

        #self.belief_space[ks_type] = update_belief_space(self.belief_space, ks_type)

        #print("Domain KS:", self.belief_space.domain_ks)
        #print("Situational KS:", self.belief_space.situational_ks)
        #print("History KS:", self.belief_space.history_ks)
        
        grad_center, grad_stdev = compute_reinforce_update(
            fitness_scores=self.fitness_scores,
            scaled_noises=self._scaled_noises,
            stdev=self._stdev,
        )
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

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

import logging
import time
from typing import Optional, Callable

import jax.numpy as jnp
import numpy as np

from evojax.task.base import VectorizedTask
from evojax.policy import PolicyNetwork
from evojax.algo import NEAlgorithm
from evojax.algo import QualityDiversityMethod
from evojax.sim_mgr import SimManager
from evojax.obs_norm import ObsNormalizer
from evojax.util import create_logger
from evojax.util import load_model
from evojax.util import save_model
from evojax.util import save_lattices


class Trainer(object):
    """A trainer that organizes the training logistics."""

    def __init__(self,
                 policy: PolicyNetwork,
                 solver: NEAlgorithm,
                 train_task: VectorizedTask,
                 test_task: VectorizedTask,
                 max_iter: int = 1000,
                 log_interval: int = 20,
                 test_interval: int = 100,
                 n_repeats: int = 1,
                 test_n_repeats: int = 1,
                 n_evaluations: int = 100,
                 seed: int = 42,
                 debug: bool = False,
                 use_for_loop: bool = False,
                 normalize_obs: bool = False,
                 model_dir: str = None,
                 log_dir: str = None,
                 logger: logging.Logger = None,
                 log_scores_fn: Optional[Callable[[int, jnp.ndarray, str], None]] = None):
        """Initialization.

        Args:
            policy - The policy network to use.
            solver - The ES algorithm for optimization.
            train_task - The task for training.
            test_task - The task for evaluation.
            max_iter - Maximum number of training iterations.
            log_interval - Interval for logging.
            test_interval - Interval for tests.
            n_repeats - Number of rollout repetitions.
            n_evaluations - Number of tests to conduct.
            seed - Random seed to use.
            debug - Whether to turn on the debug flag.
            use_for_loop - Use for loop for rollouts.
            normalize_obs - Whether to use an observation normalizer.
            model_dir - Directory to save/load model.
            log_dir - Directory to dump logs.
            logger - Logger.
            log_scores_fn - custom function to log the scores array. Expects input:
                `current_iter`: int, `scores`: jnp.ndarray, 'stage': str = "train" | "test"
        """

        if logger is None:
            self._logger = create_logger(
                name='Trainer', log_dir=log_dir, debug=debug)
        else:
            self._logger = logger

        self._log_interval = log_interval
        self._test_interval = test_interval
        self._max_iter = max_iter
        self.model_dir = model_dir
        self._log_dir = log_dir
        self.policy = policy

        self.batch_stats = None
        self.batch_stats_avg = None
        self._log_scores_fn = log_scores_fn or (lambda x, y, z: None)

        self._obs_normalizer = ObsNormalizer(
            obs_shape=train_task.obs_shape,
            dummy=not normalize_obs,
        )

        self.solver = solver
        self.sim_mgr = SimManager(
            n_repeats=n_repeats,
            test_n_repeats=test_n_repeats,
            pop_size=solver.pop_size,
            n_evaluations=n_evaluations,
            policy_net=policy,
            train_vec_task=train_task,
            valid_vec_task=test_task,
            seed=seed,
            obs_normalizer=self._obs_normalizer,
            use_for_loop=use_for_loop,
            logger=self._logger,
        )

    def run(self, demo_mode: bool = False) -> float:
        """Start the training / test process."""
        def average_batch_stats_half(batch_stats):
            def average_stats_recursive(stats):
                if isinstance(stats, dict):
                    return {k: average_stats_recursive(v) for k, v in stats.items()}
                elif isinstance(stats, jnp.ndarray):
                    # Check if the array has the expected number of dimensions
                    if stats.ndim >= 2:
                        # Split the first dimension (population) into two halves and average each
                        first_half_mean = stats[:stats.shape[0] // 2].mean(axis=0, keepdims=True)
                        second_half_mean = stats[stats.shape[0] // 2:].mean(axis=0, keepdims=True)
                        return jnp.concatenate([first_half_mean, second_half_mean], axis=0)
                    else:
                        return stats
                else:
                    raise ValueError(f"Unsupported type: {type(stats)}")
            return average_stats_recursive(batch_stats)

        def average_batch_stats_for_both_sets(batch_stats):
            def average_stats_recursive(stats):
                if isinstance(stats, dict):
                    return {k: average_stats_recursive(v) for k, v in stats.items()}
                elif isinstance(stats, jnp.ndarray):
                    if stats.ndim >= 2:
                        # Average across the entire population (assuming it's the first dimension)
                        averaged_stats = stats.mean(axis=0)
                        # Duplicate the averaged stats to have a new shape of [2, batch_stats_size]
                        return jnp.stack([averaged_stats, averaged_stats], axis=0)
                    else:
                        return stats
                else:
                    raise ValueError(f"Unsupported type: {type(stats)}")

            return average_stats_recursive(batch_stats)

        def select_two_individuals_batch_stats(batch_stats, first_idx=0, second_idx=47):
            def select_stats_recursive(stats):
                if isinstance(stats, dict):
                    return {k: select_stats_recursive(v) for k, v in stats.items()}
                elif isinstance(stats, jnp.ndarray):
                    if stats.ndim >= 2:
                        # Select two specific dimensions (individuals)
                        selected_stats_first = stats[first_idx]
                        selected_stats_second = stats[second_idx]
                        # Stack these selected stats to have a new shape of [2, batch_stats_size]
                        return jnp.stack([selected_stats_first, selected_stats_second], axis=0)
                    else:
                        return stats
                else:
                    raise ValueError(f"Unsupported type: {type(stats)}")
        
            return select_stats_recursive(batch_stats)

        def print_batch_stats_shapes(batch_stats, parent_key=''):
            for key, value in batch_stats.items():
                # Construct a new key path
                new_key = f"{parent_key}.{key}" if parent_key else key

                if isinstance(value, dict):
                    # Recursively call the function if the value is another dict
                    print_batch_stats_shapes(value, new_key)
                elif isinstance(value, jnp.ndarray):
                    # Print the shape of the ndarray
                    print(f"Shape of '{new_key}': {value.shape}")
                else:
                    # Handle other types if necessary
                    print(f"'{new_key}' is not a dict or ndarray, but a {type(value)}")

        if self.model_dir is not None and not demo_mode:
            params = self.policy.flat_transferred_params
        elif self.model_dir is not None and demo_mode:
            params, batch_stats, obs_params = load_model(model_dir=self.model_dir)
            self.sim_mgr.obs_params = obs_params
            self._logger.info(
                'Loaded model parameters from {}.'.format(self.model_dir))
        else:
            params = None

        if demo_mode:
            if params is None:
                raise ValueError('No policy parameters to evaluate.')
            self._logger.info('Start to test the parameters.')
            scores,_,_ = self.sim_mgr.eval_params(params=params, test=True, batch_stats=batch_stats)
            self._logger.info(
                '[TEST] #tests={0}, max={1:.4f}, avg={2:.4f}, min={3:.4f}, '
                'std={4:.4f}'.format(scores.size, scores.max(), scores.mean(),
                                     scores.min(), scores.std()))
            return scores.mean()
        else:
            self._logger.info(
                'Start to train for {} iterations.'.format(self._max_iter))

            if params is not None:
                # Continue training from the breakpoint.
                self.solver.best_params = params

            best_score = -float('Inf')

            for i in range(self._max_iter):
                start_time = time.perf_counter()
                params = self.solver.ask()
                self._logger.debug('solver.ask time: {0:.4f}s'.format(
                    time.perf_counter() - start_time))

                start_time = time.perf_counter()
                        
                if i == 0:
                    scores, bds, batch_stats = self.sim_mgr.eval_params(
                        params=params, test=False)
                else:
                    scores, bds, batch_stats = self.sim_mgr.eval_params(
                        params=params, test=False, batch_stats=self.batch_stats)

                # Update batch_stats
                self.batch_stats = batch_stats
                #print_batch_stats_shapes(batch_stats)
                self.batch_stats_avg = select_two_individuals_batch_stats(batch_stats)


                #scores, bds, batch_stats = self.sim_mgr.eval_params(
                #    params=params, test=False)
                
                #batch_stats = average_batch_stats_half(batch_stats)
                self._logger.debug('sim_mgr.eval_params time: {0:.4f}s'.format(
                    time.perf_counter() - start_time))

                start_time = time.perf_counter()
                if isinstance(self.solver, QualityDiversityMethod):
                    self.solver.observe_bd(bds)
                self.solver.tell(fitness=scores)
                self._logger.debug('solver.tell time: {0:.4f}s'.format(
                    time.perf_counter() - start_time))

                if i > 0 and i % self._log_interval == 0:
                    scores = np.array(scores)
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores.size, scores.max(), scores.mean(),
                            scores.min(), scores.std()))
                    self._log_scores_fn(i, scores, "train")

                if i > 0 and i % self._test_interval == 0:
                    best_params = self.solver.best_params
                    test_scores, _, _ = self.sim_mgr.eval_params(
                        params=best_params, test=True, batch_stats=self.batch_stats_avg)
                    self._logger.info(
                        '[TEST] Iter={0}, #tests={1}, max={2:.4f}, avg={3:.4f}, '
                        'min={4:.4f}, std={5:.4f}'.format(
                            i, test_scores.size, test_scores.max(),
                            test_scores.mean(), test_scores.min(),
                            test_scores.std()))
                    self._log_scores_fn(i, test_scores, "test")
                    mean_test_score = test_scores.mean()
                    save_model(
                        model_dir=self._log_dir,
                        model_name='iter_{}'.format(i),
                        params=best_params,
                        obs_params=self.sim_mgr.obs_params,
                        best=mean_test_score > best_score,
                    )
                    best_score = max(best_score, mean_test_score)

            # Test and save the final model.
            best_params = self.solver.best_params
            test_scores, _, batch_stats = self.sim_mgr.eval_params(
                params=best_params, test=True, batch_stats=self.batch_stats_avg)
            self._logger.info(
                '[TEST] Iter={0}, #tests={1}, max={2:.4f}, avg={3:.4f}, '
                'min={4:.4f}, std={5:.4f}'.format(
                    self._max_iter, test_scores.size, test_scores.max(),
                    test_scores.mean(), test_scores.min(), test_scores.std()))
            mean_test_score = test_scores.mean()
            save_model(
                model_dir=self._log_dir,
                model_name='final',
                batch_stats=batch_stats,
                params=best_params,
                obs_params=self.sim_mgr.obs_params,
                best=mean_test_score > best_score,
            )
            best_score = max(best_score, mean_test_score)
            if isinstance(self.solver, QualityDiversityMethod):
                save_lattices(
                    log_dir=self._log_dir,
                    file_name='qd_lattices',
                    fitness_lattice=self.solver.fitness_lattice,
                    params_lattice=self.solver.params_lattice,
                    occupancy_lattice=self.solver.occupancy_lattice,
                )
            self._logger.info(
                'Training done, best_score={0:.4f}'.format(best_score))

            return best_score

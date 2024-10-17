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

import jax
import jax.numpy as jnp
import numpy as np

from evojax.task.base import VectorizedTask
from evojax.policy import PolicyNetwork
from evojax.algo import NEAlgorithm
from evojax.algo import QualityDiversityMethod
from evojax.sim_mgr import SimManager
from evojax.obs_norm import ObsNormalizer
from evojax.util import create_logger
from evojax.util import load_model_gen, load_model_disc
from evojax.util import save_model
from evojax.util import save_lattices


class Trainer(object):
    """A trainer that organizes the training logistics."""

    def __init__(self,
                 policy_gen: PolicyNetwork,
                 policy_disc: PolicyNetwork,
                 solver_gen: NEAlgorithm,
                 solver_disc: NEAlgorithm,
                 train_task_gen: VectorizedTask,
                 test_task_disc: VectorizedTask,
                 train_task_disc: VectorizedTask,
                 test_task_gen: VectorizedTask,
                 max_iter: int = 1000,
                 log_interval: int = 20,
                 test_interval: int = 100,
                 n_repeats: int = 1,
                 test_n_repeats: int = 2,
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

        self.batch_stats_gen = policy_gen.flat_batch_stats_gen
        self.batch_stats_disc = policy_disc.flat_batch_stats_disc

        self.fake_imgs = None
        self.cat_codes = None

        self._log_interval = log_interval
        self._test_interval = test_interval
        self._max_iter = max_iter
        self.model_dir = model_dir
        self._log_dir = log_dir

        self._log_scores_fn = log_scores_fn or (lambda x, y, z: None)

        self._obs_normalizer = ObsNormalizer(
            obs_shape=train_task_gen.obs_shape,
            dummy=not normalize_obs,
        )

        self.solver_gen = solver_gen
        self.solver_disc = solver_disc

        self.sim_mgr_gen = SimManager(
            n_repeats=n_repeats,
            test_n_repeats=test_n_repeats,
            pop_size=solver_gen.pop_size,
            n_evaluations=n_evaluations,
            policy_net=policy_gen,
            train_vec_task=train_task_gen,
            valid_vec_task=test_task_gen,
            seed=seed,
            obs_normalizer=self._obs_normalizer,
            use_for_loop=use_for_loop,
            logger=self._logger,
        )

        self.sim_mgr_disc = SimManager(
            n_repeats=n_repeats,
            test_n_repeats=test_n_repeats,
            pop_size=solver_disc.pop_size,
            n_evaluations=n_evaluations,
            policy_net=policy_disc,
            train_vec_task=train_task_disc,
            valid_vec_task=test_task_disc,
            seed=seed + 1,
            obs_normalizer=self._obs_normalizer,
            use_for_loop=use_for_loop,
            logger=self._logger,
        )

    def run(self, demo_mode: bool = False) -> float:
        """Start the training / test process."""

        if self.model_dir is not None:
            params_gen, self.batch_stats_gen = load_model_gen(model_dir=self.model_dir)
            params_disc, self.batch_stats_disc = load_model_disc(model_dir=self.model_dir)
            #self.sim_mgr.obs_params = obs_params
            self._logger.info(
                'Loaded model parameters from {}.'.format(self.model_dir))
        else:
            params_gen, params_disc = None, None

        if demo_mode:
            if params is None:
                raise ValueError('No policy parameters to evaluate.')
            self._logger.info('Start to test the parameters.')
            scores = np.array(
                self.sim_mgr.eval_params(params=params, test=True)[0])
            self._logger.info(
                '[TEST] #tests={0}, max={1:.4f}, avg={2:.4f}, min={3:.4f}, '
                'std={4:.4f}'.format(scores.size, scores.max(), scores.mean(),
                                     scores.min(), scores.std()))
            return scores.mean()
        else:
            self._logger.info(
                'Start to train for {} iterations.'.format(self._max_iter))

            if params_gen is not None and params_disc is not None:
                # Continue training from the breakpoint.
                self.solver_gen.best_params = params_gen
                self.solver_disc.best_params = params_disc

            best_score_gen, best_score_disc = -float('Inf'), -float('Inf')

            for i in range(self._max_iter):
                # Generator step.
                params_gen = self.solver_gen.ask()
                params_disc = self.solver_disc.ask()
                
                scores_disc, bds_disc, self.batch_stats_gen, self.batch_stats_disc, _ = self.sim_mgr_disc.eval_params(
                    params_gen=params_gen, params_disc=params_disc, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc, generator=False, test=False
                )

                if isinstance(self.solver_disc, QualityDiversityMethod):
                    self.solver_disc.observe_bd(bds_disc)
               
                self.solver_disc.tell(fitness=scores_disc)
 
                scores_gen, bds_gen, self.batch_stats_gen, self.batch_stats_disc, _ = self.sim_mgr_gen.eval_params(
                params_gen=params_gen, params_disc=params_disc, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc, generator=True, test=False
                )

                if isinstance(self.solver_gen, QualityDiversityMethod):
                    self.solver_gen.observe_bd(bds_gen)
                
                self.solver_gen.tell(fitness=scores_gen)

                #self.fake_imgs = jnp.squeeze(self.fake_imgs, axis=0)

                #if (i > 1000 and i % 2 == 0) or i < 1001:
                #scores_disc, bds_disc, _, _, _, _ = self.sim_mgr_disc.eval_params(
                #    params_gen=None, params_disc=params_disc, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc, generator=False, cat_codes=self.cat_codes, fake_imgs=self.fake_imgs, test=False
                #)

                #if isinstance(self.solver_disc, QualityDiversityMethod):
                #    self.solver_disc.observe_bd(bds_disc)
                
                #if (i > 1000 and i % 2 == 0) or i < 1001:
                #self.solver_disc.tell(fitness=scores_disc)

                if i > 0 and i % self._log_interval == 0:
                    scores_gen = np.array(scores_gen)
                    self._logger.info('Generator:')
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores_gen.size, scores_gen.max(), scores_gen.mean(),
                            scores_gen.min(), scores_gen.std()))
                    scores_disc = np.array(scores_disc)
                    #self._logger.info('Discriminator:')
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores_disc.size, scores_disc.max(), scores_disc.mean(),
                            scores_disc.min(), scores_disc.std()))
                    #with open('/home/gh0st/Downloads/pgpe_main.csv', 'a') as file:
                        #file.write(f'Iter: {i}, Max: {scores.max()}, Mean: {scores.mean()}, Std: {scores.std()}, Min: {scores.min()}\n')
                    #self._log_scores_fn(i, scores, "train")

                if i > 0 and i % self._test_interval == 0:
                    best_params_gen = self.solver_gen.best_params
                    best_params_disc = self.solver_disc.best_params

                    #jax.debug.print('batch stats gen shape : {} ', self.batch_stats_gen.shape)

                    test_scores, _, _, _, fake_imgs = self.sim_mgr_gen.eval_params(
                        params_gen=best_params_gen, params_disc=best_params_disc, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc, generator=True, test=True
                    )
                    test_scores = np.array(test_scores)
                    self._logger.info(
                        '[TEST] Iter={0}, #tests={1}, max={2:.4f}, avg={3:.4f}, '
                        'min={4:.4f}, std={5:.4f}'.format(
                            i, test_scores.size, test_scores.max(),
                            test_scores.mean(), test_scores.min(),
                            test_scores.std()))
                   
                    #jax.debug.print('test scores shape : {} ', test_scores.shape)
                    filename = f"iteration-{i}.npy"
                    np.save(filename, fake_imgs[0, 0, :, :, :, :])

                    #jax.debug.print('testing, fake_imgs shape : {} ', self.fake_imgs.shape)

                    self._log_scores_fn(i, test_scores, "test")
                    mean_test_score = test_scores.mean()
                    #save_model(
                    #    model_dir=self._log_dir,
                    #    model_name='iter_{}'.format(i),
                    #    params=best_params,
                    #    obs_params=self.sim_mgr.obs_params,
                    #    best=mean_test_score > best_score,
                    #)
                    #best_score = max(best_score, mean_test_score)

            # Test and save the final model.
            best_params_gen = self.solver_gen.best_params
            best_params_disc = self.solver_disc.best_params
            #test_scores, _ = self.sim_mgr.eval_params(
            #    params=best_params, test=True)
            #self._logger.info(
            #    '[TEST] Iter={0}, #tests={1}, max={2:.4f}, avg={3:.4f}, '
            #    'min={4:.4f}, std={5:.4f}'.format(
            #        self._max_iter, test_scores.size, test_scores.max(),
            #        test_scores.mean(), test_scores.min(), test_scores.std()))
            #mean_test_score = test_scores.mean()
            save_model(
                model_dir=self._log_dir,
                model_name='final_model_gen',
                params=best_params_gen,
                obs_params=self.sim_mgr_gen.obs_params,
                batch_stats=self.batch_stats_gen,
                #best=mean_test_score > best_score,
            )
            save_model(
                model_dir=self._log_dir,
                model_name='final_model_disc',
                params=best_params_disc,
                obs_params=self.sim_mgr_disc.obs_params,
                batch_stats=self.batch_stats_disc,
                #best=mean_test_score > best_score,
            )
            #best_score = max(best_score, mean_test_score)
            #if isinstance(self.solver, QualityDiversityMethod):
            #    save_lattices(
            #        log_dir=self._log_dir,
            #        file_name='qd_lattices',
            #        fitness_lattice=self.solver.fitness_lattice,
            #        params_lattice=self.solver.params_lattice,
            #        occupancy_lattice=self.solver.occupancy_lattice,
            #    )
            self._logger.info(
                'Training done, best_score={0:.4f}'.format(best_score))

            return best_score

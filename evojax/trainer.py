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
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import he_normal
from flax import linen as nn
from torchvision import datasets

class Generator(nn.Module):
    """ Generator CNN for MNIST """

    features: int = 64
    training: bool = True

    @nn.compact
    def __call__(self, z):
        z = z.reshape((z.shape[0], 1, 1, z.shape[1]))
        x = nn.ConvTranspose(self.features*4, [3, 3], [2, 2], 'VALID', kernel_init=he_normal())(z)
        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(self.features*2, [4, 4], [1, 1], 'VALID', kernel_init=he_normal())(x)
        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(self.features, [3, 3], [2, 2], 'VALID', kernel_init=he_normal())(x)
        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(1, [4, 4], [2, 2], 'VALID', kernel_init=he_normal())(x)
        x = jnp.tanh(x)
        return x

class Discriminator(nn.Module):
    features: int = 64
    training: bool = True

    #q_cat: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
        x = nn.leaky_relu(x, 0.1)
        x = nn.Conv(self.features*2, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
        x = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(x)
        x = nn.leaky_relu(x, 0.1)
        
        # Discriminator output
        d = nn.Conv(1, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
        d = d.reshape((d.shape[0], -1))
   
        return d, x#, disc_logits, mu.squeeze(), jnp.exp(var)

class QNetwork(nn.Module):
    features: int = 64
    training: bool = True

    q_cat: int = 10

    @nn.compact
    def __call__(self, x):
        q = nn.Conv(self.features, [3, 3], [2, 2], 'VALID', kernel_init=he_normal())(x) 
        q = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(q)
        q = nn.leaky_relu(q, 0.1)
       
        disc_logits = nn.Conv(self.q_cat, [1, 1], [2, 2], 'VALID', kernel_init=he_normal())(q)
        disc_logits = disc_logits.reshape((disc_logits.shape[0], -1)) 

        return disc_logits

def sample_latent(key, shape_noise, shape_cat):
  noise_key, cat_key = jax.random.split(key, 2)
  
  # Sample irreducible noise
  noise = jax.random.normal(noise_key, shape_noise)

  # Sample categorical latent code
  code_cat = jax.random.randint(cat_key, shape_cat, 0, 10)
  code_cat = jax.nn.one_hot(code_cat, 10)

  latent = jnp.concatenate([noise, code_cat], axis=-1)

  return latent, code_cat

def sample_batch(key: jnp.ndarray,
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 batch_size: int) -> Tuple:
    ix = random.choice(
        key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(data, indices=ix, axis=0),
            jnp.take(labels, indices=ix, axis=0))

class Trainer(object):
    """A trainer that organizes the training logistics."""

    def __init__(self,
                 policy_gen: PolicyNetwork,
                 policy_disc: PolicyNetwork,
                 solver_gen: NEAlgorithm,
                 #solver_disc: NEAlgorithm,
                 #solver_q: NEAlgorithm,
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
                 batch_size: int = 128,
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
        self.batch_stats_q = policy_disc.flat_batch_stats_q

        self.batch_size = batch_size
        self.mini_batch_size = 64
        self.num_mini_batches = 10
        
        self.fake_imgs = None
        self.cat_codes = None

        self._key = jax.random.PRNGKey(44)

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
        self.solver_q = solver_q

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

        dataset = datasets.MNIST('./data', train=True, download=True)
        data = np.expand_dims(dataset.data.numpy() / 255.0, axis=-1)
        labels = dataset.targets.numpy()

        #self.sim_mgr_disc = SimManager(
        #    n_repeats=n_repeats,
        #    test_n_repeats=test_n_repeats,
        #    pop_size=solver_disc.pop_size,
        #    n_evaluations=n_evaluations,
        #    policy_net=policy_disc,
        #    train_vec_task=train_task_disc,
        #    valid_vec_task=test_task_disc,
        #    seed=seed + 1,
        #    obs_normalizer=self._obs_normalizer,
        #    use_for_loop=use_for_loop,
        #    logger=self._logger,
        #)

    def run(self, demo_mode: bool = False) -> float:

        def gather_pop_stats(belief_space):

            mean_mi = belief_space[5][9]
            mean_g = belief_space[5][9]
            mean_cond = belief_space[5][9]

            var_mi = belief_space[5][9]
            var_g = belief_space[5][9]
            var_cond = belief_space[5][9]
        
            return jnp.array([mean_mi, mean_g, mean_cond, var_mi, var_g, var_cond])

        """Start the training / test process."""

        if self.model_dir is not None:
            params_gen, self.batch_stats_gen = load_model_gen(model_dir=self.model_dir)
            params_disc, self.batch_stats_disc = load_model_disc(model_dir=self.model_dir)
            #self.sim_mgr.obs_params = obs_params
            self._logger.info(
                'Loaded model parameters from {}.'.format(self.model_dir))
        else:
            params_gen, params_disc, params_q = None, None, None

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

            if params_gen is not None and params_disc is not None and params_q is not None:
                # Continue training from the breakpoint.
                self.solver_gen.best_params = params_gen
                self.solver_disc.best_params = params_disc
                self.solver_q.best_params = params_q

            best_score_gen, best_score_disc, best_score_q = -float('Inf'), -float('Inf'), -float('Inf')

            for i in range(self._max_iter):
                # Sample latent codes.
                self._key, subkey = jax.random.split(self._key)
                shape_noise = (self.batch_size, 64)
                shape_cat = (self.batch_size,)
                latent, cat_codes = sample_latent(subkey, shape_noise, shape_cat)
              

                for mini_batch in range(self.num_mini_batches):
                    real_imgs, real_labels = sample_batch(self._key, self.data, self.labels, self.mini_batch_size)

                    (fake_imgs), vars_g = Generator().apply({'params': params_gen, 'batch_stats': self.batch_stats_gen},latent, mutable=['batch_stats'])
                    
                    (real_preds, _), vars_d = Discriminator().apply({'params': params_disc, 'batch_stats': self.batch_stats_disc}, real_imgs, mutable=['batch_stats'])
                    (fake_preds, q), vars_d = Discriminator().apply({'params': params_disc, 'batch_stats': self.batch_stats_disc}, fake_imgs, mutable=['batch_stats'])
                
                    (disc_logits, _), vars_q = QNetwork().apply({'params': params_q, 'batch_stats': self.batch_stats_q}, q, mutable=['batch_stats'])

                    real_loss = optax.sigmoid_binary_cross_entropy(real_preds, jnp.ones((self.mini_batch_size, 1), dtype=jnp.float32)).mean()
                    fake_loss = optax.sigmoid_binary_cross_entropy(fake_preds, jnp.zeros((self.mini_batch_size, 1), dtype=jnp.float32)).mean()

                    loss_disc = real_loss + fake_loss
                # Generator step.
                params_gen, belief_space = self.solver_gen.ask()
                
                

                scores_gen_adv, scores_gen_mi, disc_logits, bds_gen, self.batch_stats_gen, _, _, _, pop_stats_updated, _ = self.sim_mgr_gen.eval_params(
                params_gen=params_gen, params_disc=params_disc, params_q=params_q, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc, batch_stats_q=self.batch_stats_q, pop_stats=pop_stats, disc_reset_keys_cat_code=disc_reset_keys_cat_code, generator=True, test=False
                )

                if isinstance(self.solver_gen, QualityDiversityMethod):
                    self.solver_gen.observe_bd(bds_gen)
                
                self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_mi, disc_logits=disc_logits, adv=True)

                #top_gen_idx = self.solver_gen.get_top_idx()
                top_gen_idx = jnp.argmax(scores_gen_adv)
                # select top gen idx batch norm stats with a shape of (pop_size, num_features)
                self.batch_stats_gen = self.batch_stats_gen[top_gen_idx].flatten()
                
                params_gen, belief_space = self.solver_gen.ask()
                
                if i > 0 and i % self._log_interval == 0:
                    scores_gen_adv = np.array(scores_gen_adv)
                    self._logger.info('Generator:')
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores_gen_adv.size, scores_gen_adv.max(), scores_gen_adv.mean(),
                            scores_gen_adv.min(), scores_gen_adv.std()))
                    scores_disc = np.array(scores_real+scores_fake)
                    #self._logger.info('Discriminator:')
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores_disc.size, scores_disc.max(), scores_disc.mean(),
                            scores_disc.min(), scores_disc.std()))
                    scores_mi = np.array(scores_mi)
                    #self._logger.info('Mutual Information:')
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores_mi.size, scores_mi.max(), scores_mi.mean(),
                            scores_mi.min(), scores_mi.std()))
                    #with open('/home/gh0st/Downloads/pgpe_main.csv', 'a') as file:
                        #file.write(f'Iter: {i}, Max: {scores.max()}, Mean: {scores.mean()}, Std: {scores.std()}, Min: {scores.min()}\n')
                    #self._log_scores_fn(i, scores, "train")

                if i > 0 and i % self._test_interval == 0:
                    best_params_gen = self.solver_gen.best_params
                    best_params_disc = self.solver_disc.best_params
                    best_params_q = self.solver_q.best_params
                    #jax.debug.print('batch stats gen shape : {} ', self.batch_stats_gen.shape)

                    test_scores, _, _, _, _, _, _, fake_imgs, _, _ = self.sim_mgr_gen.eval_params(
                        params_gen=best_params_gen, params_disc=best_params_disc, params_q=best_params_q, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=self.batch_stats_disc, batch_stats_q=self.batch_stats_q, generator=True, test=True, pop_stats=pop_stats_updated, disc_reset_keys_cat_code=disc_reset_keys_cat_code
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
                    np.save(filename, fake_imgs[:, 23, :, :, :, :])

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

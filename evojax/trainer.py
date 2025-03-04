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
import optax

from functools import partial
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

# import Tuple
from typing import Tuple


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

        # Q Network
        q = nn.Conv(self.features*2, [4, 4], [2, 2], 'VALID', kernel_init=normal_init(0.02))(x)
        q = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(q)
        q = nn.leaky_relu(q, 0.1)

        q = nn.Conv(10, [1, 1], [2, 2], 'VALID', kernel_init=normal_init(0.02))(q)
        q = q.reshape((q.shape[0], -1))

        return d,q#, disc_logits, mu.squeeze(), jnp.exp(var)

#@jax.jit
#def train_step_gen(params_g, batch_stats_g, latent):
#    (fake_images), vars_g = Generator().apply({'params': params_g, 'batch_stats': batch_stats_g},latent, mutable=['batch_stats'])
#    batch_stats_g = vars_g['batch_stats']
#    return fake_images, batch_stats_g

@partial(jax.jit, static_argnames=['solver'])
def train_step_disc(params_g, batch_stats_g, params_d, batch_stats_d, data, fake_cat_input, opt_disc, solver):
        def bce_logits(logit, label):
                  """
                  Implements the BCE with logits loss, as described:
                  https://github.com/pytorch/pytorch/issues/751
                  """
                  neg_abs = -jnp.abs(logit)
                  batch_bce = jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(neg_abs))
                  return jnp.mean(batch_bce)

        def loss_mutual_information(code_cat, q_cat):
                  cat_loss = -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))
                  mi_loss = cat_loss
                  return mi_loss
            
            
        def loss_discriminator(params_d, vars_d_batch_stats):
                
                  (fake_imgs, vars_g) = Generator().apply(
                      {'params': params_g, 'batch_stats': batch_stats_g},
                      latent, mutable=['batch_stats']
                  )
                  (real_preds, _), vars_d = Discriminator().apply(
                      {'params': params_d, 'batch_stats': vars_d_batch_stats},
                      data, mutable=['batch_stats']
                  )
                  (fake_preds, q), vars_d = Discriminator().apply(
                      {'params': params_d, 'batch_stats': vars_d['batch_stats']},
                      fake_imgs, mutable=['batch_stats']
                  )
                
                  # Calculate Mutual Information loss
                  q_cat = nn.log_softmax(q, axis=-1)
                  loss_mi = loss_mutual_information(fake_cat_input, q_cat)
                
                  # real_preds reshape array of shape (64, 0) (size 0) to (64,)
                  real_loss = bce_logits(real_preds, jnp.ones((32,), dtype=jnp.int32))
                  fake_loss = bce_logits(fake_preds, jnp.zeros((32,), dtype=jnp.int32))
               
                  #jax.debug.print('real loss: {} ', real_loss)
                  #jax.debug.print('fake loss: {} ', fake_loss)

                  loss = (real_loss + fake_loss) + loss_mi* 0.1
                
                  return loss, (vars_d, vars_g)

        grad_fn_disc = jax.value_and_grad(loss_discriminator, has_aux=True)
        (loss, (vars_d,vars_g)), grads = grad_fn_disc(params_d, batch_stats_d)
        
        # apply gradients
        updates, new_opt_state = solver.update(grads, opt_disc)
        params_d = optax.apply_updates(params_d, updates)
        batch_stats_g = vars_g['batch_stats']
        # update batch stats
        batch_stats_d = vars_d['batch_stats']
        return params_d, batch_stats_d, batch_stats_g, new_opt_state

#class QNetwork(nn.Module):
#    features: int = 64
#    training: bool = True
#
#    q_cat: int = 10
#
#    @nn.compact
#    def __call__(self, x):
#        q = nn.Conv(self.features, [3, 3], [2, 2], 'VALID', kernel_init=he_normal())(x) 
#        q = nn.BatchNorm(not self.training, -1, 0.1, scale_init=normal_init(0.02))(q)
#        q = nn.leaky_relu(q, 0.1)
#       
#        disc_logits = nn.Conv(self.q_cat, [1, 1], [2, 2], 'VALID', kernel_init=he_normal())(q)
#        disc_logits = disc_logits.reshape((disc_logits.shape[0], -1)) 
#
#        return disc_logits

#@jax.jit
def sample_latent(key, shape_noise):
  noise_key, cat_key = jax.random.split(key, 2)
  
  # Sample irreducible noise
  noise = jax.random.normal(noise_key, shape_noise)

  # Sample categorical latent code
  #code_cat = jax.random.randint(cat_key, shape_cat, 0, 10)
  #code_cat = jax.nn.one_hot(code_cat, 10)

  c = jnp.tile(jnp.arange(10), 52)
  c = c[:32]

  code_cat = jax.nn.one_hot(c, 10)

  latent = jnp.concatenate([noise, code_cat], axis=-1)

  return latent, code_cat

#@jax.jit
def sample_batch(key: jnp.ndarray,
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 batch_size: int) -> Tuple:
    ix = jax.random.choice(
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
        self.mini_batch_size = 32
        self.num_mini_batches = 6
        
        self.shape_noise = (self.mini_batch_size, 64)
        self.shape_cat = (self.mini_batch_size,)

        self.fake_imgs = None
        self.cat_codes = None

        self.policy_gen = policy_gen

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
        #self.solver_disc = solver_disc
        #self.solver_q = solver_q

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

        self._key, subkey = jax.random.split(self._key)
        
        dataset = datasets.MNIST('./data', train=True, download=True)
        self.data = np.expand_dims(dataset.data.numpy() / 255.0, axis=-1)
        self.labels = dataset.targets.numpy()

        # initialize the discriminator
        variables_disc = Discriminator().init(subkey, jnp.ones((self.batch_size, 28, 28, 1), dtype=jnp.float32))
        self.params_disc, self.batch_stats_disc = variables_disc['params'], variables_disc['batch_stats']

        #self.solver_disc = optax.adam(learning_rate=0.0002, b1=0.5, b2=0.999)

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

        def bce_logits(logit, label):
            """
            Implements the BCE with logits loss, as described:
            https://github.com/pytorch/pytorch/issues/751
            """
            neg_abs = -jnp.abs(logit)
            batch_bce = jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(neg_abs))
            return jnp.mean(batch_bce)
        
        def loss_mutual_information(code_cat, q_cat):
                   cat_loss = -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))
                   mi_loss = cat_loss
                   return mi_loss
             
             
        def loss_discriminator(params_g, batch_stats_g, params_d, vars_d_batch_stats, data, fake_cat_input, latent):
                 
                   (fake_imgs, vars_g) = Generator().apply(
                       {'params': params_g, 'batch_stats': batch_stats_g},
                       latent, mutable=['batch_stats']
                   )
                   (real_preds, _), vars_d = Discriminator().apply(
                       {'params': params_d, 'batch_stats': vars_d_batch_stats},
                       data, mutable=['batch_stats']
                   )
                   (fake_preds, q), vars_d = Discriminator().apply(
                       {'params': params_d, 'batch_stats': vars_d['batch_stats']},
                       fake_imgs, mutable=['batch_stats']
                   )
                 
                   # Calculate Mutual Information loss
                   q_cat = nn.log_softmax(q, axis=-1)
                   loss_mi = loss_mutual_information(fake_cat_input, q_cat)
                 
                   # real_preds reshape array of shape (64, 0) (size 0) to (64,)
                   real_loss = bce_logits(real_preds, jnp.ones((32,), dtype=jnp.int32))
                   fake_loss = bce_logits(fake_preds, jnp.zeros((32,), dtype=jnp.int32))
                
                   loss = (real_loss + fake_loss) + loss_mi* 0.1
                 
                   return loss, (vars_d, vars_g)
        
        def fit(params_g, batch_stats_g, params_d, batch_stats_d, opt_disc, opt_state):

            @jax.jit
            def step(params_g, batch_stats_g, params_d, batch_stats_d, data, latent, fake_cat_input, opt_state):

                grad_fn_disc = jax.value_and_grad(loss_discriminator, argnums=2, has_aux=True)
                (loss, (vars_d,vars_g)), grads = grad_fn_disc(params_g, batch_stats_g, params_d, batch_stats_d, data, fake_cat_input, latent)
                
                # apply gradients
                updates, new_opt_state = opt_disc.update(grads, opt_state, params_d)
                params_d = optax.apply_updates(params_d, updates)
                batch_stats_g = vars_g['batch_stats']
                # update batch stats
                batch_stats_d = vars_d['batch_stats']
                return params_d, batch_stats_d, batch_stats_g, new_opt_state

            for i in range(self.num_mini_batches):
                # Sample batch of data.
                self._key, subkey_latent, subkey_mnist = jax.random.split(self._key, 3)
                
                data, labels = sample_batch(subkey_mnist, self.data, self.labels, self.mini_batch_size)
                #data = np.expand_dims(data / 255.0, axis=-1)


                latent, cat_codes = sample_latent(subkey_latent, self.shape_noise)

                # Train the discriminator.
                params_d, batch_stats_d, batch_stats_g, opt_state = step(
                    params_g,
                    batch_stats_g,
                    params_d,
                    batch_stats_d,
                    data,
                    latent,
                    cat_codes,
                    opt_state,
                )

            return params_d, batch_stats_d, opt_state
        #opt_disc = solver_disc.init(self.params_disc)
        opt_disc = optax.adam(learning_rate=0.0002, b1=0.5, b2=0.999) 

        opt_state = opt_disc.init(self.params_disc)
        
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

            if params_gen is not None:
                # Continue training from the breakpoint.
                self.solver_gen.best_params = params_gen

            best_score_gen, best_score_disc, best_score_q = -float('Inf'), -float('Inf'), -float('Inf')

            params_disc = self.params_disc
            
            for i in range(self._max_iter):
                params_gen, belief_space = self.solver_gen.ask()

                # Sample latent codes.
                #self._key, subkey = jax.random.split(self._key)
                #latent, cat_codes = sample_latent(subkey, shape_noise)

                # Sample batch of data.
                #data, labels = sample_batch(subkey, self.data, self.labels, self.mini_batch_size)
                
                best_params_gen = self.solver_gen.best_params
                #best_params_gen = jnp.expand_dims(best_params_gen, axis=0)
                params_gen_formatted = self.policy_gen._format_single_params_gen_fn(best_params_gen)
              
                #if self.batch_stats_gen.shape[0] != 400 and len(self.batch_stats_gen.shape) == 2: #and not test:
                #    # add pop size as first dimension to batch_stats_gen and batch_stats_disc
                #    batch_stats_gen = jnp.repeat(self.batch_stats_gen[None, :], 400, axis=0)
        
                # add dimension to self.batch_stats_gen of shape (896,) to (1, 896)
                if len(self.batch_stats_gen.shape) == 1:
                    batch_stats_gen = jnp.expand_dims(self.batch_stats_gen, axis=0)
                
                batch_stats_gen = self.policy_gen._format_batch_stats_gen_fn(batch_stats_gen)
                #jax.debug.print('batch stats gen: {} ', batch_stats_gen)
                #else:
                #    batch_stats_gen = self.batch_stats_gen

                params_disc, self.batch_stats_disc, opt_state = fit(
                    params_gen_formatted,
                    batch_stats_gen,
                    params_disc,
                    self.batch_stats_disc,
                    opt_disc,
                    opt_state,
                )
                #for mini_batch in range(self.num_mini_batches):
                #    # Sample batch of data.

                #    self._key, subkey_latent, subkey_mnist = jax.random.split(self._key, 3)
                #    

                #    data, labels = sample_batch(subkey_mnist, self.data, self.labels, self.mini_batch_size)
                #    #data = np.expand_dims(data / 255.0, axis=-1)

                #    latent, cat_codes = sample_latent(subkey_latent, shape_noise, shape_cat)
                #  
                #    #if i > 600:
                #    #    params_gen = self.solver_gen.ask_ca()
                #    #    params_gen_formatted = self.policy_gen._format_single_params_gen_fn(params_gen)
                #    #else:
                #    best_params_gen = self.solver_gen.best_params
                #    params_gen_formatted = self.policy_gen._format_single_params_gen_fn(best_params_gen)

                #    #(fake_images), vars_g = Generator().apply({'params': params_gen_formatted, 'batch_stats': batch_stats_gen},latent, mutable=['batch_stats'])
                #    #fake_images, batch_stats_gen = train_step_gen(
                #    #    params_gen_formatted,
                #    #    batch_stats_gen,
                #    #    latent,
                #    #)

                #    #batch_stats_gen = vars_g['batch_stats']
                #    #jax.debug.print('params disc shape: {} ', params_disc.shape)
                #    #jax.debug.print('batch stats disc shape: {} ', self.batch_stats_disc.shape)
                #    # Train the discriminator.
                #    params_disc, self.batch_stats_disc, opt_disc = train_step_disc(
                #        params_gen_formatted,
                #        batch_stats_gen,
                #        params_disc,
                #        self.batch_stats_disc,
                #        data,
                #        cat_codes,
                #        opt_disc,
                #        solver_disc,
                #    )
                   
                leaves_params, _ = jax.tree_flatten(params_disc) 
                flat_params_disc = jnp.concatenate([p.flatten() for p in leaves_params])

                leaves_batch_stats_disc, _ = jax.tree_flatten(self.batch_stats_disc)
                flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

                #params_gen, belief_space = self.solver_gen.ask()
                disc_reset_keys_cat_code = None
                # Generator step.
                scores_gen_adv, scores_gen_mi, disc_logits, bds_gen, self.batch_stats_gen, _, _ = self.sim_mgr_gen.eval_params(
                params_gen=params_gen, params_disc=flat_params_disc, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=flat_batch_stats_disc,  generator=True, test=False
                )

                if isinstance(self.solver_gen, QualityDiversityMethod):
                    self.solver_gen.observe_bd(bds_gen)
                
                self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_mi, disc_logits=disc_logits, adv=False)

                #top_gen_idx = self.solver_gen.get_top_idx()
                top_gen_idx = jnp.argmax(scores_gen_mi)
                # select top gen idx batch norm stats with a shape of (pop_size, num_features)
                self.batch_stats_gen = self.batch_stats_gen[top_gen_idx].flatten()
                
                params_gen, belief_space = self.solver_gen.ask()
                
                scores_gen_adv, scores_gen_mi, disc_logits, bds_gen, self.batch_stats_gen, _, _ = self.sim_mgr_gen.eval_params(
                params_gen=params_gen, params_disc=flat_params_disc, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=flat_batch_stats_disc,  generator=True, test=False
                )

                if isinstance(self.solver_gen, QualityDiversityMethod):
                    self.solver_gen.observe_bd(bds_gen)
                
                self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_mi, disc_logits=disc_logits, adv=True)

                #top_gen_idx = self.solver_gen.get_top_idx()
                top_gen_idx = jnp.argmax(scores_gen_adv)
                # select top gen idx batch norm stats with a shape of (pop_size, num_features)
                self.batch_stats_gen = self.batch_stats_gen[top_gen_idx].flatten()
                

                if i > 0 and i % self._log_interval == 0:
                    scores_gen_adv = np.array(scores_gen_adv)
                    self._logger.info('Generator:')
                    self._logger.info(
                        'Iter={0}, size={1}, max={2:.4f}, '
                        'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                            i, scores_gen_adv.size, scores_gen_adv.max(), scores_gen_adv.mean(),
                            scores_gen_adv.min(), scores_gen_adv.std()))
                    #scores_disc = np.array(scores_real+scores_fake)
                    #self._logger.info('Discriminator:')
                    #self._logger.info(
                    #    'Iter={0}, size={1}, max={2:.4f}, '
                    #    'avg={3:.4f}, min={4:.4f}, std={5:.4f}'.format(
                    #        i, scores_disc.size, scores_disc.max(), scores_disc.mean(),
                    #        scores_disc.min(), scores_disc.std()))
                    scores_mi = np.array(scores_gen_mi)
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
                    #best_params_disc = self.solver_disc.best_params
                    #best_params_q = self.solver_q.best_params
                    #jax.debug.print('batch stats gen shape : {} ', self.batch_stats_gen.shape)

                    test_scores, _, _, _, _, _, fake_imgs = self.sim_mgr_gen.eval_params(
                        params_gen=best_params_gen, params_disc=flat_params_disc, batch_stats_gen=self.batch_stats_gen, batch_stats_disc=flat_batch_stats_disc, generator=True, test=True)
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

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
def train_step_disc(state, data, fake_imgs, fake_cat_input, solver):
       
        params_d, batch_stats_d, opt_disc = state
        #def bce_logits(logit, label):
        #          """
        #          Implements the BCE with logits loss, as described:
        #          https://github.com/pytorch/pytorch/issues/751
        #          """
        #          neg_abs = -jnp.abs(logit)
        #          batch_bce = jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(neg_abs))
        #          return jnp.mean(batch_bce)

        def loss_mutual_information(code_cat, q_cat):
                  return -jnp.mean(jnp.sum(code_cat * q_cat, axis=-1))
            
            
        def loss_discriminator(params_d, vars_d_batch_stats):
                
                  #(fake_imgs, vars_g) = Generator().apply(
                  #    {'params': params_g, 'batch_stats': batch_stats_g},
                  #    latent, mutable=['batch_stats']
                  #)
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
                  #real_preds = real_preds.reshape((real_preds.shape[0],))
                  #fake_preds = fake_preds.reshape((fake_preds.shape[0],))
                  #real_loss = bce_logits(real_preds, jnp.ones((32,), dtype=jnp.int32))
                  #fake_loss = bce_logits(fake_preds, jnp.zeros((32,), dtype=jnp.int32))
              
                  # use 0.9 as the label for real images instead of 1.0
                  real_loss = optax.sigmoid_binary_cross_entropy(real_preds, jnp.ones_like(real_preds))
                  # use 0.1 as the label for fake images instead of 0.0
                  fake_loss = optax.sigmoid_binary_cross_entropy(fake_preds, jnp.zeros_like(fake_preds))

                  real_loss = jnp.mean(real_loss)
                  fake_loss = jnp.mean(fake_loss)

                  #jax.debug.print('real loss: {} ', real_loss)
                  #jax.debug.print('fake loss: {} ', fake_loss)

                  loss = (real_loss + fake_loss) / 2.0 + loss_mi*0.1
                
                  return loss, vars_d

        grad_fn_disc = jax.value_and_grad(loss_discriminator, has_aux=True)
        (loss, vars_d), grads = grad_fn_disc(params_d, batch_stats_d)
        
        # apply gradients
        updates, new_opt_state = solver.update(grads, opt_disc, params_d)
        params_d = optax.apply_updates(params_d, updates)
        #batch_stats_g = vars_g['batch_stats']
        # update batch stats
        batch_stats_d = vars_d['batch_stats']
        return (params_d, batch_stats_d, new_opt_state), loss

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

def sample_latent(key, shape_noise, shape_cat):
  noise_key, cat_key = jax.random.split(key, 2)
  
  # Sample irreducible noise
  noise = jax.random.normal(noise_key, shape_noise)

  # Sample categorical latent code
  code_cat = jax.random.randint(cat_key, shape_cat, 0, 10)
  code_cat = jax.nn.one_hot(code_cat, 10)

  #c = jnp.tile(jnp.arange(10), 52)
  #c = c[:32]

  #code_cat = jax.nn.one_hot(c, 10)

  #con = jax.random.uniform(key, (32, 2), minval=-1, maxval=1)
  
  latent = jnp.concatenate([noise, code_cat], axis=-1)

  return latent, code_cat

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
        self.mini_batch_size = 64
        self.num_mini_batches = 4
       
        self.avg_mi_loss = -2.30
        self.fake_imgs = None
        self.cat_codes = None

        self.latent_dim = 64
        self.n_classes = 10
        self.n_con = 2

        self.decay_factor = 0.9

        self.noise_dim = self.latent_dim - self.n_con
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

        #self.solver_disc = optax.adamw(learning_rate=0.0002, b1=0.5, b2=0.999, weight_decay=1e-6)
        self.solver_disc = optax.adam(learning_rate=0.0002, b1=0.5, b2=0.999)
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
            solver_disc = self.solver_disc
            opt_disc = solver_disc.init(self.params_disc)


            self._logger.info(
                'Start to train for {} iterations.'.format(self._max_iter))

            if params_gen is not None and params_disc is not None and params_q is not None:
                # Continue training from the breakpoint.
                self.solver_gen.best_params = params_gen

            best_score_gen, best_score_disc, best_score_q = -float('Inf'), -float('Inf'), -float('Inf')

            params_disc = self.params_disc
           
            num_mini_batches = self.num_mini_batches
            
            for i in range(self._max_iter):
                #params_gen, belief_space = self.solver_gen.ask()

                #if self.avg_mi_loss > -0.10:
                #    num_mini_batches = 6
                #else:
                #    num_mini_batches = 8
                
                # Sample latent codes.
                #self._key, subkey = jax.random.split(self._key)
                shape_noise = (self.mini_batch_size, self.latent_dim)
                shape_cat = (self.mini_batch_size,)
                #latent, cat_codes = sample_latent(subkey, shape_noise, shape_cat)

                #best_params_gen = self.solver_gen.best_params
                #best_params_gen = jnp.expand_dims(best_params_gen, axis=0)
                #params_gen_formatted = self.policy_gen._format_single_params_gen_fn(best_params_gen)
              
                #if self.batch_stats_gen.shape[0] != 400 and len(self.batch_stats_gen.shape) == 2: #and not test:
                #    # add pop size as first dimension to batch_stats_gen and batch_stats_disc
                #    batch_stats_gen = jnp.repeat(self.batch_stats_gen[None, :], 400, axis=0)
        
                # add dimension to self.batch_stats_gen of shape (896,) to (1, 896)
               
                if i < 1:
                    if len(self.batch_stats_gen.shape) == 1:
                        batch_stats_gen = jnp.expand_dims(self.batch_stats_gen, axis=0)
                
                    batch_stats_gen = self.policy_gen._format_batch_stats_gen_fn(batch_stats_gen)
                #jax.debug.print('batch stats gen: {} ', batch_stats_gen)
                #else:
                #    batch_stats_gen = self.batch_stats_gen
                #best_params_gen = self.solver_gen.best_params
                #params_gen_formatted = self.policy_gen._format_single_params_gen_fn(best_params_gen)

                #state = (params_gen_formatted, batch_stats_gen, params_disc, self.batch_stats_disc, opt_disc)

                for mini_batch in range(num_mini_batches):
                    # Sample batch of data.

                    self._key, subkey_latent, subkey_mnist = jax.random.split(self._key, 3)
                    

                    data, labels = sample_batch(subkey_mnist, self.data, self.labels, self.mini_batch_size)
                    #data = np.expand_dims(data / 255.0, axis=-1)

                    latent, cat_codes = sample_latent(subkey_latent, shape_noise, shape_cat)
                  
                    #if i > 600:
                    params_gen = self.solver_gen.best_params
                    params_gen_formatted = self.policy_gen._format_single_params_gen_fn(params_gen)
                    #else:
                    
                    (fake_images), vars_g = Generator(training=False).apply({'params': params_gen_formatted, 'batch_stats': batch_stats_gen},latent, mutable=['batch_stats'])
                    # reshape fake_images to (64, 28, 28, 1) from [1,1,1,64, 28, 28, 1]
                    fake_images = fake_images.reshape((self.batch_size, 28, 28, 1))
                    #batch_stats_gen = vars_g['batch_stats']
                    #jax.debug.print('fake images shape: {} ', fake_images.shape) 
                    state = (params_disc, self.batch_stats_disc, opt_disc)

                    #fake_images, batch_stats_gen = train_step_gen(
                    #    params_gen_formatted,
                    #    batch_stats_gen,
                    #    latent,
                    #)

                    #batch_stats_gen = vars_g['batch_stats']
                    #jax.debug.print('params disc shape: {} ', params_disc.shape)
                    #jax.debug.print('batch stats disc shape: {} ', self.batch_stats_disc.shape)
                    # Train the discriminator.
                    state, loss = train_step_disc(
                        state,
                        data,
                        fake_images,
                        cat_codes,
                        solver_disc,
                    )

                    #jax.debug.print('loss: {} ', loss)
                    
                    params_disc, self.batch_stats_disc, opt_disc = state 

                leaves_params, _ = jax.tree_flatten(params_disc) 
                flat_params_disc = jnp.concatenate([p.flatten() for p in leaves_params])

                leaves_batch_stats_disc, _ = jax.tree_flatten(self.batch_stats_disc)
                flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])

                leaves_batch_stats_gen, _ = jax.tree_flatten(batch_stats_gen)
                flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

                params_gen, belief_space = self.solver_gen.ask()
                disc_reset_keys_cat_code = None
                # Generator step.
                scores_gen_adv, scores_gen_mi, scores_gen_con, disc_logits, bds_gen, BN_stats_gen, _, _ = self.sim_mgr_gen.eval_params(
                params_gen=params_gen, params_disc=flat_params_disc, batch_stats_gen=flat_batch_stats_gen, batch_stats_disc=flat_batch_stats_disc,  generator=True, test=False
                )

                if isinstance(self.solver_gen, QualityDiversityMethod):
                    self.solver_gen.observe_bd(bds_gen)
                
                self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_con, disc_logits=disc_logits, adv=False)

                #jax.debug.print('first batch stats gen individual : {} ', batch_stats_gen[0])
                #top_gen_idxs = self.solver_gen.get_top_idx()
                # get top 20 index values from scores_gen_adv
                #top_gen_idxs = jnp.argsort(scores_gen_adv)[-20:]
                #top_gen_idx = jnp.argmax(scores_gen_adv)
                # select top gen idx batch norm stats with a shape of (pop_size, num_features)
                #batch_stats_gen = batch_stats_gen[top_gen_idxs]#.flatten()
                #n = batch_stats_gen.shape[0]
                #weights = self.decay_factor ** jnp.arange(n)

                #weighted_mean = jnp.average(batch_stats_gen, axis=0, weights=weights)

                #if i < 1:
                #    self.batch_stats_gen =  weighted_mean
                #else:
                #    self.batch_stats_gen = jnp.mean(self.batch_stats_gen, axis=0) * 0.9 + weighted_mean * 0.1
               
                #jax.debug.print('batch stats gen : {} ', self.batch_stats_gen)
                params_gen, belief_space = self.solver_gen.ask()
               
                best_params_gen = self.solver_gen.best_params
                best_params_gen_formatted = self.policy_gen._format_single_params_gen_fn(best_params_gen)

                self._key, subkey = jax.random.split(self._key)
                shape_noise = (self.batch_size, self.latent_dim)
                shape_cat = (self.batch_size,)
                latent, cat_codes = sample_latent(subkey, shape_noise, shape_cat)

                (fake_images), vars_g = Generator().apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},latent, mutable=['batch_stats'])
                batch_stats_gen = vars_g['batch_stats']

                leaves_batch_stats_gen, _ = jax.tree_flatten(batch_stats_gen)
                flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])

                scores_gen_adv, scores_gen_mi, scores_gen_con, disc_logits, bds_gen, BN_stats_gen, _, _ = self.sim_mgr_gen.eval_params(
                params_gen=params_gen, params_disc=flat_params_disc, batch_stats_gen=flat_batch_stats_gen, batch_stats_disc=flat_batch_stats_disc,  generator=True, test=False
                )

                if isinstance(self.solver_gen, QualityDiversityMethod):
                    self.solver_gen.observe_bd(bds_gen)
                
                self.solver_gen.tell(fitness_adv=scores_gen_adv, fitness_mi=scores_gen_mi, fitness_con=scores_gen_con, disc_logits=disc_logits, adv=True)

                #top_gen_idx = self.solver_gen.get_top_idx()
                #top_gen_idxs = self.solver_gen.get_top_idx()
                best_params_gen = self.solver_gen.best_params
                best_params_gen_formatted = self.policy_gen._format_single_params_gen_fn(best_params_gen)

                self._key, subkey = jax.random.split(self._key)
                shape_noise = (self.batch_size, self.latent_dim)
                shape_cat = (self.batch_size,)
                latent, cat_codes = sample_latent(subkey, shape_noise, shape_cat)

                (fake_images), vars_g = Generator().apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},latent, mutable=['batch_stats'])
                batch_stats_gen = vars_g['batch_stats'] 

                #(fake_images), vars_d = Generator().apply({'params': best_params_gen_formatted, 'batch_stats': batch_stats_gen},latent, mutable=['batch_stats'])
                #batch_stats_gen = vars_d['batch_stats']

                #top_gen_idxs = jnp.argsort(scores_gen_adv)[-20:]
                #top_gen_idx = jnp.argmax(scores_gen_adv)
                # select top gen idx batch norm stats with a shape of (pop_size, num_features)
                #batch_stats_gen = batch_stats_gen[top_gen_idxs]#.flatten()
             
                #weighted_mean = jnp.average(batch_stats_gen, axis=0, weights=weights)

                #self.batch_stats_gen = jnp.mean(self.batch_stats_gen, axis=0) * 0.9 + weighted_mean * 0.1 

                #jax.debug.print('batch stats gen : {} ', self.batch_stats_gen)
                self.batch_stats_gen = batch_stats_gen
                
                self.avg_mi_loss = jnp.mean(scores_gen_mi)

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
                    batch_stats_gen_leaves, _ = jax.tree_flatten(batch_stats_gen)
                    flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in batch_stats_gen_leaves])
                    #best_params_disc = self.solver_disc.best_params
                    #best_params_q = self.solver_q.best_params
                    #jax.debug.print('batch stats gen shape : {} ', self.batch_stats_gen.shape)

                    test_scores, _, _, _, _, _, _, fake_imgs = self.sim_mgr_gen.eval_params(
                        params_gen=best_params_gen, params_disc=flat_params_disc, batch_stats_gen=flat_batch_stats_gen, batch_stats_disc=flat_batch_stats_disc, generator=True, test=True)
                    test_scores = np.array(test_scores)
                    self._logger.info(
                        '[TEST] Iter={0}, #tests={1}, max={2:.4f}, avg={3:.4f}, '
                        'min={4:.4f}, std={5:.4f}'.format(
                            i, test_scores.size, test_scores.max(),
                            test_scores.mean(), test_scores.min(),
                            test_scores.std()))
                   
                    #jax.debug.print('test scores shape : {} ', test_scores.shape)
                    filename = f"iteration-{i}.npy"
                    np.save(filename, fake_imgs[23, :, :, :, :])

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

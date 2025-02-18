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
from functools import partial
from typing import Tuple
from typing import Union
import time

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map

from evojax.obs_norm import ObsNormalizer
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask
from evojax.policy.base import PolicyState
from evojax.policy.base import PolicyNetwork
from evojax.util import create_logger

#@partial(jax.jit, static_argnums=(2, 3, 4, 5))
#def get_task_reset_keys_latent(key1: jnp.ndarray,
#                        key2: jnp.ndarray,
#                        pop_size: int,
#                        n_tests: int,
#                        n_repeats: int,
#                        ma_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
#    # Split the first key
#    key1, subkey1 = random.split(key=key1)
#    
#    # Split both keys under "if testing" condition
#    key2, subkey2 = random.split(key=key2)
#    
#    reset_keys1 = random.split(subkey1, n_repeats)
#    reset_keys1 = jnp.tile(reset_keys1, (pop_size, 1))
#
#    reset_keys2 = random.split(subkey2, n_repeats)
#    reset_keys2 = jnp.tile(reset_keys2, (pop_size, 1))
#
#    return key1, reset_keys1, reset_keys2

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def get_task_reset_keys(key1: jnp.ndarray,
                        key2: jnp.ndarray,
                        key3: jnp.ndarray,
                        pop_size: int,
                        n_tests: int,
                        n_repeats: int,
                        ma_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Split the first key
    key1, subkey1 = random.split(key=key1)
    
    # Split both keys under "if testing" condition
    key2, subkey2 = random.split(key=key2)
    
    key3, subkey3 = random.split(key=key3)

    reset_keys1 = random.split(subkey1, n_repeats)
    reset_keys1 = jnp.tile(reset_keys1, (pop_size, 1))

    reset_keys2 = random.split(subkey2, n_repeats)
    reset_keys2 = jnp.tile(reset_keys2, (pop_size, 1))

    reset_keys3 = random.split(subkey3, n_repeats)
    reset_keys3 = jnp.tile(reset_keys3, (pop_size, 1))

    return key1, key2, key3, reset_keys1, reset_keys2, reset_keys3

@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def get_task_reset_keys_disc(key1: jnp.ndarray,
                        key2: jnp.ndarray,
                        key3: jnp.ndarray,
                        key4: jnp.ndarray,
                        pop_size: int,
                        n_tests: int,
                        n_repeats: int,
                        ma_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Split the first key
    key1, subkey1 = random.split(key=key1)
    
    # Split both keys under "if testing" condition
    key2, subkey2 = random.split(key=key2)
    key3, subkey3 = random.split(key=key3)
    key4, subkey4 = random.split(key=key4)

    reset_keys1 = random.split(subkey1, n_repeats)
    reset_keys1 = jnp.tile(reset_keys1, (pop_size, 1))

    reset_keys2 = random.split(subkey2, n_repeats)
    reset_keys2 = jnp.tile(reset_keys2, (pop_size, 1))

    reset_keys3 = random.split(subkey3, n_repeats)
    reset_keys3 = jnp.tile(reset_keys3, (pop_size, 1))

    reset_keys4 = random.split(subkey4, n_repeats)
    reset_keys4 = jnp.tile(reset_keys4, (pop_size, 1))

    return key1, key2, key3, reset_keys1, reset_keys2, reset_keys3, reset_keys4


#@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
#def get_task_reset_keys(key: jnp.ndarray,
#                        test: bool,
#                        pop_size: int,
#                        n_tests: int,
#                        n_repeats: int,
#                        ma_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
#    key, subkey = random.split(key=key)
#    if ma_training:
#        reset_keys = random.split(subkey, n_repeats)
#    else:
#        if test:
#            reset_keys = random.split(subkey, n_tests * n_repeats)
#        else:
#            reset_keys = random.split(subkey, n_repeats)
#            reset_keys = jnp.tile(reset_keys, (pop_size, 1))
#    return key, reset_keys


@jax.jit
def split_params_for_pmap(param: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(jnp.split(param, jax.local_device_count()))


@jax.jit
def split_states_for_pmap(
        state: Union[TaskState, PolicyState]) -> Union[TaskState, PolicyState]:
    return tree_map(split_params_for_pmap, state)


@jax.jit
def reshape_data_from_pmap(data: jnp.ndarray) -> jnp.ndarray:
    # data.shape = (#device, steps, #jobs/device, *)
    data = data.transpose([1, 0] + [i for i in range(2, data.ndim)])
    return jnp.reshape(data, (data.shape[0], data.shape[1] * data.shape[2], -1))

@jax.jit
# reshape two dim array to one dim array
def combine_ind_scalar(data: jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(data, (data.shape[0] * data.shape[1],))

@jax.jit
def merge_state_from_pmap(state: TaskState) -> TaskState:
    return jax.tree_map(
        lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])), state)


@partial(jax.jit, static_argnums=(1, 2))
def duplicate_params(params: jnp.ndarray,
                     repeats: int,
                     ma_training: bool) -> jnp.ndarray:
    if ma_training:
        return jnp.tile(params, (repeats, ) + (1,) * (params.ndim - 1))
    else:
        return jnp.repeat(params, repeats=repeats, axis=0)


@jax.jit
def update_score_and_mask(score, reward, mask, done):
    new_score = score + reward * mask
    new_mask = mask * (1 - done.ravel())
    return new_score, new_mask


@partial(jax.jit, static_argnums=(1,))
def report_score(scores, n_repeats):
    return jnp.mean(scores.ravel().reshape((-1, n_repeats)), axis=-1)


@jax.jit
def all_done(masks):
    return masks.sum() == 0


class SimManager(object):
    """Simulation manager."""

    def __init__(self,
                 n_repeats: int,
                 test_n_repeats: int,
                 pop_size: int,
                 n_evaluations: int,
                 policy_net: PolicyNetwork,
                 train_vec_task: VectorizedTask,
                 valid_vec_task: VectorizedTask,
                 seed: int = 0,
                 obs_normalizer: ObsNormalizer = None,
                 use_for_loop: bool = False,
                 logger: logging.Logger = None):
        """Initialization function.

        Args:
            n_repeats - Number of repeated parameter evaluations.
            pop_size - Population size.
            n_evaluations - Number of evaluations of the best parameter.
            policy_net - Policy network.
            train_vec_task - Vectorized tasks for training.
            valid_vec_task - Vectorized tasks for validation.
            seed - Random seed.
            obs_normalizer - Observation normalization helper.
            use_for_loop - Use for loop for rollout instead of jax.lax.scan.
            logger - Logger.
        """

        if logger is None:
            self._logger = create_logger(name='SimManager')
        else:
            self._logger = logger

        self._use_for_loop = use_for_loop
        self._logger.info('use_for_loop={}'.format(self._use_for_loop))
        self._key = random.PRNGKey(seed=seed)
        self._n_repeats = n_repeats
        self._test_n_repeats = test_n_repeats
        self._pop_size = pop_size
        self._n_evaluations = max(n_evaluations, jax.local_device_count())
        self._ma_training = train_vec_task.multi_agent_training

        self._t = 0
        self._i = 1.0
        
        self.obs_normalizer = obs_normalizer
        if self.obs_normalizer is None:
            self.obs_normalizer = ObsNormalizer(
                obs_shape=train_vec_task.obs_shape,
                dummy=True,
            )
        self.obs_params = self.obs_normalizer.get_init_params()

        self._num_device = jax.local_device_count()
        if self._pop_size % self._num_device != 0:
            raise ValueError(
                'pop_size must be multiples of GPU/TPUs: '
                'pop_size={}, #devices={}'.format(
                    self._pop_size, self._num_device))
        if self._n_evaluations % self._num_device != 0:
            raise ValueError(
                'n_evaluations must be multiples of GPU/TPUs: '
                'n_evaluations={}, #devices={}'.format(
                    self._n_evaluations, self._num_device))

        def step_once_gen(carry, input_data, task):
            (task_state, policy_state, params_gen, params_disc, params_q, obs_params, t,
             accumulated_reward_adv, accumulated_reward_mi, disc_logits, loss_g, valid_mask) = carry
            if task.multi_agent_training:
                num_tasks, num_agents = task_state.obs.shape[:2]
                task_state = task_state.replace(
                    obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:])))
            org_obs = task_state.obs
            normed_obs = self.obs_normalizer.normalize_obs(org_obs, obs_params)
            task_state = task_state.replace(obs=normed_obs)
            #jax.debug.print('task state batch stats gen shape : {}', task_state.batch_stats_gen.shape)
            #jax.debug.print('params gen shape in step_once_gen : {}', params_gen.shape)
            fake_imgs, actions, disc_logits, batch_stats_gen, batch_stats_disc, policy_state = policy_net.get_actions(
                task_state, params_gen, params_disc, params_q, policy_state)
            #act1, act2, act3 = activations
            #jax.debug.print('activations 1 shape : {}', act1.shape)
            #jax.debug.print('activations 2 shape : {}', act2.shape)
            #jax.debug.print('activations 3 shape : {}', act3.shape)
            #jax.debug.print('bin logits : {}', bin_logits)
            #leaves_batch_stats_gen = jax.tree_util.tree_flatten(batch_stats_gen[0])
            #flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])
            #leaves_batch_stats_disc = jax.tree_util.tree_flatten(batch_stats_disc[0])
            #flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])
            task_state = task_state.replace(batch_stats_gen=batch_stats_gen)
            task_state = task_state.replace(batch_stats_disc=batch_stats_disc)
            
            if task.multi_agent_training:
                task_state = task_state.replace(
                    obs=task_state.obs.reshape(
                        (num_tasks, num_agents, *task_state.obs.shape[1:])))
                actions = actions.reshape(
                    (num_tasks, num_agents, *actions.shape[1:]))
            task_state, loss_mi, loss_g, done = task.step(task_state, actions, disc_logits)
            #jax.debug.print('loss_mi : {}', loss_mi.shape)
            #jax.debug.print('loss_g : {}', loss_g.shape)
            #jax.debug.print('loss_con : {}', loss_con.shape)
           
            # get standard deviation of loss values
            #loss_g_std = jnp.std(loss_g)
            #loss_mi_std = jnp.std(loss_mi)

            # get ratio of std g to std mi
            #loss_ratio = loss_g_std / (loss_mi_std + 1e-8)
            #jax.debug.print('self t : {}', t)
           
            #loss_ratio = loss_ratio * t
            #loss_ratio = loss_ratio * 0.4
            #jax.debug.print('loss_ratio : {}', loss_ratio)
            #loss_g = (loss_g * (1 / (loss_ratio + 1e-8))) + loss_mi
            #loss_g = loss_g + (loss_mi * loss_ratio)
            # use ema to standardize the loss values
            #prev_mean_mi = task_state.mean_mi
            #prev_mean_g = task_state.mean_g
            #prev_mean_con = task_state.mean_con

            #prev_var_mi = task_state.var_mi
            #prev_var_g = task_state.var_g
            #prev_var_con = task_state.var_con

            #curr_mean_mi = jnp.mean(loss_mi)
            #curr_mean_g = jnp.mean(loss_g)
            #curr_mean_con = jnp.mean(loss_con)

            #curr_var_mi = jnp.var(loss_mi) + 1e-8
            #curr_var_g = jnp.var(loss_g) + 1e-8
            #curr_var_con = jnp.var(loss_con) + 1e-8
            
            #mean_mi = prev_mean_mi * 0.01 + curr_mean_mi * 0.94
            #mean_g = prev_mean_g * 0.01 + curr_mean_g * 0.94
            #mean_con = prev_mean_con * 0.01 + curr_mean_con * 0.94

            #var_mi = prev_var_mi * 0.01 + curr_var_mi * 0.94
            #var_g = prev_var_g * 0.01 + curr_var_g * 0.94
            #var_con = prev_var_con * 0.01 + curr_var_con * 0.94

            #mean_mi = curr_mean_mi * 0.95
            #mean_g = curr_mean_g * 0.95
            #mean_con = curr_mean_con * 0.95

            #var_mi = curr_var_mi * 0.95
            #var_g = curr_var_g * 0.95
            #var_con = curr_var_con * 0.95


            #loss_mi_std = (loss_mi - mean_mi.squeeze()) / (jnp.sqrt(var_mi.squeeze()) + 1e-8)
            # get absolute value of loss_mi_std
            #loss_mi_std = jnp.abs(loss_mi_std)
            #loss_g_std = (loss_g - mean_g.squeeze()) / (jnp.sqrt(var_g.squeeze()) + 1e-8)
            #loss_g_std = jnp.abs(loss_g_std)
            #loss_con_std = (loss_con - mean_con.squeeze()) / (jnp.sqrt(var_con.squeeze()) + 1e-8)

            #loss_con_std = jnp.abs(loss_con_std)
            reward_adv =  loss_g 
            reward_mi = loss_mi
            #reward_con = loss_con
            #jax.debug.print('loss_mi_std : {}', loss_mi_std)
            #jax.debug.print('loss_g_std : {}', loss_g_std)
            #jax.debug.print('loss_con_std : {}', loss_con_std)
            #reward = -loss_mi - loss_g - loss_con*0.1
            #bin_logits_avg = jnp.mean(bin_logits, axis=1)
            #bin_logits_avg = jnp.median(bin_logits, axis=1)
            #jax.debug.print('bin logits avg shape : {}', bin_logits_avg.shape)
            #jax.debug.print('valid mask shape : {}', valid_mask.shape)
            #reward_bin = bin_logits_avg
            if task.multi_agent_training:
                reward_adv = reward_adv.ravel()
                reward_mi = reward_mi.ravel()
                #reward_con = reward_con.ravel()
                #reward_bin = reward_bin.ravel()
                done = jnp.repeat(done, num_agents, axis=0)
            accumulated_reward_adv = accumulated_reward_adv + reward_adv * valid_mask
            accumulated_reward_mi = accumulated_reward_mi + reward_mi * valid_mask
            #accumulated_reward_con = accumulated_reward_con + reward_con * valid_mask
            #accumulated_reward_bin = accumulated_reward_bin + reward_bin * valid_mask
            #jax.debug.print('accumulated reward in gen step : {}', accumulated_reward.shape)
            valid_mask = valid_mask * (1 - done.ravel())
            return ((task_state, policy_state, params_gen, params_disc, params_q, obs_params, t,
                     accumulated_reward_adv, accumulated_reward_mi, disc_logits, loss_g, valid_mask),
                    (org_obs, valid_mask))

        def rollout_gen(task_states, policy_states, params_gen, params_disc, params_q, obs_params, t,
                    step_once_gen_fn, max_steps):
            accumulated_rewards_adv = jnp.zeros(params_gen.shape[0])
            accumulated_rewards_mi = jnp.zeros(params_gen.shape[0])
            disc_logits = jnp.zeros((128,128,10))
            loss_g = jnp.zeros(self._pop_size//2)
            #fake_imgs = jnp.zeros((64,128,28, 28, 1))
            valid_masks = jnp.ones(params_gen.shape[0])
            ((task_states, policy_states, params_gen, params_disc, params_q, obs_params, t,
              accumulated_rewards_adv, accumulated_rewards_mi, disc_logits, loss_g, valid_masks),
             (obs_set, obs_mask)) = jax.lax.scan(
                step_once_gen_fn,
                (task_states, policy_states, params_gen, params_disc, params_q, obs_params, t,
                 accumulated_rewards_adv, accumulated_rewards_mi, disc_logits, loss_g, valid_masks), (), max_steps)
            return accumulated_rewards_adv, accumulated_rewards_mi, obs_set, obs_mask, task_states, disc_logits, loss_g

        def step_once_disc(carry, input_data, task):
            (task_state, policy_state, params_gen, params_disc, params_q, obs_params,
             accumulated_reward_real, accumulated_reward_fake, accumulated_reward_mi, valid_mask, valid_mask_q) = carry
            if task.multi_agent_training:
                num_tasks, num_agents = task_state.obs.shape[:2]
                task_state = task_state.replace(
                    obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:])))
            org_obs = task_state.obs
            normed_obs = self.obs_normalizer.normalize_obs(org_obs, obs_params)
            task_state = task_state.replace(obs=normed_obs)
            #task_state = task_state.replace(fake_imgs=jnp.squeeze(task_state.fake_imgs, axis=0))
            real_preds, actions, disc_logits, batch_stats_disc, batch_stats_gen, policy_state = policy_net.get_actions(
                task_state, params_gen, params_disc, params_q, policy_state)
            task_state = task_state.replace(batch_stats_disc=batch_stats_disc)
            task_state = task_state.replace(batch_stats_gen=batch_stats_gen)
            
            if task.multi_agent_training:
                task_state = task_state.replace(
                    obs=task_state.obs.reshape(
                        (num_tasks, num_agents, *task_state.obs.shape[1:])))
                actions = actions.reshape(
                    (num_tasks, num_agents, *actions.shape[1:]))
            task_state, reward_real, reward_fake, reward_mi, done = task.step(task_state, real_preds, actions, disc_logits)
            #jax.debug.print('reward in disc step : {}', reward.shape)
            if task.multi_agent_training:
                reward_real = reward_real.ravel()
                reward_fake = reward_fake.ravel()
                reward_mi = reward_mi.ravel()
                done = jnp.repeat(done, num_agents, axis=0)
            accumulated_reward_real = accumulated_reward_real + reward_real * valid_mask
            accumulated_reward_fake = accumulated_reward_fake + reward_fake * valid_mask

            accumulated_reward_mi = accumulated_reward_mi + reward_mi * valid_mask_q
            #jax.debug.print('accumulated reward in disc step : {}', accumulated_reward.shape)
            valid_mask = valid_mask * (1 - done.ravel())
            valid_mask_q = valid_mask_q * (1 - done.ravel())
            #task_state = task_state.replace(fake_imgs=jnp.expand_dims(task_state.fake_imgs, axis=0))
            return ((task_state, policy_state, params_gen, params_disc, params_q, obs_params,
                     accumulated_reward_real, accumulated_reward_fake, accumulated_reward_mi, valid_mask, valid_mask_q),
                    (org_obs, valid_mask, valid_mask_q))

        def rollout_disc(task_states, policy_states, params_gen, params_disc, params_q, obs_params,
                    step_once_disc_fn, max_steps):
            accumulated_rewards_real = jnp.zeros(params_disc.shape[0])
            accumulated_rewards_fake = jnp.zeros(params_disc.shape[0])
            accumulated_rewards_mi = jnp.zeros(params_q.shape[0])
            #fake_preds = jnp.zeros((28, 28, 1))
            valid_masks = jnp.ones(params_disc.shape[0])
            valid_masks_q = jnp.ones(params_q.shape[0])

            ((task_states, policy_states, params_gen, params_disc, params_q, obs_params,
              accumulated_rewards_real, accumulated_rewards_fake, accumulated_rewards_mi, valid_masks, valid_masks_q),
             (obs_set, obs_mask, obs_mask_q)) = jax.lax.scan(
                step_once_disc_fn,
                (task_states, policy_states, params_gen, params_disc, params_q, obs_params,
                 accumulated_rewards_real, accumulated_rewards_fake, accumulated_rewards_mi, valid_masks, valid_masks_q), (), max_steps)
            return accumulated_rewards_real, accumulated_rewards_fake, accumulated_rewards_mi, obs_set, obs_mask, task_states

        def step_once_valid(carry, input_data, task):
            (task_state, policy_state, params_gen, params_disc, params_q, obs_params,
             accumulated_reward_adv, accumulated_reward_mi, fake_imgs, valid_mask) = carry
            if task.multi_agent_training:
                num_tasks, num_agents = task_state.obs.shape[:2]
                task_state = task_state.replace(
                    obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:])))
            org_obs = task_state.obs
            normed_obs = self.obs_normalizer.normalize_obs(org_obs, obs_params)
            task_state = task_state.replace(obs=normed_obs)
            #jax.debug.print('task state batch stats gen shape : {}', task_state.batch_stats_gen.shape)
            #jax.debug.print('params gen shape in step_once_gen : {}', params_gen.shape)
            fake_imgs, actions, disc_logits, batch_stats_gen, batch_stats_disc, policy_state = policy_net.get_actions(
                task_state, params_gen, params_disc, params_q, policy_state)
            #leaves_batch_stats_gen = jax.tree_util.tree_flatten(batch_stats_gen[0])
            #flat_batch_stats_gen = jnp.concatenate([p.flatten() for p in leaves_batch_stats_gen])
            #leaves_batch_stats_disc = jax.tree_util.tree_flatten(batch_stats_disc[0])
            #flat_batch_stats_disc = jnp.concatenate([p.flatten() for p in leaves_batch_stats_disc])
            task_state = task_state.replace(batch_stats_gen=batch_stats_gen)
            task_state = task_state.replace(batch_stats_disc=batch_stats_disc)
            
            if task.multi_agent_training:
                task_state = task_state.replace(
                    obs=task_state.obs.reshape(
                        (num_tasks, num_agents, *task_state.obs.shape[1:])))
                actions = actions.reshape(
                    (num_tasks, num_agents, *actions.shape[1:]))
            task_state, loss_mi, loss_g, done = task.step(task_state, actions, disc_logits)
            #reward = -loss_mi - loss_g - loss_con
            #bin_logits_avg = jnp.mean(bin_logits, axis=1)
            #bin_logits_avg = jnp.median(bin_logits, axis=1)
            reward_adv = -loss_g
            reward_mi = loss_mi
            #reward_con = -loss_con
            #reward_bin = bin_logits_avg

            if task.multi_agent_training:
                reward_adv = reward_adv.ravel()
                done = jnp.repeat(done, num_agents, axis=0)
            accumulated_reward_adv = accumulated_reward_adv + reward_adv * valid_mask
            accumulated_reward_mi = accumulated_reward_mi + reward_mi * valid_mask 
            #accumulated_reward_con = accumulated_reward_con + reward_con * valid_mask
            #accumulated_reward_bin = accumulated_reward_bin + reward_bin * valid_mask

            valid_mask = valid_mask * (1 - done.ravel())
            return ((task_state, policy_state, params_gen, params_disc, params_q, obs_params,
                     accumulated_reward_adv, accumulated_reward_mi, fake_imgs, valid_mask),
                    (org_obs, valid_mask))


        def rollout_valid(task_states, policy_states, params_gen, params_disc, params_q, obs_params,
                    step_once_gen_fn, max_steps):
            accumulated_rewards_adv = jnp.zeros(params_gen.shape[0])
            #accumulated_rewards_bin = jnp.zeros(params_gen.shape[0])
            accumulated_rewards_mi = jnp.zeros(params_gen.shape[0])
            #accumulated_rewards_con = jnp.zeros(params_gen.shape[0])
            fake_imgs = jnp.zeros((128,128,28, 28, 1))
            valid_masks = jnp.ones(params_gen.shape[0])
            ((task_states, policy_states, params_gen, params_disc, params_q, obs_params,
              accumulated_rewards_adv, accumulated_rewards_mi, fake_imgs, valid_masks),
             (obs_set, obs_mask)) = jax.lax.scan(
                step_once_gen_fn,
                (task_states, policy_states, params_gen, params_disc, params_q, obs_params,
                 accumulated_rewards_adv, accumulated_rewards_mi, fake_imgs, valid_masks), (), max_steps)
            return accumulated_rewards_adv, accumulated_rewards_mi, obs_set, obs_mask, task_states, fake_imgs


        self._policy_reset_fn = jax.jit(policy_net.reset)
        self._policy_act_fn = jax.jit(policy_net.get_actions)

        if (
                hasattr(train_vec_task, 'bd_extractor') and
                train_vec_task.bd_extractor is not None
        ):
            self._bd_summarize_fn = jax.jit(
                train_vec_task.bd_extractor.summarize)
        else:
            self._bd_summarize_fn = lambda x: x

        # Set up training functions.
        self._train_reset_fn = train_vec_task.reset
        self._train_step_fn = train_vec_task.step
        self._train_max_steps = train_vec_task.max_steps
        
        self._train_rollout_gen_fn = partial(
            rollout_gen,
            step_once_gen_fn=partial(step_once_gen, task=train_vec_task),
            max_steps=train_vec_task.max_steps)
        
        self._train_rollout_disc_fn = partial(
            rollout_disc,
            step_once_disc_fn=partial(step_once_disc, task=train_vec_task),
            max_steps=train_vec_task.max_steps)
        
        if self._num_device > 1:
            self._train_rollout_gen_fn = jax.jit(jax.pmap(
                self._train_rollout_gen_fn, in_axes=(0, 0, 0, 0, 0, None, None)))
            self._train_rollout_disc_fn = jax.jit(jax.pmap(
                self._train_rollout_disc_fn, in_axes=(0, 0, 0, 0, 0, None)))


        # Set up validation functions.
        self._valid_reset_fn = valid_vec_task.reset
        self._valid_step_fn = valid_vec_task.step
        self._valid_max_steps = valid_vec_task.max_steps
        self._valid_rollout_fn = partial(
            rollout_valid,
            step_once_gen_fn=partial(step_once_valid, task=valid_vec_task),
            max_steps=valid_vec_task.max_steps)
        if self._num_device > 1:
            self._valid_rollout_fn = jax.jit(jax.pmap(
                self._valid_rollout_fn, in_axes=(0, 0, 0, 0, 0, None)))

    def eval_params(self,
                    params_gen: jnp.ndarray,
                    params_disc: jnp.ndarray,
                    params_q: jnp.ndarray,
                    batch_stats_gen: dict,
                    batch_stats_disc: dict,
                    pop_stats: jnp.ndarray,
                    disc_reset_keys_cat_code: jnp.ndarray,
                    generator: bool,
                    test: bool) -> Tuple[jnp.ndarray, TaskState]:
        """Evaluate population parameters or test the best parameter.

        Args:
            params - Parameters to be evaluated.
            test - Whether we are testing the best parameter
        Returns:
            An array of fitness scores.
        """
        if self._use_for_loop:
            return self._for_loop_eval(params_gen, params_disc, params_q, batch_stats_gen, batch_stats_disc, generator, test)
        else:
            return self._scan_loop_eval(params_gen, params_disc, params_q, batch_stats_gen, batch_stats_disc, pop_stats, disc_reset_keys_cat_code, generator, test)

    def _for_loop_eval(self,
                       params: jnp.ndarray,
                       test: bool) -> Tuple[jnp.ndarray, TaskState]:
        """Rollout using for loop (no multi-device or ma_training yet)."""
        policy_reset_func = self._policy_reset_fn
        policy_act_func = self._policy_act_fn
        if test:
            n_repeats = self._test_n_repeats
            task_reset_func = self._valid_reset_fn
            task_step_func = self._valid_step_fn
            task_max_steps = self._valid_max_steps
            params = duplicate_params(
                params[None, :], self._n_evaluations, False)
        else:
            n_repeats = self._n_repeats
            task_reset_func = self._train_reset_fn
            task_step_func = self._train_step_fn
            task_max_steps = self._train_max_steps

        params = duplicate_params(params, n_repeats, self._ma_training)

        # Start rollout.
        self._key, reset_keys = get_task_reset_keys(
            self._key, test, self._pop_size, self._n_evaluations, n_repeats,
            self._ma_training)
        task_state = task_reset_func(reset_keys)
        policy_state = policy_reset_func(task_state)
        scores = jnp.zeros(params.shape[0])
        valid_mask = jnp.ones(params.shape[0])
        start_time = time.perf_counter()
        rollout_steps = 0
        sim_steps = 0
        for i in range(task_max_steps):
            actions, policy_state = policy_act_func(
                task_state, params, policy_state)
            task_state, reward, done = task_step_func(task_state, actions)
            scores, valid_mask = update_score_and_mask(
                scores, reward, valid_mask, done)
            rollout_steps += 1
            sim_steps = sim_steps + valid_mask
            if all_done(valid_mask):
                break
        time_cost = time.perf_counter() - start_time
        self._logger.debug('{} steps/s, mean.steps={}'.format(
            int(rollout_steps * task_state.obs.shape[0] / time_cost),
            sim_steps.sum() / task_state.obs.shape[0]))

        return report_score(scores, n_repeats), task_state

    def _scan_loop_eval(self,
                        params_gen: jnp.ndarray,
                        params_disc: jnp.ndarray,
                        params_q: jnp.ndarray,
                        batch_stats_gen: dict,
                        batch_stats_disc: dict,
                        pop_stats: jnp.ndarray,
                        disc_reset_keys_cat_code: jnp.ndarray,
                        generator: bool,
                        test: bool) -> Tuple[jnp.ndarray, TaskState]:
        """Rollout using jax.lax.scan."""
        policy_reset_func = self._policy_reset_fn

        # check if first dimension of batch_stats_gen and batch_stats_disc is equal to pop_size
        if batch_stats_gen.shape[0] != self._pop_size and not test:
            # add pop size as first dimension to batch_stats_gen and batch_stats_disc
            batch_stats_gen = jnp.repeat(batch_stats_gen[None, :], self._pop_size, axis=0)
            batch_stats_disc = jnp.repeat(batch_stats_disc[None, :], self._pop_size, axis=0)

        if params_gen is not None and params_gen.shape[0] != self._pop_size:
            params_gen = jnp.repeat(params_gen[None, :], self._pop_size, axis=0)
            params_disc = jnp.repeat(params_disc[None, :], self._pop_size, axis=0)
            params_q = jnp.repeat(params_q[None, :], self._pop_size, axis=0)

        self.batch_stats_gen = batch_stats_gen
        self.batch_stats_disc = batch_stats_disc

        #if params_gen is not None:
            #jax.debug.print('params gen shape before dup : {}', params_gen.shape)

        if test: 
            n_repeats = self._n_repeats
            task_reset_func = self._valid_reset_fn
            rollout_func = self._valid_rollout_fn
        else:
            n_repeats = self._n_repeats
            task_reset_func = self._train_reset_fn
            #rollout_func = self._train_rollout_fn
            if generator:
                rollout_func = self._train_rollout_gen_fn
            else:
                rollout_func = self._train_rollout_disc_fn

        #if params_gen is not None:
            #jax.debug.print('params gen shape after test dup : {}', params_gen.shape)
        # Suppose pop_size=2 and n_repeats=3.
        # For multi-agents training, params become
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        # For non-ma training, params become
        #   a1, a2, ..., an  (individual 1 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #if generator:
        #   params_gen = duplicate_params(params_gen, n_repeats, self._ma_training)
       
        #if params_gen is not None:
            #jax.debug.print('params gen shape after dup : {}', params_gen.shape)
        params_gen = duplicate_params(params_gen, n_repeats, self._ma_training) 
        params_disc = duplicate_params(params_disc, n_repeats, self._ma_training)
        params_q = duplicate_params(params_q, n_repeats, self._ma_training)

        #split the reset keys, one for noise vect and one for cat codes
        self._key, key, noise_keys, cat_keys, con_keys = random.split(self._key, 5)

        if generator:
            noise_keys, cat_keys, con_keys, reset_keys_latent, reset_keys_cat_code, reset_keys_con_code = get_task_reset_keys(
                noise_keys, cat_keys, con_keys, self._pop_size, self._n_evaluations, n_repeats, self._ma_training)
        else:
            key, noise_keys, cat_keys, reset_keys_mnist, reset_keys_latent, reset_keys_cat_code, reset_keys_con_code = get_task_reset_keys_disc(
                key, noise_keys, cat_keys, con_keys, self._pop_size, self._n_evaluations, n_repeats,self._ma_training)

        #jax.debug.print('reset keys 1 shape : {}', reset_keys1.shape)
        #jax.debug.print('reset keys 2 shape : {}', reset_keys2.shape)
        # Reset the tasks and the policy.
        if generator:
            reset_keys_cat_code = disc_reset_keys_cat_code
            task_state = task_reset_func(reset_keys_latent, reset_keys_cat_code, reset_keys_con_code)
        else:
            task_state = task_reset_func(reset_keys_mnist, reset_keys_latent, reset_keys_cat_code, reset_keys_con_code)

        #if testing:
            # obs is shape (128, 512, 74), let's look at the first individual and first image and last ten features
            #jax.debug.print('task state obs : {}', task_state.obs[0, 0:100, -10:])
            #jax.debug.print('task state obs : {}', task_state.obs)
        #task_state.set_batch_stats(self.batch_stats_gen, self.batch_stats_disc)

        #jax.debug.print('batch_stats_gen shape : {}', self.batch_stats_gen.shape)
    
        #if not generator:
        #    task_state = task_state.replace(cat_codes=cat_codes)
        #   task_state = task_state.replace(fake_imgs=fake_imgs)
            
        #if generator and not test: 
        #    mean_mi, mean_g, mean_con, var_mi, var_g, var_con = jnp.split(pop_stats,6)
        #    #jax.debug.print('mean mi shape : {}', mean_mi.shape)
        #    mean_mi = jnp.repeat(mean_mi[None, :], self._pop_size, axis=0)
        #    mean_g = jnp.repeat(mean_g[None, :], self._pop_size, axis=0)
        #    mean_con = jnp.repeat(mean_con[None, :], self._pop_size, axis=0)
        #    var_mi = jnp.repeat(var_mi[None, :], self._pop_size, axis=0)
        #    var_g = jnp.repeat(var_g[None, :], self._pop_size, axis=0)
        #    var_con = jnp.repeat(var_con[None, :], self._pop_size, axis=0)

        #    task_state = task_state.replace(mean_mi=mean_mi)
        #    task_state = task_state.replace(mean_g=mean_g)
        #    task_state = task_state.replace(mean_con=mean_con)
        #    task_state = task_state.replace(var_mi=var_mi)
        #    task_state = task_state.replace(var_g=var_g)
        #    task_state = task_state.replace(var_con=var_con)

        task_state = task_state.replace(batch_stats_gen=self.batch_stats_gen)

        task_state = task_state.replace(batch_stats_disc=self.batch_stats_disc)

        policy_state = policy_reset_func(task_state)
        
        if self._num_device > 1: #and not test:
            #if generator:
            params_gen = split_params_for_pmap(params_gen)
            batch_stats_gen = split_params_for_pmap(batch_stats_gen)
            #if not generator:
            #    fake_imgs = split_params_for_pmap(fake_imgs)
            params_disc = split_params_for_pmap(params_disc)
            task_state = split_states_for_pmap(task_state)
            params_q = split_params_for_pmap(params_q)
            policy_state = split_states_for_pmap(policy_state)
            batch_stats_disc = split_params_for_pmap(batch_stats_disc)

        #jax.debug.print('obs params : {}', self.obs_params)
        # Do the rollouts.
        #if generator:
        if test:
            scores_adv, scores_mi, all_obs, masks, final_states, fake_imgs = rollout_func(
                task_state, policy_state, params_gen, params_disc, params_q, self.obs_params)
        elif generator:
            scores_adv, scores_mi, all_obs, masks, final_states, disc_logits, loss_g = rollout_func(
            task_state, policy_state, params_gen, params_disc, params_q, self.obs_params, self._i)
        else: 
            scores_real, scores_fake, scores_mi, all_obs, masks, final_states = rollout_func(
                task_state, policy_state, params_gen, params_disc, params_q, self.obs_params)

        if self._num_device > 1:
            all_obs = reshape_data_from_pmap(all_obs)
            masks = reshape_data_from_pmap(masks)
            final_states = merge_state_from_pmap(final_states)
            if generator and not test:
                disc_logits = reshape_data_from_pmap(disc_logits)
                #disc_logits = combine_ind_scalar(disc_logits)
                loss_g = combine_ind_scalar(loss_g)
                #loss_con = combine_ind_scalar(loss_con)
            #    fake_imgs = combine_ind_scalar(fake_imgs)

        #if generator and not test:
        #    # update the population mean_mi, mean_g, mean_con, var_mi, var_g, var_con in the task state
        #    curr_mean_mi = jnp.mean(loss_mi)
        #    curr_mean_g = jnp.mean(loss_g)
        #    curr_mean_con = jnp.mean(loss_con)

        #    curr_var_mi = jnp.var(loss_mi) + 1e-8
        #    curr_var_g = jnp.var(loss_g) + 1e-8
        #    curr_var_con = jnp.var(loss_con) + 1e-8

        #    prev_mean_mi = jnp.mean(final_states.mean_mi)
        #    prev_mean_g = jnp.mean(final_states.mean_g)
        #    prev_mean_con = jnp.mean(final_states.mean_con)

        #    prev_var_mi = jnp.mean(final_states.var_mi)
        #    prev_var_g = jnp.mean(final_states.var_g)
        #    prev_var_con = jnp.mean(final_states.var_con)

        #    mean_mi = prev_mean_mi * 0.1 + curr_mean_mi * 0.9
        #    mean_g = prev_mean_g * 0.1 + curr_mean_g * 0.9
        #    mean_con = prev_mean_con * 0.1 + curr_mean_con * 0.9

        #    var_mi = prev_var_mi * 0.1 + curr_var_mi * 0.9
        #    var_g = prev_var_g * 0.1 + curr_var_g * 0.9
        #    var_con = prev_var_con * 0.1 + curr_var_con * 0.9

        #    pop_stats = jnp.array([mean_mi, mean_g, mean_con, var_mi, var_g, var_con])


        batch_stats_gen_updated = final_states.batch_stats_gen
        batch_stats_disc_updated = final_states.batch_stats_disc

        #noise_keys, cat_keys = random.split(self._key, 2)

        #if generator:
        #    self._key, reset_keys1, reset_keys2 = get_task_reset_keys_testing(
        #        noise_keys, cat_keys, testing, self._pop_size, self._n_evaluations, n_repeats, self._ma_training)

        #if generator:
        #    task_state = task_reset_func(reset_keys1, reset_keys2)

        ##if generator:
        #    cat_codes = task_state.cat_codes
        #else:
        #    cat_codes = None
        if not test:
            fake_imgs = None

        #jax.debug.print('scores shape before mean : {}', scores.shape)
        #jax.debug.print('final_states shape : {}', final_states.obs.shape)
        if not test and not self.obs_normalizer.is_dummy:
            self.obs_params = self.obs_normalizer.update_normalization_params(
                obs_buffer=all_obs, obs_mask=masks, obs_params=self.obs_params)

        if self._ma_training:
            if not test and generator:
                # In training, each agent has different parameters.
                scores_adv = jnp.mean(
                    scores_adv.ravel().reshape((n_repeats, -1)), axis=0)
                scores_mi = jnp.mean(
                    scores_mi.ravel().reshape((n_repeats, -1)), axis=0)
                #scores_con = jnp.mean(
                    #scores_con.ravel().reshape((n_repeats, -1)), axis=0)
                #scores_bin = jnp.mean(
                #    scores_bin.ravel().reshape((n_repeats, -1)), axis=0)
            elif not test and not generator:
                # In training, they share the same parameters.
                scores_real = jnp.mean(
                    scores_real.ravel().reshape((n_repeats, -1)), axis=0)
                scores_fake = jnp.mean(
                    scores_fake.ravel().reshape((n_repeats, -1)), axis=0)
                score_mi = jnp.mean(
                    scores_mi.ravel().reshape((n_repeats, -1)), axis=0)
            else:
                # In tests, they share the same parameters.
                scores_adv = jnp.mean(
                    scores_adv.ravel().reshape((n_repeats, -1)), axis=1)
                scores_mi = jnp.mean(
                    scores_mi.ravel().reshape((n_repeats, -1)), axis=1)
                #scores_con = jnp.mean(
                #    scores_con.ravel().reshape((n_repeats, -1)), axis=1)
                #scores_bin = jnp.mean(
                #    scores_bin.ravel().reshape((n_repeats, -1)), axis=1)
        elif not test and not generator:
            scores_real = jnp.mean(scores_real.ravel().reshape((-1, n_repeats)), axis=-1)
            scores_fake = jnp.mean(scores_fake.ravel().reshape((-1, n_repeats)), axis=-1)
            scores_mi = jnp.mean(scores_mi.ravel().reshape((-1, n_repeats)), axis=-1)
        else:
            scores_adv = jnp.mean(
                scores_adv.ravel().reshape((-1, n_repeats)), axis=-1)
            scores_mi = jnp.mean(
                scores_mi.ravel().reshape((-1, n_repeats)), axis=-1)
            #scores_con = jnp.mean(
            #    scores_con.ravel().reshape((-1, n_repeats)), axis=-1)
            #scores_bin = jnp.mean(
            #    scores_bin.ravel().reshape((-1, n_repeats)), axis=-1)

        #jax.debug.print('scores shape after mean : {} ', scores.shape)
        # Note: QD methods do not support ma_training for now.
        #if not self._ma_training:
        #    final_states = tree_map(
        #        lambda x: x.reshape((scores.shape[0], n_repeats, *x.shape[1:])),
        #        final_states)

        #if not generator:
        #self._t = self._t + 1
        #
        #if self._t > 7000:
        #    self._i = 0.8
        #elif self._t > 15000:
        #    self._i = 0.6
        #elif self._t > 30000:
        #    self._i = 0.4

        if not test and not generator:
            scores1 = scores_real
            scores2 = scores_fake
            scores3 = scores_mi
        elif generator and not test:
            scores1 = scores_adv
            scores2 = scores_mi
            scores3 = disc_logits
        else:
            scores1 = scores_adv
            scores2 = scores_mi
            scores3 = None
        #self._key = new_key
        return scores1, scores2, scores3, self._bd_summarize_fn(final_states), batch_stats_gen_updated, batch_stats_disc_updated, fake_imgs, pop_stats, reset_keys_cat_code

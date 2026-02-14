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

@partial(jax.jit, static_argnums=(1,2,3,4))
def get_task_reset_keys(key: jnp.ndarray,
                        pop_size: int,
                        n_tests: int,
                        n_repeats: int,
                        ma_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    # Split the first key
    key, subkey1, subkey2, subkey3 = random.split(key,4)
    
    reset_keys1 = random.split(subkey1, n_repeats)
    reset_keys1 = jnp.tile(reset_keys1, (pop_size, 1))

    reset_keys2 = random.split(subkey2, n_repeats)
    reset_keys2 = jnp.tile(reset_keys2, (pop_size, 1))

    reset_keys3 = random.split(subkey3, n_repeats)
    reset_keys3 = jnp.tile(reset_keys3, (pop_size, 1))

    return key, reset_keys1, reset_keys2, reset_keys3

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def get_task_reset_keys_disc(key: jnp.ndarray,
                        pop_size: int,
                        n_tests: int,
                        n_repeats: int,
                        ma_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Split the first key
    key, subkey1, subkey2, subkey3, subkey4 = random.split(key,5)
    
    reset_keys1 = random.split(subkey1, n_repeats)
    reset_keys1 = jnp.tile(reset_keys1, (pop_size, 1))

    reset_keys2 = random.split(subkey2, n_repeats)
    reset_keys2 = jnp.tile(reset_keys2, (pop_size, 1))

    reset_keys3 = random.split(subkey3, n_repeats)
    reset_keys3 = jnp.tile(reset_keys3, (pop_size, 1))

    reset_keys4 = random.split(subkey4, n_repeats)
    reset_keys4 = jnp.tile(reset_keys4, (pop_size, 1))

    return key, reset_keys1, reset_keys2, reset_keys3, reset_keys4

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

# reshape metrics (2,pop_size//2, 10) to (pop_size, 10) and (2, pop_size//2, 10, 128) to (pop_size, 10, 128)
@jax.jit
def reshape_metrics_from_pmap(data: jnp.ndarray) -> jnp.ndarray:
    data = data.transpose([1, 0] + [i for i in range(2, data.ndim)])
    return jnp.reshape(data, (data.shape[0] * data.shape[1], *data.shape[2:]))

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
            (task_state, policy_state, params_gen, params_disc, obs_params, t,
             accumulated_reward_adv, accumulated_reward_mi, disc_logits, mean_var_fake, sum_per_cat_code, count_per_cat_code, r_cons, r_sense, r_intra, normative_penalty, safety_ratios, spreads, valid_mask) = carry
            if task.multi_agent_training:
                num_tasks, num_agents = task_state.obs.shape[:2]
                task_state = task_state.replace(
                    obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:])))
            org_obs = task_state.obs
            normed_obs = self.obs_normalizer.normalize_obs(org_obs, obs_params)
            task_state = task_state.replace(obs=normed_obs)
            actions, disc_logits, mean_var_fake, q_flat, policy_state = policy_net.get_actions(
                task_state, params_gen, params_disc, policy_state)

            if task.multi_agent_training:
                task_state = task_state.replace(
                    obs=task_state.obs.reshape(
                        (num_tasks, num_agents, *task_state.obs.shape[1:])))
                actions = actions.reshape(
                    (num_tasks, num_agents, *actions.shape[1:]))
            task_state, loss_mi, loss_g, sum_per_cat_code, count_per_cat_code, r_cons, r_sense, r_intra, normative_pen, safety_ratios, spreads, done = task.step(task_state, actions, disc_logits, q_flat)

            reward_adv =  loss_g
            reward_mi = loss_mi

            if task.multi_agent_training:
                reward_adv = reward_adv.ravel()
                reward_mi = reward_mi.ravel()
                done = jnp.repeat(done, num_agents, axis=0)
            accumulated_reward_adv = accumulated_reward_adv + reward_adv * valid_mask
            accumulated_reward_mi = accumulated_reward_mi + reward_mi * valid_mask
            valid_mask = valid_mask * (1 - done.ravel())

            return ((task_state, policy_state, params_gen, params_disc, obs_params, t,
                     accumulated_reward_adv, accumulated_reward_mi, disc_logits, mean_var_fake, sum_per_cat_code, count_per_cat_code, r_cons, r_sense, r_intra, normative_pen, safety_ratios, spreads, valid_mask),
                    (org_obs, valid_mask))

        def rollout_gen(task_states, policy_states, params_gen, params_disc, obs_params, t,
                    step_once_gen_fn, max_steps):
            accumulated_rewards_adv = jnp.zeros(params_gen.shape[0])
            accumulated_rewards_mi = jnp.zeros(params_gen.shape[0])
            disc_logits = jnp.zeros((256,64,10))
            mean_var_fake = jnp.zeros(self._pop_size//2) #//2
            sum_per_cat_code = jnp.zeros((self._pop_size//2,10, 256))
            count_per_cat_code = jnp.zeros((self._pop_size//2,10))
            r_cons = jnp.zeros(self._pop_size//2)
            r_sense = jnp.zeros(self._pop_size//2)
            r_intra = jnp.zeros(self._pop_size//2)
            normative_penalty = jnp.zeros(self._pop_size//2)
            safety_ratios = jnp.zeros((self._pop_size//2, 10,10))
            spreads = jnp.zeros((self._pop_size//2, 10,1))
            valid_masks = jnp.ones(params_gen.shape[0])
            ((task_states, policy_states, params_gen, params_disc, obs_params, t,
              accumulated_rewards_adv, accumulated_rewards_mi, disc_logits, mean_var_fake, sum_per_cat_code, count_per_cat_code, r_cons, r_sense, r_intra, normative_penalty, safety_ratios, spreads, valid_masks),
             (obs_set, obs_mask)) = jax.lax.scan(
                step_once_gen_fn,
                (task_states, policy_states, params_gen, params_disc, obs_params, t,
                 accumulated_rewards_adv, accumulated_rewards_mi, disc_logits, mean_var_fake, sum_per_cat_code, count_per_cat_code, r_cons, r_sense, r_intra, normative_penalty, safety_ratios, spreads, valid_masks), (), max_steps)
            return accumulated_rewards_adv, accumulated_rewards_mi, obs_set, obs_mask, task_states, disc_logits, mean_var_fake, sum_per_cat_code, count_per_cat_code, r_cons, r_sense, r_intra, normative_penalty, safety_ratios, spreads

        def step_once_valid(carry, input_data, task):
            (task_state, policy_state, params_gen, params_disc, obs_params,
             accumulated_reward_adv, accumulated_reward_mi, fake_imgs, valid_mask) = carry
            if task.multi_agent_training:
                num_tasks, num_agents = task_state.obs.shape[:2]
                task_state = task_state.replace(
                    obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:])))
            org_obs = task_state.obs
            normed_obs = self.obs_normalizer.normalize_obs(org_obs, obs_params)
            task_state = task_state.replace(obs=normed_obs)
            fake_imgs, actions, disc_logits, policy_state = policy_net.get_actions(
                task_state, params_gen, params_disc, policy_state)

            if task.multi_agent_training:
                task_state = task_state.replace(
                    obs=task_state.obs.reshape(
                        (num_tasks, num_agents, *task_state.obs.shape[1:])))
                actions = actions.reshape(
                    (num_tasks, num_agents, *actions.shape[1:]))
            task_state, loss_mi, loss_g, done = task.step(task_state, actions, disc_logits)
            reward_adv = -loss_g
            reward_mi = loss_mi

            if task.multi_agent_training:
                reward_adv = reward_adv.ravel()
                done = jnp.repeat(done, num_agents, axis=0)
            accumulated_reward_adv = accumulated_reward_adv + reward_adv * valid_mask
            accumulated_reward_mi = accumulated_reward_mi + reward_mi * valid_mask

            valid_mask = valid_mask * (1 - done.ravel())
            return ((task_state, policy_state, params_gen, params_disc, obs_params,
                     accumulated_reward_adv, accumulated_reward_mi, fake_imgs, valid_mask),
                    (org_obs, valid_mask))


        def rollout_valid(task_states, policy_states, params_gen, params_disc, obs_params,
                    step_once_gen_fn, max_steps):
            accumulated_rewards_adv = jnp.zeros(params_gen.shape[0])
            accumulated_rewards_mi = jnp.zeros(params_gen.shape[0])
            fake_imgs = jnp.zeros((256,64,28, 28, 1))
            valid_masks = jnp.ones(params_gen.shape[0])
            ((task_states, policy_states, params_gen, params_disc, obs_params,
              accumulated_rewards_adv, accumulated_rewards_mi, fake_imgs, valid_masks),
             (obs_set, obs_mask)) = jax.lax.scan(
                step_once_gen_fn,
                (task_states, policy_states, params_gen, params_disc, obs_params,
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
        
        #self._train_rollout_disc_fn = partial(
        #    rollout_disc,
        #    step_once_disc_fn=partial(step_once_disc, task=train_vec_task),
        #    max_steps=train_vec_task.max_steps)
        
        if self._num_device > 1:
            self._train_rollout_gen_fn = jax.jit(jax.pmap(
                self._train_rollout_gen_fn, in_axes=(0, 0, 0, 0, None, None)))
            #self._train_rollout_disc_fn = jax.jit(jax.pmap(
            #    self._train_rollout_disc_fn, in_axes=(0, 0, 0, 0, 0, None)))


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
                self._valid_rollout_fn, in_axes=(0, 0, 0, 0, None)))

    def eval_params(self,
                    params_gen: jnp.ndarray,
                    params_disc: jnp.ndarray,
                    #params_q: jnp.ndarray,
                    batch_stats_disc: dict,
                    #batch_stats_q: dict,
                    #features: jnp.ndarray,
                    # topographic_ks tuple of two arrays
                    topographic_ks: Tuple[jnp.ndarray, jnp.ndarray],
                    normative_ks: Tuple[jnp.ndarray, jnp.ndarray],
                    #pop_stats: jnp.ndarray,
                    #disc_reset_keys_cat_code: jnp.ndarray,
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
            return self._for_loop_eval(params_gen, params_disc, batch_stats_disc, topographic_ks, normative_ks, generator, test)
        else:
            return self._scan_loop_eval(params_gen, params_disc, batch_stats_disc, topographic_ks, normative_ks, generator, test)

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
                        #params_q: jnp.ndarray,
                        batch_stats_disc: dict,
                        #features: jnp.ndarray,
                        topographic_ks: Tuple[jnp.ndarray, jnp.ndarray],
                        normative_ks: Tuple[jnp.ndarray, jnp.ndarray],
                        #batch_stats_q: dict,
                        #pop_stats: jnp.ndarray,
                        #disc_reset_keys_cat_code: jnp.ndarray,
                        generator: bool,
                        test: bool) -> Tuple[jnp.ndarray, TaskState]:
        

        history_centroids, history_velocitys = topographic_ks
        pop_avg_spread, pop_min_safety = normative_ks

        history_centroids = jnp.ravel(history_centroids)
        history_velocitys = jnp.ravel(history_velocitys)

        """Rollout using jax.lax.scan."""
        policy_reset_func = self._policy_reset_fn
        
        if params_gen is not None and params_gen.shape[0] != self._pop_size:
            params_gen = jnp.repeat(params_gen[None, :], self._pop_size, axis=0)
            
        if batch_stats_disc.shape[0] != self._pop_size and len(batch_stats_disc.shape) == 2: #and not test:
            # add pop size as first dimension to batch_stats_gen and batch_stats_disc
            batch_stats_disc = jnp.repeat(batch_stats_disc[None, :], self._pop_size, axis=0)

        if len(batch_stats_disc.shape) == 1:
            batch_stats_disc = jnp.repeat(batch_stats_disc[None, :], self._pop_size, axis=0)

        if params_disc is not None and params_disc.shape[0] != self._pop_size:
            params_disc = jnp.repeat(params_disc[None, :], self._pop_size, axis=0)

        if history_centroids.shape[0] != self._pop_size:
            history_centroids = jnp.repeat(history_centroids[None, :], self._pop_size, axis=0)
            history_velocitys = jnp.repeat(history_velocitys[None, :], self._pop_size, axis=0)
            pop_avg_spread = jnp.repeat(pop_avg_spread[None, :], self._pop_size, axis=0)
            pop_min_safety = jnp.repeat(pop_min_safety[None, :], self._pop_size, axis=0)
        
        self.batch_stats_disc = batch_stats_disc

        if test: 
            n_repeats = self._n_repeats
            task_reset_func = self._valid_reset_fn
            rollout_func = self._valid_rollout_fn
        else:
            n_repeats = self._n_repeats
            task_reset_func = self._train_reset_fn
            rollout_func = self._train_rollout_gen_fn

        params_gen = duplicate_params(params_gen, n_repeats, self._ma_training) 
        params_disc = duplicate_params(params_disc, n_repeats, self._ma_training)

        self._key, reset_keys_latent, reset_keys_cat_code, reset_keys_con_code = get_task_reset_keys(
            self._key, self._pop_size, self._n_evaluations, n_repeats, self._ma_training)

        # Reset the tasks and the policy.
            #reset_keys_cat_code = disc_reset_keys_cat_code
        task_state = task_reset_func(reset_keys_latent, reset_keys_cat_code, reset_keys_con_code)
    
        #task_state = task_state.replace(var_con=var_con)

        task_state = task_state.replace(batch_stats_disc=self.batch_stats_disc)
        task_state = task_state.replace(hist_centroids=history_centroids) 
        task_state = task_state.replace(hist_velocity=history_velocitys)
        task_state = task_state.replace(pop_avg_spread=pop_avg_spread)
        task_state = task_state.replace(pop_min_safety=pop_min_safety)

        #task_state = task_state.replace(batch_stats_q=self.batch_stats_q)

        #if not generator:
        #    digit_counts = {digit: jnp.sum(task_state.labels[0]== digit) for digit in range(10)}
        #    jax.debug.print('digit counts : {}', digit_counts)
        
        policy_state = policy_reset_func(task_state)
        
        if self._num_device > 1: #and not test:
            #if generator:
            params_gen = split_params_for_pmap(params_gen)
            #if not generator:
            #    fake_imgs = split_params_for_pmap(fake_imgs)
            params_disc = split_params_for_pmap(params_disc)
            task_state = split_states_for_pmap(task_state)
            #params_q = split_params_for_pmap(params_q)
            policy_state = split_states_for_pmap(policy_state)
            batch_stats_disc = split_params_for_pmap(batch_stats_disc)
            #batch_stats_q = split_params_for_pmap(batch_stats_q)

        #jax.debug.print('obs params : {}', self.obs_params)
        # Do the rollouts.
        #if generator:
        if test:
            scores_adv, scores_mi, all_obs, masks, final_states, fake_imgs = rollout_func(
                task_state, policy_state, params_gen, params_disc, self.obs_params)
        else:
            scores_adv, scores_mi, all_obs, masks, final_states, disc_logits, mean_var_fake, sum_per_cat_code, count_per_cat_code, r_cons, r_sense, r_intra, normative_penalty, safety_ratios, spreads = rollout_func(
            task_state, policy_state, params_gen, params_disc, self.obs_params, self._i)

        if self._num_device > 1:
            all_obs = reshape_data_from_pmap(all_obs)
            masks = reshape_data_from_pmap(masks)
            final_states = merge_state_from_pmap(final_states)
            if generator and not test:
                disc_logits = reshape_data_from_pmap(disc_logits)

                sum_per_cat_code = reshape_metrics_from_pmap(sum_per_cat_code)
                count_per_cat_code = reshape_metrics_from_pmap(count_per_cat_code)
                #r_cons = combine_ind_scalar(r_cons)

        batch_stats_disc_updated = final_states.batch_stats_disc
        #batch_stats_q_updated = final_states.batch_stats_q
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
                mean_var_fake = jnp.mean(
                    mean_var_fake.ravel().reshape((n_repeats, -1)), axis=0)
                r_cons = jnp.mean(
                    r_cons.ravel().reshape((n_repeats, -1)), axis=0)
                r_sense = jnp.mean(
                    r_sense.ravel().reshape((n_repeats, -1)), axis=0)
                r_intra = jnp.mean(
                    r_intra.ravel().reshape((n_repeats, -1)), axis=0)
                normative_penalty = jnp.mean(
                    normative_penalty.ravel().reshape((n_repeats, -1)), axis=0)
                safety_ratios = jnp.mean(
                    safety_ratios.ravel().reshape((n_repeats, -1)), axis=0)
                spreads = jnp.mean(
                    spreads.ravel().reshape((n_repeats, -1)), axis=0)
            else:
                # In tests, they share the same parameters.
                scores_adv = jnp.mean(
                    scores_adv.ravel().reshape((n_repeats, -1)), axis=1)
                scores_mi = jnp.mean(
                    scores_mi.ravel().reshape((n_repeats, -1)), axis=1)
        else:
            scores_adv = jnp.mean(
                scores_adv.ravel().reshape((-1, n_repeats)), axis=-1)
            scores_mi = jnp.mean(
                scores_mi.ravel().reshape((-1, n_repeats)), axis=-1)
            mean_var_fake = jnp.mean(
                mean_var_fake.ravel().reshape((-1, n_repeats)), axis=-1)
            r_cons = jnp.mean(
                r_cons.ravel().reshape((-1, n_repeats)), axis=-1)
            r_sense = jnp.mean(
                r_sense.ravel().reshape((-1, n_repeats)), axis=-1)
            r_intra = jnp.mean(
                r_intra.ravel().reshape((-1, n_repeats)), axis=-1)
            normative_penalty = jnp.mean(
                normative_penalty.ravel().reshape((-1, n_repeats)), axis=-1)
            safety_ratios = jnp.mean(
                safety_ratios.ravel().reshape((-1, n_repeats)), axis=-1)
            spreads = jnp.mean(
                spreads.ravel().reshape((-1, n_repeats)), axis=-1)

        if generator and not test:
            sum_pop = sum_per_cat_code.sum(axis=0)      # [K, F]
            cnt_pop = count_per_cat_code.sum(axis=0)    # [K]
            avg_per_code = sum_pop / jnp.maximum(cnt_pop[:, None], 0.00008)  # [K, F]


        #jax.debug.print('scores adv after mean : {}', scores_adv.shape)
        #jax.debug.print('mean var fake after mean : {}', mean_var_fake.shape)


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

        if generator and not test:
            scores1 = scores_adv
            scores2 = scores_mi
            scores4 = disc_logits
        else:
            scores1 = scores_adv
            scores2 = scores_mi
            scores4 = None
            avg_per_code = None
        #self._key = new_key
        return scores1, scores2, scores4, self._bd_summarize_fn(final_states), batch_stats_disc_updated, mean_var_fake, avg_per_code, r_cons, r_sense, r_intra, normative_penalty, safety_ratios, spreads

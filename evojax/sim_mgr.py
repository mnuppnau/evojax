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

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def get_task_reset_keys_testing(key1: jnp.ndarray,
                        key2: jnp.ndarray,
                        test: bool,
                        pop_size: int,
                        n_tests: int,
                        n_repeats: int,
                        ma_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Split the first key
    key1, subkey1 = random.split(key=key1)
    
    # Split both keys under "if testing" condition
    key2, subkey2 = random.split(key=key2)
    
    reset_keys1 = random.split(subkey1, n_repeats)
    reset_keys1 = jnp.tile(reset_keys1, (pop_size, 1))

    reset_keys2 = random.split(subkey2, n_repeats)
    reset_keys2 = jnp.tile(reset_keys2, (pop_size, 1))

    return key1, reset_keys1, reset_keys2

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def get_task_reset_keys(key: jnp.ndarray,
                        test: bool,
                        pop_size: int,
                        n_tests: int,
                        n_repeats: int,
                        ma_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key, subkey = random.split(key=key)
    if ma_training:
        reset_keys = random.split(subkey, n_repeats)
    else:
        if test:
            reset_keys = random.split(subkey, n_tests * n_repeats)
        else:
            reset_keys = random.split(subkey, n_repeats)
            reset_keys = jnp.tile(reset_keys, (pop_size, 1))
    return key, reset_keys


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
            (task_state, policy_state, params_gen, params_disc, obs_params,
             accumulated_reward, fake_imgs, valid_mask) = carry
            if task.multi_agent_training:
                num_tasks, num_agents = task_state.obs.shape[:2]
                task_state = task_state.replace(
                    obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:])))
            org_obs = task_state.obs
            normed_obs = self.obs_normalizer.normalize_obs(org_obs, obs_params)
            task_state = task_state.replace(obs=normed_obs)
            #jax.debug.print('task state batch stats gen shape : {}', task_state.batch_stats_gen.shape)
            #jax.debug.print('params gen shape in step_once_gen : {}', params_gen.shape)
            fake_imgs, actions, q, batch_stats_gen, batch_stats_disc, policy_state = policy_net.get_actions(
                task_state, params_gen, params_disc, policy_state)
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
            task_state, reward, done = task.step(task_state, actions, q)
            if task.multi_agent_training:
                reward = reward.ravel()
                done = jnp.repeat(done, num_agents, axis=0)
            accumulated_reward = accumulated_reward + reward * valid_mask
            valid_mask = valid_mask * (1 - done.ravel())
            return ((task_state, policy_state, params_gen, params_disc, obs_params,
                     accumulated_reward, fake_imgs, valid_mask),
                    (org_obs, valid_mask))

        def rollout_gen(task_states, policy_states, params_gen, params_disc, obs_params,
                    step_once_gen_fn, max_steps):
            accumulated_rewards = jnp.zeros(params_gen.shape[0])
            fake_imgs = jnp.zeros((64,128,28, 28, 1))
            valid_masks = jnp.ones(params_gen.shape[0])
            ((task_states, policy_states, params_gen, params_disc, obs_params,
              accumulated_rewards, fake_imgs, valid_masks),
             (obs_set, obs_mask)) = jax.lax.scan(
                step_once_gen_fn,
                (task_states, policy_states, params_gen, params_disc, obs_params,
                 accumulated_rewards, fake_imgs, valid_masks), (), max_steps)
            return accumulated_rewards, obs_set, obs_mask, task_states, fake_imgs



        def step_once_disc(carry, input_data, task):
            (task_state, policy_state, params_disc, obs_params,
             accumulated_reward, valid_mask) = carry
            if task.multi_agent_training:
                num_tasks, num_agents = task_state.obs.shape[:2]
                task_state = task_state.replace(
                    obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:])))
            org_obs = task_state.obs
            normed_obs = self.obs_normalizer.normalize_obs(org_obs, obs_params)
            task_state = task_state.replace(obs=normed_obs)
            task_state = task_state.replace(fake_imgs=jnp.squeeze(task_state.fake_imgs, axis=0))
            real_preds, actions, q, batch_stats_disc, policy_state = policy_net.get_actions(
                task_state, params_disc, task_state.fake_imgs, policy_state)
            task_state = task_state.replace(batch_stats_disc=batch_stats_disc)
            if task.multi_agent_training:
                task_state = task_state.replace(
                    obs=task_state.obs.reshape(
                        (num_tasks, num_agents, *task_state.obs.shape[1:])))
                actions = actions.reshape(
                    (num_tasks, num_agents, *actions.shape[1:]))
            task_state, reward, done = task.step(task_state, real_preds, actions, q)
            if task.multi_agent_training:
                reward = reward.ravel()
                done = jnp.repeat(done, num_agents, axis=0)
            accumulated_reward = accumulated_reward + reward * valid_mask
            valid_mask = valid_mask * (1 - done.ravel())
            task_state = task_state.replace(fake_imgs=jnp.expand_dims(task_state.fake_imgs, axis=0))
            return ((task_state, policy_state, params_disc, obs_params,
                     accumulated_reward, valid_mask),
                    (org_obs, valid_mask))

        def rollout_disc(task_states, policy_states, params_disc, obs_params,
                    step_once_disc_fn, max_steps):
            accumulated_rewards = jnp.zeros(params_disc.shape[0])
            #fake_preds = jnp.zeros((28, 28, 1))
            valid_masks = jnp.ones(params_disc.shape[0])
            ((task_states, policy_states, params_disc, obs_params,
              accumulated_rewards, valid_masks),
             (obs_set, obs_mask)) = jax.lax.scan(
                step_once_disc_fn,
                (task_states, policy_states, params_disc, obs_params,
                 accumulated_rewards, valid_masks), (), max_steps)
            return accumulated_rewards, obs_set, obs_mask, task_states

        def rollout_valid(task_states, policy_states, params_gen, params_disc, obs_params,
                    step_once_gen_fn, max_steps):
            accumulated_rewards = jnp.zeros(params_gen.shape[0])
            fake_imgs = jnp.zeros((64,10,28, 28, 1))
            valid_masks = jnp.ones(params_gen.shape[0])
            ((task_states, policy_states, params_gen, params_disc, obs_params,
              accumulated_rewards, fake_imgs, valid_masks),
             (obs_set, obs_mask)) = jax.lax.scan(
                step_once_gen_fn,
                (task_states, policy_states, params_gen, params_disc, obs_params,
                 accumulated_rewards, fake_imgs, valid_masks), (), max_steps)
            return accumulated_rewards, obs_set, obs_mask, task_states, fake_imgs



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
                self._train_rollout_gen_fn, in_axes=(0, 0, 0, 0, None)))
            self._train_rollout_disc_fn = jax.jit(jax.pmap(
                self._train_rollout_disc_fn, in_axes=(0, 0, 0, None)))


        # Set up validation functions.
        self._valid_reset_fn = valid_vec_task.reset
        self._valid_step_fn = valid_vec_task.step
        self._valid_max_steps = valid_vec_task.max_steps
        self._valid_rollout_fn = partial(
            rollout_valid,
            step_once_gen_fn=partial(step_once_gen, task=valid_vec_task),
            max_steps=valid_vec_task.max_steps)
        if self._num_device > 1:
            self._valid_rollout_fn = jax.jit(jax.pmap(
                self._valid_rollout_fn, in_axes=(0, 0, 0, 0, None)))

    def eval_params(self,
                    params_gen: jnp.ndarray = None,
                    params_disc: jnp.ndarray = None,
                    batch_stats_gen: dict = None,
                    batch_stats_disc: dict = None,
                    generator: bool = True,
                    cat_codes: jnp.ndarray = None,
                    fake_imgs: jnp.ndarray = None,
                    testing: bool = False,
                    test: bool = False) -> Tuple[jnp.ndarray, TaskState]:
        """Evaluate population parameters or test the best parameter.

        Args:
            params - Parameters to be evaluated.
            test - Whether we are testing the best parameter
        Returns:
            An array of fitness scores.
        """
        if self._use_for_loop:
            return self._for_loop_eval(params_gen, params_disc, batch_stats_gen, batch_stats_disc, generator, cat_codes, fake_imgs, testing, test)
        else:
            return self._scan_loop_eval(params_gen, params_disc, batch_stats_gen, batch_stats_disc, generator, cat_codes, fake_imgs, testing, test)

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
                        params_gen: jnp.ndarray = None,
                        params_disc: jnp.ndarray = None,
                        batch_stats_gen: dict = None,
                        batch_stats_disc: dict = None,
                        generator: bool = True,
                        cat_codes: jnp.ndarray = None,
                        fake_imgs: jnp.ndarray = None,
                        testing: bool = False,
                        test: bool = False) -> Tuple[jnp.ndarray, TaskState]:
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

        self.batch_stats_gen = batch_stats_gen
        self.batch_stats_disc = batch_stats_disc

        #if params_gen is not None:
            #jax.debug.print('params gen shape before dup : {}', params_gen.shape)

        if test:
            n_repeats = self._test_n_repeats
            task_reset_func = self._valid_reset_fn
            rollout_func = self._valid_rollout_fn
            if generator:
                params_gen = duplicate_params(
                    params_gen[None, :], self._n_evaluations, False)
            params_disc = duplicate_params(params_disc[None, :], self._n_evaluations, False) 
        elif testing: 
            n_repeats = self._n_repeats
            task_reset_func = self._valid_reset_fn
            if generator:
                rollout_func = self._train_rollout_gen_fn
            else:
                rollout_func = self._train_rollout_disc_fn
        else:
            n_repeats = self._n_repeats
            task_reset_func = self._train_reset_fn
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
        if generator:
           params_gen = duplicate_params(params_gen, n_repeats, self._ma_training)
       
        #if params_gen is not None:
            #jax.debug.print('params gen shape after dup : {}', params_gen.shape)
        
        params_disc = duplicate_params(params_disc, n_repeats, self._ma_training)

        #split the reset keys, one for noise vect and one for cat codes
        noise_keys, cat_keys = random.split(self._key, 2)

        if generator:
            self._key, reset_keys1, reset_keys2 = get_task_reset_keys_testing(
                noise_keys, cat_keys, testing, self._pop_size, self._n_evaluations, n_repeats, self._ma_training)
        else:
            self._key, reset_keys = get_task_reset_keys(
                self._key, test, self._pop_size, self._n_evaluations, n_repeats,self._ma_training)

        #jax.debug.print('reset keys 1 shape : {}', reset_keys1.shape)
        #jax.debug.print('reset keys 2 shape : {}', reset_keys2.shape)
        # Reset the tasks and the policy.
        if generator:
            task_state = task_reset_func(reset_keys1, reset_keys2)
        else:
            task_state = task_reset_func(reset_keys)

        #if testing:
            # obs is shape (128, 512, 74), let's look at the first individual and first image and last ten features
            #jax.debug.print('task state obs : {}', task_state.obs[0, 0:100, -10:])
            #jax.debug.print('task state obs : {}', task_state.obs)
        #task_state.set_batch_stats(self.batch_stats_gen, self.batch_stats_disc)

        #jax.debug.print('batch_stats_gen shape : {}', self.batch_stats_gen.shape)
    
        if not generator:
            task_state = task_state.replace(cat_codes=cat_codes)
            task_state = task_state.replace(fake_imgs=fake_imgs)
            
        if generator:
            task_state = task_state.replace(batch_stats_gen=self.batch_stats_gen)

        task_state = task_state.replace(batch_stats_disc=self.batch_stats_disc)

        policy_state = policy_reset_func(task_state)
        
        if self._num_device > 1 and not test:
            if generator:
                params_gen = split_params_for_pmap(params_gen)
                batch_stats_gen = split_params_for_pmap(batch_stats_gen)
            #if not generator:
            #    fake_imgs = split_params_for_pmap(fake_imgs)
            params_disc = split_params_for_pmap(params_disc)
            task_state = split_states_for_pmap(task_state)
            policy_state = split_states_for_pmap(policy_state)
            batch_stats_disc = split_params_for_pmap(batch_stats_disc)

        #jax.debug.print('obs params : {}', self.obs_params)
        # Do the rollouts.
        if generator:
           scores, all_obs, masks, final_states, fake_imgs = rollout_func(
                task_state, policy_state, params_gen, params_disc, self.obs_params)
        else: 
            scores, all_obs, masks, final_states = rollout_func(
                task_state, policy_state, params_disc, self.obs_params)

        if self._num_device > 1:
            all_obs = reshape_data_from_pmap(all_obs)
            masks = reshape_data_from_pmap(masks)
            final_states = merge_state_from_pmap(final_states)
            #if generator:
            #    fake_imgs = reshape_data_from_pmap(fake_imgs)

        batch_stats_gen_updated = final_states.batch_stats_gen
        batch_stats_disc_updated = final_states.batch_stats_disc

        noise_keys, cat_keys = random.split(self._key, 2)

        if generator:
            self._key, reset_keys1, reset_keys2 = get_task_reset_keys_testing(
                noise_keys, cat_keys, testing, self._pop_size, self._n_evaluations, n_repeats, self._ma_training)

        if generator:
            task_state = task_reset_func(reset_keys1, reset_keys2)

        if generator:
            cat_codes = task_state.cat_codes
        else:
            cat_codes = None
            fake_imgs = None

        #jax.debug.print('scores shape before mean : {}', scores.shape)
        #jax.debug.print('final_states shape : {}', final_states.obs.shape)
        if not test and not self.obs_normalizer.is_dummy:
            self.obs_params = self.obs_normalizer.update_normalization_params(
                obs_buffer=all_obs, obs_mask=masks, obs_params=self.obs_params)

        if self._ma_training:
            if not test:
                # In training, each agent has different parameters.
                scores = jnp.mean(
                    scores.ravel().reshape((n_repeats, -1)), axis=0)
            else:
                # In tests, they share the same parameters.
                scores = jnp.mean(
                    scores.ravel().reshape((n_repeats, -1)), axis=1)
        else:
            scores = jnp.mean(scores.ravel().reshape((-1, n_repeats)), axis=-1)

        #jax.debug.print('scores shape after mean : {} ', scores.shape)
        # Note: QD methods do not support ma_training for now.
        #if not self._ma_training:
        #    final_states = tree_map(
        #        lambda x: x.reshape((scores.shape[0], n_repeats, *x.shape[1:])),
        #        final_states)

        return scores, self._bd_summarize_fn(final_states), batch_stats_gen_updated, batch_stats_disc_updated, fake_imgs, cat_codes

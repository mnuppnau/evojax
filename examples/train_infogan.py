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

"""Train an agent for MNIST classification.

Example command to run this script: `python train_mnist.py --gpu-id=0`
"""

import argparse
import os
import shutil

from jax import random
from evojax import Trainer
from evojax.task.mnist import MNIST
from evojax.task.latent import Latent_Points
from evojax.policy.convnet import GenPolicy, DiscPolicy
from evojax.algo import PGPE_CA, PGPE_DISC, PGPE_Q
from evojax.algo.cultural.belief_space import initialize_belief_space
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size-gen', type=int, default=128, help='NE population size.')
    parser.add_argument(
        '--pop-size', type=int, default=400, help='NE population size.')
    parser.add_argument(
        '--batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument(
        '--max-iter', type=int, default=42000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=1000, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=100, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr-gen', type=float, default=0.006, help='Center learning rate.')
    parser.add_argument(
        '--std-lr-gen', type=float, default=0.089, help='Std learning rate.')
    parser.add_argument(
        '--init-std-gen', type=float, default=0.039, help='Initial std.')
    parser.add_argument(
        '--center-lr-disc', type=float, default=0.006, help='Center learning rate.')
    parser.add_argument(
        '--std-lr-disc', type=float, default=0.089, help='Std learning rate.')
    parser.add_argument(
        '--init-std-disc', type=float, default=0.039, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = './log/mnist'
    model_dir = './log/mnist/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='MNIST', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX MNIST Demo')
    logger.info('=' * 30)

    policy_gen = GenPolicy(logger=logger)
    policy_disc = DiscPolicy(policy_gen, logger=logger)

    init_params_gen = policy_gen.init_params_gen
    flat_params_gen = policy_gen.flat_params_gen
    flat_params_disc = policy_disc.flat_params_disc
    flat_params_q = policy_disc.flat_params_q

    train_task_mnist = MNIST(batch_size=config.batch_size, test=False)
    test_task_mnist = MNIST(batch_size=config.batch_size, test=True)
  
    belief_space_key = random.PRNGKey(config.seed+12)
    belief_space = initialize_belief_space(population_size=config.pop_size, param_size=policy_gen.num_params, key=belief_space_key)

    train_task_latent = Latent_Points(batch_size=config.batch_size, test=False)
    test_task_latent = Latent_Points(batch_size=config.batch_size, test=True)

    solver_gen = PGPE_CA(
        pop_size=config.pop_size,
        param_size=policy_gen.num_params,
        init_params=flat_params_gen,
        optimizer='adam',
        center_learning_rate=config.center_lr_gen,
        stdev_learning_rate=config.std_lr_gen,
        init_stdev=config.init_std_gen,
        logger=logger,
        seed=config.seed,
        belief_space=belief_space,
    )

    #solver_disc = PGPE_DISC(
    #    pop_size=config.pop_size,
    #    param_size=policy_disc.num_params,
    #    init_params=flat_params_disc,
    #    optimizer='adam',
    #    center_learning_rate=config.center_lr_disc,
    #    stdev_learning_rate=config.std_lr_disc,
    #    init_stdev=config.init_std_disc,
    #    logger=logger,
    #    seed=config.seed + 1,
    #)

    #solver_q = PGPE_Q(
    #    pop_size=config.pop_size,
    #    param_size=policy_disc.num_params_q,
    #    init_params=flat_params_q,
    #    optimizer='adam',
    #    center_learning_rate=config.center_lr_disc,
    #    stdev_learning_rate=config.std_lr_disc,
    #    init_stdev=config.init_std_disc,
    #    logger=logger,
    #    seed=config.seed + 2,
    #)
    
    # Train.
    trainer = Trainer(
        policy_gen=policy_gen,
        policy_disc=policy_disc,
        solver_gen=solver_gen,
        #solver_disc=solver_disc,
        #solver_q=solver_q,
        train_task_gen=train_task_latent,
        test_task_gen=test_task_latent,
        train_task_disc=train_task_mnist,
        test_task_disc=test_task_mnist,
        #test_task=test_task,
        #model_dir=model_dir, 
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=1,
        n_evaluations=1,
        seed=config.seed,
        batch_size=config.batch_size,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)

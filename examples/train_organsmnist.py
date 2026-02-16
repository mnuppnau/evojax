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

"""Train an InfoGAN agent for OrganSMNIST (MedMNIST) classification.

OrganSMNIST: 28x28 grayscale images, 11 organ classes.
Latent vector: 63 (noise) + 11 (discrete codes) = 74.

Example command to run this script: `python train_organsmnist.py --gpu-id=0`
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
        '--pop-size', type=int, default=512, help='NE population size.')
    parser.add_argument(
        '--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument(
        '--max-iter', type=int, default=290000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=1000, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=100, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr-gen', type=float, default=0.0038, help='Center learning rate.')
    parser.add_argument(
        '--std-lr-gen', type=float, default=0.05, help='Std learning rate.')
    parser.add_argument(
        '--init-std-gen', type=float, default=0.03, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--checkpoint-interval', type=int, default=10000,
        help='Save a checkpoint every N iterations (0 to disable).')
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Directory for checkpoints (defaults to <log_dir>/checkpoints).')
    parser.add_argument(
        '--resume-from', type=str, default=None,
        help='Path to a checkpoint .pkl file or directory to resume from.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = './log/organsmnist'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='OrganSMNIST', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX OrganSMNIST InfoGAN Demo')
    logger.info('=' * 30)

    policy_gen = GenPolicy(logger=logger)

    init_params_gen = policy_gen.init_params_gen
    flat_params_gen = policy_gen.flat_params_gen

    init_params_hypernet = policy_gen.init_params_hypernet
    flat_params_hypernet = policy_gen.flat_params_hypernet

    train_task_mnist = MNIST(batch_size=config.batch_size, test=False)
    test_task_mnist = MNIST(batch_size=config.batch_size, test=True)

    belief_space_key = random.PRNGKey(config.seed+12)
    belief_space = initialize_belief_space(population_size=config.pop_size, param_size=policy_gen.num_params_hypernet, key=belief_space_key)

    train_task_latent = Latent_Points(batch_size=config.batch_size, test=False)
    test_task_latent = Latent_Points(batch_size=config.batch_size, test=True)

    solver_hn = PGPE_CA(
        pop_size=config.pop_size,
        param_size=policy_gen.num_params_hypernet,
        init_params=flat_params_hypernet,
        optimizer='adam',
        center_learning_rate=config.center_lr_gen,
        stdev_learning_rate=config.std_lr_gen,
        init_stdev=config.init_std_gen,
        logger=logger,
        seed=config.seed,
        belief_space=belief_space,
    )

    # Train.
    trainer = Trainer(
        policy_gen=policy_gen,
        solver_hn=solver_hn,
        train_task_gen=train_task_latent,
        test_task_gen=test_task_latent,
        train_task_disc=train_task_mnist,
        test_task_disc=test_task_mnist,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=1,
        n_evaluations=1,
        seed=config.seed,
        batch_size=config.batch_size,
        log_dir=log_dir,
        checkpoint_dir=config.checkpoint_dir,
        checkpoint_interval=config.checkpoint_interval,
        resume_from=config.resume_from,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)

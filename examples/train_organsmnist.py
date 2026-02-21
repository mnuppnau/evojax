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
from evojax.policy.convnet import GenPolicy, DiscPolicy, MultiLayerAdapter
from evojax.algo import PGPE_CA, PGPE_DISC, PGPE_Q, PGPE_Layer
from evojax.algo.cultural.belief_space import initialize_belief_space
from evojax.util import create_logger, get_params_format_fn
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
        '--per-layer', action='store_true',
        help='Use per-layer HyperNetworks with independent PGPE solvers.')
    parser.add_argument(
        '--pop-size-layer', type=int, default=256,
        help='Population size per layer solver (only used with --per-layer).')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    # --- Architecture / training config (stable defaults) ---
    parser.add_argument(
        '--gen-kernel-size', type=int, default=5, choices=[3, 5],
        help='Generator conv kernel size: 5=stable (4 layers), 3=experimental (5 layers).')
    parser.add_argument(
        '--hn-hidden-dim', type=int, default=48,
        help='HyperNetwork hidden layer dimension (48=stable, 32=experimental).')
    parser.add_argument(
        '--hn-activation', type=str, default='gelu', choices=['gelu', 'tanh'],
        help='HyperNetwork activation function (gelu=stable, tanh=experimental).')
    parser.add_argument(
        '--hn-output-stddev', type=float, default=0.025,
        help='HyperNetwork output layer init stddev (0.025=stable, 0.05=experimental).')
    parser.add_argument(
        '--chunk-size', type=int, default=512,
        help='HyperNetwork chunk size (512=stable, 256=experimental).')
    parser.add_argument(
        '--d-reg', type=str, default='binary', choices=['binary', 'smooth'],
        help='Discriminator regulation mode (binary=stable, smooth=experimental).')
    parser.add_argument(
        '--fitness-mode', type=str, default='static', choices=['static', 'dynamic'],
        help='Fitness composition mode (static=stable fixed weights, dynamic=CA-modulated).')
    parser.add_argument(
        '--ca-activation-iter', type=int, default=115000,
        help='Iteration at which CA starts activating (115000=stable, 500=experimental).')
    parser.add_argument(
        '--w-adv', type=float, default=0.6, help='Static fitness weight for adversarial.')
    parser.add_argument(
        '--w-mi', type=float, default=0.1, help='Static fitness weight for MI.')
    parser.add_argument(
        '--w-div', type=float, default=0.48, help='Static fitness weight for diversity.')
    parser.add_argument(
        '--w-sense', type=float, default=0.1, help='Static fitness weight for code separation.')
    parser.add_argument(
        '--w-norm', type=float, default=0.04, help='Static fitness weight for normative penalty.')
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

    policy_gen = GenPolicy(
        logger=logger,
        gen_kernel_size=config.gen_kernel_size,
        hn_hidden_dim=config.hn_hidden_dim,
        hn_activation=config.hn_activation,
        hn_output_stddev=config.hn_output_stddev,
        chunk_size=config.chunk_size,
    )

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

    static_weights = {
        'w_adv': config.w_adv, 'w_mi': config.w_mi,
        'w_div': config.w_div, 'w_sense': config.w_sense,
        'w_norm': config.w_norm,
    }
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
        fitness_mode=config.fitness_mode,
        ca_activation_iter=config.ca_activation_iter,
        static_weights=static_weights,
    )

    # --- Per-layer HyperNetwork solvers (optional) ---
    layer_solvers = None
    multi_adapter = None

    if config.per_layer:
        multi_adapter = MultiLayerAdapter(
            policy_gen.init_params_gen, chunk_size=config.chunk_size,
            hn_hidden_dim=config.hn_hidden_dim,
            hn_activation=config.hn_activation,
            hn_output_stddev=config.hn_output_stddev,
        )
        logger.info('Per-layer mode enabled:')
        logger.info(multi_adapter.summary())

        layer_solvers = []
        # Create format functions for each layer HN (flat -> pytree)
        layer_format_fns = []

        for i in range(multi_adapter.n_hn_groups):
            hn_init = multi_adapter.init_layer_hn(
                random.PRNGKey(config.seed + i), i
            )
            n_params = multi_adapter.get_layer_hn_param_count(i)
            _, fmt_fn = get_params_format_fn(hn_init)
            layer_format_fns.append(fmt_fn)

            solver = PGPE_Layer(
                pop_size=config.pop_size_layer,
                param_size=n_params,
                optimizer='adam',
                center_learning_rate=config.center_lr_gen,
                stdev_learning_rate=config.std_lr_gen,
                init_stdev=config.init_std_gen,
                seed=config.seed + i,
                logger=logger,
            )
            layer_solvers.append(solver)
            logger.info(
                f'  Layer solver {i} [{multi_adapter.hn_group_info[i][0]}]: '
                f'{n_params} HN params, pop_size={config.pop_size_layer}'
            )

        # Misc group: direct PGPE (no HN)
        misc_solver = PGPE_Layer(
            pop_size=config.pop_size_layer,
            param_size=multi_adapter.misc_total_params,
            optimizer='adam',
            center_learning_rate=config.center_lr_gen,
            stdev_learning_rate=config.std_lr_gen,
            init_stdev=config.init_std_gen,
            seed=config.seed + 99,
            logger=logger,
        )
        layer_solvers.append(misc_solver)
        layer_format_fns.append(None)  # misc group has no HN format fn
        logger.info(
            f'  Layer solver {multi_adapter.n_hn_groups} [misc]: '
            f'{multi_adapter.misc_total_params} direct params, '
            f'pop_size={config.pop_size_layer}'
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
        layer_solvers=layer_solvers,
        multi_adapter=multi_adapter,
        layer_format_fns=layer_format_fns if config.per_layer else None,
        d_reg_mode=config.d_reg,
        fitness_mode=config.fitness_mode,
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

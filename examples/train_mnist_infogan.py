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

"""Train an InfoGAN agent for handwritten MNIST classification.

MNIST: 28x28 grayscale handwritten digits.
Latent vector (default): 62 (noise) + 10 (discrete) + 2 (continuous) = 74.

Example command to run this script: `python train_mnist_infogan.py --gpu-id=0`
"""

import argparse
import os
import shutil
import sys

# Ensure the workspace package is imported when running this script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
        '--center-lr-gen', type=float, default=0.0048, help='Center learning rate.')
    parser.add_argument(
        '--std-lr-gen', type=float, default=0.062, help='Std learning rate.')
    parser.add_argument(
        '--init-std-gen', type=float, default=0.032, help='Initial std.')
    parser.add_argument(
        '--lr-decay-coef', type=float, default=1.0,
        help='Exponential decay coefficient for center LR (1.0 = no decay).')
    parser.add_argument(
        '--lr-decay-steps', type=int, default=100000,
        help='Decay the center LR every N steps.')
    parser.add_argument(
        '--ca-blend-coeff', type=float, default=0.03,
        help='CA gradient blend coefficient (set 0.0 to disable CA blend).')
    parser.add_argument(
        '--ca-blend-start-iter', type=int, default=3000,
        help='Iteration to start ramping CA blend.')
    parser.add_argument(
        '--ca-blend-ramp-iters', type=int, default=2000,
        help='Iterations to ramp CA blend to --ca-blend-coeff.')
    parser.add_argument(
        '--ca-blend-rfl-lo', type=float, default=0.40,
        help='Optional lower real_fake_loss gate for CA blend (negative disables gate).')
    parser.add_argument(
        '--ca-blend-rfl-hi', type=float, default=0.58,
        help='Optional upper real_fake_loss gate for CA blend (negative disables gate).')
    parser.add_argument(
        '--shape-div-weight', type=float, default=0.12,
        help='Weight for conditional feature-space shape diversity reward.')
    parser.add_argument(
        '--static-fitness-weights', action='store_true',
        help='Disable adaptive fitness reweighting and use fixed weights (baseline mode).')
    parser.add_argument(
        '--static-mi-sense-ramp', action='store_true',
        help='With static weights, keep legacy MI/sense base ramps over training.')
    parser.add_argument(
        '--static-w-adv', type=float, default=None,
        help='Static adversarial weight (used only with --static-fitness-weights).')
    parser.add_argument(
        '--static-w-mi', type=float, default=None,
        help='Static MI weight (used only with --static-fitness-weights).')
    parser.add_argument(
        '--static-w-div', type=float, default=None,
        help='Static code-diversity weight (used only with --static-fitness-weights).')
    parser.add_argument(
        '--static-w-sense', type=float, default=None,
        help='Static feature-space separation weight (used only with --static-fitness-weights).')
    parser.add_argument(
        '--static-w-intra', type=float, default=None,
        help='Static intra-code variation weight (used only with --static-fitness-weights).')
    parser.add_argument(
        '--noise-dim', type=int, default=62,
        help='Noise dimensions in latent vector.')
    parser.add_argument(
        '--n-discrete-codes', type=int, default=10,
        help='Number of discrete latent codes.')
    parser.add_argument(
        '--n-continuous-codes', type=int, default=2,
        help='Number of continuous latent codes.')
    parser.add_argument(
        '--disc-features', type=int, default=48,
        help='Base discriminator width (controls D/Q parameter count).')
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
    log_dir = './log/mnist_infogan'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='MNIST-InfoGAN', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX MNIST InfoGAN Demo')
    logger.info('=' * 30)
    logger.info(
        'CA blend schedule: coeff=%.4f start=%d ramp=%d rfl_gate=[%.3f, %.3f] shape_div_weight=%.3f',
        config.ca_blend_coeff,
        config.ca_blend_start_iter,
        config.ca_blend_ramp_iters,
        config.ca_blend_rfl_lo,
        config.ca_blend_rfl_hi,
        config.shape_div_weight,
    )
    logger.info(
        'LR schedule: center=%.4f stdev=%.4f init_std=%.4f decay_coef=%.4f decay_steps=%d',
        config.center_lr_gen,
        config.std_lr_gen,
        config.init_std_gen,
        config.lr_decay_coef,
        config.lr_decay_steps,
    )
    logger.info(
        'Latent config: noise=%d discrete=%d continuous=%d total=%d',
        config.noise_dim,
        config.n_discrete_codes,
        config.n_continuous_codes,
        config.noise_dim + config.n_discrete_codes + config.n_continuous_codes,
    )
    if config.static_fitness_weights:
        logger.info(
            'Static weights: adv=%s mi=%s div=%s sense=%s intra=%s (mi_sense_ramp=%s)',
            str(config.static_w_adv),
            str(config.static_w_mi),
            str(config.static_w_div),
            str(config.static_w_sense),
            str(config.static_w_intra),
            str(config.static_mi_sense_ramp),
        )

    policy_gen = GenPolicy(
        logger=logger,
        noise_dim=config.noise_dim,
        n_discrete_codes=config.n_discrete_codes,
        n_continuous_codes=config.n_continuous_codes,
        disc_features=config.disc_features,
    )
    feature_dim = int(policy_gen.disc_feature_dim)

    init_params_gen = policy_gen.init_params_gen
    flat_params_gen = policy_gen.flat_params_gen

    init_params_hypernet = policy_gen.init_params_hypernet
    flat_params_hypernet = policy_gen.flat_params_hypernet

    train_task_mnist = MNIST(batch_size=config.batch_size, test=False)
    test_task_mnist = MNIST(batch_size=config.batch_size, test=True)

    belief_space_key = random.PRNGKey(config.seed+12)
    belief_space = initialize_belief_space(
        population_size=config.pop_size,
        param_size=policy_gen.num_params_hypernet,
        key=belief_space_key,
        features=feature_dim,
        num_codes=config.n_discrete_codes,
    )

    train_task_latent = Latent_Points(
        batch_size=config.batch_size,
        latent_dim=config.noise_dim,
        n_classes=config.n_discrete_codes,
        n_cont=config.n_continuous_codes,
        feature_dim=feature_dim,
        test=False,
    )
    test_task_latent = Latent_Points(
        batch_size=config.batch_size,
        latent_dim=config.noise_dim,
        n_classes=config.n_discrete_codes,
        n_cont=config.n_continuous_codes,
        feature_dim=feature_dim,
        test=True,
    )

    solver_hn = PGPE_CA(
        pop_size=config.pop_size,
        param_size=policy_gen.num_params_hypernet,
        init_params=flat_params_hypernet,
        optimizer='adam',
        optimizer_config={
            'center_lr_decay_coef': config.lr_decay_coef,
            'center_lr_decay_steps': config.lr_decay_steps,
        },
        center_learning_rate=config.center_lr_gen,
        stdev_learning_rate=config.std_lr_gen,
        init_stdev=config.init_std_gen,
        logger=logger,
        seed=config.seed,
        belief_space=belief_space,
        ca_blend_coeff=config.ca_blend_coeff,
        ca_blend_start_iter=config.ca_blend_start_iter,
        ca_blend_ramp_iters=config.ca_blend_ramp_iters,
        ca_blend_rfl_lo=config.ca_blend_rfl_lo,
        ca_blend_rfl_hi=config.ca_blend_rfl_hi,
        shape_div_weight=config.shape_div_weight,
        static_fitness_weights=config.static_fitness_weights,
        static_mi_sense_ramp=config.static_mi_sense_ramp,
        static_w_adv=config.static_w_adv,
        static_w_mi=config.static_w_mi,
        static_w_div=config.static_w_div,
        static_w_sense=config.static_w_sense,
        static_w_intra=config.static_w_intra,
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

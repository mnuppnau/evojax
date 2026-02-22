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
import sys

# Ensure the local workspace package is used when running this script directly
# from `examples/` (otherwise Python can import an older site-packages copy).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from jax import random
from evojax import Trainer
from evojax.task.mnist import MNIST
from evojax.task.latent import Latent_Points
from evojax.policy.convnet import GenPolicy
from evojax.algo import PGPE_CA
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
        '--fitness-mode', type=str, default='dynamic', choices=['static', 'dynamic'],
        help='`dynamic`: belief-space adaptive controls for weights/CA blend/D schedule, `static`: fixed weights.')
    parser.add_argument(
        '--ca-activation-iter', type=int, default=500,
        help='Iteration where CA distress-based weight modulation starts ramping in.')
    parser.add_argument(
        '--ca-blend-max', type=float, default=0.025,
        help='Maximum CA gradient blend ratio.')
    parser.add_argument(
        '--ca-blend-start-iter', type=int, default=500,
        help='Iteration where CA gradient blending starts.')
    parser.add_argument(
        '--ca-blend-ramp-iters', type=int, default=2000,
        help='Iterations to ramp CA blend from 0 to --ca-blend-max.')
    parser.add_argument(
        '--w-adv', type=float, default=0.53, help='Static/base weight for adversarial fitness.')
    parser.add_argument(
        '--w-mi', type=float, default=0.10, help='Static/base weight for MI fitness.')
    parser.add_argument(
        '--w-div', type=float, default=0.48, help='Static/base weight for pixel-space code diversity.')
    parser.add_argument(
        '--w-sense', type=float, default=0.10, help='Static/base weight for feature-space separation.')
    parser.add_argument(
        '--w-intra', type=float, default=0.08, help='Static/base weight for within-code variation.')
    parser.add_argument(
        '--w-cons-floor', type=float, default=0.04,
        help='Static/base weight for centroid consistency floor penalty.')
    parser.add_argument(
        '--w-norm', type=float, default=0.02, help='Static/base weight for normative penalty.')
    parser.add_argument(
        '--w-adv-ceiling', type=float, default=0.0,
        help='Penalty weight for adversarial shortcut violations above --adv-ceiling.')
    parser.add_argument(
        '--adv-ceiling', type=float, default=-0.65,
        help='Adversarial reward ceiling (logged sign). Values above this are penalized.')
    parser.add_argument(
        '--adv-ceiling-warmup-iter', type=int, default=2000,
        help='Iteration to start applying adversarial ceiling penalty.')
    parser.add_argument(
        '--gen-noise-std', type=float, default=0.1,
        help='Instance-noise std used during generator fitness rollouts.')
    parser.add_argument(
        '--sat-penalty-weight', type=float, default=0.06,
        help='Penalty weight for over-saturated generated pixels.')
    parser.add_argument(
        '--sat-threshold', type=float, default=0.95,
        help='Absolute pixel threshold used to count saturation.')
    parser.add_argument(
        '--sat-target-ratio', type=float, default=0.55,
        help='Target upper saturation ratio; values above this are penalized.')
    parser.add_argument(
        '--sat-min-ratio', type=float, default=0.10,
        help='Minimum saturation ratio before low-contrast images are penalized.')
    parser.add_argument(
        '--d-freq-phase1', type=int, default=12,
        help='Train discriminator every N iterations in phase 1.')
    parser.add_argument(
        '--d-freq-phase2', type=int, default=4,
        help='Train discriminator every N iterations in phase 2.')
    parser.add_argument(
        '--d-phase1-iters', type=int, default=2000,
        help='End iteration of discriminator phase 1.')
    parser.add_argument(
        '--d-phase2-iters', type=int, default=4000,
        help='End iteration of discriminator phase 2.')
    parser.add_argument(
        '--d-update-warmup-iters', type=int, default=5000,
        help='Always accept discriminator updates before this iteration.')
    parser.add_argument(
        '--d-min-real-fake-loss', type=float, default=0.35,
        help='After warmup, only accept discriminator updates above this loss.')
    parser.add_argument(
        '--d-rescue-rfl', type=float, default=0.55,
        help='Force discriminator updates every iteration when real_fake_loss exceeds this.')
    parser.add_argument(
        '--d-rescue-adv-max', type=float, default=-0.75,
        help='Force discriminator updates every iteration when adv_max exceeds this (logged sign).')
    parser.add_argument(
        '--d-rescue-end-iters', type=int, default=12000,
        help='Disable D-rescue override after this iteration.')
    parser.add_argument(
        '--disc-noise-std-start', type=float, default=0.1,
        help='Initial discriminator instance-noise std.')
    parser.add_argument(
        '--disc-noise-std-end', type=float, default=0.04,
        help='Final discriminator instance-noise std after annealing.')
    parser.add_argument(
        '--disc-noise-anneal-iters', type=int, default=12000,
        help='Iterations for linear discriminator instance-noise anneal.')
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
    parser.add_argument(
        '--run-demo', action='store_true',
        help='Run final demo/test pass after training.')
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
        saturation_penalty_weight=config.sat_penalty_weight,
        sat_threshold=config.sat_threshold,
        sat_target_ratio=config.sat_target_ratio,
        sat_min_ratio=config.sat_min_ratio,
    )

    init_params_gen = policy_gen.init_params_gen
    flat_params_gen = policy_gen.flat_params_gen

    init_params_hypernet = policy_gen.init_params_hypernet
    flat_params_hypernet = policy_gen.flat_params_hypernet

    train_task_mnist = MNIST(batch_size=config.batch_size, test=False)
    test_task_mnist = MNIST(batch_size=config.batch_size, test=True)

    belief_space_key = random.PRNGKey(config.seed+12)
    belief_space = initialize_belief_space(population_size=config.pop_size, param_size=policy_gen.num_params_hypernet, key=belief_space_key)

    train_task_latent = Latent_Points(
        batch_size=config.batch_size,
        noise_std=config.gen_noise_std,
        test=False,
    )
    test_task_latent = Latent_Points(
        batch_size=config.batch_size,
        noise_std=config.gen_noise_std,
        test=True,
    )

    static_weights = {
        'w_adv': config.w_adv,
        'w_mi': config.w_mi,
        'w_div': config.w_div,
        'w_sense': config.w_sense,
        'w_intra': config.w_intra,
        'w_cons_floor': config.w_cons_floor,
        'w_norm': config.w_norm,
        'w_adv_ceiling': config.w_adv_ceiling,
        'adv_ceiling': config.adv_ceiling,
        'adv_ceiling_warmup': config.adv_ceiling_warmup_iter,
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
        ca_blend_max=config.ca_blend_max,
        ca_blend_start_iter=config.ca_blend_start_iter,
        ca_blend_ramp_iters=config.ca_blend_ramp_iters,
        static_weights=static_weights,
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
        d_freq_phase1=config.d_freq_phase1,
        d_freq_phase2=config.d_freq_phase2,
        d_phase1_iters=config.d_phase1_iters,
        d_phase2_iters=config.d_phase2_iters,
        d_update_warmup_iters=config.d_update_warmup_iters,
        d_min_real_fake_loss=config.d_min_real_fake_loss,
        d_rescue_rfl=config.d_rescue_rfl,
        d_rescue_adv_max=config.d_rescue_adv_max,
        d_rescue_end_iters=config.d_rescue_end_iters,
        disc_noise_std_start=config.disc_noise_std_start,
        disc_noise_std_end=config.disc_noise_std_end,
        disc_noise_anneal_iters=config.disc_noise_anneal_iters,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Optional final demo/test pass.
    if config.run_demo:
        trainer.model_dir = log_dir
        trainer.run(demo_mode=True)


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)

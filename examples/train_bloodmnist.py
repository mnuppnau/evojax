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

"""Train Hyper-InfoGAN on BloodMNIST with PGPE + Cultural Algorithms.

BloodMNIST: 28x28 images, 8 classes.
In this branch we convert RGB to grayscale (luminance) during loading.
Latent vector: 63 noise + 8 discrete code bits.
"""

import argparse
import os
import sys

# Ensure the workspace package is imported when running this script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from jax import random

from evojax import Trainer
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
        '--noise-dim', type=int, default=63, help='Noise latent dimension.')
    parser.add_argument(
        '--n-classes', type=int, default=8, help='Number of discrete classes/codes.')
    parser.add_argument(
        '--disc-features', type=int, default=32,
        help='Base channel width for the discriminator (Option 2 capacity control).')
    parser.add_argument(
        '--max-iter', type=int, default=290000, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=1000, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=100, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument(
        '--center-lr-gen', type=float, default=0.0044, help='Center learning rate.')
    parser.add_argument(
        '--std-lr-gen', type=float, default=0.06, help='Std learning rate.')
    parser.add_argument(
        '--init-std-gen', type=float, default=0.03, help='Initial std.')
    parser.add_argument(
        '--ca-blend-coeff', type=float, default=0.0,
        help='CA gradient blend coefficient (set 0.0 to disable CA blend).')
    parser.add_argument(
        '--ca-blend-start-iter', type=int, default=4500,
        help='Iteration to start ramping CA blend.')
    parser.add_argument(
        '--ca-blend-ramp-iters', type=int, default=3000,
        help='Iterations to ramp CA blend to --ca-blend-coeff.')
    parser.add_argument(
        '--ca-blend-rfl-lo', type=float, default=0.43,
        help='Optional lower real_fake_loss gate for CA blend (negative disables gate).')
    parser.add_argument(
        '--ca-blend-rfl-hi', type=float, default=0.56,
        help='Optional upper real_fake_loss gate for CA blend (negative disables gate).')
    parser.add_argument(
        '--shape-div-weight', type=float, default=0.01,
        help='Weight for conditional feature-space shape diversity reward.')
    parser.add_argument(
        '--sat-penalty-weight', type=float, default=0.12,
        help='Max weight for output saturation penalty.')
    parser.add_argument(
        '--sat-target', type=float, default=0.10,
        help='Allowed saturation fraction before excess penalty activates.')
    parser.add_argument(
        '--sat-penalty-start-iter', type=int, default=1500,
        help='Iteration to start saturation-penalty ramp.')
    parser.add_argument(
        '--sat-penalty-ramp-iters', type=int, default=1500,
        help='Ramp length for saturation-penalty weight.')
    parser.add_argument(
        '--adv-sat-trigger', type=float, default=0.20,
        help='Mean sat_excess threshold where adversarial governor starts.')
    parser.add_argument(
        '--adv-sat-cap-min', type=float, default=0.35,
        help='Minimum cap for rank-normalized adversarial fitness term.')
    parser.add_argument(
        '--adv-sat-strength', type=float, default=0.85,
        help='Strength of saturation-aware adversarial governor (0-1).')
    parser.add_argument(
        '--disc-lr', type=float, default=7e-5,
        help='Discriminator Adam learning rate.')
    parser.add_argument(
        '--disc-update-gate', type=float, default=0.40,
        help='Only keep D update when real_fake_loss exceeds this gate.')
    parser.add_argument(
        '--disc-warmup-freq', type=int, default=5,
        help='Run one D step every N iterations during warmup.')
    parser.add_argument(
        '--disc-warmup-iters', type=int, default=3000,
        help='Number of warmup iterations to apply --disc-warmup-freq.')
    parser.add_argument(
        '--disc-input-noise-start', type=float, default=0.20,
        help='Initial discriminator input noise std (decays slowly over training).')
    parser.add_argument(
        '--disc-input-noise-end', type=float, default=0.08,
        help='Final discriminator input noise std after decay.')
    parser.add_argument(
        '--disc-input-noise-decay-iters', type=int, default=30000,
        help='Iterations for linear decay of discriminator input noise std.')
    parser.add_argument(
        '--disc-reg-enable', action='store_true',
        help='Enable discriminator regulation (soft rollback on prolonged low real_fake_loss).')
    parser.add_argument(
        '--disc-reg-rfl-threshold', type=float, default=0.35,
        help='real_fake_loss threshold for D regulation streak counting.')
    parser.add_argument(
        '--disc-reg-consecutive-logs', type=int, default=4,
        help='Consecutive low-rfl log intervals required to trigger D regulation.')
    parser.add_argument(
        '--disc-reg-blend', type=float, default=0.30,
        help='Blend factor for D regulation toward anchor D weights.')
    parser.add_argument(
        '--disc-reg-start-iter', type=int, default=2000,
        help='Earliest iteration where D regulation may trigger.')
    parser.add_argument(
        '--disc-reg-cooldown-iters', type=int, default=1000,
        help='Minimum iterations between D regulation events.')
    parser.add_argument(
        '--data-root', type=str, default='./data',
        help='Directory for MedMNIST data.')
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
    log_dir = './log/bloodmnist'
    os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='BloodMNIST', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX BloodMNIST Hyper-InfoGAN Demo')
    logger.info('=' * 30)
    logger.info(
        'CA blend: coeff=%.4f start=%d ramp=%d gate=[%.3f, %.3f] '
        'shape_div_weight=%.3f sat_penalty_weight=%.3f sat_target=%.3f '
        'sat_ramp_start=%d sat_ramp_iters=%d adv_sat_trigger=%.3f '
        'adv_sat_cap_min=%.3f adv_sat_strength=%.3f disc_features=%d '
        'n_classes=%d noise_dim=%d',
        config.ca_blend_coeff,
        config.ca_blend_start_iter,
        config.ca_blend_ramp_iters,
        config.ca_blend_rfl_lo,
        config.ca_blend_rfl_hi,
        config.shape_div_weight,
        config.sat_penalty_weight,
        config.sat_target,
        config.sat_penalty_start_iter,
        config.sat_penalty_ramp_iters,
        config.adv_sat_trigger,
        config.adv_sat_cap_min,
        config.adv_sat_strength,
        config.disc_features,
        config.n_classes,
        config.noise_dim,
    )
    logger.info(
        'D control: lr=%.6f update_gate=%.3f warmup_freq=%d warmup_iters=%d '
        'noise_std=[%.3f->%.3f] decay_iters=%d '
        'reg_enable=%s reg_thr=%.3f reg_logs=%d reg_blend=%.2f reg_start=%d reg_cooldown=%d',
        config.disc_lr,
        config.disc_update_gate,
        config.disc_warmup_freq,
        config.disc_warmup_iters,
        config.disc_input_noise_start,
        config.disc_input_noise_end,
        config.disc_input_noise_decay_iters,
        str(config.disc_reg_enable),
        config.disc_reg_rfl_threshold,
        config.disc_reg_consecutive_logs,
        config.disc_reg_blend,
        config.disc_reg_start_iter,
        config.disc_reg_cooldown_iters,
    )

    policy_gen = GenPolicy(
        n_classes=config.n_classes,
        noise_dim=config.noise_dim,
        image_size=28,
        image_channels=1,
        disc_features=config.disc_features,
        logger=logger,
    )

    belief_space_key = random.PRNGKey(config.seed + 12)
    belief_space = initialize_belief_space(
        population_size=config.pop_size,
        param_size=policy_gen.num_params_hypernet,
        key=belief_space_key,
        features=config.disc_features * 4,
        num_codes=config.n_classes,
    )

    train_task_latent = Latent_Points(
        batch_size=config.batch_size,
        latent_dim=config.noise_dim,
        n_classes=config.n_classes,
        dataset_name='bloodmnist',
        image_size=28,
        image_channels=1,
        data_root=config.data_root,
        test=False,
    )
    test_task_latent = Latent_Points(
        batch_size=config.batch_size,
        latent_dim=config.noise_dim,
        n_classes=config.n_classes,
        dataset_name='bloodmnist',
        image_size=28,
        image_channels=1,
        data_root=config.data_root,
        test=True,
    )

    solver_hn = PGPE_CA(
        pop_size=config.pop_size,
        param_size=policy_gen.num_params_hypernet,
        init_params=policy_gen.flat_params_hypernet,
        optimizer='adam',
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
        sat_penalty_weight=config.sat_penalty_weight,
        sat_target=config.sat_target,
        sat_penalty_start_iter=config.sat_penalty_start_iter,
        sat_penalty_ramp_iters=config.sat_penalty_ramp_iters,
        adv_sat_trigger=config.adv_sat_trigger,
        adv_sat_cap_min=config.adv_sat_cap_min,
        adv_sat_strength=config.adv_sat_strength,
    )

    trainer = Trainer(
        policy_gen=policy_gen,
        solver_hn=solver_hn,
        train_task_gen=train_task_latent,
        test_task_gen=test_task_latent,
        # Unused in the current training path; pass latent tasks to avoid
        # pulling unrelated datasets.
        train_task_disc=train_task_latent,
        test_task_disc=test_task_latent,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=1,
        n_evaluations=1,
        seed=config.seed,
        batch_size=config.batch_size,
        latent_dim=config.noise_dim,
        n_classes=config.n_classes,
        dataset_name='bloodmnist',
        image_size=28,
        image_channels=1,
        data_root=config.data_root,
        disc_features=config.disc_features,
        disc_lr=config.disc_lr,
        disc_update_gate=config.disc_update_gate,
        disc_warmup_freq=config.disc_warmup_freq,
        disc_warmup_iters=config.disc_warmup_iters,
        disc_input_noise_start=config.disc_input_noise_start,
        disc_input_noise_end=config.disc_input_noise_end,
        disc_input_noise_decay_iters=config.disc_input_noise_decay_iters,
        disc_reg_enable=config.disc_reg_enable,
        disc_reg_rfl_threshold=config.disc_reg_rfl_threshold,
        disc_reg_consecutive_logs=config.disc_reg_consecutive_logs,
        disc_reg_blend=config.disc_reg_blend,
        disc_reg_start_iter=config.disc_reg_start_iter,
        disc_reg_cooldown_iters=config.disc_reg_cooldown_iters,
        log_dir=log_dir,
        checkpoint_dir=config.checkpoint_dir,
        checkpoint_interval=config.checkpoint_interval,
        resume_from=config.resume_from,
        logger=logger,
    )
    trainer.run(demo_mode=False)


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)

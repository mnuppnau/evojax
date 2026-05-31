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

"""Train a HyperNet-InfoGAN hard-attention agent on IMDb sentiment.

Phase 6 deliverable: this is the IMDB-B00 training entry point. It mirrors
`train_bloodmnist.py` for CLI / config / construction, and wires together the
IMDb-specific components built in Phases 1-5:

    * `IMDb` (data provider; `evojax/task/imdb.py`)
    * `AttentionPolicy` (frozen DistilBERT + HyperNet + mask gen + Classifier
      + Q-head; `evojax/policy/attention_transformer.py`)
    * `PGPE_CA_Text` (solver fork with attention-quality fitness composition;
      `evojax/algo/pgpe_ca_text.py`)

Phase 7 owns the actual `Trainer.run()` wiring. The shared `evojax/trainer.py`
still assumes the BloodMNIST `GenPolicy` interface (float32 image data, GAN
real/fake loss in the D-step, BatchNorm variables in the policy). This script
constructs every Phase-6 component and exposes them under the well-defined
`build_components()` function so Phase 7 can drop the trainer wiring in
without touching the CLI surface. Until then, the `__main__` path stops at a
"components compiled" report and exits 0.

Example baseline command (post-Phase 7):
    python examples/train_imdb.py \\
        --gpu-id=0 \\
        --pop-size=512 --batch-size=64 \\
        --max-iter=290000 --ca-blend-coeff=0.0 \\
        --static-fitness-weights --checkpoint-interval=5000

Phase 4 testing surfaced a transient `INTERNAL: Failed to allocate ... bytes
for new constant` error that disappeared with the XLA env var below.  We set
it BEFORE any JAX import so the long Phase-8 run starts clean.
"""

import os
import sys

# Set XLA preallocate=false BEFORE any JAX import. See Phase 4 notes in
# IMDB_PLAN.md — long runs at pop_size=512 will see real GPU pressure.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# GPU selection must also happen BEFORE any JAX import: JAX reads
# CUDA_VISIBLE_DEVICES at import time, so setting it from `--gpu-id` after
# the import is a no-op. Sniff sys.argv directly for `--gpu-id N` and
# `--gpu-id=N` forms; argparse runs much later.
for _i, _arg in enumerate(sys.argv):
    if _arg == '--gpu-id' and _i + 1 < len(sys.argv):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])
        break
    if _arg.startswith('--gpu-id='):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', _arg.split('=', 1)[1])
        break

import argparse

# Ensure the workspace package is imported when running this script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jax                       # noqa: E402
import jax.numpy as jnp          # noqa: E402
from jax import random           # noqa: E402

from evojax.task.imdb import IMDb                              # noqa: E402
from evojax.policy.attention_transformer import AttentionPolicy  # noqa: E402
from evojax.algo.pgpe_ca_text import PGPE_CA_Text              # noqa: E402
from evojax.algo.cultural.belief_space import (                 # noqa: E402
    initialize_belief_space,
)
from evojax.trainer_imdb import TrainerIMDb                     # noqa: E402
from evojax import util                                         # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Core training loop
    p.add_argument('--pop-size', type=int, default=512,
                   help='NE population size.')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Batch size for training.')
    p.add_argument('--max-iter', type=int, default=290000,
                   help='Max training iterations (IMDB-B00 target).')
    p.add_argument('--test-interval', type=int, default=1000,
                   help='Test interval.')
    p.add_argument('--log-interval', type=int, default=100,
                   help='Logging interval.')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for training.')

    # PGPE optimizer schedule
    p.add_argument('--center-lr-gen', type=float, default=0.0048,
                   help='Center learning rate (mirrors BloodMNIST default).')
    p.add_argument('--std-lr-gen', type=float, default=0.062,
                   help='Std learning rate.')
    p.add_argument('--init-std-gen', type=float, default=0.032,
                   help='Initial std.')
    p.add_argument('--lr-decay-coef', type=float, default=1.0,
                   help='Exponential decay coefficient for center LR.')
    p.add_argument('--lr-decay-steps', type=int, default=100000,
                   help='Decay the center LR every N steps.')

    # CA gradient-blend schedule. Default 0.0 == baseline (CA OFF).
    p.add_argument('--ca-blend-coeff', type=float, default=0.0,
                   help='CA gradient blend coefficient. 0.0 = OFF (IMDB-B00 default).')
    p.add_argument('--ca-blend-start-iter', type=int, default=3000,
                   help='Iteration to start ramping CA blend.')
    p.add_argument('--ca-blend-ramp-iters', type=int, default=2000,
                   help='Iterations to ramp CA blend to --ca-blend-coeff.')
    p.add_argument('--ca-blend-rfl-lo', type=float, default=-1.0,
                   help='Optional lower classifier-loss gate (negative disables).')
    p.add_argument('--ca-blend-rfl-hi', type=float, default=-1.0,
                   help='Optional upper classifier-loss gate (negative disables).')

    # Shape-diversity weight (legacy slot; left at BloodMNIST default).
    p.add_argument('--shape-div-weight', type=float, default=0.12,
                   help='Weight for conditional shape diversity reward.')

    # Static / hybrid fitness weighting. Default == static (baseline).
    p.add_argument('--static-fitness-weights', action='store_true',
                   default=True,
                   help='Use fixed weights, no adaptive reweighting (IMDB-B00 default).')
    p.add_argument('--adaptive-fitness-weights', dest='static_fitness_weights',
                   action='store_false',
                   help='Enable adaptive fitness reweighting (for IMDB-A01).')
    p.add_argument('--hybrid-fitness-weights', action='store_true',
                   help='Keep w_mi static while letting w_adv / w_intra / etc adapt.')
    p.add_argument('--static-mi-sense-ramp', action='store_true',
                   help='With static weights, keep legacy MI/sense ramp.')
    p.add_argument('--static-w-adv', type=float, default=None)
    p.add_argument('--static-w-mi', type=float, default=None)
    p.add_argument('--static-w-div', type=float, default=None)
    p.add_argument('--static-w-sense', type=float, default=None)
    p.add_argument('--static-w-intra', type=float, default=None)
    p.add_argument('--static-div-ramp-target', type=float, default=None)
    p.add_argument('--static-div-ramp-start-iter', type=int, default=-1)
    p.add_argument('--static-div-ramp-end-iter', type=int, default=-1)
    p.add_argument('--static-sense-ramp-target', type=float, default=None)
    p.add_argument('--static-sense-ramp-start-iter', type=int, default=-1)
    p.add_argument('--static-sense-ramp-end-iter', type=int, default=-1)
    p.add_argument('--static-intra-ramp-target', type=float, default=None)
    p.add_argument('--static-intra-ramp-start-iter', type=int, default=-1)
    p.add_argument('--static-intra-ramp-end-iter', type=int, default=-1)

    # IMDb-specific architecture
    p.add_argument('--encoder-name', type=str, default='distilbert-base-uncased',
                   help='Pretrained encoder model identifier (HuggingFace hub).')
    p.add_argument('--seq-len', type=int, default=256,
                   help='Tokenized sequence length.')
    p.add_argument('--n-discrete-codes', type=int, default=8,
                   help='Number of discrete latent codes (K).')
    p.add_argument('--n-continuous-codes', type=int, default=2,
                   help='Number of continuous latent codes.')
    p.add_argument('--noise-dim', type=int, default=62,
                   help='Noise dimensions in latent vector.')
    p.add_argument('--score-hidden', type=int, default=64,
                   help='AttentionMaskGenerator hidden size.')
    p.add_argument('--classifier-hidden', type=int, default=128,
                   help='Classifier head MLP hidden size.')
    p.add_argument('--q-hidden', type=int, default=128,
                   help='Q-head MLP hidden size.')
    p.add_argument('--chunk-size', type=int, default=512,
                   help='HyperNet output chunk size.')

    # Logistics
    p.add_argument('--gpu-id', type=str, default=None,
                   help='GPU(s) to use (sets CUDA_VISIBLE_DEVICES).')
    p.add_argument('--checkpoint-interval', type=int, default=5000,
                   help='Save a checkpoint every N iterations (0 = disable).')
    p.add_argument('--checkpoint-dir', type=str, default=None,
                   help='Directory for checkpoints (defaults to <log_dir>/checkpoints).')
    p.add_argument('--resume-from', type=str, default=None,
                   help='Path to a checkpoint to resume from.')
    p.add_argument('--render-interval', type=int, default=0,
                   help='Auto-render an HTML attention snapshot every N iters '
                        '(0 = disable). Snapshots are written next to the '
                        'checkpoints. Final-iter snapshot is always saved when '
                        'render_interval > 0.')
    p.add_argument('--render-n-reviews', type=int, default=8,
                   help='Number of held-out test reviews per render snapshot.')
    p.add_argument('--render-seed', type=int, default=0,
                   help='Seed for selecting the held-out review subset.')
    p.add_argument('--log-dir', type=str, default='./log/imdb',
                   help='Logging directory.')
    p.add_argument('--debug', action='store_true', help='Debug mode.')

    cfg, _ = p.parse_known_args()
    return cfg


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------

def build_components(config, logger):
    """Construct policy + tasks + solver + belief space.

    Returns a dict of named components so Phase 7's Trainer wiring can pick
    them up without re-reading the config or duplicating construction logic.
    """
    # 1. Policy (frozen DistilBERT + HyperNet + AttentionMaskGenerator +
    #    Classifier + Q-head).
    policy = AttentionPolicy(
        seq_len=config.seq_len,
        n_codes=config.n_discrete_codes,
        score_hidden=config.score_hidden,
        classifier_hidden=config.classifier_hidden,
        q_hidden=config.q_hidden,
        chunk_size=config.chunk_size,
        encoder_name=config.encoder_name,
        logger=logger,
    )

    # 2. Data tasks (train + test). The trainer reads `.data` (int32 token
    #    IDs) and `.labels` directly to build class-balanced D-step batches.
    train_task_imdb = IMDb(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_codes=config.n_discrete_codes,
        n_cont=config.n_continuous_codes,
        latent_dim=config.noise_dim,
        tokenizer_name=config.encoder_name,
        test=False,
    )
    test_task_imdb = IMDb(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_codes=config.n_discrete_codes,
        n_cont=config.n_continuous_codes,
        latent_dim=config.noise_dim,
        tokenizer_name=config.encoder_name,
        test=True,
    )

    # 3. Belief space for CA. param_size = HyperNet search dim.
    #    `features` and `num_codes` mirror BloodMNIST's convention; for IMDb
    #    `features` = encoder hidden dim (the classifier's input size).
    belief_space_key = random.PRNGKey(config.seed + 12)
    belief_space = initialize_belief_space(
        population_size=config.pop_size,
        param_size=policy.num_params,
        key=belief_space_key,
        features=policy.hidden_dim,
        num_codes=config.n_discrete_codes,
    )

    # 4. Solver. PGPE_CA_Text is baseline-safe out of the box
    #    (ca_blend_coeff default = 0.0). Flatten the HyperNet pytree the
    #    same way `convnet.py:GenPolicy` does so PGPE starts at the same
    #    initialization the AttentionPolicy was constructed with.
    hn_leaves = jax.tree_util.tree_leaves(policy.init_params_hypernet)
    flat_init_hypernet = jnp.concatenate([p.flatten() for p in hn_leaves])

    solver = PGPE_CA_Text(
        pop_size=config.pop_size,
        param_size=policy.num_params,
        init_params=flat_init_hypernet,
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
        hybrid_fitness_weights=config.hybrid_fitness_weights,
        static_mi_sense_ramp=config.static_mi_sense_ramp,
        static_w_adv=config.static_w_adv,
        static_w_mi=config.static_w_mi,
        static_w_div=config.static_w_div,
        static_w_sense=config.static_w_sense,
        static_w_intra=config.static_w_intra,
        static_div_ramp_target=config.static_div_ramp_target,
        static_div_ramp_start_iter=config.static_div_ramp_start_iter,
        static_div_ramp_end_iter=config.static_div_ramp_end_iter,
        static_sense_ramp_target=config.static_sense_ramp_target,
        static_sense_ramp_start_iter=config.static_sense_ramp_start_iter,
        static_sense_ramp_end_iter=config.static_sense_ramp_end_iter,
        static_intra_ramp_target=config.static_intra_ramp_target,
        static_intra_ramp_start_iter=config.static_intra_ramp_start_iter,
        static_intra_ramp_end_iter=config.static_intra_ramp_end_iter,
    )

    return {
        'policy': policy,
        'train_task_imdb': train_task_imdb,
        'test_task_imdb': test_task_imdb,
        'belief_space': belief_space,
        'solver': solver,
        'log_dir': config.log_dir,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config):
    log_dir = config.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='IMDb', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX IMDb Hard-Attention HyperNet-InfoGAN — IMDB-B00 entry point')
    logger.info('=' * 60)
    logger.info(
        'CA blend: coeff=%.4f start=%d ramp=%d rfl_gate=[%.3f, %.3f] '
        '(coeff=0.0 means CA is OFF — IMDB-B00 default)',
        config.ca_blend_coeff,
        config.ca_blend_start_iter,
        config.ca_blend_ramp_iters,
        config.ca_blend_rfl_lo,
        config.ca_blend_rfl_hi,
    )
    logger.info(
        'PGPE LR: center=%.4f stdev=%.4f init_std=%.4f decay_coef=%.4f decay_steps=%d',
        config.center_lr_gen,
        config.std_lr_gen,
        config.init_std_gen,
        config.lr_decay_coef,
        config.lr_decay_steps,
    )
    # Latent log: report the ACTUAL latent shape produced by `IMDb.reset_fn`,
    # not the BloodMNIST-style noise+discrete+continuous breakdown. The
    # current IMDb task constructs `latent = [noise, cat_one_hot]` and does
    # not yet sample continuous codes; the actual latent dim is
    # `noise_dim + n_discrete_codes`. Continuous codes are an open Phase 7
    # design question (see IMDB_PLAN.md "Architectural gap" note).
    actual_latent_dim = config.noise_dim + config.n_discrete_codes
    logger.info(
        'Latent: noise=%d discrete=%d (actual_total=%d) — continuous codes '
        'configured to %d but NOT yet wired into the IMDb task latent or '
        'into AttentionMaskGenerator; see IMDB_PLAN.md Phase 7 prep.',
        config.noise_dim,
        config.n_discrete_codes,
        actual_latent_dim,
        config.n_continuous_codes,
    )
    logger.info(
        'Architecture: encoder=%s seq_len=%d score_hidden=%d classifier_hidden=%d q_hidden=%d',
        config.encoder_name,
        config.seq_len,
        config.score_hidden,
        config.classifier_hidden,
        config.q_hidden,
    )
    logger.info(
        'Fitness mode: %s (hybrid=%s)',
        'static' if config.static_fitness_weights else 'adaptive',
        config.hybrid_fitness_weights,
    )

    components = build_components(config, logger)
    policy = components['policy']
    solver = components['solver']
    train_task = components['train_task_imdb']
    test_task = components['test_task_imdb']

    logger.info(
        'Components ready: HyperNet search dim=%d, classifier head params=%d, '
        'Q-head params=%d, train samples=%d, test samples=%d, label balance %s',
        policy.num_params,
        policy.num_params_classifier,
        policy.num_params_q,
        len(train_task.labels),
        len(test_task.labels),
        [int((train_task.labels == c).sum()) for c in (0, 1)],
    )

    # ---- Phase 7 trainer ----
    trainer = TrainerIMDb(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        batch_size=config.batch_size,
        log_dir=log_dir,
        checkpoint_interval=config.checkpoint_interval,
        checkpoint_dir=config.checkpoint_dir,
        seed=config.seed,
        render_interval=config.render_interval,
        render_n_reviews=config.render_n_reviews,
        render_seed=config.render_seed,
        logger=logger,
    )
    trainer.run()
    logger.info('Training complete. Artifacts in %s', log_dir)
    return components


if __name__ == '__main__':
    # GPU selection was already applied from sys.argv at module load (before
    # the JAX import). `cfg.gpu_id` is still parsed for logging completeness
    # but does not need to set the env var again.
    cfg = parse_args()
    main(cfg)

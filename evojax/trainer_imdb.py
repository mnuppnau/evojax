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

"""IMDb hard-attention trainer (Phase 7 deliverable).

A fresh, focused trainer for the `AttentionPolicy` + `PGPE_CA_Text` stack.
Intentionally NOT a fork of the BloodMNIST `Trainer`: this is a much simpler
shape because (a) there is no GAN — the discriminator analog is a supervised
classifier; (b) the encoder is frozen so there is no batch-stats syncing;
(c) the IMDb task produces its own latent inside `reset_fn`, so we do not
need a separate `Latent_Points` task. The result is roughly 500 lines vs the
1856-line `trainer.py`.

Per iteration:
  1. Sample one class-balanced batch from `IMDb` (50/50 pos/neg).
  2. Encoder forward once for the whole batch (shared across the PGPE pop).
  3. G-step: PGPE asks for `pop_size` HyperNet param sets; for each individual,
     the HyperNet emits AttentionMaskGenerator weights, MaskGen scores each
     position from `(h_t, z)`, Bernoulli sampling produces masks. Per-individual
     fitness components (classifier accuracy, MI signal, attention metrics)
     are aggregated and passed to `solver.tell()`.
  4. D-step: train the classifier head via backprop on the elite mask.
  5. Q-step: train the Q-head via backprop on the elite mask.
  6. Log + (every checkpoint_interval) save params.

Deferred (will be added once the Phase 8 smoke test passes):
  - Checkpoint resume from disk.
  - Mid-training periodic test evaluation.
  - Structured TSV logs analogous to `trainer.py`.
  - Multi-GPU population sharding.
"""

import logging
import os
import pickle
import time
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import random

from evojax.task.imdb import IMDb
from evojax.policy.attention_transformer import (
    AttentionPolicy, bernoulli_hard_mask, mean_pool_masked,
)
from evojax.algo.pgpe_ca_text import PGPE_CA_Text
from evojax.task.attention_metrics import compute_all
from evojax.util import create_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_balanced_sample(
    key: jnp.ndarray,
    input_ids: np.ndarray,
    attn_masks: np.ndarray,
    labels: np.ndarray,
    class_indices,
    batch_size: int,
):
    """Sample a per-class-balanced batch of (input_ids, attn, labels).

    `class_indices` is a list of int-arrays, one per sentiment class, holding
    the row indices that belong to that class. The IMDb task exposes
    `self.labels`; the trainer builds `class_indices` once at construction.
    """
    n_classes = len(class_indices)
    per_class = batch_size // n_classes
    remainder = batch_size - per_class * n_classes
    subkeys = random.split(key, n_classes + 1)

    picks = []
    for c in range(n_classes):
        cls = class_indices[c]
        idx = random.choice(
            subkeys[c], a=cls.shape[0], shape=(per_class,), replace=False)
        picks.append(cls[idx])
    if remainder > 0:
        cls = class_indices[0]
        idx = random.choice(
            subkeys[-1], a=cls.shape[0], shape=(remainder,), replace=False)
        picks.append(cls[idx])
    ix = jnp.concatenate(picks)
    # Shuffle so positives and negatives interleave (better gradient flow).
    shuffle_perm = random.permutation(subkeys[-1], batch_size)
    ix = ix[shuffle_perm]

    return (
        jnp.take(input_ids, ix, axis=0),
        jnp.take(attn_masks, ix, axis=0),
        jnp.take(labels, ix, axis=0),
    )


def _tile_codes(batch_size: int, n_codes: int) -> jnp.ndarray:
    """Cyclic one-hot tile, same shared-z protocol as BloodMNIST."""
    reps = (batch_size + n_codes - 1) // n_codes
    c = jnp.tile(jnp.arange(n_codes), reps)[:batch_size]
    return jax.nn.one_hot(c, n_codes)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TrainerIMDb(object):
    """Minimal trainer for IMDb hard-attention HyperNet-InfoGAN."""

    def __init__(
        self,
        policy: AttentionPolicy,
        solver: PGPE_CA_Text,
        train_task: IMDb,
        test_task: IMDb,
        max_iter: int = 290000,
        log_interval: int = 100,
        test_interval: int = 1000,
        batch_size: int = 64,
        log_dir: str = './log/imdb',
        checkpoint_interval: int = 5000,
        checkpoint_dir: Optional[str] = None,
        seed: int = 42,
        cls_lr: float = 1e-3,
        q_lr: float = 1e-3,
        render_interval: int = 0,
        render_n_reviews: int = 8,
        render_seed: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        self.policy = policy
        self.solver = solver
        self.train_task = train_task
        self.test_task = test_task
        self.max_iter = int(max_iter)
        self.log_interval = int(log_interval)
        self.test_interval = int(test_interval)
        self.batch_size = int(batch_size)
        self.log_dir = log_dir
        self.checkpoint_interval = int(checkpoint_interval)
        self.checkpoint_dir = checkpoint_dir or os.path.join(log_dir, 'checkpoints')
        self.seed = int(seed)
        self._key = random.PRNGKey(self.seed)
        self._logger = logger or create_logger(name='TrainerIMDb')

        # Render-on-checkpoint settings (0 = disabled).
        self.render_interval = int(render_interval)
        self.render_n_reviews = int(render_n_reviews)
        self.render_seed = int(render_seed)
        self._render_initialized = False
        self._tokenizer = None
        self._render_indices = None

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Pre-build class indices from the train task's labels for balanced
        # sampling. Binary sentiment → 50/50 by default.
        labels_np = np.asarray(train_task.labels)
        self.class_indices = [
            jnp.asarray(np.where(labels_np == c)[0])
            for c in range(int(labels_np.max()) + 1)
        ]
        self._logger.info(
            'Class-balanced sampling: %d classes, counts %s',
            len(self.class_indices),
            [int(ci.shape[0]) for ci in self.class_indices],
        )

        # Materialize the IMDb arrays once on-device so per-iter sampling
        # is a fast `jnp.take`.
        self._train_ids = jnp.asarray(train_task.data)
        self._train_attn = jnp.asarray(train_task.attention_masks)
        self._train_labels = jnp.asarray(train_task.labels)

        self.n_codes = int(policy.n_codes)
        self.hidden_dim = int(policy.hidden_dim)

        # Backprop optimizers for the classifier and Q-head.
        self._cls_optimizer = optax.adam(cls_lr)
        self._q_optimizer = optax.adam(q_lr)
        self._params_cls = policy.init_params_classifier
        self._params_q = policy.init_params_q
        self._opt_state_cls = self._cls_optimizer.init(self._params_cls)
        self._opt_state_q = self._q_optimizer.init(self._params_q)

        # JIT'd primitives (compiled on first call).
        self._encoder_forward = policy.encoder.forward  # already JIT'd
        self._evaluate_pop = jax.jit(self._build_evaluate_pop_fn())
        self._train_cls_step = jax.jit(self._build_cls_train_step())
        self._train_q_step = jax.jit(self._build_q_train_step())

    # ------------------------------------------------------------------
    # JIT primitives
    # ------------------------------------------------------------------

    def _build_evaluate_pop_fn(self):
        """Per-iteration G-step: vmap over the population, return fitness +
        attention metrics for `solver.tell()`."""

        policy = self.policy
        n_codes = self.n_codes

        def evaluate_one(flat_params_hn, hidden, encoder_attn, cat_codes,
                          labels, mask_key):
            # Reshape flat PGPE params back into the HyperNet pytree.
            hn_params = policy._format_single_params_hn_fn(flat_params_hn)
            mask_params = policy.hypernet_to_mask_params(hn_params)

            # MaskGen forward with z-conditioning (Phase 6 fix).
            logits = policy.mask_logits(mask_params, hidden, cat_codes)
            mask = bernoulli_hard_mask(logits, mask_key, encoder_attn)

            pooled = mean_pool_masked(hidden, mask)
            cls_logits = policy.classify(
                jax.lax.stop_gradient(self._params_cls), pooled)
            q_logits = policy.predict_codes(
                jax.lax.stop_gradient(self._params_q), pooled)

            # Per-individual per-code centroid in pooled-feature space.
            # Shape (n_codes, hidden_dim). Averaged across the population
            # outside the vmap to feed Topographic KS's `avg_per_code` slot.
            cat_f = cat_codes.astype(jnp.float32)
            per_code_count = cat_f.sum(axis=0)
            per_code_centroid = (
                cat_f.T @ pooled / (per_code_count[:, None] + 1e-6))

            # f_adv = classifier accuracy (higher = better).
            cls_pred = jnp.argmax(cls_logits, axis=-1)
            f_adv = (cls_pred == labels).astype(jnp.float32).mean()

            # f_mi proxy: negative cross-entropy of Q-head predicting cat code.
            cat_idx = jnp.argmax(cat_codes, axis=-1)
            ce = optax.softmax_cross_entropy_with_integer_labels(q_logits, cat_idx)
            f_mi = -ce.mean()

            # Attention metrics (uses `compute_all` from Phase 4).
            metrics = compute_all(
                mask, encoder_attn, cat_codes, labels, n_codes=n_codes)

            # `disc_logits` slot: the solver's `tell()` does
            # `jnp.mean(disc_logits, axis=(0, 1))` to collapse to (n_codes,),
            # so the per-individual value must be 2D (batch, n_codes). After
            # vmap over the pop axis the shape passed in is
            # (pop_size, batch, n_codes), matching the BloodMNIST shape.

            # Placeholders for the solver's BloodMNIST-flavored slots that
            # do not carry meaning in IMDb (kept zero so rank_normalize is
            # well-defined; they have zero weight in the IMDb fitness anyway).
            zero = jnp.float32(0.0)

            return {
                'fitness_adv': f_adv,
                'fitness_mi': f_mi,
                'q_logits': q_logits,    # per-indiv (batch, n_codes); vmap -> (pop, batch, n_codes)
                'per_code_centroid': per_code_centroid,           # (n_codes, hidden)
                'pop_var': pooled.var(),                          # scalar diversity proxy
                'r_cons': zero,
                'r_sense': zero,
                'r_intra': zero,
                'r_shape_div': zero,
                'r_shape_div_min': zero,
                'sparsity_band_reward': metrics['sparsity_band_reward'],
                'span_continuity': metrics['span_continuity'],
                'code_disjointness': metrics['code_disjointness'],
                'position_entropy': metrics['position_entropy'],
                'sparsity_density': metrics['sparsity_density'],
                'pos_neg_attention_overlap': metrics['pos_neg_attention_overlap'],
                'normative_penalty': zero,
                'mask': mask,                                     # for D-step
            }

        def evaluate_pop(flat_params_pop, hidden, encoder_attn, cat_codes,
                         labels, mask_keys):
            return jax.vmap(
                evaluate_one,
                in_axes=(0, None, None, None, None, 0),
            )(flat_params_pop, hidden, encoder_attn, cat_codes, labels, mask_keys)

        return evaluate_pop

    def _build_cls_train_step(self):
        policy = self.policy
        optimizer = self._cls_optimizer

        def loss_fn(params_cls, hidden, mask, labels):
            pooled = mean_pool_masked(hidden, mask)
            logits = policy.classify(params_cls, pooled)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, labels).mean()

        def step(params_cls, opt_state, hidden, mask, labels):
            loss, grads = jax.value_and_grad(loss_fn)(
                params_cls, hidden, mask, labels)
            updates, opt_state = optimizer.update(grads, opt_state, params_cls)
            params_cls = optax.apply_updates(params_cls, updates)
            return params_cls, opt_state, loss

        return step

    def _build_q_train_step(self):
        policy = self.policy
        optimizer = self._q_optimizer

        def loss_fn(params_q, hidden, mask, cat_codes):
            pooled = mean_pool_masked(hidden, mask)
            logits = policy.predict_codes(params_q, pooled)
            cat_idx = jnp.argmax(cat_codes, axis=-1)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, cat_idx).mean()

        def step(params_q, opt_state, hidden, mask, cat_codes):
            loss, grads = jax.value_and_grad(loss_fn)(
                params_q, hidden, mask, cat_codes)
            updates, opt_state = optimizer.update(grads, opt_state, params_q)
            params_q = optax.apply_updates(params_q, updates)
            return params_q, opt_state, loss

        return step

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_iter(self, i: int, t_iter: float, eval_out: dict,
                  cls_loss: float, q_loss: float,
                  adv_elite_idx: int, mi_elite_idx: int) -> None:
        # Mean over the population for the summary log.
        diag = self.solver.get_diagnostics()

        def m(name):
            v = eval_out[name]
            return float(np.asarray(v).mean())

        msg = (
            f'iter={i:>6d}  '
            f'time={t_iter:.2f}s  '
            f'f_adv={m("fitness_adv"):.4f}  '
            f'f_mi={m("fitness_mi"):.4f}  '
            f'elite_adv={adv_elite_idx}  '
            f'elite_mi={mi_elite_idx}  '
            f'sparsity={m("sparsity_density"):.3f}  '
            f'span={m("span_continuity"):.2f}  '
            f'disjoint={m("code_disjointness"):.3f}  '
            f'pos_ent={m("position_entropy"):.3f}  '
            f'pos_neg={m("pos_neg_attention_overlap"):.3f}  '
            f'cls_loss={cls_loss:.4f}  '
            f'q_loss={q_loss:.4f}'
        )
        # Stitch in a few solver-side diagnostics if present.
        for k in ('w_adv', 'w_mi', 'stdev_mean', 'ca_blend'):
            if k in diag:
                msg += f'  {k}={diag[k]:.4f}'
        self._logger.info(msg)

    # ------------------------------------------------------------------
    # Checkpointing (minimal: pickle of evolved + backprop params)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Trajectory rendering (lazy init; only loads the tokenizer + picks
    # the held-out review subset the first time it's needed).
    # ------------------------------------------------------------------

    def _init_render(self) -> None:
        if self._render_initialized:
            return
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.policy.encoder_name)
        rng = np.random.default_rng(self.render_seed)
        n = min(self.render_n_reviews, len(self.test_task.labels))
        self._render_indices = rng.choice(
            len(self.test_task.labels), size=n, replace=False)
        self._render_initialized = True
        self._logger.info(
            'Render-on-checkpoint enabled: every %d iters, %d test reviews, '
            'snapshots in %s',
            self.render_interval, n, self.checkpoint_dir)

    def _render_snapshot(self, i: int) -> None:
        # Import lazily to avoid importing transformers/jax helpers when
        # render is disabled.
        from evojax.render_imdb import render_checkpoint_to_html
        self._init_render()
        out_path = os.path.join(
            self.checkpoint_dir, f'render_iter-{i:06d}.html')
        center_flat = np.asarray(self.solver._center)
        summary = render_checkpoint_to_html(
            policy=self.policy,
            center_flat=center_flat,
            tokenizer=self._tokenizer,
            test_input_ids=self.test_task.data,
            test_attn_masks=self.test_task.attention_masks,
            test_labels=self.test_task.labels,
            indices=self._render_indices,
            n_codes=self.n_codes,
            out_path=out_path,
            title_suffix=f'iter {i}',
        )
        # One-line density summary so the trajectory is greppable from
        # the log without opening every HTML file.
        d = summary['density_per_code']
        density_str = ' '.join(f'c{k}={d[k]:.3f}' for k in range(len(d)))
        self._logger.info(
            'rendered snapshot %s  density: %s',
            out_path, density_str)

    def _save_checkpoint(self, i: int) -> None:
        """Pickle the evolved + backprop parameters for later restart/analysis.

        TODO (resume path, deferred to Phase 8/9 cleanup): `center` and
        `stdev` are saved as flat numpy arrays, while the policy/solver
        carry the HyperNet as a pytree. A resume function must:
          1. Load the flat `center` array
          2. Call `policy._format_single_params_hn_fn(center)` to reshape
             into the HyperNet pytree (round-trip is needed for diagnostics
             and any future direct-pytree consumers).
          3. Assign the flat array straight back into `solver._center` and
             `solver._stdev` (the solver itself uses the flat form).
          4. Restore `_params_classifier`, `_params_q`, and re-init the
             Adam opt states (or save those too).
        """
        path = os.path.join(self.checkpoint_dir, f'iter-{i:06d}.pkl')
        payload = {
            'iter': i,
            'center': np.asarray(self.solver._center),
            'stdev': np.asarray(self.solver._stdev),
            'params_classifier': jax.tree_util.tree_map(np.asarray, self._params_cls),
            'params_q': jax.tree_util.tree_map(np.asarray, self._params_q),
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        self._logger.info('saved checkpoint %s', path)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._logger.info('Starting training for %d iterations.', self.max_iter)
        cat_codes_base = _tile_codes(self.batch_size, self.n_codes)

        for i in range(self.max_iter):
            t0 = time.time()

            # 1. Sample class-balanced batch.
            self._key, batch_key, mask_root_key, code_key = random.split(self._key, 4)
            input_ids, encoder_attn, labels = _class_balanced_sample(
                batch_key, self._train_ids, self._train_attn, self._train_labels,
                self.class_indices, self.batch_size,
            )

            # Per-iteration shuffle of cat_codes. The tile pattern is fixed
            # at `[0,1,2,...,K-1,0,1,...]`; reshuffling breaks any
            # fixed-position-→-code correspondence that the HyperNet could
            # otherwise exploit. Even though reviews are already randomly
            # sampled, position 0 always carrying code 0 (etc.) means the
            # classifier+Q could in principle learn a position-→-code prior
            # over many iterations. Shuffling is a cheap robustness guard.
            shuffle_perm = random.permutation(code_key, self.batch_size)
            cat_codes_iter = cat_codes_base[shuffle_perm]

            # 2. Encoder forward (frozen).
            hidden = self._encoder_forward(input_ids, encoder_attn)

            # 3. G-step: PGPE ask, eval population, tell.
            solutions, _ = self.solver.ask()
            mask_keys = random.split(mask_root_key, solutions.shape[0])
            eval_out = self._evaluate_pop(
                solutions, hidden, encoder_attn, cat_codes_iter, labels, mask_keys,
            )

            # 3b. Wrap into the solver's tell signature.
            # avg_per_code: population mean of per-code centroids,
            # shape (n_codes, hidden_dim). Topographic KS expects this.
            avg_per_code_current = eval_out['per_code_centroid'].mean(axis=0)
            pop_size = solutions.shape[0]
            # `safety_ratios` and `spreads` are BloodMNIST-flavored slots
            # that feed `update_normative_ks`. We pass constant `jnp.ones`
            # because (a) we don't compute analogous quantities for text
            # and (b) any CA gradient that would react to these is gated
            # by `ca_blend_coeff`, which is 0.0 for IMDB-B00. The resulting
            # `norm_*_floor`/`_ceiling` diagnostics are NOT meaningful in
            # baseline runs (Phase 5 review item #4 — see IMDB_PLAN.md).
            self.solver.tell(
                fitness_adv=eval_out['fitness_adv'],
                fitness_mi=eval_out['fitness_mi'],
                disc_logits=eval_out['q_logits'],               # (pop, batch, n_codes)
                pop_var=eval_out['pop_var'],
                avg_per_code=avg_per_code_current,
                r_cons=eval_out['r_cons'],
                r_sense=eval_out['r_sense'],
                r_intra=eval_out['r_intra'],
                r_shape_div=eval_out['r_shape_div'],
                r_shape_div_min=eval_out['r_shape_div_min'],
                sparsity_band_reward=eval_out['sparsity_band_reward'],
                span_continuity=eval_out['span_continuity'],
                code_disjointness=eval_out['code_disjointness'],
                position_entropy=eval_out['position_entropy'],
                sparsity_density=eval_out['sparsity_density'],
                pos_neg_attention_overlap=eval_out['pos_neg_attention_overlap'],
                normative_penalty=eval_out['normative_penalty'],
                safety_ratios=jnp.ones((pop_size, self.n_codes, self.n_codes), dtype=jnp.float32),
                spreads=jnp.ones((pop_size, self.n_codes, 1), dtype=jnp.float32),
                adv=True,
            )
            # Update the solver's RFL surrogate with the population-mean classifier
            # accuracy so any future health-gated CA logic still has a runtime hook.
            self.solver.set_runtime_metrics(
                real_fake_loss=float(eval_out['fitness_adv'].mean()))

            # 4. D-step: classifier backprop on the ACCURACY-elite mask.
            adv_elite_idx = int(jnp.argmax(eval_out['fitness_adv']))
            adv_elite_mask = eval_out['mask'][adv_elite_idx]
            self._params_cls, self._opt_state_cls, cls_loss = self._train_cls_step(
                self._params_cls, self._opt_state_cls, hidden,
                adv_elite_mask, labels)

            # 5. Q-step: Q-head backprop on the MI-ELITE mask, NOT the
            # accuracy-elite. Decouples Q training from f_adv so the feedback
            # loop "f_adv elite mask is code-agnostic → Q trains on
            # code-agnostic data → Q stays at chance → f_mi stays low" cannot
            # close. The MI elite reinforces whichever individual's mask
            # carries the cleanest code signal so far. Early in training all
            # MI values are near zero and the selection is effectively
            # random, which is the desired behavior.
            mi_elite_idx = int(jnp.argmax(eval_out['fitness_mi']))
            mi_elite_mask = eval_out['mask'][mi_elite_idx]
            self._params_q, self._opt_state_q, q_loss = self._train_q_step(
                self._params_q, self._opt_state_q, hidden,
                mi_elite_mask, cat_codes_iter)

            t_iter = time.time() - t0

            # 6. Log + checkpoint + (optional) render.
            if i % self.log_interval == 0:
                self._log_iter(
                    i, t_iter, eval_out,
                    float(cls_loss), float(q_loss),
                    adv_elite_idx, mi_elite_idx,
                )
            if self.checkpoint_interval > 0 and i > 0 and i % self.checkpoint_interval == 0:
                self._save_checkpoint(i)
            if (self.render_interval > 0 and i > 0
                    and i % self.render_interval == 0):
                self._render_snapshot(i)

        self._logger.info('Training complete (%d iterations).', self.max_iter)
        self._save_checkpoint(self.max_iter)
        if self.render_interval > 0:
            self._render_snapshot(self.max_iter)

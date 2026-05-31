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

"""Attention-quality metrics for the IMDb hard-attention InfoGAN.

These are the text-domain analog of the morphology metrics in
`evojax/task/latent.py` (sense / cons / intra) — specialized to discrete
binary masks over tokens. Each function is jittable and vmappable; the
caller (Phase 5 solver) is responsible for any population-axis vmapping.

Metrics:
    sparsity_density         mean attended-fraction over non-pad positions
    sparsity_band_reward     peaked reward inside a target sparsity band
    span_continuity          mean contiguous-run length of attended tokens
    position_entropy         normalized entropy of position histogram
    per_code_attention_overlap   avg pairwise cosine sim of per-code centroids
    pos_neg_attention_overlap    cosine sim between pos- and neg-class centroids
    compute_all              convenience aggregator that returns a dict

Reads carried forward from Phase 3.5 capacity check (as monitoring guards):
    `sparsity_density` must stay in roughly the 0.05--0.20 band
    `per_code_attention_overlap` must drop below 1.0 for disentanglement
"""

from typing import Dict

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Sparsity (1D mask density)
# ---------------------------------------------------------------------------

def sparsity_density(mask: jnp.ndarray,
                     encoder_attention_mask: jnp.ndarray) -> jnp.ndarray:
    """Mean attended-fraction over non-pad positions, batch-averaged.

    mask:    (batch, seq_len), 0/1
    encoder: (batch, seq_len), 0/1 (1 = non-pad)
    Returns: scalar in [0, 1].
    """
    mask_f = mask.astype(jnp.float32)
    attn_f = encoder_attention_mask.astype(jnp.float32)
    n_attended = mask_f.sum(axis=-1)
    n_valid = attn_f.sum(axis=-1) + 1e-6
    return (n_attended / n_valid).mean()


def sparsity_band_reward(
    mask: jnp.ndarray,
    encoder_attention_mask: jnp.ndarray,
    target_low: float = 0.05,
    target_high: float = 0.20,
) -> jnp.ndarray:
    """Peaked reward: 1.0 inside `[target_low, target_high]`, falls off linearly.

    Penalty distance from the band is normalized by `target_low` (NOT by the
    band width). With defaults `target_low=0.05`, `target_high=0.20` the
    reward bottoms out at 0 at density=0.00 on the low side and density=0.25
    on the high side. This is asymmetric in absolute-density terms — going
    too sparse hurts faster than going too dense — and is intentional: at
    very low density the pooled vector loses information rapidly, while at
    high density the classifier just sees more context. Phase 8 may want to
    tune `target_low` if the band is too punishing on the sparse side.
    """
    mask_f = mask.astype(jnp.float32)
    attn_f = encoder_attention_mask.astype(jnp.float32)
    n_attended = mask_f.sum(axis=-1)
    n_valid = attn_f.sum(axis=-1) + 1e-6
    density = n_attended / n_valid                                       # (B,)

    too_low = jnp.maximum(target_low - density, 0.0)
    too_high = jnp.maximum(density - target_high, 0.0)
    penalty = (too_low + too_high) / target_low
    reward = 1.0 - jnp.clip(penalty, 0.0, 1.0)
    return reward.mean()


# ---------------------------------------------------------------------------
# Span continuity
# ---------------------------------------------------------------------------

def span_continuity(mask: jnp.ndarray) -> jnp.ndarray:
    """Mean contiguous-run length of attended tokens.

    Higher = attention forms long contiguous spans (sentiment phrases).
    Lower  = attention scatters into isolated single-token picks (likely
             shortcut: punctuation / stopwords / position-only patterns).
    """
    mask_i = mask.astype(jnp.int32)
    batch = mask_i.shape[0]
    zeros = jnp.zeros((batch, 1), dtype=mask_i.dtype)
    padded = jnp.concatenate([zeros, mask_i, zeros], axis=-1)
    diffs = padded[:, 1:] - padded[:, :-1]
    n_runs = (diffs == 1).astype(jnp.float32).sum(axis=-1)               # (B,)
    total_attended = mask_i.sum(axis=-1).astype(jnp.float32)             # (B,)
    avg_run_length = total_attended / (n_runs + 1e-6)
    return avg_run_length.mean()


# ---------------------------------------------------------------------------
# Position entropy (batch-aggregated)
# ---------------------------------------------------------------------------

def position_entropy(mask: jnp.ndarray,
                     encoder_attention_mask: jnp.ndarray) -> jnp.ndarray:
    """Normalized entropy of where attention concentrates across positions.

    Computes a per-position attention frequency (averaged over the batch,
    conditional on the position being valid), normalizes it into a
    probability distribution over positions, then returns the entropy
    divided by `log(seq_len)` so the result is in [0, 1].

    High → attention is spread across positions (content-driven).
    Low  → attention concentrates at specific positions (position shortcut).
    """
    mask_f = mask.astype(jnp.float32)
    attn_f = encoder_attention_mask.astype(jnp.float32)
    # Per-position attended-fraction conditional on validity.
    numer = (mask_f * attn_f).sum(axis=0)                                # (L,)
    denom = attn_f.sum(axis=0) + 1e-6                                    # (L,)
    p = numer / denom                                                    # freqs in [0,1]
    p = p / (p.sum() + 1e-6)                                             # probabilities
    entropy = -(p * jnp.log(p + 1e-12)).sum()
    seq_len = jnp.asarray(mask.shape[-1], dtype=jnp.float32)
    max_entropy = jnp.log(seq_len)
    return entropy / (max_entropy + 1e-6)


# ---------------------------------------------------------------------------
# Cross-code attention overlap (the disentanglement guard)
# ---------------------------------------------------------------------------

def per_code_attention_overlap(
    masks: jnp.ndarray,
    cat_codes_one_hot: jnp.ndarray,
    n_codes: int,
) -> jnp.ndarray:
    """Average pairwise cosine similarity between per-code mask centroids.

    masks:             (batch, seq_len)
    cat_codes_one_hot: (batch, n_codes)
    n_codes:           static int (used to size the off-diagonal mean)

    Centroid for code k: mean mask over all batch samples assigned to k.
    Then compute the (n_codes, n_codes) cosine-similarity matrix between
    centroids and average the off-diagonal entries. The diagonal is 1 by
    construction; the off-diagonal captures inter-code redundancy.

    Returns a scalar in [-1, 1]. Lower is better for disentanglement.
    Phase 3.5 monitoring rule: stays well below 1.0 once training is healthy;
    a value pinned near 1.0 means every discrete code emits the same mask.
    """
    masks_f = masks.astype(jnp.float32)
    codes_f = cat_codes_one_hot.astype(jnp.float32)
    per_code_count = codes_f.sum(axis=0)                                 # (K,)
    code_centroids = (
        codes_f.T @ masks_f / (per_code_count[:, None] + 1e-6)           # (K, L)
    )
    norms = jnp.linalg.norm(code_centroids, axis=-1, keepdims=True) + 1e-6
    normalized = code_centroids / norms
    sim_matrix = normalized @ normalized.T
    eye = jnp.eye(n_codes)
    n_pairs = n_codes * (n_codes - 1)
    off_diag_sum = (sim_matrix * (1.0 - eye)).sum()
    return off_diag_sum / (n_pairs + 1e-6)


def pos_neg_attention_overlap(
    masks: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Cosine similarity between positive-class and negative-class centroids.

    Diagnostic, not a fitness target. Detects whether attention is being
    conditioned on the sentiment LABEL (positive vs negative). Latent-code
    conditioning is what `per_code_attention_overlap` measures separately;
    this metric is orthogonal to that one.

    A value pinned near 1.0 means the same mask is being applied regardless
    of review polarity, which would make sentiment classification rely
    entirely on the encoder's content representation rather than on
    sentiment-relevant token selection.
    """
    masks_f = masks.astype(jnp.float32)
    is_pos = (labels == 1).astype(jnp.float32)
    is_neg = (labels == 0).astype(jnp.float32)
    pos_centroid = (is_pos[:, None] * masks_f).sum(axis=0) / (
        is_pos.sum() + 1e-6)
    neg_centroid = (is_neg[:, None] * masks_f).sum(axis=0) / (
        is_neg.sum() + 1e-6)
    pos_norm = jnp.linalg.norm(pos_centroid) + 1e-6
    neg_norm = jnp.linalg.norm(neg_centroid) + 1e-6
    return (pos_centroid @ neg_centroid) / (pos_norm * neg_norm)


# ---------------------------------------------------------------------------
# Aggregator for diagnostic logging
# ---------------------------------------------------------------------------

def compute_all(
    masks: jnp.ndarray,
    encoder_attention_mask: jnp.ndarray,
    cat_codes_one_hot: jnp.ndarray,
    labels: jnp.ndarray,
    n_codes: int,
    sparsity_low: float = 0.05,
    sparsity_high: float = 0.20,
) -> Dict[str, jnp.ndarray]:
    """Compute all attention metrics in one pass.

    Caller chooses how to consume the returned scalars (rank-normalize for
    fitness, or just log for diagnostics). Splits the fitness-target metrics
    from the pure diagnostics so Phase 5 wiring can grab a focused subset.

    IMPORTANT DIRECTION NOTE for Phase 5 wiring: this aggregator emits
    `code_disjointness = 1.0 - per_code_attention_overlap`, already in the
    fitness-friendly "higher is better" direction. Use
    `compute_all['code_disjointness']` in the fitness composition; do NOT
    call the standalone `per_code_attention_overlap` for a fitness term, as
    its sign is the opposite (lower-is-better) and combining the two would
    silently invert the disentanglement pressure relative to other fitness
    components. The standalone version is fine for diagnostic logging only.
    """
    return {
        # Fitness-target metrics (Phase 5 will rank-normalize these).
        'sparsity_band_reward': sparsity_band_reward(
            masks, encoder_attention_mask, sparsity_low, sparsity_high),
        'span_continuity': span_continuity(masks),
        'code_disjointness': 1.0 - per_code_attention_overlap(
            masks, cat_codes_one_hot, n_codes),
        # Diagnostics (logged but not in fitness).
        'sparsity_density': sparsity_density(masks, encoder_attention_mask),
        'position_entropy': position_entropy(masks, encoder_attention_mask),
        'pos_neg_attention_overlap': pos_neg_attention_overlap(masks, labels),
    }

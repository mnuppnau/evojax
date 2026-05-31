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

"""HyperNet-driven hard-attention policy for IMDb sentiment classification.

Architecture roles:
  - FrozenDistilBertEncoder: pretrained DistilBERT (params NOT evolved)
  - HyperNetwork: PGPE-evolved; emits AttentionMaskGenerator weights
  - AttentionMaskGenerator: per-token MLP scoring -> Bernoulli mask
  - Classifier: backprop-trained sentiment head over masked-pooled features
  - QHead: backprop-trained code-reconstruction head for the MI term

The Bernoulli sample is hard. Backprop never needs to flow through the mask
because the HyperNet is evolved (not gradient-descended) and the encoder is
frozen; the classifier and Q-head receive ordinary gradients w.r.t. their own
parameters with the mask treated as input data.
"""

import logging
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn
from evojax.util import get_single_params_format_fn


# ---------------------------------------------------------------------------
# 1. Frozen encoder wrapper
# ---------------------------------------------------------------------------

class FrozenDistilBertEncoder:
    """Wraps `FlaxDistilBertModel` with frozen pretrained weights.

    The encoder parameters live inside this object and are NEVER exposed to
    PGPE or the policy parameter set. `forward` returns hidden states with a
    `stop_gradient` so backprop into the encoder is impossible by construction.
    """

    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        from transformers import FlaxDistilBertModel  # heavy import; defer
        self.model_name = model_name
        self._model = FlaxDistilBertModel.from_pretrained(model_name)
        self.params = self._model.params
        self.hidden_dim = int(self._model.config.hidden_size)
        self.num_layers = int(self._model.config.num_hidden_layers)

        def _forward(input_ids: jnp.ndarray,
                     attention_mask: jnp.ndarray) -> jnp.ndarray:
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                params=self.params,
                train=False,
            )
            return jax.lax.stop_gradient(out.last_hidden_state)

        self.forward = jax.jit(_forward)

    def __call__(self,
                 input_ids: jnp.ndarray,
                 attention_mask: jnp.ndarray) -> jnp.ndarray:
        return self.forward(input_ids, attention_mask)


# ---------------------------------------------------------------------------
# 2. HyperNetwork (mirrors convnet.py; weight-emission MLP)
# ---------------------------------------------------------------------------

class HyperNetwork(nn.Module):
    chunk_size: int = 512
    n_chunks: int = 100
    chunk_embed_dim: int = 16

    @nn.compact
    def __call__(self,
                 chunk_ids: jnp.ndarray,
                 context: jnp.ndarray) -> jnp.ndarray:
        chunk_emb = nn.Embed(self.n_chunks, self.chunk_embed_dim)(chunk_ids)
        x = jnp.concatenate([chunk_emb, context], axis=-1)
        x = nn.Dense(48)(x)
        x = nn.gelu(x)
        x = nn.Dense(48)(x)
        x = nn.gelu(x)
        weights = nn.Dense(
            self.chunk_size,
            kernel_init=jax.nn.initializers.normal(stddev=0.01),
        )(x)
        return weights


# ---------------------------------------------------------------------------
# 3. AttentionMaskGenerator (its weights are produced by the HyperNet)
# ---------------------------------------------------------------------------

class AttentionMaskGenerator(nn.Module):
    """Per-token MLP that scores each token from `(hidden_state_t, z)`.

    Returns per-token logits (pre-sigmoid). Bernoulli sampling and the
    encoder-padding mask are applied outside this module.

    z-conditioning (added 2026-05-16, addresses Phase 6 review finding):
    each position's score MUST depend on the latent code `z`, otherwise the
    HyperNet-emitted MaskGen produces identical masks for different codes
    and the Q-head cannot recover `z` from the mask. We broadcast `z`
    across the sequence axis and concatenate with the encoder hidden state
    at each position, so the same MLP receives `(h_t, z)` and the code
    directly steers which tokens are selected. Pass the discrete latent
    code only (e.g. the `n_codes`-dim one-hot from `state.cat_codes`); the
    62-dim noise vector would dominate the concatenation and dilute the
    code signal — see the Phase 6 review note in IMDB_PLAN.md.

    Position-blindness: still true at the MLP level — the same `(h_t, z)`
    function is applied at every position. Positional information enters
    only via the encoder hidden state (which already carries positional
    encoding). A position-bias shortcut, if it emerges, is diagnosed from
    `position_entropy` in `attention_metrics`, not from internal structure.
    """
    hidden_dim: int = 768
    score_hidden: int = 64
    z_dim: int = 8   # fixed at init time; chunk adapter walks the resulting tree

    @nn.compact
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 z: jnp.ndarray) -> jnp.ndarray:
        # hidden_states: (batch, seq_len, hidden_dim)
        # z:             (batch, z_dim)   — typically discrete one-hot
        # Broadcast z across positions and concatenate.
        z_expanded = jnp.broadcast_to(
            z[:, None, :],
            (hidden_states.shape[0], hidden_states.shape[1], z.shape[-1]),
        )
        x = jnp.concatenate([hidden_states, z_expanded], axis=-1)
        x = nn.Dense(self.score_hidden, name='score_h1')(x)
        x = nn.gelu(x)
        x = nn.Dense(1, name='score_h2')(x)
        return jnp.squeeze(x, axis=-1)  # (batch, seq_len)


# ---------------------------------------------------------------------------
# 4. ParameterAdapter for AttentionMaskGenerator
#    (Cleaner than convnet.ParameterAdapter — drops image-specific
#     depth/scale heuristics and uses only layer-id one-hot context.)
# ---------------------------------------------------------------------------

class AttentionParameterAdapter:
    """Maps a HyperNetwork output back into the AttentionMaskGenerator
    parameter pytree. The HyperNet sees one chunk at a time with a context
    vector that identifies which target-layer the chunk belongs to.
    """

    def __init__(self, target_init_params, chunk_size: int = 512):
        # Edge case: a single-layer target produces an all-zero `layer_ids`
        # array and a constant one-hot context, which collapses the HyperNet to
        # a pure chunk-embedding model. Fine for the current 2-layer
        # AttentionMaskGenerator; revisit if the target gets simpler.
        from jax import tree_util
        self.chunk_size = chunk_size
        self.target_tree = tree_util.tree_structure(target_init_params)

        flat_params, _ = tree_util.tree_flatten(target_init_params)
        self.param_sizes = [int(np.prod(p.shape)) for p in flat_params]
        self.param_shapes = [p.shape for p in flat_params]

        layer_ids_list, chunk_ids_list = [], []
        for layer_idx, size in enumerate(self.param_sizes):
            n_chunks = (size + chunk_size - 1) // chunk_size
            layer_ids_list.append(np.full(n_chunks, layer_idx, dtype=np.int32))
            chunk_ids_list.append(np.arange(n_chunks, dtype=np.int32))

        layer_ids_np = np.concatenate(layer_ids_list)
        chunk_ids_np = np.concatenate(chunk_ids_list)

        self.layer_ids = jnp.asarray(layer_ids_np)
        self.chunk_ids = jnp.asarray(chunk_ids_np)
        self.total_chunks = int(self.layer_ids.shape[0])

        self.N_LAYERS = int(layer_ids_np.max()) + 1
        self.N_CHUNKS = int(chunk_ids_np.max()) + 1
        self.CHUNK_EMBED_DIM = 16
        # Context = layer one-hot only (no spatial scale for attention MLP).
        self.CONTEXT_DIM = self.N_LAYERS

        l_oh = jax.nn.one_hot(self.layer_ids, self.N_LAYERS)
        self.static_context = l_oh

        self.layer_chunks_split = [
            (s + chunk_size - 1) // chunk_size for s in self.param_sizes]
        self.layer_chunks_split_indices = (
            np.cumsum(self.layer_chunks_split)[:-1].tolist())

    def _make_hn(self) -> HyperNetwork:
        return HyperNetwork(
            chunk_size=self.chunk_size,
            n_chunks=self.N_CHUNKS,
            chunk_embed_dim=self.CHUNK_EMBED_DIM,
        )

    def init_hypernet(self, rng):
        dummy_chunk_ids = jnp.zeros((self.total_chunks,), dtype=jnp.int32)
        dummy_context = jnp.zeros((self.total_chunks, self.CONTEXT_DIM))
        return self._make_hn().init(rng, dummy_chunk_ids, dummy_context)

    def generate_params(self, hypernet_params):
        from jax import tree_util
        flat_chunks = self._make_hn().apply(
            hypernet_params, self.chunk_ids, self.static_context)

        chunks_per_layer = jnp.split(
            flat_chunks, self.layer_chunks_split_indices)

        reshaped_params = []
        for i, chunks in enumerate(chunks_per_layer):
            flat = chunks.reshape(-1)
            reshaped = flat[:self.param_sizes[i]].reshape(self.param_shapes[i])
            reshaped_params.append(reshaped)

        return tree_util.tree_unflatten(self.target_tree, reshaped_params)


# ---------------------------------------------------------------------------
# 5. Classifier and Q heads (trained by backprop in Phase 7)
# ---------------------------------------------------------------------------

class Classifier(nn.Module):
    head_hidden: int = 128
    n_classes: int = 2

    @nn.compact
    def __call__(self, pooled: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.head_hidden, name='cls_h1')(pooled)
        x = nn.gelu(x)
        x = nn.Dense(self.n_classes, name='cls_h2')(x)
        return x


class QHead(nn.Module):
    head_hidden: int = 128
    n_codes: int = 8

    @nn.compact
    def __call__(self, pooled: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.head_hidden, name='q_h1')(pooled)
        x = nn.gelu(x)
        x = nn.Dense(self.n_codes, name='q_h2')(x)
        return x


# ---------------------------------------------------------------------------
# 6. Mask sampling + pooling helpers
# ---------------------------------------------------------------------------

def bernoulli_hard_mask(
    logits: jnp.ndarray,
    key: jnp.ndarray,
    encoder_attention_mask: jnp.ndarray,
    fallback_to_cls: bool = True,
) -> jnp.ndarray:
    """Hard Bernoulli sample of an attention mask. NON-DIFFERENTIABLE.

    Used only inside PGPE rollouts and evaluation. The HyperNet is evolved (no
    gradient required), the encoder is frozen, and the classifier / Q-head
    receive ordinary backprop with the mask treated as input data — so no
    straight-through estimator is needed and this function is never on a
    gradient path.

    When `fallback_to_cls=True` (default), the [CLS] position (index 0) is
    force-attended whenever Bernoulli sampling + padding produces an all-zero
    mask. This guards `mean_pool_masked` from a near-zero pooled vector that
    the classifier would otherwise be free to read as a class indicator
    decoupled from review content. Cost: when the HyperNet emits an empty
    mask, sparsity metrics will report 1/seq_len rather than 0 — which is
    acceptable; empty masks should be punished by the sparsity-floor fitness
    term anyway, not silently emit a degenerate pool.
    """
    probs = jax.nn.sigmoid(logits)
    sampled = (random.uniform(key, logits.shape) < probs).astype(jnp.int32)
    sampled = sampled * encoder_attention_mask.astype(jnp.int32)

    if fallback_to_cls:
        n_attended = sampled.sum(axis=1, keepdims=True)               # (B, 1)
        empty = (n_attended == 0).astype(sampled.dtype)               # (B, 1)
        cls_indicator = jnp.zeros(
            (1, sampled.shape[1]), dtype=sampled.dtype).at[0, 0].set(1)
        sampled = sampled + empty * cls_indicator                     # at most one fix per row
    return sampled


def mean_pool_masked(hidden_states: jnp.ndarray,
                     mask: jnp.ndarray) -> jnp.ndarray:
    """Mean over attended positions; length-normalized.

    Assumes `bernoulli_hard_mask(..., fallback_to_cls=True)` was used so at
    least one position is attended per sample. The 1e-6 floor remains as a
    paranoia guard against any caller bypassing the fallback.
    """
    mask_f = mask.astype(hidden_states.dtype)
    masked_h = hidden_states * mask_f[:, :, None]
    n_attended = mask_f.sum(axis=1, keepdims=True) + 1e-6
    return masked_h.sum(axis=1) / n_attended


# ---------------------------------------------------------------------------
# 7. Top-level policy
# ---------------------------------------------------------------------------

class AttentionPolicy(PolicyNetwork):
    """HyperNet-driven hard-attention policy.

    PGPE evolves `init_params_hypernet`. The HyperNet generates
    AttentionMaskGenerator params each rollout. Classifier and Q-head are
    trained separately via backprop (Phase 7).

    The full PolicyNetwork.get_actions integration is deferred to Phase 6/7;
    this class currently exposes the building blocks the trainer will need:
        - encoder forward (frozen)
        - hypernet -> attention-mask-generator params
        - mask-generator forward
        - classifier / q-head forward
        - parameter shapes / format helpers for PGPE
    """

    def __init__(
        self,
        seq_len: int = 256,
        n_codes: int = 8,
        score_hidden: int = 64,
        classifier_hidden: int = 128,
        q_hidden: int = 128,
        chunk_size: int = 512,
        encoder_name: str = 'distilbert-base-uncased',
        logger: logging.Logger = None,
    ):
        self._logger = logger or create_logger('AttentionPolicy')
        self.seq_len = int(seq_len)
        self.n_codes = int(n_codes)
        self.encoder_name = encoder_name

        # 1. Encoder (frozen pretrained DistilBERT).
        self.encoder = FrozenDistilBertEncoder(model_name=encoder_name)
        self.hidden_dim = self.encoder.hidden_dim
        self._logger.info(
            'Encoder: %s, hidden_dim=%d, num_layers=%d',
            encoder_name, self.hidden_dim, self.encoder.num_layers)

        # 2. AttentionMaskGenerator (params produced by HyperNet).
        # z-conditioned: forward signature is (hidden_states, z) where z has
        # shape (batch, n_codes). The first Dense layer's input dim grows
        # from `hidden_dim` to `hidden_dim + n_codes`, so the HyperNet
        # target params (and the chunked adapter) grow accordingly.
        self.model_mask = AttentionMaskGenerator(
            hidden_dim=self.hidden_dim,
            score_hidden=int(score_hidden),
            z_dim=self.n_codes,
        )
        dummy_hidden = jnp.zeros(
            (1, self.seq_len, self.hidden_dim), dtype=jnp.float32)
        dummy_z = jnp.zeros((1, self.n_codes), dtype=jnp.float32)
        variables_mask = self.model_mask.init(
            random.PRNGKey(0), dummy_hidden, dummy_z)
        self.init_params_mask = variables_mask['params']

        # 3. ParameterAdapter + HyperNetwork (the PGPE-evolved component).
        self.adapter = AttentionParameterAdapter(
            self.init_params_mask, chunk_size=int(chunk_size))
        self.init_params_hypernet = self.adapter.init_hypernet(
            random.PRNGKey(11))

        self.num_params, format_params_hn_fn = get_params_format_fn(
            self.init_params_hypernet)
        self._format_params_hn_fn = jax.vmap(format_params_hn_fn)
        self._format_single_params_hn_fn = get_single_params_format_fn(
            self.init_params_hypernet)

        # Total target params produced by the HN (informational; not evolved).
        self.num_params_mask_target = int(sum(
            int(np.prod(p.shape))
            for p in jax.tree_util.tree_leaves(self.init_params_mask)))
        self._logger.info(
            'AttentionPolicy.num_params (PGPE search dim, HyperNet) = %d',
            self.num_params)
        self._logger.info(
            'AttentionMaskGenerator target params (HN-emitted) = %d',
            self.num_params_mask_target)
        self._logger.info(
            'PGPE compression ratio: %.2fx (target / search)',
            self.num_params_mask_target / max(self.num_params, 1))

        # 4. Classifier head (trained via backprop in Phase 7).
        self.model_classifier = Classifier(
            head_hidden=int(classifier_hidden), n_classes=2)
        dummy_pooled = jnp.zeros((1, self.hidden_dim), dtype=jnp.float32)
        self.init_params_classifier = self.model_classifier.init(
            random.PRNGKey(2), dummy_pooled)['params']

        self.num_params_classifier, _ = get_params_format_fn(
            self.init_params_classifier)
        self._logger.info(
            'Classifier head params (backprop) = %d',
            self.num_params_classifier)

        # 5. Q head (trained via backprop in Phase 7).
        self.model_q = QHead(
            head_hidden=int(q_hidden), n_codes=self.n_codes)
        self.init_params_q = self.model_q.init(
            random.PRNGKey(3), dummy_pooled)['params']
        self.num_params_q, _ = get_params_format_fn(self.init_params_q)
        self._logger.info(
            'QHead params (backprop) = %d', self.num_params_q)

    # ------------------------------------------------------------------
    # Forward primitives the trainer / smoke tests will use.
    # ------------------------------------------------------------------

    def hypernet_to_mask_params(self, params_hypernet):
        """HyperNet params -> AttentionMaskGenerator param pytree."""
        return self.adapter.generate_params(params_hypernet)

    def mask_logits(self,
                    params_mask,
                    hidden_states: jnp.ndarray,
                    z: jnp.ndarray) -> jnp.ndarray:
        """Per-token attention logits, conditioned on the latent code `z`.

        `z` is typically the discrete one-hot code `state.cat_codes`. The
        Phase 6 review found that omitting `z` here is the architectural gap
        that breaks the InfoGAN MI signal (identical masks across codes →
        Q-head cannot recover `z` → MI stuck at zero).
        """
        return self.model_mask.apply(
            {'params': params_mask}, hidden_states, z)

    def classify(self, params_classifier, pooled: jnp.ndarray
                 ) -> jnp.ndarray:
        return self.model_classifier.apply(
            {'params': params_classifier}, pooled)

    def predict_codes(self, params_q, pooled: jnp.ndarray) -> jnp.ndarray:
        return self.model_q.apply({'params': params_q}, pooled)

    # ------------------------------------------------------------------
    # PolicyNetwork interface (full integration in Phase 7).
    # ------------------------------------------------------------------

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState
                    ) -> Tuple[jnp.ndarray, PolicyState]:
        """Stub. Phase 7 wires this through trainer's ask/evaluate path."""
        raise NotImplementedError(
            'AttentionPolicy.get_actions is not wired yet — see Phase 7.')

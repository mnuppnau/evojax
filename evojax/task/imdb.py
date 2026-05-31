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

"""IMDb sentiment-classification task for HyperNet-InfoGAN hard attention.

Mirrors BloodMNIST in role: a data provider for the trainer's classifier
(D-step) batches. The trainer reads `.data` (int32 token IDs) and `.labels`,
and Phase 7 will extend it to also read `.attention_masks`. Tokenization happens
once at construction and is cached on disk so repeat runs start instantly.
"""

import os
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState


DEFAULT_TOKENIZER = 'distilbert-base-uncased'
DEFAULT_SEQ_LEN = 256
DEFAULT_CACHE_DIR = './data/imdb_cache'


@dataclass
class State(TaskState):
    obs: jnp.ndarray              # input_ids,        shape (batch, seq_len), int32
    attention_mask: jnp.ndarray   # encoder padding,  shape (batch, seq_len), int32
    latent: jnp.ndarray           # noise + cat,      shape (batch, latent_dim + n_codes)
    cat_codes: jnp.ndarray        # one-hot discrete, shape (batch, n_codes)
    labels: jnp.ndarray           # sentiment labels, shape (batch,), int32 (0=neg, 1=pos)
    batch_stats_gen: any = None
    batch_stats_disc: any = None
    batch_stats_q: any = None


def _load_imdb_split(
    test: bool,
    tokenizer_name: str,
    seq_len: int,
    cache_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load + tokenize one IMDb split. Caches a .npz on disk for fast reload."""
    split = 'test' if test else 'train'
    cache_name = (
        f'imdb_{split}_{tokenizer_name.replace("/", "_")}_{seq_len}.npz'
    )
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return (
            cached['input_ids'],
            cached['attention_masks'],
            cached['labels'],
        )

    # First-time path: download IMDb (if needed), tokenize, persist.
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset('imdb', split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    encoded = tokenizer(
        list(ds['text']),
        padding='max_length',
        truncation=True,
        max_length=seq_len,
        return_tensors='np',
    )
    input_ids = encoded['input_ids'].astype(np.int32)
    attention_masks = encoded['attention_mask'].astype(np.int32)
    labels = np.array(ds['label'], dtype=np.int32)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        cache_path,
        input_ids=input_ids,
        attention_masks=attention_masks,
        labels=labels,
    )
    return input_ids, attention_masks, labels


class IMDb(VectorizedTask):
    """IMDb binary sentiment task with HyperNet-style latent codes.

    Pre-tokenizes the IMDb split with `DistilBertTokenizerFast` at construction
    and caches the tokenized arrays under ``cache_dir``. The trainer pulls
    class-balanced batches from ``self.data`` / ``self.labels`` directly, the
    same pattern as BloodMNIST.

    Latent structure (mirrors BloodMNIST):
        z = [noise(latent_dim - n_cont), discrete(n_codes), continuous(n_cont)]

    Note that ``n_codes`` is the count of HyperNet discrete latent codes (default
    8 for methodological continuity with BloodMNIST), not the count of sentiment
    classes. Sentiment is binary; the discrete codes are intended to disentangle
    aspects within each polarity.
    """

    def __init__(
        self,
        batch_size: int = 64,
        seq_len: int = DEFAULT_SEQ_LEN,
        n_codes: int = 8,
        n_cont: int = 2,
        latent_dim: int = 62,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        test: bool = False,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        self.max_steps = 1
        self.obs_shape = tuple([seq_len, ])
        self.act_shape = tuple([n_codes, ])

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_classes = n_codes               # discrete latent codes (HyperNet)
        self.n_sentiment_classes = 2           # binary sentiment (pos / neg)
        self.n_cont = n_cont
        self.latent_dim = latent_dim
        self.dataset_name = 'imdb'
        self.tokenizer_name = tokenizer_name
        self.noise_dim = self.latent_dim - self.n_cont

        input_ids, attention_masks, labels = _load_imdb_split(
            test=test,
            tokenizer_name=tokenizer_name,
            seq_len=seq_len,
            cache_dir=cache_dir,
        )
        # Trainer-facing attributes (mirror BloodMNIST `data` / `labels` pattern).
        # `data` is int32 here, not float32 — Phase 7 will adapt the trainer's
        # `data_raw = np.array(..., dtype=np.float32)` cast to skip when ints.
        self.data = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

        input_ids_j = jnp.asarray(input_ids)
        attention_masks_j = jnp.asarray(attention_masks)
        labels_j = jnp.asarray(labels)

        def reset_fn(key, noise_key, cat_key, con_key):
            ix = random.choice(
                key=key,
                a=input_ids_j.shape[0],
                shape=(self.batch_size,),
                replace=False,
            )
            batch_input_ids = jnp.take(input_ids_j, indices=ix, axis=0)
            batch_attention_mask = jnp.take(
                attention_masks_j, indices=ix, axis=0)
            batch_labels = jnp.take(labels_j, indices=ix, axis=0)

            batch_noise = random.normal(
                noise_key, (self.batch_size, self.latent_dim))

            # Cyclic tile of discrete codes (shared-z protocol).
            reps = (self.batch_size + self.n_classes - 1) // self.n_classes
            c = jnp.tile(jnp.arange(self.n_classes), reps)[:self.batch_size]
            batch_cat_one_hot = jax.nn.one_hot(c, self.n_classes)

            batch_latent_concat = jnp.concatenate(
                [batch_noise, batch_cat_one_hot], axis=-1)

            return State(
                obs=batch_input_ids,
                attention_mask=batch_attention_mask,
                latent=batch_latent_concat,
                cat_codes=batch_cat_one_hot,
                labels=batch_labels,
            )

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        # Placeholder. Phase 7 owns the classifier-loss step in the trainer; the
        # task itself does not need to compute the classification objective.
        # Signature matches BloodMNIST's `step_fn(state, real_preds, action, q)`
        # so trainer wiring stays uniform until we override.
        def step_fn(state, real_preds, action, q):
            return (
                state,
                jnp.zeros(()),
                jnp.zeros(()),
                jnp.zeros(()),
                jnp.ones(()),
            )

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(
        self,
        key: jnp.ndarray,
        noise_key: jnp.ndarray,
        cat_key: jnp.ndarray,
        con_key: jnp.ndarray,
    ) -> State:
        return self._reset_fn(key, noise_key, cat_key, con_key)

    def step(
        self,
        state: TaskState,
        real_preds: jnp.ndarray,
        action: jnp.ndarray,
        q: jnp.ndarray,
    ) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, real_preds, action, q)

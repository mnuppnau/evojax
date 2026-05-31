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

"""Phase 3.5 — Architectural capacity check for IMDb hard attention.

Trains a small classifier head on top of frozen-DistilBERT mean-pooled features
using the full encoder attention mask. Then evaluates that classifier across
several rule-based mask families to test whether the binary mask space can in
principle express aspect-level disentanglement on IMDb sentiment.

Decision rule:
  - If aspect-biased masks (keyword_acting / plot / pacing) preserve sentiment
    accuracy clearly above a density-matched random baseline → the mask space
    has the representational capacity → search is the bottleneck → Phase 4.
  - If aspect-biased masks collapse near chance (or near density-matched
    random) → architecture needs richer structure before the controller is
    relevant.

Run:
  python examples/capacity_check_imdb.py --seq-len=256 --epochs=3 --batch-size=64
"""

import argparse
import os
import sys
import time
from typing import Callable, Dict, Iterable, List

# Ensure the workspace package is imported when running this script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax

from evojax.task.imdb import IMDb
from evojax.policy.attention_transformer import (
    FrozenDistilBertEncoder, Classifier, mean_pool_masked,
)


# ---------------------------------------------------------------------------
# Aspect keyword lists (curated; intentionally small to test sparse signal)
# ---------------------------------------------------------------------------

ASPECT_KEYWORDS: Dict[str, List[str]] = {
    'acting': [
        # nouns / referents
        'acting', 'actor', 'actors', 'actress', 'actresses', 'cast', 'casting',
        'performance', 'performances', 'role', 'roles', 'character',
        'characters', 'played', 'plays', 'playing', 'star', 'stars',
        'starring', 'portrayal', 'portrayed', 'portrays', 'lead', 'supporting',
        'leading', 'protagonist', 'antagonist', 'hero', 'heroine', 'villain',
        'extras', 'cameo', 'ensemble', 'crew', 'directorial', 'directed',
        'directs', 'director', 'directors', 'delivery', 'delivers',
        # sentiment-bearing acting descriptors
        'brilliant', 'terrible', 'wooden', 'natural', 'charismatic',
        'awkward', 'convincing', 'unconvincing', 'compelling', 'flat',
        'expressive', 'restrained', 'over-the-top', 'subtle', 'nuanced',
        'amateur', 'amateurish', 'professional', 'stunning', 'mediocre',
        'memorable', 'forgettable', 'underrated', 'overrated', 'phenomenal',
        'embarrassing', 'cringeworthy', 'authentic', 'believable',
        'unbelievable', 'powerful', 'weak', 'strong', 'incredible',
        'impressive', 'disappointing', 'miscast',
    ],
    'plot': [
        # nouns / referents
        'plot', 'plots', 'story', 'stories', 'narrative', 'storyline',
        'twist', 'twists', 'ending', 'beginning', 'climax', 'subplot',
        'subplots', 'screenplay', 'script', 'scripted', 'premise',
        'plotline', 'storytelling', 'arc', 'ending', 'opening', 'finale',
        'dialogue', 'dialogues', 'writing', 'written', 'screenwriter',
        'mystery', 'reveal', 'resolution', 'conclusion', 'setup',
        'exposition', 'backstory', 'flashback', 'theme', 'themes',
        # sentiment-bearing plot descriptors
        'predictable', 'unpredictable', 'surprising', 'original',
        'derivative', 'formulaic', 'cliched', 'cliche', 'fresh', 'tired',
        'intricate', 'simple', 'simplistic', 'confusing', 'incoherent',
        'gripping', 'engaging', 'absorbing', 'boring', 'tedious', 'silly',
        'ridiculous', 'plausible', 'implausible', 'contrived', 'realistic',
        'unrealistic', 'compelling', 'shocking', 'satisfying',
        'unsatisfying', 'underwhelming', 'thoughtful', 'thought-provoking',
        'pointless', 'meaningful', 'profound', 'shallow', 'clever',
    ],
    'pacing': [
        # nouns / referents
        'pace', 'pacing', 'paced', 'rhythm', 'tempo', 'momentum',
        'minutes', 'hours', 'runtime', 'long', 'short', 'lengthy',
        'overlong', 'brief', 'quick',
        # sentiment-bearing pacing descriptors
        'slow', 'fast', 'boring', 'dragged', 'drags', 'rushed', 'tedious',
        'engaging', 'sluggish', 'snappy', 'plodding', 'breezy', 'lethargic',
        'meandering', 'languid', 'leisurely', 'brisk', 'taut', 'tight',
        'flabby', 'bloated', 'padded', 'overlong', 'rambling', 'crisp',
        'efficient', 'drawn-out', 'unhurried', 'tightly', 'loosely',
        'monotonous', 'repetitive', 'choppy', 'jumpy', 'frenetic',
        'measured', 'frenzied', 'lulls', 'momentum', 'patience', 'patient',
        'impatient', 'hurried', 'glacial', 'breakneck', 'leisure',
    ],
}


def words_to_token_ids(words: Iterable[str], tokenizer) -> set:
    """Map English words to the set of subword token IDs they decompose into."""
    ids = set()
    for w in words:
        # Lowercase to match DistilBERT uncased tokenizer.
        encoded = tokenizer.encode(w.lower(), add_special_tokens=False)
        ids.update(int(tid) for tid in encoded)
    return ids


# ---------------------------------------------------------------------------
# Mask family functions
# Signature: (input_ids, encoder_attn_mask, key) -> mask of shape (B, L)
# ---------------------------------------------------------------------------

def family_all_tokens(input_ids, attn, key):
    return attn.astype(jnp.int32)


def make_random_family(p: float) -> Callable:
    def fn(input_ids, attn, key):
        sampled = (random.uniform(key, attn.shape) < p).astype(jnp.int32)
        return sampled * attn.astype(jnp.int32)
    return fn


def make_keyword_family(token_ids: set) -> Callable:
    keyword_arr = jnp.asarray(sorted(token_ids), dtype=jnp.int32)

    def fn(input_ids, attn, key):
        # (B, L, K) match table -> any over K
        match = jnp.any(
            input_ids[:, :, None] == keyword_arr[None, None, :], axis=-1
        ).astype(jnp.int32)
        return match * attn.astype(jnp.int32)
    return fn


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seq-len', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--eval-samples', type=int, default=5000,
                   help='Subset of test set used for mask-family evaluation.')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    print(f'[init] loading frozen DistilBERT encoder...')
    t0 = time.time()
    encoder = FrozenDistilBertEncoder()
    print(f'[init] encoder ready ({time.time() - t0:.1f}s)')

    classifier = Classifier(head_hidden=128, n_classes=2)

    print(f'[data] loading IMDb (seq_len={args.seq_len})...')
    t0 = time.time()
    train_task = IMDb(batch_size=1, seq_len=args.seq_len)
    test_task = IMDb(batch_size=1, seq_len=args.seq_len, test=True)
    print(f'[data] train={len(train_task.labels)}, test={len(test_task.labels)} '
          f'({time.time() - t0:.1f}s)')

    train_ids = train_task.data
    train_attn = train_task.attention_masks
    train_lab = train_task.labels

    test_ids = test_task.data
    test_attn = test_task.attention_masks
    test_lab = test_task.labels

    # ----- 1. Init classifier -----
    rng = random.PRNGKey(args.seed)
    rng, init_key = random.split(rng)
    init_params = classifier.init(
        init_key, jnp.zeros((1, encoder.hidden_dim)))['params']

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(init_params)

    @jax.jit
    def train_step(params, opt_state, hidden, mask, labels):
        def loss_fn(p):
            pooled = mean_pool_masked(hidden, mask)
            logits = classifier.apply({'params': p}, pooled)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, labels).mean()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, hidden, mask, labels):
        pooled = mean_pool_masked(hidden, mask)
        logits = classifier.apply({'params': params}, pooled)
        pred = jnp.argmax(logits, axis=-1)
        return (pred == labels).astype(jnp.float32).mean()

    # ----- 2. Train reference classifier -----
    n_train = len(train_lab)
    bs = args.batch_size

    print(f'[train] {args.epochs} epochs over {n_train} samples, batch={bs}')
    params = init_params
    for epoch in range(args.epochs):
        rng, sk = random.split(rng)
        perm = np.asarray(random.permutation(sk, n_train))
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        for i in range(0, n_train - bs + 1, bs):
            ix = perm[i:i + bs]
            ids = jnp.asarray(train_ids[ix])
            mask = jnp.asarray(train_attn[ix])
            labs = jnp.asarray(train_lab[ix])
            hidden = encoder.forward(ids, mask)
            params, opt_state, loss = train_step(
                params, opt_state, hidden, mask, labs)
            epoch_loss += float(loss)
            n_batches += 1
        print(f'[train] epoch {epoch + 1}/{args.epochs}: '
              f'loss={epoch_loss / n_batches:.4f} '
              f'({time.time() - t0:.1f}s, {n_batches} batches)')

    # ----- 3. Build keyword-aspect mask families -----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    keyword_id_sets = {
        name: words_to_token_ids(words, tok)
        for name, words in ASPECT_KEYWORDS.items()
    }
    for name, ids in keyword_id_sets.items():
        sample_words = [
            tok.decode([tid]) for tid in sorted(ids)[:8]
        ]
        print(f'[masks] aspect={name}: {len(ids)} token ids '
              f'(first 8 decoded: {sample_words})')

    families: Dict[str, Callable] = {
        'all_tokens': family_all_tokens,
        'random_p005': make_random_family(0.005),
        'random_p010': make_random_family(0.010),
        'random_p050': make_random_family(0.050),
        'random_p200': make_random_family(0.200),
        'keyword_acting': make_keyword_family(keyword_id_sets['acting']),
        'keyword_plot': make_keyword_family(keyword_id_sets['plot']),
        'keyword_pacing': make_keyword_family(keyword_id_sets['pacing']),
    }

    # ----- 4. Evaluate each family on the test subset -----
    n_eval = min(args.eval_samples, len(test_lab))
    n_eval = (n_eval // bs) * bs
    print(f'[eval] evaluating {len(families)} mask families on '
          f'{n_eval} test samples...')

    results: Dict[str, Dict[str, float]] = {}
    for fam_name, fam_fn in families.items():
        accs, dens = [], []
        rng, fam_key = random.split(rng)
        for i in range(0, n_eval, bs):
            ids = jnp.asarray(test_ids[i:i + bs])
            attn = jnp.asarray(test_attn[i:i + bs])
            labs = jnp.asarray(test_lab[i:i + bs])
            fam_key, sk = random.split(fam_key)
            mask = fam_fn(ids, attn, sk)
            hidden = encoder.forward(ids, attn)
            acc = eval_step(params, hidden, mask, labs)

            # Track density relative to non-pad positions for fair reporting.
            n_attended = mask.sum(axis=-1).astype(jnp.float32)
            n_valid = attn.sum(axis=-1).astype(jnp.float32) + 1e-6
            density = (n_attended / n_valid).mean()

            accs.append(float(acc))
            dens.append(float(density))
        results[fam_name] = {
            'acc': float(np.mean(accs)),
            'density': float(np.mean(dens)),
        }

    # ----- 5. Summary + decision -----
    print('\n' + '=' * 70)
    print(f'{"PHASE 3.5 CAPACITY CHECK SUMMARY":^70}')
    print('=' * 70)
    print(f'{"family":<22}{"density":>12}{"accuracy":>12}')
    print('-' * 70)
    for fam_name in families:
        r = results[fam_name]
        print(f'{fam_name:<22}{r["density"]:>12.4f}{r["acc"]:>12.4f}')

    aspect_keys = ('keyword_acting', 'keyword_plot', 'keyword_pacing')
    upper = results['all_tokens']['acc']

    # Per-aspect interpolated-random baseline + accuracy preservation ratio.
    rand_keys = sorted(
        (k for k in results if k.startswith('random_')),
        key=lambda k: results[k]['density'])
    rand_dens = np.array([results[k]['density'] for k in rand_keys])
    rand_accs = np.array([results[k]['acc'] for k in rand_keys])

    print('-' * 70)
    print(f'upper bound (all_tokens):   acc={upper:.4f}')
    print(f'{"aspect":<22}{"density":>10}{"acc":>10}'
          f'{"interp_rand":>14}{"margin":>10}{"preservation":>14}')
    margins, preservations = [], []
    for k in aspect_keys:
        d, a = results[k]['density'], results[k]['acc']
        # Linear interpolation of random accuracy at this density.
        interp = float(np.interp(d, rand_dens, rand_accs))
        margin = a - interp
        preservation = a / upper if upper > 0 else 0.0
        margins.append(margin)
        preservations.append(preservation)
        print(f'{k:<22}{d:>10.4f}{a:>10.4f}{interp:>14.4f}'
              f'{margin:>+10.4f}{preservation:>14.2%}')

    print('\nDecision:')
    pres_min = float(min(preservations))
    margin_min = float(min(margins))
    if pres_min >= 0.90 and margin_min > 0.0:
        verdict = 'PASS'
        message = ('Aspect-biased masks preserve >=90% of upper-bound accuracy '
                   'and each beats density-interpolated random. The binary '
                   'mask space carries enough sentiment-relevant signal for '
                   'aspect-aware selection. Architecture has capacity; '
                   'search is the bottleneck. Proceed to Phase 4.')
    elif pres_min >= 0.75:
        verdict = 'AMBIGUOUS'
        message = ('Aspect masks preserve 75-90% of upper-bound accuracy. '
                   'Mask space carries signal but at the edge of usefulness. '
                   'Consider expanding keyword lists, training the classifier '
                   'with mask-augmentation, or richer mask architecture '
                   '(multi-head / hierarchical) before fully committing to '
                   'Phase 4.')
    else:
        verdict = 'FAIL'
        message = ('Aspect masks lose >25% of upper-bound accuracy. The '
                   'binary per-token mask space cannot cleanly express '
                   'sentiment-relevant aspect signal at this scale. '
                   'Architecture needs richer structure (multi-head masks, '
                   'hierarchical selection, or continuous-with-discrete-'
                   'routing) before attention-side controllers are worth '
                   'investing in.')
    print(f'  {verdict}: {message}')
    print('=' * 70)


if __name__ == '__main__':
    main()

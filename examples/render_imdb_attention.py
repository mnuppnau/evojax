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

"""CLI for rendering per-code attention masks from an IMDb checkpoint.

Loads a saved `TrainerIMDb` pickle, reconstructs the HyperNet-driven
`AttentionMaskGenerator`, and writes an HTML grid of N held-out reviews,
each shown K times (one per discrete code) with attended tokens highlighted.

Core rendering logic lives in `evojax/render_imdb.py` so the trainer's
`--render-interval` auto-snapshot path can reuse it without duplication.

Usage:
  python examples/render_imdb_attention.py \\
      --checkpoint=./log/imdb/checkpoints/iter-000200.pkl \\
      --n-reviews=8 \\
      --out=./log/imdb/render_iter-000200.html
"""

import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
for _i, _arg in enumerate(sys.argv):
    if _arg == '--gpu-id' and _i + 1 < len(sys.argv):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])
        break
    if _arg.startswith('--gpu-id='):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', _arg.split('=', 1)[1])
        break

import argparse
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np                                       # noqa: E402
import jax.numpy as jnp                                  # noqa: E402

from evojax.task.imdb import IMDb                        # noqa: E402
from evojax.policy.attention_transformer import (        # noqa: E402
    AttentionPolicy,
)
from evojax.render_imdb import (                          # noqa: E402
    build_review_payload,
    render_html_str,
    render_ansi_str,
    render_checkpoint_to_html,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--n-reviews', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--format', type=str, default='html', choices=['html', 'ansi'])
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--encoder-name', type=str, default='distilbert-base-uncased')
    p.add_argument('--seq-len', type=int, default=256)
    p.add_argument('--n-discrete-codes', type=int, default=8)
    p.add_argument('--score-hidden', type=int, default=64)
    p.add_argument('--classifier-hidden', type=int, default=128)
    p.add_argument('--q-hidden', type=int, default=128)
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--max-tokens-rendered', type=int, default=200)
    p.add_argument('--gpu-id', type=str, default=None,
                   help='(env-var sniffed pre-import; kept for log parity)')
    cfg, _ = p.parse_known_args()
    return cfg


def main():
    cfg = parse_args()
    with open(cfg.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    center_flat = ckpt['center']
    print(f'[ckpt] loaded {cfg.checkpoint}  iter={ckpt["iter"]}  '
          f'center.shape={center_flat.shape}', file=sys.stderr)

    policy = AttentionPolicy(
        seq_len=cfg.seq_len,
        n_codes=cfg.n_discrete_codes,
        score_hidden=cfg.score_hidden,
        classifier_hidden=cfg.classifier_hidden,
        q_hidden=cfg.q_hidden,
        chunk_size=cfg.chunk_size,
        encoder_name=cfg.encoder_name,
    )
    assert center_flat.size == policy.num_params, (
        f'Checkpoint param count {center_flat.size} does not match policy '
        f'num_params {policy.num_params}; check --score-hidden / --chunk-size '
        f'/ --n-discrete-codes flags match the training run.'
    )

    test_task = IMDb(
        batch_size=1, seq_len=cfg.seq_len, n_codes=cfg.n_discrete_codes,
        tokenizer_name=cfg.encoder_name, test=True,
    )
    rng = np.random.default_rng(cfg.seed)
    rev_ix = rng.choice(len(test_task.labels), size=cfg.n_reviews, replace=False)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)

    if cfg.format == 'html':
        out_path = cfg.out or (
            './render_' + os.path.basename(cfg.checkpoint).replace('.pkl', '.html')
        )
        summary = render_checkpoint_to_html(
            policy=policy,
            center_flat=center_flat,
            tokenizer=tokenizer,
            test_input_ids=test_task.data,
            test_attn_masks=test_task.attention_masks,
            test_labels=test_task.labels,
            indices=rev_ix,
            n_codes=cfg.n_discrete_codes,
            out_path=out_path,
            max_tokens=cfg.max_tokens_rendered,
            title_suffix=f'iter {ckpt["iter"]}',
        )
        print(f'[out] wrote HTML render to {summary["out_path"]}', file=sys.stderr)
    else:
        from jax.numpy import asarray as _asarray  # quiet noqa
        hn_params = policy._format_single_params_hn_fn(jnp.asarray(center_flat))
        mask_params = policy.hypernet_to_mask_params(hn_params)
        payload = build_review_payload(
            policy, mask_params, policy.encoder.forward, tokenizer,
            test_task.data[rev_ix], test_task.attention_masks[rev_ix],
            test_task.labels[rev_ix], cfg.n_discrete_codes,
        )
        body = render_ansi_str(payload, cfg.n_discrete_codes, cfg.max_tokens_rendered)
        if cfg.out:
            with open(cfg.out, 'w') as f:
                f.write(body)
            print(f'[out] wrote ANSI render to {cfg.out}', file=sys.stderr)
        else:
            print(body)
        summary = None

    if summary is not None:
        print(file=sys.stderr)
        print(f'Density per code (mean across {len(rev_ix)} reviews):', file=sys.stderr)
        for k, d in enumerate(summary['density_per_code']):
            print(f'  code {k}: {d:.4f}', file=sys.stderr)
        print(f'Mean attended-token count per code:', file=sys.stderr)
        for k, c in enumerate(summary['mean_tokens_per_code']):
            print(f'  code {k}: {c:.1f}', file=sys.stderr)


if __name__ == '__main__':
    main()

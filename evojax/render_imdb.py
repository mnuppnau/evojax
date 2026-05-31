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

"""Shared rendering primitives for IMDb hard-attention masks.

Both `examples/render_imdb_attention.py` (one-shot CLI against a saved
checkpoint) and `evojax/trainer_imdb.py` (auto-render every
`--render-interval` iters during the long Phase 8 run) use these. The
rendering is deterministic — `sigmoid(logits) > 0.5` rather than a
Bernoulli sample — so the snapshot reflects the policy's high-probability
attention pattern at that checkpoint, not one Monte Carlo draw.
"""

import html as html_lib
from typing import Dict, List

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Color palette (one background color per discrete code).
# ---------------------------------------------------------------------------

PALETTE_HSL = [
    'hsl(  0, 80%, 85%)',  # code 0  — soft red
    'hsl( 30, 80%, 85%)',  # code 1  — peach
    'hsl( 60, 80%, 80%)',  # code 2  — pale yellow
    'hsl(120, 60%, 85%)',  # code 3  — mint
    'hsl(180, 60%, 85%)',  # code 4  — cyan
    'hsl(220, 70%, 88%)',  # code 5  — sky blue
    'hsl(270, 60%, 88%)',  # code 6  — lavender
    'hsl(320, 60%, 88%)',  # code 7  — pink
]

PALETTE_ANSI_BG = [217, 223, 229, 194, 195, 189, 225, 218]


# ---------------------------------------------------------------------------
# Token helpers (DistilBERT WordPiece-aware)
# ---------------------------------------------------------------------------

def decode_token_string(tokenizer, token_id: int) -> str:
    return tokenizer.convert_ids_to_tokens(int(token_id))


def is_continuation(token_str: str) -> bool:
    return token_str.startswith('##')


def token_display(token_str: str) -> str:
    return token_str[2:] if is_continuation(token_str) else token_str


# ---------------------------------------------------------------------------
# Compute per-code masks for a batch of reviews using a saved HN center
# ---------------------------------------------------------------------------

def build_review_payload(
    policy,
    mask_params,
    encoder_forward,
    tokenizer,
    input_ids: np.ndarray,
    attn_masks: np.ndarray,
    labels: np.ndarray,
    n_codes: int,
) -> List[Dict]:
    """Run the policy K times per review (one per discrete code) and return
    the per-review render data (`tokens`, `attn_mask`, `masks_per_code`,
    `label`). Deterministic threshold mask — no Bernoulli sampling.
    """
    K = n_codes
    eye_codes = jnp.eye(K, dtype=jnp.float32)
    payload = []
    for i in range(input_ids.shape[0]):
        ids = jnp.asarray(input_ids[i:i + 1])
        attn = jnp.asarray(attn_masks[i:i + 1])
        label = int(labels[i])
        hidden = encoder_forward(ids, attn)                              # (1, L, H)
        hidden_k = jnp.broadcast_to(hidden, (K,) + hidden.shape[1:])     # (K, L, H)
        logits = policy.mask_logits(mask_params, hidden_k, eye_codes)    # (K, L)
        masks = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
        masks = masks * attn[0:1].astype(jnp.int32)
        token_ids = np.asarray(ids[0])
        tokens = [decode_token_string(tokenizer, tid) for tid in token_ids]
        payload.append({
            'tokens': tokens,
            'attn_mask': np.asarray(attn[0]),
            'masks_per_code': np.asarray(masks),
            'label': label,
        })
    return payload


# ---------------------------------------------------------------------------
# HTML / ANSI emitters
# ---------------------------------------------------------------------------

def render_html_str(reviews: List[Dict],
                    n_codes: int,
                    max_tokens: int = 200,
                    title_suffix: str = '') -> str:
    """Render reviews as a self-contained HTML document.

    `title_suffix` is appended to the page <h2> — useful for marking the
    iteration index when called from the trainer.
    """
    legend_swatches = ''.join(
        f'<span class="codeSwatch" style="background:{PALETTE_HSL[k % len(PALETTE_HSL)]}">'
        f'code {k}</span> ' for k in range(n_codes)
    )

    review_blocks = []
    for r_idx, rev in enumerate(reviews):
        tokens = rev['tokens']
        attn = rev['attn_mask']
        masks = rev['masks_per_code']
        valid_len = int(attn.sum())
        tokens = tokens[:min(valid_len, max_tokens)]
        masks = masks[:, :len(tokens)]

        per_code_rows = []
        for k in range(n_codes):
            color = PALETTE_HSL[k % len(PALETTE_HSL)]
            row_parts = []
            for t_idx, tok in enumerate(tokens):
                disp = token_display(tok)
                disp_esc = html_lib.escape(disp)
                space = '' if is_continuation(tok) else ' '
                if masks[k, t_idx]:
                    span = (
                        f'<span style="background:{color};border-radius:3px;'
                        f'padding:1px 2px;">{disp_esc}</span>'
                    )
                else:
                    span = disp_esc
                row_parts.append(f'{space}{span}')
            density = float(masks[k].sum()) / max(len(tokens), 1)
            per_code_rows.append(
                f'<tr><td class="codeLabel" '
                f'style="background:{color};">'
                f'code&nbsp;{k}<br/><small>density={density:.2f}</small></td>'
                f'<td class="codeText">{"".join(row_parts)}</td></tr>'
            )

        header = (
            f'<h3>Review #{r_idx} '
            f'<small>(label={int(rev["label"])}, len={valid_len} tokens)</small>'
            f'</h3>'
        )
        block = (
            f'{header}'
            f'<table class="codeTable">'
            f'<tbody>{"".join(per_code_rows)}</tbody>'
            f'</table>'
        )
        review_blocks.append(block)

    css = """
    body{font-family:-apple-system, system-ui, sans-serif; max-width:1100px;
         margin:24px auto; padding:0 16px; color:#222;}
    h2{margin-top:0;}
    .legend{margin:8px 0 20px 0; font-size:13px;}
    .codeSwatch{display:inline-block; padding:2px 8px; border-radius:4px;
                margin-right:4px; font-weight:600; font-size:12px;}
    .codeTable{border-collapse:collapse; width:100%; margin-bottom:28px;
               font-size:14px; line-height:1.55;}
    .codeLabel{width:80px; text-align:center; vertical-align:top;
               padding:6px 4px; border-radius:6px 0 0 6px; font-weight:600;
               font-size:12px;}
    .codeText{padding:8px 12px; vertical-align:top;}
    small{color:#666; font-size:11px;}
    h3{margin-top:36px; margin-bottom:8px; font-size:15px; color:#333;}
    """
    title = 'IMDb hard-attention — per-code mask render'
    if title_suffix:
        title = title + ' — ' + title_suffix
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{html_lib.escape(title)}</title>'
        f'<style>{css}</style></head><body>'
        f'<h2>{html_lib.escape(title)}</h2>'
        f'<div class="legend">Color legend: {legend_swatches}</div>'
        f'{"".join(review_blocks)}'
        '</body></html>'
    )


def render_ansi_str(reviews: List[Dict],
                    n_codes: int,
                    max_tokens: int = 200) -> str:
    chunks = []
    for r_idx, rev in enumerate(reviews):
        tokens = rev['tokens']
        attn = rev['attn_mask']
        masks = rev['masks_per_code']
        valid_len = int(attn.sum())
        tokens = tokens[:min(valid_len, max_tokens)]
        masks = masks[:, :len(tokens)]

        chunks.append(
            f'\n=== Review #{r_idx}  label={int(rev["label"])}  '
            f'len={valid_len} ==='
        )
        for k in range(n_codes):
            color = PALETTE_ANSI_BG[k % len(PALETTE_ANSI_BG)]
            row = []
            for t_idx, tok in enumerate(tokens):
                disp = token_display(tok)
                space = '' if is_continuation(tok) else ' '
                if masks[k, t_idx]:
                    row.append(f'{space}\033[48;5;{color}m\033[30m{disp}\033[0m')
                else:
                    row.append(f'{space}{disp}')
            density = float(masks[k].sum()) / max(len(tokens), 1)
            chunks.append(f'[code {k}  d={density:.2f}] ' + ''.join(row))
    return '\n'.join(chunks)


# ---------------------------------------------------------------------------
# End-to-end convenience: take a flat center vector → write HTML to disk
# ---------------------------------------------------------------------------

def render_checkpoint_to_html(
    policy,
    center_flat: np.ndarray,
    tokenizer,
    test_input_ids: np.ndarray,
    test_attn_masks: np.ndarray,
    test_labels: np.ndarray,
    indices: np.ndarray,
    n_codes: int,
    out_path: str,
    max_tokens: int = 200,
    title_suffix: str = '',
) -> Dict:
    """Write an HTML render of `len(indices)` reviews for the given center.

    Returns a small summary dict ({'density_per_code': List[float],
    'mean_tokens_per_code': List[float], 'out_path': str}) so callers can
    log a one-liner without re-parsing the HTML.
    """
    hn_params = policy._format_single_params_hn_fn(jnp.asarray(center_flat))
    mask_params = policy.hypernet_to_mask_params(hn_params)
    payload = build_review_payload(
        policy, mask_params, policy.encoder.forward, tokenizer,
        test_input_ids[indices], test_attn_masks[indices],
        test_labels[indices], n_codes,
    )
    html_text = render_html_str(payload, n_codes, max_tokens, title_suffix)
    with open(out_path, 'w') as f:
        f.write(html_text)

    all_masks = np.stack([r['masks_per_code'] for r in payload])
    valid_lens = np.array([int(r['attn_mask'].sum()) for r in payload])
    densities = np.array([
        all_masks[i].sum(axis=1) / max(valid_lens[i], 1)
        for i in range(len(payload))
    ])
    return {
        'out_path': out_path,
        'density_per_code': [float(densities[:, k].mean()) for k in range(n_codes)],
        'mean_tokens_per_code': [
            float(all_masks[:, k].sum(axis=1).mean()) for k in range(n_codes)
        ],
    }

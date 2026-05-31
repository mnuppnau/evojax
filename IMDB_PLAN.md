# IMDb Hard-Attention Transformer Plan

Branch: `claude/evolve-infogan-hypernetworks-s3r6d-67132f2-imdb-baseline`
Created: 2026-04-19
Parent: `claude/evolve-infogan-hypernetworks-s3r6d-67132f2-blood-baseline` at commit `cf37ef7`

This document tracks the setup tasks for porting the HyperNet-InfoGAN architecture from BloodMNIST to IMDb hard-attention. The near-term goal is to mirror the BloodMNIST workflow: establish a non-CA baseline first, then layer CA integration on top once the architecture is validated.

## Status Summary (last updated 2026-05-16)

| Phase | Status | When | Notes |
|---|---|---|---|
| 1 — Environment and dependencies | ✅ DONE | 2026-04-19 | `transformers`/`datasets`/`tokenizers` pinned `<5.0` (Flax models dropped in v5) |
| 2 — Dataset task (`evojax/task/imdb.py`) | ✅ DONE | 2026-04-22 | 25k train / 25k test, perfectly 12500/12500 balanced, on-disk `.npz` cache |
| 3 — Policy (`evojax/policy/attention_transformer.py`) | ✅ DONE | 2026-04-25 | FrozenDistilBERT + HyperNet (29,984 search params) + Classifier (98,690) + Q-head (99,464) |
| 3.5 — Architectural capacity check | ✅ DONE — **PASS** | 2026-04-25 | Aspect-biased masks preserve 92–98% of upper-bound accuracy; architecture is not the bottleneck |
| 4 — Attention fitness metrics (`evojax/task/attention_metrics.py`) | ✅ DONE | 2026-05-12 | 6 JAX-jittable metrics + `compute_all` aggregator |
| 5 — Solver fork (`evojax/algo/pgpe_ca_text.py`) | ✅ DONE | 2026-05-13 | Class `PGPE_CA_Text`; tell() takes attention metrics; review fixes applied 2026-05-16 |
| 6 — Training entry point (`examples/train_imdb.py`) | ✅ DONE | 2026-05-16 | CLI parsing + `build_components()` constructs policy / tasks / solver; verified end-to-end on small config; review fixes applied (GPU env-var ordering + latent log) |
| **3-revisit — `AttentionMaskGenerator` z-conditioning** | ✅ DONE | 2026-05-16 | MaskGen now takes `(hidden_states, z)`; `score_h1/kernel` is `(hidden_dim + z_dim, score_hidden)`; same z → identical logits, different z → non-zero diff; +512 target params, +16 PGPE params (1.64x → 1.66x compression — negligible) |
| 7 — Trainer (`evojax/trainer_imdb.py`) | ✅ DONE | 2026-05-16 | Fresh ~360-line `TrainerIMDb` (NOT a fork of `trainer.py`); JIT'd vmap eval over pop, classifier + Q-head backprop, balanced sampling, checkpoint save; `disc_logits` shape contract corrected to `(pop, batch, n_codes)`; smoke test 3-iter passes |
| 8 — IMDB-B00 baseline run | ✅ DONE | 2026-05-23 | 290000 iters, ~17h on single GPU, stable JIT 0.21s/iter; late-window f_adv=0.588, f_mi=−1.873; **documented failure mode: density-tier shortcut** (8 codes → 3 stable density tiers). See `IMDB_ABLATION_TRIAL_LOG.txt`. |
| 9 — Baseline analysis | 🔄 IN PROGRESS | 2026-05-31 | Trajectory & density-tier diagnosis done; 17 HTML renders archived at 20k strides plus iter-290000; qualitative aspect-vs-shortcut spot-check on `render_iter-290000.html` outstanding; ablation ladder defined in `IMDB_ABLATION_TRIAL_LOG.txt` |

**Artifacts created:**
- `evojax/task/imdb.py` (Phase 2)
- `evojax/policy/attention_transformer.py` (Phase 3, z-conditioning added 2026-05-16)
- `evojax/task/attention_metrics.py` (Phase 4)
- `evojax/algo/pgpe_ca_text.py` (Phase 5)
- `examples/train_imdb.py` (Phase 6)
- `evojax/trainer_imdb.py` (Phase 7)
- `evojax/render_imdb.py` + `examples/render_imdb_attention.py` (Phase 8 inspection tooling)
- `examples/capacity_check_imdb.py` (Phase 3.5)
- `setup.py` updated with `nlp` extras section (Phase 1)
- `IMDB_ABLATION_TRIAL_LOG.txt` (Phase 8/9 — first entry is IMDB-B00)

**Artifacts produced by the IMDB-B00 run** (`log/imdb/`):
- 58 PGPE checkpoints (`checkpoints/iter-005000.pkl` … `iter-290000.pkl`) plus final at iter 290000
- 17 HTML attention renders (`checkpoints/render_iter-020000.html` … `render_iter-290000.html`, every 20k iters)
- Stdout/structured log at `IMDb.txt` and the tee'd `imdb_b00.log`

**Next milestone:** Phase 9 cleanup + the first ablation. Specifically:
1. Spot-check `render_iter-290000.html` (and a 20k baseline `render_iter-020000.html`) to verify whether the density tiers correlate with token TYPES (content vs stopwords vs punctuation) or only with token COUNTS.
2. Run **IMDB-A04-hybrid** — directly tests whether the BloodMNIST hybrid winner (static `w_mi` + dynamic remainder) transfers. Highest-leverage single experiment per `IMDB_ABLATION_TRIAL_LOG.txt`.
3. Once IMDB-A04-hybrid is characterized, build the **logit-prior CA intervention hook** for IMDB-A02 — the architectural follow-through that Phase 5's design notes promised.

## Design Decisions (Resolved)

### Core architectural shift
- Generator no longer produces pixels. It produces a **binary attention mask** over tokens of an input review.
- Downstream task is sentiment classification. The "Discriminator" analog is a supervised classifier, not a real/fake discriminator.
- Intervention surface for any future CA integration is **logit priors on the attention sampler**, not gradient blending on HyperNet weights. This is the direct architectural lesson from the BloodMNIST ablation study.

### Latent structure (preserved from BloodMNIST)
- `z = [noise(62), discrete(K), continuous(2)]`
- `K` is an open question — see Open Questions below.

### Encoder choice
- **Pretrained DistilBERT (6 layers), frozen.** Isolates the research variable (the attention mechanism) from encoder representation quality. Saves PGPE rollout compute. Keeps the paper's contribution legible.

### Carried-over findings from BloodMNIST
- Class-balanced sampling is standard from day one (50/50 positive/negative).
- All fitness components pass through `rank_normalize`, never `standardize`.
- D/G asymmetry: classifier trains via backprop, HyperNet evolves via PGPE.
- Hybrid fitness weighting (static `w_mi` + dynamic `w_adv`/`w_intra`/etc.) is the default once dynamic weights are enabled.
- Baseline runs use static fitness weights with CA disabled.

### Out-of-scope anti-goals (lessons applied)
- No gradient blend at any dose in the first CA integration. Logit priors only.
- No imbalanced sampling at any phase.
- No from-scratch encoder for the baseline (reserved for a later sensitivity study if needed).
- No dynamic weighting without hybrid MI protection.

## Resolved Decisions (closed during Phases 1–5)

These were originally listed as open questions to settle before Phase 3. All have now been resolved by the actual implementation:

1. **Discrete code count `K = 8`.** Matches BloodMNIST for methodological continuity. Configurable via the `n_codes` arg on `IMDb` (task) and `AttentionPolicy` (policy).
2. **Mask sampling: hard Bernoulli, no straight-through.** Confirmed in Phase 3 design. The HyperNet is evolved by PGPE (no gradient needed), and the classifier and Q-head receive ordinary backprop with the mask treated as input data, so straight-through estimators are not required. The `bernoulli_hard_mask` helper additionally forces position 0 ([CLS]) when sampling produces an all-zero mask, to keep the mean-pool non-degenerate.
3. **HyperNet target = per-token MLP** (`AttentionMaskGenerator`, hidden=64, 2 layers, score per token). Target params: 49,281. PGPE search dim: 29,984. Compression ratio 1.64× (modest — flagged for Phase 8 measurement; could fall back to direct evolution if HyperNet inference dominates rollout time).
4. **Encoder = full frozen DistilBERT** (6 layers, 768-dim hidden, 12 attention heads, vocab 30522). `FrozenDistilBertEncoder` wraps `FlaxDistilBertModel` and applies `jax.lax.stop_gradient` so the encoder is architecturally unreachable by backprop.
5. **Pooling = `mean_pool_masked`** over attended tokens with length normalization. CLS fallback in `bernoulli_hard_mask` guards against an empty mask collapsing the pool to zero.

## Task List

### Phase 1 — Environment and dependencies (complete 2026-04-19)
- [x] Add `transformers`, `datasets`, `tokenizers` to `setup.py` `nlp` extras
  - Pinned: `transformers>=4.40,<5.0`, `datasets>=2.16,<4.0`, `tokenizers>=0.15,<1.0`
  - Upper bound on transformers is load-bearing: 5.x removed the Flax/JAX model classes
- [x] Verify JAX/Flax compatibility: `FlaxDistilBertModel` imports and runs forward passes against jax 0.4.31 / flax 0.8.4
- [x] Cache pretrained DistilBERT weights locally (`distilbert-base-uncased` resolved into `~/.cache/huggingface/`)
- [x] Confirm tokenizer and model versions installed: transformers 4.57.6, datasets 3.6.0, tokenizers 0.22.2

#### Verified backbone specs (for Phase 3 reference)
- Model: `distilbert-base-uncased` (FlaxDistilBertModel)
- Vocab size: 30522
- Hidden size: 768
- Layers: 6
- Attention heads: 12
- Tokenizer: `DistilBertTokenizerFast`
- Forward-pass output: `last_hidden_state` shape `(batch, seq_len, 768)`, `float32`

#### Note for setup.py-driven installs
- `pip install -e .[nlp]` will pull the pinned NLP stack
- The transformers package emits a deprecation warning about Flax classes; that is expected and is exactly why the pin caps below 5.x

### Phase 2 — Dataset task (complete 2026-04-22)
- [x] Created `evojax/task/imdb.py` implementing `VectorizedTask`
  - [x] Loads IMDb via `datasets.load_dataset('imdb')` (25k train / 25k test, perfectly balanced 12500/12500)
  - [x] Pre-tokenizes at construction with `DistilBertTokenizerFast`; default `seq_len=256`
  - [x] Disk cache at `./data/imdb_cache/imdb_{split}_{tokenizer}_{seq_len}.npz` (covered by `/data/` in `.gitignore`)
  - [x] `reset(key, noise_key, cat_key, con_key)` returns a `State` with `obs` (input_ids, int32), `attention_mask`, `latent`, `cat_codes`, `labels`
  - [x] Class-balanced sampling handled by the existing trainer (builds `class_indices` from `self.labels`; binary labels → 50/50)
  - [ ] (Deferred) POS tag pre-compute — belongs with Phase-5 CA re-instantiation, not the baseline

#### Verified behavior
- First-time init: ≈11s (download + tokenize + cache). Repeat init: ≈3s (cache hit).
- Tokenization sanity: decode roundtrip produces a readable `[CLS] i rented i am curious...` sample.
- `State.obs` shape `(devices, batch, seq_len)` int32; `State.latent` is `(devices, batch, 70)` for `n_codes=8, n_cont=2` (i.e., noise(62) concatenated with one-hot cat(8)).
- `step(state, real_preds, action, q)` is stubbed (returns zeros) — Phase 7 owns the classifier-loss path.

#### Trainer coupling notes (for Phase 7)
- `self.data` is `int32` token_ids; the trainer currently forces `np.array(..., dtype=np.float32)` + division by 255 at `trainer.py:1157`. Phase 7 must branch on `np.issubdtype(data.dtype, np.integer)` to skip both the float cast and the `/255` normalization, and additionally pull `self.attention_masks` alongside `self.data` in the D-step `sample_batch`.
- Default `n_codes=8` (matches BloodMNIST); configurable per entry-point CLI.
- Discrete-code count is NOT the sentiment class count; sentiment is binary.

### Phase 3 — Policy: encoder + HyperNet + attention + heads (complete 2026-04-25)
- [x] Created `evojax/policy/attention_transformer.py`
  - [x] `FrozenDistilBertEncoder` — wraps `FlaxDistilBertModel`, params held inside the wrapper, output passes through `jax.lax.stop_gradient` so the encoder is unreachable by backprop by construction
  - [x] `HyperNetwork` — chunk-embed + 2-layer MLP, mirrors `convnet.py` HN; PGPE-evolved
  - [x] `AttentionMaskGenerator` — per-token 2-layer MLP scoring; weights are produced by the HyperNet via `AttentionParameterAdapter`
  - [x] `Classifier` — 2-layer MLP head over mean-pooled masked features, returns sentiment logits
  - [x] `QHead` — 2-layer MLP head, returns logits over discrete codes for MI estimation
  - [x] `bernoulli_hard_mask` and `mean_pool_masked` helpers (zero-pad-aware)
- [x] Design choice: hard Bernoulli, no straight-through. The HyperNet is evolved (no gradient needed); the classifier and Q-head receive ordinary backprop gradients with the mask treated as input data, so straight-through is not needed.
- [x] `AttentionParameterAdapter` is a simplified version of the convnet adapter — drops the image-specific depth/scale heuristic and uses only layer-id one-hot context

#### Verified shapes and parameter counts (smoke test, seq_len=128, batch=8)
| Component | Params | Trained by | Notes |
|---|---|---|---|
| Encoder (DistilBERT) | ~66M | frozen | not exposed to PGPE |
| HyperNetwork | 29,984 | PGPE | the search space |
| AttentionMaskGenerator | 49,281 | (produced by HN) | hidden 64, scores per token |
| Classifier | 98,690 | backprop (Phase 7) | hidden 128, 2 classes |
| QHead | 99,464 | backprop (Phase 7) | hidden 128, n_codes=8 |

End-to-end forward pass confirmed:
- input_ids `(8, 128)` int32 → encoder hidden `(8, 128, 768)` float32 (5s first compile)
- mask logits `(8, 128)` → hard mask `(8, 128)` int32; random init gives ~50% mask density
- pooled `(8, 768)` → classifier `(8, 2)` and Q-head `(8, 8)`

#### Post-review hardening (Phase 3 follow-ups, applied 2026-04-25)
- `bernoulli_hard_mask` now defaults to `fallback_to_cls=True`. If Bernoulli + padding produces an all-zero mask, position 0 ([CLS]) is force-attended. Guards `mean_pool_masked` from a near-zero pool that the classifier could otherwise read as a free class indicator. Verified: very-negative logits → density goes from `[0,0,0,0]` to `[1,1,1,1]`; pool max-abs goes from `0.0e+00` to `1.0`.
- `bernoulli_hard_mask` docstring now flags non-differentiability explicitly.
- `AttentionMaskGenerator` docstring now records the position-blind design choice and points to Phase 4 `position_entropy` as the diagnostic surface.
- `AttentionParameterAdapter` notes the single-layer-target collapse edge case.
- `AttentionPolicy.__init__` now logs HN search dim, HN-emitted target params, and the compression ratio explicitly. Current ratio is **1.64x** (49,281 target / 29,984 search) — modest. Worth noting: the BloodMNIST HN was ~32k against a ~225k target generator (~7x compression). For IMDb, the target is small enough that PGPE could plausibly evolve it directly.

#### Deferred review items (revisit later, not Phase 3 defects)
- **HyperNet may be over-engineered for the small target.** Compression ratio is 1.64x, not 7x. Decision deferred to Phase 8 smoke test: if rollout time is dominated by HN inference, fall back to direct evolution of the AttentionMaskGenerator's 49k params.
- **JIT-captured encoder params (~67M).** Captured in the encoder's forward closure. Not a leak; flagged for Phase 7 trainer integration when `jax.vmap` over the rollout might interact awkwardly with the closure.
- **DistilBERT loaded at policy `__init__`.** ~400ms first call (cached afterward). For multi-process / multi-worker PGPE setups, consider lazy initialization. Deferred to Phase 6/7.
- **Validated:** `datasets 3.6.0` + `load_dataset('imdb', split='...')` works without `trust_remote_code`. Confirmed in Phase 2 smoke test (12500/12500 split balance). No Phase 1 changes needed.

### Phase 3.5 — Architectural capacity check (complete 2026-04-25, verdict: PASS)

The BloodMNIST diagnostic lesson was that a 28x28 ConvTranspose generator could not represent sub-pixel hematologic morphology regardless of controller sophistication. This phase validated that the binary per-token mask space CAN in principle express aspect-level disentanglement on IMDb before more time is spent on attention-side controllers.

Implementation: `examples/capacity_check_imdb.py`
- Trains a small `Classifier(head_hidden=128, n_classes=2)` head over `mean_pool_masked` features from a frozen DistilBERT, using the full encoder attention mask (3 epochs, batch 64, lr 1e-3).
- Reuses the same `Classifier` and pooling that Phase 7 will train via backprop, and the same `FrozenDistilBertEncoder` from `attention_transformer.py`. Whatever this experiment can express is exactly what the actual training loop will be able to express.
- Evaluates the trained head with rule-based mask families on 4992 test samples.

Mask families used (no POS tagger installed, so POS-based families were dropped; expanded keyword lists carry the load):
- `all_tokens` — upper bound
- `random_p005 / p010 / p050 / p200` — density baselines
- `keyword_acting / plot / pacing` — sentiment-bearing aspect words (~90 per aspect; nouns + descriptors)

#### Headline numbers (after expanded keyword lists, 2026-04-25)

| family | density | acc | density-interp random | margin | preservation vs upper |
|---|---|---|---|---|---|
| all_tokens | 100.0% | 0.8317 | — | — | 100.0% |
| random_p050 | 5.0% | 0.7686 | — | — | 92.4% |
| random_p200 | 20.0% | 0.8097 | — | — | 97.4% |
| keyword_acting | **6.6%** | **0.8165** | 0.7731 | **+0.043** | **98.2%** |
| keyword_plot | **2.3%** | **0.7933** | 0.6404 | **+0.153** | **95.4%** |
| keyword_pacing | **2.5%** | **0.7676** | 0.6513 | **+0.116** | **92.3%** |

#### Reads
- **Mask space carries sentiment at target density.** At the planned target sparsity of 5-20%, the random-subset baseline already preserves 92-97% of upper-bound accuracy. Binary token selection is sufficient for the sentiment signal.
- **Aspect-aware sparse selection is informative.** Each keyword aspect mask beats a density-interpolated random baseline by +4 to +15 percentage points. Aspect tokens carry more sentiment information per token than random tokens at the same density.
- **Aspect ordering is stable and sensible.** Acting > plot > pacing in accuracy preservation. People express sentiment more directly about acting than about pacing, which matches the linguistic structure of movie reviews.
- **The architecture is not the bottleneck.** Whether disentanglement EMERGES is now a training-procedure question for Phase 4-8 (PGPE search + fitness composition + Q-head MI signal).

#### Caveats / deferred refinements
- The reference classifier was trained on full attention only. A more rigorous test would train with mask-augmentation (random masks during training) so the classifier is robust to mask density and the aspect-mask comparison eliminates distribution shift. Deferred — the current preservation ratios already cleared the >=90% bar.
- POS-based mask families (adj-only, noun-only, verb-only) were dropped because no POS tagger was installed. Worth adding if Phase 4-8 results suggest the mask space carries sentiment but not aspect.
- Keyword lists are hand-curated and small (~90 words). A learned aspect classifier could give sharper masks, but the existing test was strong enough to discharge the architectural-capacity risk.

#### Phase 4 monitoring carried forward
The capacity check assumes PGPE-evolved masks reach the target sparsity (5-20%) and that different discrete codes select disjoint token subsets. Phase 4 metrics to track:
- `sparsity` — make sure masks live in the 5-20% band
- `per_class_attention_overlap` — different discrete codes should pick disjoint subsets
- Q-head accuracy on `z_discrete` recovery — direct disentanglement signal

If trained masks drift into <2% sparsity OR cross-code overlap stays near 1.0, the capacity-check assumption is violated and we revisit the architecture (multi-head masks, hierarchical selection).

### Phase 4 — Attention-specific fitness metrics (complete 2026-05-12)
- [x] Created `evojax/task/attention_metrics.py` with the JAX-jittable / vmappable metric set
  - [x] `sparsity_density` — mean attended-fraction over non-pad positions
  - [x] `sparsity_band_reward` — peaked reward inside `[low, high]` (defaults `[0.05, 0.20]`); linear falloff normalized by `target_low`
  - [x] `span_continuity` — mean contiguous-run length of attended tokens (counts runs via padded-diff trick)
  - [x] `position_entropy` — normalized entropy of the batch-aggregated position histogram, in `[0, 1]`
  - [x] `per_code_attention_overlap` — average pairwise cosine similarity of per-code mask centroids; the carried-forward Phase 3.5 monitoring metric (lower is better, target well below 1.0)
  - [x] `pos_neg_attention_overlap` — diagnostic-only, detects collapse where masks no longer condition on the latent code at all
  - [x] `compute_all` aggregator that splits fitness-target metrics from pure diagnostics
- [ ] (Deferred) `pos_diversity` — postponed to the CA re-instantiation phase since no POS tagger is installed and Phase 3.5 cleared the architecture-capacity bar without it
- [ ] (Phase 5 task) Strip BloodMNIST-specific morphology metrics (`cell_circularity`, `nucleus_offset`, etc.) from the IMDb fork of `pgpe_ca.py`. Left there for Phase 5 because the surgery is on the solver's fitness composition, not the metric definitions.

#### Sanity checks performed (2026-05-12)
Hand-crafted inputs returned the exact expected values:
- half-attended `[1,1,1,1,0,0,0,0]`: density `0.50`, band reward at `[0.40,0.60]` → `1.0`, span continuity `4.0`
- two runs of length 2 → span continuity `2.0`
- attention concentrated at position 0 → position entropy `0.0`; uniform attention → `1.0`
- identical-across-codes masks → `per_code_attention_overlap = 1.0`; disjoint per-code masks → `0.0`
- same masks for pos/neg classes → `pos_neg_attention_overlap = 1.0`; flipped → `0.0`
- `jax.jit(compute_all, static_argnames=('n_codes',))` compiles and runs

Real-policy integration check (random HyperNet init, batch=64, seq_len=128):
- sparsity_density `0.5029` — random Bernoulli, ~50% as expected
- sparsity_band_reward `0.0000` — random init is far above the `[0.05, 0.20]` band
- span_continuity `2.01` — random runs of length ~2
- position_entropy `0.9984` — uniform across positions, no shortcut yet
- code_disjointness `0.1171` — codes produce nearly identical masks at init (HyperNet has not yet learned to differentiate)
- pos_neg_attention_overlap `0.9726` — mask not yet conditioned on label, as expected

#### Phase 4 review fixes (applied 2026-05-13)
- `pos_neg_attention_overlap` docstring corrected: it detects conditioning on the sentiment LABEL, not on the latent code (latent-code conditioning is what `per_code_attention_overlap` measures).
- `sparsity_band_reward` docstring now states the asymmetry explicitly: with defaults the falloff hits 0 at density=0.00 on the low side and density=0.25 on the high side (not "one band-width" — it's one `target_low` width on either side). The asymmetry in absolute density terms is intentional: at very low density the pooled vector loses information rapidly, while denser-than-band masks just give the classifier more context.
- `compute_all` docstring now flags the direction gotcha for Phase 5 wiring (see below).

#### Note on JAX memory state during testing
A transient `INTERNAL: Failed to allocate 2359296 bytes for new constant` fired on the first integration run. Recovered immediately with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Likely GPU-memory state left over from the Phase 3.5 training session. Phase 6 must set this env var inside `train_imdb.py` (see Phase 6 task list).

### Phase 5 — Solver adaptation (complete 2026-05-13)
- [x] Forked: `evojax/algo/pgpe_ca.py` → `evojax/algo/pgpe_ca_text.py` (class `PGPE_CA_Text`)
- [x] `tell()` signature replaces 10 BloodMNIST morphology args (`cell_circularity`, `nucleus_offset`, `nuc_cell_ratio_range`, `nuc_eccentricity_range`, `cell_area_var`, `morph_dark_range`, `morph_center_edge_range`, `edge_dark_frac`, `code_proto_corr`, `proto_angle_spread`) with 6 attention args from `attention_metrics.compute_all`: `sparsity_band_reward`, `span_continuity`, `code_disjointness`, `position_entropy`, `sparsity_density`, `pos_neg_attention_overlap`
- [x] Fitness composition uses `code_disjointness` (higher-is-better) — the standalone `per_code_attention_overlap` is NOT used as a fitness term anywhere in the file
- [x] Fitness composition adds three new rank-normalized terms with starting weights:
  - `w_sparsity_band = 0.10` for `sparsity_band_reward`
  - `w_span_continuity = 0.06` for `span_continuity`
  - `w_disjoint = 0.10` for `code_disjointness`
- [x] Framework-general mechanics preserved:
  - `belief_space` initialization unchanged
  - All five KS update functions still called (`update_topographic_ks`, `update_normative_ks`, `update_domain_ks`, `update_situational_ks`, `update_history_ks`)
  - `rank_normalize` discipline applied to every fitness component
  - Hybrid weighting (`--hybrid-fitness-weights`) still works — `w_mi` pinning logic untouched
  - `compute_metric_slopes`, `compute_semantic_trap_state`, `compute_normative_state` still called
- [x] BloodMNIST-flavored slots in shared CA helpers (`update_metric_history`, `update_normative_ks`) are reused with attention metrics:
  - `update_normative_ks` cell-metric slots → `span_continuity`, `position_entropy`, `sparsity_density`
  - `update_metric_history` morphology kwargs → attention-metric averages (`avg_morph_dark_range` carries `sparsity_density`, `avg_code_proto_corr` carries `1 - code_disjointness`, `avg_proto_angle_spread` carries `position_entropy`, etc.)
  - Names inside the shared CA helpers stay BloodMNIST-flavored, but the diagnostics dict re-labels them on the way out (`sparsity_density_short`, `code_overlap_short`, `norm_span_continuity_floor`, etc.)
- [x] `bio_score` in `update_history_ks` replaced with an attention-quality score (equal weight on `sparsity_band_reward`, `span_continuity`, `code_disjointness`)
- [x] Gradient blend path preserved structurally; defaults to OFF via `ca_blend_coeff=0.0` for baseline
- [x] TODO marker placed where the future logit-prior CA intervention will hook in (in the policy/sampler, not in this solver)

#### Smoke test (2026-05-13)
End-to-end `ask()` → `tell()` with random inputs at `pop_size=8, n_codes=11, param_size=128`:
- `tell()` completed without errors, `t` advanced to 1, `best_score = 0.7486`
- 97 diagnostic keys emitted, including all attention-metric averages and the relabelled normative bounds (`norm_span_continuity_floor`, `norm_position_entropy_ceiling`, `norm_sparsity_density_low/high`)
- Sparsity bounds learned within the input range (low=0.098, high=0.331 from inputs uniform(0.05, 0.20))
- Verified: `XLA_PYTHON_CLIENT_PREALLOCATE=false` is required, same as Phase 4

#### Note on shared-CA-helper slot reuse
The CA helper modules (`belief_space.py`, `knowledge_sources.py`, `helper_functions.py`) still use BloodMNIST-flavored variable names inside (`cell_circularity_buf`, `dark_range_buf`, etc.). Rather than fork those too, the IMDb solver reuses the slots and re-labels at the diagnostic boundary. If a future phase needs the CA helpers to be task-aware (e.g., normative bounds with attention-specific direction conventions), refactoring those modules to accept a generic metric registry is the cleanest path.

#### Phase 5 review fixes (applied 2026-05-16)
- `ca_blend_coeff` default in `PGPE_CA_Text.__init__` changed from `0.05` to `0.0`. The class is now baseline-safe out of the box — `ca_blend_coeff=0.0` matches the documented IMDB-B00 configuration.
- Added a docstring NOTE under the new attention-metric args in `tell()` flagging that `disc_logits` is consumed as `(pop_size, n_codes)` and must be squeezed/averaged over the batch axis if Phase 7 produces a 3D tensor.
- Added a multi-line TODO at the `update_normative_ks` call site documenting that the BloodMNIST clip bounds inside the CA helper (`cell_circularity_floor ∈ [0.45, 0.80]`, `cell_area_var_ceiling ∈ [0.001, 0.01]`, `nucleus_offset ∈ [0.05, 0.55]`) will produce nonsensical normative bounds when the slots carry attention metrics. Acceptable for IMDB-B00 (CA off) but the `norm_*_floor`/`_ceiling` diagnostics are NOT meaningful in baseline runs.
- `best_bio_score` (the value fed to `update_history_ks` for KS rescue) now normalizes `span_continuity` by an 8.0 anchor and clips to `[0, 1]`, putting the three components on comparable scales.
- The `compute_semantic_trap_state` output `bio_score_latest` (a separate path, computed inside the CA helper from buffered metric averages with the original BloodMNIST weighting) is renamed in the diagnostic dict from `attn_quality_score_latest` to `bio_proxy_latest_raw` with an explicit comment that it can exceed 1.0 and is NOT a valid attention-quality score. The genuine normalized score is `best_bio_score` inside `tell()`.
- Smoke test re-run confirms: default ca_blend is 0.0, tell() works with span_continuity in [5, 15], `ca_blend` diagnostic is 0.0.

### Phase 6 — Training entry point (complete 2026-05-16)
- [x] Created `examples/train_imdb.py` (mirrors `train_bloodmnist.py` structure)
  - [x] `os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')` set BEFORE any `jax` import (top of file, line 50)
  - [x] CLI flags present:
    - Core: `--gpu-id`, `--pop-size` (def 512), `--batch-size` (def 64), `--max-iter` (def 290000), `--test-interval`, `--log-interval`, `--seed`
    - PGPE: `--center-lr-gen`, `--std-lr-gen`, `--init-std-gen`, `--lr-decay-coef`, `--lr-decay-steps`
    - CA: `--ca-blend-coeff` (def **0.0**, IMDB-B00 default), `--ca-blend-start-iter`, `--ca-blend-ramp-iters`, `--ca-blend-rfl-lo/hi`
    - Fitness mode: `--static-fitness-weights` (def **True**), `--adaptive-fitness-weights` (opt-in for IMDB-A01), `--hybrid-fitness-weights`, plus the full `--static-w-*` / `--static-*-ramp-*` knobs
    - IMDb-specific: `--encoder-name` (def `distilbert-base-uncased`), `--seq-len` (def 256), `--n-discrete-codes` (def 8), `--n-continuous-codes` (def 2), `--noise-dim` (def 62), `--score-hidden`, `--classifier-hidden`, `--q-hidden`, `--chunk-size`
    - Logistics: `--checkpoint-interval`, `--checkpoint-dir`, `--resume-from`, `--log-dir`, `--debug`
  - [x] `build_components(config, logger)` constructs `AttentionPolicy`, `IMDb` train/test, `belief_space`, and `PGPE_CA_Text` and returns them in a dict so Phase 7 can pick them up without re-reading the config
  - [x] Class-balanced sampling is enabled by default via the trainer's existing `class_indices` logic (binary labels → 50/50 — see Phase 2 trainer-coupling notes)

#### Verified construction run (2026-05-16)
`python examples/train_imdb.py --pop-size=32 --batch-size=8 --seq-len=128` succeeds end-to-end through component construction:
- HyperNet search dim = 29,984 (PGPE parameter count)
- Classifier head = 98,690 params (backprop, Phase 7)
- Q-head = 99,464 params (backprop, Phase 7)
- Train task: 25,000 samples, balance `[12500, 12500]`
- Test task: 25,000 samples, balance `[12500, 12500]`
- `CA blend: coeff=0.0000 (coeff=0.0 means CA is OFF — IMDB-B00 default)` confirmed in log
- `Fitness mode: static (hybrid=False)` confirmed in log

#### Phase 7 wiring deferred (documented in the entry point as a stop block)
The shared `evojax/trainer.py` still assumes the BloodMNIST `GenPolicy` interface. The entry point logs "Phase 6 stop point: components built. Phase 7 owns Trainer wiring." and returns the components dict for REPL/test access. Phase 7 changes required (all encoded in the entry-point comment):
  - branch on `np.issubdtype(data.dtype, np.integer)` in `trainer.py:1156` to skip the float cast and `/255` for int token IDs
  - thread `train_task_disc.attention_masks` through `sample_batch`
  - swap the GAN real/fake loss for classifier cross-entropy in the D-step
  - rename `real_fake_loss` → `classifier_loss` (or alias)
  - thread `attention_metrics.compute_all` outputs into `solver.tell()` per the disc_logits shape note

#### Phase 6 review fixes (applied 2026-05-16)
- **`--gpu-id` ordering.** `os.environ['CUDA_VISIBLE_DEVICES']` was being set inside `__main__` AFTER `import jax`. JAX reads the env var at import time, so the flag was a no-op. Fix: sniff `sys.argv` for `--gpu-id N` and `--gpu-id=N` forms at module load (before the JAX import) and set the env var there. Removed the no-op post-import set in `__main__`.
- **Latent dimension log clarity.** The log line previously claimed `total = noise + discrete + continuous = 72`. The actual latent that `IMDb.reset_fn` constructs is `[noise, cat_one_hot]` → dim 70. Continuous codes are configured but not wired into the latent. Log line now reports `actual_total=70` and explicitly notes the continuous codes are not yet wired in.

#### Architectural gap discovered during the Phase 6 review — RESOLVED 2026-05-16

While tracing the latent flow to fix the log mismatch, found that the original `AttentionMaskGenerator` did **not** take `z` as an input — its forward signature was `__call__(hidden_states)`. Per-individual the HyperNet emitted **one** MaskGenerator parameter set, and that MaskGenerator scored tokens from `hidden_states` alone. Consequences had this not been fixed:
- Two samples in the same batch that share encoder input but carry different discrete codes would have produced **identical** mask logits (only Bernoulli sampling noise differs).
- The Q-head therefore could not have recovered `z_discrete` from the mask, MI signal would have stayed at zero, and the InfoGAN disentanglement objective could not have made progress.
- In the BloodMNIST analog this is not a gap because `Generator.__call__(z)` directly takes `z` — `z` conditions through the Generator's forward pass while the HyperNet supplies its weights. The IMDb architecture needed the same.

Fix applied (`evojax/policy/attention_transformer.py`):
- `AttentionMaskGenerator` now declares `z_dim` as a static field and the forward signature is `__call__(hidden_states, z)`. `z` is broadcast across the sequence axis and concatenated with `hidden_states` at each position, so the first Dense layer scores tokens from `(h_t, z)` together. The same MLP runs at every position (position-blindness preserved).
- `AttentionPolicy.__init__` now passes `z_dim=self.n_codes` and uses a dummy `z` of shape `(1, n_codes)` during Flax init.
- `AttentionPolicy.mask_logits(params, hidden_states, z)` gains the third arg. Phase 7's trainer will thread `state.cat_codes` through this call.
- Parameter footprint impact: `score_h1/kernel` shape `(768, 64) → (776, 64)`, an extra 512 target params. PGPE search dim 29,984 → 30,000 (+16). Compression ratio 1.64x → 1.66x — negligible.

Sanity test performed (with random HN init, batch=8, identical hidden states across rows):
- `score_h1/kernel: (776, 64)` confirms z-dim concatenation in the MLP input.
- Two samples with the SAME one-hot `z` produce identical logits (max abs diff = 0.0).
- Two samples with DIFFERENT one-hot `z` produce non-zero logit differences (mean abs 3e-4 at random init; small because HyperNet output kernel is init-scaled to stdev=0.01, but the z signal will be amplified by training as the MI fitness term pushes the HyperNet to make code-dependent selections).

Capacity-check implication: the Phase 3.5 verdict still holds. That experiment trained a Classifier on top of rule-based masks (random / keyword) and did NOT route through the AttentionMaskGenerator. The set of representable masks is unchanged by z-conditioning; what changes is which mask the policy emits for which `z`. No re-run of the capacity check needed.

### Phase 7 — Trainer (complete 2026-05-16)

**Scope change vs. original plan.** The original task list said "audit `evojax/trainer.py` for image-specific assumptions." `trainer.py` is 1856 lines of tightly GAN-coupled BloodMNIST logic (separate `Latent_Points` task, BatchNorm batch-stats syncing, shape-diversity tracking, real/fake D-step, prototype centroid bookkeeping). Forking would have meant stripping ~60% of the file. Instead, wrote a fresh `evojax/trainer_imdb.py` (~360 lines) purpose-built for the IMDb shape: frozen encoder, no GAN, no Latent_Points, classifier CE in the D-step. Kept the trainer's public surface compatible with what `examples/train_imdb.py` builds via `build_components()`.

- [x] Created `evojax/trainer_imdb.py` with `TrainerIMDb` class
  - [x] `__init__` reads policy / solver / train_task / test_task plus standard knobs (max_iter, batch_size, log_interval, checkpoint_interval, lr); pre-builds `class_indices` from `train_task.labels` for 50/50 balanced sampling
  - [x] Adam optimizers for the Classifier and Q-head (both backprop, per Phase 3 design)
  - [x] JIT-compiled `_evaluate_pop` vmaps over the PGPE population: for each individual, HyperNet → MaskGen weights → `mask_logits(params, hidden, z=cat_codes)` → Bernoulli mask → pooled features → classifier/Q-head logits → fitness components + attention metrics via `compute_all`
  - [x] JIT-compiled `_train_cls_step` and `_train_q_step` backprop on the elite individual's mask each iteration
  - [x] `run()` main loop: balanced batch → encoder forward (once per iter, shared across pop) → solver.ask/eval/tell → classifier backprop → Q-head backprop → log + checkpoint
  - [x] Logging emits `fitness_adv`, `fitness_mi`, `sparsity`, `span_continuity`, `code_disjointness`, `position_entropy`, `pos_neg_overlap`, `cls_loss`, `q_loss`, plus a few solver diagnostics (`w_adv`, `w_mi`, `stdev_mean`, `ca_blend`)
  - [x] Minimal pickle-based checkpoint save (center, stdev, classifier params, q-head params); resume from disk deferred
  - [x] `rank_normalize` discipline preserved — fitness composition lives inside `PGPE_CA_Text.tell()`, not in the trainer
- [x] `disc_logits` shape contract fixed: solver expects `(pop_size, batch, n_codes)` (not 2D as my Phase 5 docstring claimed). Updated trainer to pass full per-individual Q-head logits, and corrected the docstring in `pgpe_ca_text.py`.
- [x] `examples/train_imdb.py` now instantiates `TrainerIMDb` and calls `trainer.run()` — the Phase-6 "stop block" is removed.

#### Verified smoke run (2026-05-16)
`python examples/train_imdb.py --pop-size=8 --batch-size=8 --seq-len=64 --max-iter=3 --log-interval=1 --score-hidden=32 --checkpoint-interval=0`
- iter 0: 12.19s (encoder JIT + trainer JIT compile)
- iter 1: 1.08s (residual recompile)
- iter 2: 0.03s (fully JIT'd; ~400× faster than iter 0)
- f_adv ~0.4–0.5 (random init, near-chance classifier), f_mi ~−2.08 (log(8) chance for Q-head), sparsity ~0.5 (random Bernoulli), ca_blend=0.0
- Checkpoint pickle written successfully

#### Deferred (Phase 8/9 cleanup)
- Checkpoint resume from disk (only save is wired today)
- Mid-training periodic test-set evaluation
- Structured TSV logs analogous to `trainer.py`
- Multi-GPU population sharding (single-GPU only today)

#### Phase 7 review fixes (applied 2026-05-16)
- **(#1, real correctness)** Q-head now trains on the **MI-elite** mask (`argmax(fitness_mi)`), not the accuracy-elite. Decouples Q's training data from the f_adv elite so the feedback loop "f_adv elite mask is code-agnostic → Q trains on code-agnostic data → Q stays at chance → f_mi stays low → PGPE doesn't pressure code-dependent masks" cannot close. Classifier still trains on the accuracy-elite mask, where it belongs.
- **(#2, robustness)** `cat_codes` is now **shuffled per iteration** via `random.permutation` so no fixed batch-position-↔-code correspondence can leak into the HyperNet. The base tile is built once at the top of `run()`; each iteration applies a fresh permutation.
- **(#3, documentation)** Added a multi-line TODO docstring on `_save_checkpoint` describing the four-step resume path (load flat → reshape pytree → restore solver flat-form → reinit Adam states or save them) so the resume PR has a clear spec.
- **(#4, documentation)** Added a comment at the `solver.tell(...)` call explaining that `safety_ratios` and `spreads` are constant `jnp.ones` placeholders, with the explicit observation that any CA gradient reading them is gated by `ca_blend_coeff=0.0` in baseline so the `norm_*_floor`/`_ceiling` diagnostics are NOT meaningful in IMDB-B00.
- **(#5, observability)** `_log_iter` now emits `elite_adv=<idx>` and `elite_mi=<idx>` so each iteration's selected individuals are visible in the log — useful for spotting cases where Q-head consistently picks individual 0 (a degenerate fixed-elite signal).
- Verified: post-fix 3-iter smoke test still passes; log shows `elite_adv=1 elite_mi=0` confirming the two elites are computed independently.

### Phase 8 — Baseline run (IMDB-B00, no CA) (complete 2026-05-23)
- [x] Small-scale smoke tests: 3-iter wiring smoke (2026-05-16), 200-iter scale smoke (2026-05-17), 3000-iter trajectory smoke (2026-05-17). The first 200-iter run surfaced a sparsity-collapse pathology; weights were re-tuned (`w_sparsity_band 0.10→0.25`, `w_disjoint 0.10→0.04`, `w_span_continuity 0.06→0.10`) and the 3000-iter smoke confirmed stability.
- [x] Full baseline: `pop_size=512, batch_size=64, --ca-blend-coeff=0.0, --static-fitness-weights, --render-interval=20000`, 290000 iterations
- [x] Checkpoints every 5k iters → 58 pickles in `log/imdb/checkpoints/`
- [x] HTML render snapshots every 20k iters → 17 files in `log/imdb/checkpoints/`
- [x] IMDB-B00 defined and recorded as reference baseline in `IMDB_ABLATION_TRIAL_LOG.txt`

#### IMDB-B00 headline numbers (late-window iter 289900)
- `f_adv = 0.588` (chance 0.5; capacity ceiling 0.83 from Phase 3.5)
- `f_mi = -1.873` (chance -2.079; ~0.30 bits of code recovery captured out of 3 max)
- `sparsity = 0.038` (just below the [0.05, 0.20] target band)
- `span_continuity = 1.88`, `code_disjointness = 0.783`
- `cls_loss = 0.130` (down 5× from 0.69), `q_loss = 0.380` (down 5× from 2.08)
- Training stable for full 17 hours at 0.21s/iter, no NaNs, no divergence

#### Documented failure mode: density-tier shortcut
Across all 17 render snapshots from iter-20k to iter-290k, the 8 discrete codes settled into 3 stable density clusters:
- Dense tier (c0, c1): density ~0.07–0.09
- Mid tier (c2, c5, c6): density ~0.04
- Sparse tier (c3, c4, c7): density ~0.015

Differences between iter-20k and iter-290k are within sampling noise. The HyperNet found the easy disentanglement axis (per-code count of attended tokens) by iter ~20k and held it for the remaining 270k iters. Q-head learned to recover *tier identity* from masks, not aspect-level semantic content. This is the direct IMDb analog of BloodMNIST BLD-B00's "size/orientation shortcut" and motivates the ablation ladder in `IMDB_ABLATION_TRIAL_LOG.txt`.

### Phase 9 — Baseline analysis and success criteria (in progress)
- [x] Trajectory recorded at 10k strides (15 sample points) plus per-iter log every 100 iters
- [x] Per-code density evolution archived across 17 render snapshots
- [x] Density-tier shortcut documented as IMDB-B00's failure mode
- [x] Late-window numbers compared to Phase 3.5 capacity-check ceiling (0.83); ~23-point accuracy gap attributed to sub-band sparsity (0.038 vs 0.05 band low)
- [x] Ablation ladder defined in `IMDB_ABLATION_TRIAL_LOG.txt` (IMDB-A01, A02, A03, A04-hybrid, A05a–e)
- [ ] Open `render_iter-290000.html` (and a 20k baseline for comparison): qualitative confirmation of whether dense-tier codes pick content tokens vs stopwords vs punctuation
- [ ] Success criteria assessment for IMDB-B00:
  - Classifier accuracy within 2-3 points of soft-attention baseline → **NOT MET** (0.59 vs 0.83 ceiling = ~24 points off)
  - Attention masks "readable" — coherent spans, not pure stopwords/punctuation → **PENDING visual inspection**
  - Q-head recovers `z_discrete` above chance → **PARTIAL** (recovers tier identity, not full code identity; ~10% of available MI captured)
  - `sparsity` stays in target range → **NEAR MISS** (0.038 vs target low 0.05)
  - Training stable, no all-ones/all-zeros collapse → **MET**
  These are the expected partial results for a baseline whose job is to characterize the architecture, not to beat soft attention. The pattern (one partial success, one documented shortcut) is the structural mirror of BLD-B00 and the right launching point for the ablation ladder.

### Phase 10 (new) — Ablation ladder
Definitions live in `IMDB_ABLATION_TRIAL_LOG.txt`. Order of execution:
1. **IMDB-A04-hybrid** — highest leverage, directly tests if the BloodMNIST hybrid winner transfers
2. **IMDB-A01** — dynamic weighting only, isolates adaptive controller
3. **IMDB-A02** — CA logit-prior intervention (requires building the prior hook in the sampler)
4. **IMDB-A03** — dynamic + logit prior combined
5. **IMDB-A05 series** — shortcut-aware CA with Domain KS detectors for punctuation, position, stopwords, and the density-tier failure mode

## Reference Files (BloodMNIST → IMDb mapping)

| Purpose | BloodMNIST file | IMDb file | Status |
|---|---|---|---|
| Task / dataset | `evojax/task/bloodmnist.py` | `evojax/task/imdb.py` | ✅ created (Phase 2) |
| Disentanglement metrics | `evojax/task/latent.py` (sense / cons / intra) | `evojax/task/attention_metrics.py` (sparsity / span / disjointness / position / pos_neg) | ✅ created (Phase 4) |
| Policy (G + D + Q + HyperNet) | `evojax/policy/convnet.py` | `evojax/policy/attention_transformer.py` | ✅ created (Phase 3) |
| Solver | `evojax/algo/pgpe_ca.py` | `evojax/algo/pgpe_ca_text.py` (`PGPE_CA_Text`) | ✅ created (Phase 5) |
| Capacity check | (n/a — BloodMNIST didn't have one) | `examples/capacity_check_imdb.py` | ✅ created (Phase 3.5) |
| Entry point | `examples/train_bloodmnist.py` | `examples/train_imdb.py` | ⏳ Phase 6 |
| Main loop | `evojax/trainer.py` | same file, adjusted for text-mode | ⏳ Phase 7 |
| Setup extras | `setup.py` `extra` | `setup.py` `nlp` extras section | ✅ added (Phase 1) |

## Post-Baseline (Deferred to After IMDB-B00)

- IMDB-A01: dynamic fitness weighting only
- IMDB-A02: CA logit-prior controller only (shortcut detectors, no gradient blend)
- IMDB-A03: hybrid weighting + CA logit prior
- IMDB-A04 series: attention-shortcut-aware CA with Domain KS detectors for punctuation, position, token frequency
- Each run includes class-balanced sampling (standard)

## Working Discipline

- Match BloodMNIST discipline: one controller layer at a time, keep architecture fixed during ablations
- Update this file after each phase completes
- After IMDB-B00 runs, establish an `IMDB_ABLATION_TRIAL_LOG.txt` analogous to `ABLATION_TRIAL_LOG.txt`
- Commit phase work as discrete commits on the IMDb branch

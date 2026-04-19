# IMDb Hard-Attention Transformer Plan

Branch: `claude/evolve-infogan-hypernetworks-s3r6d-67132f2-imdb-baseline`
Created: 2026-04-19
Parent: `claude/evolve-infogan-hypernetworks-s3r6d-67132f2-blood-baseline` at commit `cf37ef7`

This document tracks the setup tasks for porting the HyperNet-InfoGAN architecture from BloodMNIST to IMDb hard-attention. The near-term goal is to mirror the BloodMNIST workflow: establish a non-CA baseline first, then layer CA integration on top once the architecture is validated.

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

## Open Questions (decide before Phase 3)

1. **Discrete code count `K`.** Options:
   - `K = 2` — matches the binary sentiment task directly; MI target is clean but disentanglement claim is weak.
   - `K = 4-8` — lets codes disentangle aspect (plot, acting, pacing, emotional intensity) within each polarity; matches BloodMNIST `K=8` and gives a richer disentanglement signal.
   - Recommended default: `K = 8` (same as BloodMNIST) for methodological continuity.

2. **Mask sampling strategy.**
   - Bernoulli with straight-through estimator — simplest, gradients are biased but the classifier head absorbs it.
   - Concrete / Gumbel-Sigmoid relaxation — differentiable, but we rely on PGPE for the HyperNet gradient anyway.
   - Hard Bernoulli + REINFORCE gradient for the attention layer — matches Ben Goertzel framing but complicates the D-step.
   - Recommended default: Bernoulli with straight-through estimator. HyperNet is evolved (no gradient needed); classifier sees a valid masked-hidden-state vector.

3. **HyperNet target parameterization.** The HyperNet can produce:
   - Per-token query vectors that attend to encoder hidden states.
   - Full attention-layer weights (Q/K/V projections + output projection).
   - Parameters of a small MLP that takes `h_t` and outputs a logit per token.
   - Recommended default: a small MLP per head (scores each token from its encoder hidden state). Fewer params, easier chunked generation, analogous to conv layers in BloodMNIST.

4. **Encoder depth used.**
   - Full DistilBERT (6 layers, 768-dim hidden).
   - Truncated first 2-3 layers to reduce compute.
   - Recommended default: full frozen DistilBERT; compute is dominated by PGPE rollouts, encoder pass is amortized.

5. **Pooling strategy for the classifier head.**
   - Mean-pool over masked tokens (only attended positions contribute).
   - Sum-pool (preserves magnitude signal about how much was attended).
   - Attention-weighted pool using the same logits.
   - Recommended default: mean-pool over attended tokens with length normalization (guards against length confound).

## Task List

### Phase 1 — Environment and dependencies
- [ ] Add `transformers`, `datasets`, `tokenizers` to `setup.py` or equivalent extras
- [ ] Verify JAX/Flax compatibility path for DistilBERT (use `FlaxDistilBertModel`)
- [ ] Cache pretrained DistilBERT weights locally to avoid download during training
- [ ] Confirm tokenizer and model version pinning

### Phase 2 — Dataset task
- [ ] Create `evojax/task/imdb.py` implementing `VectorizedTask`
  - [ ] Load IMDb (HuggingFace `datasets.load_dataset('imdb')`)
  - [ ] Pre-tokenize entire dataset at init (fixed length, e.g., 256 tokens) to avoid per-step tokenization cost
  - [ ] Cache token_ids + attention_mask + labels as JAX arrays
  - [ ] `reset(key)` returns token_ids + attention_mask + label for a batch
  - [ ] Class-balanced batch construction (50/50 pos/neg) exposed via task state
  - [ ] Optional: pre-compute POS tags per token (once, cached) for the eventual CA metric `pos_diversity`

### Phase 3 — Policy: encoder + HyperNet + attention + heads
- [ ] Create `evojax/policy/attention_transformer.py`
  - [ ] `FrozenEncoder` wrapper around `FlaxDistilBertModel` (no parameters exposed to PGPE)
  - [ ] `LayerHyperNetwork` adapted: reuse chunk-embedding design from `convnet.py`, target attention-MLP weights instead of conv weights
  - [ ] `AttentionMaskGenerator`: takes encoder hidden states `h_t` and HyperNet-generated MLP weights, produces per-token logits, Bernoulli sampling with straight-through
  - [ ] `ClassifierHead`: mean-pool over masked tokens + 2-layer MLP + sigmoid/softmax for sentiment
  - [ ] `QHead`: predicts `z_discrete` from the pooled masked feature vector; use cross-entropy for MI estimation

### Phase 4 — Attention-specific fitness metrics
- [ ] Create `evojax/task/attention_metrics.py` or extend `latent.py`
  - [ ] `sparsity`: mean mask density per sample (target range soft bound, e.g., 5-20%)
  - [ ] `span_continuity`: mean contiguous run length of attended tokens
  - [ ] `position_entropy`: entropy of attended-token positions (detects position-biased attention)
  - [ ] `per_class_attention_overlap`: mask overlap between positive-class and negative-class centroids
  - [ ] `pos_diversity` (deferred to CA phase): entropy over POS tags of attended tokens
  - [ ] Remove BloodMNIST morphology metrics (`cell_circularity`, `nucleus_offset`, etc.) from the fitness composition

### Phase 5 — Solver adaptation
- [ ] Fork: copy `evojax/algo/pgpe_ca.py` to `evojax/algo/pgpe_ca_text.py`, or refactor to support task-specific metric registries
  - [ ] Strip BloodMNIST-specific CA instantiation (A04a-e morphology signals)
  - [ ] Keep framework-general mechanics: belief space, KS roles, `rank_normalize`, hybrid weighting
  - [ ] Gradient blend entry points exist but are disabled for baseline
  - [ ] Future CA integration will add logit-prior outputs (not gradient outputs) — leave structural hooks but not logic

### Phase 6 — Training entry point
- [ ] Create `examples/train_imdb.py` (mirror `train_bloodmnist.py`)
  - [ ] CLI flags: `--gpu-id`, `--pop-size`, `--batch-size`, `--max-iter`, `--ca-blend-coeff` (default 0.0), `--static-fitness-weights` (default True), `--hybrid-fitness-weights`, `--encoder-name` (default `distilbert-base-uncased`), `--seq-len`, `--K` (discrete code count)
  - [ ] Wire task, policy, solver, trainer
  - [ ] Class-balanced sampling enabled by default

### Phase 7 — Trainer adjustments
- [ ] Audit `evojax/trainer.py` for image-specific assumptions
  - [ ] D-step: replace GAN real/fake loss with classifier cross-entropy
  - [ ] Rename `real_fake_loss` diagnostic to `classifier_loss` (or keep RFL name as a health proxy alias)
  - [ ] Preserve D-freeze threshold logic but re-tune for classifier dynamics (classifier converges faster than GAN D)
  - [ ] Log-line order: `fitness_adv`, `fitness_mi`, `sparsity`, `span_continuity`, `position_entropy`, `per_class_overlap`, `classifier_loss`
  - [ ] Keep `rank_normalize` discipline in fitness composition

### Phase 8 — Baseline run (IMDB-B00, no CA)
- [ ] Small-scale smoke test: `pop_size=32`, 5k iterations, verify no NaNs, sparsity stays in range, classifier loss decreases
- [ ] Full baseline: `pop_size=512`, `batch_size=64`, `--ca-blend-coeff=0.0`, `--static-fitness-weights`, class-balanced, target 290k iterations
- [ ] Checkpoint every 5k iterations
- [ ] Define this run as `IMDB-B00` reference baseline

### Phase 9 — Baseline analysis and success criteria
- [ ] Record metrics at 20k, 60k, 100k, 150k, 290k
- [ ] Qualitative inspection: render attended spans per discrete code across 10-20 sample reviews
- [ ] Success criteria for IMDB-B00:
  - Classifier accuracy within 2-3 points of a standard soft-attention baseline on the same frozen encoder
  - Attention masks pass the "readable" bar: attended tokens form coherent spans, not stop-words or punctuation only
  - `Q`-head recovers `z_discrete` above chance (accuracy > 1/K + margin)
  - `sparsity` stays inside the target range through training
  - Training is stable (no mode collapse to all-ones or all-zeros masks)
- [ ] Document failure modes observed, as BloodMNIST's analog of "size/orientation shortcut" — these will seed the Domain KS for the eventual CA integration

## Reference Files (from BloodMNIST, to mirror)

| Purpose | BloodMNIST file | IMDb file (to create) |
|---|---|---|
| Task / dataset | `evojax/task/bloodmnist.py` | `evojax/task/imdb.py` |
| Latent / shared-z metric | `evojax/task/latent.py` | extend for attention metrics |
| Policy (G + D + Q + HyperNet) | `evojax/policy/convnet.py` | `evojax/policy/attention_transformer.py` |
| Solver | `evojax/algo/pgpe_ca.py` | `evojax/algo/pgpe_ca_text.py` |
| Entry point | `examples/train_bloodmnist.py` | `examples/train_imdb.py` |
| Main loop | `evojax/trainer.py` (modified) | same file, adjusted for text |

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

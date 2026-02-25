# EvoJAX Hyper-InfoGAN + CA Plan (BloodMNIST Pivot)

Branch: `claude/bloodmnist-unsupervised-ablation`  
Date: 2026-02-25

## 1) Research Pivot Summary

We are pivoting from OrganSMNIST to BloodMNIST for the **method-development track** while preserving the long-term goal:

- Fully unsupervised InfoGAN-style latent discovery.
- CA-guided evolutionary stabilization (PGPE + belief space/KS).
- Future transfer to IMDB + transformer hard-attention (non-backprop-centered).

Reason for pivot:

- OrganSMNIST runs frequently produce repeated within-code shapes despite improved D/G balance.
- BloodMNIST has stronger natural morphology separation and is a better near-term benchmark for discrete unsupervised structure learning.

Important constraint:

- Training remains **unsupervised** (no class labels in optimization losses).
- Labels are only used for dataset loading and external evaluation.

---

## 2) Current Codebase State (This Branch)

### 2.1 Dataset/Model Adaptation

Code has been updated to support BloodMNIST (8 classes) end-to-end:

- `examples/train_bloodmnist.py` added.
- `GenPolicy` now supports configurable:
  - `n_classes`
  - `noise_dim`
  - `image_size`
  - `image_channels`
- Generator output channels are configurable (`1` grayscale / `3` RGB).
- Trainer now supports configurable:
  - `dataset_name` (`organsmnist` / `bloodmnist`)
  - `n_classes`
  - `latent_dim`
  - `image_size`
  - `image_channels`
  - `data_root`
- BloodMNIST Option-A grayscale pivot is now implemented:
  - RGB BloodMNIST is converted to luminance at load time in:
    - `evojax/trainer.py`
    - `evojax/task/latent.py`
  - `examples/train_bloodmnist.py` now runs with `image_channels=1`.

### 2.2 CA + Shape-Diversity Controller (from prior phase)

The collapse-sensitive Route 2 controller remains active:

- `shape_score = 0.7 * shape_div_min + 0.3 * shape_div_mean`
- adaptive `w_shape` with clamp
- MI guard when worst-code shape diversity collapses
- shape-div slopes added to metric history (`shape_short`, `shape_med`)

### 2.3 Belief Space / KS Generalization

- Normative KS initialization now uses dynamic `num_codes`.
- Belief-space init now passes `num_codes` through to normative/topographic setup.
- Sim manager rollout buffers now infer code dimension from task state.

---

## 3) Immediate Execution Plan

## Phase A - BloodMNIST Bring-Up (No Objective Changes)

- [ ] A1: Run smoke training (2k-4k) on BloodMNIST with current settings.
  - Confirm no shape errors in rollout, D-step, checkpointing, or image export.
- [ ] A2: Verify logging fields exist and are finite:
  - `r_shape_div_avg`
  - `r_shape_div_min_avg`
  - `shape_short`, `shape_med`
  - `mi_guard`, `w_shape`
- [ ] A3: Save first baseline run artifacts at checkpoints 5k / 8k / 11k.

Success criteria:

- Training runs without runtime shape bugs.
- D/G balance remains in usable range (`real_fake_loss` occupancy in `[0.4, 0.6]`).
- Metrics TSV and KS TSV are complete for BloodMNIST runs.

## Phase B - BloodMNIST Ablations (Disentanglement-Focused)

- [x] B1: Baseline no-CA run (`ca_blend_coeff=0.0`).
  - Outcome (11.5k): persistent D dominance (`real_fake_loss` mostly `<0.30`).
- [ ] B1b: No-CA with D-control retune.
  - New knobs: `disc_update_gate`, `disc_warmup_freq`, `disc_warmup_iters`, `disc_lr`.
  - Initial setting: gate `0.40`, warmup `1/5` until `3k`, D lr `7e-5`.
- [ ] B1c: No-CA with grayscale BloodMNIST (Option A).
  - Motivation: remove independent RGB shortcut; focus PGPE + InfoGAN on morphology.
  - Implemented as luminance conversion in both trainer data path and latent RFF real-class path.
- [ ] B1d: No-CA anti-saturation Generator activation update.
  - Replace intermediate `tanh` with `leaky_relu(0.2)` (both policy and trainer Generator copies).
  - Keep bounded output but soften final clamp with `tanh(x / 2.0)`.
  - Goal: increase phenotype sensitivity for PGPE perturbations.
- [ ] B1e: No-CA explicit saturation penalty in PGPE fitness.
  - Add `sat_frac = mean(|x| > 0.85)` per member from generated pixels.
  - Penalize only excess above `sat_target=0.10` with ramped weight.
  - Goal: prevent collapse into binary/saturated shortcuts after ~3k-5k.
- [ ] B1f: Reflection-padding Generator convs + stronger sat penalty sweep.
  - Replace zero-padding (`SAME`) with explicit reflection padding + `VALID` convs.
  - Run with `sat_penalty_weight` in `0.25-0.30`.
  - Goal: remove edge-line shortcut and raise anti-saturation selection pressure.
- [ ] B1g: Saturation-aware adversarial governor.
  - Add dynamic cap on rank-normalized adversarial term when `sat_excess` is high.
  - Suppress `adv_distress`-driven `w_adv` boosts during high-saturation phases.
  - Goal: keep adversarial signal informative without letting it overwhelm anti-saturation pressure.
- [ ] B1h: Discriminator regulation (soft rollback on prolonged low rfl).
  - Trigger when `real_fake_loss` stays below threshold for K log intervals.
  - Apply partial blend of D params/batch-stats toward anchor D state; reset D optimizer.
  - Goal: recover from stuck-strong D regimes without permanently shrinking model capacity.
- [ ] B1i: Reduced discriminator capacity (Option 2).
  - Lower discriminator base channels for BloodMNIST (`disc_features=32`).
  - Propagate discriminator feature width into belief-space topographic feature dim.
  - Goal: narrow D/G learning-speed gap by construction.
- [ ] B1j: Decaying discriminator-input noise schedule.
  - Increase D input noise early and decay slowly over training.
  - Apply the same schedule to both D-step updates and rollout/eval discriminator inputs.
  - Fix topographic KS broadcast edge-case at `disc_features=16` (`8*64=512`) so
    centroid tensors are never interpreted as per-pop scalars.
  - Goal: regularize early D shortcuts without permanently washing out signal.
- [ ] B2: CA blend run (`0.03`) with same seed/config.
- [ ] B3: CA blend run (`0.015`) with same seed/config.
- [ ] B4: T05 controller run (collapse-sensitive shape score + MI guard).
- [ ] B5: Shape-weight sweep on T05 (`shape_div_weight`: `0.08`, `0.12`, `0.16`).

Success criteria:

- Improve within-code structural diversity at 11k without destabilizing adv fitness.
- Avoid MI “improvement” that corresponds to repeated per-code prototypes.

## Phase C - Unsupervised Global Structure (No Labels in Loss)

- [ ] C1: Add unsupervised prototype alignment in discriminator feature space:
  - Build prototypes by online clustering of real data features.
  - Align code centroids to prototypes with permutation-invariant objective.
- [ ] C2: Keep conditional local diversity objective (T05) active.
- [ ] C3: Compare against T05-only to validate additive value.

Success criteria:

- Better code-level semantic separation than T05-only.
- Preserved within-code variation.
- No supervised labels used for optimization.

---

## 4) Suggested BloodMNIST Baseline Command

```bash
python examples/train_bloodmnist.py \
  --gpu-id='0,1' \
  --checkpoint-interval=5000 \
  --ca-blend-coeff=0.0 \
  --ca-blend-start-iter=4500 \
  --ca-blend-ramp-iters=3000 \
  --ca-blend-rfl-lo=0.43 \
  --ca-blend-rfl-hi=0.56 \
  --shape-div-weight=0.12
```

---

## 5) Paper-Oriented Tracking Rules

- Maintain `ABLATION_TRIAL_LOG.txt` after every code/config change.
- For each trial, record:
  - exact command/config
  - metric window summary (`1k-12k`)
  - checkpoint image notes (especially 11k panel)
  - interpretation + next step
- Keep `PLAN.md` synchronized when tasks are completed or pivoted.

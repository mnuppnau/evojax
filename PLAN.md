# EvoJAX Hyper-InfoGAN Baseline Plan

Branch: `claude/evolve-infogan-hypernetworks-s3r6d-67132f2-mnist-debug`  
Date: 2026-03-05

This plan replaces older BloodMNIST-first planning.  
We have pivoted back to MNIST digits to lock a stable, debuggable base architecture before re-expanding to BloodMNIST and other datasets.

## 1. Objective

- Establish one stable **foundational baseline** for HyperNetwork + PGPE InfoGAN.
- Use this as the reference for all future changes, ablations, and dataset transfers.
- Limit immediate experiments to small sweeps of learning rates and static fitness weights.

## 2. Current Baseline (Locked Structure)

- Dataset path: MNIST digits.
- Input/output scale: real data in `[0, 1]`, generator output via `sigmoid`.
- Latent structure: `z = [noise(62), discrete(10), continuous(2)]` (total 74).
- HyperNetwork:
  - Learned chunk embeddings.
  - Two hidden layers of width 48.
  - Correct per-layer chunk reconstruction (no padding-misaligned flatten split).
- Generator:
  - Split latent pathways:
    - `seed_base` sees **noise only**.
    - `seed_code` sees **discrete code only**.
    - `seed_cont` sees **continuous codes only**.
  - `code_seed_scale = 3.0` (amplified code signal).
- Discriminator/Q:
  - D backbone uses full-coverage SAME padding.
  - Q head is spatial (no GAP), with bottlenecked `1x1 -> 7x7 VALID` path.
  - Brightness guard before Q (`h - spatial_mean(h)`) to reduce trivial intensity watermarking.
- Dynamic feature plumbing:
  - `feature_dim` now follows policy output dynamically (`disc_feature_dim`).
  - No remaining hard-coded `256` assumptions in rollout carry or centroids.

## 3. Major Bugs Fixed (Pivot-Critical)

1. HyperNetwork chunk misalignment during parameter reconstruction.  
2. Dataset/output mismatch when returning to MNIST (`[-1,1]` vs `[0,1]`, `tanh` vs `sigmoid`).  
3. Discriminator/Q blind spots from VALID/stride geometry dropping edge information.  
4. Q-head GAP shortcut enabling steganographic code encoding.  
5. Latent leak where noise path consumed full `z`, allowing suppression of code effect.  
6. Hard-coded feature dimensions (`256`) causing shape/carry failures after D/Q changes.  
7. Remainder-code bias in latent batching and preview panel imbalance.

## 4. Current Run Readout (Stable Ground)

Source: `log/mnist_infogan/metrics.tsv`, latest around `25.6k` iters.

- `real_fake_loss`: mostly healthy (`~0.49` to `~0.58` in recent window).
- `fitness_adv` (`adv_avg`): stable around `-1.1` to `-1.3`.
- `fitness_mi`:
  - now positive in recent window (`mi_avg` up to ~`0.05`, `mi_max` up to ~`0.08`).
  - no late-run MI crash in this run.
- Separation/tightness:
  - `r_sense_avg` stabilized around `~0.14` to `~0.16`.
  - `r_intra_avg` remains non-collapsed.
- CA state:
  - baseline run is static-weight style with CA blend effectively off (`ca_blend=0.0`).

Image trajectory (`iteration-*.npy` snapshots):

- Digits remain recognizable with stable contrast.
- Diversity across codes is materially better than prior collapsed runs.
- Some residual artifacts and uneven code quality remain, but training is no longer in the previous catastrophic failure modes.

## 5. Baseline Configuration to Reuse

- Keep this architecture fixed while tuning:
  - split latent paths,
  - spatial Q head (no GAP),
  - brightness guard,
  - dynamic feature-dim wiring,
  - corrected chunk reconstruction.
- Keep CA blend off during baseline tuning (`ca_blend_coeff=0.0`).
- Keep morphology penalties off for MNIST baseline (`w_dark_range`, `w_center_edge_range`, `w_edge_dark_penalty`, `w_code_corr` at `0.0`).

## 6. Immediate Tuning Plan (Small, Controlled)

Phase T1: learning-rate micro-sweep (one variable at a time).

- `center_lr_gen`: test around current value (for example `0.0042`, `0.0048`, `0.0054`).
- `std_lr_gen`: test small neighborhood around current (`~0.05` to `~0.07`).
- Keep all fitness weights fixed during each LR sweep.

Phase T2: static fitness-weight micro-sweep.

- Anchor on current approximate regime:
  - `w_adv ~ 0.53`
  - `w_mi ~ 0.34`
  - `w_div ~ 0.16`
  - `w_sense ~ 0.18`
  - `w_intra ~ 0.05`
- Sweep one weight at a time with small deltas only.

Phase T3: select baseline checkpoint.

- Choose checkpoint by joint criteria:
  - stable `real_fake_loss` band,
  - positive/stable MI trend,
  - visible per-code digit differentiation,
  - minimal shortcut artifacts.

## 7. Transfer Rules for BloodMNIST and Future Datasets

- Reuse this exact architecture and bug-fixed plumbing as the default template.
- Change only dataset adapter settings first:
  - class count,
  - channel handling,
  - normalization/output activation pairing.
- Do not reintroduce architecture changes during initial transfer.
- Re-enable CA blend only after non-CA baseline is stable on target dataset.

## 8. Working Discipline

- One-variable changes only.
- Every run must log:
  - exact command,
  - checkpoint source,
  - metrics and KS TSVs,
  - fixed-interval image snapshots.
- Keep `PLAN.md` and `ABLATION_TRIAL_LOG.txt` updated after each configuration change.

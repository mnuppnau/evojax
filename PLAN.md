# EvoJAX Hyper-InfoGAN + Cultural Algorithms Plan (Branch `claude/evolve-infogan-hypernetworks-s3r6d-stable-67132f2`)

## Section A: Current Snapshot

### A.1 Research Direction

This branch is currently the best practical baseline for OrganSMNIST among the recent runs:

- It keeps discriminator learning healthy (`real_fake_loss` usually in the useful `0.4-0.6` band).
- It produces meaningfully better visual diversity/contrast than the recent unstable runs.
- It already contains the core CA structure (belief space + KS + CA-guided fitness shaping), but CA guidance is only partially functioning due to at least one critical bug (documented below).

This aligns with the medium-term goal:

- stabilize InfoGAN-style evolution on a harder dataset (OrganSMNIST),
- then transfer the CA mechanisms to IMDB/Transformer with minimal retuning.

### A.2 Current Architecture (as implemented on `67132f2`)

1. **Generator training**:
- HyperNetwork evolved with PGPE (`evojax/algo/pgpe_ca.py`).
- PGPE population default: `512`.
- HyperNetwork params: `33,344` (log output).
- Generator params: `188,889` (log output).

2. **Discriminator training**:
- Backprop with Adam `lr=1e-4`, `b1=0.5`, `b2=0.999` (`evojax/trainer.py`).
- D updates every `3` iterations before `1500`, then every iteration.
- D update accepted only when `real_fake_loss > 0.3`.

3. **Fitness components used in PGPE** (`evojax/algo/pgpe_ca.py`):
- `fitness_adv`, `fitness_mi`, `r_sense`, `pop_var`, `r_intra`, `cons_shortfall`, `normative_penalty`.
- Rank-normalization used before weighting.
- CA slope-driven weight modulation for `w_adv`, `w_div`, `w_sense`, `w_intra`.

4. **Belief space / CA components**:
- Domain, situational, historical, topographic, normative KS.
- Metric-history slopes (short/medium/long windows).
- CA gradient blend into PGPE gradient is now runtime-configurable:
  - `--ca-blend-coeff`
  - `--ca-blend-start-iter`
  - `--ca-blend-ramp-iters`
  - `--ca-blend-rfl-lo`, `--ca-blend-rfl-hi` (optional real/fake-loss gate band)

5. **Task**:
- OrganSMNIST, 11 discrete codes.
- Latent input: 63 noise + 11 one-hot code.

### A.3 What Is Working vs Not Working

**Working now**
- Training does not immediately collapse.
- Discriminator signal is generally in a healthy range.
- Generated samples show increased complexity vs recent unstable branches.

**Not yet solved**
- Convergence is still inconsistent in the critical adversarial phase.
- Disentanglement is improved but still partial.
- CA guidance is active, but net benefit is still phase-dependent and requires
  better objective alignment.
- MI improvement is still partially achieved via within-code prototype collapse
  (high `I(c;x)` with weak `I(z;x|c)`), which hurts OrganSMNIST realism.

---

## Section B: Empirical Summary (Current 12k Run)

Source: `log/organsmnist/OrganSMNIST.txt` and `iteration-*.npy` from this branch/run.

### B.1 Training Dynamics

Across 120 logged points (iterations `100..12000`):

- `real_fake_loss` mean `0.4725` (min `0.3036`, max `0.6259`).
- `102/120` logged points had `real_fake_loss` in `[0.4, 0.6]`.
- `fitness_adv max > -0.6` occurred `7/120` times (brief spikes, not persistent collapse).
- `r_sense avg` increased from early regime (~`0.03`) to later regime (~`0.078` on 7k-12k window).
- `r_intra avg` remained high overall (~`0.936` full run mean), with occasional dips.

Interpretation:

- This confirms your observation: this branch is materially healthier than the recent unstable runs.
- The adversarial game is active rather than frozen, and diversity signals are present.

### B.2 Image-Side Proxy Metrics (from `iteration-*.npy`)

Computed over saved snapshots `1000..12000`:

- Detail proxy (mean abs Laplacian): first-3 snapshots `0.2402` -> last-3 snapshots `0.5724`.
- Between-code variance proxy: first-3 `0.5390` -> last-3 `0.7052`.
- Intra/Between ratio: first-3 `0.571` -> last-3 `0.281` (better code separation trend).

Interpretation:

- Structural complexity and inter-code separation are improving through training.
- This branch is a valid foundation for stabilization + disentanglement work.

### B.3 Post-Phase-1 Ablation (6k, CA Blend ON vs OFF)

Comparison run window: `100..5900`.

- `real_fake_loss` in `[0.4, 0.6]`: OFF `0.475` vs ON `0.237`
- `real_fake_loss < 0.4`: OFF `0.492` vs ON `0.712`
- `fitness_mi` mean: OFF `-1.0598` vs ON `-2.3182`
- `r_sense` mean: OFF `0.0463` vs ON `0.0099`
- Laplacian detail proxy (2k-5k): OFF `0.2704` vs ON `0.0610`
- Between-code variance (2k-5k): OFF `0.8983` vs ON `0.1586`

Interpretation:

- For this branch/regime, early fixed CA blend (`0.05`) was too aggressive.
- Disabling CA blend improved discriminator balance and visual detail.
- CA should phase in later, not from iteration 0.

---

## Section C: Change History and What to Reuse Incrementally

### C.1 From `285f5df` -> `67132f2` (important deltas)

Main deltas observed in code history:

1. Early D throttling schedule (`d_freq=3` before 1.5k, then `1`).
2. D update acceptance changed to strict threshold (`real_fake_loss > 0.3` only).
3. CA metric history expanded (added `avg_r_intra`, `avg_fitness_adv` slopes).
4. Fitness formula added:
- `r_intra` reward term.
- `r_cons` floor penalty (`cons_shortfall`).
5. Code-pixel diversity made brightness-normalized in generator fitness path.

### C.2 What should be retried from newer branch, but incrementally

From `claude/evolve-infogan-hypernetworks-s3r6d` (large unstable integration), reintroduce only in controlled stages:

1. Better metric/KS logging (`metrics.tsv`, `ks_weights.tsv`) for diagnosis.
2. Runtime-controlled CA knobs (blend schedule, adaptive D controls), but one mechanism at a time.
3. Capacity upgrades (HN and Generator) only after CA correctness bugs are fixed and baseline is re-validated.

### C.3 What to defer

Per-layer multi-hypernetwork evolution should be deferred until:

- single-hypernetwork training is reliably stable on OrganSMNIST,
- CA guidance is validated as active and beneficial,
- ablations show capacity-limited failure rather than controller failure.

---

## Section D: Bug / Risk Audit (Current Branch)

Status note: the major Phase 1 correctness bugs listed below were fixed in-code;
they are retained here as audit history and regression checks.

### D.1 Critical (must fix first)

1. **CA guidance likely disabled by NaN entropy path**
- File: `evojax/algo/pgpe_ca.py:824`, `evojax/algo/pgpe_ca.py:825`
- Issue: entropy uses raw logits as if probabilities: `log(mean_disc_logit)`.
- Evidence from checkpoint (`checkpoint_latest.pkl`):
  - metric history entropy buffer contains all NaNs,
  - `ent_long` slope is NaN,
  - CA guidance output is non-finite,
  - `has_ca_data` gate evaluates false, so blend is skipped.
- Impact: belief-space guidance is not effectively influencing PGPE center/stdev.

2. **HyperNetwork chunk-ID truncation / representational collapse**
- File: `evojax/policy/convnet.py:153`, `evojax/policy/convnet.py:159`
- Issue: `N_CHUNKS` hard-coded to `100`, but actual max chunk id is `199` with current generator/chunk size.
- Effect: chunk IDs `>=100` map to all-zero one-hot; many chunks lose identity.
- Impact: weaker controllability and reduced expressive capacity, directly hurting image quality/disentanglement.

### D.2 High priority

3. **Hard-coded population reshape in PGPE tell**
- File: `evojax/algo/pgpe_ca.py:796`, `evojax/algo/pgpe_ca.py:797`
- Issue: forces `spreads` and `safety_ratios` to population `512`.
- Impact: non-general, brittle, blocks reproducible scaling and future task transfer.

4. **Normative clamp logic is overwritten**
- File: `evojax/algo/cultural/knowledge_sources.py:444` and `evojax/algo/cultural/knowledge_sources.py:457`
- Issue: clamped `elite_ratios_clamped` path is computed, then replaced by unclamped recomputation.
- Impact: normative KS behavior does not match intended robust safety handling.

5. **Latent task hard-codes feature width `256`**
- File: `evojax/task/latent.py:489`, `evojax/task/latent.py:490`
- Issue: reshapes historical centroids/velocity to fixed feature dim.
- Impact: breaks when discriminator feature dimension changes; blocks easy model scaling/transfer.

### D.3 Medium priority

6. **Trainer demo mode references undefined variable**
- File: `evojax/trainer.py:1075`
- Issue: `params` is undefined in demo branch.
- Impact: demo/test path is broken.

7. **KS scoring bias not parameterized**
- File: `evojax/algo/cultural/helper_functions.py:313`
- Issue: hard-coded `+0.897` bias in domain score.
- Impact: can dominate KS selection in opaque ways; should be explicit/configurable.

8. **Shape flattening obscures structure in sim manager**
- File: `evojax/sim_mgr.py:618` and `evojax/sim_mgr.py:643`
- Issue: safety/spread tensors are flattened/reduced then later re-shaped externally.
- Impact: reduces clarity, creates hidden coupling with hard-coded reshape logic.

---

## Section E: Incremental Execution Plan (Tasks + Subtasks)

## Phase 0: Lock Baseline + Instrumentation (no behavior change)

- [x] Task 0.1: Add structured metric logging for each training run.
  - [x] Write `metrics.tsv` with iteration-level summaries.
  - [x] Write `ks_weights.tsv` with active weights, slopes, and KS-selection stats.
  - [x] Keep existing text log for backward compatibility.

- [ ] Task 0.2: Add a tiny checkpoint-inspection utility.
  - [ ] Print NaN counts in metric buffers.
  - [ ] Print CA blend activation rate.
  - [ ] Print KS selection distribution.

Success criteria:
- Diagnostics can confirm whether CA guidance is actually active during a run.

## Phase 1: Correctness Fixes (minimal, isolated)

- [x] Task 1.1: Fix entropy path in CA history update.
  - [x] Convert logits to probabilities (`softmax`) before entropy.
  - [x] Guard with finite checks and fallback value.
  - [x] Verify checkpoint no longer accumulates NaN-driven CA disablement.

- [x] Task 1.2: Fix dynamic chunk dimensions in `ParameterAdapter`.
  - [x] Replace hard-coded `N_LAYERS/N_CHUNKS`.
  - [x] Derive from actual adapter maps.
  - [x] Validate no out-of-range chunk IDs in one-hot path.

- [x] Task 1.3: Remove hard-coded `512` from PGPE reshape path.
  - [x] Reshape based on runtime population and `n_classes`.
  - [x] Keep shape logic dynamic for spread/safety tensors.

- [x] Task 1.4: Fix normative clamp overwrite.
  - [x] Keep single clamped path.
  - [x] Preserve elite-safety computation on clamped ratios only.

- [x] Task 1.5: Remove hard-coded `256` in latent KS reshapes.
  - [x] Infer feature dimension dynamically via reshape.
  - [x] Remove fixed discriminator-feature assumption from latent task.

Phase 1 completion notes:
- CA slope computation now sanitizes NaNs so legacy checkpoints can recover without manual reset.
- CA guidance from the latest checkpoint is finite and blend-eligible again (`has_ca_data=True` in local checkpoint inspection).
- HyperNetwork input dimensionality increased (dynamic chunk encoding), so `num_params_hypernet` changed from `33344` to `37952` for this generator layout.
- `examples/train_organsmnist.py` now forces workspace-local imports to avoid accidental use of stale site-packages code.

Success criteria:
- Same config still runs.
- CA guidance finite and blend gate can turn on.
- No regressions in 2k smoke run.

## Phase 2: Re-validate `67132f2` behavior after bug fixes

- [ ] Task 2.1: Run 3 seeds to 12k with unchanged training knobs (except explicit CA-blend ablation).
  - [x] Seed-1 A/B completed (fixed CA blend `0.05` vs `0.0`) through ~6k.
  - [ ] Add seeds 2-3 for confirmation.
  - [ ] Compare `real_fake_loss` occupancy in `[0.4, 0.6]`.
  - [ ] Compare `fitness_adv max` excursions above `-0.6`.
  - [ ] Compare disentanglement proxies from `iteration-*.npy`.
  - [x] Added structured A/B comparer utility: `scripts/compare_metrics_tsv.py`.

- [ ] Task 2.2: Define pass/fail gate.
  - [ ] Pass if at least 2/3 seeds match or improve current baseline.
  - [ ] Fail => rollback only the offending fix set, not whole branch.

- [x] Task 2.4: Route 2 instrumentation for conditional diversity objective.
  - [x] Added `r_shape_div` in latent task as feature-space conditional diversity proxy (`Var[f_D(G(z,c)) | c]`).
  - [x] Propagated `r_shape_div` through `sim_mgr`, `trainer`, and `metrics.tsv`.
  - [x] Added `w_shape`/`shape_div_avg` diagnostics to `ks_weights.tsv`.

- [x] Task 2.5: Route 2 objective + Domain KS archive pivot.
  - [x] Added `rank_normalize(r_shape_div) * w_shape` to PGPE fitness blend.
  - [x] Extended Domain KS archive schema with `r_shape_div`.
  - [x] Extended Domain KS Pareto axes to `[|adv|, |mi|, -shape_div, |entropy|]`.
  - [x] Updated Domain KS stagnation selector to prefer highest `r_shape_div`.

- [x] Task 2.6: Route 2 collapse-sensitive refinement (T05).
  - [x] Replaced mean-only shape reward with weighted min+mean score:
    `shape_score = 0.7 * shape_div_min + 0.3 * shape_div_mean`.
  - [x] Added `r_shape_div_min` to rollout/trainer logging (`metrics.tsv`).
  - [x] Added shape-div trend to CA metric history and slopes (`shape_short`, `shape_med`).
  - [x] Made `w_shape` CA-adaptive with strict clamp (`0.02..0.20`).
  - [x] Added MI guard: when `shape_div_min` drops below target, reduce `w_mi`.

- [x] Task 2.3: Add delayed/ramped CA blend control path.
  - [x] `PGPE_CA` accepts CA schedule/gate args.
  - [x] `Trainer` passes runtime `real_fake_loss` to solver.
  - [x] `train_organsmnist.py` exposes CLI knobs and logs schedule.

Current default schedule:
- `--ca-blend-coeff=0.05`
- `--ca-blend-start-iter=3000`
- `--ca-blend-ramp-iters=2000`
- `--ca-blend-rfl-lo=0.40`
- `--ca-blend-rfl-hi=0.58`

## Phase 3: Controlled Stabilization Knobs (one-at-a-time experiments)

- [ ] Task 3.1: D-schedule A/B tests (single variable changes).
  - [ ] A: current schedule (`3->1`, threshold `0.3`).
  - [ ] B: phase schedule (`3->2->1`) with same threshold.
  - [ ] C: same schedule with threshold sweep (`0.30`, `0.33`, `0.36`).

- [ ] Task 3.2: CA blend ramp tests.
  - [ ] Start iteration sweep (`0`, `500`, `1000`).
  - [ ] Blend max sweep (`0.01`, `0.025`, `0.05`).
  - [ ] Keep all other knobs fixed.

- [ ] Task 3.3: Weight adaptation logic tests.
  - [ ] Compare current distress logic vs inverted logic from later branch.
  - [ ] Keep D schedule fixed while testing this.

- [x] Task 3.4: Add conservative MI/sense distress adaptation.
  - [x] MI distress: increase `w_mi` when short/medium MI slopes are negative.
  - [x] Sense deficit: increase `w_sense` when `r_sense` remains below target.
  - [x] Keep bounded clamps (`w_mi <= 0.35`, `w_sense <= 0.25`) to avoid instability.
  - [ ] Validate through 12k A/B against prior delayed-CA run.

Success criteria:
- Reduce frequency of bad adversarial dips without flattening image detail.

## Phase 4: Image Quality + Disentanglement (capacity changes only after stability)

- [ ] Task 4.1: Generator capacity ladder.
  - [ ] Small conv-depth increase first.
  - [ ] Then feature-width increase.
  - [ ] Re-evaluate chunk-id distribution after each change.

- [ ] Task 4.2: HyperNetwork capacity ladder.
  - [ ] Expand hidden width incrementally.
  - [ ] Monitor sensitivity of `fitness_adv` and `real_fake_loss`.

- [ ] Task 4.3: Disentanglement-focused evaluation.
  - [ ] Track within-code diversity and between-code separation explicitly.
  - [ ] Add code-conditioned fixed-z panel and fixed-code varying-z panel outputs.

---

## Section F: CA Generalization Plan for IMDB + Transformer

Goal: make CA task-agnostic, with task-specific adapters only.

### F.1 CA core vs task adapters

- [ ] Separate CA core signals from image-only signals.
- [ ] Define a generic metric interface:
  - `quality_signal`
  - `info_signal`
  - `separation_signal`
  - `intra_variation_signal`
  - `safety_signal`
- [ ] Map OrganSMNIST metrics and future IMDB/Transformer metrics to this interface.

### F.2 Remove image hard-codings

- [ ] Eliminate fixed constants (`11`, `28x28`, `256`) from CA-relevant paths.
- [ ] Centralize task metadata (num classes, feature dims, batch struct) in one config object.

### F.3 Transformer readiness (without committing to per-layer HN yet)

- [ ] Keep single-HN evolution as baseline path.
- [ ] Add optional module for per-block parameter grouping later.
- [ ] Gate per-layer/per-block evolution behind demonstrated need.

Recommendation:
- Do **not** make per-layer hypernetwork evolution a prerequisite for IMDB transition.
- First prove CA-correctness + stable adaptation on one robust baseline pipeline.

---

## Section G: Immediate Next Actions

1. Run 12k Route 2 validation with current conservative CA settings:
   - `python examples/train_organsmnist.py --gpu-id='0,1' --checkpoint-interval=5000 --ca-blend-coeff=0.015 --ca-blend-start-iter=4500 --ca-blend-ramp-iters=3000 --ca-blend-rfl-lo=0.43 --ca-blend-rfl-hi=0.56 --shape-div-weight=0.12`
2. Compare new run vs prior controls with structured logs:
   - `python scripts/compare_metrics_tsv.py --control <control_metrics.tsv> --treatment <route2_metrics.tsv> --iter-min 1000 --iter-max 12000`
   - Primary: `adv_max_mean`, `real_fake_loss in [0.4,0.6]`, `mi_avg_mean`, `r_shape_div_mean`, `r_shape_div_min_mean`.
3. If `r_shape_div` improves but `mi_avg` regresses hard:
   - lower `w_shape` first (before touching D schedule),
   - keep Domain KS 4-axis archive in place,
   - re-test with one-variable change only.
4. Keep `ABLATION_TRIAL_LOG.txt` updated after each code/config change so image checkpoints (11k) map to exact trial conditions.

---

## Section H: Route 2 Pivot (I(c;x) + I(z;x|c))

### H.1 Why we pivoted

Recent A/B runs showed that stronger CA blend often improved D/G balance (`real_fake_loss`)
but produced weaker within-code variation. This is consistent with MI-only pressure
favoring code prototypes: codes become distinguishable, while variation from `z` collapses.

### H.2 Core hypothesis

For OrganSMNIST (and likely many medical datasets), useful disentanglement requires:

- high `I(c;x)` (code distinguishability), and
- high `I(z;x|c)` (meaningful within-code shape variation).

Optimizing only the first can improve MI numerically while harming visual realism/diversity.

### H.3 Code changes made for Route 2

1. Added `r_shape_div` conditional diversity proxy:
   - `evojax/task/latent.py` computes per-code feature variance in discriminator feature space.
2. Threaded `r_shape_div` through the training stack:
   - `evojax/sim_mgr.py` rollout carry/aggregation/returns.
   - `evojax/trainer.py` logging + TSV metrics (`r_shape_div_avg`).
3. Updated optimizer objective and CA archive:
   - `evojax/algo/pgpe_ca.py` adds `w_shape * rank_normalize(r_shape_div)`.
   - `evojax/algo/cultural/knowledge_sources.py` adds Domain KS `r_shape_div` archive field.
   - Domain Pareto front now includes `-shape_div` (maximize shape diversity under minimization sorter).
   - `examples/train_organsmnist.py` exposes `--shape-div-weight` for A/B sweeps without code edits.
4. Added collapse-sensitive controller path:
   - shape reward now uses min+mean blend to punish worst-code collapse,
   - CA tracks shape-div slopes for adaptive `w_shape`,
   - MI guard prevents MI-only prototype shortcuts when shape minima collapse.
4. Updated experiment tooling:
   - `scripts/compare_metrics_tsv.py` now reports `r_shape_div_mean`.

### H.4 Route 2 success criteria

- `real_fake_loss` occupancy in `[0.4, 0.6]` remains near/above current baseline.
- `mi_avg` does not improve solely via visual prototype collapse.
- `r_shape_div_avg` rises with clear within-code structural variability in saved panels.
- Domain KS winner distribution should show less single-mode lock-in over long runs.

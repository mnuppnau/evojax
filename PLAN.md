# EvoJAX Hyper-InfoGAN BloodMNIST Plan

Branch: `claude/evolve-infogan-hypernetworks-s3r6d-67132f2-blood-baseline`  
Date: 2026-03-19

This plan tracks the restored BloodMNIST baseline and the completed / next CA ablations.
The immediate goal is not to chase a perfect BloodMNIST run. The goal is to lock a defensible non-CA baseline, measure what the current CA integration actually helps with, and then move to a morphology-aware controller.

## 1. Objective

- Keep the current HyperNetwork / Generator / Discriminator architecture fixed.
- Lock the restored BloodMNIST baseline rerun as the new `BLD-B00` reference.
- Run CA ablations one layer at a time so each effect is measurable.
- Determine whether the current CA helps primarily with:
  - adversarial stability
  - exploration vs commitment timing
  - escaping prototype hardening
  - or actual hematologic morphology disentanglement
- Only after that, move to morphology-aware controller changes.

## 2. Locked Baseline Architecture

- Dataset:
  - BloodMNIST converted to grayscale
  - real data in `[0, 1]`
  - `8` discrete classes
- Generator output:
  - `sigmoid`
- Latent:
  - `z = [noise(62), discrete(8), continuous(2)]`
- HyperNetwork:
  - learned chunk embeddings
  - hidden widths `48 -> 48`
  - correct per-layer chunk reconstruction
- Generator:
  - split latent paths:
    - `seed_base(noise only)`
    - `seed_code(discrete only)`
    - `seed_cont(continuous only)`
  - `code_seed_scale = 3.0`
- Discriminator / Q:
  - SAME-padded full-coverage backbone
  - spatial Q head without GAP
  - brightness guard before Q
- Plumbing:
  - dynamic discriminator feature dimension
  - no hard-coded feature-size assumptions in rollout / centroid state

## 3. Current BloodMNIST Baseline Read

Restored baseline rerun:
- CA disabled: `--ca-blend-coeff=0.0`
- static phased weights enabled
- current checkpoint window discussed: `~165k`

Baseline command family:

```bash
python examples/train_bloodmnist.py \
  --gpu-id='0,1' \
  --checkpoint-interval=5000 \
  --ca-blend-coeff=0.0 \
  --static-fitness-weights \
  --static-mi-sense-ramp \
  --static-w-div=0.18 \
  --static-w-sense=0.08 \
  --static-w-intra=0.03 \
  --static-div-ramp-target=0.16 \
  --static-div-ramp-start-iter=12000 \
  --static-div-ramp-end-iter=28000 \
  --static-sense-ramp-target=0.14 \
  --static-sense-ramp-start-iter=12000 \
  --static-sense-ramp-end-iter=28000 \
  --static-intra-ramp-target=0.04 \
  --static-intra-ramp-start-iter=12000 \
  --static-intra-ramp-end-iter=28000 \
  --disc-features=48
```

Current interpretation of the restored baseline:
- late-window metrics are stable and match the original `BLD-B00` regime closely:
  - `150k-160k`:
    - `mi_avg = 0.0859`
    - `r_sense_avg = 0.1404`
    - `r_intra_avg = 0.7493`
    - `real_fake_loss = 0.5818`
    - `stdev_mean = 0.0255`
    - `code_proto_corr_avg = 0.7032`
  - `160k-165.3k`:
    - `mi_avg = 0.0863`
    - `r_sense_avg = 0.1378`
    - `r_intra_avg = 0.7394`
    - `real_fake_loss = 0.5805`
    - `stdev_mean = 0.0252`
    - `code_proto_corr_avg = 0.7029`
- images are clean and smooth:
  - circular cell bodies
  - no major edge-connection artifact
- disentanglement remains partial and still dominated by easy axes:
  - cell size
  - darkness / contrast
  - nucleus placement / orientation

Working conclusion:
- this rerun is a valid replacement for the overwritten original `BLD-B00`
- it remains the correct reference baseline for all BloodMNIST CA comparisons
- it is not yet a morphology-disentangled solution

## 4. What The Current CA Is Likely To Help With

The current CA is most likely to help with global training dynamics, not directly with hematologic semantics.

Most plausible benefits of the current CA:
- adjust pressure between `adv / mi / div / sense / intra` when training drifts
- delay premature prototype hardening
- preserve exploration longer when the run begins collapsing onto cheap axes
- help PGPE escape late plateaus or duplicate prototype basins

What the current CA probably does **not** yet know how to do well:
- distinguish hematologic morphology from cheap shortcuts such as:
  - cell size
  - nucleus angle / clock-face rotation
  - global darkness differences
- reward true morphology-specific factors such as:
  - nucleus-to-cytoplasm ratio
  - lobulation / segmentation
  - contour roughness
  - chromatin / texture structure

Therefore:
- current CA may improve disentanglement indirectly
- current CA alone is unlikely to guarantee hematologic morphology disentanglement
- if controller-only ablations plateau on size/orientation codes, morphology-aware signals will still be needed

## 4.1 CA Design Principle

The intended CA design is:
- general at the InfoGAN optimization-framework level
- instantiated at the dataset-semantic level

What should remain general across datasets:
- belief-space structure
- the roles of Domain / Normative / Historical / Situational / Topographic KS
- controller logic for:
  - exploration vs exploitation
  - collapse detection
  - plateau rescue
  - adversarial-health regulation

What should become dataset-specific:
- the semantic signals that tell the controller what counts as meaningful disentanglement
- the proxy measurements used to detect shortcut solutions
- the acceptable ranges or soft bounds for those semantic factors

Working interpretation for this project:
- Domain KS should be the semantic interpreter of the dataset
- Normative KS should convert those learned regularities into soft acceptable regions
- Historical KS should preserve earlier diverse states that can rescue the run from shortcut traps
- Topographic KS should track output geometry, but not define semantics by itself

For BloodMNIST specifically:
- a fully dataset-agnostic CA is not the target
- the target is a framework-general CA whose Domain and Normative sources can be instantiated with hematologic morphology signals
- this is the cleanest path toward a controller that is reusable across InfoGAN tasks without pretending that all datasets share the same semantics

## 4.2 Future Transfer To IMDb Hard-Attention

The same CA principle should transfer to a future HyperNetwork / PGPE hard-attention system for IMDb, but again only at the framework level.

What should transfer unchanged:
- belief-space mechanics
- KS roles
- explore / exploit / rescue logic
- generic optimization-health monitoring

What should be re-instantiated for hard attention:
- what counts as a shortcut in the attention policy
- what counts as meaningful semantic separation
- what soft bounds should constrain attention behavior

Working interpretation for IMDb:
- Domain KS should learn whether attention selections are focusing on sentiment-bearing spans or cheap shortcuts such as punctuation, position bias, or highly frequent tokens
- Normative KS should express soft acceptable regions for attention behavior, such as sparsity, continuity, class-conditional diversity, or token-coverage ranges
- Historical KS should preserve pre-collapse attention policies before the run narrows into trivial token-selection patterns
- Topographic KS should track output / attention geometry, but should not define semantic usefulness by itself

This keeps the CA general as an optimizer-controller while allowing the semantic interface to change from morphology on BloodMNIST to attention structure on IMDb.

## 5. Ordered CA Ablation Ladder

### BLD-B00 — Locked non-CA baseline
Purpose:
- anchor run for all BloodMNIST comparisons

Status:
- restored / reference baseline
- rerun at `~165k` matches the original `BLD-B00` behavior closely

Question:
- how far can the fixed phased-weight baseline go without CA?

### BLD-A01 — Dynamic fitness weighting only
Purpose:
- isolate adaptive fitness weighting without CA gradient guidance

Configuration change:
- turn off `--static-fitness-weights`
- keep `--ca-blend-coeff=0.0`
- keep architecture, LR schedule, and `disc-features` fixed

Question:
- does adaptive weighting improve late morphology separation, or does it mostly improve health / exploration timing?

Result:
- stable, but negative as a disentanglement ablation
- by `~100k`, the run remained behind the static baseline on morphology separation
- main behavior:
  - good adversarial health
  - clean cells
  - persistent one-template / size-orientation coding
- observed controller tendency:
  - `w_adv` stayed high
  - `w_mi` stayed too low
  - `w_div` stayed suppressed
  - `w_intra` stayed too high
- conclusion:
  - dynamic weighting alone acts as a conservative stability controller
  - it does not adequately prioritize BloodMNIST disentanglement

### BLD-A02 — CA gradient blend only
Purpose:
- isolate KS-guided gradient blending while leaving the baseline fitness schedule fixed

Configuration change:
- keep the baseline static schedule
- set `--ca-blend-coeff=0.015`
- keep blend ramp / RFL gating as currently implemented

Question:
- can KS guidance alone push PGPE off the current size/orientation attractors without changing the base fitness landscape?

Status:
- completed
- negative ablation

Result:
- CA blend alone did not improve BloodMNIST disentanglement over `BLD-B00`
- typical late-window behavior (`~100k-119k`):
  - `mi_avg ~ 0.003`
  - `r_sense_avg ~ 0.114`
  - `r_intra_avg ~ 0.660`
  - `real_fake_loss ~ 0.518`
  - `code_proto_corr_avg ~ 0.612`
- interpretation:
  - the lower prototype correlation was misleading because the model still used the orientation / clock-face shortcut
  - CA blend alone was semantically ineffective even when image quality remained usable

### BLD-A03 — Full current CA
Purpose:
- test the combined controller: adaptive weighting + CA blend

Configuration change:
- disable `--static-fitness-weights`
- set `--ca-blend-coeff=0.03`
- keep all other architecture and LR settings fixed

Question:
- does the full current CA outperform either component alone, or does it simply add variance?

Status:
- completed
- negative ablation

Result:
- full current CA performed worse than `BLD-B00`, `BLD-A01`, and `BLD-A02`
- typical late-window behavior (`~120k-128k`):
  - `mi_avg ~ -0.101`
  - `r_sense_avg ~ 0.069`
  - `r_intra_avg ~ 0.850`
  - `real_fake_loss ~ 0.443`
  - `code_proto_corr_avg ~ 0.632`
  - `stdev_mean ~ 0.032`
- observed controller behavior:
  - `w_adv` remained high (`~0.62`)
  - `w_mi` remained too low (`~0.27`)
  - `w_div` stayed suppressed (`~0.12`)
  - `w_intra` stayed too high (`~0.10`)
  - `ca_grad_norm` dominated `reinforce_grad_norm`
- interpretation:
  - the current CA signal is too strong relative to the semantic usefulness of its inputs
  - full current CA amplifies the wrong controller priors instead of helping PGPE escape the BloodMNIST shortcut

### BLD-A04 — Morphology-aware CA
Purpose:
- instantiate the CA with BloodMNIST-specific semantic signals
- move from shortcut detection alone to biology-aware rescue selection

Required code work:
- A04a (implemented / completed negative ablation):
  - compute `proto_angle_spread` from per-code prototype dark-mass centroids
  - extend CA metric history with:
    - `morph_dark_range`
    - `code_proto_corr`
    - `proto_angle_spread`
  - detect a semantic shortcut when:
    - MI is present
    - prototype correlation stays high
    - angle spread is non-trivial
    - morphology diversity remains weak
  - concrete rescue action:
    - reduce Domain KS confidence
    - boost Historical KS and Topographic KS guidance
    - apply a modest temporary stdev exploration bump
  - keep fitness terms unchanged
  - keep Normative KS unchanged
- A04a result:
  - detector worked mechanically
  - orientation-heavy separation was reduced
  - but the run still converged toward smoother shared-template cells rather than better hematologic morphology
  - conclusion:
    - shortcut detection alone is not enough
    - the controller needs constructive biological signals
- A04b (implemented / completed mixed-negative ablation):
  - add biological morphology metrics:
    - `nuc_cell_ratio_range`
    - `nuc_eccentricity_range`
  - extend CA metric history with those biology metrics
  - extend Historical KS with a biology-aware rescue score
  - when a semantic trap is active:
    - select Historical rescue candidates by biology score instead of entropy alone
  - log biology-aware controller diagnostics:
    - `nuc_ratio_latest`
    - `nuc_ecc_latest`
    - `bio_score_latest`
- A04b result:
  - the run produced smoother images and some nucleus variation
  - but whole-cell shape degraded late and the run remained behind `BLD-B00`
  - working conclusion:
    - constructive nucleus signals alone are not enough
    - the controller also needs explicit cell-body integrity signals
- A04c (implemented / completed partial-positive ablation):
  - add cell-body integrity metrics:
    - `cell_circularity`
    - `cell_area_var`
    - `nucleus_offset`
  - extend CA metric history and diagnostics with those signals
  - fold those signals into semantic trap detection
  - extend Historical rescue scoring so rescue favors:
    - stronger whole-cell integrity
    - better nucleus placement inside the cell body
  - observed outcome through `~195k`:
    - nuclei are more clearly distinguished than in `A04a/A04b`
    - semantic trap activity is effectively suppressed
    - prototype correlation is modestly lower than `BLD-B00`
    - but `mi_avg` remains weak/noisy and adversarial health remains behind `BLD-B00`
    - cell-body quality still lags the baseline
  - working conclusion:
    - whole-cell integrity signals improved controller perception
    - but rescue-only controller changes are still not enough
    - the next step is to let Normative KS preserve acceptable cell-body structure
- A04d (implemented / completed mixed-result ablation):
  - keep the baseline run schedule fixed
  - move the most reliable A04c morphology signals into Normative KS soft bounds:
    - `cell_circularity`
    - `nucleus_offset`
    - optional `cell_area_var` stability bound
  - use those bounds as controller-side acceptance regions, not as a new external fitness term
  - implemented controller behavior:
    - Normative KS now tracks soft bounds for circularity, area stability, and nucleus offset
    - center guidance penalizes Domain KS confidence when the current run violates those bounds
    - center guidance boosts Historical/Situational rescue under morphology violations
    - stdev guidance is mildly damped when morphology violates the learned bounds
  - observed outcome through `~200k`:
    - adversarial health and MI were modestly better than `A04c`
    - semantic shortcut pressure remained low
    - but morphology quality was still not better than `BLD-B00`
    - learned normative bounds stayed too permissive to force stronger cell-body integrity
  - intended effect:
    - preserve coherent cell bodies while the controller continues pushing nucleus-level variation
    - reduce late drift into smooth but structurally weak cells
  - working conclusion:
    - soft normative bounds helped stabilize the run somewhat
    - but the current bound-learning rule mostly ratified the existing mediocre elite pool
    - if CA work continues, the next change should tighten how Normative KS learns or sets those bounds
- A04e (implemented / next run):
  - keep the A04d controller structure fixed
  - tighten Normative KS learning with a warmup-gated asymmetric ratchet:
    - tighten quickly when elite morphology improves
    - relax extremely slowly after warmup
  - use actual iteration `t` for the warmup gate
  - add deadbands so bounds do not ratchet on quantile noise
  - intended effect:
    - preserve the best morphology standards discovered so far
    - stop Normative KS from trailing the current mediocre elite pool

Recommended A04d run mode:
- keep the static baseline schedule fixed
- enable CA blend
- do not enable adaptive weighting yet

Question:
- can Normative KS soft bounds turn the A04c partial positive into a controller that preserves cell-body integrity while improving morphology disentanglement?

Design note:
- this phase is not a shift away from a general CA
- it is the intended dataset-instantiation phase of a framework-general controller
- the Domain KS should learn which morphology factors matter
- the Normative KS should then express soft bounds / acceptable regions for those factors

## 6. Evaluation Protocol For Every BloodMNIST Run

Fixed checkpoints to compare:
- `20k`
- `60k`
- `100k`
- `150k`
- `164k/165k` for the restored baseline reference

Metrics to record side-by-side:
- `mi_avg`
- `mi_max`
- `r_sense_avg`
- `r_intra_avg`
- `real_fake_loss`
- `spread_avg`
- `code_proto_corr_avg`
- `morph_dark_range_avg`
- `morph_center_edge_range_avg`
- `proto_angle_spread_avg`
- `nuc_cell_ratio_range_avg`
- `nuc_eccentricity_range_avg`
- `cell_circularity_avg`
- `cell_area_var_avg`
- `nucleus_offset_avg`

Image questions:
- Are edge artifacts absent?
- Are codes still separating mostly by size / darkness / nucleus angle?
- Do multiple codes still look like the same template rotated around the center?
- Are any codes beginning to differ by more meaningful morphology?
- Is within-code variation still coherent?

Primary success criterion for CA runs:
- lower prototype correlation and visibly more morphology-specific differences than `BLD-B00`, without destabilizing adversarial health

Secondary success criterion:
- better or equal image quality with the same clean circular-cell baseline look

Failure criterion:
- CA increases MI or `r_sense` while images still reduce to one cell template with orientation-only variation

## 7. Next Actions

1. Archive the restored `BLD-B00` rerun at:
   - `20k`
   - `60k`
   - `100k`
   - `150k`
   - `164k/165k`
2. Treat `BLD-A01`, `BLD-A02`, and `BLD-A03` as completed controller-only ablations.
3. Treat `BLD-A04a` as a completed negative-but-informative ablation:
   - trap detection worked
   - rescue leverage was too weak without constructive biology signals
4. Treat `BLD-A04b` as a mixed-negative ablation:
   - some nucleus variation improved
   - whole-cell integrity regressed
5. Treat `BLD-A04c` as a completed partial-positive ablation:
   - nucleus-level variation improved
   - semantic trap remained suppressed
   - but MI and adversarial health still lagged `BLD-B00`
   - cell-body quality still needs Normative protection
6. Treat `BLD-A04d` as a completed mixed-result ablation:
   - slightly healthier than `A04c`
   - but still not competitive with `BLD-B00`
   - learned Normative KS bounds were too permissive
7. Run `BLD-A04e` next:
   - same A04d configuration
   - stricter Normative KS learning via warmup-gated asymmetric ratchet
   - no new fitness terms
8. If BloodMNIST CA work continues beyond `A04e`, the next change should be stricter Normative KS priors or elite selection:
   - tighter elite selection or stronger priors for `cell_circularity` / `nucleus_offset`
   - possibly per-code norms only after a cleaner global morphology controller exists
9. Preserve the same CA design principle for future IMDb hard-attention work:
   - general controller framework
   - task-specific semantic interface

## 8. Working Discipline

- One controller layer at a time.
- Keep architecture fixed during CA ablations.
- Keep the same logging/checkpoint cadence across runs.
- Every accepted run change must be reflected in:
  - `PLAN.md`
  - `ABLATION_TRIAL_LOG.txt`
- Controller-only BloodMNIST ablations are complete.
- Next BloodMNIST controller changes must be morphology-aware, not just another generic controller reweighting.

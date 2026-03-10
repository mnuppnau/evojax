# EvoJAX Hyper-InfoGAN BloodMNIST Plan

Branch: `claude/evolve-infogan-hypernetworks-s3r6d-67132f2-blood-baseline`  
Date: 2026-03-10

This plan tracks the current BloodMNIST baseline and the next CA ablations.
The immediate goal is not to chase a perfect BloodMNIST run. The goal is to lock a defensible non-CA baseline, then measure what the current CA integration actually helps with.

## 1. Objective

- Keep the current HyperNetwork / Generator / Discriminator architecture fixed.
- Finish characterizing the current BloodMNIST baseline run with CA disabled.
- Run CA ablations one layer at a time so each effect is measurable.
- Determine whether the current CA helps primarily with:
  - adversarial stability
  - exploration vs commitment timing
  - escaping prototype hardening
  - or actual hematologic morphology disentanglement
- Only after that, decide whether morphology-aware controller changes are needed.

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

Current baseline run:
- CA disabled: `--ca-blend-coeff=0.0`
- static phased weights enabled
- current checkpoint window discussed: `~58k`

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

Current interpretation of the baseline:
- good adversarial health
- clean circular cell bodies
- no major edge-connection artifact
- partial disentanglement
- current code separation is dominated by easy axes:
  - cell size
  - darkness / contrast
  - nucleus placement / orientation
- current controller-free baseline is already substantially cleaner than older BloodMNIST runs

Working conclusion:
- this baseline is valid
- it is not yet a morphology-disentangled solution
- it is good enough to anchor CA ablations

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

## 5. Ordered CA Ablation Ladder

### BLD-B00 — Locked non-CA baseline
Purpose:
- anchor run for all BloodMNIST comparisons

Status:
- active / reference baseline
- continue to at least `100k` before judging ceiling

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

### BLD-A02 — CA gradient blend only
Purpose:
- isolate KS-guided gradient blending while leaving the baseline fitness schedule fixed

Configuration change:
- keep the baseline static schedule
- set `--ca-blend-coeff=0.015`
- keep blend ramp / RFL gating as currently implemented

Question:
- can KS guidance alone push PGPE off the current size/orientation attractors without changing the base fitness landscape?

### BLD-A03 — Full current CA
Purpose:
- test the combined controller: adaptive weighting + CA blend

Configuration change:
- disable `--static-fitness-weights`
- set `--ca-blend-coeff=0.03`
- keep all other architecture and LR settings fixed

Question:
- does the full current CA outperform either component alone, or does it simply add variance?

### BLD-A04 — Morphology-aware CA (future code task)
Purpose:
- only run this if BLD-A01 through BLD-A03 still separate mostly by size / orientation

Required code work:
- feed morphology-aware signals into the controller, not just generic GAN health
- candidate signals already logged in the pipeline:
  - `code_proto_corr`
  - `morph_dark_range`
  - `morph_center_edge_range`
- likely need additional morphology metrics later:
  - nucleus-body ratio
  - lobulation / connected components
  - radial intensity profile
  - texture / granularity descriptors

Question:
- once the controller sees morphology-related failure modes explicitly, can it push separation beyond the current clock-face / size shortcuts?

## 6. Evaluation Protocol For Every BloodMNIST Run

Fixed checkpoints to compare:
- `20k`
- `60k`
- `100k`
- `150k` if the run remains healthy

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

1. Continue `BLD-B00` to at least `100k` unless adversarial health breaks.
2. Archive `BLD-B00` metrics and snapshots at `20k / 60k / 100k`.
3. Run `BLD-A01` next:
   - dynamic fitness weighting only
   - no gradient blend
4. Only after `BLD-A01` is complete, run `BLD-A02` and `BLD-A03`.
5. Do not change architecture or dataset preprocessing during this controller ablation phase.

## 8. Working Discipline

- One controller layer at a time.
- Keep architecture fixed during CA ablations.
- Keep the same logging/checkpoint cadence across runs.
- Every accepted run change must be reflected in:
  - `PLAN.md`
  - `ABLATION_TRIAL_LOG.txt`
- Do not add morphology penalties or new KS inputs until the controller-only ablations are measured cleanly.

# EvoJAX Hyper-InfoGAN Baseline Plan

Branch: `claude/evolve-infogan-hypernetworks-s3r6d-67132f2-mnist-debug`  
Date: 2026-03-06

This plan now tracks the MNIST-first baseline program.  
The immediate goal is not broader ablation coverage; it is to lock one reliable MNIST configuration and only then reuse it for BloodMNIST and other datasets.

## 1. Objective

- Keep the now-stable MNIST architecture fixed.
- Use the archived stable MNIST run as the reference pack.
- Use one controlled weight tweak at a time from that reference.
- Do not broaden the search again until we know whether the `w_sense` adjustment actually helps semantic code separation.

## 2. Locked Architecture

- Dataset/output scale:
  - MNIST real data in `[0, 1]`
  - generator output via `sigmoid`
- Latent:
  - `z = [noise(62), discrete(10), continuous(2)]`
- HyperNetwork:
  - learned chunk embeddings
  - hidden widths `48 -> 48`
  - correct chunk reconstruction
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
  - no hard-coded `256` assumptions

## 3. Baseline Reference Pack

Reference run: archived stable MNIST run (`R00` in `ABLATION_TRIAL_LOG.txt`).

Reference command family:

```bash
python examples/train_mnist_infogan.py \
  --gpu-id='0,1' \
  --checkpoint-interval=5000 \
  --ca-blend-coeff=0.0 \
  --static-fitness-weights
```

Important nuance:
- This was legacy-static behavior, not true static.
- MI and sense still ramped over time.

Effective reference regime:
- `w_adv = 0.53`
- `w_div = 0.16`
- `w_intra = 0.05`
- `w_mi = 0.30 -> 0.50` (gated ramp)
- `w_sense = 0.14 -> 0.34` (gated ramp)
- `center_lr = 0.0048`
- `std_lr = 0.062`
- `init_std = 0.032`

Known reference weakness:
- late semantic aliasing across a few code columns
- especially in the `1 / 4 / 8 / 9` family

## 4. Current Active Run

Active run: `B01` in `ABLATION_TRIAL_LOG.txt`.

Purpose:
- test only one hypothesis:
  - slightly lower early separation pressure should reduce style-splitting into duplicate code columns

Run source:
- resumed from `checkpoint_10000.pkl` from the archived stable run

Current configuration:

```bash
python examples/train_mnist_infogan.py \
  --gpu-id='0,1' \
  --checkpoint-interval=5000 \
  --resume-from=log/mnist_infogan/checkpoints/checkpoint_10000.pkl \
  --ca-blend-coeff=0.0 \
  --static-fitness-weights \
  --static-mi-sense-ramp \
  --static-w-sense=0.10
```

Interpretation of this setup:
- architecture unchanged
- PGPE learning rates unchanged
- only `w_sense` base was reduced
- MI/sense ramp intentionally preserved to reproduce the old stable dynamics as closely as possible

## 5. Current Readout

Source: `log/mnist_infogan/metrics.tsv`, current window around `10.4k-12.3k`.

- `real_fake_loss`: `0.522 - 0.595`, last `0.590`
- `adv_avg`: `-1.211 - -0.920`, last `-0.946`
- `mi_avg`: `-0.025 - 0.024`, last `0.002`
- `mi_max`: `0.030 - 0.051`, last `0.045`
- `r_sense_avg`: `0.109 - 0.162`, last `0.146`
- `r_intra_avg`: `0.461 - 0.967`, last `0.811`
- `spread_avg`: `0.014 - 0.039`, last `0.026`

Early image read:
- the early straight-`1` / slanted-`1` split did not stay fixed
- the slanted-`1` column started moving toward `2`
- that is the main behavioral change to verify against the archived reference

## 6. Comparison Schedule

Compare `B01` against the archived stable reference at:

- absolute iterations:
  - `12k`
  - `15k`
  - `20k`
  - `25k`
  - `35k`
- relative-to-resume windows:
  - `+2k`
  - `+5k`
  - `+10k`
  - `+15k`

What to inspect in images:
- whether duplicate code columns disappear or persist
- whether a style variant becomes a new semantic digit rather than another `1`
- whether any other code column collapses while the `1` split improves
- whether within-code samples remain coherent and readable

What to inspect in metrics:
- `real_fake_loss` stays in a healthy band (`~0.52-0.60`)
- `mi_avg` stays near or above `0` by `15k-20k`
- `r_sense_avg` remains non-collapsed but does not force style-only separation
- `r_intra_avg` stays healthy and does not collapse to near-identical samples

## 7. Decision Rules

- Continue without changes through `20k` unless D/G balance breaks.
- If the duplicate-column issue is clearly reduced by `20k`, keep this run going.
- If duplicate columns are still fixed by `25k-30k`, stop and retune again.
- Do not change learning rates before the `20k` check; this run is meant to isolate the `w_sense` change only.

## 8. Future Transfer Rule

- Once MNIST baseline behavior is locked, reuse this exact architecture for BloodMNIST.
- On transfer, change only:
  - dataset adapter
  - class count
  - channel handling
  - normalization/output pairing
- Keep CA off first on any new dataset.

## 9. Working Discipline

- One-variable changes only.
- Every run must preserve:
  - exact command
  - checkpoint source
  - metrics TSV
  - image snapshots at fixed checkpoints
- `PLAN.md` and `ABLATION_TRIAL_LOG.txt` must be updated immediately after each accepted run change.

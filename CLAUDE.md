# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EvoJAX is a hardware-accelerated neuroevolution toolkit built on JAX + Flax. This fork extends it with an InfoGAN architecture using HyperNetworks and Cultural Algorithms (CA) for medical image generation (BloodMNIST). See `PLAN.md` for the current experiment state, ablation results, and next steps.

## Common Commands

```bash
# Install (editable, with extras)
pip install -e .[extra]

# Run tests
pytest -W ignore::DeprecationWarning

# Lint (matches CI settings)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=100 --max-line-length=127 --statistics

# Train BloodMNIST baseline (CA disabled)
python examples/train_bloodmnist.py \
  --gpu-id='0,1' --pop-size=512 --batch-size=64 \
  --max-iter=290000 --ca-blend-coeff=0.0 \
  --static-fitness-weights --checkpoint-interval=5000

# Train MNIST InfoGAN
python examples/train_mnist_infogan.py \
  --gpu-id=0,1 --checkpoint-interval=5000 \
  --ca-blend-coeff=0.0 --static-fitness-weights \
  --static-mi-sense-ramp
```

## Architecture

### Three-Interface Design

All components implement one of three base interfaces:

| Interface | Location | Method |
|-----------|----------|--------|
| `NEAlgorithm` | `evojax/algo/base.py` | `ask()` → params, `tell(fitness)` |
| `PolicyNetwork` | `evojax/policy/base.py` | `get_actions(t_states, params, p_states)` |
| `VectorizedTask` | `evojax/task/base.py` | `reset(key)`, `step(state, action)` |

### Orchestration Layer

- **`evojax/trainer.py`** — Main training loop. Handles D-step (backprop), G-step (evolutionary), per-layer mode, fitness composition, and checkpoint save/load.
- **`evojax/sim_mgr.py`** — Simulation manager. Runs vectorized rollouts across devices, manages parameter distribution.

### InfoGAN / HyperNetwork Extension (this fork)

- **`evojax/policy/convnet.py`** — Generator (ConvTranspose), Discriminator (Conv + Q-head), and LayerHyperNetwork (generates per-layer weights from latent codes via chunk embeddings).
- **`evojax/task/latent.py`** — Latent vector task for InfoGAN. Computes r_sense, r_cons, r_intra metrics for mutual information and disentanglement.
- **`evojax/algo/pgpe_ca.py`** — PGPE solver extended with Cultural Algorithms. Contains `compute_ca_weights()`, `update_belief_space_from_metrics()`, fitness computation, and `rank_normalize`.
- **`evojax/algo/pgpe_layer.py`** — Lightweight per-layer PGPE variant (no CA). Used when `--per-layer` flag is set.
- **`evojax/algo/cultural/`** — CA framework: `belief_space.py`, `knowledge_sources.py` (Domain, Normative, Historical, Situational, Topographic), `helper_functions.py`.

### Training Modes

- **Standard mode**: Single PGPE_CA solver evolves all generator HyperNetwork params.
- **Per-layer mode** (`--per-layer`): 5 HN groups (Conv_0-3 + Dense_0, params >1000) each with own PGPE_Layer solver + LayerHyperNetwork, plus 1 misc group (direct evolution). Shared PGPE_CA computes fitness weights only.

## Key Technical Details

- **D/G asymmetry**: Discriminator trains via backprop (fast); Generator evolves via PGPE (slow). D-freeze threshold prevents D dominance.
- **Fitness composition**: All fitness components must go through `rank_normalize` (not `standardize`) to prevent any single component from dominating.
- **Latent structure**: `z = [noise(62), discrete(8), continuous(2)]` = 72 dims. Structured as n_sets of n_classes sharing same z-base.
- **Log line order** in trainer.py output: fitness_adv, fitness_mi, r_cons, r_sense, r_intra, norm_pen, safety_ratios, spreads, real_fake_loss.

## JAX/Flax Gotchas

- `jax.tree_util.tree_flatten_with_path` returns `DictKey` objects — access via `p.key`, not `str(p)`.
- Discriminator in `trainer.py` uses `train=True` with `mutable=['batch_stats']`; in `convnet.py` uses `train=False` with `mutable=False`.
- `PGPE_Layer.tell()` expects pre-combined fitness (no internal rank normalization).
- Normative KS shapes: `(pop_size, 11, 1)` spreads, `(pop_size, 11, 11)` safety_ratios.

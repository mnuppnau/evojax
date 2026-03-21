# EvoJAX: Hyper-InfoGAN with Cultural Algorithms

This project is a specialized research fork of [EvoJAX](https://github.com/google/evojax), a hardware-accelerated neuroevolution toolkit built on JAX. This fork extends the framework to implement a **Hyper-InfoGAN** architecture optimized using **Cultural Algorithms (CA)** for medical image generation, specifically targeting the **BloodMNIST** dataset.

## Project Overview

The core objective is to evolve a Generator network that learns disentangled representations of hematologic morphology (e.g., cell size, nucleus shape, cytoplasmic ratio) without explicit labels.

- **Architecture**: Uses a **HyperNetwork** to generate weights for the Generator layers from latent codes. This reduces the parameter space for evolution.
- **Optimization**: Employs **Neuroevolution (PGPE)** for the Generator/HyperNetwork and **Gradient Descent (Adam)** for the Discriminator/Q-head.
- **Cultural Algorithms (CA)**: Adds a meta-heuristic layer (Belief Space) to guide the evolution, helping escape local optima (like "prototype hardening") and balancing multiple fitness objectives (adversarial loss vs. mutual information).

## Key Components & Architecture

### 1. Evolutionary Algorithms (`evojax/algo/`)
- `pgpe_ca.py`: PGPE solver extended with Cultural Algorithms (Domain, Normative, Historical, Situational, Topographic knowledge sources).
- `pgpe_layer.py`: Per-layer PGPE variant for modular evolution of network layers.
- `cultural/`: Core CA framework including belief space and knowledge source implementations.

### 2. Policy Networks (`evojax/policy/`)
- `convnet.py`: Contains the InfoGAN components:
    - **Generator**: ConvTranspose network generating images from latent codes.
    - **Discriminator/Q-head**: Backbone for real/fake classification and latent code recovery.
    - **HyperNetwork**: Generates per-layer weights for the Generator using learned chunk embeddings.

### 3. Tasks (`evojax/task/`)
- `latent.py`: Latent vector task that computes InfoGAN-specific metrics:
    - `r_sense`: Sensitivity (mutual information).
    - `r_cons`: Consistency of generated features.
    - `r_intra`: Intra-class diversity.

### 4. Orchestration (`evojax/trainer.py`)
- Manages the **D-step** (backprop for Discriminator) and **G-step** (evolution for Generator).
- Handles fitness composition, rank normalization, and checkpointing.

## Key Commands

### Installation
```bash
# Install in editable mode with extra dependencies
pip install -e .[extra]
```

### Training
The project uses highly specific training configurations documented in `PLAN.md`.

**BloodMNIST Baseline (CA disabled):**
```bash
python examples/train_bloodmnist.py \
  --gpu-id='0,1' --pop-size=512 --batch-size=64 \
  --max-iter=290000 --ca-blend-coeff=0.0 \
  --static-fitness-weights --checkpoint-interval=5000
```

**MNIST InfoGAN:**
```bash
python examples/train_mnist_infogan.py \
  --gpu-id=0,1 --checkpoint-interval=5000 \
  --ca-blend-coeff=0.0 --static-fitness-weights
```

### Development & Testing
```bash
# Run tests
pytest -W ignore::DeprecationWarning

# Linting
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Development Conventions

- **JAX/Flax**: Strict adherence to functional programming patterns in JAX. Note that `jax.tree_util.tree_flatten_with_path` returns `DictKey` objects.
- **Fitness Normalization**: Always use `rank_normalize` (in `evojax/algo/pgpe_ca.py`) instead of standard scaling to prevent dominant fitness components.
- **Latent Structure**: The standard latent vector is `z = [noise(62), discrete(8), continuous(2)]`.
- **Documentation**: Refer to `PLAN.md` for the current ablation study status and `CLAUDE.md` for engineering-specific "gotchas".

## Important Files
- `PLAN.md`: The "Source of Truth" for current experiments, ablation results, and future strategy.
- `CLAUDE.md`: High-level summary of architecture and common commands.
- `ABLATION_TRIAL_LOG.txt`: Detailed log of experimental runs.
- `evojax/trainer.py`: The main entry point for the training logic.

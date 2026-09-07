# Evolutionary hard-attention research core

This is a reduced research fork of [EvoJAX](https://github.com/google/evojax),
preparing PGPE, HyperNetwork indirect encoding, and cultural memory for the
next IMDb study. It is **not** a drop-in replacement for upstream EvoJAX or
the BloodMNIST trainer.

Current branch: `research/imdb-hard-attention`. The committed BloodMNIST
implementation and paper remain available at `ee1bf0a` on
`claude/evolve-infogan-hypernetworks-s3r6d-67132f2-blood-baseline`.
Papers, figures, datasets, result archives, checkpoints, and local environments
have not been removed or regenerated.

## Current scope

Implemented and CPU-tested:

- Antithetic PGPE with Adam/SGD, correct optimizer counters, bounded sigma,
  neutral tied ranks, and strict ask/tell sequencing.
- Independently ranked fitness components with explicit objective direction.
- Bounded Pareto archives that store actual evaluated population rows and raw,
  signed metrics under a declared fixed evaluator protocol.
- Metric histories with explicit occupancy and actual-generation slopes.
- A task-independent chunked HyperNetwork adapter with per-tensor truncation.
- Common-data/common-random-number evaluation and versioned PGPE checkpoints.

Not implemented yet: an IMDb data pipeline, pretrained encoder loading,
attention masks, classifier/Q heads, an adaptive cultural controller, or a
multi-device text-training loop. See [IMDB_PLAN.md](IMDB_PLAN.md).
The retained archive/history are primitives, not a claim that the original
five-knowledge-source controller has been ported.

## Local checks

The validated environment is Python 3.11.5, JAX/JAXlib 0.4.31, Flax 0.8.4,
Optax 0.2.4, and NumPy 2.1.3. Use the existing environment without reinstalling:

```bash
JAX_PLATFORMS=cpu new-env/bin/python -m unittest discover -s tests -v
JAX_PLATFORMS=cpu new-env/bin/python -m examples.smoke_core
```

The smoke example is a small quadratic optimization, not a training experiment.
It downloads nothing and uses no dataset. For a fresh Python 3.11 environment,
`python -m pip install -e .` installs the CPU-compatible core. GPU installation
is intentionally not forced by package metadata; existing CUDA dependencies
are left alone. The paper figure script additionally needs Matplotlib
(`python -m pip install -e '.[paper]'` in a suitable environment).

## API rules

```python
from evojax.algo import PGPE
from evojax.fitness import combine_fitness

solver = PGPE(pop_size=32, param_size=8, seed=42, solution_ranking=False)
population = solver.ask()
metrics = {"loss": (population ** 2).mean(axis=1)}
fitness = combine_fitness(metrics, {"loss": 1.0}, maximize={"loss": False})
solver.tell(fitness)
```

PGPE maximizes fitness. Set `solution_ranking=False` for an already
rank-composed fitness; otherwise a second ranking discards component spacing.
The compatibility property `best_params` means the distribution center, not
the best evaluated individual.

Archives use `{"q": True}` to maximize a signed Q proxy, never `abs(q)`.
For cross-generation retention, measurements must come from the same fixed
evaluator **and data/protocol**. An identifier records this contract but cannot
enforce evaluator purity. Co-adapted Q scores can be logged or used for current
training; do not compare stale Q scores as stable semantic archive objectives.

The optimizer owns sigma. There are no CA center/sigma blend methods, cultural
sigma floors, automatic archive restores, or mask-temperature controls.
Selection-mediated influence will be implemented separately after the baseline.

## Checkpoints and compatibility

Use `evojax.checkpoint.save_checkpoint` only at a completed-generation boundary.
It saves PGPE center, sigma, Adam/SGD state and counters, sampling RNG, and
configuration. The caller supplies external state (data/evaluation RNGs, learned
heads and optimizers, archive/history snapshots, split/model IDs) through
`extra_state`. External state must be validated by the task before resuming.
Use dictionaries, lists, strings, scalars and arrays supported by Flax msgpack;
typed JAX keys should be stored as raw key data with their implementation.

Legacy pickle checkpoints are not migrated or loaded. Exact resume is tested
for the new core on the same software/backend, not promised across JAX/CUDA
versions. The optimizer fixes change future trajectories and do not repair or
validate earlier BloodMNIST experiments.

See [the code review](docs/CODE_REVIEW.md) for defects, evidence, verification,
known limitations, and [the removal manifest](docs/RETIRED_CODE.md) for recovery.
Original EvoJAX attribution and the Apache-2.0 [license](LICENSE) are retained.

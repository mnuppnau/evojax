# Code review and research-core cleanup

Reviewed: 2026-09-06.
Baseline: `ee1bf0ae5d4b0ea9fdfa62192d4a2ca2f215913e`.
Work branch: `research/imdb-hard-attention`.

## Outcome and scope

The new branch contains a smaller, task-independent evolutionary core.
Confirmed indexing and objective-direction defects are fixed in its replacement
archive path. PGPE's Adam counter, tie ranking, initialization, state protocol
and checkpoint handling are corrected and regression-tested. Image-specific
CA guidance, direct optimizer blending, GAN training and unrelated upstream
EvoJAX tasks/backends were retired, not carried forward behind disabled flags.

Review covered the original scientific path (trainer, PGPE/CA, policy/adapter,
simulation manager, image/latent tasks, cultural memory and checkpointing),
all retained/new package code, tests, public imports, dependencies/CI, and
retained paper-analysis utilities. Other upstream algorithms, demos and
benchmark configs were inventoried for dependencies and retired; they were
not all behaviorally revalidated or given individual correctness proofs.
Local environments and binary research artifacts were not audited as source.

This is not a backport to the BloodMNIST branch. It is not a completed IMDb
implementation. The replacement archive/history are generic primitives, not
the original five-source controller with merely renamed metrics.

## Priority findings

Locations below refer to the baseline revision, including files now removed.
Use `git show ee1bf0a:<path>` to inspect them.

### R01 — High: incorrect Adam update count (new finding, fixed)

`evojax/algo/pgpe.py:237` and `evojax/algo/pgpe_ca.py:1417` pass
`self._t // self._lr_decay_steps` as the optimizer update index. That index
drives Adam's bias correction as well as the learning-rate schedule. With the
CA default interval of 100,000, it remains zero for the first 100,000 updates
while the moment accumulators continue to change. A decay coefficient of 1.0
does not neutralize this error.

The new implementation uses Optax's actual per-update Adam counter. Only the
learning-rate schedule divides its counter by the decay interval. A regression
compares five deliberately varied gradient updates against a separate NumPy
Adam calculation, including an interval boundary.

Impact: this defect is not confined to blend-on configurations. For any
reported runs using this implementation, the center updates were not standard
Adam as described by the nominal hyperparameters. It does not by itself
establish which configuration would win after correction.

### R02 — High: antithetic archive indexing (previously identified, fixed)

`evojax/algo/pgpe_ca.py:1095` assumes a positive half followed by a negative
half. Sampling actually emits `[+eps0, -eps0, +eps1, -eps1, ...]`.
The selected fitness and archived parameter vector can therefore describe
different individuals.

The replacement archives `population[argmax(fitness)]` directly. Tests force
every row of an eight-member antithetic population to win and compare the
stored vector and metrics to that exact evaluated row. No reconstruction from
a direction index, current center or sign is needed.

The original paired RE update did not have this indexing defect. That narrower
statement must not be read as saying the old optimizer had no other defects
(see R01 and R04).

### R03 — High: signed objectives reduced to absolute values (previously identified, fixed)

`evojax/algo/cultural/knowledge_sources.py:459,517,599` puts
`abs(Q)` into minimization objectives. A larger positive Q proxy can then
lose to one closer to zero. Situational selection also minimizes the absolute
combined fitness. Historical retention minimizes entropy although ordinary
rescue retrieval selects maximum entropy (`knowledge_sources.py:954`).

The new archive declares each objective as maximize or minimize and preserves
its sign. Tests check positive Q versus near-zero/negative Q, loss minimization,
and entropy maximization. Population rank composition also uses explicit
directions; reversing ranks handles unsigned loss arrays without integer
wraparound.

These are archive selection errors, not evidence that the generator's active
Q fitness itself was minimized. The continuous Q term can make the correctly
signed weighted proxy positive; removing its absolute value is necessary.

### R04 — High: tied values created artificial selection pressure (new finding, fixed)

`evojax/algo/pgpe.py:49` and `evojax/algo/pgpe_ca.py:122,163`
assign different ranks to equal scores according to sort order. A constant
component can thus exert pressure despite containing no measured distinction.
Interleaved positive/negative pairs make order-dependent treatment especially
undesirable.

Average ranks now preserve ties. Constant finite scores are exactly neutral.
Tests cover ties, permutations, constants, singleton input, independently
weighted components and the distinction between one and two rankings.
Nonfinite active fitness fails at the host boundary instead of silently
participating in training.

### R05 — High: archive placeholders could be selected as elites (new finding, replaced)

`knowledge_sources.py:58–98` initializes unwritten records with zero
parameters and metric/entropy values of 1000. Retrieval uses maximum entropy
or maximum combined score (`knowledge_sources.py:890–892,954`), so those
sentinels can defeat valid records while they remain in the archive. In
`pgpe_ca.py:1217`, testing whether the resulting guidance contains a
nonzero coordinate is not an archive-occupancy check.

The new archive contains only real records; zero-valued parameters are valid
data, not an empty marker. There is no implicit rescue/restore or sigma state
in cultural memory. Tests cover empty memory, valid zero vectors, bounded
capacity and checkpoint restoration.

### R06 — High: checkpoints omitted independent random streams (new finding, replaced)

`evojax/util.py:287–303` stores the trainer RNG but not the PGPE sampler
RNG or simulation-manager RNG. Restoring center, sigma and Adam moments cannot
reproduce uninterrupted sampling. The referenced compatibility pytrees in
`load_checkpoint` were not actually used to validate the restored state.
The old format also executes pickle and is inappropriate for untrusted files.

New checkpoints have an explicit version, complete PGPE configuration,
center/sigma, optimizer state/counters, iteration and sampler key. Writes
serialize first, use a unique temporary file in the destination directory,
then atomically replace the destination. Invalid solver loads are rejected
before state mutation. There is no silent migration of old pickles.

Tests compare subsequent populations and updates exactly after both in-memory
and on-disk restore, and confirm failed saves preserve an existing checkpoint.
The future task must separately save and validate all external RNGs, head
optimizers, split/model IDs, evaluator protocol and cultural state.

## Other confirmed defects

| ID / priority | Baseline evidence | Disposition |
|---|---|---|
| R07 / Medium | `pgpe.py:167–169`: only Python float initializes sigma; documented vector input and integer input leave state unset. Negative inputs were silently absolutized. | Scalar/vector initialization and bounds are validated; invalid signs, shapes, nonfinite and nonrepresentable float32 settings are rejected. |
| R08 / Medium | `pgpe.py:63` and the CA equivalent squeeze away the direction axis for population size 2. | Pair axes retained; smallest antithetic population tested. |
| R09 / Medium | `pgpe.py:256` and CA setter update center but not the optimizer state holding the old center. | Warm start resets center optimizer/counter explicitly; exact resume uses state loading. |
| R10 / Medium | `pgpe_ca.py:392` discards the supplied `stdev_max_change` and forces 0.1. | Configured relative/absolute sigma limits are honored and tested. No CA override. |
| R11 / Medium | No consistent ask/tell generation-boundary checks. Repeated/misordered calls can reuse or replace samples. | Explicit pending-population state; duplicate calls, malformed scores, mid-generation save/restore and center changes are rejected. |
| R12 / Medium | `trainer.py:1068` hard-codes seed 44; `policy/convnet.py:765,779,1085` embeds other fixed seeds. | New PGPE accepts an actual seed; HyperNetwork initialization requires a caller key. This was not a between-seed validation of the historical study. |
| R13 / Medium | `trainer.py:1494–1511` regenerates the nominally fixed latent panel on resume and advances the restored trainer RNG. | Image path retired. Future audit panels must be explicitly stored or generated from a separate persisted key. |
| R14 / Medium | `knowledge_sources.py:649` treats nonzero centroid velocity as initialization; legitimate stationary centroids have zero velocity. First velocity also includes displacement from arbitrary initial centroids. | Co-adapted topographic state retired. New history uses actual populated rows, not numerical sentinels. |
| R15 / Medium | `sim_mgr.py:447,451,473`: non-scan fallback takes the wrong number of evaluation/reset-key arguments. | Incompatible simulation manager retired; replacement pure population evaluation is tested with common inputs/keys. |
| R16 / Medium | ClipUp divides by the gradient norm without guarding a zero norm. | Unused optimizer removed; supported center optimizers are Adam and SGD. |
| R17 / Low | `task/bloodmnist.py:91–110` advertises a different noise width and constructs a 70-entry latent without continuous codes. | Inconsistent side API retired. The active generator path uses the separate latent task; this does not establish that the reported generator received malformed 72-entry inputs. |
| R18 / Medium | Root imports eagerly pull in the image trainer and optional ES/RL dependencies; policy `__all__` names undefined `ConvNetPolicy`; old tests target removed APIs. | Minimal public exports and dependency-isolated import tests replace them. |
| R19 / Medium | setup/CI require unused image/RL packages and a forced CUDA JAX extra; Python settings disagree; publishing workflows target an upstream-style distribution. | Narrow, locally tested Python 3.11/core dependencies; CPU CI; no automatic publishing; obsolete Pipfile removed. |

Duplicate image architectures and adapters, commented-out experimental
implementations, one-off source patchers, stale demos and unused alternative
optimizers were removed. The retained adapter independently truncates padding
for every target tensor, supports explicit structural context and requires a
caller-provided initialization key. Tests cover scalar/tensor leaves, padding,
dtype preservation, deterministic initialization, JIT, vmap and gradients.

## Research-design hazards, not isolated code bugs

- Archived Q/feature metrics from a changing evaluator are not comparable
  across generations. Neither are within-population ranks or differently
  weighted combined scores. A frozen evaluator still needs a fixed data and
  evaluation protocol for historical comparisons. The replacement archive
  requires a declared evaluator identifier, but the caller must actually
  honor that contract.
- The stored sigma snapshot was shared across an original population, so the
  antithetic indexing error did not corrupt that numerical snapshot. However,
  faulty objective ordering and retention can still change WHICH snapshot is
  retrieved. Consequently, old sigma targets are not a clean intervention
  independent of all archive defects.
- Co-adapted Q remains a possible training signal, not a semantic oracle.
  Frozen evaluators used for fitness are also exposed to selection pressure;
  keep a separate final audit and intervention-based tests.
- The old direct CA blend and morphology schedules are not task-neutral
  reusable controls. They were removed. An adaptive sigma floor or an archive
  reset would also change exploration/trajectory and should not be treated as
  exempt from this concern.
- Deterministic top-k removes mask-sampling temperature, but it does not prove
  that shrinking PGPE sigma monotonically commits masks through a nonlinear
  HyperNetwork. Measure realized mask changes.
- Frozen contextual token features can leak information from unselected
  tokens. Cached-feature routing is not automatically a faithful explanation
  or an intervention on internal self-attention. The new roadmap explicitly
  requires sufficiency/deletion or masked-input re-encoding audits.

## Retained legacy tooling: documented open issues

These scripts remain unchanged to preserve the publication-analysis workflow.
They are not imported or exercised by the new training core.

1. `full-paper/analyze_bloodmnist_results.py`: `run_label` silently maps an
   unknown directory to A04-hybrid; repeated labels overwrite earlier traces
   and panels. A future archive with additional/renamed runs could therefore
   mislabel results. The current seven-run archive is the intended input.
   Before reusing the script with new archives, require an explicit
   run-to-directory map and reject duplicates/unknown runs.
2. That script assumes nonempty, well-formed TSVs, checks missing metric
   traces but not knowledge traces before plotting, and interprets a 50-row
   moving average as 5,000 iterations without checking cadence. Add fail-fast
   schema/cadence validation before generalizing it.
3. `scripts/compare_metrics_tsv.py`: malformed/missing/nonfinite iteration
   fields fail during integer conversion; duplicate or unordered rows are
   not rejected. Its delta formatter treats `rfl_mean` as a percentage
   because the name starts with `rfl_`, although mean loss is not a
   fraction. Use the script only on its expected archived TSV format until
   those parser/display issues are corrected.

No figures or paper tables were regenerated in this cleanup. These input
robustness/display findings alone do not demonstrate that the present paper
numbers are wrong.

## Verification

Run locally, using existing `new-env` with CPU JAX:

```bash
JAX_PLATFORMS=cpu new-env/bin/python -m unittest discover -s tests -v
JAX_PLATFORMS=cpu new-env/bin/python -m examples.smoke_core
new-env/bin/python -m pip check
new-env/bin/python setup.py --name --version
git diff --check
```

All 45 tests passed locally. The suite covers archive indexing and objective signs; ties/nonfinite fitness;
Adam step/decay correctness; sigma bounds and ask/tell protocol; full PGPE
resume; atomic failed writes; archive/history schemas; common-random-number
evaluation; HyperNetwork padding and transformations; and optional-dependency
isolation. The default smoke objective decreased from 1.000000 to 0.001209 in
40 updates, with 20 real archive records. This is a plumbing check, not evidence
about IMDb or clinical performance.

No training runs, model/dataset downloads, dependency installations, GPU
benchmarks or manuscript edits were performed. A fresh CI installation and
multi-GPU performance have not been validated locally. Existing dependency
metadata passes `pip check`; pins reflect the installed working environment,
not a complete transitive lockfile or a claim of latest-library compatibility.

## Remaining boundaries and handoff

- The active evaluator maps over one device and materializes population-sized
  arrays. Input validation and archives run on the host. Benchmark memory,
  host synchronization and microbatching before a large model/population.
- PGPE is float32. Resume equivalence is tested on the same backend/software,
  not across hardware/version changes. Checkpoints are generation-boundary
  snapshots, not mid-evaluation recovery.
- Archive objectives need fixed-protocol reevaluation if evaluator/data change.
  The archive currently admits one scalar-fitness winner per population, then
  applies bounded Pareto retention across generations; it is not a full
  population Pareto optimizer.
- External task/head state is the caller's responsibility and must be validated
  before training resumes. The new core does not yet define a task checkpoint
  schema or an attention controller.
- Some `venv-infogan-mnist` launcher/environment files were already tracked.
  They and all other local environments were deliberately left alone; ignoring
  a path does not untrack existing files. This remains repository hygiene debt.
- APIs intentionally break from the image trainer; no compatibility shim loads
  obsolete controllers or silently adapts old pickles. Package publishing is
  disabled on this branch.

The historical branch and artifacts are unchanged; all removed code is
recoverable from the baseline. See [RETIRED_CODE.md](RETIRED_CODE.md) for the
exact manifest and [../IMDB_PLAN.md](../IMDB_PLAN.md) for the next implementation
steps. Before journal submission, check which original revision produced each
run and qualify the Adam description and archive mechanism claims if these
defects were present. This audit does not require new BloodMNIST runs, but
corrected future experiments must be described as a new implementation.

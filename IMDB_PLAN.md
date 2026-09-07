# Next study: evolutionary code-conditioned hard attention

Status: core cleanup complete; transformer experiment **not implemented**.
Branch: `research/imdb-hard-attention`, based on BloodMNIST commit `ee1bf0a`.
Planning horizon: approximately five months, with a sixth month only as buffer.

## Research question and scope

Test whether PGPE can learn useful, code-dependent **discrete routing** over a
frozen pretrained representation, and whether cultural influence through
selection improves that routing. This addresses the learning-rule/architecture
question without claiming to evolve language competence from scratch.

The repository's earlier IMDb plan already selected frozen-encoder sentiment
classification. Keep that as the bounded first study; autoregressive review
generation, LoRA, multiple semantic code factors and content-dependent
internal head gating are separate extensions, not prerequisites for this paper.

Use a small frozen pretrained encoder, initially on IMDb sentiment. Pin the
exact model/tokenizer revision and data splits when the pipeline is built.
The particular encoder and feature-cache design still need a short feasibility
check; do not add/download an ecosystem of model dependencies preemptively.

## Proposed baseline

1. Precompute or cache frozen token representations in inference mode. Keep
   token IDs, padding masks and split provenance alongside features. Benchmark
   cache size, I/O, sequence length and population microbatching before choosing
   a training budget.
2. Evolve a small code-conditioned scoring module, directly and through the
   retained HyperNetwork adapter. Compare measured search dimensions; indirect
   encoding is not automatically a compression when the target is tiny.
3. Apply deterministic top-k selection to eligible token positions, with a
   declared tie-break, no padding/special-token selection, and explicit handling
   of reviews with fewer than k eligible tokens. Start with one small
   categorical code (provisional K=4) and a fixed sparsity budget. Do not inherit
   image noise dimensions or continuous codes without a testable use.
4. Pool selected representations for sentiment prediction. Establish a
   supervised classifier baseline first; if classifier or Q heads subsequently
   co-adapt with the router, label their scores as training signals and retain
   independent frozen/held-out semantic audits.
5. Train an auxiliary Q head only on information available through the routed
   representation, not the code or gate logits themselves. Test code recovery
   after token/position shortcut interventions.
6. Evaluate every population member on the same minibatch and random draws,
   with fresh generation-level randomness and separate held-out audit keys.
   Save all head optimizers, RNG streams, split/model IDs and cultural state.

This is hard **token selection over contextual features**, not a replacement
of the encoder's internal self-attention. Selected contextual vectors may
already encode information from unselected tokens. Token highlights alone
therefore do not establish faithful rationales: evaluate masked-input
re-encoding or equivalent deletion/sufficiency interventions separately.
An internal attention-head intervention would prevent straightforward reuse
of cached full-encoder features and needs a separate compute budget.

## Controls and metrics

Keep the first question small enough to replicate:

- Full-feature/soft-attention supervised baseline: establish representation
  quality and the task ceiling for this setup.
- Matched hard-routing architecture trained with a differentiable relaxation
  or straight-through estimator, versus PGPE. Match parameterization, data,
  sparsity and tuning budgets; report wall time and evaluations as well.
- Direct-encoding versus HyperNetwork PGPE only after both can solve a short
  pilot. This isolates encoding from optimization.
- No-CA versus a single selection-mediated cultural intervention after the
  baseline passes checks. Do not begin with a large controller matrix.

Primary task metric: held-out sentiment accuracy and its control/sparsity
tradeoff. Separately measure Q recovery, code redundancy, mask overlap,
position/frequency shortcuts and mask stability. A code is not a clinical,
topic, style or sentiment category merely because Q can recover it. Sentiment
labels support the classification task; they do not make latent-code discovery
fully unsupervised if used to shape code semantics.

Reserve independent held-out evaluation from the start. If a frozen evaluator
is used for fitness, it is a training judge, not an independent final audit.
Use additional held-out protocols/interventions to assess gaming.

## Cultural influence: after the baseline

The retained `ParetoArchive` and `MetricHistory` supply storage, not a
ready-made controller. Implement one small, explicit fitness-weighting or
mask-prior term first. Define its update/acceptance rule and ablate it.

- No direct center/sigma blend, cultural sigma floor, automatic archive reset,
  or separate mask temperature. A floor can also obstruct contraction; a
  one-sided intervention is not automatically harmless.
- A cultural prior may change fitness/selection, not add directly to routing
  logits or mutate PGPE state.
- Retain raw metrics evaluated under the same fixed model AND data protocol.
  A frozen encoder alone does not make scores from changing minibatches or
  changing heads comparable across generations.
- Wrong priors can still dominate selection or entrench early mistakes. Bound
  their weight and assess diversity rather than asserting they self-correct.
- Measure actual mask-change frequencies and emitted logit margins. Shrinking
  parameter-space sigma does not guarantee monotonic circuit commitment
  through a nonlinear HyperNetwork.

The old five-source controller and morphology-specific schedules were retired.
Do not describe the generic replacement as an already validated transfer of
A04-hybrid, or make a predicted failure of CA blending a required outcome.

## Milestones and stopping rules

| Window | Deliverable | Gate before expanding |
|---|---|---|
| Month 1 | Prospectus, fixed data/model protocol, encoder cache, supervised baseline, runtime pilot | Fits hardware and solves sentiment |
| Month 2 | Small code-conditioned deterministic router, Q and shortcut diagnostics | Nontrivial code-dependent routing on held-out examples |
| Month 3 | Matched optimizer/encoding comparisons | Short replicated runs establish what is worth scaling |
| Month 4 | One selection-mediated CA intervention and targeted replications | Added effect survives the chosen controls |
| Month 5 | Final evaluation, figures, paper and dissertation integration | Freeze scope and report negative results honestly |
| Month 6, if available | Buffer and only decisive follow-ups | No new architectural dependency |

Do not inherit 512 individuals or 290,000 iterations from BloodMNIST. Set
population size, budget and replication count using measured pilot cost and
variance. No further BloodMNIST training is required for this code-cleanup
task. The new optimizer corrections mean old and new trajectories are not
interchangeable.

## Implementation queue

- [x] Separate reusable PGPE, rank shaping, HyperNetwork, archive/history.
- [x] Fix indexing/sign/counter defects and add CPU regression tests.
- [x] Replace legacy checkpointing; remove image/RL and direct-blend paths.
- [ ] Decide encoder revision, sequence length and feature-cache schema.
- [ ] Implement and test data splits, feature extraction and pad-safe top-k.
- [ ] Establish supervised task baseline and benchmark population evaluation.
- [ ] Add code injection/Q with leakage and shortcut tests.
- [ ] Run small matched controls before any full research run.
- [ ] Add one CA selection channel and complete the bounded ablation.

See [docs/CODE_REVIEW.md](docs/CODE_REVIEW.md) for the audit and limitations.

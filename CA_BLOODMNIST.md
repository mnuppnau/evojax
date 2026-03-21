# CA + BloodMNIST Deep Design Record

## Purpose

This document records the low-level rationale behind the BloodMNIST phase of the project.

It is meant to sit below `PLAN.md` and `ABLATION_TRIAL_LOG.txt` in detail. The goal is to capture:

- what the BloodMNIST setup actually is in this codebase
- what the Cultural Algorithm (CA) is doing at the optimization level
- what the CA is *supposed* to do at the dataset-semantic level
- what the ablations have shown so far
- what future BloodMNIST and IMDb directions still make sense

This is not a paper draft. It is a working design and interpretation document.

---

## 1. BloodMNIST In This Repository

### 1.1 Dataset definition in code

The BloodMNIST pipeline in this repo is defined primarily in:

- `examples/train_bloodmnist.py`
- `evojax/task/bloodmnist.py`
- `evojax/task/latent.py`
- `evojax/policy/convnet.py`
- `evojax/trainer.py`

The implementation assumptions are:

- dataset file: `./data/bloodmnist.npz`
- train/test splits are read directly from that `.npz`
- images are converted from RGB stain images to grayscale
- pixel range is normalized to `[0, 1]`
- image size is `28 x 28 x 1`
- discrete code count is `8`
- continuous code count is `2`
- noise dimension is `62`
- total latent size is `72`

The grayscale conversion in `evojax/task/bloodmnist.py` is:

- `0.2989 * R + 0.5870 * G + 0.1140 * B`
- then divided by `255.0`

So this BloodMNIST pipeline is not using the original color information. That matters. Some morphology and stain cues that may help class separation in the original dataset are compressed away before training.

### 1.2 What makes BloodMNIST harder than MNIST here

MNIST rewards clean stroke topology and relatively simple mode separation. BloodMNIST is harder in this setup because:

- class structure is morphology-heavy rather than topology-heavy
- morphology differences can be subtle at `28 x 28`
- grayscale conversion removes stain/color structure
- the generator can find cheap shortcuts that still satisfy MI and D/Q enough to survive

The main shortcut family we observed is:

- one generic blood-cell template
- differences expressed through:
  - overall size
  - darkness / contrast
  - nucleus position around the center
  - nucleus eccentricity without true whole-cell re-structuring

That is the reason BloodMNIST is the right dataset for testing whether the CA can become semantically aware instead of remaining only a generic optimization controller.

---

## 2. Locked Baseline System

### 2.1 High-level architecture

The baseline system is a HyperNetwork-conditioned InfoGAN trained with mixed optimization:

- PGPE optimizes the HyperNetwork
- the HyperNetwork generates Generator weights
- the Discriminator and Q-head are trained with backpropagation

The key architectural pieces are:

- HyperNetwork parameter count: about `32k`
- generated Generator parameter count: about `225k`
- Discriminator parameter count: about `455k` at `--disc-features=48`

This matters because the project is explicitly comparing:

- direct PGPE over a large Generator parameter space
- vs.
- PGPE over a smaller HyperNetwork that indirectly generates Generator weights

### 2.2 Generator structure

The baseline Generator is intentionally split at the latent-entry level.

The latent is:

- `noise(62)`
- `discrete(8)`
- `continuous(2)`

The Generator uses split latent seed paths:

- `seed_base(noise only)`
- `seed_code(discrete only)`
- `seed_cont(continuous only)`

This is important because one of the core baseline hypotheses is that latent leakage should be reduced structurally, not only by losses.

### 2.3 Discriminator / Q structure

The Discriminator side has:

- a shared convolutional backbone
- an adversarial D-head
- a Q-head without global average pooling
- a brightness guard before Q

The Q-head is intentionally spatial. That is a meaningful baseline decision because global pooling tends to destroy the location-sensitive signal that helped the latent structure stabilize in this codebase.

### 2.4 Locked baseline schedule for BloodMNIST

The restored non-CA baseline `BLD-B00` uses:

- `--ca-blend-coeff=0.0`
- `--static-fitness-weights`
- static phased `div / sense / intra` schedule
- discriminator width `48`

Representative command family:

```bash
./new-env/bin/python examples/train_bloodmnist.py --gpu-id='0,1' --checkpoint-interval=5000 --ca-blend-coeff=0.0 --static-fitness-weights --static-mi-sense-ramp --static-w-div=0.18 --static-w-sense=0.08 --static-w-intra=0.03 --static-div-ramp-target=0.16 --static-div-ramp-start-iter=12000 --static-div-ramp-end-iter=28000 --static-sense-ramp-target=0.14 --static-sense-ramp-start-iter=12000 --static-sense-ramp-end-iter=28000 --static-intra-ramp-target=0.04 --static-intra-ramp-start-iter=12000 --static-intra-ramp-end-iter=28000 --disc-features=48
```

The baseline remains the reference because it is the cleanest BloodMNIST run so far even though it still only achieves partial disentanglement.

---

## 3. What The CA Is At The Optimization Level

### 3.1 CA role in this system

At the optimization level, the CA is not a second optimizer that replaces PGPE.

It is a controller layered on top of PGPE that modifies:

- how fitness pressures are weighted
- where the center update is nudged
- how exploration (`stdev`) is nudged
- when rescue behavior should override normal exploitation

That distinction matters.

The CA is not supposed to redefine the optimization problem from scratch. It is supposed to shape the search trajectory of PGPE using accumulated knowledge.

### 3.2 Belief-space structure

The belief space currently contains:

- `meta`
- `Domain KS`
- `Situational KS`
- `Historical KS`
- `Topographic KS`
- `Normative KS`
- `metric_history`

At a high level:

- `Domain KS` stores elite parameter-space patterns and should become the semantic interpreter
- `Situational KS` stores current best / exploitative guidance
- `Historical KS` stores earlier strong states for rescue
- `Topographic KS` tracks output-space geometry
- `Normative KS` stores soft acceptable regions / standards
- `metric_history` stores rolling controller signals and slopes

### 3.3 How the CA actually affects PGPE

In `PGPE_CA.tell(...)`, the algorithm:

1. evaluates the population
2. computes the normal REINFORCE-style PGPE updates
3. asks the belief space for center/stdev guidance
4. converts that guidance into CA gradient directions
5. rescales the CA directions to match REINFORCE magnitude
6. blends them using `ca_blend`

Important implication:

- the CA does not directly overwrite the parameter center
- it contributes a *directional bias* to PGPE
- the effective influence is controlled by both blend scheduling and runtime gates

This means the CA is fundamentally a meta-controller over PGPE dynamics.

### 3.4 Runtime gates already present

The system already gates CA influence using runtime health:

- `real_fake_loss` gate for CA blend
- objective-health gates for some weight schedules
- semantic trap triggers in the morphology-aware phases
- normative-violation triggers in `A04d/A04e`

So the CA is already a conditional controller, not a permanently full-strength overlay.

### 3.5 Why this matters for the research story

This is the reusable, framework-level part of the CA.

Across tasks, the general optimization-level responsibilities are:

- monitor adversarial balance
- monitor MI progress
- monitor collapse risk
- monitor commitment vs exploration timing
- rescue runs from shortcut basins or plateaus
- reweight or redirect search pressure without discarding PGPE

That part is general enough to transfer beyond BloodMNIST.

---

## 4. What The CA Must Become At The Semantic Level

### 4.1 General principle

The CA should be:

- general at the optimization-framework level
- task-instantiated at the semantic level

That means:

- the *machinery* should transfer
- the *meaning of a good solution* should not be assumed to transfer

This is the central design stance of the project.

### 4.2 Why dataset-agnostic semantics are not enough

If the CA only sees generic GAN-health signals, it can help with:

- stability
- exploration timing
- avoiding total collapse

But it cannot tell whether the model is separating codes on the *right factors*.

That is exactly what happened on BloodMNIST:

- generic controller signals were not enough
- the model found shortcuts that looked acceptable to MI and D/Q
- the CA could stabilize those shortcuts unless it was given morphology-aware signals

So the CA cannot remain purely statistical if the goal is meaningful disentanglement.

### 4.3 Semantic responsibilities of each KS

#### Domain KS

Domain KS should be the semantic interpreter.

For BloodMNIST, that means learning whether the current latent separation is happening along:

- meaningful morphology
- or shortcuts such as size, angle, and contrast

Domain KS is the natural place to say:

- this is a real morphological direction
- this is just clock-face coding
- this elite pool is semantically weak even if it is numerically stable

#### Normative KS

Normative KS should not just track generic spread or safety forever.

Once semantic control matters, Normative KS should express soft acceptable regions for the factors that Domain KS has identified as meaningful.

For BloodMNIST, that means soft morphology standards such as:

- minimum acceptable cell-body circularity
- acceptable nucleus-offset band
- maximum acceptable cross-code whole-cell area variance

Normative KS should answer:

- what structural properties must remain valid while exploration continues?

#### Historical KS

Historical KS is the rescue archive.

It should preserve states from before the run narrowed into a shortcut basin.

On BloodMNIST, that means rescue should prefer history states with:

- better morphology
- less prototype collapse
- less shortcut coding

#### Situational KS

Situational KS remains the exploitative current-best source.

Once Normative KS becomes meaningful, Situational guidance should only remain dominant when the current best solution is not violating basic morphology standards.

#### Topographic KS

Topographic KS tracks output-space structure.

It is useful for exploration pressure and output geometry, but it should not define semantic validity by itself.

This is especially important in BloodMNIST because output geometry can look diverse while still being morphologically trivial.

---

## 5. BloodMNIST-Specific Failure Modes

### 5.1 The one-template failure

The main failure mode across many runs has been:

- one generic blood-cell template reused across codes
- code identity carried mainly by easy factors

Observed easy factors:

- cell size
- darkness / contrast
- nucleus position around the center
- nucleus eccentricity without true whole-cell restructuring

### 5.2 The clock-face exploit

This was the most obvious early semantic shortcut.

A single generic cell template could satisfy Q by moving a dark mass around the center like positions on a clock face.

This is why explicit orientation-aware measurement was necessary.

### 5.3 Why BloodMNIST is a good CA testbed

BloodMNIST exposes whether the CA can distinguish:

- mathematically valid code separation
- from
- medically weak semantic separation

That is exactly the gap we want the CA to close.

---

## 6. Morphology Metrics Added So Far

The morphology-aware work so far introduced controller signals derived from per-code prototype images in `evojax/policy/convnet.py`.

### 6.1 Earlier generic / shortcut metrics

- `morph_dark_range`
  - range of dark-pixel fraction across codes
- `morph_center_edge_range`
  - range of center-vs-edge contrast across codes
- `edge_dark_frac`
  - dark mass in the border ring; artifact proxy
- `code_proto_corr`
  - mean pairwise prototype correlation across codes
- `proto_angle_spread`
  - whether code separation is mainly driven by dark-mass rotation around the center

These are useful for diagnosing the shortcut, but they are not enough to define meaningful blood-cell morphology.

### 6.2 First biology-oriented metrics

- `nuc_cell_ratio_range`
  - variation in nucleus-to-cell area ratio across codes
- `nuc_eccentricity_range`
  - variation in nucleus eccentricity across codes

These help detect whether different codes are producing different kinds of nuclei, but by themselves they did not protect whole-cell shape.

### 6.3 Whole-cell integrity metrics

- `cell_circularity`
  - whole-cell compactness via `4*pi*area / perimeter^2`
- `cell_area_var`
  - cross-code variation in whole-cell area
- `nucleus_offset`
  - nucleus centroid offset relative to cell centroid, normalized by cell-equivalent radius

These matter because the controller must distinguish:

- moving the nucleus *within* the cell body
- from
- deforming or translating the whole cell template

---

## 7. CA Ablation Story On BloodMNIST

### 7.1 BLD-B00: no CA

Result:

- strongest BloodMNIST run so far
- stable adversarial balance
- clean circular cells
- partial disentanglement only
- separation still mostly dominated by easy axes

Interpretation:

- this is the correct baseline
- it proves the HyperNetwork + PGPE setup is viable on BloodMNIST
- it does not prove morphology disentanglement is solved

### 7.2 A01: dynamic weights only

Result:

- stable but too conservative
- underweighted MI
- poor morphology separation

Interpretation:

- generic adaptive weighting acts as a stability controller
- not enough for semantic disentanglement

### 7.3 A02: blend only

Result:

- negative
- did not break the shortcut

Interpretation:

- KS guidance without semantic signals was not enough

### 7.4 A03: full generic CA

Result:

- negative
- stronger controller did not help because the controller priors were not semantically meaningful

Interpretation:

- generic CA pieces alone can be neutral or harmful if they steer confidently using the wrong signals

### 7.5 A04a: shortcut detection

Added:

- `proto_angle_spread`
- explicit semantic trap detection
- rescue bias toward Historical / Topographic guidance

Result:

- useful negative
- reduced the shortcut somewhat
- but drifted toward smooth shared-template cells

Interpretation:

- detecting the bad direction is not enough
- the controller also needs a constructive target

### 7.6 A04b: nucleus-oriented biology signals

Added:

- `nuc_cell_ratio_range`
- `nuc_eccentricity_range`
- biology-aware historical rescue

Result:

- mixed-negative
- better nucleus variation
- whole-cell shape degraded late

Interpretation:

- nucleus signals alone are not enough
- the controller still lacked explicit whole-cell integrity signals

### 7.7 A04c: whole-cell integrity signals

Added:

- `cell_circularity`
- `cell_area_var`
- `nucleus_offset`

Result:

- partial positive
- better nucleus distinction
- lower shortcut pressure
- still not better than `BLD-B00`

Interpretation:

- the controller finally became meaningfully morphology-aware
- but rescue-only control still did not preserve whole-cell morphology strongly enough

### 7.8 A04d: Normative KS soft bounds

Added:

- Normative KS soft bounds for morphology
- normative violation state feeds back into center and stdev guidance

Result:

- mixed result
- slightly healthier than A04c
- still not better than `BLD-B00`
- learned bounds were too permissive

Interpretation:

- the controller started protecting morphology in principle
- but Normative KS was acting too much like a moving average of the current elite pool

### 7.9 A04e: asymmetric normative ratchet

Current intended test:

- preserve the A04d controller structure
- change only the Normative KS learning rule
- tighten quickly when elites improve
- relax extremely slowly after warmup
- add deadbands so the bounds do not move on noise

Interpretation:

- this is the clean test of whether Normative KS can become a real standard rather than a trailing summary

---

## 8. Optimization-Level Hypothesis

At the optimization level, the current hypothesis is:

- the CA is most useful as an adaptive controller over PGPE
- not as a replacement for PGPE
- and not as a fully generic semantic system

In that role, the CA should help PGPE by:

- deciding when to explore vs exploit
- deciding when to rescue the run from a shortcut basin
- deciding when current elites are semantically weak even if numerically acceptable
- preserving structural integrity while latent separation improves

In plain terms:

- the optimization framework gives the system the *capacity* to evolve
- the dataset-semantic belief space gives it the *criteria* for evolving meaningfully

That is the core project thesis.

---

## 9. Dataset-Semantic Hypothesis For BloodMNIST

The dataset-semantic hypothesis is:

- meaningful BloodMNIST disentanglement requires the controller to know something about blood-cell morphology
- not merely about adversarial health or MI

Specifically, the CA must eventually distinguish between:

- valid nucleus movement inside a coherent cell body
- invalid whole-cell deformation used as a shortcut
- valid variation in nucleus/cytoplasm structure
- invalid reuse of one cell template across many codes

A controller that only watches:

- `mi_avg`
- `real_fake_loss`
- `r_sense`
- entropy

cannot reliably make that distinction.

This is why BloodMNIST is not just a harder benchmark. It is the dataset that forces the CA architecture to confront semantic instantiation directly.

---

## 10. Future Directions For BloodMNIST

### 10.1 Immediate next CA directions

If CA work continues on BloodMNIST, the highest-value next directions are:

1. Stricter Normative KS learning
- asymmetric ratchet
- stronger priors
- tighter elite filtering
- health-gated tightening

2. Better elite semantics
- do not let Normative KS learn mainly from a mediocre elite pool
- require better morphology before standards ratchet downward or stabilize

3. Per-code morphology only after global integrity is solved
- per-code norms are tempting
- but should come after global cell-body integrity is robust
- otherwise the controller may overconstrain noisy early code allocations

### 10.2 Additional BloodMNIST metrics that may still help

If further semantic instrumentation is needed:

- nucleus connected-components or lobulation proxy
- contour roughness / convexity proxy
- nucleus-to-cell overlap or boundary proximity statistics
- texture-sensitive nucleus density measures
- per-code morphological consistency over time, not just per-generation snapshots

### 10.3 What not to do first

The current evidence suggests avoiding these as the immediate next step:

- more generic controller tuning without semantic changes
- only adjusting D strength without fixing morphology semantics
- adding many new signals at once without isolating the change
- hard class-specific norms before a cleaner global morphology controller exists

---

## 11. Future Directions For IMDb / Hard-Attention Transformers

### 11.1 What should transfer directly

The following should transfer from BloodMNIST to a future IMDb HyperNetwork/PGPE hard-attention project:

- belief-space architecture
- Domain / Normative / Historical / Situational / Topographic KS roles
- PGPE + HyperNetwork optimization framing
- CA as a controller over search rather than a replacement optimizer
- explore / exploit / rescue logic
- the principle of semantic trap detection

### 11.2 What must be re-instantiated semantically

For a hard-attention sentiment model, the semantic interface must change completely.

BloodMNIST semantics are morphology-based. IMDb semantics will be attention-structure-based.

Possible shortcut families on IMDb:

- punctuation fixation
- positional bias
- overuse of a few high-frequency sentiment tokens
- trivial attention sparsity that ignores sentiment-bearing spans
- class-separating token heuristics that are not robust semantic evidence

### 11.3 Possible Domain KS signals for IMDb

Domain KS for IMDb could eventually learn from signals such as:

- attention sparsity
- attention continuity / fragmentation
- fraction of attention on sentiment-bearing tokens
- token-position entropy
- diversity of attention locations across codes or prompts
- overlap between attended tokens and model saliency or rationale proxies

### 11.4 Possible Normative KS signals for IMDb

Normative KS could express soft acceptable ranges for:

- minimum and maximum sparsity
- acceptable continuity of attended spans
- lower bounds on sentiment-token coverage
- upper bounds on positional collapse
- code-conditional diversity of attended evidence

### 11.5 Research connection back to BloodMNIST

BloodMNIST is the morphology case study.
IMDb would be the attention-structure case study.

The shared research claim would be:

- the CA is reusable as an optimization-level controller
- but its semantic interface must be instantiated to the domain

On BloodMNIST:

- semantic interface = morphology

On IMDb:

- semantic interface = attention behavior over text

That is the longer-term unifying idea.

---

## 12. Current Bottom Line

The BloodMNIST work has already established several important points:

1. HyperNetworks make PGPE viable on BloodMNIST in a way direct large-space evolution did not.
2. The best current BloodMNIST result is still the non-CA baseline `BLD-B00`.
3. Generic CA signals alone do not solve morphology disentanglement.
4. BloodMNIST requires dataset-semantic controller inputs.
5. Domain KS and Normative KS are the key places where that semantic interface must be instantiated.
6. The CA should be evaluated as:
   - a general optimizer-controller framework
   - with domain-specific semantic instrumentation

That is the design position to preserve going forward.

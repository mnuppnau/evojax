# Retired code manifest

Date: 2026-09-06. New branch: `research/imdb-hard-attention`.

173 tracked files were removed from the working tree. Every listed file is
recoverable from baseline commit `ee1bf0ae5d4b0ea9fdfa62192d4a2ca2f215913e`,
which remains on the BloodMNIST branch. No history was rewritten.

Read a retired file without changing branches:

```bash
git show ee1bf0a:evojax/algo/pgpe_ca.py
```

For an old experiment, use a separate checkout of the BloodMNIST revision and
its original environment, not a mixture of old trainers with the new API.
No old checkpoints were converted. Papers, figures, result archives, datasets,
logs, model checkpoints, and virtual environments were not deleted.

The removals are intentional API breaks: no `Trainer`, `SimManager`,
`PGPE_CA`, `GenPolicy`, `DiscPolicy`, RL tasks, or optional ES backends are
exported. Reusable functionality is now in PGPE, fitness, evaluation,
checkpoint, HyperNetwork, archive and history modules. A transformer trainer
and the new cultural controller are still future work.

The publication-analysis scripts are retained separately:
`full-paper/analyze_bloodmnist_results.py` and
`scripts/compare_metrics_tsv.py`. They are not imported by the new core.
Benchmark output images/logs remain; their obsolete training drivers do not.

## Old training, optimizer, policy and task code (46)

- `evojax/algo/README.md`
- `evojax/algo/ars.py`
- `evojax/algo/ars_native.py`
- `evojax/algo/cma_evosax.py`
- `evojax/algo/cma_jax.py`
- `evojax/algo/cma_wrapper.py`
- `evojax/algo/crfmnes.py`
- `evojax/algo/cultural/base.py`
- `evojax/algo/cultural/belief_space.py`
- `evojax/algo/cultural/helper_functions.py`
- `evojax/algo/cultural/knowledge_sources.py`
- `evojax/algo/cultural/population_space.py`
- `evojax/algo/fcrfmc.py`
- `evojax/algo/fpgpec.py`
- `evojax/algo/iamalgam.py`
- `evojax/algo/map_elites.py`
- `evojax/algo/open_es.py`
- `evojax/algo/pgpe_ca.py`
- `evojax/algo/pgpe_disc.py`
- `evojax/algo/pgpe_q.py`
- `evojax/algo/sep_cma_es.py`
- `evojax/algo/simple_ga.py`
- `evojax/obs_norm.py`
- `evojax/policy/base.py`
- `evojax/policy/convnet.py`
- `evojax/policy/mlp.py`
- `evojax/policy/mlp_pi.py`
- `evojax/policy/seq2seq.py`
- `evojax/sim_mgr.py`
- `evojax/task/README.md`
- `evojax/task/__init__.py`
- `evojax/task/base.py`
- `evojax/task/bloodmnist.py`
- `evojax/task/brax_task.py`
- `evojax/task/cartpole.py`
- `evojax/task/flocking.py`
- `evojax/task/latent.py`
- `evojax/task/ma_waterworld.py`
- `evojax/task/mdkp.py`
- `evojax/task/mnist.py`
- `evojax/task/procgen_task.py`
- `evojax/task/seq2seq.py`
- `evojax/task/slimevolley.py`
- `evojax/task/waterworld.py`
- `evojax/trainer.py`
- `evojax/util.py`

## Retired examples and notebooks (29)

- `examples/notebooks/AbstractPainting01.ipynb`
- `examples/notebooks/AbstractPainting02.ipynb`
- `examples/notebooks/BraxMetaLearning.ipynb`
- `examples/notebooks/BraxTasks.ipynb`
- `examples/notebooks/EncirclingAgents.ipynb`
- `examples/notebooks/FlockingExploration.ipynb`
- `examples/notebooks/FlockingSimple.ipynb`
- `examples/notebooks/GymnaxEvosax.ipynb`
- `examples/notebooks/HighResGIFfromSVG.ipynb`
- `examples/notebooks/LeniaEvoJAX.ipynb`
- `examples/notebooks/SpikingNeuralNetworksMNIST.ipynb`
- `examples/notebooks/TutorialAlgorithmImplementation.ipynb`
- `examples/notebooks/TutorialCallEvoJAXAlgorithmDirectly.ipynb`
- `examples/notebooks/TutorialNonVectorTask.ipynb`
- `examples/notebooks/TutorialPolicyImplementation.ipynb`
- `examples/notebooks/TutorialTaskImplementation.ipynb`
- `examples/train_ant_map_elites.py`
- `examples/train_bloodmnist.py`
- `examples/train_cartpole.py`
- `examples/train_infogan.py`
- `examples/train_mdkp.py`
- `examples/train_mnist.py`
- `examples/train_mnist_infogan.py`
- `examples/train_organsmnist.py`
- `examples/train_seq2seq.py`
- `examples/train_slimevolley.py`
- `examples/train_waterworld.py`
- `examples/train_waterworld_ma.py`
- `infogan.ipynb`

## Unused benchmark drivers and configurations (75)

- `scripts/benchmarks/Readme.md`
- `scripts/benchmarks/configs/ARS/brax_ant.yaml`
- `scripts/benchmarks/configs/ARS/cartpole_easy.yaml`
- `scripts/benchmarks/configs/ARS/cartpole_hard.yaml`
- `scripts/benchmarks/configs/ARS/mnist.yaml`
- `scripts/benchmarks/configs/ARS/search.yaml`
- `scripts/benchmarks/configs/ARS/waterworld.yaml`
- `scripts/benchmarks/configs/ARS/waterworld_ma.yaml`
- `scripts/benchmarks/configs/ARS_native/brax_ant.yaml`
- `scripts/benchmarks/configs/ARS_native/cartpole_easy.yaml`
- `scripts/benchmarks/configs/ARS_native/cartpole_hard.yaml`
- `scripts/benchmarks/configs/ARS_native/mnist.yaml`
- `scripts/benchmarks/configs/ARS_native/search.yaml`
- `scripts/benchmarks/configs/ARS_native/waterworld.yaml`
- `scripts/benchmarks/configs/ARS_native/waterworld_ma.yaml`
- `scripts/benchmarks/configs/CMA_ES/brax_ant.yaml`
- `scripts/benchmarks/configs/CMA_ES/cartpole_easy.yaml`
- `scripts/benchmarks/configs/CMA_ES/cartpole_hard.yaml`
- `scripts/benchmarks/configs/CMA_ES/mnist.yaml`
- `scripts/benchmarks/configs/CMA_ES/waterworld.yaml`
- `scripts/benchmarks/configs/CMA_ES/waterworld_ma.yaml`
- `scripts/benchmarks/configs/CMA_ES_JAX/cartpole_easy.yaml`
- `scripts/benchmarks/configs/CMA_ES_JAX/cartpole_hard.yaml`
- `scripts/benchmarks/configs/CMA_ES_JAX/mnist.yaml`
- `scripts/benchmarks/configs/CMA_ES_JAX/waterworld.yaml`
- `scripts/benchmarks/configs/CRFMNES/brax_ant.yaml`
- `scripts/benchmarks/configs/CRFMNES/cartpole_easy.yaml`
- `scripts/benchmarks/configs/CRFMNES/cartpole_hard.yaml`
- `scripts/benchmarks/configs/CRFMNES/mnist.yaml`
- `scripts/benchmarks/configs/CRFMNES/slimevolley.yaml`
- `scripts/benchmarks/configs/CRFMNES/waterworld.yaml`
- `scripts/benchmarks/configs/CRFMNES/waterworld_ma.yaml`
- `scripts/benchmarks/configs/FCRFMC/brax_ant.yaml`
- `scripts/benchmarks/configs/FCRFMC/cartpole_easy.yaml`
- `scripts/benchmarks/configs/FCRFMC/cartpole_hard.yaml`
- `scripts/benchmarks/configs/FCRFMC/mnist.yaml`
- `scripts/benchmarks/configs/FCRFMC/slimevolley.yaml`
- `scripts/benchmarks/configs/FCRFMC/waterworld.yaml`
- `scripts/benchmarks/configs/FCRFMC/waterworld_ma.yaml`
- `scripts/benchmarks/configs/FPGPEC/brax_ant.yaml`
- `scripts/benchmarks/configs/FPGPEC/cartpole_easy.yaml`
- `scripts/benchmarks/configs/FPGPEC/cartpole_hard.yaml`
- `scripts/benchmarks/configs/FPGPEC/mnist.yaml`
- `scripts/benchmarks/configs/FPGPEC/search.yaml`
- `scripts/benchmarks/configs/FPGPEC/slimevolley.yaml`
- `scripts/benchmarks/configs/FPGPEC/waterworld.yaml`
- `scripts/benchmarks/configs/FPGPEC/waterworld_ma.yaml`
- `scripts/benchmarks/configs/OpenES/brax_ant.yaml`
- `scripts/benchmarks/configs/OpenES/cartpole_easy.yaml`
- `scripts/benchmarks/configs/OpenES/cartpole_hard.yaml`
- `scripts/benchmarks/configs/OpenES/mnist.yaml`
- `scripts/benchmarks/configs/OpenES/search.yaml`
- `scripts/benchmarks/configs/PGPE/brax_ant.yaml`
- `scripts/benchmarks/configs/PGPE/brax_test.yaml`
- `scripts/benchmarks/configs/PGPE/cartpole_easy.yaml`
- `scripts/benchmarks/configs/PGPE/cartpole_hard.yaml`
- `scripts/benchmarks/configs/PGPE/mnist.yaml`
- `scripts/benchmarks/configs/PGPE/search.yaml`
- `scripts/benchmarks/configs/PGPE/waterworld.yaml`
- `scripts/benchmarks/configs/PGPE/waterworld_ma.yaml`
- `scripts/benchmarks/configs/Sep_CMA_ES/brax_ant.yaml`
- `scripts/benchmarks/configs/Sep_CMA_ES/cartpole_easy.yaml`
- `scripts/benchmarks/configs/Sep_CMA_ES/cartpole_hard.yaml`
- `scripts/benchmarks/configs/Sep_CMA_ES/mnist.yaml`
- `scripts/benchmarks/configs/Sep_CMA_ES/waterworld.yaml`
- `scripts/benchmarks/configs/Sep_CMA_ES/waterworld_ma.yaml`
- `scripts/benchmarks/configs/iAMaLGaM/brax_ant.yaml`
- `scripts/benchmarks/configs/iAMaLGaM/cartpole_easy.yaml`
- `scripts/benchmarks/configs/iAMaLGaM/cartpole_hard.yaml`
- `scripts/benchmarks/configs/iAMaLGaM/mnist.yaml`
- `scripts/benchmarks/configs/iAMaLGaM/waterworld.yaml`
- `scripts/benchmarks/configs/iAMaLGaM/waterworld_ma.yaml`
- `scripts/benchmarks/problems.py`
- `scripts/benchmarks/train.py`
- `scripts/benchmarks/viz_grid.ipynb`

## Obsolete root probes and tests (20)

- `analyze_images.py`
- `check_translation.py`
- `fix_q_trunk.py`
- `measure_blood.py`
- `measure_separation.py`
- `see_images.py`
- `test_adapter.py`
- `test_adapter2.py`
- `test_fix_adapter.py`
- `test_fix_adapter_jax.py`
- `test_q_grad.py`
- `test_receptive_field.py`
- `test_resize.py`
- `test_shapes.py`
- `test_sigmoid.py`
- `test_split.py`
- `tests/test_algo.py`
- `tests/test_import.py`
- `tests/test_init.py`
- `tests/test_init_extra.py`

## Redundant environment and publishing configuration (3)

- `.github/workflows/publish-to-pypi.yml`
- `.github/workflows/publish-to-testpypi.yml`
- `Pipfile`

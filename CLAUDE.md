# Research fork working notes

Read [README.md](README.md), [IMDB_PLAN.md](IMDB_PLAN.md), and
[docs/CODE_REVIEW.md](docs/CODE_REVIEW.md) before changes.

This branch contains a reduced PGPE/HyperNetwork core, not an image trainer.
Keep cultural influence out of optimizer center/sigma updates. Preserve raw
metric signs, archive provenance, common-random-number evaluation, and full
checkpoint state. Do not claim frozen representations make selected tokens
causally faithful without interventions.

Use the existing environment for CPU checks:
`JAX_PLATFORMS=cpu new-env/bin/python -m unittest discover -s tests -v`.

Do not launch experiments or download models/datasets without an explicit task.
Preserve papers, results, checkpoints, local environments and unrelated changes.

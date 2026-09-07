# Core smoke example

From the repository root:

```bash
JAX_PLATFORMS=cpu new-env/bin/python -m examples.smoke_core
```

This checks a tiny analytic objective through population evaluation, PGPE,
signed fitness shaping, archive retention and metric history. It is not an
IMDb or BloodMNIST run. Optionally add `--checkpoint /tmp/core-smoke.msgpack`
to save the completed state; an existing file at that exact path is replaced.

The retired image/RL examples are available in commit `ee1bf0a`.

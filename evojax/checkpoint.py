"""Atomic, versioned checkpoints for the retained PGPE research core.

No pickle execution and no implicit migration of BloodMNIST experiments.
The caller must supply all external state: task/evaluation PRNG keys,
trainable heads, fixed panels, cultural memory, and configuration metadata.
"""

import os
from pathlib import Path
import tempfile

from flax import serialization


def save_checkpoint(path, solver, *, extra_state=None):
    """Atomically replace one checkpoint after a completed PGPE generation."""
    path = Path(path)
    payload = dict(version=1, solver=solver.save_state(), extra_state=extra_state)
    # Validate/serialize before creating a file or replacing a good checkpoint.
    data = serialization.msgpack_serialize(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_checkpoint(path, solver):
    """Validate/restore PGPE and return the caller's external state.

    Optimizer configuration and all optimizer arrays must match. Callers
    should validate their returned external state before resuming a task.
    """
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    if not isinstance(payload, dict) or set(payload) != {"version", "solver", "extra_state"} or payload["version"] != 1:
        raise ValueError("unsupported checkpoint schema")
    solver.load_state(payload["solver"])
    return payload["extra_state"]

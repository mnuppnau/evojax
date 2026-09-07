"""Explicit, bounded metric history, independent of any image task."""

from collections import deque

import numpy as np


class MetricHistory:
    """Fit slopes against actual generation numbers, excluding unwritten slots."""

    def __init__(self, capacity=100):
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or capacity < 2:
            raise ValueError("history capacity must be an integer >= 2")
        self.capacity = int(capacity)
        self._rows = deque(maxlen=self.capacity)
        self._names = None

    def append(self, generation, metrics):
        if isinstance(generation, bool) or not isinstance(generation, (int, np.integer)) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if self._rows and generation <= self._rows[-1][0]:
            raise ValueError("history generations must strictly increase")
        if not metrics or any(not isinstance(k, str) or not k for k in metrics):
            raise ValueError("metrics must have nonempty string names")
        if self._names is not None and set(metrics) != self._names:
            raise ValueError("history metric schema cannot change")
        values = {k: float(v) for k, v in metrics.items()}
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("history metrics must be finite")
        self._names = set(values)
        self._rows.append((int(generation), values))

    def slope(self, name, window):
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or not 2 <= window <= self.capacity:
            raise ValueError("slope window must be an integer in [2, capacity]")
        if self._names is not None and name not in self._names:
            raise KeyError(name)
        if len(self._rows) < window:
            return 0.0
        rows = list(self._rows)[-window:]
        # Subtract the origin before converting to float, so long-running
        # generation counters do not lose resolution during regression.
        x = np.array([step - rows[0][0] for step, _ in rows], dtype=np.float64)
        y = np.array([metrics[name] for _, metrics in rows], dtype=np.float64)
        x -= x.mean()
        return float(x @ (y - y.mean()) / (x @ x))

    def save_state(self):
        return dict(version=1, capacity=self.capacity,
                    rows=[dict(generation=g, metrics=dict(m)) for g, m in self._rows])

    def load_state(self, state):
        if (set(state) != {"version", "capacity", "rows"} or state["version"] != 1
                or state["capacity"] != self.capacity
                or not isinstance(state["rows"], list)
                or len(state["rows"]) > self.capacity):
            raise ValueError("incompatible history checkpoint")
        candidate = MetricHistory(self.capacity)
        for row in state["rows"]:
            if set(row) != {"generation", "metrics"}:
                raise ValueError("invalid history row")
            candidate.append(row["generation"], row["metrics"])
        self._rows, self._names = candidate._rows, candidate._names

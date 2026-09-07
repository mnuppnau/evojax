"""Bounded elite archives with explicit metric direction and provenance.

Archive records are selected directly from the evaluated population. Only
stable raw measurements belong in cross-generation comparisons: neither
within-population ranks nor scores from changing evaluators are comparable.
"""

from dataclasses import dataclass
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ArchiveRecord:
    params: object
    metrics: object
    generation: int
    population_index: int


def _pareto_order(costs):
    """Minimization fronts, then crowding distance, then newest record."""
    dominates = (costs[:, None] <= costs[None, :]).all(axis=-1)
    dominates &= (costs[:, None] < costs[None, :]).any(axis=-1)
    remaining = np.ones(len(costs), dtype=bool)
    order = []
    while remaining.any():
        front = np.flatnonzero(remaining & ~dominates[remaining].any(axis=0))
        distance = np.zeros(len(front))
        for column in costs.T:
            values = column[front]
            spread = np.ptp(values)
            if spread == 0:
                continue
            indices = np.argsort(values, kind="stable")
            distance[indices[[0, -1]]] = np.inf
            if len(front) > 2:
                distance[indices[1:-1]] += (
                    values[indices[2:]] - values[indices[:-2]]
                ) / spread
        local_order = np.lexsort((-front, -distance))
        order.extend(front[local_order].tolist())
        remaining[front] = False
    return order


class ParetoArchive:
    """Retain elites by Pareto front and crowding distance.

    Each objective maps its name to True (maximize) or False (minimize).
    Empty slots do not exist, so initialization cannot supply bogus elites.
    evaluator_id must identify a fixed evaluator/protocol; change it by
    creating a new archive or reevaluating old records, not relabeling them.
    No search-distribution standard deviations are retained or restored.
    """

    def __init__(self, objectives, *, evaluator_id, capacity=20):
        if not objectives or any(not isinstance(k, str) or not k for k in objectives):
            raise ValueError("objectives must have nonempty string names")
        if any(not isinstance(v, (bool, np.bool_)) for v in objectives.values()):
            raise ValueError("each objective direction must be a boolean")
        if not isinstance(evaluator_id, str) or not evaluator_id:
            raise ValueError("a nonempty evaluator_id is required")
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.objectives = MappingProxyType({k: bool(v) for k, v in objectives.items()})
        self.evaluator_id = evaluator_id
        self.capacity = int(capacity)
        self._records = []
        self._param_size = None

    @property
    def records(self):
        return tuple(self._records)

    def add_population(self, population, fitness, metrics, *, generation, evaluator_id):
        """Archive the exact row selected by the generation's scalar fitness.

        Capture population and metrics from the same evaluation, before a
        subsequent ask. Metrics are raw objective values; fitness is used only
        to select this generation's candidate, never to compare generations.
        """
        if evaluator_id != self.evaluator_id:
            raise ValueError("cannot compare archive records from different evaluators")
        if isinstance(generation, bool) or not isinstance(generation, (int, np.integer)) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if not np.isrealobj(population) or not np.isrealobj(fitness):
            raise ValueError("population and fitness must be real-valued")
        population = np.asarray(population, dtype=np.float32)
        fitness = np.asarray(fitness)
        if population.ndim != 2 or min(population.shape) < 1 or not np.isfinite(population).all():
            raise ValueError("population must be a finite, nonempty matrix")
        if fitness.shape != (len(population),) or not np.isfinite(fitness).all():
            raise ValueError("fitness must be finite with one value per population row")
        if self._param_size is not None and population.shape[1] != self._param_size:
            raise ValueError("archive parameter dimension cannot change")
        if set(metrics) != set(self.objectives):
            raise ValueError("metrics must match the archive's objective schema")
        metric_arrays = {k: np.asarray(v) for k, v in metrics.items()}
        if any(not np.isrealobj(v) or v.shape != fitness.shape
               or not np.isfinite(v).all() for v in metric_arrays.values()):
            raise ValueError("archive metrics must be finite population vectors")
        index = int(np.argmax(fitness))
        record = ArchiveRecord(
            params=jnp.asarray(population[index].copy()),
            metrics=MappingProxyType({k: float(v[index]) for k, v in metric_arrays.items()}),
            generation=int(generation), population_index=index,
        )
        records = self._records + [record]
        # Preserve the SIGN of each objective. abs(Q) would prefer near-zero
        # Q to a better positive Q, the defect in the BloodMNIST archives.
        costs = np.array([
            [(-1 if maximize else 1) * r.metrics[name]
             for name, maximize in self.objectives.items()]
            for r in records
        ])
        order = _pareto_order(costs)[:self.capacity]
        self._records = [records[i] for i in order]
        self._param_size = population.shape[1]
        return record

    def save_state(self):
        return dict(
            version=1, objectives=dict(self.objectives), evaluator_id=self.evaluator_id,
            capacity=self.capacity,
            records=[dict(params=np.array(r.params, copy=True), metrics=dict(r.metrics),
                          generation=r.generation, population_index=r.population_index)
                     for r in self._records],
        )

    def load_state(self, state):
        expected = {"version", "objectives", "evaluator_id", "capacity", "records"}
        if (set(state) != expected or state["version"] != 1
                or state["objectives"] != dict(self.objectives)
                or state["evaluator_id"] != self.evaluator_id
                or state["capacity"] != self.capacity):
            raise ValueError("incompatible archive checkpoint")
        if not isinstance(state["records"], list) or len(state["records"]) > self.capacity:
            raise ValueError("invalid archive records")
        records = []
        size = None
        for raw in state["records"]:
            if set(raw) != {"params", "metrics", "generation", "population_index"}:
                raise ValueError("invalid archive record schema")
            if not np.isrealobj(raw["params"]):
                raise ValueError("archive parameters must be real-valued")
            params = np.asarray(raw["params"], dtype=np.float32)
            if params.ndim != 1 or not params.size or not np.isfinite(params).all():
                raise ValueError("invalid archive parameters")
            if size is not None and params.size != size:
                raise ValueError("inconsistent archive parameter dimensions")
            size = params.size
            if set(raw["metrics"]) != set(self.objectives):
                raise ValueError("invalid archive metric names")
            metrics = {k: float(v) for k, v in raw["metrics"].items()}
            if not all(np.isfinite(v) for v in metrics.values()):
                raise ValueError("invalid archive metric values")
            for name in ("generation", "population_index"):
                value = raw[name]
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                    raise ValueError(f"invalid archive {name}")
            records.append(ArchiveRecord(
                jnp.asarray(params.copy()), MappingProxyType(metrics),
                int(raw["generation"]), int(raw["population_index"]),
            ))
        self._records, self._param_size = records, size

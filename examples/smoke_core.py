"""Dataset-free smoke check of the retained evolutionary research core."""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from evojax.algo import PGPE
from evojax.algo.cultural import MetricHistory, ParetoArchive
from evojax.checkpoint import save_checkpoint
from evojax.evaluation import evaluate_population
from evojax.fitness import combine_fitness


def measure(params, target, key):
    del key  # The analytic objective is deterministic.
    return {"loss": jnp.mean((params - target) ** 2)}


def run(steps=40, seed=42, checkpoint=None):
    if steps < 1:
        raise ValueError("steps must be positive")
    solver = PGPE(32, 8, init_params=jnp.ones(8), seed=seed,
                  center_learning_rate=0.05, init_stdev=0.1,
                  stdev_learning_rate=0.01, solution_ranking=False)
    archive = ParetoArchive({"loss": False}, evaluator_id="quadratic-zero-8d-v1")
    history = MetricHistory()
    target = jnp.zeros(8)
    evaluation_key = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
    initial = float(measure(solver.center, target, evaluation_key)["loss"])
    for _ in range(steps):
        population = solver.ask()
        evaluation_key, key = jax.random.split(evaluation_key)
        metrics = evaluate_population(measure, population, target, key)
        fitness = combine_fitness(metrics, {"loss": 1.0}, maximize={"loss": False})
        archive.add_population(population, fitness, metrics,
                               generation=solver.iteration,
                               evaluator_id=archive.evaluator_id)
        solver.tell(fitness)
        history.append(solver.iteration, {"loss": float(jnp.min(metrics["loss"]))})
    final = float(measure(solver.center, target, evaluation_key)["loss"])
    if checkpoint is not None:
        save_checkpoint(checkpoint, solver, extra_state={
            "evaluation_key": np.asarray(evaluation_key),
            "archive": archive.save_state(), "history": history.save_state(),
            "task": archive.evaluator_id,
        })
    print(f"CPU-compatible smoke: {steps} updates; center loss {initial:.6f} -> {final:.6f}")
    print(f"Valid archive records: {len(archive.records)}; sigma mean: {float(solver.stdev.mean()):.6f}")
    return initial, final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", help="Optional new-format checkpoint path")
    args = parser.parse_args()
    run(args.steps, args.seed, args.checkpoint)

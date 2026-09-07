from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from evojax.algo import PGPE
from evojax.algo.cultural import MetricHistory, ParetoArchive
from evojax.checkpoint import load_checkpoint, save_checkpoint
from evojax.evaluation import evaluate_population


def noisy_score(params, batch, key):
    return {"score": jnp.dot(params, batch) + jax.random.normal(key)}


class CheckpointEvaluationTests(unittest.TestCase):
    def test_population_receives_common_batch_and_random_key(self):
        population = jnp.tile(jnp.array([1.0, 2.0]), (8, 1))
        batch = jnp.array([2.0, 3.0])
        first = evaluate_population(noisy_score, population, batch, jax.random.PRNGKey(1))
        np.testing.assert_array_equal(first["score"], np.repeat(first["score"][0], 8))
        again = evaluate_population(noisy_score, population, batch, jax.random.PRNGKey(1))
        np.testing.assert_array_equal(first["score"], again["score"])
        other = evaluate_population(noisy_score, population, batch, jax.random.PRNGKey(2))
        self.assertFalse(np.array_equal(first["score"], other["score"]))

    def test_disk_roundtrip_preserves_rng_optimizer_and_external_state(self):
        original = PGPE(8, 3, seed=21)
        population = original.ask()
        original.tell(-jnp.square(population).sum(axis=1))
        archive = ParetoArchive({"q": True}, evaluator_id="fixed")
        archive.add_population(population, np.arange(8.0), {"q": np.arange(8.0)},
                               generation=0, evaluator_id="fixed")
        history = MetricHistory()
        history.append(0, {"q": 1.0})
        extra = {"training_key": np.asarray(jax.random.PRNGKey(5)),
                 "archive": archive.save_state(), "history": history.save_state()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.msgpack"
            save_checkpoint(path, original, extra_state=extra)
            restored = PGPE(8, 3, seed=999)
            loaded = load_checkpoint(path, restored)
            self.assertEqual(restored.iteration, 1)
            np.testing.assert_array_equal(original.ask(), restored.ask())
            np.testing.assert_array_equal(loaded["training_key"], extra["training_key"])
            copy_archive = ParetoArchive({"q": True}, evaluator_id="fixed")
            copy_archive.load_state(loaded["archive"])
            np.testing.assert_array_equal(copy_archive.records[0].params, population[-1])
            MetricHistory().load_state(loaded["history"])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_save_preserves_previous_checkpoint(self):
        solver = PGPE(4, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.msgpack"
            save_checkpoint(path, solver)
            previous = path.read_bytes()
            solver.ask()
            with self.assertRaises(RuntimeError):
                save_checkpoint(path, solver)
            self.assertEqual(path.read_bytes(), previous)

    def test_incompatible_checkpoint_does_not_modify_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.msgpack"
            save_checkpoint(path, PGPE(4, 2))
            solver = PGPE(8, 2, init_params=[1, 2])
            with self.assertRaises(ValueError):
                load_checkpoint(path, solver)
            np.testing.assert_array_equal(solver.center, [1, 2])

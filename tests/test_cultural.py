import copy
import unittest

import numpy as np

from evojax.algo import PGPE
from evojax.algo.cultural import MetricHistory, ParetoArchive


class ArchiveTests(unittest.TestCase):
    def test_every_antithetic_index_records_the_evaluated_row(self):
        population = PGPE(8, 3, seed=123).ask()
        for index in range(8):
            archive = ParetoArchive({"q": True}, evaluator_id="frozen-v1")
            fitness = np.zeros(8)
            fitness[index] = 1
            record = archive.add_population(population, fitness, {"q": np.arange(8.0)},
                                            generation=0, evaluator_id="frozen-v1")
            np.testing.assert_array_equal(record.params, population[index])
            self.assertEqual(record.metrics["q"], index)
            self.assertEqual(record.population_index, index)

    def test_positive_q_is_preferred_to_zero(self):
        archive = ParetoArchive({"q": True}, evaluator_id="fixed-q", capacity=1)
        for generation, q in enumerate([-0.1, 0.001, 0.09, 0.0]):
            archive.add_population([[generation]], [1.0], {"q": [q]},
                                   generation=generation, evaluator_id="fixed-q")
        self.assertEqual(archive.records[0].metrics["q"], 0.09)

    def test_minimization_and_entropy_direction_are_explicit(self):
        archive = ParetoArchive({"loss": False, "entropy": True}, evaluator_id="fixed", capacity=1)
        archive.add_population([[0]], [1], {"loss": [2], "entropy": [0]},
                               generation=0, evaluator_id="fixed")
        archive.add_population([[1]], [1], {"loss": [1], "entropy": [1]},
                               generation=1, evaluator_id="fixed")
        np.testing.assert_array_equal(archive.records[0].params, [1])

    def test_empty_archive_has_no_sentinel_candidates(self):
        archive = ParetoArchive({"q": True}, evaluator_id="fixed")
        self.assertEqual(len(archive.records), 0)
        archive.add_population([[0, 0]], [0], {"q": [-100]},
                               generation=0, evaluator_id="fixed")
        self.assertEqual(len(archive.records), 1)
        self.assertEqual(archive.records[0].metrics["q"], -100)
        self.assertFalse(hasattr(archive.records[0], "stdev"))

    def test_pareto_capacity_retains_tradeoff_extremes(self):
        archive = ParetoArchive({"a": True, "b": True}, evaluator_id="fixed", capacity=2)
        for i, values in enumerate([(1, 0), (0, 1), (0.1, 0.1), (-1, -1)]):
            archive.add_population([[i]], [1], {"a": [values[0]], "b": [values[1]]},
                                   generation=i, evaluator_id="fixed")
        self.assertEqual({float(r.params[0]) for r in archive.records}, {0, 1})

    def test_checkpoint_preserves_records_and_rejects_mixed_evaluators(self):
        archive = ParetoArchive({"q": True}, evaluator_id="fixed")
        archive.add_population([[1, 2]], [1], {"q": [0.09]}, generation=3, evaluator_id="fixed")
        restored = ParetoArchive({"q": True}, evaluator_id="fixed")
        restored.load_state(archive.save_state())
        np.testing.assert_array_equal(restored.records[0].params, [1, 2])
        self.assertEqual(restored.records[0].generation, 3)
        with self.assertRaises(ValueError):
            restored.add_population([[2, 3]], [1], {"q": [0.1]}, generation=4, evaluator_id="changed")
        self.assertEqual(len(restored.records), 1)

    def test_bad_measurements_cannot_mutate_archive(self):
        archive = ParetoArchive({"q": True}, evaluator_id="fixed")
        with self.assertRaises(ValueError):
            archive.add_population([[1], [2]], [1, 2], {"q": [0, np.nan]},
                                   generation=0, evaluator_id="fixed")
        self.assertEqual(len(archive.records), 0)
        archive.add_population([[1]], [1], {"q": [0.09]}, generation=0, evaluator_id="fixed")
        state = copy.deepcopy(archive.save_state())
        state["records"][0]["params"] = np.array([np.nan])
        with self.assertRaises(ValueError):
            archive.load_state(state)
        np.testing.assert_array_equal(archive.records[0].params, [1])

    def test_complex_archive_inputs_are_not_silently_projected(self):
        archive = ParetoArchive({"q": True}, evaluator_id="fixed")
        for population, fitness, metrics in [
            ([[1 + 2j]], [1], {"q": [1]}),
            ([[1]], [1 + 2j], {"q": [1]}),
            ([[1]], [1], {"q": [1 + 2j]}),
        ]:
            with self.assertRaises(ValueError):
                archive.add_population(population, fitness, metrics,
                                       generation=0, evaluator_id="fixed")
        self.assertEqual(len(archive.records), 0)


class HistoryTests(unittest.TestCase):
    def test_warmup_rollover_and_actual_generation_spacing(self):
        history = MetricHistory(4)
        self.assertEqual(history.slope("q", 3), 0)
        for generation in range(0, 80, 10):
            history.append(generation, {"q": generation * 2 + 5})
        self.assertAlmostEqual(history.slope("q", 4), 2)
        self.assertEqual(len(history.save_state()["rows"]), 4)

    def test_invalid_history_update_is_rejected(self):
        history = MetricHistory()
        history.append(1, {"q": 0.1})
        for step, metrics in [(1, {"q": 0.2}), (2, {"q": np.nan}), (2, {"different": 1})]:
            with self.assertRaises(ValueError):
                history.append(step, metrics)
        self.assertEqual(len(history.save_state()["rows"]), 1)
        with self.assertRaises(ValueError):
            history.slope("q", 1)
        with self.assertRaises(KeyError):
            history.slope("typo", 20)

    def test_roundtrip_keeps_history_and_schema(self):
        history = MetricHistory(4)
        for i in range(6):
            history.append(i, {"q": 3 * i})
        restored = MetricHistory(4)
        restored.load_state(history.save_state())
        self.assertAlmostEqual(restored.slope("q", 4), 3)
        restored.append(6, {"q": 18})
        self.assertAlmostEqual(restored.slope("q", 4), 3)

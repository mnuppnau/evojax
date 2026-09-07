"""Regression tests for bugs found while retiring the BloodMNIST trainer."""

import copy
import unittest

import jax.numpy as jnp
import numpy as np

from evojax.algo import PGPE
from evojax.algo.pgpe import compute_reinforce_update, update_stdev


class PGPETests(unittest.TestCase):
    def test_antithetic_pairs_are_interleaved(self):
        center = np.array([2.0, -3.0, 1.0])
        solver = PGPE(8, 3, init_params=center, seed=17)
        population = np.asarray(solver.ask())
        np.testing.assert_allclose((population[::2] + population[1::2]) / 2, np.tile(center, (4, 1)))
        self.assertFalse(np.array_equal(population[0], population[2]))

    def test_smallest_population_preserves_gradient_axes(self):
        mu, sigma = compute_reinforce_update(
            jnp.array([3.0, 1.0]), jnp.array([[0.2, -0.1]]), jnp.ones(2),
        )
        np.testing.assert_allclose(mu, [0.2, -0.1])
        np.testing.assert_array_equal(sigma, [0.0, 0.0])

    def test_adam_uses_actual_step_and_separate_decay_schedule(self):
        beta1, beta2, epsilon, lr = 0.8, 0.95, 1e-7, 0.01
        solver = PGPE(
            2, 1, init_stdev=0.3, solution_ranking=False,
            center_learning_rate=lr, stdev_learning_rate=0.0,
            optimizer_config=dict(beta1=beta1, beta2=beta2, epsilon=epsilon,
                                  center_lr_decay_coef=0.5, center_lr_decay_steps=2),
        )
        expected = m = v = 0.0
        for t, gradient in enumerate([1.0, 2.0, -0.5, 1.0, -2.0]):
            population = np.asarray(solver.ask())
            perturbation = float(solver._scaled_noises[0, 0])
            solver.tell(np.array([gradient / perturbation, -gradient / perturbation]))
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            rate = lr * 0.5 ** (t // 2)
            expected += rate * (m / (1 - beta1 ** (t + 1))) / (
                np.sqrt(v / (1 - beta2 ** (t + 1))) + epsilon
            )
            self.assertEqual(population.shape, (2, 1))
            np.testing.assert_allclose(solver.center, [expected], atol=2e-7)

    def test_constant_ranked_fitness_adds_no_selection_pressure(self):
        solver = PGPE(8, 3)
        center, sigma = np.array(solver.center), np.array(solver.stdev)
        solver.ask()
        solver.tell(np.ones(8))
        np.testing.assert_array_equal(solver.center, center)
        np.testing.assert_array_equal(solver.stdev, sigma)

    def test_sgd_updates_and_resume_use_actual_steps(self):
        solver = PGPE(2, 1, optimizer="sgd", solution_ranking=False,
                      center_learning_rate=0.01, stdev_learning_rate=0)
        expected = 0.0
        for gradient in (1.0, -0.5, 2.0):
            solver.ask()
            noise = float(solver._scaled_noises[0, 0])
            solver.tell([gradient / noise, -gradient / noise])
            expected += 0.01 * gradient
            np.testing.assert_allclose(solver.center, [expected], atol=1e-7)
        restored = PGPE(2, 1, optimizer="sgd", solution_ranking=False,
                        center_learning_rate=0.01, stdev_learning_rate=0)
        restored.load_state(solver.save_state())
        np.testing.assert_array_equal(solver.ask(), restored.ask())

    def test_sigma_relative_and_absolute_limits(self):
        sigma = update_stdev(jnp.array([0.01, 9.9, 0.005]), 1.0,
                             jnp.array([-100.0, 100.0, -100.0]), 0.1, 0.005, 10.0)
        np.testing.assert_allclose(sigma, [0.009, 10.0, 0.005], rtol=1e-6)

    def test_vector_and_integer_initial_sigma(self):
        np.testing.assert_allclose(PGPE(4, 2, init_stdev=[0.1, 0.2]).stdev, [0.1, 0.2])
        np.testing.assert_allclose(PGPE(4, 2, init_stdev=1).stdev, [1.0, 1.0])

    def test_invalid_initialization_rejected(self):
        cases = [dict(pop_size=3), dict(pop_size=0), dict(param_size=0),
                 dict(init_stdev=[0.1]), dict(init_stdev=np.nan),
                 dict(init_stdev=0.0), dict(init_params=[1.0]),
                 dict(optimizer="clipup"), dict(center_learning_rate=-1),
                 dict(stdev_max_change=-0.1), dict(stdev_min=0),
                 dict(stdev_min=1e-60), dict(init_params=[1 + 2j, 3 + 4j]),
                 dict(init_stdev=0.1 + 0.1j),
                 dict(optimizer_config={"beta2": 1 - 1e-12}),
                 dict(optimizer_config={"center_lr_decay_steps": 0}),
                 dict(optimizer_config={"unknown": True})]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                PGPE(**(dict(pop_size=4, param_size=2) | case))

    def test_ask_tell_protocol_and_invalid_fitness_retry(self):
        solver = PGPE(4, 2)
        with self.assertRaises(RuntimeError):
            solver.tell(np.ones(4))
        solver.ask()
        for action in (solver.ask, solver.save_state):
            with self.assertRaises(RuntimeError):
                action()
        for scores in ([1.0], [1.0, 2.0, 3.0, np.nan], np.ones((4, 1))):
            with self.assertRaises(ValueError):
                solver.tell(scores)
        self.assertEqual(solver.iteration, 0)
        solver.tell(np.ones(4))
        self.assertEqual(solver.iteration, 1)
        with self.assertRaises(RuntimeError):
            solver.tell(np.ones(4))

    def test_setting_center_resets_adam_instead_of_snapping_back(self):
        solver = PGPE(4, 2)
        solver.ask()
        solver.tell(np.arange(4.0))
        solver.best_params = [5.0, -4.0]
        solver.ask()
        solver.tell(np.ones(4))
        np.testing.assert_array_equal(solver.center, [5.0, -4.0])

    def test_resume_matches_uninterrupted_population_and_update(self):
        original = PGPE(8, 3, seed=42)
        for _ in range(3):
            population = original.ask()
            original.tell(-jnp.square(population - 1).sum(axis=1))
        restored = PGPE(8, 3, seed=999)
        restored.load_state(original.save_state())
        for _ in range(3):
            a, b = original.ask(), restored.ask()
            np.testing.assert_array_equal(a, b)
            original.tell(-jnp.square(a - 1).sum(axis=1))
            restored.tell(-jnp.square(b - 1).sum(axis=1))
            np.testing.assert_array_equal(original.center, restored.center)
            np.testing.assert_array_equal(original.stdev, restored.stdev)

    def test_restore_rejects_bad_state_without_partial_mutation(self):
        solver = PGPE(4, 2)
        original = solver.save_state()
        variants = []
        for key, value in [("center", [1.0]), ("stdev", [0.0, 0.0]),
                           ("key", np.zeros(2)), ("iteration", -1)]:
            state = copy.deepcopy(original)
            state[key] = value
            variants.append(state)
        state = copy.deepcopy(original)
        state["config"]["center_learning_rate"] = 9.0
        variants.append(state)
        state = copy.deepcopy(original)
        state["opt_state"]["0"]["count"] = np.array(3, dtype=np.int32)
        variants.append(state)
        for state in variants:
            with self.subTest(state_key=list(state)), self.assertRaises(ValueError):
                solver.load_state(state)
            np.testing.assert_array_equal(solver.center, original["center"])
            self.assertEqual(solver.iteration, 0)

    def test_save_state_is_an_independent_snapshot(self):
        solver = PGPE(4, 2)
        state = solver.save_state()
        state["center"][0] = 99
        state["config"]["optimizer_config"]["beta1"] = 0
        self.assertEqual(float(solver.center[0]), 0)
        self.assertEqual(solver.save_state()["config"]["optimizer_config"]["beta1"], 0.9)

    def test_short_quadratic_optimization_improves(self):
        solver = PGPE(32, 8, init_params=np.ones(8), center_learning_rate=0.05)
        before = float(jnp.square(solver.center).sum())
        for _ in range(40):
            population = solver.ask()
            solver.tell(-jnp.square(population).sum(axis=1))
        self.assertLess(float(jnp.square(solver.center).sum()), before * 0.25)

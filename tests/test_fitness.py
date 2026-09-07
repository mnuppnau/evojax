import unittest

import jax.numpy as jnp
import numpy as np

from evojax.fitness import combine_fitness, rank_normalize


class FitnessTests(unittest.TestCase):
    def test_ties_receive_average_rank(self):
        np.testing.assert_allclose(rank_normalize(jnp.array([2.0, 1.0, 2.0, 4.0])),
                                   [0.0, -1.0, 0.0, 1.0])

    def test_constant_and_singleton_scores_are_neutral(self):
        np.testing.assert_array_equal(rank_normalize(jnp.ones(512)), np.zeros(512))
        np.testing.assert_array_equal(rank_normalize(jnp.array([7.0])), [0.0])

    def test_permutations_preserve_ranks_and_ties(self):
        values = np.array([3.0, -2.0, 3.0, 5.0, -2.0])
        permutation = np.array([4, 0, 2, 1, 3])
        np.testing.assert_array_equal(rank_normalize(values[permutation]),
                                      np.asarray(rank_normalize(values))[permutation])

    def test_invalid_kernel_entries_cannot_win(self):
        np.testing.assert_array_equal(rank_normalize(jnp.array([1.0, 2.0, np.nan, np.inf])),
                                      [-1.0, 1.0, -1.0, -1.0])
        np.testing.assert_array_equal(rank_normalize(jnp.array([np.nan])), [-1.0])

    def test_signed_objectives_and_zero_weight_diagnostic(self):
        scores = combine_fitness(
            {"q": [0.001, 0.09], "loss": [2.0, 1.0], "unused": [np.nan]},
            {"q": 0.3, "loss": 0.7, "unused": 0},
            maximize={"q": True, "loss": False},
        )
        np.testing.assert_allclose(scores, [-1.0, 1.0])

    def test_composite_weighting_is_not_ranked_again(self):
        scores = combine_fitness({"a": [0, 1, 2], "b": [2, 0, 1]}, {"a": 0.8, "b": 0.2})
        np.testing.assert_allclose(scores, [-0.6, -0.2, 0.8], atol=1e-7)

    def test_unsigned_loss_is_minimized_without_wraparound(self):
        scores = combine_fitness({"loss": np.array([0, 1, 2], dtype=np.uint32)},
                                 {"loss": 1}, maximize={"loss": False})
        np.testing.assert_array_equal(scores, [1, 0, -1])

    def test_large_integer_scores_keep_distinct_order(self):
        values = np.array([2 ** 25, 2 ** 25 + 1, 2 ** 25 + 2], dtype=np.int32)
        np.testing.assert_array_equal(rank_normalize(values), [-1, 0, 1])

    def test_complex_measurements_are_rejected(self):
        with self.assertRaises(ValueError):
            rank_normalize(jnp.array([1 + 2j, 3 + 4j]))
        with self.assertRaises(ValueError):
            combine_fitness({"q": [1 + 2j, 3 + 4j]}, {"q": 1})

    def test_bad_active_components_fail_fast(self):
        for components, weights in [
            ({"a": [1, np.nan]}, {"a": 1}),
            ({"a": [1, 2]}, {"a": -1}),
            ({"a": [1, 2]}, {"a": 0}),
            ({"a": [1, 2]}, {"missing": 1}),
            ({"a": [1, 2], "b": [1]}, {"a": 1, "b": 1}),
        ]:
            with self.subTest(components=components), self.assertRaises(ValueError):
                combine_fitness(components, weights)

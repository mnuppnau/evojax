import unittest

import jax
import jax.numpy as jnp
import numpy as np

from evojax.policy import ParameterAdapter


class HyperNetworkTests(unittest.TestCase):
    def test_per_tensor_padding_never_shifts_later_tensors(self):
        target = {"a": jnp.arange(5.0), "b": jnp.array(8.0),
                  "c": jnp.arange(9.0).reshape(3, 3)}
        adapter = ParameterAdapter(target, chunk_size=4)
        chunks = jnp.array([
            [0, 1, 2, 3], [4, 999, 999, 999],
            [8, 999, 999, 999],
            [0, 1, 2, 3], [4, 5, 6, 7], [8, 999, 999, 999],
        ], dtype=jnp.float32)
        rebuilt = jax.jit(adapter.reconstruct)(chunks)
        for name in target:
            np.testing.assert_array_equal(rebuilt[name], target[name])
            self.assertEqual(rebuilt[name].shape, target[name].shape)

    def test_seed_reproducibility_jit_vmap_and_gradients(self):
        adapter = ParameterAdapter({"weight": jnp.zeros((3, 2)), "bias": jnp.zeros(2)},
                                   chunk_size=4, hidden_dim=8, chunk_embed_dim=4)
        first = adapter.init_hypernet(jax.random.PRNGKey(1))
        repeated = adapter.init_hypernet(jax.random.PRNGKey(1))
        other = adapter.init_hypernet(jax.random.PRNGKey(2))
        out = jax.jit(adapter.generate_params)(first)
        again = adapter.generate_params(repeated)
        np.testing.assert_array_equal(out["weight"], again["weight"])
        self.assertFalse(np.array_equal(out["weight"], adapter.generate_params(other)["weight"]))
        population = jax.tree_util.tree_map(lambda a, b: jnp.stack((a, b)), first, other)
        batched = jax.jit(jax.vmap(adapter.generate_params))(population)
        self.assertEqual(batched["weight"].shape, (2, 3, 2))
        grads = jax.grad(lambda p: sum(jnp.square(v).sum()
                        for v in jax.tree_util.tree_leaves(adapter.generate_params(p))))(first)
        leaves = jax.tree_util.tree_leaves(grads)
        self.assertTrue(all(np.isfinite(g).all() for g in leaves))
        self.assertTrue(any(np.any(g != 0) for g in leaves))

    def test_declared_context_and_dtypes_are_preserved(self):
        adapter = ParameterAdapter({"a": jnp.ones(2, dtype=jnp.float16)},
                                   chunk_size=4, tensor_context=[[1, 0, 0]])
        self.assertEqual(adapter.static_context.shape, (1, 4))
        reconstructed = adapter.reconstruct(jnp.ones((1, 4)))
        self.assertEqual(reconstructed["a"].dtype, jnp.float16)

    def test_invalid_trees_chunks_and_context_fail(self):
        for tree, options in [
            ({}, {}), ({"x": jnp.ones(2, dtype=jnp.int32)}, {}),
            ({"x": jnp.zeros(0)}, {}), ({"x": jnp.ones(2)}, {"chunk_size": 0}),
            ({"x": jnp.ones(2)}, {"tensor_context": [[np.nan]]}),
            ({"x": jnp.ones(2)}, {"tensor_context": [[1 + 2j]]}),
            ({"x": jnp.ones(2)}, {"tensor_context": [[1], [2]]}),
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                ParameterAdapter(tree, **options)
        adapter = ParameterAdapter({"x": jnp.ones(2)}, chunk_size=4)
        with self.assertRaises(ValueError):
            adapter.reconstruct(jnp.ones((2, 4)))

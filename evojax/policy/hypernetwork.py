# Copyright 2022 The EvoJAX Authors. Licensed under Apache-2.0.
"""Chunked indirect encoding extracted from the BloodMNIST generator.

No image architecture, latent-code dimensions, or pretrained model is
embedded here. The caller owns model initialization and its random key.
"""

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class HyperNetwork(nn.Module):
    chunk_size: int
    n_chunks: int
    chunk_embed_dim: int = 16
    hidden_dim: int = 48

    @nn.compact
    def __call__(self, chunk_ids, context):
        embedded = nn.Embed(self.n_chunks, self.chunk_embed_dim)(chunk_ids)
        x = jnp.concatenate((embedded, context), axis=-1)
        x = nn.gelu(nn.Dense(self.hidden_dim)(x))
        x = nn.gelu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(
            self.chunk_size, kernel_init=jax.nn.initializers.normal(stddev=0.01),
        )(x)


class ParameterAdapter:
    """Emit a floating-point parameter tree with independent tensor padding.

    Default context is tensor identity plus two historical positional
    coordinates (depth and scale), NOT measured network depth/resolution.
    Supply tensor_context shaped (number_of_tensors, context_width) to replace
    those coordinates with declared structural features such as module role.
    Tensor order follows JAX tree flattening, not execution order.
    """

    def __init__(self, target_init_params, chunk_size=512, *, tensor_context=None,
                 chunk_embed_dim=16, hidden_dim=48):
        for name, value in (("chunk_size", chunk_size), ("chunk_embed_dim", chunk_embed_dim),
                            ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        leaves, self.target_tree = jax.tree_util.tree_flatten(target_init_params)
        if not leaves:
            raise ValueError("target parameter tree must not be empty")
        leaves = [jnp.asarray(leaf) for leaf in leaves]
        if any(not leaf.size or not jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in leaves):
            raise ValueError("target leaves must be nonempty floating-point tensors")
        self.chunk_size = int(chunk_size)
        self.param_shapes = tuple(leaf.shape for leaf in leaves)
        self.param_dtypes = tuple(leaf.dtype for leaf in leaves)
        self.param_sizes = tuple(int(leaf.size) for leaf in leaves)
        self.num_target_params = sum(self.param_sizes)
        counts = [(size + self.chunk_size - 1) // self.chunk_size for size in self.param_sizes]
        self.layer_chunks_split_indices = tuple(np.cumsum(counts)[:-1].tolist())
        self.total_chunks = sum(counts)
        self.N_LAYERS = len(leaves)
        self.N_CHUNKS = max(counts)
        self.chunk_ids = jnp.asarray(np.concatenate([np.arange(n) for n in counts]), dtype=jnp.int32)
        self.layer_ids = jnp.asarray(np.repeat(np.arange(len(leaves)), counts), dtype=jnp.int32)
        if tensor_context is None:
            tensor_context = np.column_stack((
                np.linspace(0, 1, len(leaves)), np.linspace(0.25, 1, len(leaves)),
            ))
        if not np.isrealobj(tensor_context):
            raise ValueError("tensor_context must be real-valued")
        context = np.asarray(tensor_context, dtype=np.float32)
        if context.ndim != 2 or context.shape[0] != len(leaves) or not np.isfinite(context).all():
            raise ValueError("tensor_context must be finite with one row per target tensor")
        self.static_context = jnp.concatenate((
            jax.nn.one_hot(self.layer_ids, len(leaves)),
            jnp.asarray(context)[self.layer_ids],
        ), axis=-1)
        self.CONTEXT_DIM = self.static_context.shape[-1]
        self.hypernetwork = HyperNetwork(self.chunk_size, self.N_CHUNKS,
                                         int(chunk_embed_dim), int(hidden_dim))

    def init_hypernet(self, key):
        return self.hypernetwork.init(key, self.chunk_ids, self.static_context)

    def reconstruct(self, chunks):
        """Truncate padding separately per tensor before rebuilding its tree."""
        if chunks.shape != (self.total_chunks, self.chunk_size):
            raise ValueError("chunk matrix does not match the target parameter schema")
        parts = jnp.split(chunks, self.layer_chunks_split_indices, axis=0)
        leaves = [
            part.reshape(-1)[:size].reshape(shape).astype(dtype)
            for part, size, shape, dtype in zip(parts, self.param_sizes, self.param_shapes, self.param_dtypes)
        ]
        return jax.tree_util.tree_unflatten(self.target_tree, leaves)

    def generate_params(self, hypernet_params):
        chunks = self.hypernetwork.apply(hypernet_params, self.chunk_ids, self.static_context)
        return self.reconstruct(chunks)

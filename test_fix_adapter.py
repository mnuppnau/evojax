import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

param_sizes = [64, 4608, 64]
chunk_size = 512

# Create dummy chunks
total_chunks = sum((s + chunk_size - 1) // chunk_size for s in param_sizes)
dummy_chunks = jnp.arange(total_chunks * chunk_size).reshape(total_chunks, chunk_size)

# Old way (buggy)
raw_stream = dummy_chunks.reshape(-1)
split_indices = np.cumsum(param_sizes)[:-1]
total_gen_params = sum(param_sizes)
valid_stream = raw_stream[:total_gen_params]
param_list_old = jnp.split(valid_stream, split_indices)

print("Old Way:")
for i, p in enumerate(param_list_old):
    print(f"Layer {i} (size {param_sizes[i]}): start={p[0]}, end={p[-1]}")

# New way (correct)
# We need to reshape each layer's chunks back into a stream, THEN truncate.
layer_chunks_split = []
idx = 0
for size in param_sizes:
    n_chunks = (size + chunk_size - 1) // chunk_size
    layer_chunks_split.append(n_chunks)

chunks_per_layer = jnp.split(dummy_chunks, np.cumsum(layer_chunks_split)[:-1])

param_list_new = []
for i, chunks in enumerate(chunks_per_layer):
    flat = chunks.reshape(-1)
    param_list_new.append(flat[:param_sizes[i]])

print("\nNew Way:")
for i, p in enumerate(param_list_new):
    print(f"Layer {i} (size {param_sizes[i]}): start={p[0]}, end={p[-1]}")


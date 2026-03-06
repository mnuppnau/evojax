import jax
import jax.numpy as jnp
import numpy as np

param_sizes = [64, 4608, 64]
chunk_size = 512

layer_chunks_split = [(s + chunk_size - 1) // chunk_size for s in param_sizes]
split_indices = np.cumsum(layer_chunks_split)[:-1].tolist()

def generate_params(flat_chunks):
    chunks_per_layer = jnp.split(flat_chunks, split_indices)
    reshaped_params = []
    for i, chunks in enumerate(chunks_per_layer):
        flat = chunks.reshape(-1)
        reshaped_params.append(flat[:param_sizes[i]])
    return reshaped_params

dummy_chunks = jnp.arange(sum(layer_chunks_split) * chunk_size).reshape(-1, chunk_size)
res = jax.jit(generate_params)(dummy_chunks)
print("JIT compiled successfully.")
for i, p in enumerate(res):
    print(f"Layer {i}: {p[0]} to {p[-1]}")

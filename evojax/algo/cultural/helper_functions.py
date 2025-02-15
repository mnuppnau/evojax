import jax
import jax.numpy as jnp

from jax import lax
from jax import jit
from jax import random
from jax import vmap
from jax import ops 
from jax.lax import scan, fori_loop


@jax.jit
def non_dominated_sort_lax(objectives: jnp.ndarray) -> jnp.ndarray:
    """
    Perform non-dominated sorting on a set of points in multi-objective space,
    using jax.lax.while_loop for the iterative rank assignment.
    
    Args:
        objectives (jnp.ndarray): Array of shape (N, M), where
            N = number of points,
            M = number of objectives (assume minimization).
            
    Returns:
        jnp.ndarray of shape (N,):
            The integer Pareto rank of each point (0 = best/front, 1 = next front, etc.).
            Points that cannot be assigned (e.g., if a front is not found) remain at -1.
    """
    # --- Step A: Build the "dominates" matrix ---
    # dominates[i, j] = True if point i dominates point j (all dims <=, at least one dim <)
    less_equal = objectives[:, None, :] <= objectives[None, :, :]  # (N, N, M)
    strictly_less = objectives[:, None, :] < objectives[None, :, :]  # (N, N, M)
    all_le = jnp.all(less_equal, axis=-1)    # (N, N)
    any_lt = jnp.any(strictly_less, axis=-1) # (N, N)
    dominates = jnp.logical_and(all_le, any_lt)  # (N, N)
    
    # --- Step B: Iteratively identify Pareto layers using lax.while_loop ---
    N = objectives.shape[0]
    init_ranks = -1 * jnp.ones((N,), dtype=jnp.int32)  # -1 => unassigned
    init_rank_idx = jnp.int32(0)
    
    # A boolean mask of which points are still unranked:
    init_unranked = (init_ranks == -1)  # True/False array
    init_done = False  # Will indicate if we should stop

    # Pack into a "carry" tuple to pass between iterations
    carry_init = (init_ranks, init_rank_idx, init_unranked, init_done)

    def cond_fun(carry):
        """Return True if we should continue; False if done."""
        ranks, current_rank, unranked, done = carry
        return jnp.logical_not(done)

    def body_fun(carry):
        """One iteration of finding the next front and assigning ranks."""
        ranks, current_rank, unranked, done = carry
        
        # Check if there are still unranked points
        still_unranked = jnp.any(unranked)  # bool
        
        # For each j, check if it is dominated by any unranked i:
        # dominators[i, j] = (dominates[i, j] & unranked[i])
        dominators = jnp.logical_and(dominates, unranked[:, None])
        dominated_by_unranked = jnp.any(dominators, axis=0)
        
        # The next front = unranked points NOT dominated by any unranked
        front_mask = jnp.logical_and(unranked, jnp.logical_not(dominated_by_unranked))
        
        # If front_mask is empty, we can't assign a next layer.
        # So we set a "done" condition to break out of the loop.
        no_front = jnp.logical_not(jnp.any(front_mask))
        
        # We stop if EITHER we have no unranked points left OR no new front is found
        done_cond = jnp.logical_or(jnp.logical_not(still_unranked), no_front)
        
        # Tentative updates if we are NOT done:
        new_ranks = jnp.where(front_mask, current_rank, ranks)
        new_unranked = jnp.logical_and(unranked, jnp.logical_not(front_mask))
        new_current_rank = current_rank + 1
        new_done = jnp.logical_or(done, done_cond)  # once done => always done
        
        # If done_cond is True, keep the old values (no update):
        new_ranks = jnp.where(done_cond, ranks, new_ranks)
        new_unranked = jnp.where(done_cond, unranked, new_unranked)
        new_current_rank = jnp.where(done_cond, current_rank, new_current_rank)
        
        return (new_ranks, new_current_rank, new_unranked, new_done)

    # Run the while_loop
    final_ranks, _, _, _ = lax.while_loop(cond_fun, body_fun, carry_init)
    return final_ranks

@jit
def compute_distances(embedding, centroids):
    return jnp.sqrt(jnp.sum((embedding - centroids)**2, axis=-1))

@jit
def calculate_entropy_sampling(key, population, num_samples=800):
    n_models = population.shape[0]
    pop_norm = population / jnp.linalg.norm(population, axis=1, keepdims=True)

    key1, key2, key3 = random.split(key, 3)
    idx1 = random.randint(key1, (num_samples,), 0, n_models)
    idx2 = random.randint(key2, (num_samples,), 0, n_models)

    vec1 = pop_norm[idx1]
    vec2 = pop_norm[idx2]

    cosine_sim = jnp.sum(vec1 * vec2, axis=1)

    mean_cosine_sim = jnp.mean(cosine_sim)
    return key3, 1 - mean_cosine_sim 


@jit
def calculate_slopes(best_fitness_window, best_fitness_window_mi, norm_entropy_window):
    best_fitness_slope = calculate_slope(best_fitness_window)
    best_fitness_slope_mi = calculate_slope(best_fitness_window_mi)
    norm_entropy_slope = calculate_slope(norm_entropy_window)
    
    # Normalize the slope values
    #slope_values = jnp.array([avg_fitness_slope, best_fitness_slope, norm_entropy_slope])
    #normalized_slopes = normalize_slopes(slope_values)

    #avg_fitness_slope = normalized_slopes[0]
    #best_fitness_slope = normalized_slopes[1]
    #norm_entropy_slope = normalized_slopes[2]

    stagnation_slope = calculate_stagnation_slope(best_fitness_slope)
    stagnation_slope = -stagnation_slope
    #stagnation_slope = calculate_slope(best_fitness_window)
    return best_fitness_slope, best_fitness_slope_mi, norm_entropy_slope, stagnation_slope

@jit
def scale_arrays(arrays, ref_index=2):
    # Choose the reference array
    ref_array = arrays[ref_index]
    
    # Compute the mean of the reference array
    ref_mean = jnp.mean(ref_array)
    
    # Scale each array based on the reference mean
    scaled_arrays = []
    for array in arrays:
        array_mean = jnp.mean(array)
        scale_factor = ref_mean / array_mean
        scaled_array = array * scale_factor
        scaled_arrays.append(scaled_array)
    
    return scaled_arrays[0],scaled_arrays[1], scaled_arrays[2]

@jit
def calculate_slope(y):
    # Create an array of x-coordinates (indices)
    x = jnp.arange(len(y))
    
    # Calculate the mean of x and y
    mean_x = jnp.mean(x)
    mean_y = jnp.mean(y)
    
    # Calculate the slope using the formula: slope = (x - mean_x) * (y - mean_y) / (x - mean_x)^2
    numerator = jnp.sum((x - mean_x) * (y - mean_y))
    denominator = jnp.sum((x - mean_x) ** 2)
    slope = numerator / denominator
    
    return slope

@jit
def calculate_stagnation_slope(slope, flatness_threshold=0.000000078, max_scale=2):
    # Normalize the slope by the flatness threshold
    normalized_slope = jnp.abs(slope) / flatness_threshold
    
    # Calculate scale factor using an exponential decay function
    # Ensures that the factor is within 0 to max_scale
    scale = max_scale / (1 + normalized_slope ** 2)  # Using squared to enhance the effect of smaller slopes
    return scale

@jit
def update_ks_weights(best_fitness_slope, best_fitness_slope_mi, norm_entropy_slope, stagnation_slope, best_fitness_variance_ratio):
    domain_weight = best_fitness_slope
    situational_weight = best_fitness_slope
    history_weight = best_fitness_slope_mi
    topographic_weight = norm_entropy_slope

    total_weight = jnp.abs(domain_weight) + jnp.abs(situational_weight) + jnp.abs(history_weight) + jnp.abs(topographic_weight)
    domain_weight /= total_weight
    situational_weight /= total_weight
    history_weight /= total_weight
    topographic_weight /= total_weight
    topographic_weight = topographic_weight * best_fitness_variance_ratio 

    #history_weight = history_weight * 0.9

    return jnp.array([domain_weight, situational_weight, history_weight, topographic_weight])


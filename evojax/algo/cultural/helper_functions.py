import jax
import jax.numpy as jnp

from jax import lax
from jax import jit
from jax import random
from jax import vmap
from jax import ops 
from jax.lax import scan, fori_loop

@jit
def initialize_centroids(embeddings, k, key):
    indices = random.choice(key, jnp.arange(embeddings.shape[0]), shape=(k,), replace=False)
    return jnp.take(embeddings, indices, axis=0)

@jit
def compute_distances(embedding, centroids):
    return jnp.sqrt(jnp.sum((embedding - centroids)**2, axis=-1))

@jit
def assign_clusters(embeddings, centroids):
    distances = vmap(compute_distances, in_axes=(0, None))(embeddings, centroids)
    return jnp.argmin(distances, axis=-1)

@jit
def update_centroids(embeddings, assignments, k):
    def update_centroid(i):
        mask = jnp.equal(assignments, i)
        masked_embeddings = jnp.where(mask[:, None], embeddings, 0)
        return jnp.sum(masked_embeddings, axis=0) / jnp.sum(mask)

    return vmap(update_centroid)(jnp.arange(k))

def kmeans_step(state, _):
    centroids, embeddings, k = state
    assignments = assign_clusters(embeddings, centroids)
    centroids = update_centroids(embeddings, assignments, k)
    return (centroids, embeddings, k), None

@jit
def kmeans(embeddings, k=10, num_iters=60, seed=0):
    key = random.PRNGKey(seed)
    centroids = initialize_centroids(embeddings, k, key)
    initial_state = (centroids, embeddings, k)

    final_state, _ = lax.scan(kmeans_step, initial_state, None, length=num_iters)
    final_centroids, embeddings, k = final_state
    assignments = assign_clusters(embeddings, final_centroids)
    return final_centroids, assignments

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

    domain_weight = 99
    #topographic_weight = 99
    #history_weight = history_weight * 0.9

    return jnp.array([domain_weight, situational_weight, history_weight, topographic_weight])

@jit
def situational_score(adv_slope_short, mi_slope_short):
    """
    We want a high situational score if both adversarial and MI slopes
    are negative (indicating improvement).
    """
    # Suppose adv_slope_short ranges ~[-2, +2], likewise for mi_slope_short.
    # A negative slope -> improvement.

    # Convert slope to a "signal" in [0, 1], where negative slope => near 1
    # and positive slope => near 0.
    # We'll do a simple exponential transform:
    adv_signal = jnp.exp(-jnp.clip(adv_slope_short, -2.0, 2.0))
    mi_signal  = jnp.exp(-jnp.clip(mi_slope_short, -2.0, 2.0))

    # If slope is -2, exp(-(-2)) = exp(2) ~ 7.39 => strong improvement
    # If slope is  2, exp(-(2))  = exp(-2) ~ 0.135 => poor improvement

    # Combine them:
    raw_score = (adv_signal + mi_signal) / 2.0

    # Now raw_score could range roughly from ~0.135 to ~7.39. 
    # You might clamp or scale that further:
    scaled_score = jnp.clip(raw_score, 0.0, 5.0) 
    return scaled_score
#def situational_score(adv_slope_short, mi_slope_short):
#    # We want negative slopes to yield higher scores.
#    # For example, transform slopes into a [0,1] range by taking e^-slope if slope>0 or something similar.
#    # Alternatively, just clamp negative slopes to a positive range. Simplest approach:
#    
#    # Convert negative slope to a positive number (no improvement => 0).
#    adv_signal = -jnp.clip(adv_slope_short, -1.0, 1.0)
#    mi_signal  = -jnp.clip(mi_slope_short, -1.0, 1.0)
#    
#    # If both slopes are negative, the sum is high => exploit
#    raw_score = (adv_signal + mi_signal) / 2.0  # average them
#    # Ensure it's in [0,1]
#    return jnp.clip(raw_score, 0.0, 1.0)

@jit
def historical_score(entropy_long, adv_slope_short):
    # We want to reintroduce diversity when entropy is *low* 
    # and when there's no short-term improvement (adversarial slope >= 0).
    
    # "Low" entropy => high 'need' for historical injection
    # Let's define a function that flips the entropy scale into [0,1].
    # Suppose we assume typical entropy is around 2 to 3 for 10-class. 
    # We can clamp or scale it:
    inv_entropy = jnp.clip(3.0 - entropy_long, 0.0, 3.0) / 3.0  
    # This yields 1.0 if entropy_long is 0.0, and near 0.0 if entropy_long is ~3.0
    
    # Also, if short-term adv slope >= 0 => no improvement => want reintroduction
    no_improvement = jnp.clip(adv_slope_short, 0.0, 1.0)  # slope>0 => positive => no improvement
    # Combine them (could average, multiply, etc.):
    raw_score = inv_entropy * (1.0 + no_improvement)
    # Normalize or clip
    return jnp.clip(raw_score, 0.0, 2.0)

@jit
def topographic_score(entropy_long, adv_slope_med, spread_short=0.0):
    # Maybe you want topographic exploration if entropy_long is starting to slip but not fully collapsed,
    # or if adv_slope_med is near 0 => no big improvement.

    mid_entropy_drop = jnp.clip(2.5 - entropy_long, 0.0, 2.5) / 2.5  # partial "need" for exploring codes
    slow_improvement = jnp.clip(-adv_slope_med, 0.0, 1.0)  # if adv_slope_med>0 => no improvement => 0 here

    # Layer-A-only mode: keep spread slopes logged, but do not let spread drive
    # KS scoring until baseline behavior is stable.
    _ = spread_short
    raw_score = mid_entropy_drop + slow_improvement
    return jnp.clip(raw_score, 0.0, 2.0)

@jit
def domain_score(adv_slope_med, mi_slope_med, entropy_long, spread_med=0.0):
    # Example: moderate negative slopes => stable improvement => Domain KS can refine
    moderate_adv = jnp.clip(-adv_slope_med, 0.0, 1.0)
    moderate_mi  = jnp.clip(-mi_slope_med, 0.0, 1.0)
    # If entropy_long is also moderate (e.g., ~2), that might be "good enough" => domain
    normal_entropy = jnp.exp(-jnp.abs(2.0 - entropy_long))  # peak around 2.0

    # Layer-A-only mode: keep spread slopes logged, but do not let spread drive
    # KS scoring until baseline behavior is stable.
    _ = spread_med
    raw_score = (moderate_adv + moderate_mi) + normal_entropy + 0.897
    return jnp.clip(raw_score, 0.0, 2.0)

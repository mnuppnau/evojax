import jax
import jax.numpy as jnp

from jax import jit
from jax import random
from jax import vmap
from jax import ops 
from jax.lax import scan, fori_loop

@jit
def initialize_centroids(embeddings, k, key):
    """
    This function initializes k centroids randomly.

    Args:
        embeddings (jax.numpy.ndarray): The input embeddings.
        k (int): The number of clusters.
        key (jax.random.PRNGKey): The random key.

    Returns:
        jax.numpy.ndarray: The initialized centroids.
    """
    indices = random.choice(key, jnp.arange(embeddings.shape[0]), shape=(10,), replace=False)
    return jnp.take(embeddings, indices, axis=0)

@jit
def compute_distances(embedding, centroids):
    """
    This function computes the distance from each centroid to an embedding.

    Args:
        embedding (jax.numpy.ndarray): The input embedding.
        centroids (jax.numpy.ndarray): The centroids.

    Returns:
        jax.numpy.ndarray: The distances.
    """
    return jnp.sqrt(jnp.sum((embedding - centroids)**2, axis=-1))

@jit
def assign_clusters(embeddings, centroids):
    """
    This function assigns each embedding to the nearest centroid.

    Args:
        embeddings (jax.numpy.ndarray): The input embeddings.
        centroids (jax.numpy.ndarray): The centroids.

    Returns:
        jax.numpy.ndarray: The cluster assignments for each embedding.
    """
    distances = vmap(compute_distances, in_axes=(0, None))(embeddings, centroids)
    return jnp.argmin(distances, axis=-1)

@jit
def update_centroids(embeddings, assignments, k):
    """
    This function updates the centroids by computing the mean of all embeddings in each cluster.

    Args:
        embeddings (jax.numpy.ndarray): The input embeddings.
        assignments (jax.numpy.ndarray): The cluster assignments for each embedding.
        K (int): The number of clusters.

    Returns:
        jax.numpy.ndarray: The updated centroids.
    """
    def update_centroid(i):
        mask = jnp.equal(assignments, i)
        masked_embeddings = jnp.where(mask[:, None], embeddings, 0)
        return jnp.sum(masked_embeddings, axis=0) / jnp.sum(mask)

    return jax.vmap(update_centroid)(jnp.arange(6))

def kmeans(embeddings, k=6, num_iters=100, seed=0):
    """
    This function applies the K-Means algorithm to input embeddings.

    Args:
        embeddings (jax.numpy.ndarray): The input embeddings.
        k (int): The number of clusters.
        num_iters (int, optional): The number of iterations to run the K-Means algorithm. Default is 100.
        seed (int, optional): The random seed for centroid initialization. Default is 0.

    Returns:
        tuple: The final centroids and the cluster assignments for each embedding.
    """
    key = random.PRNGKey(seed)
    centroids = initialize_centroids(embeddings, k, key)

    for _ in range(num_iters):
        assignments = assign_clusters(embeddings, centroids)
        centroids = update_centroids(embeddings, assignments, k)

    return centroids, assignments


def least_frequent_cluster(arr):
    unique_elements, counts = jnp.unique(arr, return_counts=True)
    min_count_index = jnp.argmin(counts)
    least_frequent_element = unique_elements[min_count_index]
    first_index = jnp.where(arr == least_frequent_element)[0][0]
    return least_frequent_element, first_index

@jit
def calculate_entropy(population):
    # Normalize the population matrix
    population_norm = population / jnp.linalg.norm(population, axis=1, keepdims=True)
    
    # Calculate the cosine similarity matrix
    cosine_sim = jnp.dot(population_norm, population_norm.T)
    
    # Calculate the mean cosine similarity
    mean_cosine_sim = jnp.mean(cosine_sim)
    
    # Convert cosine similarity to cosine distance
    mean_cosine_distance = 1 - mean_cosine_sim
    
    return mean_cosine_distance

#@jit
#def calculate_slope(values):
#    n = len(values)
#    x = jnp.arange(n)
#    x_mean = jnp.mean(x)
#    y_mean = jnp.mean(values)
    
#    numerator = jnp.sum((x - x_mean) * (values - y_mean))
#    denominator = jnp.sum((x - x_mean) ** 2)
    
#    slope = numerator / denominator
#    return slope

@jit
def calculate_slopes(avg_fitness_window, best_fitness_window, norm_entropy_window):
    avg_fitness_slope = calculate_slope(avg_fitness_window)
    best_fitness_slope = calculate_slope(best_fitness_window)
    norm_entropy_slope = calculate_slope(norm_entropy_window)
    
    # Normalize the slope values
    #slope_values = jnp.array([avg_fitness_slope, best_fitness_slope, norm_entropy_slope])
    #normalized_slopes = normalize_slopes(slope_values)

    #avg_fitness_slope = normalized_slopes[0]
    #best_fitness_slope = normalized_slopes[1]
    #norm_entropy_slope = normalized_slopes[2]

    stagnation_slope = calculate_stagnation_slope(best_fitness_slope)
    stagnation_slope = -stagnation_slope
    return avg_fitness_slope, best_fitness_slope, norm_entropy_slope, stagnation_slope

def normalize_slopes(slope_values):
    min_value = jnp.min(slope_values)
    max_value = jnp.max(slope_values)
    normalized_slopes = (slope_values - min_value) / (max_value - min_value)
    return normalized_slopes

def combined_min_max_scale(arr1, arr2, arr3):
    combined = jnp.concatenate([arr1, arr2, arr3])
    min_val = jnp.min(combined)
    max_val = jnp.max(combined)
    scaled1 = (arr1 - min_val) / (max_val - min_val)
    scaled2 = (arr2 - min_val) / (max_val - min_val)
    scaled3 = (arr3 - min_val) / (max_val - min_val)
    return scaled1, scaled2, scaled3

def combined_z_score_standardize(arr1, arr2, arr3):
    combined = jnp.concatenate([arr1, arr2, arr3])
    mean_val = jnp.mean(combined)
    std_dev = jnp.std(combined)
    standardized1 = (arr1 - mean_val) / std_dev
    standardized2 = (arr2 - mean_val) / std_dev
    standardized3 = (arr3 - mean_val) / std_dev
    return standardized1, standardized2, standardized3

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

def normalize_arrays(arrays):
    # Concatenate the arrays along a new axis
    concatenated = jnp.stack(arrays)
    
    # Compute the minimum and maximum values along the first axis
    min_val = jnp.min(concatenated, axis=0)
    max_val = jnp.max(concatenated, axis=0)
    
    # Normalize the arrays using min-max normalization
    normalized = (concatenated - min_val) / (max_val - min_val)
    
    # Split the normalized arrays back into separate arrays
    normalized_arrays = jnp.split(normalized, len(arrays))

    return normalized_arrays[0][0], normalized_arrays[1][0], normalized_arrays[2][0]

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
def calculate_slope_old(window):
    #if len(window) < 2:
    #    return 0.0
    x = jnp.arange(len(window), dtype=jnp.float32)  # Convert x to float32
    y = jnp.array(window, dtype=jnp.float32)        # Convert y to float32 
    slope = jnp.polyfit(x, y, 1)[0]
    return slope

def update_weights(avg_fitness_slope, best_fitness_slope, norm_entropy_slope, stagnation_slope):
    domain_weight = best_fitness_slope
    situational_weight = avg_fitness_slope
    history_weight = norm_entropy_slope
    topographic_weight = stagnation_slope

    total_weight = jnp.abs(domain_weight) + jnp.abs(situational_weight) + jnp.abs(history_weight) + jnp.abs(topographic_weight)
    domain_weight /= total_weight
    situational_weight /= total_weight
    history_weight /= total_weight
    topographic_weight /= total_weight

    #history_weight = history_weight * 0.9

    return jnp.array([domain_weight, situational_weight, history_weight, topographic_weight])

def compute_cluster_weights(assignments, fitness_values):
    num_clusters = jnp.max(assignments) + 1
    cluster_sums = jnp.zeros(num_clusters)
    cluster_counts = jnp.zeros(num_clusters)

    def update_cluster_stats(carry, x):
        cluster_sums, cluster_counts = carry
        assignment, fitness = x
        cluster_sums = cluster_sums.at[assignment].add(fitness)
        cluster_counts = cluster_counts.at[assignment].add(1)
        return (cluster_sums, cluster_counts), None

    (cluster_sums, cluster_counts), _ = jax.lax.scan(update_cluster_stats, (cluster_sums, cluster_counts), (assignments, fitness_values))

    cluster_weights = cluster_sums / cluster_counts
    normalized_weights = cluster_weights / jnp.sum(cluster_weights)  # Normalize the weights
    return normalized_weights

def inverse_fitness_values(fitness_values, epsilon=1e-2):
    min_fitness = jnp.max(fitness_values)
    shifted_fitness = jnp.abs(fitness_values - min_fitness)
    inverse_fitness = 1 / (shifted_fitness + epsilon)
    normalized_inverse_fitness = inverse_fitness / jnp.sum(inverse_fitness)
    return normalized_inverse_fitness

def average_activations(tuple_list):
    # Stack arrays in each position of the tuples
    stack_1 = jnp.stack([t[0] for t in tuple_list])
    stack_2 = jnp.stack([t[1] for t in tuple_list])
    stack_3 = jnp.stack([t[2] for t in tuple_list])
    
    # Compute the average along the new axis (axis 0)
    average_1 = jnp.mean(stack_1, axis=0)
    average_2 = jnp.mean(stack_2, axis=0)
    average_3 = jnp.mean(stack_3, axis=0)
    
    return (average_1, average_2, average_3)

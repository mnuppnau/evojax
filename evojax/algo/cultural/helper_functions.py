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

    return jax.vmap(update_centroid)(jnp.arange(10))

def kmeans(embeddings, k=10, num_iters=100, seed=0):
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
    slope_values = jnp.array([avg_fitness_slope, best_fitness_slope, norm_entropy_slope])
    normalized_slopes = normalize_slopes(slope_values)

    avg_fitness_slope = normalized_slopes[0]
    best_fitness_slope = normalized_slopes[1]
    norm_entropy_slope = normalized_slopes[2]

    stagnation_slope = 1 - best_fitness_slope
    return avg_fitness_slope, best_fitness_slope, norm_entropy_slope, stagnation_slope

def normalize_slopes(slope_values):
    min_value = jnp.min(slope_values)
    max_value = jnp.max(slope_values)
    normalized_slopes = (slope_values - min_value) / (max_value - min_value)
    return normalized_slopes

@jit
def calculate_slope(window):
    if len(window) < 2:
        return 0.0
    x = jnp.arange(len(window), dtype=jnp.float32)  # Convert x to float32
    y = jnp.array(window, dtype=jnp.float32)        # Convert y to float32 
    slope = jnp.polyfit(x, y, 1)[0]
    return slope

def update_weights(avg_fitness_slope, best_fitness_slope, norm_entropy_slope, stagnation_slope):
    domain_weight = best_fitness_slope
    situational_weight = avg_fitness_slope
    history_weight = norm_entropy_slope
    topographic_weight = stagnation_slope

    total_weight = domain_weight + situational_weight + history_weight + topographic_weight
    domain_weight /= total_weight
    situational_weight /= total_weight
    history_weight /= total_weight
    topographic_weight /= total_weight

    history_weight = history_weight * 0.9

    return jnp.array([domain_weight, situational_weight, history_weight, topographic_weight])

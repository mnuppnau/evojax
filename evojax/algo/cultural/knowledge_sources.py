# Description: This file contains the implementation of the knowledge sources (KS) used in the knowledge-based optimization (KBO) framework.

from typing import Dict, List, Optional
import jax.numpy as jnp
import jax
import jax.tree_util
import jax.lax
import numpy as np
import copy

from evojax.algo.cultural.population_space import Individual
from evojax.algo.cultural.helper_functions import (
    kmeans,
    calculate_slopes,
    update_ks_weights,
    least_frequent_cluster,
    scale_arrays,
    compute_cluster_weights,
    inverse_fitness_values,
    average_activations,
)


def initialize_domain_ks(param_size: int):
    return (
        jnp.zeros(param_size),  # center
        jnp.ones(param_size),  # stdev
        jnp.array([0.0]),  # fitness_score
        [
            (
                jnp.zeros((90, 28, 28, 8)),
                jnp.zeros((90, 14, 14, 16)),
                jnp.zeros((90, 784)),
            )
            for _ in range(40)
        ],  # activations
        (
            jnp.zeros((90, 28, 28, 8)),
            jnp.zeros((90, 14, 14, 16)),
            jnp.zeros((90, 784)),
        ),  # average activations
    )


def initialize_situational_ks(param_size: int, max_individuals: int = 100):
    # Pre-allocate arrays with zeros for each individual property, given max_individuals
    # Assuming 'center' and 'stdev' are of size 'param_size'
    return (
        jnp.zeros((param_size, max_individuals)),  # center
        jnp.zeros((param_size, max_individuals)),  # stdev
        jnp.zeros(max_individuals),  # fitness_score
    )


def initialize_history_ks(
    param_size: int, decay_factor: float = 0.8, num_iterations: int = 5000
):
    # Pre-allocate arrays with zeros for each individual property, given num_iterations
    # Assuming 'center' and 'stdev' are of size 'param_size'
    return (
        jnp.zeros((param_size, num_iterations)),  # center
        jnp.zeros((param_size, num_iterations)),  # stdev
        jnp.zeros(num_iterations),  # fitness_score
    )


def initialize_topographic_ks(
    param_size: int, num_clusters: int = 6, max_individuals: int = 100
):
    return (
        jnp.zeros((param_size, max_individuals)),  # center
        jnp.zeros((param_size, max_individuals)),  # stdev
        jnp.zeros(max_individuals),  # fitness_scores
        jnp.zeros(num_clusters, param_size),  # cluster_centroids_center
        jnp.zeros(num_clusters, param_size),  # cluster_centroids_stdev
        jnp.zeros(max_individuals),  # cluster_assignments_center
        jnp.zeros(max_individuals),  # cluster_assignments_stdev
        jnp.zeros(max_individuals),  # cluster_weights_center
        jnp.zeros(max_individuals),  # cluster_weights_stdev
    )


def initialize_normative_ks(param_size: int):
    return (
        jnp.ones(40),  # rolling_avg_fitness
        jnp.ones(40),  # rolling_best_fitness
        jnp.ones(40),  # rolling_norm_entropy
        jnp.array([0.0]),  # avg_fitness_slope
        jnp.array([0.0]),  # best_fitness_slope
        jnp.array([0.0]),  # norm_entropy_slope
        jnp.array([0.0]),  # stagnation_slope
        jnp.ones(60),  # rolling_best_fitness_variance
        jnp.array([0.0]),  # best_fitness_variance_ratio
    )


@jax.jit
def update_knowledge_sources(
    belief_space, best_individual, activations, num_iterations=5000, max_individuals=100
):

    best_center, best_stdev, best_fitness_value = best_individual

    domain_activations = belief_space[1][3]

    domain_activations.insert(0, activations)

    domain_activations.pop()

    avg_activations = average_activations(domain_activations)

    updated_domain_ks = (
        best_center,
        best_stdev,
        best_fitness_value,
        domain_activations,
        avg_activations,
    )

    updated_belief_space_domain = (
        belief_space[:1] + (updated_domain_ks,) + belief_space[2:]
    )

    sit_center, sit_stdev, sit_fitness_value = updated_belief_space_domain[2]

    best_center_reshaped = best_center.reshape(-1, 1)

    sit_updated_center = jnp.concatenate([best_center_reshaped, sit_center], axis=1)[
        :, :max_individuals
    ]
    # print('updated center shape : ', updated_center.shape)
    sit_updated_stdev = jnp.concatenate([best_stdev[:, None], sit_stdev], axis=1)[
        :, :max_individuals
    ]

    sit_updated_fitness_value = jnp.concatenate(
        [best_fitness_value, sit_fitness_value], axis=0
    )[:max_individuals]

    # Construct the updated situational knowledge source
    updated_situational_ks = (
        sit_updated_center,
        sit_updated_stdev,
        sit_updated_fitness_value,
    )

    # Reconstruct the belief space with the updated situational KS
    updated_belief_space_situational = (
        updated_belief_space_domain[:2]
        + (updated_situational_ks,)
        + updated_belief_space_domain[3:]
    )

    hist_center, hist_stdev, hist_fitness_value = updated_belief_space_domain[3]

    hist_updated_center = jnp.concatenate([best_center_reshaped, hist_center], axis=1)[
        :, :num_iterations
    ]
    hist_updated_stdev = jnp.concatenate([best_stdev[:, None], hist_stdev], axis=1)[
        :, :num_iterations
    ]

    hist_updated_fitness_value = jnp.concatenate(
        [best_fitness_value, hist_fitness_value], axis=0
    )[:num_iterations]

    # Construct the updated situational knowledge source
    updated_history_ks = (
        hist_updated_center,
        hist_updated_stdev,
        hist_updated_fitness_value,
    )

    updated_belief_space_history = (
        updated_belief_space_situational[:3]
        + (updated_history_ks,)
        + updated_belief_space_situational[4:]
    )

    return updated_belief_space_history


def add_ind_topographic_ks(
    belief_space, grad_center, grad_stdev, best_score, max_individuals=100
):
    topographic_ks = belief_space[4]

    center, stdev, fitness = topographic_ks[:3]

    grad_center = grad_center.reshape(-1, 1)
    grad_stdev = grad_stdev.reshape(-1, 1)

    updated_center = jnp.concatenate([grad_center, center], axis=1)[:, :max_individuals]

    updated_stdev = jnp.concatenate([grad_stdev, stdev], axis=1)[:, :max_individuals]

    updated_fitness = jnp.concatenate([best_score, fitness], axis=0)[:max_individuals]

    updated_topographic_ks = (
        updated_center,
        updated_stdev,
        updated_fitness,
        topographic_ks[3],
        topographic_ks[4],
        topographic_ks[5],
        topographic_ks[6],
        topographic_ks[7],
        topographic_ks[8],
    )

    centroids_center, assignments_center = kmeans(updated_center.T)
    centroids_stdev, assignments_stdev = kmeans(updated_stdev.T)

    inv_fitness = inverse_fitness_values(updated_fitness)

    cluster_weights_center = compute_cluster_weights(assignments_center, inv_fitness)

    cluster_weights_stdev = compute_cluster_weights(assignments_stdev, inv_fitness)

    final_topographic_ks = (
        updated_topographic_ks[0],
        updated_topographic_ks[1],
        updated_topographic_ks[2],
        centroids_center,
        centroids_stdev,
        assignments_center,
        assignments_stdev,
        cluster_weights_center,
        cluster_weights_stdev,
    )

    updated_belief_space_topographic = (
        belief_space[:4] + (final_topographic_ks,) + belief_space[5:]
    )

    return updated_belief_space_topographic


def update_topographic_ks(
    belief_space, grad_center, grad_stdev, best_score, max_individuals=100
):
    topographic_ks = belief_space[4]

    center, stdev, fitness = topographic_ks[:3]

    grad_center = grad_center.reshape(-1, 1)
    grad_stdev = grad_stdev.reshape(-1, 1)

    updated_center = jnp.concatenate([grad_center, center], axis=1)[:, :max_individuals]

    updated_stdev = jnp.concatenate([grad_stdev, stdev], axis=1)[:, :max_individuals]

    updated_fitness = jnp.concatenate([best_score, fitness], axis=0)[:max_individuals]

    updated_topographic_ks = (
        updated_center,
        updated_stdev,
        updated_fitness,
        topographic_ks[3],
        topographic_ks[4],
        topographic_ks[5],
        topographic_ks[6],
        topographic_ks[7],
        topographic_ks[8],
    )

    centroids_center, assignments_center = kmeans(updated_center.T)
    centroids_stdev, assignments_stdev = kmeans(updated_stdev.T)

    inv_fitness = inverse_fitness_values(updated_fitness)

    cluster_weights_center = compute_cluster_weights(assignments_center, inv_fitness)

    cluster_weights_stdev = compute_cluster_weights(assignments_stdev, inv_fitness)

    final_topographic_ks = (
        updated_topographic_ks[0],
        updated_topographic_ks[1],
        updated_topographic_ks[2],
        centroids_center,
        centroids_stdev,
        assignments_center,
        assignments_stdev,
        cluster_weights_center,
        cluster_weights_stdev,
    )

    updated_belief_space_topographic = (
        belief_space[:4] + (final_topographic_ks,) + belief_space[5:]
    )

    return updated_belief_space_topographic


@jax.jit
def update_normative_ks(
    belief_space, best_fitness, avg_fitness, norm_entropy 
):
    normative_ks = belief_space[5]

    one_dim_avg_fitness = jnp.array([avg_fitness])
    one_dim_best_fitness = jnp.array([best_fitness])
    one_dim_norm_entropy = jnp.array([norm_entropy])

    updated_rolling_avg_fitness = jnp.concatenate(
        [normative_ks[0], -one_dim_avg_fitness], axis=0
    )[1:]

    updated_rolling_best_fitness = jnp.concatenate(
        [normative_ks[1], -one_dim_best_fitness], axis=0
    )[1:]
    
    updated_rolling_norm_entropy = jnp.concatenate(
        [normative_ks[2], one_dim_norm_entropy], axis=0
    )[1:]

    (
        scaled_rolling_avg_fitness,
        scaled_rolling_best_fitness,
        scaled_rolling_norm_entropy,
    ) = scale_arrays(
        [
            updated_rolling_avg_fitness,
            updated_rolling_best_fitness,
            updated_rolling_norm_entropy,
        ]
    )

    scaled_rolling_best_fitness_variance = jnp.var(scaled_rolling_best_fitness)

    one_dim_best_fitness_variance = jnp.array([scaled_rolling_best_fitness_variance])

    updated_rolling_best_fitness_variance = jnp.concatenate(
        [normative_ks[7], one_dim_best_fitness_variance], axis=0
    )[1:]

    top_20_variances = jax.lax.top_k(updated_rolling_best_fitness_variance, 20)[0]

    average_rolling_best_fitness_variance = jnp.mean(top_20_variances)

    best_fitness_variance_ratio = (
        scaled_rolling_best_fitness_variance / average_rolling_best_fitness_variance
    )

    avg_fitness_slope, best_fitness_slope, norm_entropy_slope, stagnation_slope = (
        calculate_slopes(
            avg_fitness_window=scaled_rolling_avg_fitness,
            best_fitness_window=scaled_rolling_best_fitness,
            norm_entropy_window=scaled_rolling_norm_entropy,
        )
    )

    ks_weights = update_ks_weights(
        avg_fitness_slope,
        best_fitness_slope,
        norm_entropy_slope,
        stagnation_slope,
        best_fitness_variance_ratio,
    )

    updated_normative_ks = (
        updated_rolling_avg_fitness,
        updated_rolling_best_fitness,
        updated_rolling_norm_entropy,
        avg_fitness_slope,
        best_fitness_slope,
        norm_entropy_slope,
        stagnation_slope,
        updated_rolling_best_fitness_variance,
        best_fitness_variance_ratio,
    )

    updated_belief_space_normative = (
        belief_space[:5] + (updated_normative_ks,)
    )

    return updated_belief_space_normative, ks_weights


def get_center_guidance(belief_space, t, center):
    topographic_ks = belief_space[4]
    normative_ks = belief_space[5]

    best_fitness_variance_ratio = normative_ks[8]
    avg_fitness_slope = normative_ks[3]
    best_fitness_slope = normative_ks[4]
    norm_entropy_slope = normative_ks[5]
    stagnation_slope = normative_ks[6]

    ks_weights = update_ks_weights(
        avg_fitness_slope,
        best_fitness_slope,
        norm_entropy_slope,
        stagnation_slope,
        best_fitness_variance_ratio,
    )

    min_index = jnp.argmin(ks_weights)
    result = jnp.zeros_like(ks_weights, dtype=jnp.int32)

    ks_weights = result.at[min_index].set(1)

    decay_factor_history = 0.994
    decay_factor_situational = 0.95

    max_iterations = 5000

    arr = jnp.array([t, 100])
    n = jnp.min(arr)

    domain_ks_center = belief_space[1][0]
    situational_ks_center = belief_space[2][0]  # [:,:n]
    history_ks_center = belief_space[3][0]  # [:,:t]

    topographic_ks_center = topographic_ks[4]

    situational_valid_columns_mask = jnp.arange(situational_ks_center.shape[1]) < t

    situational_masked_data = situational_ks_center * situational_valid_columns_mask

    column_indices_situational = jnp.arange(100)
    weights_situational = jnp.exp(
        -decay_factor_situational * column_indices_situational
    )

    masked_weights_situational = weights_situational * situational_valid_columns_mask

    weighted_sums_situational = jnp.sum(
        situational_masked_data * masked_weights_situational, axis=1
    )
    sum_of_masked_weights_situational = jnp.sum(masked_weights_situational)

    situational_weighted_averages = (
        weighted_sums_situational / sum_of_masked_weights_situational
    )

    history_valid_columns_mask = jnp.arange(history_ks_center.shape[1]) < t

    history_masked_data = history_ks_center * history_valid_columns_mask

    column_indices = jnp.arange(max_iterations)
    weights = jnp.exp(-decay_factor_history * column_indices)

    masked_weights = weights * history_valid_columns_mask

    weighted_sums = jnp.sum(history_masked_data * masked_weights, axis=1)
    sum_of_masked_weights = jnp.sum(masked_weights)

    history_weighted_averages = weighted_sums / sum_of_masked_weights

    domain_ks_center_weighted = domain_ks_center * ks_weights[0]
    situational_row_averages_weighted = situational_weighted_averages * ks_weights[1]
    history_row_averages_weighted = history_weighted_averages * ks_weights[2]
    topographic_ks_center_weighted = center * ks_weights[3]

    return (
        jnp.sum(
            jnp.array(
                [
                    domain_ks_center_weighted,
                    situational_row_averages_weighted,
                    history_row_averages_weighted,
                    topographic_ks_center_weighted,
                ]
            ),
            axis=0,
        )
    )


def get_stdev_guidance(belief_space, t, stdev):
    topographic_ks = belief_space[4]
    normative_ks = belief_space[5]

    best_fitness_variance_ratio = normative_ks[8]
    decay_factor_historical = 0.994
    decay_factor_situational = 0.95

    max_iterations = 5000

    avg_fitness_slope = normative_ks[3]
    best_fitness_slope = normative_ks[4]
    norm_entropy_slope = normative_ks[5]
    stagnation_slope = normative_ks[6]

    ks_weights = update_ks_weights(
        avg_fitness_slope,
        best_fitness_slope,
        norm_entropy_slope,
        stagnation_slope,
        best_fitness_variance_ratio,
    )

    max_index = jnp.argmax(ks_weights)
    result = jnp.zeros_like(ks_weights, dtype=jnp.int32)

    ks_weights = result.at[max_index].set(1)

    arr = jnp.array([t, 100])
    n = jnp.min(arr)

    domain_ks_stdev = belief_space[1][1]
    situational_ks_stdev = belief_space[2][1]  # [:,:n]
    history_ks_stdev = belief_space[3][1]  # [:,:t]

    situational_valid_columns_mask = jnp.arange(situational_ks_stdev.shape[1]) < t

    situational_masked_data = situational_ks_stdev * situational_valid_columns_mask

    column_indices_situational = jnp.arange(100)
    weights_situational = jnp.exp(
        -decay_factor_situational * column_indices_situational
    )

    masked_weights_situational = weights_situational * situational_valid_columns_mask

    weighted_sums_situational = jnp.sum(
        situational_masked_data * masked_weights_situational, axis=1
    )
    sum_of_masked_weights_situational = jnp.sum(masked_weights_situational)

    situational_weighted_averages = (
        weighted_sums_situational / sum_of_masked_weights_situational
    )

    history_valid_columns_mask = jnp.arange(history_ks_stdev.shape[1]) < t

    history_masked_data = history_ks_stdev * history_valid_columns_mask

    column_indices = jnp.arange(max_iterations)
    weights = jnp.exp(-decay_factor_historical * column_indices)

    masked_weights = weights * history_valid_columns_mask

    weighted_sums = jnp.sum(history_masked_data * masked_weights, axis=1)
    sum_of_masked_weights = jnp.sum(masked_weights)

    history_weighted_averages = weighted_sums / sum_of_masked_weights

    domain_ks_stdev_weighted = domain_ks_stdev * ks_weights[0]
    situational_row_averages_weighted = situational_weighted_averages * ks_weights[1]
    history_row_averages_weighted = history_weighted_averages * ks_weights[2]

    topographic_ks_stdev = stdev * ks_weights[3]

    return (
        domain_ks_stdev_weighted
        + situational_row_averages_weighted
        + history_row_averages_weighted
        + topographic_ks_stdev
    )

# class DomainKS:
#    def __init__(self):
#        self.individual = None
#        self.assigned_indexes = None
#        self.individual_count = 0
#
#    def __repr__(self):
#        if self.individual is None:
#            return "DomainKS(individual=None)"
#        else:
#            return f"DomainKS(individual={self.individual})"
#
#    def accept(self, individual: Individual):
#        self.individual = individual
#        self.individual_count = 1
#
#    def update(self):
#        pass
#
#    def get_center_guidance(self) -> Optional[jnp.ndarray]:
#        if self.individual is not None:
#            #print("center guidance : ", self.individual.center)
#            return self.individual.center
#        return None
#
#    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
#        if self.individual is not None:
#            return self.individual.stdev
#        return None
#
#    def adjust_noise(self, scaled_noises: jnp.ndarray, index: int, index_counter: int) -> jnp.ndarray:
#        if self.individual is not None and index == 0:
#            # Adjust the noise for the first index if an individual is available
#            scaled_noises = scaled_noises.at[index].set(scaled_noises[index] * self.individual.noise_magnitude)
#        return scaled_noises
#
# class SituationalKS:
#    def __init__(self, max_individuals: int = 100):
#        self.max_individuals = max_individuals
#        self.individuals: List[Individual] = []
#        self.assigned_indexes = None
#        self.individual_count = 0
#
#    def __repr__(self):
#        individual_str = "\n".join([str(individual) for individual in self.individuals])
#        return f"SituationalKS(max_individuals={self.max_individuals}, individuals=[\n{individual_str}\n])"
#
#    def accept(self, individual: Individual):
#        self.individuals.append(individual)
#        #self.individual_count += 1
#        self.individual_count = min(len(self.individuals), self.max_individuals)
#        #print("individual count in situational KS : ", self.individual_count)
#
#    def update(self):
#        self.individuals.sort(key=lambda x: x.fitness_score)
#        self.individuals = self.individuals[:self.max_individuals]
#
#    def get_center_guidance(self) -> Optional[jnp.ndarray]:
#        if self.individuals:
#            centers = jnp.array([ind.center for ind in self.individuals])
#            weights = jnp.array([1.0 - jnp.abs(ind.fitness_score) for ind in self.individuals])
#            weights /= jnp.sum(weights)  # Normalize the weights
#            return jnp.average(centers, weights=weights, axis=0)
#        return None
#
#    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
#        if self.individuals:
#            stdevs = jnp.array([ind.stdev for ind in self.individuals])
#            weights = jnp.array([1.0 - jnp.abs(ind.fitness_score) for ind in self.individuals])
#            weights /= jnp.sum(weights)  # Normalize the weights
#            return jnp.average(stdevs, weights=weights, axis=0)
#        return None
#
#    #def get_center_guidance(self) -> Optional[jnp.ndarray]:
#    #    if self.individuals:
#    #        centers = jnp.array([ind.center for ind in self.individuals])
#    #        weights = jnp.array([jnp.absolute(ind.fitness_score) for ind in self.individuals])
#    #        #print("average center : ", jnp.average(centers, weights=weights, axis=0))
#    #        return jnp.average(centers, weights=weights, axis=0)
#    #    return None
#
#    #def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
#    #    if self.individuals:
#    #        stdevs = jnp.array([ind.stdev for ind in self.individuals])
#    #        weights = jnp.array([jnp.absolute(ind.fitness_score) for ind in self.individuals])
#    #        return jnp.average(stdevs, weights=weights, axis=0)
#    #    return None
#
#    def adjust_noise(self, scaled_noises: jnp.ndarray, index: int, index_counter: int) -> jnp.ndarray:
#        #if index < len(self.individuals):
#            # Adjust the noise for the assigned index if an individual is available
#        scaled_noises = scaled_noises.at[index].set(scaled_noises[index] * self.individuals[index_counter].noise_magnitude)
#        return scaled_noises
#
# class HistoryKS:
#    def __init__(self, decay_factor: float = 0.8):
#        self.individuals: List[Individual] = []
#        self.decay_factor = decay_factor
#        self.assigned_indexes = None
#        self.individual_count = 0
#
#    def __repr__(self):
#        individual_str = "\n".join([str(individual) for individual in self.individuals])
#        return f"HistoryKS(decay_factor={self.decay_factor}, individuals=[\n{individual_str}\n])"
#
#    def accept(self, individual: Individual):
#        self.individuals.append(individual)
#        self.individual_count = len(self.individuals)
#        #self.individual_count += 1
#        #print("individual count in history KS : ", self.individual_count)
#
#    def update(self):
#        pass
#
#    def get_center_guidance(self) -> Optional[jnp.ndarray]:
#        if self.individuals:
#            centers = jnp.array([ind.center for ind in self.individuals])
#            #weights = jnp.array([1.0 - ind.fitness_score for ind in self.individuals])
#            #weights /= jnp.sum(weights)  # Normalize the weights
#            weights = jnp.power(self.decay_factor, jnp.arange(len(self.individuals)))
#            return jnp.average(centers, weights=weights, axis=0)
#        return None
#
#    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
#        if self.individuals:
#            stdevs = jnp.array([ind.stdev for ind in self.individuals])
#            #weights = jnp.array([1.0 - ind.fitness_score for ind in self.individuals])
#            #weights /= jnp.sum(weights)  # Normalize the weights
#            weights = jnp.power(self.decay_factor, jnp.arange(len(self.individuals)))
#            return jnp.average(stdevs, weights=weights, axis=0)
#        return None
#
#    def adjust_noise(self, scaled_noises: jnp.ndarray, index: int, index_counter: int) -> jnp.ndarray:
#        #if index < len(self.individuals):
#            # Adjust the noise for the assigned index if an individual is available
#        scaled_noises = scaled_noises.at[index].set(scaled_noises[index] * self.individuals[index_counter].noise_magnitude)
#        return scaled_noises
#
# class TopographicKS:
#    def __init__(self, num_clusters: int = 5):
#        self.center = None
#        self.stdev = None
#        self.num_clusters = num_clusters
#
#    def accept(self, individual: Dict):
#        # Implement the acceptance criteria and update logic for TopographicKS
#        pass
#
#    def update(self):
#        # Implement the update logic for TopographicKS (e.g., clustering)
#        pass
#
# class NormativeKS:
#    def __init__(self):
#        self.center = None
#        self.stdev = None
#
#    def accept(self, individual: Dict):
#        # Implement the acceptance criteria and update logic for NormativeKS
#        pass
#
#    def update(self):
#        # Implement the update logic for NormativeKS (e.g., updating statistics)
#        pass

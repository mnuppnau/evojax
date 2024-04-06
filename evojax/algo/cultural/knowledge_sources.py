# Description: This file contains the implementation of the knowledge sources (KS) used in the knowledge-based optimization (KBO) framework.

from typing import Dict, List, Optional
import jax.numpy as jnp
import jax
import jax.tree_util
import jax.lax
import numpy as np
import copy

from evojax.algo.cultural.population_space import Individual

def initialize_domain_ks(param_size: int):
    return (
        jnp.zeros(param_size), # center
        jnp.ones(param_size), # stdev
        jnp.array([1.0]), # noise_magnitude
        jnp.array([0.0]) # fitness_score
        )

def initialize_situational_ks(param_size: int, max_individuals: int = 100):
    # Pre-allocate arrays with zeros for each individual property, given max_individuals
    # Assuming 'center' and 'stdev' are of size 'param_size'
    return (
        jnp.zeros((param_size, max_individuals)), # center
        jnp.zeros((param_size, max_individuals)), # stdev
        jnp.ones(max_individuals), # noise_magnitude
        jnp.zeros(max_individuals) # fitness_score
    )

def initialize_history_ks(param_size: int, decay_factor: float = 0.8, num_iterations: int = 5000):
    # Pre-allocate arrays with zeros for each individual property, given num_iterations
    # Assuming 'center' and 'stdev' are of size 'param_size'
    return (
        jnp.zeros((param_size, num_iterations)), # center
        jnp.zeros((param_size, num_iterations)), # stdev
        jnp.ones(num_iterations), # noise_magnitude
        jnp.zeros(num_iterations) # fitness_score
    )

@jax.jit
def update_knowledge_sources(belief_space, best_individual, max_individuals=100):
    
    updated_belief_space_domain = belief_space[:1] + (best_individual,) + belief_space[2:]

    center, stdev, noise_magnitude, fitness_value = updated_belief_space_domain[2]
       
    best_center, best_stdev, best_noise_magnitude, best_fitness_value = best_individual
   
    #print('center shape : ', center.shape)
    #print('best center shape : ', best_center.shape)

    best_center_reshaped = best_center.reshape(-1, 1)

    updated_center = jnp.concatenate([best_center_reshaped, center], axis=1)[:, :max_individuals]
    #print('updated center shape : ', updated_center.shape) 
    updated_stdev = jnp.concatenate([best_stdev[:, None], stdev], axis=1)[:, :max_individuals]
    
    updated_noise_magnitude = jnp.concatenate([best_noise_magnitude, noise_magnitude], axis=0)[:max_individuals]
    
    updated_fitness_value = jnp.concatenate([best_fitness_value, fitness_value], axis=0)[:max_individuals]
     
    
    # Construct the updated situational knowledge source
    updated_situational_ks = (updated_center, updated_stdev, updated_noise_magnitude, updated_fitness_value)
    
    # Reconstruct the belief space with the updated situational KS
    updated_belief_space_situational = updated_belief_space_domain[:2] + (updated_situational_ks,) + updated_belief_space_domain[3:]
    
    updated_history_ks = belief_space[3]  # Placeholder operation
    updated_belief_space_history = updated_belief_space_situational[:3] + (updated_history_ks,) + updated_belief_space_situational[4:]
    
    return updated_belief_space_history

def get_center_guidance(belief_space,t):
    decay_factor_history = 0.20
    decay_factor_situational = 0.08

    max_iterations = 5000

    arr = jnp.array([t,100])
    n = jnp.min(arr)

    ks_weights = jnp.array([.2, .4, .4])
    #print("n : ", n)
    #print("t : ", t)
    domain_ks_center = belief_space[1][0]
    situational_ks_center = belief_space[2][0] #[:,:n]
    history_ks_center = belief_space[3][0] #[:,:t]


    situational_valid_columns_mask = jnp.arange(situational_ks_center.shape[1]) < t
    
    # Apply the mask to zero out columns beyond the current iteration.
    situational_masked_data = situational_ks_center * situational_valid_columns_mask

    column_indices_situational = jnp.arange(100)
    weights_situational = jnp.exp(-decay_factor_situational * column_indices_situational)
    
    masked_weights_situational = weights_situational * situational_valid_columns_mask

    # Compute weighted sums and the sum of masked weights for normalization
    weighted_sums_situational = jnp.sum(situational_masked_data * masked_weights_situational, axis=1)
    sum_of_masked_weights_situational = jnp.sum(masked_weights_situational)

    # Calculate the weighted average
    situational_weighted_averages = weighted_sums_situational / sum_of_masked_weights_situational



    #print('situation center : ', situational_ks_center)
    #print('situation center shape : ', situational_ks_center.shape)

    # Compute the weighted average of the centers
    #situational_ks_center_weights = jnp.array([1.0 - jnp.abs(fitness_score) for fitness_score in belief_space[2][3][:n]])
    #situational_ks_center_weights /= jnp.sum(situational_ks_center_weights)  # Normalize the weights
    #situational_ks_center_avg = jnp.average(situational_ks_center, weights=situational_ks_center_weights, axis=1)

    history_valid_columns_mask = jnp.arange(history_ks_center.shape[1]) < t
    
    # Apply the mask to zero out columns beyond the current iteration.
    history_masked_data = history_ks_center * history_valid_columns_mask

    column_indices = jnp.arange(max_iterations)
    weights = jnp.exp(-decay_factor_situational * column_indices)
    
    masked_weights = weights * history_valid_columns_mask

    # Compute weighted sums and the sum of masked weights for normalization
    weighted_sums = jnp.sum(history_masked_data * masked_weights, axis=1)
    sum_of_masked_weights = jnp.sum(masked_weights)

    # Calculate the weighted average
    history_weighted_averages = weighted_sums / sum_of_masked_weights

    domain_ks_center_weighted = domain_ks_center * ks_weights[0]
    situational_row_averages_weighted = situational_weighted_averages * ks_weights[1]
    history_row_averages_weighted = history_weighted_averages * ks_weights[2]
    
    #history_row_sums = history_masked_data.sum(axis=1)

    #history_row_averages = history_row_sums / t


    #history_ks_center_weights = jnp.power(decay_factor, jnp.arange(t))
    #history_ks_center_avg = jnp.average(history_ks_center, weights=history_ks_center_weights, axis=1)

    return domain_ks_center_weighted + situational_row_averages_weighted + history_row_averages_weighted
    #return jnp.mean(jnp.array([domain_ks_center_weighted, situational_row_averages_weighted, history_row_averages_weighted]), axis=0)
    #return domain_ks_center

def get_stdev_guidance(belief_space,t):
    decay_factor_historical = 0.20
    decay_factor_situational = 0.08

    max_iterations = 5000

    ks_weights = jnp.array([.2, .4, .4])
    
    arr = jnp.array([t,100])
    n = jnp.min(arr)

    domain_ks_stdev = belief_space[1][1]
    situational_ks_stdev = belief_space[2][1]#[:,:n]
    history_ks_stdev = belief_space[3][1]#[:,:t]


    situational_valid_columns_mask = jnp.arange(situational_ks_stdev.shape[1]) < t
    
    # Apply the mask to zero out columns beyond the current iteration.
    situational_masked_data = situational_ks_stdev * situational_valid_columns_mask

    column_indices_situational = jnp.arange(100)
    weights_situational = jnp.exp(-decay_factor_situational * column_indices_situational)
    
    masked_weights_situational = weights_situational * situational_valid_columns_mask

    # Compute weighted sums and the sum of masked weights for normalization
    weighted_sums_situational = jnp.sum(situational_masked_data * masked_weights_situational, axis=1)
    sum_of_masked_weights_situational = jnp.sum(masked_weights_situational)

    # Calculate the weighted average
    situational_weighted_averages = weighted_sums_situational / sum_of_masked_weights_situational

    #print('situation stdev : ', situational_ks_stdev)
    #print('situation stdev shape : ', situational_ks_stdev.shape)

    # Compute the weighted average of the stdevs
    #situational_ks_stdev_weights = jnp.array([1.0 - jnp.abs(fitness_score) for fitness_score in belief_space[2][3][:n]])
    #situational_ks_stdev_weights /= jnp.sum(situational_ks_stdev_weights)  # Normalize the weights
    #situational_ks_stdev_avg = jnp.average(situational_ks_stdev, weights=situational_ks_stdev_weights, axis=1)

    history_valid_columns_mask = jnp.arange(history_ks_stdev.shape[1]) < t
    
    # Apply the mask to zero out columns beyond the current iteration.
    history_masked_data = history_ks_stdev * history_valid_columns_mask

    column_indices = jnp.arange(max_iterations)
    weights = jnp.exp(-decay_factor_historical * column_indices)
    
    masked_weights = weights * history_valid_columns_mask

    # Compute weighted sums and the sum of masked weights for normalization
    weighted_sums = jnp.sum(history_masked_data * masked_weights, axis=1)
    sum_of_masked_weights = jnp.sum(masked_weights)

    # Calculate the weighted average
    history_weighted_averages = weighted_sums / sum_of_masked_weights

    # Compute the weighted average of the standard deviations
    #situational_ks_stdev_weights = jnp.array([1.0 - jnp.abs(fitness_score) for fitness_score in belief_space[2][3][:n]])
    #situational_ks_stdev_weights /= jnp.sum(situational_ks_stdev_weights)  # Normalize the weights
    #situational_ks_stdev_avg = jnp.average(situational_ks_stdev, axis=1)

    #history_ks_stdev_weights = jnp.power(decay_factor, jnp.arange(t))
    #history_ks_stdev_avg = jnp.average(history_ks_stdev, weights=history_ks_stdev_weights, axis=1)

    domain_ks_stdev_weighted = domain_ks_stdev * weights[0]
    situational_row_averages_weighted = situational_weighted_averages * weights[1]
    history_row_averages_weighted = history_weighted_averages * weights[2]

    return domain_ks_stdev_weighted + situational_row_averages_weighted + history_row_averages_weighted
    #return jnp.mean(jnp.array([domain_ks_stdev_weighted, situational_row_averages_weighted, history_row_averages_weighted]), axis=0)
    #return domain_ks_stdev

#class DomainKS:
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
#class SituationalKS:
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
#class HistoryKS:
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
#class TopographicKS:
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
#class NormativeKS:
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

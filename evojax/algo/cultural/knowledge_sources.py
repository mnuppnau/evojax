# Description: This file contains the implementation of the knowledge sources (KS) used in the knowledge-based optimization (KBO) framework.

from typing import Dict, List, Optional
import jax.numpy as jnp

from evojax.algo.cultural.population_space import Individual

def initialize_domain_ks():
    return {
        'individuals': [],  # List to store historical individuals; consider using a JAX-friendly structure if needed.
        'assigned_indexes': None,  
        'individual_count': 0,
    }

def initialize_situational_ks(max_individuals: int = 100):
    return {
        'individuals': [],  # List to store historical individuals; consider using a JAX-friendly structure if needed.
        'assigned_indexes': None,
        'individual_count': 0,
        'max_individuals': max_individuals,
    }

def initialize_history_ks(decay_factor=0.8):
    return {
        'individuals': [],  # List to store historical individuals; consider using a JAX-friendly structure if needed.
        'assigned_indexes': None,
        'individual_count': 0,
        'decay_factor': decay_factor,  # Decay factor to reduce the influence of older individuals.
    }

def accept_domain_ks(ks, individual):
    ks_updated = ks.copy()  # Copy for immutability
    ks_updated['individual'] = individual
    ks_updated['individual_count'] = 1
    ks_updated = update_domain_ks(ks_updated)
    return ks_updated

def accept_situational_ks(ks, individual):
    ks_updated = ks.copy()
    ks_updated['individuals'].append(individual)
    if len(ks_updated['individuals']) > ks_updated['max_individuals']:
        ks_updated = update_situational_ks(ks_updated)
    return ks_updated

def accept_history_ks(ks, individual):
    ks_updated = ks.copy()
    ks_updated['individuals'].append(individual)
    ks_updated = update_history_ks(ks_updated)
    return ks_updated

def update_domain_ks(ks):
    return ks

def update_situational_ks(ks):
    ks['individuals'] = sorted(ks['individuals'], key=lambda x: x['fitness_score'])[:ks['max_individuals']]
    return ks

def update_history_ks(ks):
    return ks

def update_belief_space(belief_space, ks):
    if ks == 'situational':
        return update_situational_ks(belief_space['situational'])
    elif ks == 'history':
        return update_history_ks(belief_space['history'])
    else:
        return update_domain_ks(belief_space['domain'])


def get_center_guidance(ks_data, ks_name):
    def get_center_guidance_domain_ks(ks_data):
        if ks_data.get('individual') is not None:
            return ks_data['individual']['center']
        return None
    
    def get_center_guidance_situational_ks(ks_data):
        if ks_data.get('individuals'):
            centers = jnp.array([ind['center'] for ind in ks_data['individuals']])
            # Assuming situational KS uses fitness-based weights for averaging
            weights = jnp.array([1.0 - jnp.abs(ind['fitness_score']) for ind in ks_data['individuals']])
            weights /= jnp.sum(weights)  # Normalize the weights
            return jnp.average(centers, weights=weights, axis=0)
        return None
    
    def get_center_guidance_history_ks(ks_data):
        if ks_data.get('individuals'):
            centers = jnp.array([ind['center'] for ind in ks_data['individuals']])
            # Assuming history KS uses time-based decay for weighting
            weights = jnp.power(ks_data['decay_factor'], jnp.arange(len(ks_data['individuals'])))
            return jnp.average(centers, weights=weights, axis=0)
        return None
    
    if ks_name == 'situational':
        return get_center_guidance_situational_ks(ks_data)
    elif ks_name == 'history':
        return get_center_guidance_history_ks(ks_data)
    else:
        return get_center_guidance_domain_ks(ks_data)

def get_stdev_guidance(ks_data, ks_name):
    # Similarly, define stdev guidance functions for each KS
    def get_stdev_guidance_domain_ks(ks_data):
        if ks_data.get('individual') is not None:
            return ks_data['individual']['stdev']
        return None
    
    def get_stdev_guidance_situational_ks(ks_data):
        if ks_data.get('individuals'):
            stdevs = jnp.array([ind['stdev'] for ind in ks_data['individuals']])
            weights = jnp.array([1.0 - jnp.abs(ind['fitness_score']) for ind in ks_data['individuals']])
            weights /= jnp.sum(weights)  # Normalize the weights
            return jnp.average(stdevs, weights=weights, axis=0)
        return None
    
    def get_stdev_guidance_history_ks(ks_data):
        if ks_data.get('individuals'):
            stdevs = jnp.array([ind['stdev'] for ind in ks_data['individuals']])
            weights = jnp.power(ks_data['decay_factor'], jnp.arange(len(ks_data['individuals'])))
            return jnp.average(stdevs, weights=weights, axis=0)
        return None

    if ks_name == 'situational':
        return get_stdev_guidance_situational_ks(ks_data)
    elif ks_name == 'history':
        return get_stdev_guidance_history_ks(ks_data)
    else:
        return get_stdev_guidance_domain_ks(ks_data)

def adjust_noise(ks_data, scaled_noises, index, index_counter):
    if index_counter < len(ks_data['individuals']) and ks_data['individuals'][index_counter] is not None:
        # Adjust the noise for the first index if an individual is available
        return scaled_noises.at[index].set(scaled_noises[index] * ks_data['individuals'][index_counter]['noise_magnitude'])
    return scaled_noises

def adjust_noise_domain_ks(scaled_noises, individual, assigned_indexes):
    # Check if the individual exists and assigned_indexes includes the first position
    if individual is not None and 0 in assigned_indexes:
        # Adjust noise only for the specific index
        index_to_adjust = assigned_indexes.index(0)  # Get the actual index to adjust
        adjusted_noise = scaled_noises.at[index_to_adjust].set(scaled_noises[index_to_adjust] * individual['noise_magnitude'])
        return adjusted_noise
    return scaled_noises

def adjust_noise_situational_ks(scaled_noises, individual, assigned_indexes):
    # Check if the individual exists and assigned_indexes includes the first position
    if individual is not None and 0 in assigned_indexes:
        # Adjust noise only for the specific index
        index_to_adjust = assigned_indexes.index(0)  # Get the actual index to adjust
        adjusted_noise = scaled_noises.at[index_to_adjust].set(scaled_noises[index_to_adjust] * individual['noise_magnitude'])
        return adjusted_noise
    return scaled_noises

def adjust_noise_history_ks(scaled_noises, individual, assigned_indexes):
    # Check if the individual exists and assigned_indexes includes the first position
    if individual is not None and 0 in assigned_indexes:
        # Adjust noise only for the specific index
        index_to_adjust = assigned_indexes.index(0)  # Get the actual index to adjust
        adjusted_noise = scaled_noises.at[index_to_adjust].set(scaled_noises[index_to_adjust] * individual['noise_magnitude'])
        return adjusted_noise
    return scaled_noises

class DomainKS:
    def __init__(self):
        self.individual = None
        self.assigned_indexes = None
        self.individual_count = 0

    def __repr__(self):
        if self.individual is None:
            return "DomainKS(individual=None)"
        else:
            return f"DomainKS(individual={self.individual})"

    def accept(self, individual: Individual):
        self.individual = individual
        self.individual_count = 1

    def update(self):
        pass

    def get_center_guidance(self) -> Optional[jnp.ndarray]:
        if self.individual is not None:
            #print("center guidance : ", self.individual.center)
            return self.individual.center
        return None

    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
        if self.individual is not None:
            return self.individual.stdev
        return None

    def adjust_noise(self, scaled_noises: jnp.ndarray, index: int, index_counter: int) -> jnp.ndarray:
        if self.individual is not None and index == 0:
            # Adjust the noise for the first index if an individual is available
            scaled_noises = scaled_noises.at[index].set(scaled_noises[index] * self.individual.noise_magnitude)
        return scaled_noises

class SituationalKS:
    def __init__(self, max_individuals: int = 100):
        self.max_individuals = max_individuals
        self.individuals: List[Individual] = []
        self.assigned_indexes = None
        self.individual_count = 0

    def __repr__(self):
        individual_str = "\n".join([str(individual) for individual in self.individuals])
        return f"SituationalKS(max_individuals={self.max_individuals}, individuals=[\n{individual_str}\n])"

    def accept(self, individual: Individual):
        self.individuals.append(individual)
        #self.individual_count += 1
        self.individual_count = min(len(self.individuals), self.max_individuals)
        #print("individual count in situational KS : ", self.individual_count)

    def update(self):
        self.individuals.sort(key=lambda x: x.fitness_score)
        self.individuals = self.individuals[:self.max_individuals]

    def get_center_guidance(self) -> Optional[jnp.ndarray]:
        if self.individuals:
            centers = jnp.array([ind.center for ind in self.individuals])
            weights = jnp.array([1.0 - jnp.abs(ind.fitness_score) for ind in self.individuals])
            weights /= jnp.sum(weights)  # Normalize the weights
            return jnp.average(centers, weights=weights, axis=0)
        return None
    
    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
        if self.individuals:
            stdevs = jnp.array([ind.stdev for ind in self.individuals])
            weights = jnp.array([1.0 - jnp.abs(ind.fitness_score) for ind in self.individuals])
            weights /= jnp.sum(weights)  # Normalize the weights
            return jnp.average(stdevs, weights=weights, axis=0)
        return None

    #def get_center_guidance(self) -> Optional[jnp.ndarray]:
    #    if self.individuals:
    #        centers = jnp.array([ind.center for ind in self.individuals])
    #        weights = jnp.array([jnp.absolute(ind.fitness_score) for ind in self.individuals])
    #        #print("average center : ", jnp.average(centers, weights=weights, axis=0))
    #        return jnp.average(centers, weights=weights, axis=0)
    #    return None

    #def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
    #    if self.individuals:
    #        stdevs = jnp.array([ind.stdev for ind in self.individuals])
    #        weights = jnp.array([jnp.absolute(ind.fitness_score) for ind in self.individuals])
    #        return jnp.average(stdevs, weights=weights, axis=0)
    #    return None

    def adjust_noise(self, scaled_noises: jnp.ndarray, index: int, index_counter: int) -> jnp.ndarray:
        #if index < len(self.individuals):
            # Adjust the noise for the assigned index if an individual is available
        scaled_noises = scaled_noises.at[index].set(scaled_noises[index] * self.individuals[index_counter].noise_magnitude)
        return scaled_noises

class HistoryKS:
    def __init__(self, decay_factor: float = 0.8):
        self.individuals: List[Individual] = []
        self.decay_factor = decay_factor
        self.assigned_indexes = None
        self.individual_count = 0

    def __repr__(self):
        individual_str = "\n".join([str(individual) for individual in self.individuals])
        return f"HistoryKS(decay_factor={self.decay_factor}, individuals=[\n{individual_str}\n])"

    def accept(self, individual: Individual):
        self.individuals.append(individual)
        self.individual_count = len(self.individuals)
        #self.individual_count += 1
        #print("individual count in history KS : ", self.individual_count)

    def update(self):
        pass

    def get_center_guidance(self) -> Optional[jnp.ndarray]:
        if self.individuals:
            centers = jnp.array([ind.center for ind in self.individuals])
            #weights = jnp.array([1.0 - ind.fitness_score for ind in self.individuals])
            #weights /= jnp.sum(weights)  # Normalize the weights
            weights = jnp.power(self.decay_factor, jnp.arange(len(self.individuals))) 
            return jnp.average(centers, weights=weights, axis=0)
        return None
    
    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
        if self.individuals:
            stdevs = jnp.array([ind.stdev for ind in self.individuals])
            #weights = jnp.array([1.0 - ind.fitness_score for ind in self.individuals])
            #weights /= jnp.sum(weights)  # Normalize the weights
            weights = jnp.power(self.decay_factor, jnp.arange(len(self.individuals))) 
            return jnp.average(stdevs, weights=weights, axis=0)
        return None

    def adjust_noise(self, scaled_noises: jnp.ndarray, index: int, index_counter: int) -> jnp.ndarray:
        #if index < len(self.individuals):
            # Adjust the noise for the assigned index if an individual is available
        scaled_noises = scaled_noises.at[index].set(scaled_noises[index] * self.individuals[index_counter].noise_magnitude)
        return scaled_noises

class TopographicKS:
    def __init__(self, num_clusters: int = 5):
        self.center = None
        self.stdev = None 
        self.num_clusters = num_clusters

    def accept(self, individual: Dict):
        # Implement the acceptance criteria and update logic for TopographicKS
        pass

    def update(self):
        # Implement the update logic for TopographicKS (e.g., clustering)
        pass

class NormativeKS:
    def __init__(self):
        self.center = None
        self.stdev = None

    def accept(self, individual: Dict):
        # Implement the acceptance criteria and update logic for NormativeKS
        pass

    def update(self):
        # Implement the update logic for NormativeKS (e.g., updating statistics)
        pass

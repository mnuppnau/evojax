# Description: This file contains the implementation of the knowledge sources (KS) used in the knowledge-based optimization (KBO) framework.

from typing import Dict, List, Optional
import jax.numpy as jnp

from evojax.algo.cultural.population_space import Individual

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
    def __init__(self, max_individuals: int = 40):
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
            weights = jnp.array([1.0 - ind.fitness_score for ind in self.individuals])
            weights /= jnp.sum(weights)  # Normalize the weights
            return jnp.average(centers, weights=weights, axis=0)
        return None
    
    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
        if self.individuals:
            stdevs = jnp.array([ind.stdev for ind in self.individuals])
            weights = jnp.array([1.0 - ind.fitness_score for ind in self.individuals])
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
            weights = jnp.array([1.0 - ind.fitness_score for ind in self.individuals])
            weights /= jnp.sum(weights)  # Normalize the weights
            return jnp.average(centers, weights=weights, axis=0)
        return None
    
    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
        if self.individuals:
            stdevs = jnp.array([ind.stdev for ind in self.individuals])
            weights = jnp.array([1.0 - ind.fitness_score for ind in self.individuals])
            weights /= jnp.sum(weights)  # Normalize the weights
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

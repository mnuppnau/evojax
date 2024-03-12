# Description: This file contains the implementation of the knowledge sources (KS) used in the knowledge-based optimization (KBO) framework.

from typing import Dict, List, Optional
import jax.numpy as jnp

from evojax.algo.cultural.population_space import Individual

class DomainKS:
    def __init__(self):
        self.individual = None
        self.assigned_indexes = None
        self.individual_count = 0

    def accept(self, individual: Individual):
        self.individual = individual
        self.individual_count += 1

    def update(self):
        pass

    def get_center_guidance(self) -> Optional[jnp.ndarray]:
        if self.individual is not None:
            return self.individual.center
        return None

    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
        if self.individual is not None:
            return self.individual.stdev
        return None

    def adjust_noise(self, scaled_noises: jnp.ndarray, index: int) -> jnp.ndarray:
        if self.individual is not None and index == 0:
            # Adjust the noise for the first index if an individual is available
            scaled_noises = scaled_noises.at[index].set(scaled_noises[index] * self.individual.noise_magnitude)
        return scaled_noises

class SituationalKS:
    def __init__(self, max_individuals: int = 10):
        self.max_individuals = max_individuals
        self.individuals: List[Individual] = []
        self.assigned_indexes = None
        self.individual_count = 0

    def accept(self, individual: Individual):
        self.individuals.append(individual)
        self.individual_count += 1

    def update(self):
        self.individuals.sort(key=lambda x: x.fitness_score, reverse=True)
        self.individuals = self.individuals[:self.max_individuals]

    def get_center_guidance(self) -> Optional[jnp.ndarray]:
        if self.individuals:
            centers = jnp.array([ind.center for ind in self.individuals])
            weights = jnp.array([ind.fitness_score for ind in self.individuals])
            return jnp.average(centers, weights=weights, axis=0)
        return None

    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
        if self.individuals:
            stdevs = jnp.array([ind.stdev for ind in self.individuals])
            weights = jnp.array([ind.fitness_score for ind in self.individuals])
            return jnp.average(stdevs, weights=weights, axis=0)
        return None

    def adjust_noise(self, scaled_noises: jnp.ndarray, index: int) -> jnp.ndarray:
        if index < len(self.individuals):
            # Adjust the noise for the assigned index if an individual is available
            scaled_noises = scaled_noises.at[index].set(scaled_noises[index] * self.individuals[index].noise_magnitude)
        return scaled_noises

class HistoryKS:
    def __init__(self, decay_factor: float = 0.9):
        self.individuals: List[Individual] = []
        self.decay_factor = decay_factor
        self.assigned_indexes = None
        self.individual_count = 0

    def accept(self, individual: Individual):
        self.individuals.append(individual)
        self.individual_count += 1

    def update(self):
        pass

    def get_center_guidance(self) -> Optional[jnp.ndarray]:
        if self.individuals:
            centers = jnp.array([ind.center for ind in self.individuals])
            weights = jnp.power(self.decay_factor, jnp.arange(len(self.individuals)))
            return jnp.average(centers, weights=weights, axis=0)
        return None

    def get_stdev_guidance(self) -> Optional[jnp.ndarray]:
        if self.individuals:
            stdevs = jnp.array([ind.stdev for ind in self.individuals])
            weights = jnp.power(self.decay_factor, jnp.arange(len(self.individuals)))
            return jnp.average(stdevs, weights=weights, axis=0)
        return None

    def adjust_noise(self, scaled_noises: jnp.ndarray, index: int) -> jnp.ndarray:
        if index < len(self.individuals):
            # Adjust the noise for the assigned index if an individual is available
            scaled_noises = scaled_noises.at[index].set(scaled_noises[index] * self.individuals[index].noise_magnitude)
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

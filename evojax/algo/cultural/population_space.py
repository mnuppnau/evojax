import jax.numpy as jnp
import numpy as np

def create_individual(center: jnp.ndarray, stdev: jnp.ndarray, noise_magnitude: float = 0.0, fitness_score: float = 0.0):
    return {
        'center': center,
        'stdev': stdev,
        'noise_magnitude': noise_magnitude,
        'fitness_score': fitness_score
    }

def initialize_population(pop_size: int, center: jnp.ndarray, stdev: jnp.ndarray):
    return [create_individual(center, stdev) for _ in range(pop_size)]

def update_population(fitness_scores: jnp.ndarray, center: jnp.ndarray, stdev: jnp.ndarray, scaled_noises: jnp.ndarray):
    # Compute noise magnitudes for all individuals at once
    noise_magnitudes = jnp.linalg.norm(scaled_noises, axis=1)

    # Update all individuals in the population at once
    # Here, 'center' and 'stdev' are assumed to be the same for all individuals
    # This creates a new representation for the entire population
    updated_population = {
        'center': jnp.tile(center[None, :], (fitness_scores.shape[0], 1)),  # Replicate the center for each individual
        'stdev': jnp.tile(stdev[None, :], (fitness_scores.shape[0], 1)),     # Replicate the stdev for each individual
        'noise_magnitude': noise_magnitudes,  # Already vectorized
        'fitness_score': fitness_scores       # Directly use the provided scores
    }
    return updated_population

class Individual:
    def __init__(self, center: jnp.ndarray, stdev: jnp.ndarray, noise_magnitude: float, fitness_score: float):
        self.center = center
        self.stdev = stdev
        self.noise_magnitude = noise_magnitude
        self.fitness_score = fitness_score

    def __repr__(self):
         return f"Individual(fitness_score={self.fitness_score}, noise_magnitude={self.noise_magnitude})"

class PopulationSpace:
    def __init__(self, pop_size: int):
        self.pop_size = pop_size
        self.individuals: List[Individual] = []

    def initialize(self, center: jnp.ndarray, stdev: jnp.ndarray):
        for _ in range(self.pop_size):
            individual = Individual(center, stdev, noise_magnitude=0.0, fitness_score=0.0)
            self.individuals.append(individual)

    def update(self, fitness_scores: jnp.ndarray, center: jnp.ndarray, stdev: jnp.ndarray, scaled_noises: jnp.ndarray):
        for i, individual in enumerate(self.individuals):
            individual.center = center
            individual.stdev = stdev
            individual.noise_magnitude = jnp.linalg.norm(scaled_noises[i])
            individual.fitness_score = fitness_scores[i]

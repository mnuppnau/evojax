import jax.numpy as jnp
import numpy as np

class Individual:
    def __init__(self, center: jnp.ndarray, stdev: jnp.ndarray, noise_magnitude: float, fitness_score: float):
        self.center = center
        self.stdev = stdev
        self.noise_magnitude = noise_magnitude
        self.fitness_score = fitness_score

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

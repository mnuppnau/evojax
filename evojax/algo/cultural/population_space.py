import jax.numpy as jnp
import numpy as np

def update_population(center: jnp.ndarray, stdev: jnp.ndarray, fitness_scores: jnp.ndarray):
    
    updated_population = (
        center,                   
        stdev,                    
        fitness_scores         
    )
   
    best_index = jnp.argmax(fitness_scores)
   
    best_fitness_scores = fitness_scores[best_index]

    center = jnp.atleast_1d(center)
    stdev = jnp.atleast_1d(stdev)
    best_fitness_scores = jnp.atleast_1d(best_fitness_scores)

    return updated_population, (center, stdev, best_fitness_scores)

class Individual:
    def __init__(self, center: jnp.ndarray, stdev: jnp.ndarray, fitness_score: float):
        self.center = center
        self.stdev = stdev
        self.fitness_score = fitness_score

    def __repr__(self):
         return f"Individual(fitness_score={self.fitness_score})"

class PopulationSpace:
    def __init__(self, pop_size: int):
        self.pop_size = pop_size
        self.individuals: List[Individual] = []

    def initialize(self, center: jnp.ndarray, stdev: jnp.ndarray):
        for _ in range(self.pop_size):
            individual = Individual(center, stdev, noise_magnitude=0.0, fitness_score=0.0)
            self.individuals.append(individual)

    def update(self, fitness_scores: jnp.ndarray, center: jnp.ndarray, stdev: jnp.ndarray):
        for i, individual in enumerate(self.individuals):
            individual.center = center
            individual.stdev = stdev
            individual.fitness_score = fitness_scores[i]

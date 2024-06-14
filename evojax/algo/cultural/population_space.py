import jax.numpy as jnp
import numpy as np

def initialize_population(param_size: int, pop_size: int):
    pop_center_grad = jnp.zeros((pop_size, param_size))
    pop_stdev_grad = jnp.zeros((pop_size, param_size))
    pop_connections = jnp.zeros(pop_size)
    pop_connection_weights = jnp.zeros(pop_size)

def update_population(center: jnp.ndarray, stdev: jnp.ndarray, fitness_scores: jnp.ndarray, scaled_noises: jnp.ndarray):
    
    noise_magnitudes = jnp.linalg.norm(scaled_noises, axis=1)

    updated_population = (
        center,                   
        stdev,                    
        noise_magnitudes,      
        fitness_scores         
    )
   
    best_index = jnp.argmax(fitness_scores)
   
    # Extract the corresponding values for the best individual
    best_noise_magnitude = noise_magnitudes[best_index]  # Assuming noise_magnitude is 1D: individuals
    best_fitness_scores = fitness_scores[best_index]

    center = jnp.atleast_1d(center)
    stdev = jnp.atleast_1d(stdev)
    best_noise_magnitude = jnp.atleast_1d(best_noise_magnitude)
    best_fitness_scores = jnp.atleast_1d(best_fitness_scores)

    return updated_population, (center, stdev, best_noise_magnitude, best_fitness_scores)

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

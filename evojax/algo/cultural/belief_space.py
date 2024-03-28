import random
import jax.numpy as jnp
import jax

from jax import random
from typing import List, Tuple ,Dict
from evojax.algo.cultural.knowledge_sources import initialize_domain_ks, initialize_situational_ks, initialize_history_ks, adjust_noise
from evojax.algo.cultural.knowledge_sources import accept_domain_ks, accept_situational_ks, accept_history_ks, get_center_guidance, get_stdev_guidance


def initialize_belief_space(population_size: int, key: int, num_clusters: int = 3):
    belief_space = {
        'population_size': population_size,
        'domain_ks': initialize_domain_ks(),
        'situational_ks': initialize_situational_ks(),
        'history_ks': initialize_history_ks(),
        #'topographic_ks': initialize_topographic_ks(num_clusters),
        #'normative_ks': initialize_normative_ks(),
        'learning_rate': 0.45
    }
    return assign_indexes_to_knowledge_sources(belief_space, key)

def assign_indexes_to_knowledge_sources(belief_space, key):
    population_size = belief_space['population_size']
    indexes = jnp.arange(1, population_size)  # Exclude index 0

    shuffled_numbers = jax.random.shuffle(key, indexes)

    set1 = shuffled_numbers[:5]
    set2 = shuffled_numbers[5:10]

    belief_space['domain_ks']['assigned_indexes'] = [0]  # Domain KS gets the first index
    # Update other KS based on your logic; here's an example for situational and history:
    belief_space['situational_ks']['assigned_indexes'] = set1.tolist()
    belief_space['history_ks']['assigned_indexes'] = set2.tolist()
    # If there are more KS, assign them indexes in similar fashion.

    # Return the updated belief space
    return belief_space

# Mapping functions to their corresponding KS
accept_function_mapping = {
    'domain_ks': accept_domain_ks,
    'situational_ks': accept_situational_ks,
    'history_ks': accept_history_ks
}


def influence(belief_space, scaled_noises):
    for ks_name in ['domain_ks','situational_ks','history_ks']:  # Add or remove KS names as necessary
        ks_data = belief_space[ks_name]
        index_counter = 0
        for i in ks_data['assigned_indexes']:
            scaled_noises = adjust_noise(ks_data,scaled_noises, i, index_counter)
            index_counter += 1
    return scaled_noises

def get_updated_params(belief_space, center, stdev):
    combined_guidance_center = combine_guidance(belief_space, center, "center")
    combined_guidance_stdev = combine_guidance(belief_space, stdev, "stdev")

    new_center = (1 - belief_space['learning_rate']) * center + belief_space['learning_rate'] * combined_guidance_center
    new_stdev = (1 - belief_space['learning_rate']) * stdev + belief_space['learning_rate'] * combined_guidance_stdev

    return new_center, new_stdev

def combine_guidance(belief_space, current_value, param_type):
    guidance_values = []

    for ks_name in ['domain_ks','situational_ks','history_ks']:  # Add or remove KS names as necessary
        ks_data = belief_space[ks_name]
        guidance_value = None
        if param_type == "center":
            guidance_value = get_center_guidance(ks_data, ks_name)
        elif param_type == "stdev":
            guidance_value = get_stdev_guidance(ks_data, ks_name)
        
        if guidance_value is not None:
            guidance_values.append(guidance_value)

    if guidance_values:
        combined_guidance = jnp.mean(jnp.stack(guidance_values), axis=0)
    else:
        combined_guidance = current_value

    return combined_guidance

class BeliefSpace:
    def __init__(self, population_size: int, num_clusters: int = 3):
        self.population_size = population_size
        self.domain_ks = DomainKS()
        self.situational_ks = SituationalKS()
        self.history_ks = HistoryKS()
        self.topographic_ks = TopographicKS(num_clusters)
        self.normative_ks = NormativeKS()
        self.assign_indexes_to_knowledge_sources()
        self.learning_rate = 0.45

    def assign_indexes_to_knowledge_sources(self):
        indexes = list(range(1, self.population_size))  # Exclude index 0
        for ks in [self.situational_ks, self.history_ks]:
            ks.assigned_indexes = random.sample(indexes, k=len(indexes) // 12)
            indexes = [i for i in indexes if i not in ks.assigned_indexes]
        self.domain_ks.assigned_indexes = [0]  # Assign index 0 to DomainKS

    def accept(self, individuals: List[Dict]):
        best_individual = min(individuals, key=lambda x: x.fitness_score)
        self.domain_ks.accept(best_individual)
        self.situational_ks.accept(best_individual)
        self.history_ks.accept(best_individual)

    def update(self):
        # Update the knowledge sources based on newly accepted values
        self.domain_ks.update()
        self.situational_ks.update()
        self.history_ks.update()
        #self.topographic_ks.update()
        #self.normative_ks.update()

    def influence(self, scaled_noises: jnp.ndarray) -> jnp.ndarray:
        for ks in [self.domain_ks, self.situational_ks, self.history_ks]:
            index_counter = 0
            for i in ks.assigned_indexes:
                if index_counter < ks.individual_count:
                    scaled_noises = ks.adjust_noise(scaled_noises, i, index_counter)
                    index_counter += 1
        return scaled_noises

    def get_updated_params(self, center: jnp.ndarray, stdev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        combined_guidance_center = self.combine_guidance(center, "center")
        combined_guidance_stdev = self.combine_guidance(stdev, "stdev")

        new_center = (1 - self.learning_rate) * center + self.learning_rate * combined_guidance_center
        new_stdev = (1 - self.learning_rate) * stdev + self.learning_rate * combined_guidance_stdev

        return new_center, new_stdev

    def combine_guidance(self, current_value: jnp.ndarray, param_type: str) -> jnp.ndarray:
        guidance_values = []

        for ks in [ self.history_ks]:
            if param_type == "center":
                guidance_value = ks.get_center_guidance()
            elif param_type == "stdev":
                guidance_value = ks.get_stdev_guidance()
            else:
                raise ValueError(f"Invalid parameter type: {param_type}")

            if guidance_value is not None:
                guidance_values.append(guidance_value)

        if guidance_values:
            combined_guidance = jnp.mean(jnp.array(guidance_values), axis=0)
        else:
            combined_guidance = current_value

        return combined_guidance

import random as jrandom
import jax.numpy as jnp
import jax

#from jax import jrandom
from typing import List, Tuple ,Dict
from evojax.algo.cultural.knowledge_sources import initialize_domain_ks, initialize_situational_ks, initialize_history_ks, initialize_topographic_ks, initialize_normative_ks, get_center_guidance, get_stdev_guidance
#from evojax.algo.cultural.knowledge_sources import accept_domain_ks, accept_situational_ks, accept_history_ks, get_center_guidance, get_stdev_guidance

def initialize_belief_space(population_size: int, param_size: int, key: int , scaled_noises_adjustment_rate: float ,num_clusters: int = 3):
    belief_space = (
        jnp.array([population_size]), 
        initialize_domain_ks(param_size),
        initialize_situational_ks(param_size),
        initialize_history_ks(param_size),
        initialize_topographic_ks(param_size, num_clusters),
        initialize_normative_ks(param_size),
        jnp.array([0.6]),
        generate_scaled_noises_indexes(population_size, key, scaled_noises_adjustment_rate),
    )
    return belief_space

def generate_scaled_noises_indexes(population_size, key, scaled_noises_adjustment_rate):
    indexes = jnp.arange(1, population_size)
    
    shuffled_numbers = jax.random.shuffle(key, indexes)

    max_index = int(population_size * scaled_noises_adjustment_rate)
   
    shuffled_numbers_subset = shuffled_numbers[:max_index]

    return shuffled_numbers_subset

def influence(belief_space, scaled_noises):
    # Extracted and concatenated noise magnitudes
    domain_noise = belief_space[1][2]  # Domain KS noise magnitude
    situational_noise = belief_space[2][2][:len(belief_space[7]) // 2]  # Half from situational KS
    history_noise = belief_space[3][2][:(len(belief_space[7]) + 1) // 2]  # Half (rounded up) from history KS
    
    noise_magnitudes = jnp.concatenate([domain_noise, situational_noise, history_noise])
    
    noise_magnitudes = noise_magnitudes[:len(belief_space[7])]
    
    noise_magnitudes = noise_magnitudes[:, None]  # Properly broadcasting across columns
    
    selected_rows = scaled_noises[belief_space[7], :]
    
    adjusted_rows = selected_rows * noise_magnitudes
    
    adjusted_scaled_noises = scaled_noises.at[belief_space[7], :].set(adjusted_rows)

    return adjusted_scaled_noises

@jax.jit
def get_updated_params(belief_space, stdev, t):
    learning_rate = belief_space[6][0] 
    combined_guidance_center, topographic_center_weight = combine_center_guidance(belief_space, t)
    combined_guidance_stdev, topographic_stdev_weight = combine_stdev_guidance(belief_space, t, stdev)

    #new_center = (1 - ((learning_rate//2)-(.1*topographic_center_weight))) * center + ((learning_rate//2)-(.1*topographic_center_weight)) * combined_guidance_center
    #new_stdev = ((1 - ((learning_rate)+(.3*topographic_stdev_weight))) * stdev + ((learning_rate)+(0.7*topographic_stdev_weight)) * combined_guidance_stdev)

    new_center = combined_guidance_center
    new_stdev = combined_guidance_stdev

    return new_center, new_stdev

def combine_center_guidance(belief_space, t):
    return get_center_guidance(belief_space, t)

def combine_stdev_guidance(belief_space,t, stdev):
    return get_stdev_guidance(belief_space, t, stdev)

#class BeliefSpace:
#    def __init__(self, population_size: int, num_clusters: int = 3):
#        self.population_size = population_size
#        self.domain_ks = DomainKS()
#        self.situational_ks = SituationalKS()
#        self.history_ks = HistoryKS()
#        self.topographic_ks = TopographicKS(num_clusters)
#        self.normative_ks = NormativeKS()
#        self.assign_indexes_to_knowledge_sources()
#        self.learning_rate = 0.45
#
#    def assign_indexes_to_knowledge_sources(self):
#        indexes = list(range(1, self.population_size))  # Exclude index 0
#        for ks in [self.situational_ks, self.history_ks]:
#            ks.assigned_indexes = random.sample(indexes, k=len(indexes) // 12)
#            indexes = [i for i in indexes if i not in ks.assigned_indexes]
#        self.domain_ks.assigned_indexes = [0]  # Assign index 0 to DomainKS
#
#    def accept(self, individuals: List[Dict]):
#        best_individual = min(individuals, key=lambda x: x.fitness_score)
#        self.domain_ks.accept(best_individual)
#        self.situational_ks.accept(best_individual)
#        self.history_ks.accept(best_individual)
#
#    def update(self):
#        # Update the knowledge sources based on newly accepted values
#        self.domain_ks.update()
#        self.situational_ks.update()
#        self.history_ks.update()
#        #self.topographic_ks.update()
#        #self.normative_ks.update()
#
#    def influence(self, scaled_noises: jnp.ndarray) -> jnp.ndarray:
#        for ks in [self.domain_ks, self.situational_ks, self.history_ks]:
#            index_counter = 0
#            for i in ks.assigned_indexes:
#                if index_counter < ks.individual_count:
#                    scaled_noises = ks.adjust_noise(scaled_noises, i, index_counter)
#                    index_counter += 1
#        return scaled_noises
#
#    def get_updated_params(self, center: jnp.ndarray, stdev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
#        combined_guidance_center = self.combine_guidance(center, "center")
#        combined_guidance_stdev = self.combine_guidance(stdev, "stdev")
#
#        new_center = (1 - self.learning_rate) * center + self.learning_rate * combined_guidance_center
#        new_stdev = (1 - self.learning_rate) * stdev + self.learning_rate * combined_guidance_stdev
#
#        return new_center, new_stdev
#
#    def combine_guidance(self, current_value: jnp.ndarray, param_type: str) -> jnp.ndarray:
#        guidance_values = []
#
#        for ks in [ self.history_ks]:
#            if param_type == "center":
#                guidance_value = ks.get_center_guidance()
#            elif param_type == "stdev":
#                guidance_value = ks.get_stdev_guidance()
#            else:
#                raise ValueError(f"Invalid parameter type: {param_type}")
#
#            if guidance_value is not None:
#                guidance_values.append(guidance_value)
#
#        if guidance_values:
#            combined_guidance = jnp.mean(jnp.array(guidance_values), axis=0)
#        else:
#            combined_guidance = current_value
#
#        return combined_guidance

import random
import jax.numpy as jnp
import jax
from typing import List, Tuple ,Dict
from evojax.algo.cultural.knowledge_sources import DomainKS, SituationalKS, HistoryKS, TopographicKS, NormativeKS

class BeliefSpace:
    def __init__(self, population_size: int, num_clusters: int = 3):
        self.population_size = population_size
        self.domain_ks = DomainKS()
        self.situational_ks = SituationalKS()
        self.history_ks = HistoryKS()
        self.topographic_ks = TopographicKS(num_clusters)
        self.normative_ks = NormativeKS()
        self.assign_indexes_to_knowledge_sources()
        self.learning_rate = 0.2

    def assign_indexes_to_knowledge_sources(self):
        indexes = list(range(1, self.population_size))  # Exclude index 0
        for ks in [self.situational_ks, self.history_ks]:
            ks.assigned_indexes = random.sample(indexes, k=len(indexes) // 16)
            indexes = [i for i in indexes if i not in ks.assigned_indexes]
        self.domain_ks.assigned_indexes = [0]  # Assign index 0 to DomainKS

    def accept(self, individuals: List[Dict]):
        # Accept individuals into the belief space based on certain criteria
        best_individual = min(individuals, key=lambda x: x.fitness_score)
        #print("best individual : ", best_individual)
        #print("accepting individual into domain_ks,...")
        self.domain_ks.accept(best_individual)
        #print("accepting individual into situational_ks,...")
        self.situational_ks.accept(best_individual)
        #print("accepting individual into history_ks,...")
        self.history_ks.accept(best_individual)

        #print(f"Accepted best individual: {best_individual}")

    def acceptance_criteria(self, individual: Dict) -> bool:
        # Define the acceptance criteria for individuals
        # Return True if the individual meets the criteria, False otherwise
        pass

    def update(self):
        # Update the knowledge sources based on newly accepted values
        #print("updating domain_ks,...")
        self.domain_ks.update()
        #print("updating situational_ks,...")
        self.situational_ks.update()
        #print("updating history_ks,...")
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

        #print("center : ", center)
        #print("original stdev : ", stdev)
        #print("adjusted stdev : ", combined_guidance_stdev)
        
        new_center = (1 - self.learning_rate) * center + self.learning_rate * combined_guidance_center
        new_stdev = (1 - self.learning_rate) * stdev + self.learning_rate * combined_guidance_stdev

        return new_center, new_stdev

    def combine_guidance(self, current_value: jnp.ndarray, param_type: str) -> jnp.ndarray:
        guidance_values = []

        for ks in [self.domain_ks, self.situational_ks, self.history_ks]:
            #print("ks : ", ks)
            #print("param_type : ", param_type)
            if param_type == "center":
                guidance_value = ks.get_center_guidance()
            elif param_type == "stdev":
                guidance_value = ks.get_stdev_guidance()
            else:
                raise ValueError(f"Invalid parameter type: {param_type}")

            #print("guidance_value : ", guidance_value)
            if guidance_value is not None:
                guidance_values.append(guidance_value)

        if guidance_values:
            combined_guidance = jnp.mean(jnp.array(guidance_values), axis=0)
        else:
            combined_guidance = current_value

        return combined_guidance

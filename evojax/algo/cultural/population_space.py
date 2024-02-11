import numpy as np

class PopulationSpace:
    """Manages the population of individuals and their interactions."""
    
    def __init__(self, pop_size: int, individual_structure: Dict[str, Any], num_knowledge_sources: int):
        self.pop_size = pop_size
        self.individuals = [self._create_individual(i, individual_structure) for i in range(pop_size)]
        self.num_knowledge_sources = num_knowledge_sources  # Total number of available KSs

    def _create_individual(self, id: int, structure: Dict[str, Any]) -> Individual:
        """Create a new individual based on the provided structure."""
        individual_data = {k: v for k, v in structure.items()}
        return Individual(id=id, data=individual_data)

    def connect_individuals(self, connection_strategy: str = 'random', **kwargs):
        """Connect individuals using the specified strategy."""
        if connection_strategy == 'random':
            self._random_connections(**kwargs)
        # Implement other strategies as needed.

    def _random_connections(self, num_connections: int = 2):
        """Randomly connect individuals with a specified number of connections."""
        for individual in self.individuals:
            potential_neighbors = [i for i in self.individuals if i.id != individual.id]
            individual.neighbors = np.random.choice(potential_neighbors, num_connections, replace=False).tolist()

    def assign_knowledge_sources_randomly(self):
        """Randomly assign a Knowledge Source to each individual."""
        for individual in self.individuals:
            # Assign a random KS from the available pool
            individual.knowledge_source_id = np.random.randint(0, self.num_knowledge_sources)

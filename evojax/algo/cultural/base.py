from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import numpy as np
import jax.numpy as jnp

class CulturalAlgorithm(ABC):
    """Base class for cultural algorithms in EvoJAX."""
    pop_size: int

    @abstractmethod
    def initialize_population(self) -> None:
        """Initialize the population space with individuals."""
        raise NotImplementedError()

    @abstractmethod
    def ask(self) -> jnp.ndarray:
        """Ask the algorithm for a population of parameters.

        This method should consider both the individual parameters and the influence
        of the Knowledge Sources and Belief Space.

        Returns:
            A Jax array of shape (population_size, param_size).
        """
        raise NotImplementedError()

    @abstractmethod
    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        """Report the fitness of the population to the algorithm.

        This method should also handle the selection of promising individuals to update
        the Belief Space and influence the Knowledge Sources.

        Args:
            fitness - The fitness scores array.
        """
        raise NotImplementedError()

    @abstractmethod
    def update_knowledge_sources(self) -> None:
        """Update the Knowledge Sources based on the information from the Belief Space."""
        raise NotImplementedError()

    @abstractmethod
    def distribute_knowledge(self) -> None:
        """Distribute knowledge from the Knowledge Sources to the population."""
        raise NotImplementedError()

    def save_state(self) -> Any:
        """Optionally, save the state of the algorithm.

        Returns:
            Saved state.
        """
        return None

    def load_state(self, saved_state: Any) -> None:
        """Optionally, load the saved state of the algorithm.

        Args:
            saved_state - The result of self.save_state().
        """
        pass

    @property
    @abstractmethod
    def best_params(self) -> jnp.ndarray:
        """Return the best parameters found so far.

        Implementations should define how the best parameters are determined, potentially
        considering both the individual fitness and the influence of cultural factors.

        Returns:
            The best parameter set as a Jax array.
        """
        raise NotImplementedError()

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        """Optionally, set the best parameters manually.

        Args:
            params - The parameter set to be considered the best.
        """
        pass


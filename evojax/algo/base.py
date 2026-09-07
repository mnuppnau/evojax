# Copyright 2022 The EvoJAX Authors. Licensed under Apache-2.0.
"""Minimal ask/tell interface for evolutionary optimization."""

from abc import ABC, abstractmethod


class NEAlgorithm(ABC):
    """Fitness is maximized; one tell must correspond to one ask."""

    @abstractmethod
    def ask(self):
        """Return candidate parameters shaped (population, parameters)."""

    @abstractmethod
    def tell(self, fitness):
        """Update from one finite scalar fitness per candidate."""

    @abstractmethod
    def save_state(self):
        """Return complete state at a generation boundary."""

    @abstractmethod
    def load_state(self, state):
        """Validate and restore state at a generation boundary."""

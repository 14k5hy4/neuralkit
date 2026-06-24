"""Abstract base class for all layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class Layer(ABC):
    """Base class that every layer should inherit from.

    Subclasses must implement forward() and backward(). Layers that have
    learnable parameters should also override the params and grads properties.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        ...

    @property
    def params(self) -> Dict[str, np.ndarray]:
        """Return dict of learnable parameters. Empty by default."""
        return {}

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        """Return gradients for learnable parameters."""
        return {}

    @property
    def name(self) -> str:
        return self.__class__.__name__

"""Flatten layer — reshapes multi-dimensional input to 2D."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy import ndarray

from neuralkit.layers.base import Layer


class Flatten(Layer):
    """Flatten layer that reshapes input to (batch_size, -1).

    Useful when connecting from convolutional or multi-dimensional
    layers to dense layers.
    """

    def __init__(self) -> None:
        self._input_shape: Optional[tuple] = None

    def forward(self, x: ndarray) -> ndarray:
        """Flatten all dimensions except the batch dimension."""
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: ndarray) -> ndarray:
        """Reshape gradient back to original input shape."""
        assert self._input_shape is not None, "forward() must be called first"
        return grad_output.reshape(self._input_shape)

    @property
    def params(self) -> Dict[str, ndarray]:
        return {}

    @property
    def grads(self) -> Dict[str, ndarray]:
        return {}

    @property
    def name(self) -> str:
        return "Flatten"

    def __repr__(self) -> str:
        return "Flatten()"

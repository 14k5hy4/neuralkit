"""Activation function wrappers as Layer subclasses.

These let you use activations as standalone layers in Sequential,
instead of passing them to Dense.

Example::
    model = Sequential([
        Dense(784, 128),
        ReLULayer(),
        Dense(128, 10),
    ])
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy import ndarray

from neuralkit.layers.base import Layer
from neuralkit.activations import ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish


class ReLULayer(Layer):
    """ReLU activation as a layer."""

    def __init__(self) -> None:
        self._act = ReLU()

    def forward(self, x: ndarray) -> ndarray:
        return self._act.forward(x)

    def backward(self, grad_output: ndarray) -> ndarray:
        return self._act.backward(grad_output)

    @property
    def params(self) -> Dict[str, ndarray]:
        return {}

    @property
    def grads(self) -> Dict[str, ndarray]:
        return {}

    def __repr__(self) -> str:
        return "ReLULayer()"


class SigmoidLayer(Layer):
    """Sigmoid activation as a layer."""

    def __init__(self) -> None:
        self._act = Sigmoid()

    def forward(self, x: ndarray) -> ndarray:
        return self._act.forward(x)

    def backward(self, grad_output: ndarray) -> ndarray:
        return self._act.backward(grad_output)

    @property
    def params(self) -> Dict[str, ndarray]:
        return {}

    @property
    def grads(self) -> Dict[str, ndarray]:
        return {}

    def __repr__(self) -> str:
        return "SigmoidLayer()"


class TanhLayer(Layer):
    """Tanh activation as a layer."""

    def __init__(self) -> None:
        self._act = Tanh()

    def forward(self, x: ndarray) -> ndarray:
        return self._act.forward(x)

    def backward(self, grad_output: ndarray) -> ndarray:
        return self._act.backward(grad_output)

    @property
    def params(self) -> Dict[str, ndarray]:
        return {}

    @property
    def grads(self) -> Dict[str, ndarray]:
        return {}

    def __repr__(self) -> str:
        return "TanhLayer()"


class LeakyReLULayer(Layer):
    """LeakyReLU activation as a layer."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope
        self._act = LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: ndarray) -> ndarray:
        return self._act.forward(x)

    def backward(self, grad_output: ndarray) -> ndarray:
        return self._act.backward(grad_output)

    @property
    def params(self) -> Dict[str, ndarray]:
        return {}

    @property
    def grads(self) -> Dict[str, ndarray]:
        return {}

    def __repr__(self) -> str:
        return f"LeakyReLULayer(negative_slope={self.negative_slope})"


class ELULayer(Layer):
    """ELU activation as a layer."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._act = ELU(alpha=alpha)

    def forward(self, x: ndarray) -> ndarray:
        return self._act.forward(x)

    def backward(self, grad_output: ndarray) -> ndarray:
        return self._act.backward(grad_output)

    @property
    def params(self) -> Dict[str, ndarray]:
        return {}

    @property
    def grads(self) -> Dict[str, ndarray]:
        return {}

    def __repr__(self) -> str:
        return f"ELULayer(alpha={self.alpha})"


class SwishLayer(Layer):
    """Swish activation as a layer."""

    def __init__(self) -> None:
        self._act = Swish()

    def forward(self, x: ndarray) -> ndarray:
        return self._act.forward(x)

    def backward(self, grad_output: ndarray) -> ndarray:
        return self._act.backward(grad_output)

    @property
    def params(self) -> Dict[str, ndarray]:
        return {}

    @property
    def grads(self) -> Dict[str, ndarray]:
        return {}

    def __repr__(self) -> str:
        return "SwishLayer()"

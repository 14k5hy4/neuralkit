"""Fully-connected (dense) layer implementation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy import ndarray

from neuralkit.layers.base import Layer


class Dense(Layer):
    """A standard fully-connected layer.

    Computes y = x @ W + b during the forward pass and propagates
    gradients through during backward.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output neurons.
    activation : object, optional
        Activation function instance with forward()/backward() methods.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation=None,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        # He initialization — good default for ReLU-family networks
        scale = np.sqrt(2.0 / input_dim)
        self.W: ndarray = np.random.randn(input_dim, output_dim) * scale
        self.b: ndarray = np.zeros((1, output_dim))

        # cached for backward pass
        self._input: Optional[ndarray] = None
        self._grad_W: Optional[ndarray] = None
        self._grad_b: Optional[ndarray] = None

    def forward(self, x: ndarray) -> ndarray:
        """Forward pass: linear transform then optional activation.

        Args:
            x: Input of shape (batch_size, input_dim).

        Returns:
            Output of shape (batch_size, output_dim).
        """
        self._input = x
        out = x @ self.W + self.b

        if self.activation is not None:
            out = self.activation.forward(out)
        return out

    def backward(self, grad_output: ndarray) -> ndarray:
        """Backward pass — computes parameter gradients and returns input gradient."""
        if self.activation is not None:
            grad_output = self.activation.backward(grad_output)

        batch_size = self._input.shape[0]
        self._grad_W = self._input.T @ grad_output / batch_size
        self._grad_b = np.mean(grad_output, axis=0, keepdims=True)

        # FIXME: handle case where input might have been 1-d (no batch dim)
        grad_input = grad_output @ self.W.T
        return grad_input

    @property
    def params(self) -> Dict[str, ndarray]:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> Dict[str, ndarray]:
        return {"W": self._grad_W, "b": self._grad_b}

    def __repr__(self) -> str:
        act_name = self.activation.__class__.__name__ if self.activation else "None"
        return f"Dense({self.input_dim}, {self.output_dim}, activation={act_name})"

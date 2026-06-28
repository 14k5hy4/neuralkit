"""Dropout regularization layer (inverted dropout)."""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from neuralkit.layers.base import Layer


class Dropout(Layer):
    """Randomly zeroes elements during training to prevent overfitting.

    Uses inverted dropout — scales activations by 1/(1-rate) during
    training so that no scaling is needed at test time.

    Args:
        rate: Fraction of inputs to drop. Default 0.5.
    """

    def __init__(self, rate: float = 0.5) -> None:
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"dropout rate must be in [0, 1), got {rate}")
        self.rate = rate
        self.training = True
        self._mask: ndarray | None = None

    def forward(self, x: ndarray) -> ndarray:
        if not self.training or self.rate == 0.0:
            return x

        self._mask = (np.random.rand(*x.shape) > self.rate).astype(x.dtype)
        # scale by 1/(1-rate) so expected value stays the same
        return x * self._mask / (1.0 - self.rate)

    def backward(self, grad_output: ndarray) -> ndarray:
        """Same mask applied to gradients."""
        if not self.training or self.rate == 0.0:
            return grad_output
        return grad_output * self._mask / (1.0 - self.rate)

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    # FIXME: should we reset the mask between forward calls?

    def __repr__(self) -> str:
        return f"Dropout(rate={self.rate})"

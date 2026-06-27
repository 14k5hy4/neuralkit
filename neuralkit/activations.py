"""Activation functions for neural network layers.

Each activation is implemented as a class with forward() and backward()
methods so it can be slotted into a layer pipeline later.
"""

from __future__ import annotations

import numpy as np


class Sigmoid:
    """Logistic sigmoid activation: σ(x) = 1 / (1 + exp(-x)).

    Caches the output of forward() so backward() can reuse it
    without recomputing.
    """

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid(x).

        Args:
            x: Input array of any shape.

        Returns:
            Array of same shape with values in (0, 1).
        """
        # clip to ~88 since exp(88) is near float64 max
        x_safe = np.clip(x, -88, 88)
        self._out = 1.0 / (1.0 + np.exp(-x_safe))
        return self._out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradient: dL/dx = dL/dy * σ(x) * (1 - σ(x)).

        Args:
            grad_output: Upstream gradient, same shape as forward output.

        Returns:
            Gradient with respect to the input.
        """
        assert self._out is not None, "forward() must be called before backward()"
        return grad_output * self._out * (1.0 - self._out)


class ReLU:
    """Rectified Linear Unit: f(x) = max(0, x)."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute ReLU(x).

        Args:
            x: Input array of any shape.

        Returns:
            Array with negative values zeroed out.
        """
        self._mask = (x > 0).astype(x.dtype)
        return x * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Gradient is 1 where input was positive, 0 otherwise.

        Args:
            grad_output: Upstream gradient.

        Returns:
            Gradient with respect to the input.
        """
        assert self._mask is not None, "forward() must be called before backward()"
        return grad_output * self._mask


class Tanh:
    """Hyperbolic tangent activation."""

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute tanh(x).

        Args:
            x: Input array of any shape.

        Returns:
            Array with values in (-1, 1).
        """
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Gradient: dL/dx = dL/dy * (1 - tanh²(x)).

        Args:
            grad_output: Upstream gradient.

        Returns:
            Gradient with respect to the input.
        """
        assert self._out is not None, "forward() must be called before backward()"
        return grad_output * (1.0 - self._out ** 2)


class Softmax:
    """Softmax activation: turns logits into a probability distribution.

    Uses the log-sum-exp trick for numerical stability (subtracts the
    row-wise max before exponentiating).
    """

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax along the last axis."""
        # subtract max for numerical stability
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        self._out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self._out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Softmax backward pass (Jacobian-vector product).

        This is the full Jacobian version. For classification with
        cross-entropy, prefer SoftmaxCrossEntropy which is simpler and
        more stable.
        """
        assert self._out is not None, "forward() must be called before backward()"
        s = self._out
        # for each sample: dout * (diag(s) - s @ s.T)
        # vectorized version
        return s * (grad_output - np.sum(grad_output * s, axis=-1, keepdims=True))

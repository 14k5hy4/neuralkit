"""Loss functions with forward and backward passes.

Each loss computes a scalar loss value in forward() and the gradient
of the loss with respect to predictions in backward().
"""

from __future__ import annotations

import numpy as np


class MSELoss:
    """Mean Squared Error loss.

    L = (1/n) * Σ (y_pred - y_true)²
    """

    def __init__(self) -> None:
        self._diff: np.ndarray | None = None
        self._n: int = 0

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute MSE loss.

        Args:
            y_pred: Predicted values, shape (n,) or (n, d).
            y_true: Target values, same shape as y_pred.

        Returns:
            Scalar loss value.
        """
        self._diff = y_pred - y_true
        self._n = y_pred.size
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        """Compute gradient of MSE w.r.t. predictions.

        dL/dy_pred = (2/n) * (y_pred - y_true)

        Returns:
            Gradient array, same shape as y_pred.
        """
        assert self._diff is not None, "forward() must be called before backward()"
        return (2.0 / self._n) * self._diff


class CrossEntropyLoss:
    """Binary / multi-class cross-entropy loss.

    Expects predictions to already be probabilities (e.g. after softmax
    or sigmoid). We clip predictions for numerical stability.

    For multi-class (one-hot targets):
        L = -(1/n) * Σ y_true * log(y_pred)

    For binary (scalar targets):
        L = -(1/n) * Σ [y*log(p) + (1-y)*log(1-p)]
    """

    _EPS = 1e-12  # small constant to avoid log(0)

    def __init__(self) -> None:
        self._y_pred: np.ndarray | None = None
        self._y_true: np.ndarray | None = None
        self._n: int = 0

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss.

        Args:
            y_pred: Predicted probabilities, shape (n,) or (n, c).
            y_true: Ground-truth labels (one-hot or binary), same shape.

        Returns:
            Scalar loss value.
        """
        clipped = np.clip(y_pred, self._EPS, 1.0 - self._EPS)
        self._y_pred = clipped
        self._y_true = y_true
        self._n = y_pred.shape[0]

        # General form that works for both binary and multi-class
        loss = -np.sum(
            y_true * np.log(clipped)
            + (1.0 - y_true) * np.log(1.0 - clipped)
        )
        return float(loss / self._n)

    def backward(self) -> np.ndarray:
        """Compute gradient of cross-entropy w.r.t. predicted probabilities.

        dL/dy_pred = -(1/n) * (y_true/y_pred - (1-y_true)/(1-y_pred))

        Returns:
            Gradient array, same shape as y_pred.
        """
        assert self._y_pred is not None, "forward() must be called before backward()"
        grad = -(self._y_true / self._y_pred) + (1.0 - self._y_true) / (1.0 - self._y_pred)
        return grad / self._n


# TODO: implement Softmax + CrossEntropy fused version for numerical stability

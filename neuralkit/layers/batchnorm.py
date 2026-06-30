"""Batch normalization layer."""

from __future__ import annotations

from typing import Dict
import numpy as np
from numpy import ndarray

from neuralkit.layers.base import Layer


class BatchNorm(Layer):
    """Batch normalization (Ioffe & Szegedy, 2015).

    Normalizes activations to zero mean and unit variance per feature,
    then applies learnable scale (gamma) and shift (beta). Tracks
    running statistics for use during inference.

    Args:
        num_features: Number of input features (channels).
        eps: Small value added to denominator for stability.
        momentum: Factor for running mean/var update. The running stats
            are updated as: running = (1 - momentum) * running + momentum * batch.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # learnable parameters
        self.gamma: ndarray = np.ones((1, num_features))
        self.beta: ndarray = np.zeros((1, num_features))

        # running stats for inference
        self.running_mean: ndarray = np.zeros((1, num_features))
        self.running_var: ndarray = np.ones((1, num_features))

        # cached values for backward
        self._x_norm: ndarray | None = None
        self._std: ndarray | None = None
        self._x_centered: ndarray | None = None
        self._batch_size: int = 0

        self._grad_gamma: ndarray | None = None
        self._grad_beta: ndarray | None = None

    def forward(self, x: ndarray) -> ndarray:
        if self.training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            # update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            self._x_centered = x - mean
            self._std = np.sqrt(var + self.eps)
            self._x_norm = self._x_centered / self._std
            self._batch_size = x.shape[0]
        else:
            # use running stats at inference time
            self._x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self._x_norm + self.beta

    def backward(self, grad_output: ndarray) -> ndarray:
        """Backprop through batch norm — follows the derivation from the paper."""
        n = self._batch_size

        self._grad_gamma = np.sum(grad_output * self._x_norm, axis=0, keepdims=True)
        self._grad_beta = np.sum(grad_output, axis=0, keepdims=True)

        # gradient w.r.t normalized input
        dx_norm = grad_output * self.gamma

        # TODO: optimize this for large batch sizes
        dvar = np.sum(dx_norm * self._x_centered * -0.5 * self._std**(-3), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -1.0 / self._std, axis=0, keepdims=True)

        dx = dx_norm / self._std + dvar * 2.0 * self._x_centered / n + dmean / n
        return dx

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    @property
    def params(self) -> Dict[str, ndarray]:
        return {"gamma": self.gamma, "beta": self.beta}

    @property
    def grads(self) -> Dict[str, ndarray]:
        return {"gamma": self._grad_gamma, "beta": self._grad_beta}

    def __repr__(self) -> str:
        return f"BatchNorm({self.num_features})"

"""Stochastic Gradient Descent optimizer with momentum and weight decay."""

from __future__ import annotations

from typing import List
import numpy as np


class SGD:
    """SGD with optional momentum and weight decay (L2 regularization).

    Parameters
    ----------
    lr : float
        Learning rate.
    momentum : float
        Momentum factor. 0.0 means vanilla SGD.
    weight_decay : float
        L2 regularization coefficient.
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocities: dict = {}

    def step(self, layers: List) -> None:
        """Update parameters for all layers that have gradients."""
        for i, layer in enumerate(layers):
            params = layer.params
            grads = layer.grads
            if not params:
                continue

            for key in params:
                if grads.get(key) is None:
                    continue

                grad = grads[key]

                # L2 regularization adds param * weight_decay to gradient
                if self.weight_decay != 0.0:
                    grad = grad + self.weight_decay * params[key]

                # velocity tracking per (layer_index, param_name)
                vel_key = (i, key)
                if self.momentum != 0.0:
                    if vel_key not in self._velocities:
                        self._velocities[vel_key] = np.zeros_like(params[key])
                    v = self._velocities[vel_key]
                    v = self.momentum * v + grad
                    self._velocities[vel_key] = v
                    params[key] -= self.lr * v
                else:
                    params[key] -= self.lr * grad

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.momentum})"

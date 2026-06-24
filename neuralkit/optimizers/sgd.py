"""Stochastic Gradient Descent optimizer."""

from __future__ import annotations

from typing import Dict, List
import numpy as np


class SGD:
    """Basic SGD optimizer.

    Updates each parameter by subtracting lr * gradient.

    Args:
        lr: Learning rate (step size). Default 0.01.
    """

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, layers: List) -> None:
        """Update parameters for all layers that have gradients.

        Parameters
        ----------
        layers : list
            List of Layer objects. Each layer exposes params and grads dicts.
        """
        for layer in layers:
            params = layer.params
            grads = layer.grads
            if not params:
                continue
            for key in params:
                if grads.get(key) is not None:
                    params[key] -= self.lr * grads[key]

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr})"

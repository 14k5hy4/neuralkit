"""RMSProp optimizer (Hinton, 2012).

Divides the learning rate by a running average of the magnitudes
of recent gradients. Prevents the learning rate from becoming
too large or too small.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class RMSProp:
    """RMSProp optimizer following Hinton's original formulation.

    Args:
        lr: Learning rate. Default 0.01.
        alpha: Smoothing constant (decay rate for running average).
            Default 0.99.
        eps: Small constant for numerical stability.
        centered: If True, compute the centered RMSProp (divides by
            an estimate of the variance instead of the uncentered
            second moment).
        clip_value: If set, clip gradient values to [-clip_value, clip_value].
        clip_norm: If set, clip gradient norm to this max.
    """

    def __init__(
        self,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        centered: bool = False,
        clip_value: Optional[float] = None,
        clip_norm: Optional[float] = None,
    ) -> None:
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.centered = centered
        self.clip_value = clip_value
        self.clip_norm = clip_norm

        self._v: dict = {}     # running avg of squared gradients
        self._g_avg: dict = {} # running avg of gradients (centered only)

    def _clip_gradient(self, grad: np.ndarray) -> np.ndarray:
        if self.clip_value is not None:
            grad = np.clip(grad, -self.clip_value, self.clip_value)
        if self.clip_norm is not None:
            norm = np.linalg.norm(grad)
            if norm > self.clip_norm:
                grad = grad * (self.clip_norm / norm)
        return grad

    def step(self, layers: List) -> None:
        """Update all layer parameters."""
        for i, layer in enumerate(layers):
            params = layer.params
            grads = layer.grads
            if not params:
                continue

            for key in params:
                if grads.get(key) is None:
                    continue

                g = self._clip_gradient(grads[key])
                pid = (i, key)

                # init running averages
                if pid not in self._v:
                    self._v[pid] = np.zeros_like(params[key])
                    if self.centered:
                        self._g_avg[pid] = np.zeros_like(params[key])

                # update running average of squared gradient
                self._v[pid] = self.alpha * self._v[pid] + (1 - self.alpha) * (g ** 2)

                if self.centered:
                    # running average of gradient
                    self._g_avg[pid] = self.alpha * self._g_avg[pid] + (1 - self.alpha) * g
                    # centered second moment: E[g²] - E[g]²
                    denom = self._v[pid] - self._g_avg[pid] ** 2
                    params[key] -= self.lr * g / (np.sqrt(denom) + self.eps)
                else:
                    params[key] -= self.lr * g / (np.sqrt(self._v[pid]) + self.eps)

    def __repr__(self) -> str:
        return f"RMSProp(lr={self.lr}, alpha={self.alpha}, centered={self.centered})"

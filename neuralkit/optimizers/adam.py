"""Adam optimizer (Kingma & Ba, 2014).

Adaptive learning rate optimizer using first and second moment estimates
with bias correction.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class Adam:
    """Adam optimizer following the original paper.

    Args:
        lr: Learning rate (alpha in the paper). Default 0.001.
        beta1: Exponential decay rate for first moment. Default 0.9.
        beta2: Exponential decay rate for second moment. Default 0.999.
        eps: Small constant for numerical stability.
        clip_value: If set, clip gradient values to [-clip_value, clip_value].
        clip_norm: If set, clip gradient norm to this max value.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        clip_value: Optional[float] = None,
        clip_norm: Optional[float] = None,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip_value = clip_value
        self.clip_norm = clip_norm

        self._m: dict = {}  # first moment estimates
        self._v: dict = {}  # second moment estimates
        self._t: int = 0    # timestep

    def _clip_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Apply gradient clipping if configured."""
        if self.clip_value is not None:
            grad = np.clip(grad, -self.clip_value, self.clip_value)
        if self.clip_norm is not None:
            norm = np.linalg.norm(grad)
            if norm > self.clip_norm:
                grad = grad * (self.clip_norm / norm)
        return grad

    def step(self, layers: List) -> None:
        """Update all layer parameters using Adam rule."""
        self._t += 1

        for i, layer in enumerate(layers):
            params = layer.params
            grads = layer.grads
            if not params:
                continue

            for key in params:
                if grads.get(key) is None:
                    continue

                g = self._clip_gradient(grads[key])
                param_id = (i, key)

                # init moment estimates on first use
                if param_id not in self._m:
                    self._m[param_id] = np.zeros_like(params[key])
                    self._v[param_id] = np.zeros_like(params[key])

                # update biased moments
                self._m[param_id] = self.beta1 * self._m[param_id] + (1 - self.beta1) * g
                self._v[param_id] = self.beta2 * self._v[param_id] + (1 - self.beta2) * (g ** 2)

                # bias correction
                m_hat = self._m[param_id] / (1 - self.beta1 ** self._t)
                v_hat = self._v[param_id] / (1 - self.beta2 ** self._t)

                params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    # TODO: add amsgrad variant

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"

"""Learning rate schedulers.

Each scheduler wraps an optimizer and adjusts its lr attribute
based on the current epoch or metric value.
"""

from __future__ import annotations

import math
from typing import Optional


class StepLR:
    """Decay learning rate by a factor every step_size epochs.

    Args:
        optimizer: Optimizer instance with an lr attribute.
        step_size: Decay lr every this many epochs.
        gamma: Multiplicative factor. Default 0.1.
    """

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1) -> None:
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self._initial_lr = optimizer.lr
        self._epoch = 0

    def step(self) -> None:
        self._epoch += 1
        if self._epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    @property
    def lr(self) -> float:
        return self.optimizer.lr


class ExponentialLR:
    """Exponentially decay the learning rate each epoch.

    new_lr = initial_lr * gamma^epoch
    """

    def __init__(self, optimizer, gamma: float = 0.95) -> None:
        self.optimizer = optimizer
        self.gamma = gamma
        self._initial_lr = optimizer.lr
        self._epoch = 0

    def step(self) -> None:
        self._epoch += 1
        self.optimizer.lr = self._initial_lr * (self.gamma ** self._epoch)

    @property
    def lr(self) -> float:
        return self.optimizer.lr


class CosineAnnealingLR:
    """Cosine annealing schedule.

    Decays lr from initial value to eta_min following a cosine curve
    over T_max epochs, then restarts.

    Args:
        optimizer: Optimizer with lr attribute.
        T_max: Maximum number of epochs for one cycle.
        eta_min: Minimum learning rate. Default 0.
    """

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0) -> None:
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self._initial_lr = optimizer.lr
        self._epoch = 0

    def step(self) -> None:
        self._epoch += 1
        self.optimizer.lr = self.eta_min + (self._initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * self._epoch / self.T_max)
        ) / 2

    @property
    def lr(self) -> float:
        return self.optimizer.lr


class ReduceLROnPlateau:
    """Reduce lr when a metric has stopped improving.

    Args:
        optimizer: Optimizer with lr attribute.
        patience: Number of epochs with no improvement before reducing.
        factor: Factor to multiply lr by. Default 0.1.
        min_lr: Lower bound on lr.
        mode: 'min' for metrics that should decrease, 'max' for increase.
    """

    def __init__(
        self,
        optimizer,
        patience: int = 10,
        factor: float = 0.1,
        min_lr: float = 1e-6,
        mode: str = "min",
    ) -> None:
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode

        self._best: Optional[float] = None
        self._wait = 0

    def step(self, metric_value: float) -> None:
        """Call with the current metric value after each epoch."""
        if self._best is None:
            self._best = metric_value
            return

        improved = (
            metric_value < self._best if self.mode == "min"
            else metric_value > self._best
        )

        if improved:
            self._best = metric_value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self._wait = 0

    @property
    def lr(self) -> float:
        return self.optimizer.lr

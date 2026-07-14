"""Callback system for the training loop.

Callbacks receive training state at various points and can
modify behaviour (e.g. stop training early).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np


class Callback:
    """Base class for all callbacks.

    Override any of the hook methods to add custom logic.
    """

    def on_train_begin(self, logs: Dict[str, Any] = None) -> None:
        pass

    def on_train_end(self, logs: Dict[str, Any] = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        pass


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric name to watch (e.g. 'val_loss').
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        restore_best_weights: If True, restore model weights from
            the epoch with the best value of the monitored metric.
        mode: 'min' if lower is better, 'max' if higher is better.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode

        self.best: Optional[float] = None
        self.best_epoch: int = 0
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.stop_training: bool = False
        self._best_weights: Optional[List] = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.best is None:
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self._save_weights(logs.get("_model"))
            return

        improved = (
            current < self.best - self.min_delta if self.mode == "min"
            else current > self.best + self.min_delta
        )

        if improved:
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self._save_weights(logs.get("_model"))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                if self.restore_best_weights and self._best_weights is not None:
                    self._restore_weights(logs.get("_model"))
                    print(f"Restoring model weights from epoch {self.best_epoch}")
                print(f"Early stopping at epoch {epoch}")

    def _save_weights(self, model) -> None:
        if model is None:
            return
        self._best_weights = []
        for layer in model.layers:
            params = layer.params
            if params:
                self._best_weights.append({k: v.copy() for k, v in params.items()})
            else:
                self._best_weights.append(None)

    def _restore_weights(self, model) -> None:
        if model is None or self._best_weights is None:
            return
        for layer, saved in zip(model.layers, self._best_weights):
            if saved is not None:
                params = layer.params
                for k, v in saved.items():
                    params[k][:] = v


class ModelCheckpoint(Callback):
    """Save the model when a monitored metric improves.

    Args:
        filepath: Directory path to save the model to.
        monitor: Metric to monitor.
        mode: 'min' or 'max'.
    """

    def __init__(
        self,
        filepath: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best: Optional[float] = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        model = logs.get("_model")
        if model is None:
            return

        if self.best is None:
            self.best = current
            if hasattr(model, 'save'):
                model.save(self.filepath)
                print(f"Checkpoint saved (epoch {epoch}, {self.monitor}={current:.6f})")
            return

        improved = (
            current < self.best if self.mode == "min"
            else current > self.best
        )
        if improved:
            self.best = current
            if hasattr(model, 'save'):
                model.save(self.filepath)
                print(f"Checkpoint saved (epoch {epoch}, {self.monitor}={current:.6f})")


class LearningRateLogger(Callback):
    """Log the learning rate at each epoch."""

    def __init__(self) -> None:
        self.lr_history: List[float] = []

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        logs = logs or {}
        optimizer = logs.get("_optimizer")
        if optimizer is not None and hasattr(optimizer, 'lr'):
            self.lr_history.append(optimizer.lr)

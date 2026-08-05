"""Training loop for neuralkit models."""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from neuralkit.data.loader import DataLoader, ArrayDataset
from neuralkit.exceptions import ConfigurationError, ShapeMismatchError

import warnings


def _progress_bar(current: int, total: int, width: int = 30, metrics: str = "") -> str:
    """Build a text progress bar string.

    Args:
        current: Current step (1-indexed).
        total: Total steps.
        width: Character width of the bar.
        metrics: Metrics string to append.

    Returns:
        Formatted progress bar like: [████████░░░░░░░░] 50% — loss: 0.1234
    """
    frac = current / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{frac * 100:5.1f}%"
    line = f"\r  [{bar}] {pct} — epoch {current}/{total}"
    if metrics:
        line += f" — {metrics}"
    return line


class Trainer:
    """Handles the training loop for a Sequential model.

    Args:
        model: A Sequential model instance.
        optimizer: An optimizer instance (e.g. SGD, Adam).
        loss_fn: Loss function with forward() and backward() methods.
        metrics: Optional list of metric functions. Each should take
            (y_true, y_pred) and return a float.
    """

    def __init__(self, model, optimizer, loss_fn, metrics: Optional[List] = None, callbacks: Optional[List] = None, regularizer=None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        self.regularizer = regularizer

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        val_data: Optional[tuple] = None,
        verbose: Union[bool, int] = 1,
        callbacks: Optional[List] = None,
    ) -> Dict[str, List[float]]:
        """Train the model on the given data.

        Parameters
        ----------
        x : np.ndarray
            Training inputs, shape (n_samples, n_features).
        y : np.ndarray
            Training targets.
        epochs : int
            Number of passes through the dataset.
        batch_size : int, optional
            If None, use the full dataset each step.
        val_data : tuple, optional
            (x_val, y_val) for validation tracking.
        verbose : int or bool
            0 or False = silent, 1 or True = progress bar,
            2 = detailed per-epoch logging.
        callbacks : list, optional
            Additional callbacks for this training run.

        Returns
        -------
        dict
            Training history with loss, val_loss, and metric values.
        """
        # normalize verbose: True -> 1, False -> 0
        if isinstance(verbose, bool):
            verbose = 1 if verbose else 0

        history: Dict[str, List[float]] = {"loss": []}

        # validate inputs
        if x.shape[0] != y.shape[0]:
            raise ShapeMismatchError(
                expected=f"{x.shape[0]} samples",
                got=f"{y.shape[0]} samples",
                context="x and y must have the same number of samples",
            )
        if hasattr(self.optimizer, 'lr') and self.optimizer.lr > 10.0:
            warnings.warn(
                f"Learning rate {self.optimizer.lr} is very large. "
                f"This may cause training instability.",
                stacklevel=2,
            )
        if val_data is not None and (not isinstance(val_data, tuple) or len(val_data) != 2):
            raise ConfigurationError("val_data must be a tuple of (x_val, y_val)")

        if val_data is not None:
            history["val_loss"] = []

        # init metric history
        for m in self.metrics:
            name = m.__name__ if hasattr(m, '__name__') else str(m)
            history[name] = []
            if val_data is not None:
                history[f"val_{name}"] = []

        if batch_size is not None:
            loader = DataLoader(
                ArrayDataset(x, y),
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            loader = None

        # merge callbacks from constructor and fit() call
        all_callbacks = self.callbacks + (callbacks or [])

        # notify: train begin
        for cb in all_callbacks:
            cb.on_train_begin({"epochs": epochs})

        t_start = time.time()

        for epoch in range(1, epochs + 1):
            # notify: epoch begin
            for cb in all_callbacks:
                cb.on_epoch_begin(epoch)

            if hasattr(self.model, 'train'):
                self.model.train()

            if loader is not None:
                batch_losses = []
                for x_batch, y_batch in loader:
                    loss = self._train_step(x_batch, y_batch)
                    batch_losses.append(loss)
                epoch_loss = float(np.mean(batch_losses))
            else:
                epoch_loss = self._train_step(x, y)

            history["loss"].append(epoch_loss)

            # compute train metrics on full data
            if self.metrics:
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                train_pred = self.model.forward(x)
                for m in self.metrics:
                    name = m.__name__ if hasattr(m, '__name__') else str(m)
                    val = m(y, self._to_labels(train_pred))
                    history[name].append(float(val))

            # validation
            if val_data is not None:
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                x_val, y_val = val_data
                val_pred = self.model.forward(x_val)
                val_loss = self.loss_fn.forward(val_pred, y_val)
                history["val_loss"].append(float(val_loss))

                for m in self.metrics:
                    name = m.__name__ if hasattr(m, '__name__') else str(m)
                    val = m(y_val, self._to_labels(val_pred))
                    history[f"val_{name}"].append(float(val))

            # --- logging ---
            metrics_str = self._format_metrics(epoch_loss, history, val_data)

            if verbose == 1:
                # progress bar mode
                bar = _progress_bar(epoch, epochs, metrics=metrics_str)
                sys.stdout.write(bar)
                sys.stdout.flush()
                if epoch == epochs:
                    elapsed = time.time() - t_start
                    sys.stdout.write(f"  [{elapsed:.1f}s]\n")
                    sys.stdout.flush()

            elif verbose >= 2:
                # detailed mode — print every epoch
                elapsed = time.time() - t_start
                print(f"epoch {epoch}/{epochs} — {metrics_str} [{elapsed:.1f}s]")

            # notify: epoch end
            epoch_logs = dict(history)
            epoch_logs["_model"] = self.model
            epoch_logs["_optimizer"] = self.optimizer
            for cb in all_callbacks:
                cb.on_epoch_end(epoch, epoch_logs)

            # check if any callback requested stop
            if any(getattr(cb, 'stop_training', False) for cb in all_callbacks):
                if verbose == 1:
                    # finish the progress bar line
                    elapsed = time.time() - t_start
                    sys.stdout.write(f"  [stopped @ epoch {epoch}, {elapsed:.1f}s]\n")
                    sys.stdout.flush()
                break

        # notify: train end
        for cb in all_callbacks:
            cb.on_train_end(history)

        return history

    def _format_metrics(self, epoch_loss: float, history: Dict, val_data) -> str:
        """Format metrics into a compact string for display."""
        parts = [f"loss: {epoch_loss:.6f}"]
        if val_data is not None:
            parts.append(f"val_loss: {history['val_loss'][-1]:.6f}")
        for m in self.metrics:
            name = m.__name__ if hasattr(m, '__name__') else str(m)
            parts.append(f"{name}: {history[name][-1]:.4f}")
            if val_data is not None and f"val_{name}" in history:
                parts.append(f"val_{name}: {history[f'val_{name}'][-1]:.4f}")
        return " — ".join(parts)

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model on test data and return metric results.

        Returns dict with 'loss' and each metric name.
        """
        if hasattr(self.model, 'eval'):
            self.model.eval()

        pred = self.model.forward(x)
        loss = self.loss_fn.forward(pred, y)

        results: Dict[str, float] = {"loss": float(loss)}
        for m in self.metrics:
            name = m.__name__ if hasattr(m, '__name__') else str(m)
            results[name] = float(m(y, self._to_labels(pred)))

        return results

    def _train_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Single forward + backward + update step."""
        predictions = self.model.forward(x_batch)
        loss = self.loss_fn.forward(predictions, y_batch)

        # add regularization penalty to loss
        if self.regularizer is not None:
            loss = loss + self.regularizer.penalty(self.model.layers)

        grad = self.loss_fn.backward()
        self.model.backward(grad)

        # add regularizer gradients to param gradients
        if self.regularizer is not None:
            for layer in self.model.layers:
                grads = layer.grads
                for key, param in layer.params.items():
                    if grads.get(key) is not None:
                        grads[key] = grads[key] + self.regularizer.grad(param)

        self.optimizer.step(self.model.layers)
        return loss

    @staticmethod
    def _to_labels(pred: np.ndarray) -> np.ndarray:
        """Convert probabilities to class labels (argmax for multi-class)."""
        if pred.ndim == 2 and pred.shape[1] > 1:
            return np.argmax(pred, axis=1)
        return (pred > 0.5).astype(int).ravel()

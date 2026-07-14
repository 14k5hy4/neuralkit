"""Training loop for neuralkit models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from neuralkit.data.loader import DataLoader, ArrayDataset


class Trainer:
    """Handles the training loop for a Sequential model.

    Args:
        model: A Sequential model instance.
        optimizer: An optimizer instance (e.g. SGD, Adam).
        loss_fn: Loss function with forward() and backward() methods.
        metrics: Optional list of metric functions. Each should take
            (y_true, y_pred) and return a float.
    """

    def __init__(self, model, optimizer, loss_fn, metrics: Optional[List] = None, callbacks: Optional[List] = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.callbacks = callbacks or []

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        val_data: Optional[tuple] = None,
        verbose: bool = True,
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
        verbose : bool
            Whether to print loss each epoch.

        Returns
        -------
        dict
            Training history with loss, val_loss, and metric values.
        """
        history: Dict[str, List[float]] = {"loss": []}
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

            # logging
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                msg = f"epoch {epoch}/{epochs} — loss: {epoch_loss:.6f}"
                if val_data is not None:
                    msg += f" — val_loss: {history['val_loss'][-1]:.6f}"
                for m in self.metrics:
                    name = m.__name__ if hasattr(m, '__name__') else str(m)
                    msg += f" — {name}: {history[name][-1]:.4f}"
                print(msg)

            # notify: epoch end — pass internal refs so callbacks can act
            epoch_logs = dict(history)
            epoch_logs["_model"] = self.model
            epoch_logs["_optimizer"] = self.optimizer
            for cb in all_callbacks:
                cb.on_epoch_end(epoch, epoch_logs)

            # check if any callback requested stop
            if any(getattr(cb, 'stop_training', False) for cb in all_callbacks):
                break

        # notify: train end
        for cb in all_callbacks:
            cb.on_train_end(history)

        return history

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

        grad = self.loss_fn.backward()
        self.model.backward(grad)

        self.optimizer.step(self.model.layers)
        return loss

    @staticmethod
    def _to_labels(pred: np.ndarray) -> np.ndarray:
        """Convert probabilities to class labels (argmax for multi-class)."""
        if pred.ndim == 2 and pred.shape[1] > 1:
            return np.argmax(pred, axis=1)
        return (pred > 0.5).astype(int).ravel()

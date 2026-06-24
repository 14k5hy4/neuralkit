"""Training loop for neuralkit models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class Trainer:
    """Handles the training loop for a Sequential model.

    Args:
        model: A Sequential model instance.
        optimizer: An optimizer instance (e.g. SGD).
        loss_fn: Loss function with forward() and backward() methods.
    """

    def __init__(self, model, optimizer, loss_fn) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        verbose: bool = True,
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
            If None, use the full dataset each step (batch gradient descent).
        verbose : bool
            Whether to print loss each epoch.

        Returns
        -------
        dict
            Training history with 'loss' key.
        """
        n_samples = x.shape[0]
        history: Dict[str, List[float]] = {"loss": []}

        for epoch in range(1, epochs + 1):
            if batch_size is None:
                # full-batch
                epoch_loss = self._train_step(x, y)
            else:
                # mini-batch training
                indices = np.random.permutation(n_samples)
                batch_losses = []

                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch_idx = indices[start:end]
                    x_batch = x[batch_idx]
                    y_batch = y[batch_idx]
                    loss = self._train_step(x_batch, y_batch)
                    batch_losses.append(loss)

                epoch_loss = float(np.mean(batch_losses))

            history["loss"].append(epoch_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                print(f"epoch {epoch}/{epochs} — loss: {epoch_loss:.6f}")

        return history

    def _train_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Single forward + backward + update step."""
        predictions = self.model.forward(x_batch)
        loss = self.loss_fn.forward(predictions, y_batch)

        grad = self.loss_fn.backward()
        self.model.backward(grad)

        # update weights
        self.optimizer.step(self.model.layers)
        return loss

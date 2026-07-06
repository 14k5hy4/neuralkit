"""Training loop for neuralkit models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from neuralkit.data.loader import DataLoader, ArrayDataset


class Trainer:
    """Handles the training loop for a Sequential model.

    Args:
        model: A Sequential model instance.
        optimizer: An optimizer instance (e.g. SGD, Adam).
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
        val_data: Optional[tuple] = None,
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
        val_data : tuple, optional
            (x_val, y_val) for validation loss tracking.
        verbose : bool
            Whether to print loss each epoch.

        Returns
        -------
        dict
            Training history with 'loss' and optionally 'val_loss' keys.
        """
        history: Dict[str, List[float]] = {"loss": []}
        if val_data is not None:
            history["val_loss"] = []

        # set up data loader for mini-batch or full-batch
        if batch_size is not None:
            loader = DataLoader(
                ArrayDataset(x, y),
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            loader = None

        for epoch in range(1, epochs + 1):
            # training
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

            # validation
            if val_data is not None:
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                x_val, y_val = val_data
                val_pred = self.model.forward(x_val)
                val_loss = self.loss_fn.forward(val_pred, y_val)
                history["val_loss"].append(float(val_loss))

            # logging
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                msg = f"epoch {epoch}/{epochs} — loss: {epoch_loss:.6f}"
                if val_data is not None:
                    msg += f" — val_loss: {history['val_loss'][-1]:.6f}"
                print(msg)

        return history

    def _train_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Single forward + backward + update step."""
        predictions = self.model.forward(x_batch)
        loss = self.loss_fn.forward(predictions, y_batch)

        grad = self.loss_fn.backward()
        self.model.backward(grad)

        self.optimizer.step(self.model.layers)
        return loss

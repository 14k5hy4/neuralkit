"""Regression example — fitting a sine wave with noise.

Shows how to use neuralkit for regression tasks with MSE loss
and evaluation using regression metrics.
"""

import os
import numpy as np

from neuralkit.model import Sequential
from neuralkit.layers import Dense
from neuralkit.activations import ReLU, Tanh
from neuralkit.losses import MSELoss
from neuralkit.optimizers import Adam
from neuralkit.trainer import Trainer
from neuralkit.data.splits import train_test_split
from neuralkit.metrics.regression import mse, rmse, r2_score
from neuralkit.utils.visualization import plot_training_history


def generate_sine_data(n_samples: int = 200, noise: float = 0.1, seed: int = 42):
    """Generate noisy sine wave data."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, (n_samples, 1))
    y = np.sin(X) + rng.normal(0, noise, X.shape)
    return X, y


def print_predictions(x, y_true, y_pred, n=20):
    """Simple text-based comparison of predictions vs actual."""
    print(f"\n{'x':>8} {'actual':>10} {'predicted':>10} {'error':>10}")
    print("-" * 42)

    order = np.argsort(x.ravel())
    step = max(1, len(order) // n)

    for i in order[::step]:
        xi = x.ravel()[i]
        yi = y_true.ravel()[i]
        pi = y_pred.ravel()[i]
        err = abs(yi - pi)
        print(f"{xi:>8.3f} {yi:>10.4f} {pi:>10.4f} {err:>10.4f}")


def main():
    np.random.seed(42)

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # generate data
    X, y = generate_sine_data(n_samples=300, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_seed=42,
    )
    print(f"Generated sine data: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # build model
    model = Sequential([
        Dense(1, 32, activation=ReLU()),
        Dense(32, 16, activation=Tanh()),
        Dense(16, 1),
    ])

    model.summary()

    # train
    trainer = Trainer(
        model=model,
        optimizer=Adam(lr=0.005),
        loss_fn=MSELoss(),
    )

    history = trainer.fit(
        X_train, y_train,
        epochs=300,
        batch_size=32,
        val_data=(X_test, y_test),
        verbose=True,
    )

    # evaluate
    print("\n--- Test Metrics ---")
    model.eval()
    y_pred = model.forward(X_test)

    print(f"MSE:  {mse(y_test, y_pred):.6f}")
    print(f"RMSE: {rmse(y_test, y_pred):.6f}")
    print(f"R²:   {r2_score(y_test, y_pred):.4f}")

    print_predictions(X_test, y_test, y_pred)

    print(f"\nFinal train loss: {history['loss'][-1]:.6f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.6f}")

    # save training plot
    plot_training_history(history, save_path=os.path.join(out_dir, "regression_training.png"))


if __name__ == "__main__":
    main()

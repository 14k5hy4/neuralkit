"""XOR example — proves the framework works end-to-end.

Trains a tiny 2-layer network to learn the XOR function.
"""

import numpy as np

from neuralkit.layers.dense import Dense
from neuralkit.activations import Sigmoid
from neuralkit.losses import MSELoss
from neuralkit.optimizers.sgd import SGD
from neuralkit.model import Sequential
from neuralkit.trainer import Trainer


def main():
    # XOR inputs and targets
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    np.random.seed(42)

    model = Sequential([
        Dense(2, 8, activation=Sigmoid()),
        Dense(8, 1, activation=Sigmoid()),
    ])

    optimizer = SGD(lr=2.0)
    loss_fn = MSELoss()
    trainer = Trainer(model, optimizer, loss_fn)

    print("Training on XOR...")
    history = trainer.fit(X, y, epochs=3000, verbose=True)

    # show predictions
    preds = model.predict(X)
    print("\nResults:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {preds[i][0]:.4f} (expected {y[i][0]:.0f})")

    print(f"\nFinal loss: {history['loss'][-1]:.6f}")


if __name__ == "__main__":
    main()

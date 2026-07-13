"""Iris classification example — full neuralkit pipeline.

Loads a hardcoded subset of the Iris dataset, preprocesses with
StandardScaler, trains a small network, and prints metrics.
"""

import numpy as np

from neuralkit.model import Sequential
from neuralkit.layers import Dense
from neuralkit.activations import ReLU
from neuralkit.losses import SoftmaxCrossEntropy
from neuralkit.optimizers import Adam
from neuralkit.trainer import Trainer
from neuralkit.data.transforms import StandardScaler, OneHotEncoder
from neuralkit.data.splits import train_test_split
from neuralkit.metrics import accuracy, classification_report


def load_iris_subset():
    """Hardcoded subset of Iris dataset (all 150 samples, 4 features, 3 classes).

    Features: sepal_length, sepal_width, petal_length, petal_width
    Labels: 0=setosa, 1=versicolor, 2=virginica
    """
    # fmt: off
    data = np.array([
        [5.1,3.5,1.4,0.2,0],[4.9,3.0,1.4,0.2,0],[4.7,3.2,1.3,0.2,0],[4.6,3.1,1.5,0.2,0],
        [5.0,3.6,1.4,0.2,0],[5.4,3.9,1.7,0.4,0],[4.6,3.4,1.4,0.3,0],[5.0,3.4,1.5,0.2,0],
        [4.4,2.9,1.4,0.2,0],[4.9,3.1,1.5,0.1,0],[5.4,3.7,1.5,0.2,0],[4.8,3.4,1.6,0.2,0],
        [4.8,3.0,1.4,0.1,0],[4.3,3.0,1.1,0.1,0],[5.8,4.0,1.2,0.2,0],[5.7,4.4,1.5,0.4,0],
        [5.4,3.9,1.3,0.4,0],[5.1,3.5,1.4,0.3,0],[5.7,3.8,1.7,0.3,0],[5.1,3.8,1.5,0.3,0],
        [5.4,3.4,1.7,0.2,0],[5.1,3.7,1.5,0.4,0],[4.6,3.6,1.0,0.2,0],[5.1,3.3,1.7,0.5,0],
        [4.8,3.4,1.9,0.2,0],[5.0,3.0,1.6,0.2,0],[5.0,3.4,1.6,0.4,0],[5.2,3.5,1.5,0.2,0],
        [5.2,3.4,1.4,0.2,0],[4.7,3.2,1.6,0.2,0],[4.8,3.1,1.6,0.2,0],[5.4,3.4,1.5,0.4,0],
        [5.2,4.1,1.5,0.1,0],[5.5,4.2,1.4,0.2,0],[4.9,3.1,1.5,0.2,0],[5.0,3.2,1.2,0.2,0],
        [5.5,3.5,1.3,0.2,0],[4.9,3.6,1.4,0.1,0],[4.4,3.0,1.3,0.2,0],[5.1,3.4,1.5,0.2,0],
        [5.0,3.5,1.3,0.3,0],[4.5,2.3,1.3,0.3,0],[4.4,3.2,1.3,0.2,0],[5.0,3.5,1.6,0.6,0],
        [5.1,3.8,1.9,0.4,0],[4.8,3.0,1.4,0.3,0],[5.1,3.8,1.6,0.2,0],[4.6,3.2,1.4,0.2,0],
        [5.3,3.7,1.5,0.2,0],[5.0,3.3,1.4,0.2,0],[7.0,3.2,4.7,1.4,1],[6.4,3.2,4.5,1.5,1],
        [6.9,3.1,4.9,1.5,1],[5.5,2.3,4.0,1.3,1],[6.5,2.8,4.6,1.5,1],[5.7,2.8,4.5,1.3,1],
        [6.3,3.3,4.7,1.6,1],[4.9,2.4,3.3,1.0,1],[6.6,2.9,4.6,1.3,1],[5.2,2.7,3.9,1.4,1],
        [5.0,2.0,3.5,1.0,1],[5.9,3.0,4.2,1.5,1],[6.0,2.2,4.0,1.0,1],[6.1,2.9,4.7,1.4,1],
        [5.6,2.9,3.6,1.3,1],[6.7,3.1,4.4,1.4,1],[5.6,3.0,4.5,1.5,1],[5.8,2.7,4.1,1.0,1],
        [6.2,2.2,4.5,1.5,1],[5.6,2.5,3.9,1.1,1],[5.9,3.2,4.8,1.8,1],[6.1,2.8,4.0,1.3,1],
        [6.3,2.5,4.9,1.5,1],[6.1,2.8,4.7,1.2,1],[6.4,2.9,4.3,1.3,1],[6.6,3.0,4.4,1.4,1],
        [6.8,2.8,4.8,1.4,1],[6.7,3.0,5.0,1.7,1],[6.0,2.9,4.5,1.5,1],[5.7,2.6,3.5,1.0,1],
        [5.5,2.4,3.8,1.1,1],[5.5,2.4,3.7,1.0,1],[5.8,2.7,3.9,1.2,1],[6.0,2.7,5.1,1.6,1],
        [5.4,3.0,4.5,1.5,1],[6.0,3.4,4.5,1.6,1],[6.7,3.1,4.7,1.5,1],[6.3,2.3,4.4,1.3,1],
        [5.6,3.0,4.1,1.3,1],[5.5,2.5,4.0,1.3,1],[5.5,2.6,4.4,1.2,1],[6.1,3.0,4.6,1.4,1],
        [5.8,2.6,4.0,1.2,1],[5.0,2.3,3.3,1.0,1],[5.6,2.7,4.2,1.3,1],[5.7,3.0,4.2,1.2,1],
        [5.7,2.9,4.2,1.3,1],[6.2,2.9,4.3,1.3,1],[5.1,2.5,3.0,1.1,1],[5.7,2.8,4.1,1.3,1],
        [6.3,3.3,6.0,2.5,2],[5.8,2.7,5.1,1.9,2],[7.1,3.0,5.9,2.1,2],[6.3,2.9,5.6,1.8,2],
        [6.5,3.0,5.8,2.2,2],[7.6,3.0,6.6,2.1,2],[4.9,2.5,4.5,1.7,2],[7.3,2.9,6.3,1.8,2],
        [6.7,2.5,5.8,1.8,2],[7.2,3.6,6.1,2.5,2],[6.5,3.2,5.1,2.0,2],[6.4,2.7,5.3,1.9,2],
        [6.8,3.0,5.5,2.1,2],[5.7,2.5,5.0,2.0,2],[5.8,2.8,5.1,2.4,2],[6.4,3.2,5.3,2.3,2],
        [6.5,3.0,5.5,1.8,2],[7.7,3.8,6.7,2.2,2],[7.7,2.6,6.9,2.3,2],[6.0,2.2,5.0,1.5,2],
        [6.9,3.2,5.7,2.3,2],[5.6,2.8,4.9,2.0,2],[7.7,2.8,6.7,2.0,2],[6.3,2.7,4.9,1.8,2],
        [6.7,3.3,5.7,2.1,2],[7.2,3.2,6.0,1.8,2],[6.2,2.8,4.8,1.8,2],[6.1,3.0,4.9,1.8,2],
        [6.4,2.8,5.6,2.1,2],[7.2,3.0,5.8,1.6,2],[7.4,2.8,6.1,1.9,2],[7.9,3.8,6.4,2.0,2],
        [6.4,2.8,5.6,2.2,2],[6.3,2.8,5.1,1.5,2],[6.1,2.6,5.6,1.4,2],[7.7,3.0,6.1,2.3,2],
        [6.3,3.4,5.6,2.4,2],[6.4,3.1,5.5,1.8,2],[6.0,3.0,4.8,1.8,2],[6.9,3.1,5.4,2.1,2],
        [6.7,3.1,5.6,2.4,2],[6.9,3.1,5.1,2.3,2],[5.8,2.7,5.1,1.9,2],[6.8,3.2,5.9,2.3,2],
        [6.7,3.3,5.7,2.5,2],[6.7,3.0,5.2,2.3,2],[6.3,2.5,5.0,1.9,2],[6.5,3.0,5.2,2.0,2],
        [6.2,3.4,5.4,2.3,2],[5.9,3.0,5.1,1.8,2],
    ])
    # fmt: on
    X = data[:, :4]
    y = data[:, 4].astype(int)
    return X, y


def main():
    np.random.seed(42)

    # load and preprocess
    X, y = load_iris_subset()
    print(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = OneHotEncoder()
    y_onehot = encoder.fit_transform(y)

    # split
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
        X, y_onehot, y, test_size=0.2, random_seed=42, stratify=y,
    )

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # build model
    model = Sequential([
        Dense(4, 16, activation=ReLU()),
        Dense(16, 8, activation=ReLU()),
        Dense(8, 3),  # raw logits, SoftmaxCrossEntropy handles softmax
    ])

    model.summary()

    # train
    trainer = Trainer(
        model=model,
        optimizer=Adam(lr=0.01),
        loss_fn=SoftmaxCrossEntropy(),
        metrics=[accuracy],
    )

    history = trainer.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        verbose=True,
    )

    # evaluate
    print("\n--- Test Results ---")
    model.eval()
    test_pred = model.forward(X_test)
    pred_labels = np.argmax(test_pred, axis=1)

    print(f"Test accuracy: {accuracy(y_test_labels, pred_labels):.4f}")
    print()
    print(classification_report(y_test_labels, pred_labels))


if __name__ == "__main__":
    main()

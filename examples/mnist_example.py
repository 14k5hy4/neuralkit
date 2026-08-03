"""MNIST digit classification — neuralkit showcase example.

Trains a deeper network with dropout and batch normalization on the
MNIST handwritten digit dataset (10 classes, 784 features).

Prerequisites:
    Download MNIST from http://yann.lecun.com/exdb/mnist/ and place
    the four .gz files in data/mnist/.
"""

import os
import numpy as np

from neuralkit.model import Sequential
from neuralkit.layers import Dense, Dropout, BatchNorm
from neuralkit.activations import ReLU
from neuralkit.losses import SoftmaxCrossEntropy
from neuralkit.optimizers import Adam
from neuralkit.trainer import Trainer
from neuralkit.data.datasets import load_mnist
from neuralkit.data.transforms import OneHotEncoder
from neuralkit.metrics import accuracy, classification_report, confusion_matrix
from neuralkit.utils.visualization import plot_training_history, plot_confusion_matrix


def main():
    np.random.seed(42)

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # load data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
    try:
        X_train, y_train, X_test, y_test = load_mnist(data_dir)
    except FileNotFoundError as e:
        print(e)
        print("\nTo run this example, download MNIST and place files in data/mnist/")
        return

    print(f"MNIST loaded: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"Image shape: {X_train.shape[1]} features (28x28 flattened)")

    # one-hot encode labels for softmax cross entropy
    encoder = OneHotEncoder()
    y_train_oh = encoder.fit_transform(y_train)
    y_test_oh = encoder.fit_transform(y_test)

    # use a subset for faster training (optional - comment out for full training)
    n_train = 10000
    n_test = 2000
    X_train, y_train_oh, y_train = X_train[:n_train], y_train_oh[:n_train], y_train[:n_train]
    X_test, y_test_oh, y_test = X_test[:n_test], y_test_oh[:n_test], y_test[:n_test]

    print(f"Using subset: {n_train} train, {n_test} test")

    # build a deeper network
    model = Sequential([
        Dense(784, 256, activation=ReLU()),
        BatchNorm(256),
        Dropout(rate=0.3),
        Dense(256, 128, activation=ReLU()),
        BatchNorm(128),
        Dropout(rate=0.2),
        Dense(128, 64, activation=ReLU()),
        Dense(64, 10),  # logits for SoftmaxCrossEntropy
    ])

    model.summary()

    # train
    trainer = Trainer(
        model=model,
        optimizer=Adam(lr=0.001, clip_norm=1.0),
        loss_fn=SoftmaxCrossEntropy(),
        metrics=[accuracy],
    )

    history = trainer.fit(
        X_train, y_train_oh,
        epochs=30,
        batch_size=64,
        val_data=(X_test, y_test_oh),
        verbose=True,
    )

    # evaluate
    print("\n--- Test Results ---")
    model.eval()
    test_pred = model.forward(X_test)
    pred_labels = np.argmax(test_pred, axis=1)

    test_acc = accuracy(y_test, pred_labels)
    print(f"Test accuracy: {test_acc:.4f}")
    print()
    print(classification_report(y_test, pred_labels))

    # save plots
    plot_training_history(history, save_path=os.path.join(out_dir, "mnist_training.png"))

    cm = confusion_matrix(y_test, pred_labels)
    plot_confusion_matrix(
        cm,
        class_names=[str(i) for i in range(10)],
        save_path=os.path.join(out_dir, "mnist_confusion_matrix.png"),
    )

    print(f"\nFinal train loss: {history['loss'][-1]:.4f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()

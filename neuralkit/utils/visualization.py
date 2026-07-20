"""Visualization utilities for training and evaluation.

Uses matplotlib for plotting. All functions can optionally save
plots to a file path.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _check_matplotlib():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> None:
    """Plot training loss and metric curves.

    Args:
        history: Dict returned by Trainer.fit() with keys like
            'loss', 'val_loss', 'accuracy', etc.
        save_path: If provided, save the figure to this path.
        figsize: Figure size (width, height).
    """
    _check_matplotlib()

    # separate loss from other metrics
    loss_keys = [k for k in history if "loss" in k]
    metric_keys = [k for k in history if "loss" not in k and not k.startswith("_")]

    n_plots = 1 + (1 if metric_keys else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # plot losses
    ax = axes[0]
    for key in loss_keys:
        label = key.replace("_", " ")
        ax.plot(history[key], label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # plot metrics
    if metric_keys:
        ax = axes[1]
        for key in metric_keys:
            label = key.replace("_", " ")
            ax.plot(history[key], label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        ax.set_title("Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (6, 5),
) -> None:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array of shape (n_classes, n_classes).
        class_names: Optional list of class label names.
        save_path: If provided, save the figure.
        figsize: Figure size.
    """
    _check_matplotlib()

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    # rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # text annotations
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    resolution: float = 0.02,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> None:
    """Plot decision boundary for a 2D classification problem.

    Only works with 2-feature inputs.

    Args:
        model: Trained model with a forward() method.
        X: Input data of shape (n_samples, 2).
        y: True labels.
        resolution: Grid resolution for the boundary.
        save_path: If provided, save the figure.
        figsize: Figure size.
    """
    _check_matplotlib()

    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary only works with 2D input features")

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.forward(grid)

    if preds.ndim == 2 and preds.shape[1] > 1:
        Z = np.argmax(preds, axis=1)
    else:
        Z = (preds > 0.5).astype(int).ravel()

    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k", s=40)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary")
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Decision boundary saved to {save_path}")
    else:
        plt.show()
    plt.close()

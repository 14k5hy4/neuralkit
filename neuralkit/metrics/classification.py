"""Classification metrics: accuracy, precision, recall, F1.

All functions take y_true and y_pred as numpy arrays.
Supports both binary and multi-class classification.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground truth labels (integer encoded).
        y_pred: Predicted labels (integer encoded).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute precision score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: 'macro' for unweighted mean, 'micro' for global.
    """
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "micro":
        tp = np.sum(y_true == y_pred)
        return float(tp / len(y_pred)) if len(y_pred) > 0 else 0.0

    precisions = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(p)

    return float(np.mean(precisions))


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute recall score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: 'macro' or 'micro'.
    """
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "micro":
        tp = np.sum(y_true == y_pred)
        return float(tp / len(y_true)) if len(y_true) > 0 else 0.0

    recalls = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(r)

    return float(np.mean(recalls))


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute F1 score (harmonic mean of precision and recall).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: 'macro' or 'micro'.
    """
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "micro":
        p = precision(y_true, y_pred, average="micro")
        r = recall(y_true, y_pred, average="micro")
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    # FIXME: weighted average not implemented yet
    f1s = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f)

    return float(np.mean(f1s))

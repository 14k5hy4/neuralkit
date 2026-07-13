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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix.

    Returns:
        Matrix of shape (n_classes, n_classes) where element [i, j]
        is the count of samples with true label i predicted as j.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Generate a text report showing main classification metrics per class."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))

    lines = []
    lines.append(f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    lines.append("")

    total_support = 0
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        support = tp + fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        lines.append(f"{str(cls):>12} {p:>10.4f} {r:>10.4f} {f:>10.4f} {support:>10}")
        total_support += support

    lines.append("")
    acc = accuracy(y_true, y_pred)
    macro_p = precision(y_true, y_pred, average="macro")
    macro_r = recall(y_true, y_pred, average="macro")
    macro_f = f1_score(y_true, y_pred, average="macro")

    lines.append(f"{'accuracy':>12} {'':>10} {'':>10} {acc:>10.4f} {total_support:>10}")
    lines.append(f"{'macro avg':>12} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f:>10.4f} {total_support:>10}")

    return "\n".join(lines)

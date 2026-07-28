"""K-fold cross-validation utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def k_fold_split(
    n_samples: int,
    k: int = 5,
    shuffle: bool = True,
    random_seed: Optional[int] = None,
    stratify: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split indices into k folds for cross-validation.

    Args:
        n_samples: Total number of samples.
        k: Number of folds.
        shuffle: Whether to shuffle before splitting.
        random_seed: Random seed for reproducibility.
        stratify: If provided, preserve class proportions in each fold.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    if stratify is not None:
        return _stratified_k_fold(n_samples, k, stratify, shuffle, random_seed)

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(indices)

    fold_size = n_samples // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, val_idx))

    return folds


def _stratified_k_fold(
    n_samples: int,
    k: int,
    y: np.ndarray,
    shuffle: bool,
    random_seed: Optional[int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold: keeps class balance across folds."""
    rng = np.random.RandomState(random_seed)
    classes = np.unique(y)

    # group indices by class
    class_indices = {}
    for cls in classes:
        idx = np.where(y == cls)[0]
        if shuffle:
            rng.shuffle(idx)
        class_indices[cls] = idx

    # build folds
    folds_val = [[] for _ in range(k)]
    for cls in classes:
        idx = class_indices[cls]
        fold_size = len(idx) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(idx)
            folds_val[i].extend(idx[start:end])

    folds = []
    all_idx = np.arange(n_samples)
    for i in range(k):
        val_idx = np.array(folds_val[i])
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
        folds.append((train_idx, val_idx))

    return folds


def cross_validate(
    model_fn: Callable,
    optimizer_fn: Callable,
    loss_fn,
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    epochs: int = 100,
    batch_size: Optional[int] = None,
    metrics: Optional[List] = None,
    stratify: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate a model across k folds.

    Args:
        model_fn: Callable that returns a fresh model instance.
        optimizer_fn: Callable that returns a fresh optimizer instance.
        loss_fn: Loss function instance (will be reused).
        x: Input data.
        y: Target data.
        k: Number of folds.
        epochs: Training epochs per fold.
        batch_size: Mini-batch size (None for full-batch).
        metrics: List of metric functions.
        stratify: Labels for stratified splitting.
        random_seed: Seed for reproducibility.
        verbose: Print per-fold results.

    Returns:
        Dict with metric names mapped to {'mean': ..., 'std': ...}.
    """
    from neuralkit.trainer import Trainer

    folds = k_fold_split(len(x), k=k, stratify=stratify, random_seed=random_seed)
    metrics = metrics or []
    all_results: Dict[str, List[float]] = {"loss": []}
    for m in metrics:
        name = m.__name__ if hasattr(m, '__name__') else str(m)
        all_results[name] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_fn()
        optimizer = optimizer_fn()
        trainer = Trainer(model, optimizer, loss_fn, metrics=metrics)

        trainer.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False,
        )

        results = trainer.evaluate(x_val, y_val)

        for key, val in results.items():
            if key in all_results:
                all_results[key].append(val)

        if verbose:
            metrics_str = " — ".join(f"{k}: {v:.4f}" for k, v in results.items())
            print(f"Fold {fold_idx + 1}/{k}: {metrics_str}")

    # compute mean and std
    summary = {}
    for key, values in all_results.items():
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    if verbose:
        print("\n--- Cross-validation summary ---")
        for key, stats in summary.items():
            print(f"{key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return summary

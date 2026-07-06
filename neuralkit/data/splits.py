"""Dataset splitting utilities."""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np


def train_test_split(
    *arrays: np.ndarray,
    test_size: float = 0.2,
    random_seed: Optional[int] = None,
    stratify: Optional[np.ndarray] = None,
) -> list:
    """Split arrays into train and test subsets.

    Args:
        *arrays: Arrays to split (must have same length on axis 0).
        test_size: Fraction of data to use for test set.
        random_seed: Seed for reproducibility.
        stratify: If provided, use these labels for stratified splitting.

    Returns:
        List of arrays: [train_1, test_1, train_2, test_2, ...]
    """
    if not arrays:
        raise ValueError("need at least one array to split")

    n = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != n:
            raise ValueError("all arrays must have the same number of samples")

    rng = np.random.RandomState(random_seed)

    if stratify is not None:
        train_idx, test_idx = _stratified_split(stratify, test_size, rng)
    else:
        indices = rng.permutation(n)
        split_point = int(n * (1 - test_size))
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]

    result = []
    for arr in arrays:
        result.append(arr[train_idx])
        result.append(arr[test_idx])
    return result


def train_val_test_split(
    *arrays: np.ndarray,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_seed: Optional[int] = None,
    stratify: Optional[np.ndarray] = None,
) -> list:
    """Split arrays into train, validation, and test subsets.

    First splits off test, then splits remaining into train/val.
    """
    if not arrays:
        raise ValueError("need at least one array to split")

    # first split: separate test set
    remaining_size = 1.0 - test_size
    split1 = train_test_split(
        *arrays,
        test_size=test_size,
        random_seed=random_seed,
        stratify=stratify,
    )

    # extract train+val arrays and test arrays
    train_val_arrays = [split1[i] for i in range(0, len(split1), 2)]
    test_arrays = [split1[i] for i in range(1, len(split1), 2)]

    # compute adjusted val_size relative to remaining data
    adjusted_val = val_size / remaining_size

    # stratify on the train+val portion if needed
    strat = None
    if stratify is not None:
        # the stratify array was the last one passed originally,
        # but we need the train portion of it
        strat_idx = list(arrays).index(stratify) if stratify is not None else None
        if strat_idx is not None:
            strat = train_val_arrays[strat_idx]

    seed2 = random_seed + 1 if random_seed is not None else None
    split2 = train_test_split(
        *train_val_arrays,
        test_size=adjusted_val,
        random_seed=seed2,
        stratify=strat,
    )

    train_arrays = [split2[i] for i in range(0, len(split2), 2)]
    val_arrays = [split2[i] for i in range(1, len(split2), 2)]

    # interleave: train_1, val_1, test_1, train_2, val_2, test_2, ...
    result = []
    for t, v, te in zip(train_arrays, val_arrays, test_arrays):
        result.extend([t, v, te])
    return result


def _stratified_split(
    labels: np.ndarray,
    test_size: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices while maintaining class proportions."""
    classes = np.unique(labels)
    train_indices = []
    test_indices = []

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * test_size))
        test_indices.extend(cls_idx[:n_test])
        train_indices.extend(cls_idx[n_test:])

    return np.array(train_indices), np.array(test_indices)

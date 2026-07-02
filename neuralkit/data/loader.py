"""Data loading utilities — Dataset and DataLoader."""

from __future__ import annotations

from typing import Iterator, Tuple, Optional
import numpy as np


class Dataset:
    """Base class for all datasets.

    Subclasses should implement __len__ and __getitem__.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError


class ArrayDataset(Dataset):
    """Wraps numpy arrays as a dataset.

    Args:
        *arrays: One or more numpy arrays. All must have the same
            length along the first axis.
    """

    def __init__(self, *arrays: np.ndarray) -> None:
        if not arrays:
            raise ValueError("need at least one array")
        n = arrays[0].shape[0]
        for i, arr in enumerate(arrays):
            if arr.shape[0] != n:
                raise ValueError(
                    f"array {i} has length {arr.shape[0]}, expected {n}"
                )
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, index):
        if len(self.arrays) == 1:
            return self.arrays[0][index]
        return tuple(a[index] for a in self.arrays)


class DataLoader:
    """Iterates over a dataset in batches.

    Supports shuffling and dropping the last incomplete batch.

    Args:
        dataset: A Dataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle indices each epoch.
        drop_last: If True, drop the last batch if it's smaller
            than batch_size.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        n = len(self.dataset)
        indices = np.arange(n)

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_idx = indices[start:end]
            yield self.dataset[batch_idx]

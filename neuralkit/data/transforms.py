"""Data preprocessing transforms and scalers.

Sklearn-style fit/transform API for common preprocessing operations.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np


class Normalize:
    """Normalize data to zero mean and unit variance using provided values.

    Unlike StandardScaler, this doesn't learn stats from data — you
    pass in the mean and std directly. Useful when you already know
    dataset statistics (e.g. ImageNet normalization).

    Args:
        mean: Per-feature means.
        std: Per-feature standard deviations.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Undo the normalization."""
        return x * self.std + self.mean


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance.

    Learns statistics from data via fit(), then applies them via transform().
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, x: np.ndarray) -> "StandardScaler":
        """Compute mean and std from training data."""
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0)
        # avoid division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self._fitted, "call fit() before transform()"
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Convenience method: fit then transform in one call."""
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self._fitted, "call fit() before inverse_transform()"
        return x * self.std_ + self.mean_


class MinMaxScaler:
    """Scale features to a given range (default [0, 1]).

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, feature_range: tuple = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)
        # handle constant features
        diff = self.max_ - self.min_
        diff[diff == 0] = 1.0
        self.max_ = self.min_ + diff
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self._fitted, "call fit() first"
        lo, hi = self.feature_range
        x_std = (x - self.min_) / (self.max_ - self.min_)
        return x_std * (hi - lo) + lo

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    # FIXME: inverse_transform not implemented yet


class OneHotEncoder:
    """Encode integer labels as one-hot vectors."""

    def __init__(self) -> None:
        self.num_classes_: Optional[int] = None
        self._fitted = False

    def fit(self, y: np.ndarray) -> "OneHotEncoder":
        """Learn the number of classes from the labels."""
        self.num_classes_ = int(np.max(y)) + 1
        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        assert self._fitted, "call fit() before transform()"
        y_int = y.astype(int).ravel()
        one_hot = np.zeros((len(y_int), self.num_classes_))
        one_hot[np.arange(len(y_int)), y_int] = 1.0
        return one_hot

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)


class Compose:
    """Chain multiple transforms together.

    Example::
        transform = Compose([
            StandardScaler(),
        ])
        # must call fit on individual transforms first, or use
        # transforms that don't need fitting (like Normalize)
    """

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            if hasattr(t, '__call__'):
                x = t(x)
            elif hasattr(t, 'transform'):
                x = t.transform(x)
        return x

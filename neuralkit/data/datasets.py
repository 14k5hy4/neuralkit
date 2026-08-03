"""Built-in dataset loaders.

Provides utilities to load common datasets for quick experimentation.
"""

from __future__ import annotations

import gzip
import os
import struct
from typing import Tuple

import numpy as np


def load_mnist(data_dir: str = "data/mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset from local IDX files.

    Expects the standard MNIST files in data_dir:
        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz
        t10k-images-idx3-ubyte.gz
        t10k-labels-idx1-ubyte.gz

    If files don't exist, prints instructions for downloading.

    Returns:
        (X_train, y_train, X_test, y_test) with images as float32
        arrays normalized to [0, 1] and labels as int arrays.
    """
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    # check all files exist
    for key, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"MNIST file not found: {fpath}\n"
                f"Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/ "
                f"and place the .gz files in '{data_dir}/'."
            )

    X_train = _read_images(os.path.join(data_dir, files["train_images"]))
    y_train = _read_labels(os.path.join(data_dir, files["train_labels"]))
    X_test = _read_images(os.path.join(data_dir, files["test_images"]))
    y_test = _read_labels(os.path.join(data_dir, files["test_labels"]))

    return X_train, y_train, X_test, y_test


def _read_images(filepath: str) -> np.ndarray:
    """Read MNIST image file (IDX3 format, gzipped)."""
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number {magic} for images"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows * cols).astype(np.float32) / 255.0
    return images


def _read_labels(filepath: str) -> np.ndarray:
    """Read MNIST label file (IDX1 format, gzipped)."""
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number {magic} for labels"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)

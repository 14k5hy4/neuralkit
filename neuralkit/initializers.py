"""Weight initialization strategies.

Functions return numpy arrays of the requested shape, initialized
according to various schemes from the literature.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def zeros(shape: Tuple[int, ...]) -> np.ndarray:
    """Initialize all weights to zero."""
    return np.zeros(shape)


def ones(shape: Tuple[int, ...]) -> np.ndarray:
    """Initialize all weights to one."""
    return np.ones(shape)


def constant(shape: Tuple[int, ...], value: float = 0.0) -> np.ndarray:
    """Initialize all weights to a constant value."""
    return np.full(shape, value)


def xavier_uniform(shape: Tuple[int, ...]) -> np.ndarray:
    """Xavier/Glorot uniform initialization (Glorot & Bengio, 2010).

    Draws from U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out)).
    Good default for sigmoid and tanh networks.
    """
    fan_in, fan_out = _compute_fans(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


def xavier_normal(shape: Tuple[int, ...]) -> np.ndarray:
    """Xavier/Glorot normal initialization.

    Draws from N(0, std) where std = sqrt(2 / (fan_in + fan_out)).
    """
    fan_in, fan_out = _compute_fans(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std


def he_uniform(shape: Tuple[int, ...]) -> np.ndarray:
    """He uniform initialization (He et al., 2015).

    Draws from U(-limit, limit) where limit = sqrt(6 / fan_in).
    Designed for ReLU networks.
    """
    fan_in, _ = _compute_fans(shape)
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)


def he_normal(shape: Tuple[int, ...]) -> np.ndarray:
    """He normal initialization (He et al., 2015).

    Draws from N(0, std) where std = sqrt(2 / fan_in).
    Default for ReLU-family networks.
    """
    fan_in, _ = _compute_fans(shape)
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


def lecun_normal(shape: Tuple[int, ...]) -> np.ndarray:
    """LeCun normal initialization.

    Draws from N(0, std) where std = sqrt(1 / fan_in).
    Works well with SELU activation.
    """
    fan_in, _ = _compute_fans(shape)
    std = np.sqrt(1.0 / fan_in)
    return np.random.randn(*shape) * std


def _compute_fans(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Compute fan_in and fan_out for a weight tensor shape."""
    if len(shape) < 1:
        raise ValueError("shape must have at least 1 dimension")
    if len(shape) == 1:
        return shape[0], shape[0]
    # for 2D (dense): fan_in = shape[0], fan_out = shape[1]
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    return fan_in, fan_out

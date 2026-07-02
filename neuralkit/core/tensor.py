"""Basic tensor wrapper around numpy arrays.

Provides a thin abstraction over np.ndarray with common operations
needed for building neural networks from scratch.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


class Tensor:
    """A simple tensor class wrapping a numpy ndarray.

    This is intentionally minimal — just enough to build layers on top of.
    We store the raw numpy data and expose arithmetic ops that return
    new Tensor instances.

    Attributes:
        data: The underlying numpy array.
    """

    def __init__(self, data: Union[np.ndarray, list, float]) -> None:
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def zeros(shape: Tuple[int, ...]) -> "Tensor":
        """Create a tensor filled with zeros."""
        return Tensor(np.zeros(shape))

    @staticmethod
    def ones(shape: Tuple[int, ...]) -> "Tensor":
        """Create a tensor filled with ones."""
        return Tensor(np.ones(shape))

    @staticmethod
    def random(shape: Tuple[int, ...], seed: Optional[int] = None) -> "Tensor":
        """Create a tensor with random values drawn from U(0, 1)."""
        rng = np.random.default_rng(seed)
        return Tensor(rng.random(shape))

    @staticmethod
    def randn(shape: Tuple[int, ...], seed: Optional[int] = None) -> "Tensor":
        """Create a tensor with values drawn from N(0, 1)."""
        rng = np.random.default_rng(seed)
        return Tensor(rng.standard_normal(shape))

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------

    def add(self, other: "Tensor") -> "Tensor":
        """Element-wise addition."""
        return Tensor(self.data + other.data)

    def subtract(self, other: "Tensor") -> "Tensor":
        """Element-wise subtraction."""
        return Tensor(self.data - other.data)

    def multiply(self, other: "Tensor") -> "Tensor":
        """Element-wise (Hadamard) multiplication."""
        return Tensor(self.data * other.data)

    def divide(self, other: "Tensor") -> "Tensor":
        """Element-wise division."""
        return Tensor(self.data / other.data)

    def power(self, exponent: float) -> "Tensor":
        """Raise each element to the given power."""
        return Tensor(np.power(self.data, exponent))

    def dot(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication via np.dot."""
        return Tensor(np.dot(self.data, other.data))

    def transpose(self) -> "Tensor":
        """Return the transpose of this tensor."""
        return Tensor(self.data.T)

    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor":
        """Reshape tensor to new_shape. Follows numpy reshape semantics."""
        return Tensor(self.data.reshape(new_shape))

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        """Sum elements along an axis (or all elements if axis is None)."""
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims))

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        """Mean of elements along an axis."""
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims))

    # ------------------------------------------------------------------
    # Operator overloads (convenience)
    # ------------------------------------------------------------------

    def __add__(self, other: "Tensor") -> "Tensor":
        return self.add(other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self.subtract(other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        return self.multiply(other)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        return self.divide(other)

    def __pow__(self, exponent: float) -> "Tensor":
        return self.power(exponent)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.dot(other)

    def __neg__(self) -> "Tensor":
        return Tensor(-self.data)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """Return the raw numpy array."""
        return self.data

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.data.shape[0]

    def __repr__(self) -> str:
        # show small tensors inline, larger ones just show shape
        if self.size <= 10:
            return f"Tensor({self.data}, dtype={self.dtype})"
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Tensor):
            return NotImplemented
        return np.array_equal(self.data, other.data)

    # TODO: add __array__ protocol for seamless numpy interop

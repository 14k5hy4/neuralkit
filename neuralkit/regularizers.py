"""Weight regularization: L1, L2, and ElasticNet.

Regularizers compute a penalty term that gets added to the loss,
and provide a gradient contribution for backpropagation.
"""

from __future__ import annotations

from typing import List
import numpy as np


class L1:
    """L1 (Lasso) regularization — encourages sparsity.

    penalty = lambda_ * sum(|w|)

    Args:
        lambda_: Regularization strength. Default 0.01.
    """

    def __init__(self, lambda_: float = 0.01) -> None:
        self.lambda_ = lambda_

    def penalty(self, layers: List) -> float:
        """Compute the L1 penalty over all trainable params."""
        total = 0.0
        for layer in layers:
            for param in layer.params.values():
                total += np.sum(np.abs(param))
        return self.lambda_ * total

    def grad(self, param: np.ndarray) -> np.ndarray:
        """Gradient of L1 penalty: lambda * sign(w)."""
        return self.lambda_ * np.sign(param)


class L2:
    """L2 (Ridge) regularization — discourages large weights.

    penalty = 0.5 * lambda_ * sum(w²)

    Args:
        lambda_: Regularization strength. Default 0.01.
    """

    def __init__(self, lambda_: float = 0.01) -> None:
        self.lambda_ = lambda_

    def penalty(self, layers: List) -> float:
        """Compute the L2 penalty over all trainable params."""
        total = 0.0
        for layer in layers:
            for param in layer.params.values():
                total += np.sum(param ** 2)
        return 0.5 * self.lambda_ * total

    def grad(self, param: np.ndarray) -> np.ndarray:
        """Gradient of L2 penalty: lambda * w."""
        return self.lambda_ * param


class ElasticNet:
    """ElasticNet — combination of L1 and L2 regularization.

    penalty = l1_ratio * L1 + (1 - l1_ratio) * L2

    Args:
        lambda_: Overall regularization strength.
        l1_ratio: Mix between L1 and L2. 1.0 = pure L1, 0.0 = pure L2.
    """

    def __init__(self, lambda_: float = 0.01, l1_ratio: float = 0.5) -> None:
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio
        self._l1 = L1(lambda_=lambda_ * l1_ratio)
        self._l2 = L2(lambda_=lambda_ * (1 - l1_ratio))

    def penalty(self, layers: List) -> float:
        return self._l1.penalty(layers) + self._l2.penalty(layers)

    def grad(self, param: np.ndarray) -> np.ndarray:
        return self._l1.grad(param) + self._l2.grad(param)

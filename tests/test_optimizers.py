"""Tests for optimizers (SGD, Adam)."""

import unittest
import numpy as np

from neuralkit.layers.dense import Dense
from neuralkit.optimizers.sgd import SGD
from neuralkit.optimizers.adam import Adam


class _QuadraticLayer:
    """Fake layer wrapping a single parameter for testing optimizers.

    Minimizes f(x) = 0.5 * ||x - target||^2, so gradient is (x - target).
    """

    def __init__(self, init_val: np.ndarray, target: np.ndarray):
        self._param = init_val.copy()
        self._target = target
        self._grad = np.zeros_like(init_val)

    def compute_grad(self):
        self._grad = self._param - self._target

    @property
    def params(self):
        return {"x": self._param}

    @property
    def grads(self):
        return {"x": self._grad}


class TestSGD(unittest.TestCase):

    def test_basic_update(self):
        """Parameter should move toward target after one step."""
        layer = _QuadraticLayer(np.array([5.0]), np.array([0.0]))
        opt = SGD(lr=0.1)

        layer.compute_grad()
        opt.step([layer])

        # should have moved from 5.0 toward 0.0
        self.assertLess(layer.params["x"][0], 5.0)

    def test_convergence(self):
        """SGD should converge on a simple quadratic."""
        target = np.array([3.0, -2.0])
        layer = _QuadraticLayer(np.array([10.0, 10.0]), target)
        opt = SGD(lr=0.1)

        for _ in range(200):
            layer.compute_grad()
            opt.step([layer])

        np.testing.assert_allclose(layer.params["x"], target, atol=0.01)

    def test_momentum(self):
        target = np.array([1.0])
        layer = _QuadraticLayer(np.array([10.0]), target)
        opt = SGD(lr=0.05, momentum=0.9)

        for _ in range(200):
            layer.compute_grad()
            opt.step([layer])

        np.testing.assert_allclose(layer.params["x"], target, atol=0.05)


class TestAdam(unittest.TestCase):

    def test_convergence(self):
        """Adam should converge on a quadratic."""
        target = np.array([3.0, -1.0, 7.0])
        layer = _QuadraticLayer(np.zeros(3), target)
        opt = Adam(lr=0.1)

        for _ in range(300):
            layer.compute_grad()
            opt.step([layer])

        np.testing.assert_allclose(layer.params["x"], target, atol=0.05)

    def test_with_dense_layer(self):
        """Adam should be able to update Dense layer params without error."""
        np.random.seed(0)
        layer = Dense(2, 1)
        x = np.array([[1.0, 2.0]])
        y = np.array([[1.0]])

        opt = Adam(lr=0.01)
        initial_W = layer.W.copy()

        out = layer.forward(x)
        grad = 2 * (out - y)  # MSE gradient
        layer.backward(grad)
        opt.step([layer])

        # weights should have changed
        self.assertFalse(np.array_equal(layer.W, initial_W))


if __name__ == "__main__":
    unittest.main()

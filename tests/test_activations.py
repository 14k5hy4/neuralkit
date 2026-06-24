"""Tests for activation functions."""

import unittest

import numpy as np

from neuralkit.activations import ReLU, Sigmoid, Tanh


class TestSigmoid(unittest.TestCase):
    """Test sigmoid activation forward and backward."""

    def setUp(self) -> None:
        self.act = Sigmoid()

    def test_output_range(self) -> None:
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        out = self.act.forward(x)
        self.assertTrue(np.all(out > 0))
        self.assertTrue(np.all(out < 1))

    def test_known_value(self) -> None:
        out = self.act.forward(np.array([0.0]))
        np.testing.assert_almost_equal(out, [0.5])

    def test_gradient_shape(self) -> None:
        x = np.random.randn(3, 4)
        self.act.forward(x)
        grad = self.act.backward(np.ones_like(x))
        self.assertEqual(grad.shape, x.shape)


class TestReLU(unittest.TestCase):
    """Test ReLU activation forward and backward."""

    def setUp(self) -> None:
        self.act = ReLU()

    def test_positive_passthrough(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        out = self.act.forward(x)
        np.testing.assert_array_equal(out, x)

    def test_negative_zeroed(self) -> None:
        x = np.array([-1.0, -0.5, 0.0])
        out = self.act.forward(x)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

    def test_backward_mask(self) -> None:
        x = np.array([-2.0, 3.0, -1.0, 5.0])
        self.act.forward(x)
        grad = self.act.backward(np.ones_like(x))
        expected = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(grad, expected)


class TestTanh(unittest.TestCase):
    """Test tanh activation forward and backward."""

    def setUp(self) -> None:
        self.act = Tanh()

    def test_output_range(self) -> None:
        x = np.linspace(-5, 5, 100)
        out = self.act.forward(x)
        self.assertTrue(np.all(out >= -1))
        self.assertTrue(np.all(out <= 1))

    def test_zero_input(self) -> None:
        out = self.act.forward(np.array([0.0]))
        np.testing.assert_almost_equal(out, [0.0])

    def test_gradient_at_zero(self) -> None:
        """Gradient of tanh at x=0 should be 1 (since tanh'(0) = 1 - 0² = 1)."""
        self.act.forward(np.array([0.0]))
        grad = self.act.backward(np.array([1.0]))
        np.testing.assert_almost_equal(grad, [1.0])


if __name__ == "__main__":
    unittest.main()

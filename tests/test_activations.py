"""Tests for activation functions."""

import unittest

import numpy as np

from neuralkit.activations import ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish


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


class TestLeakyReLU(unittest.TestCase):

    def test_positive_passthrough(self):
        act = LeakyReLU(negative_slope=0.01)
        x = np.array([1.0, 3.0, 5.0])
        np.testing.assert_array_equal(act.forward(x), x)

    def test_negative_slope(self):
        act = LeakyReLU(negative_slope=0.2)
        x = np.array([-5.0, -1.0])
        out = act.forward(x)
        np.testing.assert_allclose(out, [-1.0, -0.2])

    def test_gradient_numerical(self):
        act = LeakyReLU(negative_slope=0.1)
        x = np.array([-2.0, -0.5, 0.5, 2.0])
        act.forward(x)
        grad = act.backward(np.ones_like(x))
        expected = np.array([0.1, 0.1, 1.0, 1.0])
        np.testing.assert_allclose(grad, expected)


class TestELU(unittest.TestCase):

    def test_positive_passthrough(self):
        act = ELU(alpha=1.0)
        x = np.array([1.0, 2.0])
        np.testing.assert_array_equal(act.forward(x), x)

    def test_negative_exponential(self):
        act = ELU(alpha=1.0)
        x = np.array([0.0, -1.0])
        out = act.forward(x)
        expected = np.array([0.0, np.exp(-1.0) - 1.0])
        np.testing.assert_allclose(out, expected, atol=1e-7)

    def test_backward_shape(self):
        act = ELU()
        x = np.random.randn(4, 3)
        act.forward(x)
        grad = act.backward(np.ones_like(x))
        self.assertEqual(grad.shape, x.shape)


class TestSwish(unittest.TestCase):

    def test_zero(self):
        act = Swish()
        out = act.forward(np.array([0.0]))
        np.testing.assert_almost_equal(out, [0.0])

    def test_positive(self):
        """Swish(x) for large positive x ≈ x."""
        act = Swish()
        out = act.forward(np.array([10.0]))
        self.assertAlmostEqual(float(out[0]), 10.0, places=3)

    def test_gradient_numerical(self):
        """Check gradient with finite differences."""
        act = Swish()
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        eps = 1e-5

        numerical = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy(); x_plus[i] += eps
            x_minus = x.copy(); x_minus[i] -= eps
            # swish = x * sigmoid(x)
            def swish(z): return z / (1 + np.exp(-z))
            numerical[i] = (swish(x_plus[i]) - swish(x_minus[i])) / (2 * eps)

        act.forward(x)
        analytical = act.backward(np.ones_like(x))
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()

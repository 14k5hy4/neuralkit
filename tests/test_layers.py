"""Tests for layer implementations."""

import unittest
import numpy as np

from neuralkit.layers.dense import Dense
from neuralkit.activations import ReLU, Sigmoid


class TestDenseLayer(unittest.TestCase):

    def test_forward_shape(self):
        """Output shape should be (batch_size, output_dim)."""
        layer = Dense(4, 3)
        x = np.random.randn(8, 4)
        out = layer.forward(x)
        self.assertEqual(out.shape, (8, 3))

    def test_forward_with_activation(self):
        layer = Dense(3, 2, activation=ReLU())
        x = np.random.randn(5, 3)
        out = layer.forward(x)
        self.assertEqual(out.shape, (5, 2))
        # ReLU should produce non-negative outputs
        self.assertTrue(np.all(out >= 0))

    def test_backward_shape(self):
        """Backward should return gradient with same shape as input."""
        layer = Dense(4, 3)
        x = np.random.randn(8, 4)
        layer.forward(x)
        grad_out = np.random.randn(8, 3)
        grad_in = layer.backward(grad_out)
        self.assertEqual(grad_in.shape, (8, 4))

    def test_gradient_numerical(self):
        """Check analytical gradients against numerical (finite differences)."""
        np.random.seed(42)
        layer = Dense(3, 2)
        x = np.random.randn(4, 3)

        # compute analytical gradients
        out = layer.forward(x)
        grad_out = np.ones_like(out)
        layer.backward(grad_out)
        analytical_grad_W = layer.grads["W"].copy()

        # numerical gradient checking for W
        eps = 1e-5
        numerical_grad_W = np.zeros_like(layer.W)
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                layer.W[i, j] += eps
                out_plus = layer.forward(x).sum()
                layer.W[i, j] -= 2 * eps
                out_minus = layer.forward(x).sum()
                layer.W[i, j] += eps  # restore
                numerical_grad_W[i, j] = (out_plus - out_minus) / (2 * eps)

        np.testing.assert_allclose(
            analytical_grad_W, numerical_grad_W, rtol=1e-4, atol=1e-6,
            err_msg="Dense W gradient doesn't match numerical gradient"
        )

    def test_params_dict(self):
        layer = Dense(5, 3)
        params = layer.params
        self.assertIn("W", params)
        self.assertIn("b", params)
        self.assertEqual(params["W"].shape, (5, 3))
        self.assertEqual(params["b"].shape, (1, 3))

    def test_single_sample(self):
        """Should work with batch_size=1."""
        layer = Dense(4, 2)
        x = np.random.randn(1, 4)
        out = layer.forward(x)
        grad = layer.backward(np.ones_like(out))
        self.assertEqual(out.shape, (1, 2))
        self.assertEqual(grad.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()

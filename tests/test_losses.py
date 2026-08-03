"""Tests for loss functions."""

import unittest

import numpy as np

from neuralkit.losses import (
    MSELoss, CrossEntropyLoss, SoftmaxCrossEntropy,
    HuberLoss, BinaryCrossEntropyLoss,
)


class TestMSELoss(unittest.TestCase):

    def test_zero_loss(self):
        loss = MSELoss()
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(loss.forward(y, y), 0.0)

    def test_known_value(self):
        loss = MSELoss()
        pred = np.array([[2.0]])
        true = np.array([[1.0]])
        # MSE = (2-1)^2 / 1 = 1.0
        self.assertAlmostEqual(loss.forward(pred, true), 1.0)

    def test_backward_shape(self):
        loss = MSELoss()
        pred = np.random.randn(4, 3)
        true = np.random.randn(4, 3)
        loss.forward(pred, true)
        grad = loss.backward()
        self.assertEqual(grad.shape, pred.shape)


class TestHuberLoss(unittest.TestCase):

    def test_zero_loss(self):
        loss = HuberLoss(delta=1.0)
        y = np.array([[1.0, 2.0]])
        self.assertAlmostEqual(loss.forward(y, y), 0.0)

    def test_small_error_equals_mse(self):
        """For errors smaller than delta, Huber should behave like 0.5*MSE."""
        loss_huber = HuberLoss(delta=10.0)
        loss_mse = MSELoss()

        pred = np.array([[1.5]])
        true = np.array([[1.0]])
        huber_val = loss_huber.forward(pred, true)
        # Huber with large delta: 0.5 * 0.5^2 = 0.125
        self.assertAlmostEqual(huber_val, 0.125)

    def test_large_error_is_linear(self):
        """For errors larger than delta, should be approximately linear."""
        loss = HuberLoss(delta=1.0)
        pred = np.array([[10.0]])
        true = np.array([[0.0]])
        # Huber: delta * |diff| - 0.5 * delta^2 = 1.0 * 10 - 0.5 = 9.5
        val = loss.forward(pred, true)
        self.assertAlmostEqual(val, 9.5)

    def test_backward_shape(self):
        loss = HuberLoss()
        pred = np.random.randn(5, 2)
        true = np.random.randn(5, 2)
        loss.forward(pred, true)
        grad = loss.backward()
        self.assertEqual(grad.shape, pred.shape)

    def test_gradient_clipped(self):
        """Gradient should be clipped to [-delta, delta]."""
        loss = HuberLoss(delta=1.0)
        pred = np.array([[100.0]])
        true = np.array([[0.0]])
        loss.forward(pred, true)
        grad = loss.backward()
        # gradient should be delta * sign(diff) / n = 1.0 * 1 / 1 = 1.0
        self.assertAlmostEqual(grad[0, 0], 1.0)


class TestBinaryCrossEntropyLoss(unittest.TestCase):

    def test_perfect_prediction(self):
        loss = BinaryCrossEntropyLoss()
        pred = np.array([[0.999], [0.001]])
        true = np.array([[1.0], [0.0]])
        val = loss.forward(pred, true)
        # should be close to 0
        self.assertLess(val, 0.01)

    def test_worst_prediction(self):
        loss = BinaryCrossEntropyLoss()
        pred = np.array([[0.001], [0.999]])
        true = np.array([[1.0], [0.0]])
        val = loss.forward(pred, true)
        # should be large
        self.assertGreater(val, 3.0)

    def test_backward_shape(self):
        loss = BinaryCrossEntropyLoss()
        pred = np.array([[0.7], [0.3], [0.9]])
        true = np.array([[1.0], [0.0], [1.0]])
        loss.forward(pred, true)
        grad = loss.backward()
        self.assertEqual(grad.shape, pred.shape)

    def test_numerical_gradient(self):
        """Verify gradient numerically."""
        loss = BinaryCrossEntropyLoss()
        pred = np.array([[0.6], [0.4]])
        true = np.array([[1.0], [0.0]])

        loss.forward(pred, true)
        analytical = loss.backward()

        # numerical gradient
        eps = 1e-5
        numerical = np.zeros_like(pred)
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred_plus = pred.copy()
                pred_plus[i, j] += eps
                pred_minus = pred.copy()
                pred_minus[i, j] -= eps
                l_plus = loss.forward(pred_plus, true)
                l_minus = loss.forward(pred_minus, true)
                numerical[i, j] = (l_plus - l_minus) / (2 * eps)

        np.testing.assert_allclose(analytical, numerical, rtol=1e-4)


class TestSoftmaxCrossEntropy(unittest.TestCase):

    def test_backward_shape(self):
        loss = SoftmaxCrossEntropy()
        logits = np.random.randn(4, 3)
        targets = np.zeros((4, 3))
        targets[np.arange(4), np.random.randint(0, 3, 4)] = 1.0
        loss.forward(logits, targets)
        grad = loss.backward()
        self.assertEqual(grad.shape, logits.shape)


if __name__ == "__main__":
    unittest.main()

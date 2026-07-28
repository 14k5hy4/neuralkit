"""Tests for classification and regression metrics."""

import unittest

import numpy as np

from neuralkit.metrics import (
    accuracy, precision, recall, f1_score,
    confusion_matrix,
    mse, rmse, mae, r2_score,
)


class TestAccuracy(unittest.TestCase):

    def test_perfect(self):
        y = np.array([0, 1, 2, 0])
        self.assertEqual(accuracy(y, y), 1.0)

    def test_half(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.5)


class TestConfusionMatrix(unittest.TestCase):

    def test_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(cm, expected)

    def test_multiclass_shape(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape, (3, 3))
        # perfect predictions - diagonal should equal counts
        np.testing.assert_array_equal(np.diag(cm), [2, 2, 2])


class TestPrecisionRecall(unittest.TestCase):

    def test_perfect_precision(self):
        y_true = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(precision(y_true, y_true), 1.0)

    def test_perfect_recall(self):
        y_true = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(recall(y_true, y_true), 1.0)


class TestF1Score(unittest.TestCase):

    def test_perfect(self):
        y = np.array([0, 1, 2, 0, 1])
        self.assertAlmostEqual(f1_score(y, y), 1.0)

    def test_between_zero_and_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        score = f1_score(y_true, y_pred)
        self.assertGreater(score, 0)
        self.assertLess(score, 1)


class TestMSE(unittest.TestCase):

    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mse(y, y), 0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])  # error of 1 on last
        # MSE = (0 + 0 + 1) / 3
        self.assertAlmostEqual(mse(y_true, y_pred), 1.0 / 3.0)


class TestRMSE(unittest.TestCase):

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = np.sqrt(1.0 / 3.0)
        self.assertAlmostEqual(rmse(y_true, y_pred), expected)


class TestMAE(unittest.TestCase):

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        self.assertAlmostEqual(mae(y_true, y_pred), 0.5)


class TestR2Score(unittest.TestCase):

    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(r2_score(y, y), 1.0)

    def test_bad_model(self):
        """Predicting the mean should give R² = 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full_like(y_true, y_true.mean())
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.0)


if __name__ == "__main__":
    unittest.main()

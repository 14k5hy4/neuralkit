"""Tests for Sequential model."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from neuralkit.model import Sequential
from neuralkit.layers import Dense
from neuralkit.activations import ReLU, Sigmoid


class TestSequentialForwardBackward(unittest.TestCase):

    def test_forward_shape(self):
        model = Sequential([
            Dense(4, 8, activation=ReLU()),
            Dense(8, 3),
        ])
        x = np.random.randn(5, 4)
        out = model.forward(x)
        self.assertEqual(out.shape, (5, 3))

    def test_backward_runs(self):
        """Backward should run without errors and produce gradients."""
        model = Sequential([
            Dense(2, 4, activation=ReLU()),
            Dense(4, 1, activation=Sigmoid()),
        ])
        x = np.random.randn(3, 2)
        out = model.forward(x)
        grad = np.ones_like(out)
        # shouldn't throw
        model.backward(grad)
        # check gradients exist
        for layer in model.layers:
            if layer.params:
                for g in layer.grads.values():
                    self.assertIsNotNone(g)

    def test_predict_same_as_forward_eval(self):
        model = Sequential([
            Dense(3, 5, activation=ReLU()),
            Dense(5, 2),
        ])
        x = np.random.randn(4, 3)
        model.eval()
        expected = model.forward(x)
        result = model.predict(x)
        np.testing.assert_array_equal(result, expected)


class TestSequentialSaveLoad(unittest.TestCase):

    def setUp(self):
        self.tmpdir = os.path.join(tempfile.gettempdir(), "neuralkit_test_save")
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def tearDown(self):
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_save_load_roundtrip(self):
        np.random.seed(123)
        model = Sequential([
            Dense(4, 8, activation=ReLU()),
            Dense(8, 2),
        ])
        x = np.random.randn(3, 4)
        out_before = model.forward(x)

        model.save(self.tmpdir)

        loaded = Sequential.load(self.tmpdir)
        out_after = loaded.forward(x)

        np.testing.assert_allclose(out_before, out_after, atol=1e-10)

    def test_save_creates_files(self):
        model = Sequential([Dense(2, 3)])
        model.save(self.tmpdir)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "architecture.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "weights.npz")))


class TestSequentialSummary(unittest.TestCase):

    def test_summary_runs(self):
        model = Sequential([
            Dense(10, 5, activation=ReLU()),
            Dense(5, 1),
        ])
        # should print without error
        model.summary()


if __name__ == "__main__":
    unittest.main()

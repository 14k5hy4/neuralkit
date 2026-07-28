"""Tests for data utilities: DataLoader, transforms, splits."""

import unittest

import numpy as np

from neuralkit.data.loader import ArrayDataset, DataLoader
from neuralkit.data.transforms import StandardScaler, MinMaxScaler, OneHotEncoder
from neuralkit.data.splits import train_test_split, train_val_test_split
from neuralkit.data.cross_validation import k_fold_split


class TestArrayDataset(unittest.TestCase):

    def test_length(self):
        ds = ArrayDataset(np.zeros((10, 3)), np.zeros(10))
        self.assertEqual(len(ds), 10)

    def test_getitem(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        ds = ArrayDataset(X, y)
        x_i, y_i = ds[3]
        np.testing.assert_array_equal(x_i, [6, 7])
        self.assertEqual(y_i, 3)


class TestDataLoader(unittest.TestCase):

    def test_batch_sizes(self):
        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        ds = ArrayDataset(X, y)
        loader = DataLoader(ds, batch_size=3, shuffle=False, drop_last=False)
        batches = list(loader)
        # 10 / 3 = 3 full + 1 partial
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[-1][0].shape[0], 1)  # last batch has 1

    def test_drop_last(self):
        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        ds = ArrayDataset(X, y)
        loader = DataLoader(ds, batch_size=3, shuffle=False, drop_last=True)
        batches = list(loader)
        self.assertEqual(len(batches), 3)  # drops last batch of 1


class TestStandardScaler(unittest.TestCase):

    def test_fit_transform(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # mean should be ~0, std ~1
        np.testing.assert_allclose(X_scaled.mean(axis=0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), [1, 1], atol=1e-10)


class TestMinMaxScaler(unittest.TestCase):

    def test_range(self):
        X = np.array([[1.0], [5.0], [10.0]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        self.assertAlmostEqual(X_scaled.min(), 0.0)
        self.assertAlmostEqual(X_scaled.max(), 1.0)


class TestOneHotEncoder(unittest.TestCase):

    def test_encoding(self):
        y = np.array([0, 1, 2, 0])
        enc = OneHotEncoder()
        encoded = enc.fit_transform(y)
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ])
        np.testing.assert_array_equal(encoded, expected)


class TestTrainTestSplit(unittest.TestCase):

    def test_sizes(self):
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_seed=42,
        )
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)

    def test_no_overlap(self):
        X = np.arange(50).reshape(50, 1)
        y = np.arange(50)
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_seed=0)
        train_set = set(X_train.ravel())
        test_set = set(X_test.ravel())
        self.assertEqual(len(train_set & test_set), 0)


class TestKFoldSplit(unittest.TestCase):

    def test_fold_count(self):
        folds = k_fold_split(100, k=5)
        self.assertEqual(len(folds), 5)

    def test_no_overlap_in_folds(self):
        folds = k_fold_split(100, k=5, random_seed=42)
        for train_idx, val_idx in folds:
            overlap = set(train_idx) & set(val_idx)
            self.assertEqual(len(overlap), 0)

    def test_full_coverage(self):
        folds = k_fold_split(100, k=5, random_seed=42)
        all_val = np.concatenate([v for _, v in folds])
        self.assertEqual(len(set(all_val)), 100)


if __name__ == "__main__":
    unittest.main()

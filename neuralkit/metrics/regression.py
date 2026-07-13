"""Regression metrics: MSE, RMSE, MAE, R²."""

from __future__ import annotations
import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination).

    R² = 1 - SS_res / SS_tot
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # TODO: handle ss_tot == 0 edge case (constant target)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

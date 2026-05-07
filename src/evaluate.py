from __future__ import annotations

from typing import Dict

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
	denom = np.clip(np.abs(y_true), epsilon, None)
	return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	return {
		"rmse": rmse(y_true, y_pred),
		"mae": mae(y_true, y_pred),
		"mape": mape(y_true, y_pred),
	}

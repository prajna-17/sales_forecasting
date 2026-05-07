from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from keras.models import load_model


def _holiday_flag_for_date(date_value: pd.Timestamp) -> int:
	cal = USFederalHolidayCalendar()
	holidays = cal.holidays(start=date_value, end=date_value)
	return int(date_value in set(holidays))


def _xgb_recursive_forecast(model, history: List[float], start_date: pd.Timestamp, steps: int) -> np.ndarray:
	preds: List[float] = []
	history_buffer = list(history)
	for i in range(steps):
		current_date = start_date + pd.Timedelta(days=i)
		lag_1 = history_buffer[-1]
		lag_7 = history_buffer[-7] if len(history_buffer) >= 7 else history_buffer[-1]
		lag_30 = history_buffer[-30] if len(history_buffer) >= 30 else history_buffer[-1]
		last_7 = history_buffer[-7:] if len(history_buffer) >= 7 else history_buffer
		rolling_mean = float(np.mean(last_7))
		rolling_std = float(np.std(last_7))
		row = np.array(
			[[
				lag_1,
				lag_7,
				lag_30,
				rolling_mean,
				rolling_std,
				current_date.dayofweek,
				current_date.month,
				_holiday_flag_for_date(current_date),
			]]
		)
		pred = float(model.predict(row)[0])
		pred = max(0.0, pred)
		preds.append(pred)
		history_buffer.append(pred)
	return np.array(preds)


def _lstm_recursive_forecast(model, scaler, last_window: np.ndarray, steps: int) -> np.ndarray:
	preds_scaled: List[float] = []
	window = last_window.copy()
	for _ in range(steps):
		x = window.reshape(1, window.shape[0], 1)
		pred_scaled = float(model.predict(x, verbose=0)[0, 0])
		preds_scaled.append(pred_scaled)
		window = np.append(window[1:], pred_scaled)

	preds_array = np.array(preds_scaled).reshape(-1, 1)
	pred_unscaled = scaler.inverse_transform(preds_array).reshape(-1)
	pred_unscaled = np.maximum(0.0, pred_unscaled)
	return pred_unscaled


def load_registry(models_dir: str = "models") -> Dict:
	registry_path = Path(models_dir) / "model_registry.json"
	if not registry_path.exists():
		raise FileNotFoundError(f"Model registry not found: {registry_path}")

	with registry_path.open("r", encoding="utf-8") as f:
		return json.load(f)


def available_states(models_dir: str = "models") -> List[str]:
	registry = load_registry(models_dir=models_dir)
	return sorted(list(registry.get("states", {}).keys()))


def forecast_next_8_weeks(state: str, models_dir: str = "models") -> Tuple[pd.DataFrame, Dict]:
	registry = load_registry(models_dir=models_dir)
	states = registry.get("states", {})
	if state not in states:
		raise ValueError(
			f"State '{state}' not found in trained registry. Available: {sorted(states.keys())}"
		)

	state_meta = states[state]
	model_type = state_meta["best_model"]
	state_dir = Path(state_meta["state_dir"])

	history_df = pd.read_csv(state_meta["history_path"], parse_dates=["Date"]) 
	history_df = history_df.sort_values("Date")
	last_date = pd.to_datetime(history_df["Date"].iloc[-1])

	horizon_days = int(state_meta.get("horizon_days", 56))
	future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

	if model_type == "sarima":
		model = joblib.load(state_meta["best_model_path"])
		daily_pred = model.forecast(steps=horizon_days)
		daily_pred = np.maximum(0.0, np.array(daily_pred, dtype=float))
	elif model_type == "prophet":
		model = joblib.load(state_meta["best_model_path"])
		future_df = pd.DataFrame({"ds": future_dates})
		forecast_df = model.predict(future_df)
		daily_pred = np.maximum(0.0, forecast_df["yhat"].to_numpy(dtype=float))
	elif model_type == "xgboost":
		payload = joblib.load(state_meta["best_model_path"])
		model = payload["model"]
		history_values = history_df["Total"].tolist()
		daily_pred = _xgb_recursive_forecast(model, history_values, future_dates[0], horizon_days)
	elif model_type == "lstm":
		model = load_model(state_meta["best_model_path"])
		meta = joblib.load(state_meta["lstm_meta_path"])
		scaler = meta["scaler"]
		last_window = np.array(meta["last_window"], dtype=float)
		daily_pred = _lstm_recursive_forecast(model, scaler, last_window, horizon_days)
	else:
		raise ValueError(f"Unsupported model type: {model_type}")

	daily_forecast = pd.DataFrame({"Date": future_dates, "Forecast": daily_pred})
	weekly_forecast = (
		daily_forecast.set_index("Date")
		.resample("W-SUN")["Forecast"]
		.sum()
		.head(8)
		.reset_index()
		.rename(columns={"Date": "Week_Ending", "Forecast": "Predicted_Sales"})
	)

	meta = {
		"state": state,
		"best_model": model_type,
		"metrics": state_meta.get("metrics", {}),
	}
	return weekly_forecast, meta

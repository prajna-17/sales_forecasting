from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.layers import Dense, Input, LSTM
from keras.models import Sequential
from xgboost import XGBRegressor

from src.evaluate import evaluate_forecast
from src.preprocessing import load_and_prepare_data, time_train_validation_split

plt.switch_backend("Agg")


def _safe_clip(values: np.ndarray) -> np.ndarray:
	return np.maximum(0.0, np.array(values, dtype=float))


def _sarima_train_and_predict(train: pd.Series, val_horizon: int) -> Tuple[np.ndarray, object]:
	model = SARIMAX(
		train,
		order=(1, 1, 1),
		seasonal_order=(1, 1, 1, 7),
		enforce_stationarity=False,
		enforce_invertibility=False,
	)
	fit = model.fit(disp=False, maxiter=50)
	pred = fit.forecast(steps=val_horizon)
	return _safe_clip(pred), fit


def _prophet_train_and_predict(train_df: pd.DataFrame, val_horizon: int) -> Tuple[np.ndarray, Prophet]:
	frame = train_df[["Date", "Total"]].rename(columns={"Date": "ds", "Total": "y"})
	model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
	model.fit(frame)
	future = model.make_future_dataframe(periods=val_horizon, freq="D", include_history=False)
	pred = model.predict(future)["yhat"].to_numpy()
	return _safe_clip(pred), model


def _holiday_flag(date_value: pd.Timestamp) -> int:
	from pandas.tseries.holiday import USFederalHolidayCalendar

	cal = USFederalHolidayCalendar()
	holidays = cal.holidays(start=date_value, end=date_value)
	return int(date_value in set(holidays))


def _xgb_recursive_predict(model: XGBRegressor, history: List[float], start_date: pd.Timestamp, steps: int) -> np.ndarray:
	preds = []
	history_buffer = list(history)
	for i in range(steps):
		current_date = start_date + pd.Timedelta(days=i)
		lag_1 = history_buffer[-1]
		lag_7 = history_buffer[-7] if len(history_buffer) >= 7 else history_buffer[-1]
		lag_30 = history_buffer[-30] if len(history_buffer) >= 30 else history_buffer[-1]
		last_7 = history_buffer[-7:] if len(history_buffer) >= 7 else history_buffer
		rolling_mean = float(np.mean(last_7))
		rolling_std = float(np.std(last_7))
		x = np.array(
			[[
				lag_1,
				lag_7,
				lag_30,
				rolling_mean,
				rolling_std,
				current_date.dayofweek,
				current_date.month,
				_holiday_flag(current_date),
			]]
		)
		pred = float(model.predict(x)[0])
		pred = max(0.0, pred)
		preds.append(pred)
		history_buffer.append(pred)
	return np.array(preds, dtype=float)


def _xgb_prepare_supervised(state_df: pd.DataFrame) -> pd.DataFrame:
	work = state_df.copy().sort_values("Date")
	work["lag_1"] = work["Total"].shift(1)
	work["lag_7"] = work["Total"].shift(7)
	work["lag_30"] = work["Total"].shift(30)
	work["rolling_mean"] = work["Total"].shift(1).rolling(7).mean()
	work["rolling_std"] = work["Total"].shift(1).rolling(7).std()
	work["day_of_week"] = work["Date"].dt.dayofweek
	work["month"] = work["Date"].dt.month
	work["holiday_flag"] = work["Date"].apply(_holiday_flag)
	return work.dropna().reset_index(drop=True)


def _xgb_train_and_predict(train_df: pd.DataFrame, val_horizon: int) -> Tuple[np.ndarray, XGBRegressor]:
	supervised = _xgb_prepare_supervised(train_df)
	feature_cols = [
		"lag_1",
		"lag_7",
		"lag_30",
		"rolling_mean",
		"rolling_std",
		"day_of_week",
		"month",
		"holiday_flag",
	]

	x_train = supervised[feature_cols]
	y_train = supervised["Total"]

	model = XGBRegressor(
		n_estimators=120,
		learning_rate=0.05,
		max_depth=5,
		subsample=0.8,
		colsample_bytree=0.8,
		objective="reg:squarederror",
		random_state=42,
	)
	model.fit(x_train, y_train)

	history = train_df.sort_values("Date")["Total"].tolist()
	start_date = pd.to_datetime(train_df["Date"].max()) + pd.Timedelta(days=1)
	pred = _xgb_recursive_predict(model, history, start_date, val_horizon)
	return pred, model


def _make_lstm_sequences(values: np.ndarray, lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
	x, y = [], []
	for i in range(lookback, len(values)):
		x.append(values[i - lookback : i, 0])
		y.append(values[i, 0])
	if not x:
		return np.empty((0, lookback, 1)), np.empty((0,))
	x_arr = np.array(x).reshape(len(x), lookback, 1)
	y_arr = np.array(y)
	return x_arr, y_arr


def _lstm_recursive_predict(model: Sequential, scaler: MinMaxScaler, train_scaled: np.ndarray, steps: int, lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
	window = train_scaled[-lookback:].flatten()
	preds_scaled: List[float] = []

	for _ in range(steps):
		x = window.reshape(1, lookback, 1)
		pred_scaled = float(model.predict(x, verbose=0)[0, 0])
		preds_scaled.append(pred_scaled)
		window = np.append(window[1:], pred_scaled)

	pred_arr = np.array(preds_scaled).reshape(-1, 1)
	pred = scaler.inverse_transform(pred_arr).flatten()
	pred = np.maximum(0.0, pred)
	return pred, window


def _lstm_train_and_predict(train: pd.Series, val_horizon: int, lookback: int = 30) -> Tuple[np.ndarray, Sequential, MinMaxScaler, np.ndarray]:
	values = train.to_numpy(dtype=float).reshape(-1, 1)
	scaler = MinMaxScaler()
	scaled = scaler.fit_transform(values)
	lookback = min(lookback, max(2, len(values) - 1))

	x_train, y_train = _make_lstm_sequences(scaled, lookback=lookback)
	if x_train.shape[0] == 0:
		raise ValueError("Not enough data for LSTM lookback window.")

	model = Sequential(
		[
			Input(shape=(lookback, 1)),
			LSTM(64),
			Dense(32, activation="relu"),
			Dense(1),
		]
	)
	model.compile(optimizer="adam", loss="mse")
	model.fit(x_train, y_train, epochs=8, batch_size=32, verbose=0)

	pred, last_window = _lstm_recursive_predict(model, scaler, scaled, val_horizon, lookback=lookback)
	return pred, model, scaler, last_window


def _plot_validation(state: str, val_dates: pd.Series, actual: np.ndarray, predictions: Dict[str, np.ndarray], output_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(val_dates, actual, label="Actual", linewidth=2)
	for model_name, y_hat in predictions.items():
		ax.plot(val_dates, y_hat, label=model_name)
	ax.set_title(f"Validation Forecast Comparison - {state}")
	ax.set_xlabel("Date")
	ax.set_ylabel("Sales")
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path)
	plt.close(fig)


def _plot_future_forecast(state: str, daily_forecast: pd.DataFrame, output_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(daily_forecast["Date"], daily_forecast["Forecast"], label="Forecast", color="tab:orange")
	ax.set_title(f"Next 8 Weeks Forecast - {state}")
	ax.set_xlabel("Date")
	ax.set_ylabel("Predicted Sales")
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path)
	plt.close(fig)


def train_all_states(
	data_dir: str = "data",
	models_dir: str = "models",
	outputs_dir: str = "outputs",
	horizon_days: int = 56,
	dataset_path: str | None = None,
	target_state: str | None = None,
) -> Dict:
	models_path = Path(models_dir)
	outputs_path = Path(outputs_dir)
	models_path.mkdir(parents=True, exist_ok=True)
	outputs_path.mkdir(parents=True, exist_ok=True)

	data_df, data_file = load_and_prepare_data(data_dir=data_dir, dataset_path=dataset_path)
	if target_state:
		data_df = data_df[data_df["State"].astype(str).str.casefold() == target_state.casefold()].copy()
		if data_df.empty:
			raise ValueError(f"State '{target_state}' not found in the dataset.")

	comparison_rows = []
	registry = {
		"created_at": datetime.now(timezone.utc).isoformat(),
		"dataset": str(data_file),
		"horizon_days": horizon_days,
		"states": {},
	}

	for state, state_df in data_df.groupby("State"):
		state_df = state_df.sort_values("Date").reset_index(drop=True)
		state_dir = models_path / state.lower().replace(" ", "_")
		state_dir.mkdir(parents=True, exist_ok=True)

		train_df, val_df = time_train_validation_split(state_df, horizon_days=horizon_days)
		y_true = val_df["Total"].to_numpy(dtype=float)
		val_horizon = len(val_df)

		metrics_by_model: Dict[str, Dict[str, float]] = {}
		preds_by_model: Dict[str, np.ndarray] = {}
		trained_models = {}

		try:
			sarima_pred, sarima_fit = _sarima_train_and_predict(train_df["Total"], val_horizon)
			preds_by_model["sarima"] = sarima_pred
			metrics_by_model["sarima"] = evaluate_forecast(y_true, sarima_pred)
			trained_models["sarima"] = sarima_fit
		except Exception:
			metrics_by_model["sarima"] = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")}

		try:
			prophet_pred, prophet_model = _prophet_train_and_predict(train_df, val_horizon)
			preds_by_model["prophet"] = prophet_pred
			metrics_by_model["prophet"] = evaluate_forecast(y_true, prophet_pred)
			trained_models["prophet"] = prophet_model
		except Exception:
			metrics_by_model["prophet"] = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")}

		try:
			xgb_pred, xgb_model = _xgb_train_and_predict(train_df, val_horizon)
			preds_by_model["xgboost"] = xgb_pred
			metrics_by_model["xgboost"] = evaluate_forecast(y_true, xgb_pred)
			trained_models["xgboost"] = xgb_model
		except Exception:
			metrics_by_model["xgboost"] = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")}

		try:
			lstm_pred, lstm_model, lstm_scaler, lstm_last_window = _lstm_train_and_predict(train_df["Total"], val_horizon)
			preds_by_model["lstm"] = lstm_pred
			metrics_by_model["lstm"] = evaluate_forecast(y_true, lstm_pred)
			trained_models["lstm"] = {
				"model": lstm_model,
				"scaler": lstm_scaler,
				"last_window": lstm_last_window,
			}
		except Exception:
			metrics_by_model["lstm"] = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")}

		best_model = min(metrics_by_model, key=lambda name: metrics_by_model[name]["rmse"])

		for model_name, metric_values in metrics_by_model.items():
			comparison_rows.append(
				{
					"state": state,
					"model": model_name,
					"rmse": metric_values["rmse"],
					"mae": metric_values["mae"],
					"mape": metric_values["mape"],
					"is_best": model_name == best_model,
				}
			)

		# Refit best model on full data for production 8-week forecasting.
		best_model_path = ""
		lstm_meta_path = ""
		history_path = state_dir / "history.csv"
		state_df[["Date", "Total"]].to_csv(history_path, index=False)

		full_series = state_df["Total"]
		last_date = pd.to_datetime(state_df["Date"].max())
		future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

		def _forecast_from_trained_model(model_name: str) -> np.ndarray:
			if model_name not in trained_models:
				return np.repeat(float(full_series.iloc[-1]), horizon_days)
			if model_name == "sarima":
				return _safe_clip(trained_models["sarima"].forecast(steps=horizon_days))
			if model_name == "prophet":
				future_df = pd.DataFrame({"ds": future_dates})
				pred_df = trained_models["prophet"].predict(future_df)
				return _safe_clip(pred_df["yhat"].to_numpy())
			if model_name == "xgboost":
				history = state_df["Total"].tolist()
				return _xgb_recursive_predict(trained_models["xgboost"], history, future_dates[0], horizon_days)
			if model_name == "lstm":
				lstm_bundle = trained_models["lstm"]
				return _lstm_recursive_predict(
					lstm_bundle["model"],
					lstm_bundle["scaler"],
					np.asarray(lstm_bundle["last_window"]),
					horizon_days,
				)[0]
			raise ValueError(f"Unsupported fallback model: {model_name}")

		try:
			if best_model == "sarima":
				_, full_fit = _sarima_train_and_predict(full_series, val_horizon=horizon_days)
				best_model_path = str(state_dir / "sarima_best.pkl")
				joblib.dump(full_fit, best_model_path)
				daily_pred = _safe_clip(full_fit.forecast(steps=horizon_days))
			elif best_model == "prophet":
				_, full_prophet = _prophet_train_and_predict(state_df, val_horizon=horizon_days)
				best_model_path = str(state_dir / "prophet_best.pkl")
				joblib.dump(full_prophet, best_model_path)
				pred_df = full_prophet.predict(pd.DataFrame({"ds": future_dates}))
				daily_pred = _safe_clip(pred_df["yhat"].to_numpy())
			elif best_model == "xgboost":
				_, full_xgb = _xgb_train_and_predict(state_df, val_horizon=horizon_days)
				best_model_path = str(state_dir / "xgboost_best.pkl")
				joblib.dump({"model": full_xgb}, best_model_path)
				history = state_df["Total"].tolist()
				daily_pred = _xgb_recursive_predict(full_xgb, history, future_dates[0], horizon_days)
			else:
				_, full_lstm, full_scaler, last_window = _lstm_train_and_predict(state_df["Total"], val_horizon=horizon_days)
				best_model_path = str(state_dir / "lstm_best.keras")
				full_lstm.save(best_model_path)
				lstm_meta_path = str(state_dir / "lstm_meta.pkl")
				joblib.dump({"scaler": full_scaler, "last_window": last_window.tolist()}, lstm_meta_path)
				daily_pred = _lstm_recursive_predict(
					full_lstm,
					full_scaler,
					full_scaler.transform(state_df["Total"].to_numpy(dtype=float).reshape(-1, 1)),
					horizon_days,
				)[0]
		except Exception:
			print(f"[WARN] Full retrain failed for {state}; using validation-trained {best_model} artifact.")
			best_model_path = str(state_dir / f"{best_model}_best_fallback")
			daily_pred = _forecast_from_trained_model(best_model)
			if best_model == "lstm":
				lstm_bundle = trained_models["lstm"]
				best_model_path = str(state_dir / "lstm_best_fallback.keras")
				lstm_bundle["model"].save(best_model_path)
				lstm_meta_path = str(state_dir / "lstm_meta.pkl")
				joblib.dump({"scaler": lstm_bundle["scaler"], "last_window": lstm_bundle["last_window"].tolist()}, lstm_meta_path)
			elif best_model == "sarima":
				best_model_path = str(state_dir / "sarima_best_fallback.pkl")
				joblib.dump(trained_models["sarima"], best_model_path)
			elif best_model == "prophet":
				best_model_path = str(state_dir / "prophet_best_fallback.pkl")
				joblib.dump(trained_models["prophet"], best_model_path)
			elif best_model == "xgboost":
				best_model_path = str(state_dir / "xgboost_best_fallback.pkl")
				joblib.dump({"model": trained_models["xgboost"]}, best_model_path)

		daily_forecast = pd.DataFrame({"Date": future_dates, "Forecast": daily_pred})
		weekly_forecast = (
			daily_forecast.set_index("Date")
			.resample("W-SUN")["Forecast"]
			.sum()
			.head(8)
			.reset_index()
			.rename(columns={"Date": "Week_Ending", "Forecast": "Predicted_Sales"})
		)

		weekly_path = outputs_path / f"{state.lower().replace(' ', '_')}_next_8_weeks.csv"
		weekly_forecast.to_csv(weekly_path, index=False)

		comparison_plot = outputs_path / f"{state.lower().replace(' ', '_')}_validation_compare.png"
		_plot_validation(state, val_df["Date"], y_true, preds_by_model, comparison_plot)

		forecast_plot = outputs_path / f"{state.lower().replace(' ', '_')}_future_forecast.png"
		_plot_future_forecast(state, daily_forecast, forecast_plot)

		registry["states"][state] = {
			"best_model": best_model,
			"best_model_path": best_model_path,
			"lstm_meta_path": lstm_meta_path,
			"metrics": metrics_by_model,
			"state_dir": str(state_dir),
			"history_path": str(history_path),
			"horizon_days": horizon_days,
			"weekly_forecast_path": str(weekly_path),
		}

	comparison_df = pd.DataFrame(comparison_rows)
	comparison_out = outputs_path / "model_comparison.csv"
	comparison_df.to_csv(comparison_out, index=False)

	registry_path = models_path / "model_registry.json"
	with registry_path.open("w", encoding="utf-8") as f:
		json.dump(registry, f, indent=2)

	return {
		"registry_path": str(registry_path),
		"comparison_path": str(comparison_out),
		"states": sorted(registry["states"].keys()),
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train time series models for state-level sales forecasting")
	parser.add_argument("--data-dir", type=str, default="data")
	parser.add_argument("--dataset-path", type=str, default=None)
	parser.add_argument("--models-dir", type=str, default="models")
	parser.add_argument("--outputs-dir", type=str, default="outputs")
	parser.add_argument("--horizon-days", type=int, default=56)
	parser.add_argument("--state", type=str, default=None, help="Optional single-state training mode for faster demo runs")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	result = train_all_states(
		data_dir=args.data_dir,
		models_dir=args.models_dir,
		outputs_dir=args.outputs_dir,
		horizon_days=args.horizon_days,
		dataset_path=args.dataset_path,
		target_state=args.state,
	)
	print(json.dumps(result, indent=2))

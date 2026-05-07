# State-Level Time Series Forecasting System

Production-style end-to-end forecasting backend for predicting the next 8 weeks of sales for each state.

## Problem Statement

- Forecast target: `Total`
- Forecast granularity: `State + Date`
- Forecast horizon: next 8 weeks (56 days aggregated weekly)
- Dataset columns expected:
  - `State`
  - `Date`
  - `Total`
  - `Category`

## Project Structure

project/

- data/
- notebooks/
- models/
- outputs/
- app/
  - main.py
  - predict.py
  - utils.py
- src/
  - preprocessing.py
  - feature_engineering.py
  - train_models.py
  - evaluate.py
  - forecasting.py
- requirements.txt
- README.md
- .gitignore

## What This System Does

1. Loads Excel/CSV data and validates schema.
2. Cleans and aggregates sales by `State + Date`.
3. Handles missing dates via daily reindexing per state.
4. Handles missing values using interpolation + forward/backward fill.
5. Trains and compares 4 models per state:
   - SARIMA
   - Prophet
   - XGBoost (with lag and calendar features)
   - LSTM (TensorFlow/Keras)
6. Evaluates with RMSE, MAE, MAPE on strict time-based validation split.
7. Auto-selects best model per state using lowest RMSE.
8. Saves model artifacts and registry for serving.
9. Exposes REST API with Swagger docs.
10. Saves plots and comparison outputs.

## Feature Engineering

Implemented engineered features:

- `lag_1`
- `lag_7`
- `lag_30`
- `rolling_mean`
- `rolling_std`
- `day_of_week`
- `month`
- `holiday_flag`

## Setup

### 1) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## Train Models

Full training for all states:

```powershell
python -m src.train_models --data-dir data --models-dir models --outputs-dir outputs --horizon-days 56
```

Fast demo training for a single state:

```powershell
python -m src.train_models --data-dir data --models-dir models --outputs-dir outputs --horizon-days 56 --state Texas
```

Notes:

- The trainer automatically picks the first Excel/CSV file in `data/` if `--dataset-path` is not supplied.
- You can provide a specific file path:

```powershell
python -m src.train_models --data-dir data --dataset-path "Forecasting Case- Study.xlsx - Sheet1.csv"
```

## Run API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

- `http://127.0.0.1:8000/docs`

## API Endpoints

- `GET /health`
- `GET /states`
- `GET /forecast?state=Texas`

### Example Request

```http
GET /forecast?state=Texas
```

### Example Response

```json
{
  "state": "Texas",
  "best_model": "xgboost",
  "metrics": {
    "sarima": { "rmse": 132.8, "mae": 101.1, "mape": 12.4 },
    "prophet": { "rmse": 128.2, "mae": 95.3, "mape": 11.8 },
    "xgboost": { "rmse": 121.5, "mae": 89.9, "mape": 10.9 },
    "lstm": { "rmse": 140.7, "mae": 105.2, "mape": 13.7 }
  },
  "forecast_next_8_weeks": [
    { "week_ending": "2026-05-10", "predicted_sales": 8123.41 },
    { "week_ending": "2026-05-17", "predicted_sales": 8450.11 }
  ]
}
```

## Generated Outputs

After training:

- `models/model_registry.json`
- `models/<state_slug>/...` model artifacts
- `outputs/model_comparison.csv`
- `outputs/<state>_validation_compare.png`
- `outputs/<state>_future_forecast.png`
- `outputs/<state>_next_8_weeks.csv`

## Notebook

Notebook path:

- `notebooks/state_sales_forecasting.ipynb`

Run cells sequentially:

1. Imports
2. Load and inspect data
3. Train all state models
4. Review model comparison
5. Generate sample 8-week forecast

## Design Notes

- Validation split is strictly temporal (last 56 days as validation) to avoid leakage.
- Best model is selected independently for each state.
- Forecasting endpoint reads persisted model registry and artifacts.
- Service is stateless at runtime.

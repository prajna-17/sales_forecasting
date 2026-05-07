from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from app.predict import get_state_forecast, list_states

app = FastAPI(
	title="State Sales Forecast API",
	description="Forecast next 8 weeks of state-level sales from historical data.",
	version="1.0.0",
)


@app.get("/")
def root() -> dict:
	return {
		"message": "State-level time series forecasting service is running.",
		"docs": "/docs",
	}


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.get("/states")
def states() -> dict:
	try:
		available = list_states(models_dir="models")
		return {"count": len(available), "states": available}
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/forecast")
def forecast(state: str = Query(..., description="State name, for example Texas")) -> dict:
	try:
		return get_state_forecast(state=state, models_dir="models")
	except ValueError as exc:
		raise HTTPException(status_code=404, detail=str(exc)) from exc
	except FileNotFoundError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

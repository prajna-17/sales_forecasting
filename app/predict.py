from __future__ import annotations

from typing import Dict, List

from app.utils import normalize_state_name, read_registry
from src.forecasting import forecast_next_8_weeks


def list_states(models_dir: str = "models") -> List[str]:
	registry = read_registry(models_dir=models_dir)
	return sorted(list(registry.get("states", {}).keys()))


def get_state_forecast(state: str, models_dir: str = "models") -> Dict:
	clean_state = normalize_state_name(state)
	try:
		forecast_df, meta = forecast_next_8_weeks(clean_state, models_dir=models_dir)
	except (FileNotFoundError, ValueError):
		from src.train_models import train_all_states

		train_all_states(data_dir="data", models_dir=models_dir, outputs_dir="outputs", horizon_days=56, target_state=clean_state)
		forecast_df, meta = forecast_next_8_weeks(clean_state, models_dir=models_dir)

	records = [
		{
			"week_ending": str(row["Week_Ending"].date()),
			"predicted_sales": float(row["Predicted_Sales"]),
		}
		for _, row in forecast_df.iterrows()
	]

	return {
		"state": meta["state"],
		"best_model": meta["best_model"],
		"metrics": meta["metrics"],
		"forecast_next_8_weeks": records,
	}

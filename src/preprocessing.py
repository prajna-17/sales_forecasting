from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

REQUIRED_COLUMNS = ["State", "Date", "Total", "Category"]


def _resolve_dataset_path(data_dir: Path, dataset_path: Optional[str] = None) -> Path:
	if dataset_path:
		candidate = Path(dataset_path)
		if not candidate.is_absolute():
			candidate = data_dir / candidate
		if not candidate.exists():
			raise FileNotFoundError(f"Dataset not found: {candidate}")
		return candidate

	patterns = ["*.xlsx", "*.xls", "*.csv"]
	for pattern in patterns:
		files = sorted(data_dir.glob(pattern))
		if files:
			return files[0]

	raise FileNotFoundError(
		f"No dataset found in {data_dir}. Provide an Excel/CSV file with required columns."
	)


def load_raw_dataset(data_dir: str = "data", dataset_path: Optional[str] = None) -> Tuple[pd.DataFrame, Path]:
	data_folder = Path(data_dir)
	data_file = _resolve_dataset_path(data_folder, dataset_path)

	if data_file.suffix.lower() in {".xlsx", ".xls"}:
		df = pd.read_excel(data_file)
	else:
		df = pd.read_csv(data_file)

	missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
	if missing:
		raise ValueError(f"Dataset is missing required columns: {missing}")

	return df, data_file


def preprocess_sales_data(df: pd.DataFrame) -> pd.DataFrame:
	work = df.copy()
	work["State"] = work["State"].astype(str).str.strip()
	work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
	work["Total"] = pd.to_numeric(work["Total"], errors="coerce")

	work = work.dropna(subset=["State", "Date"])
	work["Total"] = work["Total"].fillna(0.0)

	# State + Date level aggregation is mandatory for this assignment.
	aggregated = (
		work.groupby(["State", "Date"], as_index=False)["Total"]
		.sum()
		.sort_values(["State", "Date"])
	)

	all_states = []
	for state, state_df in aggregated.groupby("State"):
		state_df = state_df.sort_values("Date").set_index("Date")
		full_range = pd.date_range(state_df.index.min(), state_df.index.max(), freq="D")
		state_filled = state_df.reindex(full_range)
		state_filled.index.name = "Date"
		state_filled["State"] = state

		# Interpolate and then fill edge gaps to handle missing values robustly.
		state_filled["Total"] = (
			state_filled["Total"]
			.interpolate(method="time")
			.ffill()
			.bfill()
			.fillna(0.0)
		)
		all_states.append(state_filled.reset_index())

	clean = pd.concat(all_states, ignore_index=True)
	clean["Category"] = "ALL"
	clean = clean[["State", "Date", "Total", "Category"]].sort_values(["State", "Date"])
	return clean.reset_index(drop=True)


def load_and_prepare_data(data_dir: str = "data", dataset_path: Optional[str] = None) -> Tuple[pd.DataFrame, Path]:
	raw_df, data_file = load_raw_dataset(data_dir=data_dir, dataset_path=dataset_path)
	clean_df = preprocess_sales_data(raw_df)
	return clean_df, data_file


def time_train_validation_split(state_df: pd.DataFrame, horizon_days: int = 56) -> Tuple[pd.DataFrame, pd.DataFrame]:
	ordered = state_df.sort_values("Date").reset_index(drop=True)
	if len(ordered) <= horizon_days:
		raise ValueError(
			f"Not enough rows ({len(ordered)}) for horizon {horizon_days}."
		)

	split_idx = len(ordered) - horizon_days
	train_df = ordered.iloc[:split_idx].copy()
	val_df = ordered.iloc[split_idx:].copy()
	return train_df, val_df

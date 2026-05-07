from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

LAG_FEATURES = ["lag_1", "lag_7", "lag_30"]
ROLLING_FEATURES = ["rolling_mean", "rolling_std"]
CALENDAR_FEATURES = ["day_of_week", "month", "holiday_flag"]
FEATURE_COLUMNS = LAG_FEATURES + ROLLING_FEATURES + CALENDAR_FEATURES


def _holiday_series(date_index: pd.Series) -> pd.Series:
	calendar = USFederalHolidayCalendar()
	holidays = calendar.holidays(start=date_index.min(), end=date_index.max())
	return date_index.isin(holidays).astype(int)


def create_features(df: pd.DataFrame, target_col: str = "Total") -> pd.DataFrame:
	work = df.copy().sort_values("Date")

	work["lag_1"] = work[target_col].shift(1)
	work["lag_7"] = work[target_col].shift(7)
	work["lag_30"] = work[target_col].shift(30)
	work["rolling_mean"] = work[target_col].shift(1).rolling(window=7).mean()
	work["rolling_std"] = work[target_col].shift(1).rolling(window=7).std()

	work["day_of_week"] = work["Date"].dt.dayofweek
	work["month"] = work["Date"].dt.month
	work["holiday_flag"] = _holiday_series(work["Date"])

	return work

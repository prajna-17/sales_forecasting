from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def read_registry(models_dir: str = "models") -> Dict:
	registry_path = Path(models_dir) / "model_registry.json"
	if not registry_path.exists():
		raise FileNotFoundError(
			f"Model registry not found at {registry_path}. Run training first."
		)

	with registry_path.open("r", encoding="utf-8") as f:
		return json.load(f)


def normalize_state_name(state: str) -> str:
	return state.strip()

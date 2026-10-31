from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import Settings, get_settings
from app.data import FEATURE_COLUMNS


class ModelNotFoundError(FileNotFoundError):
    pass


@lru_cache
def load_bundle(model_path: str) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise ModelNotFoundError(
            f"Model not found at {model_path}. Run: python scripts/train_model.py"
        )
    return joblib.load(path)


def predict_units(payload: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or get_settings()
    bundle = load_bundle(cfg.model_path)
    model = bundle["model"]
    features = bundle.get("features", FEATURE_COLUMNS)

    missing = [col for col in features if col not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    frame = pd.DataFrame([{col: payload[col] for col in features}])
    prediction = float(model.predict(frame)[0])
    return {
        "predicted_units": round(prediction, 2),
        "features_used": features,
        "model_path": cfg.model_path,
    }

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import Settings
from app.data import FEATURE_COLUMNS, generate_raw_orders, prepare_and_engineer, save_to_sqlite


def train_and_persist(settings: Settings) -> dict:
    raw = generate_raw_orders(seed=settings.random_seed)
    features = prepare_and_engineer(raw)
    save_to_sqlite(features, settings.database_url)

    x = features[FEATURE_COLUMNS]
    y = features["units_sold"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=False
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                GradientBoostingRegressor(random_state=settings.random_seed),
            ),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "features": FEATURE_COLUMNS,
    }

    Path(settings.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, settings.model_path)
    Path(settings.metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

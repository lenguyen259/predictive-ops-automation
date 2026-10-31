from pathlib import Path

from fastapi.testclient import TestClient

from app.config import get_settings
from app.data import FEATURE_COLUMNS, generate_raw_orders, prepare_and_engineer
from app.main import app
from app.predict import load_bundle
from app.train import train_and_persist


def test_feature_engineering_creates_expected_columns():
    raw = generate_raw_orders(n_days=60, seed=1)
    features = prepare_and_engineer(raw)
    for col in FEATURE_COLUMNS + ["units_sold"]:
        assert col in features.columns
    assert len(features) < len(raw)  # lag/rolling drop early rows
    assert features["units_sold"].min() >= 0


def test_train_predict_api(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    db_path = tmp_path / "data.db"

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    get_settings.cache_clear()
    load_bundle.cache_clear()

    settings = get_settings()
    metrics = train_and_persist(settings)
    assert metrics["mae"] >= 0
    assert Path(settings.model_path).exists()

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    predict = client.post(
        "/predict",
        json={
            "lag_1": 90,
            "lag_7": 85,
            "rolling_mean_7": 88,
            "rolling_std_7": 5,
            "promo_flag": 1,
            "price": 8.9,
            "day_of_week": 2,
            "month": 6,
        },
    )
    assert predict.status_code == 200, predict.text
    assert predict.json()["predicted_units"] > 0

    get_settings.cache_clear()
    load_bundle.cache_clear()

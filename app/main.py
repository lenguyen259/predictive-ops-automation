import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException

from app.config import Settings, get_settings
from app.predict import ModelNotFoundError, load_bundle, predict_units
from app.schemas import HealthResponse, MetricsResponse, PredictRequest, PredictResponse
from app.train import train_and_persist

app = FastAPI(
    title="Predictive Ops Automation",
    description=(
        "Data preparation, feature engineering, model training, testing, and "
        "REST deployment for demand forecasting / ops automation."
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    model_loaded = Path(settings.model_path).exists()
    return HealthResponse(status="ok", app=settings.app_name, model_loaded=model_loaded)


@app.post("/train", response_model=MetricsResponse)
def train(settings: Annotated[Settings, Depends(get_settings)]) -> MetricsResponse:
    metrics = train_and_persist(settings)
    load_bundle.cache_clear()
    return MetricsResponse(metrics=metrics)


@app.get("/metrics", response_model=MetricsResponse)
def metrics(settings: Annotated[Settings, Depends(get_settings)]) -> MetricsResponse:
    path = Path(settings.metrics_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    return MetricsResponse(metrics=json.loads(path.read_text(encoding="utf-8")))


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PredictResponse:
    try:
        result = predict_units(body.model_dump(), settings)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse.model_validate(result)


@app.get("/features")
def list_features() -> dict[str, Any]:
    from app.data import FEATURE_COLUMNS

    return {
        "features": FEATURE_COLUMNS,
        "description": {
            "lag_1": "Units sold yesterday",
            "lag_7": "Units sold 7 days ago",
            "rolling_mean_7": "7-day trailing mean",
            "rolling_std_7": "7-day trailing std",
            "promo_flag": "Promotion active (0/1)",
            "price": "Unit price",
            "day_of_week": "Monday=0 ... Sunday=6",
            "month": "Calendar month",
        },
    }

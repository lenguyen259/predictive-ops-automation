from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    model_loaded: bool


class PredictRequest(BaseModel):
    lag_1: float
    lag_7: float
    rolling_mean_7: float
    rolling_std_7: float
    promo_flag: int = Field(ge=0, le=1)
    price: float
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)


class PredictResponse(BaseModel):
    predicted_units: float
    features_used: list[str]
    model_path: str


class MetricsResponse(BaseModel):
    metrics: dict[str, Any]

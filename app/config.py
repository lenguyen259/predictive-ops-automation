from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "predictive-ops-automation"
    database_url: str = "sqlite:///./predictive_ops.db"
    model_path: str = "artifacts/demand_model.joblib"
    metrics_path: str = "artifacts/metrics.json"
    random_seed: int = 42


@lru_cache
def get_settings() -> Settings:
    return Settings()

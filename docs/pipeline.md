# Predictive Ops Pipeline

## Stages

1. **Data preparation** — generate/clean order history, clip invalid values, fill missing prices
2. **Feature engineering** — lags, rolling stats, calendar and promo/price features
3. **Model development** — `StandardScaler` + `GradientBoostingRegressor`
4. **Testing** — unit tests for features + API contract tests for `/predict`
5. **Deployment** — FastAPI REST service + Docker image

## SQL persistence

Engineered features are written to SQLite (`orders_features`) so downstream automation jobs can query with SQL.

## Monitoring hooks

- `GET /health` reports whether the model artifact is loaded
- `GET /metrics` exposes offline MAE / RMSE / R² after training
- Retrain via `POST /train` or `python scripts/train_model.py`

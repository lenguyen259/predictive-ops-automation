# Predictive Ops Automation

End-to-end **data science + automation** service for demand forecasting:

prepare data → engineer features → train model → validate → deploy REST API.

Maps directly to AI & Automation Engineer expectations around predictive models, SQL-backed data workflows, testing, and deployment.

## Quick start

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python scripts/train_model.py
uvicorn app.main:app --reload
pytest
```

## API

- `GET /health`
- `POST /train`
- `GET /metrics`
- `GET /features`
- `POST /predict`

Example prediction body:

```json
{
  "lag_1": 90,
  "lag_7": 85,
  "rolling_mean_7": 88,
  "rolling_std_7": 5,
  "promo_flag": 1,
  "price": 8.9,
  "day_of_week": 2,
  "month": 6
}
```

## Stack

Python, pandas, scikit-learn, SQLAlchemy/SQLite, FastAPI, Pytest, Docker.

## Docs

See [docs/pipeline.md](docs/pipeline.md).

## License

MIT

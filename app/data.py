"""Synthetic operational dataset + SQL-friendly persistence helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "promo_flag",
    "price",
    "day_of_week",
    "month",
]


def generate_raw_orders(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    base = 80 + 15 * np.sin(np.arange(n_days) / 7 * 2 * np.pi)
    promo = rng.binomial(1, 0.18, size=n_days)
    price = 9.5 + rng.normal(0, 0.4, size=n_days) - 0.8 * promo
    noise = rng.normal(0, 6, size=n_days)
    demand = np.clip(base + 18 * promo - 3 * (price - 9.5) + noise, 10, None)

    return pd.DataFrame(
        {
            "order_date": dates,
            "sku": "SKU-DEMO-01",
            "promo_flag": promo.astype(int),
            "price": np.round(price, 2),
            "units_sold": np.round(demand, 0).astype(int),
        }
    )


def prepare_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean + feature-engineer a demand forecasting frame."""
    frame = df.copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"])
    frame = frame.sort_values("order_date").drop_duplicates(["order_date", "sku"])
    frame["units_sold"] = frame["units_sold"].clip(lower=0)
    frame["price"] = frame["price"].fillna(frame["price"].median())
    frame["promo_flag"] = frame["promo_flag"].fillna(0).astype(int)

    frame["lag_1"] = frame["units_sold"].shift(1)
    frame["lag_7"] = frame["units_sold"].shift(7)
    frame["rolling_mean_7"] = frame["units_sold"].shift(1).rolling(7).mean()
    frame["rolling_std_7"] = frame["units_sold"].shift(1).rolling(7).std()
    frame["day_of_week"] = frame["order_date"].dt.dayofweek
    frame["month"] = frame["order_date"].dt.month
    frame = frame.dropna().reset_index(drop=True)
    return frame


def save_to_sqlite(df: pd.DataFrame, db_path: str, table: str = "orders_features") -> None:
    path = Path(db_path.replace("sqlite:///", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)


def load_feature_frame_from_sql(db_path: str, table: str = "orders_features") -> pd.DataFrame:
    path = Path(db_path.replace("sqlite:///", ""))
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)

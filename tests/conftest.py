"""Shared test fixtures for FinAI test suite.

Provides synthetic data fixtures that all test modules can reuse. Every
fixture is deterministic (seeded) and runs without external services.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ #
#  Single-stock OHLCV price data
# ------------------------------------------------------------------ #

@pytest.fixture()
def synthetic_price_df() -> pd.DataFrame:
    """Single stock OHLCV DataFrame with DatetimeIndex (300 business days).

    Columns: open, high, low, close, volume.
    """
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)

    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "open": price + np.random.randn(n) * 0.2,
            "high": price + abs(np.random.randn(n)) * 1.0,
            "low": price - abs(np.random.randn(n)) * 1.0,
            "close": price,
            "volume": np.random.randint(1000, 50000, size=n).astype(float),
        },
        index=dates,
    )
    return df


# ------------------------------------------------------------------ #
#  Multi-stock price data (flat format used by backtest engine)
# ------------------------------------------------------------------ #

@pytest.fixture()
def synthetic_multi_stock_prices() -> pd.DataFrame:
    """Multi-stock flat price DataFrame with columns:
    date, stock_id, open, high, low, close, volume.

    5 stocks x 120 business days.
    """
    np.random.seed(42)
    n_days = 120
    n_stocks = 5
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    records = []
    for i in range(n_stocks):
        sid = f"{2300 + i}"
        base = np.random.uniform(100, 500)
        rets = np.random.normal(0.0003, 0.015, n_days)
        closes = base * np.cumprod(1 + rets)
        for j, d in enumerate(dates):
            o = closes[j] * (1 + np.random.normal(0, 0.003))
            h = max(o, closes[j]) * 1.005
            lo = min(o, closes[j]) * 0.995
            records.append({
                "date": d,
                "stock_id": sid,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(lo, 2),
                "close": round(closes[j], 2),
                "volume": int(np.random.uniform(1e6, 1e7)),
            })
    return pd.DataFrame(records)


# ------------------------------------------------------------------ #
#  Feature data from FeatureEngine
# ------------------------------------------------------------------ #

@pytest.fixture()
def synthetic_features(synthetic_price_df) -> pd.DataFrame:
    """Technical features computed from synthetic_price_df via FeatureEngine.

    Returns the feature-only columns (no raw OHLCV), with NaN rows dropped.
    """
    from data.processors.features import FeatureEngine

    engine = FeatureEngine()
    feats = engine.compute_all(synthetic_price_df)
    return feats.dropna()


# ------------------------------------------------------------------ #
#  Trained models (Ridge + RF + XGBoost + Ensemble)
# ------------------------------------------------------------------ #

@pytest.fixture()
def trained_models(synthetic_features) -> dict:
    """Dict of model_name -> fitted model, trained on synthetic features.

    Uses a simple random target with embedded signal for reproducibility.
    """
    from models.baseline import RidgePredictor
    from models.ensemble import EnsemblePredictor
    from models.tree_models import RandomForestPredictor, XGBoostPredictor

    np.random.seed(42)
    X = synthetic_features
    # Create target with weak signal for stable training
    y = pd.Series(
        X.iloc[:, 0] * 0.1 + np.random.randn(len(X)) * 0.5,
        index=X.index,
        name="target",
    )

    ridge = RidgePredictor(alpha=1.0)
    ridge.fit(X, y)

    rf = RandomForestPredictor(params={
        "n_estimators": 50, "max_depth": 5, "random_state": 42, "n_jobs": 1,
    })
    rf.fit(X, y)

    xgb = XGBoostPredictor(params={
        "n_estimators": 50, "max_depth": 4, "learning_rate": 0.1, "random_state": 42,
    })
    xgb.fit(X, y)

    ensemble = EnsemblePredictor(
        models={"random_forest": rf, "xgboost": xgb},
        weights={"random_forest": 0.4, "xgboost": 0.6},
    )
    ensemble._is_fitted = True

    return {
        "ridge": ridge,
        "random_forest": rf,
        "xgboost": xgb,
        "ensemble": ensemble,
    }


# ------------------------------------------------------------------ #
#  Mock Supabase client
# ------------------------------------------------------------------ #

@pytest.fixture()
def mock_supabase_client() -> MagicMock:
    """Mock SupabaseLoader that returns empty data for all queries.

    Useful for testing code that depends on SupabaseLoader without
    connecting to a real database.
    """
    loader = MagicMock()

    loader.get_prices.return_value = pd.DataFrame()
    loader.get_features.return_value = pd.DataFrame()
    loader.get_latest_predictions.return_value = pd.DataFrame()
    loader.get_backtest_history.return_value = []
    loader.get_active_model.return_value = None

    empty_resp = MagicMock()
    empty_resp.data = []
    loader.client.table.return_value.select.return_value.execute.return_value = empty_resp

    return loader


# ------------------------------------------------------------------ #
#  Benchmark data for backtest
# ------------------------------------------------------------------ #

@pytest.fixture()
def synthetic_benchmark(synthetic_multi_stock_prices) -> pd.DataFrame:
    """Benchmark price DataFrame with columns: date, close.

    Covers the same date range as synthetic_multi_stock_prices.
    """
    np.random.seed(99)
    dates = sorted(synthetic_multi_stock_prices["date"].unique())
    n = len(dates)
    bench_close = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n))
    return pd.DataFrame({"date": dates, "close": bench_close})

"""Backtest endpoints."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import get_db
from data.loaders import DatabaseLoader

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# In-memory task progress tracking
_backtest_status: dict[str, dict] = {}


# ------------------------------------------------------------------ #
#  Response models
# ------------------------------------------------------------------ #

class BacktestSummary(BaseModel):
    id: str
    run_date: Optional[str] = None
    model_type: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    metrics: Optional[dict] = None
    config: Optional[dict] = None
    created_at: Optional[str] = None


class BacktestListResponse(BaseModel):
    results: list[BacktestSummary]
    message: Optional[str] = None


class BacktestRunRequest(BaseModel):
    model_type: str = "ensemble"
    mode: Literal["run", "walk_forward"] = "run"
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    initial_capital: float = 10_000_000
    config: Optional[dict] = None


class BacktestRunResponse(BaseModel):
    status: str
    message: str
    task_id: Optional[str] = None


class BacktestTaskStatus(BaseModel):
    task_id: str
    status: str  # pending / running / success / failed
    progress: Optional[str] = None
    error: Optional[str] = None


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #

@router.get("/results", response_model=BacktestListResponse)
def list_backtest_results(
    limit: int = Query(10, ge=1, le=50),
    db: Optional[DatabaseLoader] = Depends(get_db),
) -> BacktestListResponse:
    """List recent backtest results."""
    if db is None:
        return BacktestListResponse(results=[], message="Supabase not configured")

    try:
        data = db.get_backtest_history(limit=limit)
        results = [BacktestSummary(**r) for r in data]
        return BacktestListResponse(results=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status/{task_id}", response_model=BacktestTaskStatus)
def get_backtest_status(task_id: str) -> BacktestTaskStatus:
    """Get the progress of a running backtest task."""
    info = _backtest_status.get(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return BacktestTaskStatus(task_id=task_id, **info)


@router.get("/{result_id}", response_model=BacktestSummary)
def get_backtest_result(
    result_id: str,
    db: Optional[DatabaseLoader] = Depends(get_db),
) -> BacktestSummary:
    """Get a specific backtest result by ID."""
    if db is None:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    try:
        resp = (
            db.client.table("backtest_results")
            .select("*")
            .eq("id", result_id)
            .execute()
        )
        if not resp.data:
            raise HTTPException(status_code=404, detail="Backtest result not found")
        return BacktestSummary(**resp.data[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{result_id}")
def delete_backtest_result(
    result_id: str,
    db: Optional[DatabaseLoader] = Depends(get_db),
) -> dict:
    """Delete a backtest result by ID."""
    if db is None:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    try:
        db.delete_backtest(result_id)
        return {"status": "ok", "message": f"Deleted {result_id}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/run", response_model=BacktestRunResponse)
def run_backtest(
    request: BacktestRunRequest,
    background_tasks: BackgroundTasks,
    db: Optional[DatabaseLoader] = Depends(get_db),
) -> BacktestRunResponse:
    """Trigger a new backtest run (async)."""
    if db is None:
        return BacktestRunResponse(
            status="error", message="Supabase not configured — cannot run backtest"
        )

    task_id = str(uuid.uuid4())
    _backtest_status[task_id] = {
        "status": "pending",
        "progress": "排程中...",
        "error": None,
    }

    background_tasks.add_task(_execute_backtest, db, request, task_id)
    return BacktestRunResponse(
        status="accepted",
        message=f"Backtest for {request.model_type} scheduled",
        task_id=task_id,
    )



def _update_progress(task_id: str | None, status: str, progress: str, error: str | None = None) -> None:
    """Update in-memory backtest task progress."""
    if task_id and task_id in _backtest_status:
        _backtest_status[task_id] = {
            "status": status,
            "progress": progress,
            "error": error,
        }


def _execute_backtest(db: DatabaseLoader, request: BacktestRunRequest, task_id: str | None = None) -> None:
    """Background task that runs a backtest and saves results.

    Loads price data and predictions from Supabase, runs the backtest
    engine, and persists results back to the database.
    """
    from loguru import logger

    import numpy as np
    import pandas as pd

    from backtest.engine import BacktestEngine

    logger.info(
        "Background backtest started: model_type={}, mode={}, period={}~{}",
        request.model_type,
        request.mode,
        request.period_start,
        request.period_end,
    )

    try:
        period_start = request.period_start or "2023-01-01"
        period_end = request.period_end or str(date.today())

        _update_progress(task_id, "running", "載入股票資料...")

        # Load available stock IDs from the database
        stock_resp = (
            db.client.table("stock_prices")
            .select("stock_id")
            .gte("date", period_start)
            .lte("date", period_end)
            .execute()
        )
        if not stock_resp.data:
            logger.error("No stock data found for period {}~{}", period_start, period_end)
            _update_progress(task_id, "failed", "失敗", "找不到該期間的股票資料")
            return

        stock_ids = list({r["stock_id"] for r in stock_resp.data})
        logger.info("Found {} stocks in period", len(stock_ids))

        _update_progress(task_id, "running", f"載入 {len(stock_ids)} 檔股票價格...")

        # Load price data
        price_data = db.get_prices(stock_ids, period_start, period_end)
        if price_data.empty:
            logger.error("Price data is empty, aborting backtest")
            _update_progress(task_id, "failed", "失敗", "價格資料為空")
            return

        # Load benchmark data (0050.TW)
        benchmark_data = db.get_prices(["0050.TW"], period_start, period_end)
        if benchmark_data.empty:
            logger.warning("No benchmark data for 0050.TW, using synthetic benchmark")
            dates = sorted(price_data["date"].unique())
            benchmark_data = pd.DataFrame({
                "date": dates,
                "close": np.cumprod(1 + np.random.default_rng(42).normal(0.0003, 0.01, len(dates))) * 100,
            })
        else:
            benchmark_data = benchmark_data[["date", "close"]]

        _update_progress(task_id, "running", "載入特徵資料...")

        # Load features (needed for both modes to generate model-specific predictions)
        feature_data = db.get_features(stock_ids, period_start, period_end)
        if feature_data.empty:
            logger.error("Feature data is empty, aborting backtest")
            _update_progress(task_id, "failed", "失敗", "特徵資料為空")
            return

        _update_progress(task_id, "running", "計算預測目標值...")

        # Ensure target column exists
        if "target" not in feature_data.columns:
            # Compute forward return as target from price data
            price_pivot = price_data.pivot_table(
                index="date", columns="stock_id", values="close",
            )
            fwd_ret = price_pivot.pct_change().shift(-1).stack().reset_index()
            fwd_ret.columns = ["date", "stock_id", "target"]
            feature_data = feature_data.merge(
                fwd_ret, on=["date", "stock_id"], how="left",
            )
            feature_data["target"] = feature_data["target"].fillna(0)

        _update_progress(task_id, "running", f"載入 {request.model_type} 模型...")

        # Load the specified model
        model = _get_predictor(db, request.model_type)

        # Initialize engine
        engine = BacktestEngine()

        if request.mode == "walk_forward":
            _update_progress(task_id, "running", "執行 Walk Forward 回測...")
            result = engine.walk_forward(
                feature_data=feature_data,
                price_data=price_data,
                benchmark_data=benchmark_data,
                model=model,
                initial_capital=request.initial_capital,
            )
        else:
            _update_progress(task_id, "running", "訓練模型並產生預測...")
            # Static run mode — single train/predict split using specified model
            predictions = _generate_predictions(
                feature_data, price_data, model, request.model_type,
            )
            if predictions.empty:
                logger.error("Failed to generate predictions for {}", request.model_type)
                _update_progress(task_id, "failed", "失敗", "無法產生預測")
                return

            _update_progress(task_id, "running", "執行回測模擬...")
            result = engine.run(
                predictions=predictions,
                price_data=price_data,
                benchmark_data=benchmark_data,
                initial_capital=request.initial_capital,
            )

        _update_progress(task_id, "running", "儲存結果...")

        # Save results to Supabase
        db.save_backtest({
            "run_date": str(date.today()),
            "model_type": request.model_type,
            "period_start": period_start,
            "period_end": period_end,
            "metrics": _sanitize_metrics(result.metrics),
            "config": result.config,
        })

        total_return = result.metrics.get("total_return", 0)
        _update_progress(task_id, "success", f"完成！報酬率 {total_return:+.2%}")

        logger.info(
            "Backtest completed and saved: mode={}, final_value={:,.0f}",
            request.mode,
            result.metrics.get("final_value", 0),
        )

    except Exception as exc:
        logger.exception("Backtest execution failed: {}", exc)
        _update_progress(task_id, "failed", "失敗", str(exc))


def _generate_predictions(
    feature_data: "pd.DataFrame",
    price_data: "pd.DataFrame",
    model: object,
    model_type: str,
) -> "pd.DataFrame":
    """Generate predictions by training on early data and predicting on later data.

    Splits feature data into train (first 60%) and test (last 40%) by date.
    Trains the model on the train split and generates predictions for the
    test split, which are then used for backtesting.

    Args:
        feature_data: DataFrame with date, stock_id, target, and feature columns.
        price_data: DataFrame with price data (unused here but kept for signature).
        model: Predictor object with fit/predict interface.
        model_type: Model type string for logging.

    Returns:
        DataFrame with date, stock_id, predicted_return, score columns.
    """
    import pandas as pd
    from loguru import logger

    feature_data = feature_data.copy()
    feature_data["date"] = pd.to_datetime(feature_data["date"])

    dates = sorted(feature_data["date"].unique())
    if len(dates) < 10:
        logger.error("Not enough dates ({}) to split train/test", len(dates))
        return pd.DataFrame()

    # Split: first 60% for training, last 40% for testing
    split_idx = int(len(dates) * 0.6)
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])

    train_df = feature_data[feature_data["date"].isin(train_dates)]
    test_df = feature_data[feature_data["date"].isin(test_dates)]

    meta_cols = {"date", "stock_id", "target"}
    feature_cols = [c for c in feature_data.columns if c not in meta_cols]

    if not feature_cols:
        logger.error("No feature columns found")
        return pd.DataFrame()

    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].copy()
    X_test = test_df[feature_cols].copy()

    # Drop columns that are entirely NaN
    valid_cols = X_train.columns[X_train.notna().any()]
    X_train = X_train[valid_cols]
    X_test = X_test[valid_cols]

    # Fill remaining NaN with column median
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    y_train = y_train.fillna(0)

    logger.info(
        "Static backtest {}: train {} dates ({} rows), test {} dates ({} rows), {} features",
        model_type, len(train_dates), len(X_train), len(test_dates), len(X_test), len(valid_cols),
    )

    # Train and predict
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result = test_df[["date", "stock_id"]].copy()
    result["predicted_return"] = preds
    result["score"] = preds
    return result


def _get_predictor(db: DatabaseLoader, model_type: str):
    """Load active model from DB or create a fresh predictor of the correct type.

    Attempts to load the active model from DB. If unavailable, creates
    a new predictor instance matching the requested model_type so that
    different model types produce genuinely different backtest results.
    """
    from loguru import logger

    active = db.get_active_model(model_type)
    if active and active.get("file_path"):
        import pickle
        from pathlib import Path

        path = Path(active["file_path"])
        if path.exists():
            with open(path, "rb") as f:
                logger.info("Loaded saved model for {} from {}", model_type, path)
                return pickle.load(f)

    logger.info("No saved model for {}, creating fresh predictor", model_type)
    return _create_fresh_predictor(model_type)


def _create_fresh_predictor(model_type: str):
    """Create a fresh (unfitted) predictor instance for the given model type."""
    from models.baseline import RidgePredictor
    from models.ensemble import EnsemblePredictor
    from models.lightgbm_model import LightGBMPredictor
    from models.tree_models import RandomForestPredictor, XGBoostPredictor

    if model_type == "ridge":
        return RidgePredictor(alpha=1.0)

    if model_type == "random_forest":
        return RandomForestPredictor()

    if model_type == "xgboost":
        return XGBoostPredictor()

    if model_type == "lightgbm":
        return LightGBMPredictor()

    if model_type == "ensemble":
        rf = RandomForestPredictor()
        xgb = XGBoostPredictor()
        return EnsemblePredictor(
            models={"random_forest": rf, "xgboost": xgb},
            weights={"random_forest": 0.4, "xgboost": 0.6},
        )

    # Unknown model type — default to Ridge
    return RidgePredictor(alpha=1.0)


def _sanitize_metrics(metrics: dict) -> dict:
    """Convert numpy/pandas types in metrics dict to JSON-safe values."""
    import numpy as np

    sanitized = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            sanitized[k] = _sanitize_metrics(v)
        elif isinstance(v, (np.integer, np.floating)):
            sanitized[k] = v.item()
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            sanitized[k] = None
        else:
            sanitized[k] = v
    return sanitized

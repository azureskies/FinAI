"""Backtest endpoints."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import get_db
from data.loaders.supabase import SupabaseLoader

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


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


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #

@router.get("/results", response_model=BacktestListResponse)
def list_backtest_results(
    limit: int = Query(10, ge=1, le=50),
    db: Optional[SupabaseLoader] = Depends(get_db),
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


@router.get("/{result_id}", response_model=BacktestSummary)
def get_backtest_result(
    result_id: str,
    db: Optional[SupabaseLoader] = Depends(get_db),
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


@router.post("/run", response_model=BacktestRunResponse)
def run_backtest(
    request: BacktestRunRequest,
    background_tasks: BackgroundTasks,
    db: Optional[SupabaseLoader] = Depends(get_db),
) -> BacktestRunResponse:
    """Trigger a new backtest run (async)."""
    if db is None:
        return BacktestRunResponse(
            status="error", message="Supabase not configured — cannot run backtest"
        )

    background_tasks.add_task(_execute_backtest, db, request)
    return BacktestRunResponse(
        status="accepted",
        message=f"Backtest for {request.model_type} scheduled",
    )


def _execute_backtest(db: SupabaseLoader, request: BacktestRunRequest) -> None:
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
            return

        stock_ids = list({r["stock_id"] for r in stock_resp.data})
        logger.info("Found {} stocks in period", len(stock_ids))

        # Load price data
        price_data = db.get_prices(stock_ids, period_start, period_end)
        if price_data.empty:
            logger.error("Price data is empty, aborting backtest")
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

        # Initialize engine
        engine = BacktestEngine()

        if request.mode == "walk_forward":
            # Load features for walk-forward mode
            feature_data = db.get_features(stock_ids, period_start, period_end)
            if feature_data.empty:
                logger.error("Feature data is empty, cannot run walk_forward")
                return

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

            # Use active model or a simple predictor as fallback
            model = _get_predictor(db, request.model_type)

            result = engine.walk_forward(
                feature_data=feature_data,
                price_data=price_data,
                benchmark_data=benchmark_data,
                model=model,
                initial_capital=request.initial_capital,
            )
        else:
            # Static run mode — use latest predictions
            predictions = db.get_latest_predictions()
            if predictions.empty:
                logger.error("No predictions available, aborting backtest")
                return

            result = engine.run(
                predictions=predictions,
                price_data=price_data,
                benchmark_data=benchmark_data,
                initial_capital=request.initial_capital,
            )

        # Save results to Supabase
        db.save_backtest({
            "run_date": str(date.today()),
            "model_type": request.model_type,
            "period_start": period_start,
            "period_end": period_end,
            "metrics": _sanitize_metrics(result.metrics),
            "config": result.config,
        })

        logger.info(
            "Backtest completed and saved: mode={}, final_value={:,.0f}",
            request.mode,
            result.metrics.get("final_value", 0),
        )

    except Exception as exc:
        logger.exception("Backtest execution failed: {}", exc)


def _get_predictor(db: SupabaseLoader, model_type: str):
    """Load active model or return a simple fallback predictor.

    Attempts to load the active model from DB. If unavailable, returns
    a simple Ridge regression as a fallback.
    """
    from sklearn.linear_model import Ridge

    active = db.get_active_model(model_type)
    if active and active.get("file_path"):
        import pickle
        from pathlib import Path

        path = Path(active["file_path"])
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)

    # Fallback: simple Ridge predictor
    return Ridge(alpha=1.0)


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

"""Dashboard aggregated data endpoints."""

from __future__ import annotations

import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import get_db
from data.loaders import DatabaseLoader


def _safe_float(v: object) -> Optional[float]:
    """Convert to float, returning None for NaN/inf."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# ------------------------------------------------------------------ #
#  Response models
# ------------------------------------------------------------------ #

class DashboardSummary(BaseModel):
    stocks_count: int = 0
    latest_prediction_date: Optional[str] = None
    predictions_count: int = 0
    active_models: int = 0
    backtest_runs: int = 0
    message: Optional[str] = None


class TopPick(BaseModel):
    stock_id: str
    predicted_return: float
    score: float
    date: Optional[str] = None


class ScoredStock(BaseModel):
    stock_id: str
    stock_name: Optional[str] = None
    composite_score: Optional[float] = None
    momentum_score: Optional[float] = None
    trend_score: Optional[float] = None
    volatility_score: Optional[float] = None
    volume_score: Optional[float] = None
    ai_score: Optional[float] = None
    risk_level: Optional[str] = None
    max_drawdown: Optional[float] = None
    volatility_ann: Optional[float] = None
    win_rate: Optional[float] = None
    predicted_return: Optional[float] = None
    date: Optional[str] = None


class TopPicksResponse(BaseModel):
    picks: list[ScoredStock]
    message: Optional[str] = None


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #

@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(
    db: Optional[DatabaseLoader] = Depends(get_db),
) -> DashboardSummary:
    """Get overall dashboard statistics."""
    if db is None:
        return DashboardSummary(message="Supabase not configured")

    try:
        # Count distinct stocks
        price_resp = db.client.table("stock_prices").select("stock_id").execute()
        stock_ids = {r["stock_id"] for r in price_resp.data}

        # Latest predictions
        pred_df = db.get_latest_predictions()
        latest_date = None
        pred_count = 0
        if not pred_df.empty:
            latest_date = str(pred_df["date"].iloc[0])[:10]
            pred_count = len(pred_df)

        # Active models
        active_resp = (
            db.client.table("model_versions")
            .select("id")
            .eq("is_active", True)
            .execute()
        )

        # Backtest runs
        bt_resp = db.client.table("backtest_results").select("id").execute()

        return DashboardSummary(
            stocks_count=len(stock_ids),
            latest_prediction_date=latest_date,
            predictions_count=pred_count,
            active_models=len(active_resp.data),
            backtest_runs=len(bt_resp.data),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/top-picks", response_model=TopPicksResponse)
def top_picks(
    n: int = Query(10, ge=1, le=5000, description="Number of top picks to return"),
    db: Optional[DatabaseLoader] = Depends(get_db),
) -> TopPicksResponse:
    """Get top N stocks by latest prediction score."""
    if db is None:
        return TopPicksResponse(picks=[], message="Supabase not configured")

    try:
        # Load stock name mapping
        name_map: dict[str, str] = {}
        if hasattr(db, "get_stock_name_map"):
            name_map = db.get_stock_name_map()

        # Try scores-based ranking first
        if hasattr(db, "get_latest_scores"):
            scores_df = db.get_latest_scores(limit=n)
            if not scores_df.empty:
                # Merge with predictions for predicted_return
                pred_df = db.get_latest_predictions()
                pred_map = {}
                if not pred_df.empty:
                    pred_map = dict(
                        zip(pred_df["stock_id"], pred_df["predicted_return"])
                    )

                picks = [
                    ScoredStock(
                        stock_id=row["stock_id"],
                        stock_name=name_map.get(row["stock_id"]),
                        composite_score=_safe_float(row.get("composite_score")),
                        momentum_score=_safe_float(row.get("momentum_score")),
                        trend_score=_safe_float(row.get("trend_score")),
                        volatility_score=_safe_float(row.get("volatility_score")),
                        volume_score=_safe_float(row.get("volume_score")),
                        ai_score=_safe_float(row.get("ai_score")),
                        risk_level=row.get("risk_level"),
                        max_drawdown=_safe_float(row.get("max_drawdown")),
                        volatility_ann=_safe_float(row.get("volatility_ann")),
                        win_rate=_safe_float(row.get("win_rate")),
                        predicted_return=_safe_float(pred_map.get(row["stock_id"])),
                        date=str(row.get("date", ""))[:10],
                    )
                    for _, row in scores_df.iterrows()
                ]
                return TopPicksResponse(picks=picks)

        # Fallback to predictions-based ranking
        df = db.get_latest_predictions()
        if df.empty:
            return TopPicksResponse(picks=[], message="No predictions available")

        top = df.head(n)
        picks = [
            ScoredStock(
                stock_id=row["stock_id"],
                stock_name=name_map.get(row["stock_id"]),
                predicted_return=row["predicted_return"],
                composite_score=row.get("score"),
                date=str(row["date"])[:10],
            )
            for _, row in top.iterrows()
        ]
        return TopPicksResponse(picks=picks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

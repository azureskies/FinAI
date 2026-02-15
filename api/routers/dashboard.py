"""Dashboard aggregated data endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import get_db
from data.loaders.supabase import SupabaseLoader

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


class TopPicksResponse(BaseModel):
    picks: list[TopPick]
    message: Optional[str] = None


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #

@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(
    db: Optional[SupabaseLoader] = Depends(get_db),
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
    n: int = Query(10, ge=1, le=50, description="Number of top picks to return"),
    db: Optional[SupabaseLoader] = Depends(get_db),
) -> TopPicksResponse:
    """Get top N stocks by latest prediction score."""
    if db is None:
        return TopPicksResponse(picks=[], message="Supabase not configured")

    try:
        df = db.get_latest_predictions()
        if df.empty:
            return TopPicksResponse(picks=[], message="No predictions available")

        top = df.head(n)
        picks = [
            TopPick(
                stock_id=row["stock_id"],
                predicted_return=row["predicted_return"],
                score=row["score"],
                date=str(row["date"])[:10],
            )
            for _, row in top.iterrows()
        ]
        return TopPicksResponse(picks=picks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

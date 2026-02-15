"""Backtest endpoints."""

from __future__ import annotations

from typing import Optional

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
    period_start: Optional[str] = None
    period_end: Optional[str] = None
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

    This is a placeholder — the real implementation would invoke
    the backtest engine from the backtest module.
    """
    from loguru import logger

    logger.info(
        "Background backtest started: model_type={}, period={}~{}",
        request.model_type,
        request.period_start,
        request.period_end,
    )
    # TODO: integrate with backtest.engine.BacktestEngine

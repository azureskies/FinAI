"""Monitoring API endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import get_db
from data.loaders import DatabaseLoader

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


# ------------------------------------------------------------------ #
#  Response models
# ------------------------------------------------------------------ #

class CheckResultItem(BaseModel):
    name: str
    status: str
    message: str
    details: dict = {}


class HealthResponse(BaseModel):
    timestamp: str
    overall_status: str
    checks: list[CheckResultItem]
    message: Optional[str] = None


class PipelineRunItem(BaseModel):
    id: str
    pipeline_name: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str
    metrics: Optional[dict] = None
    error: Optional[str] = None


class PipelineRunsResponse(BaseModel):
    runs: list[PipelineRunItem]
    message: Optional[str] = None


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #

@router.get("/health", response_model=HealthResponse)
def health_status(
    db: Optional[DatabaseLoader] = Depends(get_db),
) -> HealthResponse:
    """Run health checks and return status."""
    if db is None:
        return HealthResponse(
            timestamp="",
            overall_status="unknown",
            checks=[],
            message="Supabase not configured",
        )

    try:
        from monitoring.health import HealthChecker

        checker = HealthChecker(db)
        report = checker.run_all_checks()
        return HealthResponse(
            timestamp=report.timestamp,
            overall_status=report.overall_status.value,
            checks=[
                CheckResultItem(
                    name=c.name,
                    status=c.status.value,
                    message=c.message,
                    details=c.details,
                )
                for c in report.checks
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/pipeline-runs", response_model=PipelineRunsResponse)
def pipeline_runs(
    pipeline_name: Optional[str] = Query(None, description="Filter by pipeline name"),
    limit: int = Query(10, ge=1, le=50, description="Max results"),
    db: Optional[DatabaseLoader] = Depends(get_db),
) -> PipelineRunsResponse:
    """Get recent pipeline execution history."""
    if db is None:
        return PipelineRunsResponse(runs=[], message="Supabase not configured")

    try:
        query = (
            db.client.table("pipeline_runs")
            .select("*")
            .order("start_time", desc=True)
            .limit(limit)
        )
        if pipeline_name:
            query = query.eq("pipeline_name", pipeline_name)

        resp = query.execute()
        runs = [
            PipelineRunItem(
                id=r["id"],
                pipeline_name=r["pipeline_name"],
                start_time=r.get("start_time"),
                end_time=r.get("end_time"),
                status=r.get("status", "unknown"),
                metrics=r.get("metrics"),
                error=r.get("error"),
            )
            for r in resp.data
        ]
        return PipelineRunsResponse(runs=runs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

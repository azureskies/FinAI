"""Pipeline operation endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from api.dependencies import get_db
from data.loaders import DatabaseLoader

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# Module-level dict to track task execution status
_task_status: dict[str, dict] = {}


# ------------------------------------------------------------------ #
#  Request / Response models
# ------------------------------------------------------------------ #

class PipelineRunRequest(BaseModel):
    mode: Literal["full", "incremental"] = "incremental"
    stock_ids: Optional[list[str]] = None


class PipelineRunResponse(BaseModel):
    status: str
    message: str
    task_id: str


class PipelineTaskStatus(BaseModel):
    task_id: str
    status: str  # pending / running / success / failed
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: Optional[str] = None
    error: Optional[str] = None


class PipelineStatusResponse(BaseModel):
    tasks: list[PipelineTaskStatus]


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #

@router.post("/daily-update", response_model=PipelineRunResponse)
def trigger_daily_update(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    db: Optional[DatabaseLoader] = Depends(get_db),
) -> PipelineRunResponse:
    """Trigger the daily update pipeline as a background task."""
    task_id = str(uuid.uuid4())
    _task_status[task_id] = {
        "status": "pending",
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "progress": "Queued",
        "error": None,
    }

    background_tasks.add_task(_run_daily_update, task_id, request, db)
    return PipelineRunResponse(
        status="accepted",
        message=f"Daily update pipeline ({request.mode}) scheduled",
        task_id=task_id,
    )


@router.get("/status", response_model=PipelineStatusResponse)
def get_pipeline_status() -> PipelineStatusResponse:
    """Return status of all tracked pipeline tasks (most recent first)."""
    tasks = [
        PipelineTaskStatus(task_id=tid, **info)
        for tid, info in reversed(list(_task_status.items()))
    ]
    return PipelineStatusResponse(tasks=tasks[:20])


# ------------------------------------------------------------------ #
#  Background task
# ------------------------------------------------------------------ #

def _run_daily_update(
    task_id: str,
    request: PipelineRunRequest,
    db: Optional[DatabaseLoader],
) -> None:
    """Execute the daily update pipeline in the background."""
    from loguru import logger

    _task_status[task_id]["status"] = "running"
    _task_status[task_id]["progress"] = "Starting pipeline..."

    try:
        import sys
        # Build argv for daily_update.main()
        argv_backup = sys.argv
        sys.argv = ["daily_update"]

        if request.mode == "full":
            sys.argv.append("--full-refresh")

        if request.stock_ids and len(request.stock_ids) == 1:
            sys.argv.extend(["--stock-id", request.stock_ids[0]])

        sys.argv.append("--no-notify")

        _task_status[task_id]["progress"] = "Running daily update..."

        from scripts.daily_update import main as daily_update_main
        daily_update_main()

        _task_status[task_id]["status"] = "success"
        _task_status[task_id]["progress"] = "Completed"
        _task_status[task_id]["finished_at"] = datetime.now().isoformat()
        logger.info("Pipeline task {} completed successfully", task_id)

    except SystemExit:
        # daily_update calls sys.exit(1) on failure
        _task_status[task_id]["status"] = "failed"
        _task_status[task_id]["error"] = "Pipeline exited with error"
        _task_status[task_id]["finished_at"] = datetime.now().isoformat()
        logger.error("Pipeline task {} failed (SystemExit)", task_id)

    except Exception as exc:
        _task_status[task_id]["status"] = "failed"
        _task_status[task_id]["error"] = str(exc)
        _task_status[task_id]["finished_at"] = datetime.now().isoformat()
        logger.exception("Pipeline task {} failed: {}", task_id, exc)

    finally:
        sys.argv = argv_backup

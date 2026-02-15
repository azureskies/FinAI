"""Pipeline execution tracker.

Records pipeline runs (daily_update, weekly_retrain, monthly_report)
in the Supabase pipeline_runs table for monitoring and auditing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

from data.loaders.supabase import SupabaseLoader


class PipelineTracker:
    """Track pipeline execution in Supabase."""

    def __init__(self, db: SupabaseLoader) -> None:
        self.db = db

    def start_run(
        self,
        pipeline_name: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Record the start of a pipeline run.

        Args:
            pipeline_name: Pipeline identifier (e.g. "daily_update").
            metadata: Optional metadata dict.

        Returns:
            Run ID (UUID string).
        """
        record = {
            "pipeline_name": pipeline_name,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "metadata": metadata or {},
        }
        resp = self.db.client.table("pipeline_runs").insert(record).execute()
        run_id = resp.data[0]["id"]
        logger.info("Pipeline run started: {} (id={})", pipeline_name, run_id)
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str = "success",
        metrics: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record the completion of a pipeline run.

        Args:
            run_id: ID returned from start_run.
            status: Final status ("success", "failed", "warning").
            metrics: Optional pipeline metrics.
            error: Error message if failed.
        """
        update: dict = {
            "end_time": datetime.now().isoformat(),
            "status": status,
        }
        if metrics is not None:
            update["metrics"] = metrics
        if error is not None:
            update["error"] = error

        self.db.client.table("pipeline_runs").update(update).eq("id", run_id).execute()
        logger.info("Pipeline run finished: {} (status={})", run_id, status)

    def get_recent_runs(
        self,
        pipeline_name: str,
        n: int = 10,
    ) -> list[dict]:
        """Get recent pipeline runs.

        Args:
            pipeline_name: Pipeline identifier.
            n: Maximum number of runs to return.

        Returns:
            List of run dicts ordered by start_time desc.
        """
        resp = (
            self.db.client.table("pipeline_runs")
            .select("*")
            .eq("pipeline_name", pipeline_name)
            .order("start_time", desc=True)
            .limit(n)
            .execute()
        )
        return resp.data

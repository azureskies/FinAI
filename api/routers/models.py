"""Model management endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import get_db
from data.loaders.supabase import SupabaseLoader

router = APIRouter(prefix="/api/models", tags=["models"])


# ------------------------------------------------------------------ #
#  Response models
# ------------------------------------------------------------------ #

class ModelVersion(BaseModel):
    id: str
    model_type: Optional[str] = None
    metrics: Optional[dict] = None
    file_path: Optional[str] = None
    storage_path: Optional[str] = None
    is_active: Optional[bool] = None
    description: Optional[str] = None
    created_at: Optional[str] = None


class ModelListResponse(BaseModel):
    models: list[ModelVersion]
    message: Optional[str] = None


class ModelMetricsResponse(BaseModel):
    model_id: str
    model_type: Optional[str] = None
    metrics: dict
    message: Optional[str] = None


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #

@router.get("", response_model=ModelListResponse)
def list_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    db: Optional[SupabaseLoader] = Depends(get_db),
) -> ModelListResponse:
    """List all model versions."""
    if db is None:
        return ModelListResponse(models=[], message="Supabase not configured")

    try:
        query = db.client.table("model_versions").select("*").order("created_at", desc=True)
        if model_type:
            query = query.eq("model_type", model_type)
        resp = query.execute()
        models = [ModelVersion(**r) for r in resp.data]
        return ModelListResponse(models=models)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/active", response_model=ModelListResponse)
def get_active_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    db: Optional[SupabaseLoader] = Depends(get_db),
) -> ModelListResponse:
    """Get currently active model(s)."""
    if db is None:
        return ModelListResponse(models=[], message="Supabase not configured")

    try:
        query = (
            db.client.table("model_versions")
            .select("*")
            .eq("is_active", True)
        )
        if model_type:
            query = query.eq("model_type", model_type)
        resp = query.execute()
        models = [ModelVersion(**r) for r in resp.data]
        return ModelListResponse(models=models)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{model_id}/metrics", response_model=ModelMetricsResponse)
def get_model_metrics(
    model_id: str,
    db: Optional[SupabaseLoader] = Depends(get_db),
) -> ModelMetricsResponse:
    """Get performance metrics for a specific model version."""
    if db is None:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    try:
        resp = (
            db.client.table("model_versions")
            .select("id, model_type, metrics")
            .eq("id", model_id)
            .execute()
        )
        if not resp.data:
            raise HTTPException(status_code=404, detail="Model not found")
        row = resp.data[0]
        return ModelMetricsResponse(
            model_id=row["id"],
            model_type=row.get("model_type"),
            metrics=row.get("metrics", {}),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

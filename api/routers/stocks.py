"""Stock data endpoints."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.dependencies import get_db
from data.loaders.supabase import SupabaseLoader

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


# ------------------------------------------------------------------ #
#  Response models
# ------------------------------------------------------------------ #

class StockItem(BaseModel):
    stock_id: str
    name: Optional[str] = None


class StockListResponse(BaseModel):
    stocks: list[StockItem]
    message: Optional[str] = None


class PriceRecord(BaseModel):
    date: str
    stock_id: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    adj_close: Optional[float] = None


class PriceResponse(BaseModel):
    prices: list[PriceRecord]
    message: Optional[str] = None


class FeatureRecord(BaseModel):
    date: str
    stock_id: str
    features: dict


class FeatureResponse(BaseModel):
    features: list[FeatureRecord]
    message: Optional[str] = None


class PredictionRecord(BaseModel):
    date: str
    stock_id: str
    predicted_return: float
    score: float
    model_version: Optional[str] = None


class PredictionResponse(BaseModel):
    predictions: list[PredictionRecord]
    message: Optional[str] = None


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #

@router.get("", response_model=StockListResponse)
def list_stocks(db: Optional[SupabaseLoader] = Depends(get_db)) -> StockListResponse:
    """List all stocks in the universe."""
    if db is None:
        return StockListResponse(stocks=[], message="Supabase not configured")

    try:
        resp = (
            db.client.table("stock_prices")
            .select("stock_id")
            .execute()
        )
        unique_ids = sorted({r["stock_id"] for r in resp.data})
        stocks = [StockItem(stock_id=sid) for sid in unique_ids]
        return StockListResponse(stocks=stocks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{stock_id}/prices", response_model=PriceResponse)
def get_prices(
    stock_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    db: Optional[SupabaseLoader] = Depends(get_db),
) -> PriceResponse:
    """Get price history for a stock."""
    if db is None:
        return PriceResponse(prices=[], message="Supabase not configured")

    sd = start_date or str(date.today() - timedelta(days=365))
    ed = end_date or str(date.today())

    try:
        df = db.get_prices([stock_id], sd, ed)
        if df.empty:
            return PriceResponse(prices=[], message="No price data found")
        records = df.to_dict("records")
        prices = [
            PriceRecord(**{**r, "date": str(r["date"])[:10]}) for r in records
        ]
        return PriceResponse(prices=prices)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{stock_id}/features", response_model=FeatureResponse)
def get_features(
    stock_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    db: Optional[SupabaseLoader] = Depends(get_db),
) -> FeatureResponse:
    """Get computed features for a stock."""
    if db is None:
        return FeatureResponse(features=[], message="Supabase not configured")

    sd = start_date or str(date.today() - timedelta(days=365))
    ed = end_date or str(date.today())

    try:
        df = db.get_features([stock_id], sd, ed)
        if df.empty:
            return FeatureResponse(features=[], message="No feature data found")
        result = []
        for _, row in df.iterrows():
            meta = {"date": str(row["date"])[:10], "stock_id": row["stock_id"]}
            feat_data = {k: v for k, v in row.items() if k not in ("date", "stock_id")}
            result.append(FeatureRecord(
                date=meta["date"], stock_id=meta["stock_id"], features=feat_data
            ))
        return FeatureResponse(features=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{stock_id}/predictions", response_model=PredictionResponse)
def get_predictions(
    stock_id: str,
    db: Optional[SupabaseLoader] = Depends(get_db),
) -> PredictionResponse:
    """Get latest predictions for a stock."""
    if db is None:
        return PredictionResponse(predictions=[], message="Supabase not configured")

    try:
        df = db.get_latest_predictions()
        if df.empty:
            return PredictionResponse(predictions=[], message="No predictions found")
        filtered = df[df["stock_id"] == stock_id]
        if filtered.empty:
            return PredictionResponse(predictions=[], message="No predictions for this stock")
        records = filtered.to_dict("records")
        preds = [
            PredictionRecord(**{**r, "date": str(r["date"])[:10]}) for r in records
        ]
        return PredictionResponse(predictions=preds)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

"""Supabase database interface for FinAI platform.

Provides CRUD operations for stock prices, features, model versions,
predictions, and backtest results stored in Supabase (PostgreSQL).
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from supabase import Client, create_client

load_dotenv()

_BATCH_SIZE = 500


class SupabaseLoader:
    """Supabase database interface for FinAI platform."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None) -> None:
        """Initialize Supabase client from arguments or .env."""
        self.url = url or os.getenv("SUPABASE_URL", "")
        self.key = key or os.getenv("SUPABASE_KEY", "")
        if not self.url or not self.key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set "
                "via arguments or environment variables"
            )
        self.client: Client = create_client(self.url, self.key)
        logger.info("Supabase client initialized ({})", self.url[:40] + "...")

    # ------------------------------------------------------------------ #
    #  Price data
    # ------------------------------------------------------------------ #

    def upsert_prices(self, df: pd.DataFrame) -> int:
        """Upsert price data.

        Args:
            df: DataFrame with columns:
                date, stock_id, open, high, low, close, volume, adj_close.

        Returns:
            Number of rows upserted.
        """
        required = {"date", "stock_id", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {required - set(df.columns)}")

        records = _df_to_records(df, date_cols=["date"])
        total = 0
        for batch in _chunked(records, _BATCH_SIZE):
            resp = (
                self.client.table("stock_prices")
                .upsert(batch, on_conflict="date,stock_id")
                .execute()
            )
            total += len(resp.data)
        logger.info("Upserted {} price rows", total)
        return total

    def get_prices(
        self,
        stock_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Get price data for stocks in date range.

        Args:
            stock_ids: List of stock codes.
            start_date: Start date (inclusive), e.g. "2024-01-01".
            end_date: End date (inclusive).

        Returns:
            DataFrame with price columns, indexed by (date, stock_id).
        """
        resp = (
            self.client.table("stock_prices")
            .select("date, stock_id, open, high, low, close, volume, adj_close")
            .in_("stock_id", stock_ids)
            .gte("date", start_date)
            .lte("date", end_date)
            .order("date")
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------ #
    #  Features
    # ------------------------------------------------------------------ #

    def upsert_features(self, df: pd.DataFrame) -> int:
        """Upsert computed features.

        Args:
            df: DataFrame with columns: date, stock_id, and feature columns.
                All feature columns (except date/stock_id) are stored as JSONB.

        Returns:
            Number of rows upserted.
        """
        meta_cols = {"date", "stock_id"}
        feat_cols = [c for c in df.columns if c not in meta_cols]

        records = []
        for _, row in df.iterrows():
            records.append({
                "date": _to_date_str(row["date"]),
                "stock_id": row["stock_id"],
                "features": {c: _safe_value(row[c]) for c in feat_cols},
            })

        total = 0
        for batch in _chunked(records, _BATCH_SIZE):
            resp = (
                self.client.table("stock_features")
                .upsert(batch, on_conflict="date,stock_id")
                .execute()
            )
            total += len(resp.data)
        logger.info("Upserted {} feature rows", total)
        return total

    def get_features(
        self,
        stock_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Get features for stocks in date range.

        Returns:
            DataFrame with date, stock_id, and unpacked feature columns.
        """
        resp = (
            self.client.table("stock_features")
            .select("date, stock_id, features")
            .in_("stock_id", stock_ids)
            .gte("date", start_date)
            .lte("date", end_date)
            .order("date")
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()

        rows = []
        for r in resp.data:
            row = {"date": r["date"], "stock_id": r["stock_id"]}
            row.update(r.get("features", {}))
            rows.append(row)
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------ #
    #  Model versions
    # ------------------------------------------------------------------ #

    def save_model_version(
        self,
        model_type: str,
        metrics: dict,
        file_path: str,
        description: str = "",
    ) -> str:
        """Save model metadata and upload .pkl to Supabase Storage.

        Args:
            model_type: e.g. "ridge", "xgboost", "ensemble".
            metrics: Model evaluation metrics dict.
            file_path: Local path to the model .pkl file.
            description: Optional description.

        Returns:
            version_id (UUID string).
        """
        # Upload model file to storage
        storage_path = ""
        if file_path and os.path.isfile(file_path):
            fname = os.path.basename(file_path)
            storage_path = f"{model_type}/{fname}"
            with open(file_path, "rb") as f:
                self.client.storage.from_("models").upload(
                    storage_path, f.read(), {"content-type": "application/octet-stream"}
                )
            logger.info("Uploaded model to storage: {}", storage_path)

        record = {
            "model_type": model_type,
            "metrics": metrics,
            "file_path": file_path,
            "storage_path": storage_path,
            "is_active": False,
            "description": description,
        }
        resp = self.client.table("model_versions").insert(record).execute()
        version_id = resp.data[0]["id"]
        logger.info("Saved model version {} (type={})", version_id, model_type)
        return version_id

    def get_active_model(self, model_type: str) -> Optional[dict]:
        """Get currently active model metadata.

        Args:
            model_type: e.g. "ridge", "xgboost", "ensemble".

        Returns:
            Model version dict or None if no active model.
        """
        resp = (
            self.client.table("model_versions")
            .select("*")
            .eq("model_type", model_type)
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        return resp.data[0]

    def set_active_model(self, version_id: str) -> None:
        """Set a model version as active (deactivate others of same type).

        Args:
            version_id: UUID of the model version to activate.
        """
        # Get model type first
        resp = (
            self.client.table("model_versions")
            .select("model_type")
            .eq("id", version_id)
            .execute()
        )
        if not resp.data:
            raise ValueError(f"Model version {version_id} not found")
        model_type = resp.data[0]["model_type"]

        # Deactivate all of same type
        self.client.table("model_versions").update({"is_active": False}).eq(
            "model_type", model_type
        ).execute()

        # Activate target
        self.client.table("model_versions").update({"is_active": True}).eq(
            "id", version_id
        ).execute()
        logger.info("Activated model {} (type={})", version_id, model_type)

    # ------------------------------------------------------------------ #
    #  Predictions
    # ------------------------------------------------------------------ #

    def save_predictions(self, predictions: pd.DataFrame) -> int:
        """Save model predictions.

        Args:
            predictions: DataFrame with columns:
                date, stock_id, predicted_return, score, model_version.

        Returns:
            Number of rows saved.
        """
        required = {"date", "stock_id", "predicted_return", "score", "model_version"}
        if not required.issubset(predictions.columns):
            raise ValueError(f"Missing columns: {required - set(predictions.columns)}")

        records = _df_to_records(predictions, date_cols=["date"])
        total = 0
        for batch in _chunked(records, _BATCH_SIZE):
            resp = (
                self.client.table("predictions")
                .upsert(batch, on_conflict="date,stock_id,model_version")
                .execute()
            )
            total += len(resp.data)
        logger.info("Saved {} prediction rows", total)
        return total

    def get_latest_predictions(self) -> pd.DataFrame:
        """Get most recent predictions (latest date).

        Returns:
            DataFrame with prediction columns.
        """
        # Get the latest prediction date
        resp = (
            self.client.table("predictions")
            .select("date")
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()

        latest_date = resp.data[0]["date"]
        resp = (
            self.client.table("predictions")
            .select("date, stock_id, predicted_return, score, model_version")
            .eq("date", latest_date)
            .order("score", desc=True)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------ #
    #  Backtest results
    # ------------------------------------------------------------------ #

    def save_backtest(self, result: dict) -> str:
        """Save backtest result summary.

        Args:
            result: Dict with keys:
                run_date, model_type, period_start, period_end, metrics, config.

        Returns:
            Backtest result UUID.
        """
        record = {
            "run_date": str(result.get("run_date", date.today())),
            "model_type": result.get("model_type", ""),
            "period_start": str(result["period_start"]) if "period_start" in result else None,
            "period_end": str(result["period_end"]) if "period_end" in result else None,
            "metrics": result.get("metrics", {}),
            "config": result.get("config", {}),
        }
        resp = self.client.table("backtest_results").insert(record).execute()
        result_id = resp.data[0]["id"]
        logger.info("Saved backtest result {}", result_id)
        return result_id

    def get_backtest_history(self, limit: int = 10) -> list[dict]:
        """Get recent backtest results.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of backtest result dicts ordered by run_date desc.
        """
        resp = (
            self.client.table("backtest_results")
            .select("*")
            .order("run_date", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data


# ---------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------- #


def _df_to_records(df: pd.DataFrame, date_cols: Optional[list[str]] = None) -> list[dict]:
    """Convert DataFrame to list of dicts with proper date serialization."""
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            val = row[col]
            if date_cols and col in date_cols:
                rec[col] = _to_date_str(val)
            else:
                rec[col] = _safe_value(val)
        records.append(rec)
    return records


def _to_date_str(val) -> str:
    """Convert various date types to ISO date string."""
    if isinstance(val, str):
        return val
    if hasattr(val, "isoformat"):
        return val.isoformat()[:10]
    return str(val)


def _safe_value(val):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if pd.isna(val):
        return None
    if hasattr(val, "item"):
        return val.item()
    return val


def _chunked(lst: list, size: int):
    """Yield successive chunks from list."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

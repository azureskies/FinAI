"""Shared dependencies for API endpoints."""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger

from data.loaders.supabase import SupabaseLoader

_loader: Optional[SupabaseLoader] = None


def get_db() -> Optional[SupabaseLoader]:
    """Get the shared SupabaseLoader instance.

    Returns None if Supabase is not configured, allowing endpoints
    to return fallback/empty data with an appropriate message.
    """
    global _loader
    if _loader is not None:
        return _loader

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        logger.warning("Supabase not configured — API will return empty data")
        return None

    try:
        _loader = SupabaseLoader(url, key)
        return _loader
    except Exception as exc:
        logger.error("Failed to initialize SupabaseLoader: {}", exc)
        return None

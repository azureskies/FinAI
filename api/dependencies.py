"""Shared dependencies for API endpoints."""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger

from data.loaders import DatabaseLoader

_loader: Optional[DatabaseLoader] = None


def get_db() -> Optional[DatabaseLoader]:
    """Get the shared DatabaseLoader instance.

    Returns None if database cannot be initialized, allowing endpoints
    to return fallback/empty data with an appropriate message.
    """
    global _loader
    if _loader is not None:
        return _loader

    backend = os.getenv("DB_BACKEND", "sqlite")

    try:
        if backend == "supabase":
            url = os.getenv("SUPABASE_URL", "")
            key = os.getenv("SUPABASE_KEY", "")
            if not url or not key:
                logger.warning("Supabase not configured — API will return empty data")
                return None
            _loader = DatabaseLoader(url, key)
        else:
            _loader = DatabaseLoader()
            _loader.init_schema()
        return _loader
    except Exception as exc:
        logger.error("Failed to initialize database: {}", exc)
        return None

"""Database loader factory.

Defaults to SQLite (zero-config). Set DB_BACKEND=supabase to use Supabase.
"""

import os

_BACKEND = os.getenv("DB_BACKEND", "sqlite")

if _BACKEND == "supabase":
    from data.loaders.supabase import SupabaseLoader as DatabaseLoader
else:
    from data.loaders.sqlite_loader import SQLiteLoader as DatabaseLoader

__all__ = ["DatabaseLoader"]

"""SQLite database interface for FinAI platform.

Drop-in replacement for SupabaseLoader. Stores all data locally in a
single SQLite file. Includes a lightweight query builder that mimics the
Supabase client API so existing code using ``db.client.table(...)`` works
without modification.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd
from loguru import logger

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "finai.db"
)
_DEFAULT_DB_PATH = os.path.abspath(_DEFAULT_DB_PATH)
_BATCH_SIZE = 500


# ====================================================================== #
#  Lightweight query builder (mimics Supabase PostgREST client)
# ====================================================================== #

class _QueryResponse:
    """Mimics the Supabase APIResponse."""

    def __init__(self, data: list[dict], count: Optional[int] = None) -> None:
        self.data = data
        self.count = count


class _TableQuery:
    """Chainable query builder for a single table."""

    def __init__(self, conn: sqlite3.Connection, table: str) -> None:
        self._conn = conn
        self._table = table
        self._mode = "select"
        self._select_cols = "*"
        self._count_mode: Optional[str] = None
        self._wheres: list[str] = []
        self._params: list[Any] = []
        self._order_col: Optional[str] = None
        self._order_desc = False
        self._limit_n: Optional[int] = None
        self._data: Any = None
        self._on_conflict: Optional[str] = None

    # -- Select ------------------------------------------------------
    def select(self, columns: str = "*", count: Optional[str] = None) -> "_TableQuery":
        self._select_cols = columns
        self._count_mode = count
        return self

    # -- Filters -----------------------------------------------------
    def eq(self, col: str, val: Any) -> "_TableQuery":
        self._wheres.append(f'"{col}" = ?')
        self._params.append(val)
        return self

    def in_(self, col: str, vals: list) -> "_TableQuery":
        if not vals:
            self._wheres.append("1 = 0")
            return self
        ph = ",".join("?" for _ in vals)
        self._wheres.append(f'"{col}" IN ({ph})')
        self._params.extend(vals)
        return self

    def gte(self, col: str, val: Any) -> "_TableQuery":
        self._wheres.append(f'"{col}" >= ?')
        self._params.append(val)
        return self

    def lte(self, col: str, val: Any) -> "_TableQuery":
        self._wheres.append(f'"{col}" <= ?')
        self._params.append(val)
        return self

    # -- Ordering / limit --------------------------------------------
    def order(self, col: str, desc: bool = False) -> "_TableQuery":
        self._order_col = col
        self._order_desc = desc
        return self

    def limit(self, n: int) -> "_TableQuery":
        self._limit_n = n
        return self

    # -- Mutations ---------------------------------------------------
    def insert(self, data: dict | list[dict]) -> "_TableQuery":
        self._mode = "insert"
        self._data = data if isinstance(data, list) else [data]
        return self

    def upsert(self, data: dict | list[dict], on_conflict: str = "") -> "_TableQuery":
        self._mode = "upsert"
        self._data = data if isinstance(data, list) else [data]
        self._on_conflict = on_conflict
        return self

    def update(self, data: dict) -> "_TableQuery":
        self._mode = "update"
        self._data = data
        return self

    def delete(self) -> "_TableQuery":
        self._mode = "delete"
        return self

    # -- Execute -----------------------------------------------------
    def execute(self) -> _QueryResponse:
        if self._mode == "select":
            return self._exec_select()
        if self._mode == "insert":
            return self._exec_insert()
        if self._mode == "upsert":
            return self._exec_upsert()
        if self._mode == "update":
            return self._exec_update()
        if self._mode == "delete":
            return self._exec_delete()
        raise ValueError(f"Unknown mode: {self._mode}")

    # -- Private helpers ---------------------------------------------
    def _where_clause(self) -> str:
        if not self._wheres:
            return ""
        return " WHERE " + " AND ".join(self._wheres)

    def _exec_select(self) -> _QueryResponse:
        cols = self._select_cols.replace(" ", "")
        sql = f'SELECT {cols} FROM "{self._table}"'
        sql += self._where_clause()
        if self._order_col:
            direction = "DESC" if self._order_desc else "ASC"
            sql += f' ORDER BY "{self._order_col}" {direction}'
        if self._limit_n is not None:
            sql += f" LIMIT {self._limit_n}"

        cur = self._conn.execute(sql, self._params)
        col_names = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(col_names, r)) for r in cur.fetchall()]

        # Deserialize JSON columns
        for row in rows:
            for k, v in row.items():
                if isinstance(v, str) and v.startswith(("{", "[")):
                    try:
                        row[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        pass

        count = None
        if self._count_mode == "exact":
            count_sql = f'SELECT COUNT(*) FROM "{self._table}"'
            count_sql += self._where_clause()
            count = self._conn.execute(count_sql, self._params).fetchone()[0]

        return _QueryResponse(rows, count=count)

    def _exec_insert(self) -> _QueryResponse:
        if not self._data:
            return _QueryResponse([])
        inserted = []
        for rec in self._data:
            rec = self._serialize_record(rec)
            cols = list(rec.keys())
            vals = list(rec.values())
            ph = ",".join("?" for _ in cols)
            col_str = ",".join(f'"{c}"' for c in cols)
            sql = f'INSERT INTO "{self._table}" ({col_str}) VALUES ({ph})'
            self._conn.execute(sql, vals)
            inserted.append(rec)
        self._conn.commit()
        return _QueryResponse(inserted)

    def _exec_upsert(self) -> _QueryResponse:
        if not self._data:
            return _QueryResponse([])
        upserted = []
        for rec in self._data:
            rec = self._serialize_record(rec)
            cols = list(rec.keys())
            vals = list(rec.values())
            ph = ",".join("?" for _ in cols)
            col_str = ",".join(f'"{c}"' for c in cols)
            update_cols = [c for c in cols if c not in (self._on_conflict or "").split(",")]
            if update_cols:
                update_str = ",".join(f'"{c}"=excluded."{c}"' for c in update_cols)
                conflict = self._on_conflict or ""
                conflict_str = ",".join(f'"{c.strip()}"' for c in conflict.split(",") if c.strip())
                sql = (
                    f'INSERT INTO "{self._table}" ({col_str}) VALUES ({ph}) '
                    f"ON CONFLICT ({conflict_str}) DO UPDATE SET {update_str}"
                )
            else:
                sql = f'INSERT OR IGNORE INTO "{self._table}" ({col_str}) VALUES ({ph})'
            self._conn.execute(sql, vals)
            upserted.append(rec)
        self._conn.commit()
        return _QueryResponse(upserted)

    def _exec_update(self) -> _QueryResponse:
        rec = self._serialize_record(self._data)
        set_parts = [f'"{k}" = ?' for k in rec.keys()]
        vals = list(rec.values()) + self._params
        sql = f'UPDATE "{self._table}" SET {",".join(set_parts)}'
        sql += self._where_clause()
        self._conn.execute(sql, vals)
        self._conn.commit()
        return _QueryResponse([rec])

    def _exec_delete(self) -> _QueryResponse:
        sql = f'DELETE FROM "{self._table}"'
        sql += self._where_clause()
        self._conn.execute(sql, self._params)
        self._conn.commit()
        return _QueryResponse([])

    @staticmethod
    def _serialize_record(rec: dict) -> dict:
        """Serialize complex values (dicts/lists) to JSON strings for SQLite."""
        out = {}
        for k, v in rec.items():
            if isinstance(v, (dict, list)):
                out[k] = json.dumps(v, default=str)
            elif isinstance(v, bool):
                out[k] = int(v)
            elif pd.isna(v) if not isinstance(v, str) else False:
                out[k] = None
            elif hasattr(v, "item"):
                out[k] = v.item()
            else:
                out[k] = v
        return out


class _StorageBucket:
    """Mimics Supabase Storage bucket for local file storage."""

    def __init__(self, base_dir: str, bucket: str) -> None:
        self._dir = os.path.join(base_dir, bucket)
        os.makedirs(self._dir, exist_ok=True)

    def upload(self, path: str, data: bytes, _options: dict | None = None) -> None:
        full = os.path.join(self._dir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(data)

    def download(self, path: str) -> bytes:
        full = os.path.join(self._dir, path)
        with open(full, "rb") as f:
            return f.read()


class _StorageClient:
    """Mimics Supabase storage client."""

    def __init__(self, base_dir: str) -> None:
        self._base = base_dir

    def from_(self, bucket: str) -> _StorageBucket:
        return _StorageBucket(self._base, bucket)

    def create_bucket(self, name: str, options: dict | None = None) -> None:
        os.makedirs(os.path.join(self._base, name), exist_ok=True)


class _SQLiteClient:
    """Mimics supabase.Client — provides .table() and .storage."""

    def __init__(self, conn: sqlite3.Connection, storage_dir: str) -> None:
        self._conn = conn
        self.storage = _StorageClient(storage_dir)

    def table(self, name: str) -> _TableQuery:
        return _TableQuery(self._conn, name)


# ====================================================================== #
#  SQLiteLoader — main class
# ====================================================================== #

class SQLiteLoader:
    """SQLite database interface for FinAI platform.

    API-compatible with SupabaseLoader. All ``db.client.table(...)`` calls
    work transparently through the built-in query builder.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", _DEFAULT_DB_PATH)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        storage_dir = os.path.join(os.path.dirname(self.db_path), "storage")
        self.client = _SQLiteClient(self._conn, storage_dir)

        logger.info("SQLite database initialized: {}", self.db_path)

    def init_schema(self) -> None:
        """Create all tables (idempotent)."""
        schema_path = os.path.join(
            os.path.dirname(__file__), "schema_sqlite.sql"
        )
        with open(schema_path, "r", encoding="utf-8") as f:
            sql = f.read()
        self._conn.executescript(sql)
        self._conn.commit()
        logger.info("SQLite schema initialized")

    # ------------------------------------------------------------------ #
    #  Price data
    # ------------------------------------------------------------------ #

    def upsert_prices(self, df: pd.DataFrame) -> int:
        required = {"date", "stock_id", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {required - set(df.columns)}")

        total = 0
        for _, row in df.iterrows():
            self._conn.execute(
                """INSERT INTO stock_prices (date, stock_id, open, high, low, close, volume, adj_close)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (date, stock_id) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume, adj_close=excluded.adj_close""",
                (
                    _to_date_str(row["date"]),
                    row["stock_id"],
                    _safe(row.get("open")),
                    _safe(row.get("high")),
                    _safe(row.get("low")),
                    _safe(row.get("close")),
                    _safe(row.get("volume")),
                    _safe(row.get("adj_close")),
                ),
            )
            total += 1
        self._conn.commit()
        logger.info("Upserted {} price rows", total)
        return total

    def get_prices(
        self,
        stock_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not stock_ids:
            return pd.DataFrame()
        ph = ",".join("?" for _ in stock_ids)
        sql = (
            f"SELECT date, stock_id, open, high, low, close, volume, adj_close "
            f"FROM stock_prices "
            f"WHERE stock_id IN ({ph}) AND date >= ? AND date <= ? "
            f"ORDER BY date"
        )
        params = list(stock_ids) + [start_date, end_date]
        df = pd.read_sql_query(sql, self._conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------ #
    #  Features
    # ------------------------------------------------------------------ #

    def upsert_features(self, df: pd.DataFrame) -> int:
        meta_cols = {"date", "stock_id"}
        feat_cols = [c for c in df.columns if c not in meta_cols]
        total = 0
        for _, row in df.iterrows():
            features = {c: _safe(row[c]) for c in feat_cols}
            self._conn.execute(
                """INSERT INTO stock_features (date, stock_id, features)
                   VALUES (?, ?, ?)
                   ON CONFLICT (date, stock_id) DO UPDATE SET features=excluded.features""",
                (_to_date_str(row["date"]), row["stock_id"], json.dumps(features, default=str)),
            )
            total += 1
        self._conn.commit()
        logger.info("Upserted {} feature rows", total)
        return total

    def get_features(
        self,
        stock_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not stock_ids:
            return pd.DataFrame()
        ph = ",".join("?" for _ in stock_ids)
        sql = (
            f"SELECT date, stock_id, features "
            f"FROM stock_features "
            f"WHERE stock_id IN ({ph}) AND date >= ? AND date <= ? "
            f"ORDER BY date"
        )
        params = list(stock_ids) + [start_date, end_date]
        cur = self._conn.execute(sql, params)
        rows_raw = cur.fetchall()
        if not rows_raw:
            return pd.DataFrame()

        rows = []
        for r in rows_raw:
            row_dict: dict[str, Any] = {"date": r[0], "stock_id": r[1]}
            feats = json.loads(r[2]) if r[2] else {}
            row_dict.update(feats)
            rows.append(row_dict)
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
        version_id = str(uuid.uuid4())

        # Upload model file to local storage
        storage_path = ""
        if file_path and os.path.isfile(file_path):
            fname = os.path.basename(file_path)
            storage_path = f"{model_type}/{fname}"
            with open(file_path, "rb") as f:
                self.client.storage.from_("models").upload(storage_path, f.read())
            logger.info("Stored model locally: {}", storage_path)

        self._conn.execute(
            """INSERT INTO model_versions (id, model_type, metrics, file_path, storage_path,
               is_active, description, created_at)
               VALUES (?, ?, ?, ?, ?, 0, ?, ?)""",
            (
                version_id,
                model_type,
                json.dumps(metrics, default=str),
                file_path,
                storage_path,
                description,
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()
        logger.info("Saved model version {} (type={})", version_id, model_type)
        return version_id

    def get_active_model(self, model_type: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT * FROM model_versions WHERE model_type = ? AND is_active = 1 LIMIT 1",
            (model_type,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        result = dict(zip(cols, row))
        if isinstance(result.get("metrics"), str):
            result["metrics"] = json.loads(result["metrics"])
        return result

    def set_active_model(self, version_id: str) -> None:
        cur = self._conn.execute(
            "SELECT model_type FROM model_versions WHERE id = ?", (version_id,)
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Model version {version_id} not found")
        model_type = row[0]

        self._conn.execute(
            "UPDATE model_versions SET is_active = 0 WHERE model_type = ?",
            (model_type,),
        )
        self._conn.execute(
            "UPDATE model_versions SET is_active = 1 WHERE id = ?",
            (version_id,),
        )
        self._conn.commit()
        logger.info("Activated model {} (type={})", version_id, model_type)

    # ------------------------------------------------------------------ #
    #  Predictions
    # ------------------------------------------------------------------ #

    def save_predictions(self, predictions: pd.DataFrame) -> int:
        required = {"date", "stock_id", "predicted_return", "score", "model_version"}
        if not required.issubset(predictions.columns):
            raise ValueError(f"Missing columns: {required - set(predictions.columns)}")

        total = 0
        for _, row in predictions.iterrows():
            self._conn.execute(
                """INSERT INTO predictions (date, stock_id, predicted_return, score, model_version)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (date, stock_id, model_version) DO UPDATE SET
                     predicted_return=excluded.predicted_return, score=excluded.score""",
                (
                    _to_date_str(row["date"]),
                    row["stock_id"],
                    _safe(row["predicted_return"]),
                    _safe(row["score"]),
                    row.get("model_version"),
                ),
            )
            total += 1
        self._conn.commit()
        logger.info("Saved {} prediction rows", total)
        return total

    def get_latest_predictions(self) -> pd.DataFrame:
        cur = self._conn.execute(
            "SELECT date FROM predictions ORDER BY date DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return pd.DataFrame()
        latest_date = row[0]

        df = pd.read_sql_query(
            "SELECT date, stock_id, predicted_return, score, model_version "
            "FROM predictions WHERE date = ? ORDER BY score DESC",
            self._conn,
            params=[latest_date],
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------ #
    #  Backtest results
    # ------------------------------------------------------------------ #

    def save_backtest(self, result: dict) -> str:
        result_id = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO backtest_results
               (id, run_date, model_type, period_start, period_end, metrics, config, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result_id,
                str(result.get("run_date", date.today())),
                result.get("model_type", ""),
                str(result["period_start"]) if "period_start" in result else None,
                str(result["period_end"]) if "period_end" in result else None,
                json.dumps(result.get("metrics", {}), default=str),
                json.dumps(result.get("config", {}), default=str),
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()
        logger.info("Saved backtest result {}", result_id)
        return result_id

    def get_backtest_history(self, limit: int = 10) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        rows = []
        for r in cur.fetchall():
            row = dict(zip(cols, r))
            for key in ("metrics", "config"):
                if isinstance(row.get(key), str):
                    try:
                        row[key] = json.loads(row[key])
                    except (json.JSONDecodeError, ValueError):
                        pass
            rows.append(row)
        return rows

    def delete_backtest(self, result_id: str) -> None:
        """Delete a backtest result by ID."""
        self.client.table("backtest_results").delete().eq("id", result_id).execute()
        logger.info("Deleted backtest result {}", result_id)

    # ------------------------------------------------------------------ #
    #  Data discovery
    # ------------------------------------------------------------------ #

    def get_available_stock_ids(self, table: str = "stock_features") -> list[str]:
        """Return distinct stock_id values present in *table*."""
        cur = self._conn.execute(
            f'SELECT DISTINCT stock_id FROM "{table}" ORDER BY stock_id'
        )
        return [row[0] for row in cur.fetchall()]

    def get_date_range(self, table: str = "stock_features") -> tuple[str, str] | None:
        """Return (min_date, max_date) for *table*, or None if empty."""
        cur = self._conn.execute(
            f'SELECT MIN(date), MAX(date) FROM "{table}"'
        )
        row = cur.fetchone()
        if row and row[0] and row[1]:
            return (str(row[0])[:10], str(row[1])[:10])
        return None

    # ------------------------------------------------------------------ #
    #  Stock universe
    # ------------------------------------------------------------------ #

    def get_last_date_per_stock(self, table: str = "stock_prices") -> dict[str, str]:
        """Return {stock_id: last_date} for incremental fetching."""
        cur = self._conn.execute(
            f'SELECT stock_id, MAX(date) FROM "{table}" GROUP BY stock_id'
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def upsert_universe(self, df: pd.DataFrame) -> int:
        """Save stock universe snapshot."""
        required = {"stock_id"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {required - set(df.columns)}")

        total = 0
        now = datetime.now().isoformat()
        for _, row in df.iterrows():
            self._conn.execute(
                """INSERT INTO stock_universe (stock_id, stock_name, market, listing_date, market_cap, avg_volume, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (stock_id) DO UPDATE SET
                     stock_name=excluded.stock_name, market=excluded.market,
                     listing_date=excluded.listing_date, market_cap=excluded.market_cap,
                     avg_volume=excluded.avg_volume, updated_at=excluded.updated_at""",
                (
                    row["stock_id"],
                    row.get("stock_name"),
                    row.get("market"),
                    _to_date_str(row["listing_date"]) if pd.notna(row.get("listing_date")) else None,
                    _safe(row.get("market_cap")),
                    _safe(row.get("avg_volume")),
                    now,
                ),
            )
            total += 1
        self._conn.commit()
        logger.info("Upserted {} universe records", total)
        return total

    def get_stock_name_map(self) -> dict[str, str]:
        """Return {stock_id: stock_name} mapping from stock_universe."""
        rows = self._conn.execute(
            "SELECT stock_id, stock_name FROM stock_universe WHERE stock_name IS NOT NULL"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------ #
    #  Stock scores
    # ------------------------------------------------------------------ #

    def save_scores(self, df: pd.DataFrame) -> int:
        """Save scoring results to stock_scores table."""
        required = {"date", "stock_id", "composite_score"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {required - set(df.columns)}")

        total = 0
        for _, row in df.iterrows():
            self._conn.execute(
                """INSERT INTO stock_scores
                   (date, stock_id, composite_score, momentum_score, trend_score,
                    volatility_score, volume_score, ai_score, risk_level,
                    max_drawdown, volatility_ann, win_rate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (date, stock_id) DO UPDATE SET
                     composite_score=excluded.composite_score,
                     momentum_score=excluded.momentum_score,
                     trend_score=excluded.trend_score,
                     volatility_score=excluded.volatility_score,
                     volume_score=excluded.volume_score,
                     ai_score=excluded.ai_score,
                     risk_level=excluded.risk_level,
                     max_drawdown=excluded.max_drawdown,
                     volatility_ann=excluded.volatility_ann,
                     win_rate=excluded.win_rate""",
                (
                    _to_date_str(row["date"]),
                    row["stock_id"],
                    _safe(row.get("composite_score")),
                    _safe(row.get("momentum_score")),
                    _safe(row.get("trend_score")),
                    _safe(row.get("volatility_score")),
                    _safe(row.get("volume_score")),
                    _safe(row.get("ai_score")),
                    row.get("risk_level"),
                    _safe(row.get("max_drawdown")),
                    _safe(row.get("volatility_ann")),
                    _safe(row.get("win_rate")),
                ),
            )
            total += 1
        self._conn.commit()
        logger.info("Saved {} score rows", total)
        return total

    def get_latest_scores(self, limit: int = 50) -> pd.DataFrame:
        """Get latest scores sorted by composite_score DESC."""
        cur = self._conn.execute(
            "SELECT date FROM stock_scores ORDER BY date DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return pd.DataFrame()
        latest_date = row[0]

        df = pd.read_sql_query(
            "SELECT date, stock_id, composite_score, momentum_score, trend_score, "
            "volatility_score, volume_score, ai_score, risk_level, "
            "max_drawdown, volatility_ann, win_rate "
            "FROM stock_scores WHERE date = ? ORDER BY composite_score DESC LIMIT ?",
            self._conn,
            params=[latest_date, limit],
        )
        return df

    def get_stock_score(self, stock_id: str) -> Optional[dict]:
        """Get single stock's latest score + factors."""
        cur = self._conn.execute(
            "SELECT date, stock_id, composite_score, momentum_score, trend_score, "
            "volatility_score, volume_score, ai_score, risk_level, "
            "max_drawdown, volatility_ann, win_rate "
            "FROM stock_scores WHERE stock_id = ? ORDER BY date DESC LIMIT 1",
            (stock_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------- #

def _to_date_str(val: Any) -> str:
    if isinstance(val, str):
        return val[:10]
    if hasattr(val, "isoformat"):
        return val.isoformat()[:10]
    return str(val)[:10]


def _safe(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "item"):
        return val.item()
    return val

"""Supabase project setup and initial data fetch.

Validates connection, creates tables via SQL, creates storage bucket,
and optionally runs first data fetch.

Usage:
    python -m scripts.setup_supabase
    python -m scripts.setup_supabase --init-data
    python -m scripts.setup_supabase --check-only
"""

from __future__ import annotations

import argparse
import sys
import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supabase setup and validation")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check connection, do not create tables",
    )
    parser.add_argument(
        "--init-data",
        action="store_true",
        help="Run initial data fetch after setup",
    )
    parser.add_argument(
        "--stocks",
        type=str,
        default=None,
        help="Comma-separated stock IDs for initial fetch (default: top 5)",
    )
    return parser.parse_args()


def check_env() -> tuple[str, str]:
    """Validate environment variables."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")

    if not url or url == "https://your-project.supabase.co":
        logger.error("SUPABASE_URL not configured. Set it in .env file.")
        logger.info("Get your URL from: https://supabase.com/dashboard → Settings → API")
        sys.exit(1)

    if not key or key == "your-anon-key":
        logger.error("SUPABASE_KEY not configured. Set it in .env file.")
        logger.info("Get your anon key from: https://supabase.com/dashboard → Settings → API")
        sys.exit(1)

    return url, key


def check_connection(url: str, key: str) -> bool:
    """Test Supabase connection."""
    try:
        from supabase import create_client

        client = create_client(url, key)
        # Simple query to verify connection
        client.table("stock_prices").select("id").limit(1).execute()
        logger.info("Supabase connection OK: {}", url[:50] + "...")
        return True
    except Exception as e:
        error_msg = str(e)
        if "relation" in error_msg and "does not exist" in error_msg:
            logger.warning("Connected but tables not created yet")
            return True
        logger.error("Connection failed: {}", e)
        return False


def create_tables(url: str, key: str) -> bool:
    """Create database tables using schema.sql via Supabase RPC or direct SQL."""
    schema_path = os.path.join(os.path.dirname(__file__), "..", "data", "loaders", "schema.sql")
    schema_path = os.path.abspath(schema_path)

    if not os.path.exists(schema_path):
        logger.error("Schema file not found: {}", schema_path)
        return False

    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()

    try:
        from supabase import create_client

        client = create_client(url, key)

        # Split SQL into individual statements
        statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]

        for stmt in statements:
            if not stmt:
                continue
            try:
                client.rpc("exec_sql", {"query": stmt + ";"}).execute()
            except Exception:
                # RPC may not exist, try postgrest approach
                # Tables are typically created via Supabase dashboard SQL editor
                pass

        logger.info("Table creation attempted via RPC ({} statements)", len(statements))
        return True
    except Exception as e:
        logger.warning("Could not create tables via API: {}", e)
        logger.info("Please create tables manually:")
        logger.info("  1. Go to Supabase Dashboard → SQL Editor")
        logger.info("  2. Paste contents of: {}", schema_path)
        logger.info("  3. Click 'Run'")
        return False


def create_storage_bucket(url: str, key: str) -> bool:
    """Create the 'models' storage bucket."""
    try:
        from supabase import create_client

        client = create_client(url, key)
        try:
            client.storage.create_bucket("models", options={"public": False})
            logger.info("Created storage bucket: models")
        except Exception as e:
            if "already exists" in str(e).lower() or "Duplicate" in str(e):
                logger.info("Storage bucket 'models' already exists")
            else:
                logger.warning("Could not create storage bucket: {}", e)
                logger.info("Create it manually: Dashboard → Storage → New bucket → 'models'")
        return True
    except Exception as e:
        logger.warning("Storage bucket setup failed: {}", e)
        return False


def verify_tables(url: str, key: str) -> dict:
    """Verify all required tables exist and return row counts."""
    from supabase import create_client

    client = create_client(url, key)
    tables = ["stock_prices", "stock_features", "model_versions",
              "predictions", "backtest_results", "pipeline_runs"]

    results = {}
    for table in tables:
        try:
            resp = client.table(table).select("id", count="exact").limit(0).execute()
            count = resp.count if resp.count is not None else 0
            results[table] = {"exists": True, "rows": count}
        except Exception as e:
            if "does not exist" in str(e):
                results[table] = {"exists": False, "rows": 0}
            else:
                results[table] = {"exists": True, "rows": -1, "error": str(e)}

    return results


def initial_data_fetch(stock_ids: list[str]) -> None:
    """Run initial data fetch for specified stocks."""
    from datetime import date, timedelta

    from data.collectors.price import PriceCollector
    from data.processors.cleaning import DataCleaner
    from data.processors.features import FeatureEngine
    from data.loaders.supabase import SupabaseLoader

    end_date = str(date.today())
    start_date = str(date.today() - timedelta(days=365))

    logger.info("=== Initial Data Fetch ===")
    logger.info("Stocks: {}", stock_ids)
    logger.info("Period: {} to {}", start_date, end_date)

    # 1. Fetch prices
    collector = PriceCollector()
    cleaner = DataCleaner()
    feature_engine = FeatureEngine()
    db = SupabaseLoader()

    price_data = collector.fetch_batch(stock_ids, start_date, end_date)
    logger.info("Fetched price data for {}/{} stocks", len(price_data), len(stock_ids))

    if not price_data:
        logger.error("No price data fetched")
        return

    # 2. Clean & validate
    valid_stocks: dict = {}
    for sid, df in price_data.items():
        if df.empty:
            logger.warning("{}: empty, skipping", sid)
            continue
        cleaned, report = cleaner.validate_price(df)
        filled = cleaner.fill_missing(cleaned)
        valid_stocks[sid] = filled

    logger.info("Valid stocks: {}/{}", len(valid_stocks), len(price_data))

    # 3. Save prices to Supabase
    for sid, df in valid_stocks.items():
        try:
            price_df = df.reset_index()
            price_df.columns = [c.lower() for c in price_df.columns]
            if "date" not in price_df.columns:
                price_df = price_df.rename(columns={price_df.columns[0]: "date"})
            price_df["stock_id"] = sid
            db.upsert_prices(price_df)
            logger.info("{}: saved {} price rows", sid, len(price_df))
        except Exception as e:
            logger.error("{}: failed to save prices — {}", sid, e)

    # 4. Compute & save features
    import pandas as pd

    all_features = []
    for sid, df in valid_stocks.items():
        try:
            feats = feature_engine.compute_all(df)
            feats["stock_id"] = sid
            feats.index.name = "date"
            all_features.append(feats.reset_index())
        except Exception as e:
            logger.error("{}: feature computation failed — {}", sid, e)

    if all_features:
        features_df = pd.concat(all_features, ignore_index=True)
        try:
            db.upsert_features(features_df)
            logger.info("Saved {} feature rows", len(features_df))
        except Exception as e:
            logger.error("Failed to save features: {}", e)
    else:
        logger.warning("No features computed")

    # 5. Summary
    logger.info("=== Initial Fetch Complete ===")
    logger.info("Stocks: {}", len(valid_stocks))
    total_prices = sum(len(df) for df in valid_stocks.values())
    logger.info("Total price rows: {}", total_prices)
    if all_features:
        logger.info("Total feature rows: {}", len(features_df))


def main() -> None:
    args = parse_args()

    logger.info("=== FinAI Supabase Setup ===")

    # 1. Check environment
    url, key = check_env()

    # 2. Test connection
    if not check_connection(url, key):
        sys.exit(1)

    if args.check_only:
        table_status = verify_tables(url, key)
        logger.info("Table status:")
        for table, info in table_status.items():
            status = "OK" if info["exists"] else "MISSING"
            rows = info["rows"]
            logger.info("  {}: {} ({} rows)", table, status, rows)
        return

    # 3. Create tables
    logger.info("Step 1: Creating tables...")
    create_tables(url, key)

    # 4. Create storage bucket
    logger.info("Step 2: Creating storage bucket...")
    create_storage_bucket(url, key)

    # 5. Verify
    logger.info("Step 3: Verifying tables...")
    table_status = verify_tables(url, key)
    all_exist = all(info["exists"] for info in table_status.values())

    for table, info in table_status.items():
        status = "OK" if info["exists"] else "MISSING"
        logger.info("  {}: {}", table, status)

    if not all_exist:
        logger.warning("Some tables are missing. Please create them manually via SQL Editor.")
        logger.info("Schema file: data/loaders/schema.sql")
    else:
        logger.info("All tables ready!")

    # 6. Initial data fetch
    if args.init_data:
        default_stocks = ["2330", "2317", "2454", "2881", "2882"]
        stock_ids = args.stocks.split(",") if args.stocks else default_stocks
        initial_data_fetch(stock_ids)


if __name__ == "__main__":
    main()

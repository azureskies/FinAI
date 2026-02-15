"""Database setup and initial data fetch.

Creates SQLite database, initializes schema, and optionally fetches
initial price data for Taiwan stocks.

Usage:
    python -m scripts.setup_db
    python -m scripts.setup_db --init-data
    python -m scripts.setup_db --check-only
    python -m scripts.setup_db --init-data --stocks 2330,2317,2454
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Database setup and validation")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check database status, do not create tables",
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


def verify_tables(db) -> dict:
    """Verify all required tables exist and return row counts."""
    tables = [
        "stock_prices", "stock_features", "model_versions",
        "predictions", "backtest_results", "pipeline_runs",
    ]
    results = {}
    for table in tables:
        try:
            resp = db.client.table(table).select("*", count="exact").limit(0).execute()
            count = resp.count if resp.count is not None else 0
            results[table] = {"exists": True, "rows": count}
        except Exception as e:
            if "no such table" in str(e).lower():
                results[table] = {"exists": False, "rows": 0}
            else:
                results[table] = {"exists": True, "rows": -1, "error": str(e)}
    return results


def initial_data_fetch(stock_ids: list[str], db) -> None:
    """Run initial data fetch for specified stocks."""
    from datetime import date, timedelta

    import pandas as pd

    from data.collectors.price import PriceCollector
    from data.processors.cleaning import DataCleaner
    from data.processors.features import FeatureEngine

    end_date = str(date.today())
    start_date = str(date.today() - timedelta(days=365))

    logger.info("=== Initial Data Fetch ===")
    logger.info("Stocks: {}", stock_ids)
    logger.info("Period: {} to {}", start_date, end_date)

    collector = PriceCollector()
    cleaner = DataCleaner()
    feature_engine = FeatureEngine()

    # 1. Fetch prices
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

    # 3. Save prices
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

    logger.info("=== FinAI Database Setup ===")

    from data.loaders import DatabaseLoader

    db = DatabaseLoader()

    # Initialize schema (creates tables if not exist)
    if not args.check_only:
        logger.info("Step 1: Initializing database schema...")
        if hasattr(db, "init_schema"):
            db.init_schema()
        logger.info("Schema initialized")

    # Verify tables
    logger.info("Step 2: Verifying tables...")
    table_status = verify_tables(db)

    for table, info in table_status.items():
        status = "OK" if info["exists"] else "MISSING"
        rows = info["rows"]
        logger.info("  {}: {} ({} rows)", table, status, rows)

    all_exist = all(info["exists"] for info in table_status.values())

    if not all_exist:
        logger.error("Some tables are missing. Run without --check-only to create them.")
        sys.exit(1)
    else:
        logger.info("All tables ready!")

    # Initial data fetch
    if args.init_data:
        default_stocks = ["2330", "2317", "2454", "2881", "2882"]
        stock_ids = args.stocks.split(",") if args.stocks else default_stocks
        initial_data_fetch(stock_ids, db)


if __name__ == "__main__":
    main()

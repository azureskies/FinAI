"""Daily data update pipeline — runs after market close.

Fetches latest prices, computes features, generates predictions,
and stores everything in Supabase.

Usage:
    python -m scripts.daily_update
    python -m scripts.daily_update --dry-run
    python -m scripts.daily_update --stock-id 2330
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd
import yaml
from loguru import logger

from data.collectors.price import PriceCollector
from data.loaders.supabase import SupabaseLoader
from data.processors.features import FeatureEngine

_CONFIG_PATH = "configs/data_sources.yaml"
_LOOKBACK_DAYS = 300  # enough history for longest indicator window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily data update pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without writing to Supabase",
    )
    parser.add_argument(
        "--stock-id",
        type=str,
        default=None,
        help="Run for a single stock instead of the full universe",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date (default: today), format YYYY-MM-DD",
    )
    return parser.parse_args()


def _load_universe(stock_id: str | None) -> list[str]:
    """Load stock universe from config or use single stock."""
    if stock_id:
        return [stock_id]

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Try to load from Supabase universe table or fallback to a default list
    # For now, use a representative set of large-cap TW stocks
    default_universe = [
        "2330", "2317", "2454", "2308", "2881",
        "2882", "2891", "2303", "1301", "1303",
        "2412", "3711", "2886", "2884", "3008",
        "2357", "2382", "6505", "1326", "2002",
    ]
    logger.info("Using default universe: {} stocks", len(default_universe))
    return default_universe


def main() -> None:
    args = parse_args()
    target_date = args.date or str(date.today())
    start_date = str(date.fromisoformat(target_date) - timedelta(days=_LOOKBACK_DAYS))

    logger.info("=== Daily Update Pipeline ===")
    logger.info("Target date: {} | Dry run: {}", target_date, args.dry_run)

    # Initialize components
    collector = PriceCollector()
    feature_engine = FeatureEngine()
    db = None if args.dry_run else SupabaseLoader()

    stock_ids = _load_universe(args.stock_id)
    logger.info("Processing {} stocks", len(stock_ids))

    # ------------------------------------------------------------------ #
    # 1. Fetch latest price data
    # ------------------------------------------------------------------ #
    logger.info("Step 1: Fetching price data...")
    price_data = collector.fetch_batch(stock_ids, start_date, target_date)
    logger.info("Fetched price data for {}/{} stocks", len(price_data), len(stock_ids))

    if not price_data:
        logger.error("No price data fetched — aborting")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Data quality checks
    # ------------------------------------------------------------------ #
    logger.info("Step 2: Running data quality checks...")
    valid_stocks: dict[str, pd.DataFrame] = {}
    for sid, df in price_data.items():
        if df.empty:
            logger.warning("{}: empty DataFrame, skipping", sid)
            continue
        missing_rate = df[["close"]].isna().mean().iloc[0]
        if missing_rate > 0.1:
            logger.warning("{}: too many missing values ({:.1%}), skipping", sid, missing_rate)
            continue
        valid_stocks[sid] = df

    logger.info("Quality check passed: {}/{} stocks", len(valid_stocks), len(price_data))

    # ------------------------------------------------------------------ #
    # 3. Compute features
    # ------------------------------------------------------------------ #
    logger.info("Step 3: Computing features...")
    all_features: list[pd.DataFrame] = []
    for sid, df in valid_stocks.items():
        try:
            feats = feature_engine.compute_all(df)
            feats["stock_id"] = sid
            feats.index.name = "date"
            all_features.append(feats.reset_index())
        except Exception as e:
            logger.error("{}: feature computation failed — {}", sid, e)

    if not all_features:
        logger.error("No features computed — aborting")
        sys.exit(1)

    features_df = pd.concat(all_features, ignore_index=True)
    logger.info("Computed features: {} rows, {} columns", *features_df.shape)

    # ------------------------------------------------------------------ #
    # 4. Load active model and generate predictions
    # ------------------------------------------------------------------ #
    logger.info("Step 4: Generating predictions...")
    predictions_list: list[dict] = []
    active_model = None
    model_version_id = None

    if db:
        active_model = db.get_active_model("ensemble")
        if active_model:
            model_version_id = active_model["id"]

    if active_model:
        logger.info("Active model: {} ({})", active_model["model_type"], model_version_id)
        # Use the latest row per stock for prediction
        latest_features = features_df.sort_values("date").groupby("stock_id").tail(1)
        feat_cols = [
            c for c in latest_features.columns if c not in {"date", "stock_id"}
        ]
        X = latest_features[feat_cols].fillna(0)

        # Load model from Supabase storage or local cache
        model_path = active_model.get("file_path", "")
        model = None

        if model_path and db:
            import tempfile
            try:
                local_path = os.path.join(tempfile.gettempdir(), os.path.basename(model_path))
                if not os.path.exists(local_path):
                    storage_data = db.client.storage.from_("models").download(model_path)
                    with open(local_path, "wb") as f:
                        f.write(storage_data)
                    logger.info("Downloaded model from storage: {}", model_path)
                from models.training import ModelTrainer
                trainer = ModelTrainer()
                model = trainer.load_model(local_path)
                logger.info("Loaded model: {}", type(model).__name__)
            except Exception as e:
                logger.warning("Failed to load model from storage: {}", e)

        if model is not None and hasattr(model, "predict"):
            scores = model.predict(X)
        else:
            logger.warning("Model not available, falling back to standardized feature mean")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scores = X_scaled.mean(axis=1)

        for idx, (_, row) in enumerate(latest_features.iterrows()):
            predictions_list.append({
                "date": target_date,
                "stock_id": row["stock_id"],
                "predicted_return": float(scores[idx]),
                "score": float(scores[idx]),
                "model_version": model_version_id,
            })
    else:
        logger.warning("No active model found — skipping predictions")

    # ------------------------------------------------------------------ #
    # 5. Save everything to Supabase
    # ------------------------------------------------------------------ #
    if args.dry_run:
        logger.info("[DRY RUN] Would save {} price dfs, {} feature rows, {} predictions",
                     len(valid_stocks), len(features_df), len(predictions_list))
    else:
        logger.info("Step 5: Saving to Supabase...")

        # Save prices
        for sid, df in valid_stocks.items():
            try:
                price_df = df.reset_index()
                if "stock_id" not in price_df.columns:
                    price_df["stock_id"] = sid
                db.upsert_prices(price_df)
            except Exception as e:
                logger.error("{}: failed to save prices — {}", sid, e)

        # Save features
        try:
            db.upsert_features(features_df)
        except Exception as e:
            logger.error("Failed to save features: {}", e)

        # Save predictions
        if predictions_list:
            try:
                pred_df = pd.DataFrame(predictions_list)
                db.save_predictions(pred_df)
            except Exception as e:
                logger.error("Failed to save predictions: {}", e)

    # ------------------------------------------------------------------ #
    # 6. Log summary
    # ------------------------------------------------------------------ #
    logger.info("=== Daily Update Summary ===")
    logger.info("Stocks processed: {}/{}", len(valid_stocks), len(stock_ids))
    logger.info("Feature rows: {}", len(features_df))
    logger.info("Predictions: {}", len(predictions_list))
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()

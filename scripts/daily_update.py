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
import time
from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from data.collectors.price import PriceCollector
from data.loaders import DatabaseLoader
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
    parser.add_argument(
        "--notify",
        action="store_true",
        default=True,
        help="Send alerts on completion/failure (default: True)",
    )
    parser.add_argument(
        "--no-notify",
        dest="notify",
        action="store_false",
        help="Disable alert notifications",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Force full data refresh instead of incremental",
    )
    return parser.parse_args()


def _load_universe(stock_id: str | None) -> tuple[list[str], dict[str, str]]:
    """Load stock universe from UniverseCollector or use single stock.

    Returns:
        (stock_ids, market_map) where market_map is {stock_id: "twse"|"otc"}.
    """
    if stock_id:
        return [stock_id], {stock_id: "twse"}

    try:
        from data.collectors.universe import UniverseCollector
        collector = UniverseCollector()
        universe_df = collector.get_universe(date.today())
        if not universe_df.empty and "stock_id" in universe_df.columns:
            # Exclude emerging market stocks (low liquidity)
            if "market" in universe_df.columns:
                universe_df = universe_df[
                    universe_df["market"].isin(["twse", "otc", "tpex"])
                ]
            stock_ids = universe_df["stock_id"].tolist()

            # Build market map: tpex is also OTC for yfinance purposes
            market_map: dict[str, str] = {}
            if "market" in universe_df.columns:
                for _, row in universe_df.iterrows():
                    mkt = row["market"]
                    market_map[row["stock_id"]] = "otc" if mkt in ("otc", "tpex") else "twse"
            else:
                market_map = {sid: "twse" for sid in stock_ids}

            logger.info("Loaded {} stocks from UniverseCollector", len(stock_ids))

            # Save universe to DB
            db = DatabaseLoader()
            if hasattr(db, "init_schema"):
                db.init_schema()
            if hasattr(db, "upsert_universe"):
                db.upsert_universe(universe_df)
            db.close()

            return stock_ids, market_map
    except Exception as e:
        logger.warning("Failed to fetch universe from API: {}", e)

    # Fallback to default list (all TWSE)
    default_universe = [
        "2330", "2317", "2454", "2308", "2881",
        "2882", "2891", "2303", "1301", "1303",
        "2412", "3711", "2886", "2884", "3008",
        "2357", "2382", "6505", "1326", "2002",
    ]
    logger.info("Using default universe: {} stocks", len(default_universe))
    return default_universe, {sid: "twse" for sid in default_universe}


def main() -> None:
    args = parse_args()
    target_date = args.date or str(date.today())
    start_date = str(date.fromisoformat(target_date) - timedelta(days=_LOOKBACK_DAYS))

    logger.info("=== Daily Update Pipeline ===")
    logger.info("Target date: {} | Dry run: {}", target_date, args.dry_run)

    # Initialize components
    collector = PriceCollector()
    feature_engine = FeatureEngine()
    if args.dry_run:
        db = None
    else:
        db = DatabaseLoader()
        if hasattr(db, "init_schema"):
            db.init_schema()

    # Pipeline tracking and alerting
    tracker = None
    alert_mgr = None
    run_id = None

    if db and not args.dry_run:
        from monitoring.alerts import AlertLevel, AlertManager
        from monitoring.tracker import PipelineTracker

        tracker = PipelineTracker(db)
        alert_mgr = AlertManager() if args.notify else None
        run_id = tracker.start_run(
            "daily_update", metadata={"target_date": target_date}
        )

    stock_ids, market_map = _load_universe(args.stock_id)
    logger.info("Processing {} stocks", len(stock_ids))

    try:
        # -------------------------------------------------------------- #
        # 1. Fetch latest price data (incremental or full)
        # -------------------------------------------------------------- #
        logger.info("Step 1: Fetching price data...")
        if args.full_refresh or db is None:
            logger.info("Full refresh mode — fetching all data via batch download")
            price_data = collector.fetch_batch(
                stock_ids, start_date, target_date, market_map=market_map,
            )
        else:
            last_dates = db.get_last_date_per_stock()
            price_data = {}
            batch_size = 50
            for batch_start in range(0, len(stock_ids), batch_size):
                batch = stock_ids[batch_start:batch_start + batch_size]
                logger.info(
                    "Batch {}/{}: processing {} stocks",
                    batch_start // batch_size + 1,
                    (len(stock_ids) + batch_size - 1) // batch_size,
                    len(batch),
                )
                for sid in batch:
                    mkt = market_map.get(sid, "twse")
                    last = last_dates.get(sid)
                    if last:
                        df = collector.fetch_incremental(sid, last, market=mkt)
                    else:
                        df = collector.fetch(sid, start_date, target_date, market=mkt)
                    if df is not None and not df.empty:
                        price_data[sid] = df
                    time.sleep(collector.rate_limit)
        logger.info("Fetched price data for {}/{} stocks", len(price_data), len(stock_ids))

        if not price_data:
            raise RuntimeError("No price data fetched")

        # -------------------------------------------------------------- #
        # 2. Data quality checks
        # -------------------------------------------------------------- #
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

        # -------------------------------------------------------------- #
        # 2.5: For incremental mode, load full history for feature computation
        # -------------------------------------------------------------- #
        if not args.full_refresh and db is not None:
            logger.info("Step 2.5: Loading full history for feature computation...")
            for sid in list(valid_stocks.keys()):
                full_df = db.get_prices([sid], start_date, target_date)
                if not full_df.empty:
                    full_df = full_df.set_index("date")
                    if "stock_id" in full_df.columns:
                        full_df = full_df.drop(columns=["stock_id"])
                    full_df["stock_id"] = sid
                    valid_stocks[sid] = full_df

        # -------------------------------------------------------------- #
        # 3. Compute features
        # -------------------------------------------------------------- #
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
            raise RuntimeError("No features computed")

        features_df = pd.concat(all_features, ignore_index=True)
        logger.info("Computed features: {} rows, {} columns", *features_df.shape)

        # -------------------------------------------------------------- #
        # 4. Load active model and generate predictions
        # -------------------------------------------------------------- #
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

        # -------------------------------------------------------------- #
        # 5.5 Score all stocks
        # -------------------------------------------------------------- #
        logger.info("Step 5.5: Computing stock scores...")
        scores_df = pd.DataFrame()
        try:
            from models.scoring import StockScorer
            scorer = StockScorer()

            # Prepare latest features per stock
            latest_features = features_df.sort_values("date").groupby("stock_id").tail(1)

            # Prepare predictions
            pred_df = pd.DataFrame(predictions_list) if predictions_list else None

            # Prepare price data for risk metrics (all scored stocks, batched)
            all_prices = pd.DataFrame()
            if db is not None:
                scored_ids = latest_features["stock_id"].tolist()
                batch_sz = 500
                price_parts = []
                for bi in range(0, len(scored_ids), batch_sz):
                    batch_ids = scored_ids[bi:bi + batch_sz]
                    p = db.get_prices(batch_ids, start_date, target_date)
                    if not p.empty:
                        price_parts.append(p)
                if price_parts:
                    all_prices = pd.concat(price_parts, ignore_index=True)

            scores_df = scorer.score_universe(latest_features, pred_df, all_prices)
            if not scores_df.empty:
                scores_df["date"] = target_date
                logger.info("Computed scores for {} stocks", len(scores_df))
        except Exception as e:
            logger.error("Failed to compute scores: {}", e)

        # -------------------------------------------------------------- #
        # 5. Save everything to Supabase
        # -------------------------------------------------------------- #
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

            # Save scores
            if not scores_df.empty:
                try:
                    db.save_scores(scores_df)
                except Exception as e:
                    logger.error("Failed to save scores: {}", e)

        # -------------------------------------------------------------- #
        # 6. Log summary and record success
        # -------------------------------------------------------------- #
        summary = {
            "stocks_processed": len(valid_stocks),
            "stocks_total": len(stock_ids),
            "feature_rows": len(features_df),
            "predictions": len(predictions_list),
            "scores": len(scores_df),
        }
        logger.info("=== Daily Update Summary ===")
        logger.info("Stocks processed: {}/{}", len(valid_stocks), len(stock_ids))
        logger.info("Feature rows: {}", len(features_df))
        logger.info("Predictions: {}", len(predictions_list))
        logger.info("Pipeline completed successfully")

        if tracker and run_id:
            tracker.finish_run(run_id, status="success", metrics=summary)
        if alert_mgr:
            from monitoring.alerts import AlertLevel
            alert_mgr.send_alert(
                level=AlertLevel.INFO,
                title="Daily update completed",
                message=(
                    f"Stocks: {len(valid_stocks)}/{len(stock_ids)}, "
                    f"Features: {len(features_df)} rows, "
                    f"Predictions: {len(predictions_list)}"
                ),
            )

    except Exception as exc:
        logger.error("Daily update pipeline failed: {}", exc)
        if tracker and run_id:
            tracker.finish_run(run_id, status="failed", error=str(exc))
        if alert_mgr:
            from monitoring.alerts import AlertLevel
            alert_mgr.send_alert(
                level=AlertLevel.CRITICAL,
                title="Daily update FAILED",
                message=str(exc),
            )
        sys.exit(1)


if __name__ == "__main__":
    main()

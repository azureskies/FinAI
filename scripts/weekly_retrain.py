"""Weekly model retraining — runs on weekends.

Loads historical data, trains all models, compares with current active model,
and promotes the new model if it shows meaningful improvement.

Usage:
    python -m scripts.weekly_retrain
    python -m scripts.weekly_retrain --dry-run
    python -m scripts.weekly_retrain --model-type xgboost
    python -m scripts.weekly_retrain --optimize
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from data.loaders.supabase import SupabaseLoader
from data.processors.features import FeatureEngine
from models.tree_models import RandomForestPredictor, XGBoostPredictor

_MODEL_PARAMS_PATH = "configs/model_params.yaml"
_IC_IMPROVEMENT_THRESHOLD = 0.05  # 5% relative improvement to promote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly model retraining pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate without saving to Supabase",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Train only a specific model type (ridge, random_forest, xgboost, ensemble)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable Optuna hyperparameter tuning before training",
    )
    return parser.parse_args()


def _load_model_params() -> dict:
    with open(_MODEL_PARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Information Coefficient (Spearman rank correlation)."""
    if len(y_true) < 10:
        return 0.0
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic) if not np.isnan(ic) else 0.0


def _train_ridge(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> Ridge:
    model = Ridge(alpha=params.get("alpha", 1.0))
    model.fit(X_train, y_train)
    return model


def _train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 10),
        min_samples_leaf=params.get("min_samples_leaf", 20),
        max_features=params.get("max_features", "sqrt"),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", -1),
    )
    model.fit(X_train, y_train)
    return model


def _train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed, skipping")
        return None

    model = xgb.XGBRegressor(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 10),
        random_state=params.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    return model


def _save_model_file(model, model_type: str) -> str:
    """Save model to a temporary pickle file and return the path."""
    tmpdir = tempfile.mkdtemp(prefix="finai_model_")
    file_path = os.path.join(tmpdir, f"{model_type}_{date.today().isoformat()}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    return file_path


def main() -> None:
    args = parse_args()
    params = _load_model_params()

    logger.info("=== Weekly Retrain Pipeline ===")
    logger.info("Dry run: {}", args.dry_run)

    # ------------------------------------------------------------------ #
    # 1. Load historical data from Supabase
    # ------------------------------------------------------------------ #
    logger.info("Step 1: Loading historical data...")
    db = SupabaseLoader()

    # Pipeline tracking and alerting
    tracker = None
    alert_mgr = None
    run_id = None

    if not args.dry_run:
        from monitoring.alerts import AlertLevel, AlertManager
        from monitoring.tracker import PipelineTracker

        tracker = PipelineTracker(db)
        alert_mgr = AlertManager()
        run_id = tracker.start_run(
            "weekly_retrain",
            metadata={"optimize": args.optimize, "model_type": args.model_type},
        )

    try:
        _run_retrain_pipeline(args, params, db, tracker, alert_mgr, run_id)
    except Exception as exc:
        logger.error("Weekly retrain pipeline failed: {}", exc)
        if tracker and run_id:
            tracker.finish_run(run_id, status="failed", error=str(exc))
        if alert_mgr:
            from monitoring.alerts import AlertLevel
            alert_mgr.send_alert(
                level=AlertLevel.CRITICAL,
                title="Weekly retrain FAILED",
                message=str(exc),
            )
        sys.exit(1)


def _run_retrain_pipeline(args, params, db, tracker, alert_mgr, run_id) -> None:
    """Core retrain pipeline logic, extracted for error handling."""
    training_cfg = params.get("training", {}).get("time_split", {})
    train_start = training_cfg.get("train_start", "2018-01-01")
    test_end = training_cfg.get("test_end", str(date.today()))
    val_start = training_cfg.get("val_start", "2022-01-01")

    # Load a representative universe
    default_universe = [
        "2330", "2317", "2454", "2308", "2881",
        "2882", "2891", "2303", "1301", "1303",
        "2412", "3711", "2886", "2884", "3008",
        "2357", "2382", "6505", "1326", "2002",
    ]

    features_df = db.get_features(default_universe, train_start, test_end)
    if features_df.empty:
        raise RuntimeError("No feature data found")
    logger.info("Loaded features: {} rows", len(features_df))

    prices_df = db.get_prices(default_universe, train_start, test_end)
    if prices_df.empty:
        raise RuntimeError("No price data found")

    # ------------------------------------------------------------------ #
    # 2. Prepare features and target
    # ------------------------------------------------------------------ #
    logger.info("Step 2: Preparing features and target...")
    target_days = params.get("prediction", {}).get("target_days", 5)

    # Compute forward return as target
    prices_df = prices_df.sort_values(["stock_id", "date"])
    prices_df["forward_return"] = (
        prices_df.groupby("stock_id")["close"]
        .transform(lambda s: s.shift(-target_days) / s - 1)
    )

    # Merge features with target
    merged = features_df.merge(
        prices_df[["date", "stock_id", "forward_return"]],
        on=["date", "stock_id"],
        how="inner",
    )
    merged = merged.dropna(subset=["forward_return"])
    logger.info("Merged dataset: {} rows", len(merged))

    # Split train/validation
    meta_cols = {"date", "stock_id", "forward_return"}
    feat_cols = [c for c in merged.columns if c not in meta_cols]

    train_mask = merged["date"] < val_start
    val_mask = merged["date"] >= val_start

    X_train = merged.loc[train_mask, feat_cols].fillna(0)
    y_train = merged.loc[train_mask, "forward_return"]
    X_val = merged.loc[val_mask, feat_cols].fillna(0)
    y_val = merged.loc[val_mask, "forward_return"]

    logger.info("Train: {} rows | Validation: {} rows", len(X_train), len(X_val))

    if len(X_train) < 100 or len(X_val) < 50:
        raise RuntimeError("Insufficient data for training")

    # ------------------------------------------------------------------ #
    # 3. Optuna hyperparameter optimization (optional)
    # ------------------------------------------------------------------ #
    optuna_cfg = params.get("optuna", {})
    default_n_trials = optuna_cfg.get("n_trials", 50)
    default_timeout = optuna_cfg.get("timeout", 3600)
    # Per-model timeout: divide total timeout among optimizable models
    per_model_timeout = 600  # 10 minutes per model as reasonable default

    optimized_params: dict[str, dict] = {}  # model_type -> best_params

    if args.optimize:
        logger.info("Step 3a: Running Optuna hyperparameter optimization...")
        optimize_targets = {
            "random_forest": RandomForestPredictor,
            "xgboost": XGBoostPredictor,
        }
        model_types_to_train = (
            [args.model_type] if args.model_type else ["ridge", "random_forest", "xgboost"]
        )

        for model_type in model_types_to_train:
            if model_type not in optimize_targets:
                continue

            logger.info("Optimizing {} with Optuna (n_trials={}, timeout={}s)...",
                        model_type, default_n_trials, per_model_timeout)
            try:
                predictor = optimize_targets[model_type]()
                best_params = predictor.optimize_hyperparameters(
                    X_train, y_train,
                    n_trials=default_n_trials,
                    timeout=per_model_timeout,
                )
                optimized_params[model_type] = best_params
                logger.info("{}: Optuna best params: {}", model_type, best_params)
            except Exception as e:
                logger.error("{}: Optuna optimization failed — {}", model_type, e)

    # ------------------------------------------------------------------ #
    # 3b. Train all models
    # ------------------------------------------------------------------ #
    logger.info("Step 3b: Training models...")
    model_configs = params.get("models", {})
    model_types_to_train = (
        [args.model_type] if args.model_type else ["ridge", "random_forest", "xgboost"]
    )

    trainers = {
        "ridge": _train_ridge,
        "random_forest": _train_random_forest,
        "xgboost": _train_xgboost,
    }

    trained_models: dict[str, tuple] = {}  # model_type -> (model, metrics)

    for model_type in model_types_to_train:
        if model_type not in trainers:
            logger.warning("Unknown model type: {}, skipping", model_type)
            continue

        # Use optimized params if available, otherwise fall back to config
        if model_type in optimized_params:
            model_params = optimized_params[model_type]
            logger.info("Training {} with Optuna-optimized params...", model_type)
        else:
            model_params = model_configs.get(model_type, {})
            logger.info("Training {} with default params...", model_type)

        try:
            model = trainers[model_type](X_train, y_train, model_params)
            if model is None:
                continue

            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_ic = _compute_ic(y_train.values, train_pred)
            val_ic = _compute_ic(y_val.values, val_pred)

            metrics = {
                "train_ic": round(train_ic, 4),
                "val_ic": round(val_ic, 4),
                "train_size": len(X_train),
                "val_size": len(X_val),
                "n_features": len(feat_cols),
            }
            # Store best params in metadata when optimize was used
            if model_type in optimized_params:
                metrics["optimized_params"] = optimized_params[model_type]

            trained_models[model_type] = (model, metrics)
            logger.info("{}: train_ic={:.4f}, val_ic={:.4f}", model_type, train_ic, val_ic)

        except Exception as e:
            logger.error("{}: training failed — {}", model_type, e)

    # Build ensemble
    if "ensemble" in (model_types_to_train if args.model_type else ["ensemble"]):
        ensemble_cfg = model_configs.get("ensemble", {})
        weights = ensemble_cfg.get("weights", {"random_forest": 0.4, "xgboost": 0.6})

        ensemble_pred = np.zeros(len(X_val))
        total_weight = 0.0
        ensemble_components = {}

        for mt, w in weights.items():
            if mt in trained_models:
                model, _ = trained_models[mt]
                ensemble_pred += w * model.predict(X_val)
                total_weight += w
                ensemble_components[mt] = w

        if total_weight > 0:
            ensemble_pred /= total_weight
            ensemble_ic = _compute_ic(y_val.values, ensemble_pred)
            ensemble_metrics = {
                "val_ic": round(ensemble_ic, 4),
                "components": ensemble_components,
                "n_features": len(feat_cols),
            }
            trained_models["ensemble"] = (ensemble_components, ensemble_metrics)
            logger.info("ensemble: val_ic={:.4f}", ensemble_ic)

    # ------------------------------------------------------------------ #
    # 4 & 5. Compare with current active model and conditionally promote
    # ------------------------------------------------------------------ #
    logger.info("Step 4: Comparing with current active models...")
    promotion_log: list[str] = []

    for model_type, (model, metrics) in trained_models.items():
        new_ic = metrics.get("val_ic", 0.0)
        current = db.get_active_model(model_type)

        should_promote = False
        if current is None:
            logger.info("{}: no active model found — will promote", model_type)
            should_promote = True
        else:
            current_ic = current.get("metrics", {}).get("val_ic", 0.0)
            improvement = (new_ic - current_ic) / max(abs(current_ic), 1e-6)
            logger.info(
                "{}: current_ic={:.4f}, new_ic={:.4f}, improvement={:.1%}",
                model_type, current_ic, new_ic, improvement,
            )
            if improvement > _IC_IMPROVEMENT_THRESHOLD:
                should_promote = True

        if should_promote and not args.dry_run:
            # Save model file (skip for ensemble dict)
            file_path = ""
            if not isinstance(model, dict):
                file_path = _save_model_file(model, model_type)

            version_id = db.save_model_version(
                model_type=model_type,
                metrics=metrics,
                file_path=file_path,
            )
            db.set_active_model(version_id)
            promotion_log.append(f"{model_type} -> {version_id} (ic={new_ic:.4f})")
        elif should_promote:
            promotion_log.append(f"[DRY RUN] {model_type} would be promoted (ic={new_ic:.4f})")

    # ------------------------------------------------------------------ #
    # 7. Log comparison report and record completion
    # ------------------------------------------------------------------ #
    logger.info("=== Weekly Retrain Summary ===")
    logger.info("Models trained: {}", list(trained_models.keys()))
    for model_type, (_, metrics) in trained_models.items():
        logger.info("  {}: {}", model_type, metrics)
    if promotion_log:
        logger.info("Promotions:")
        for line in promotion_log:
            logger.info("  {}", line)
    else:
        logger.info("No models promoted (no significant improvement)")
    logger.info("Pipeline completed successfully")

    # Record pipeline success
    summary_metrics = {
        "models_trained": list(trained_models.keys()),
        "promotions": promotion_log,
        "model_ics": {mt: m[1].get("val_ic", 0) for mt, m in trained_models.items()},
    }
    if tracker and run_id:
        tracker.finish_run(run_id, status="success", metrics=summary_metrics)

    if alert_mgr:
        from monitoring.alerts import AlertLevel

        # Send promotion alert if any models were promoted
        if promotion_log and not args.dry_run:
            alert_mgr.send_alert(
                level=AlertLevel.WARNING,
                title="Model promoted",
                message="\n".join(promotion_log),
            )

        # Send completion summary
        ic_summary = ", ".join(
            f"{mt}: IC={m[1].get('val_ic', 0):.4f}"
            for mt, m in trained_models.items()
        )
        alert_mgr.send_alert(
            level=AlertLevel.INFO,
            title="Weekly retrain completed",
            message=f"Models: {ic_summary}. Promotions: {len(promotion_log)}",
        )


if __name__ == "__main__":
    main()

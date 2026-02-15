"""Unified training pipeline for all models."""

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.stats import spearmanr

from models.baseline import RidgePredictor
from models.ensemble import EnsemblePredictor
from models.tree_models import RandomForestPredictor, XGBoostPredictor


class ModelTrainer:
    """Unified training pipeline."""

    def __init__(self, config_path: str = "configs/model_params.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        logger.info(f"ModelTrainer initialized with config: {config_path}")

    def create_target(self, df: pd.DataFrame, days: Optional[int] = None) -> pd.Series:
        """Create forward return target (future N-day return).

        IMPORTANT: This creates lookahead, only use for training labels.

        Args:
            df: DataFrame with 'close' column.
            days: Forward return period. Defaults to config value.

        Returns:
            Series of forward N-day returns.
        """
        if days is None:
            days = self.config["prediction"]["target_days"]

        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        target = df["close"].shift(-days) / df["close"] - 1
        logger.info(
            f"Created {days}-day forward return target, "
            f"valid samples: {target.notna().sum()}/{len(target)}"
        )
        return target

    def time_split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split data by time period from config.

        Args:
            df: DataFrame with DatetimeIndex.

        Returns:
            Dict with 'train', 'val', 'test' DataFrames.
        """
        split_cfg = self.config["training"]["time_split"]

        train_start = split_cfg["train_start"]
        train_end = split_cfg["train_end"]
        val_start = split_cfg["val_start"]
        val_end = split_cfg["val_end"]
        test_start = split_cfg["test_start"]
        test_end = split_cfg["test_end"]

        train = df.loc[train_start:train_end]
        val = df.loc[val_start:val_end]
        test = df.loc[test_start:test_end]

        logger.info(
            f"Time split - train: {len(train)} rows ({train_start}~"
            f"{train_end}), val: {len(val)} rows, test: {len(test)} rows"
        )
        return {"train": train, "val": val, "test": test}

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Train baseline + RF + XGBoost + Ensemble, return all models.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.

        Returns:
            Dict of model name -> fitted model.
        """
        models = {}
        model_cfg = self.config["models"]

        # Ridge baseline
        logger.info("Training Ridge baseline...")
        ridge = RidgePredictor(alpha=model_cfg["ridge"]["alpha"])
        ridge.fit(X_train, y_train)
        val_preds = ridge.predict(X_val)
        ic, _ = spearmanr(y_val, val_preds)
        logger.info(f"Ridge val IC: {ic:.4f}")
        models["ridge"] = ridge

        # Random Forest
        logger.info("Training Random Forest...")
        rf_params = {k: v for k, v in model_cfg["random_forest"].items() if k != "description"}
        rf = RandomForestPredictor(params=rf_params)
        rf.fit(X_train, y_train)
        val_preds = rf.predict(X_val)
        ic, _ = spearmanr(y_val, val_preds)
        logger.info(f"Random Forest val IC: {ic:.4f}")
        models["random_forest"] = rf

        # XGBoost
        logger.info("Training XGBoost...")
        xgb_params = {k: v for k, v in model_cfg["xgboost"].items() if k != "description"}
        xgb = XGBoostPredictor(params=xgb_params)
        xgb.fit(X_train, y_train)
        val_preds = xgb.predict(X_val)
        ic, _ = spearmanr(y_val, val_preds)
        logger.info(f"XGBoost val IC: {ic:.4f}")
        models["xgboost"] = xgb

        # Ensemble
        logger.info("Creating Ensemble...")
        ensemble_weights = model_cfg["ensemble"]["weights"]
        ensemble = EnsemblePredictor(
            models={"random_forest": rf, "xgboost": xgb},
            weights=ensemble_weights,
        )
        # Ensemble sub-models are already fitted, just mark as fitted
        ensemble._is_fitted = True
        val_preds = ensemble.predict(X_val)
        ic, _ = spearmanr(y_val, val_preds)
        logger.info(f"Ensemble val IC: {ic:.4f}")
        models["ensemble"] = ensemble

        return models

    def compare_models(
        self,
        models: dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Compare all models on test set.

        Metrics: IC, Rank IC, Sharpe (of top quintile), MSE.

        Args:
            models: Dict of model name -> fitted model.
            X_test: Test features.
            y_test: Test target.

        Returns:
            DataFrame with model comparison metrics.
        """
        results = []

        for name, model in models.items():
            preds = model.predict(X_test)

            # IC (Information Coefficient) = Spearman correlation
            ic, _ = spearmanr(y_test, preds)

            # Rank IC = Spearman correlation of ranks
            rank_preds = pd.Series(preds).rank()
            rank_actual = y_test.rank()
            rank_ic, _ = spearmanr(rank_actual, rank_preds)

            # MSE
            mse = float(np.mean((y_test.values - preds) ** 2))

            # Sharpe of top quintile
            pred_series = pd.Series(preds, index=y_test.index)
            top_quintile_mask = pred_series >= pred_series.quantile(0.8)
            top_returns = y_test[top_quintile_mask]
            if len(top_returns) > 1 and top_returns.std() > 0:
                sharpe = float(top_returns.mean() / top_returns.std() * np.sqrt(252))
            else:
                sharpe = 0.0

            results.append({
                "model": name,
                "IC": round(ic, 4),
                "Rank_IC": round(rank_ic, 4),
                "Sharpe_Top20": round(sharpe, 4),
                "MSE": round(mse, 6),
            })
            logger.info(f"{name} - IC: {ic:.4f}, Rank IC: {rank_ic:.4f}, Sharpe: {sharpe:.4f}")

        return pd.DataFrame(results).set_index("model")

    def save_model(self, model: Any, path: str) -> None:
        """Save model to pickle.

        Args:
            model: Fitted model object.
            path: File path to save to.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> Any:
        """Load model from pickle.

        Args:
            path: File path to load from.

        Returns:
            Loaded model object.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    np.random.seed(42)
    n_samples, n_features = 1000, 30

    # Create synthetic data with date index
    dates = pd.date_range("2018-01-01", periods=n_samples, freq="B")
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
        index=dates,
    )
    # Simulate a target with some signal
    signal = X["feat_0"] * 0.3 + X["feat_1"] * 0.2 + np.random.randn(n_samples) * 0.5
    y = pd.Series(signal, index=dates, name="target")

    # Also create a price column for target creation test
    df_with_close = X.copy()
    df_with_close["close"] = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_samples))

    trainer = ModelTrainer(config_path="configs/model_params.yaml")

    # Test target creation
    target = trainer.create_target(df_with_close, days=5)
    print(f"Target shape: {target.shape}, NaN count: {target.isna().sum()}")

    # Simple train/val/test split for demo (since dates may not match config exactly)
    split_1 = int(n_samples * 0.6)
    split_2 = int(n_samples * 0.8)

    X_train, y_train = X.iloc[:split_1], y.iloc[:split_1]
    X_val, y_val = X.iloc[split_1:split_2], y.iloc[split_1:split_2]
    X_test, y_test = X.iloc[split_2:], y.iloc[split_2:]

    # Train all models
    models = trainer.train_all_models(X_train, y_train, X_val, y_val)
    print(f"\nTrained models: {list(models.keys())}")

    # Compare models
    comparison = trainer.compare_models(models, X_test, y_test)
    print(f"\nModel comparison:\n{comparison}")

    # Test save/load
    trainer.save_model(models["xgboost"], "/tmp/finai_test_xgb.pkl")
    loaded = trainer.load_model("/tmp/finai_test_xgb.pkl")
    loaded_preds = loaded.predict(X_test)
    print(f"\nLoaded model predictions match: {np.allclose(models['xgboost'].predict(X_test), loaded_preds)}")

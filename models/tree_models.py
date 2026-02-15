"""Tree-based models: Random Forest and XGBoost."""

from typing import Optional

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


class RandomForestPredictor:
    """Random Forest for feature importance analysis."""

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        defaults = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
        # Merge defaults with user params (user params take priority)
        merged = {**defaults, **self.params}
        # Remove non-sklearn keys
        merged.pop("description", None)
        self.model = RandomForestRegressor(**merged)
        self.feature_names: list[str] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestPredictor":
        """Fit Random Forest model.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            self
        """
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self._is_fitted = True
        logger.info(
            f"RandomForest fitted: n_estimators={self.model.n_estimators}, "
            f"max_depth={self.model.max_depth}, features={len(self.feature_names)}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using Random Forest.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance from tree splits.

        Returns:
            DataFrame with feature and importance columns, sorted descending.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        })
        return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> dict:
        """Use Optuna for hyperparameter tuning with TimeSeriesSplit.

        Args:
            X: Feature matrix.
            y: Target values.
            n_trials: Number of Optuna trials.
            timeout: Maximum optimization time in seconds (None = no limit).

        Returns:
            Best hyperparameters dict.
        """
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]
                ),
                "random_state": 42,
                "n_jobs": -1,
            }
            ic_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = RandomForestRegressor(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                ic, _ = spearmanr(y_val, preds)
                if np.isnan(ic):
                    ic = 0.0
                ic_scores.append(ic)

            return float(np.mean(ic_scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        logger.info(f"RF best IC: {study.best_value:.4f}, params: {best_params}")

        # Update model with best params
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1
        self.model = RandomForestRegressor(**best_params)
        return best_params


class XGBoostPredictor:
    """XGBoost for best performance."""

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "random_state": 42,
        }
        merged = {**defaults, **self.params}
        merged.pop("description", None)
        self.model = XGBRegressor(**merged)
        self.feature_names: list[str] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostPredictor":
        """Fit XGBoost model.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            self
        """
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self._is_fitted = True
        logger.info(
            f"XGBoost fitted: n_estimators={self.model.n_estimators}, "
            f"max_depth={self.model.max_depth}, lr={self.model.learning_rate}, "
            f"features={len(self.feature_names)}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using XGBoost.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance from XGBoost.

        Returns:
            DataFrame with feature and importance columns, sorted descending.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        })
        return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> dict:
        """Use Optuna with TimeSeriesSplit cross-validation.

        Args:
            X: Feature matrix.
            y: Target values.
            n_trials: Number of Optuna trials.
            timeout: Maximum optimization time in seconds (None = no limit).

        Returns:
            Best hyperparameters dict.
        """
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
                "random_state": 42,
            }
            ic_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = XGBRegressor(**params)
                model.fit(X_tr, y_tr, verbose=False)
                preds = model.predict(X_val)
                ic, _ = spearmanr(y_val, preds)
                if np.isnan(ic):
                    ic = 0.0
                ic_scores.append(ic)

            return float(np.mean(ic_scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        logger.info(f"XGBoost best IC: {study.best_value:.4f}, params: {best_params}")

        # Update model with best params
        best_params["random_state"] = 42
        self.model = XGBRegressor(**best_params)
        return best_params


if __name__ == "__main__":
    np.random.seed(42)
    n_samples, n_features = 500, 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randn(n_samples), name="target")

    # Test RandomForest
    rf = RandomForestPredictor()
    rf.fit(X, y)
    rf_preds = rf.predict(X)
    print(f"RF predictions shape: {rf_preds.shape}")
    print(f"RF top 5 features:\n{rf.get_feature_importance().head()}\n")

    # Test XGBoost
    xgb = XGBoostPredictor()
    xgb.fit(X, y)
    xgb_preds = xgb.predict(X)
    print(f"XGBoost predictions shape: {xgb_preds.shape}")
    print(f"XGBoost top 5 features:\n{xgb.get_feature_importance().head()}\n")

    # Test Optuna tuning (with fewer trials for quick test)
    print("Running RF Optuna tuning (5 trials)...")
    rf_best = rf.optimize_hyperparameters(X, y, n_trials=5)
    print(f"RF best params: {rf_best}\n")

    print("Running XGBoost Optuna tuning (5 trials)...")
    xgb_best = xgb.optimize_hyperparameters(X, y, n_trials=5)
    print(f"XGBoost best params: {xgb_best}")

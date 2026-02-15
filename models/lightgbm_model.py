"""LightGBM model for stock return prediction."""

from typing import Optional

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb


class LightGBMPredictor:
    """LightGBM for fast gradient boosting prediction."""

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        # Merge defaults with user params (user params take priority)
        merged = {**defaults, **self.params}
        # Remove non-lgb keys
        merged.pop("description", None)
        self.model = lgb.LGBMRegressor(**merged)
        self.feature_names: list[str] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMPredictor":
        """Fit LightGBM model.

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
            "LightGBM fitted: n_estimators={}, max_depth={}, lr={}, "
            "num_leaves={}, features={}",
            self.model.n_estimators,
            self.model.max_depth,
            self.model.learning_rate,
            self.model.num_leaves,
            len(self.feature_names),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using LightGBM.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance from LightGBM.

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
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            }
            ic_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = lgb.LGBMRegressor(**params)
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
        logger.info("LightGBM best IC: {:.4f}, params: {}", study.best_value, best_params)

        # Update model with best params
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1
        best_params["verbosity"] = -1
        self.model = lgb.LGBMRegressor(**best_params)
        return best_params


if __name__ == "__main__":
    np.random.seed(42)
    n_samples, n_features = 500, 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randn(n_samples), name="target")

    # Test LightGBM
    lgbm = LightGBMPredictor()
    lgbm.fit(X, y)
    preds = lgbm.predict(X)
    print(f"LightGBM predictions shape: {preds.shape}")
    print(f"LightGBM top 5 features:\n{lgbm.get_feature_importance().head()}\n")

    # Test Optuna tuning (with fewer trials for quick test)
    print("Running LightGBM Optuna tuning (5 trials)...")
    best = lgbm.optimize_hyperparameters(X, y, n_trials=5)
    print(f"LightGBM best params: {best}")

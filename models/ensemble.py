"""Ensemble model combining Random Forest and XGBoost."""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class EnsemblePredictor:
    """Ensemble of RF + XGBoost with weighted average."""

    def __init__(self, models: dict, weights: Optional[dict] = None):
        """Initialize ensemble.

        Args:
            models: Dict of model name -> model instance.
                    e.g. {'random_forest': RFPredictor, 'xgboost': XGBPredictor}
            weights: Dict of model name -> weight. If None, equal weights.
                     e.g. {'random_forest': 0.4, 'xgboost': 0.6}
        """
        self.models = models
        if weights is None:
            n = len(models)
            self.weights = {name: 1.0 / n for name in models}
        else:
            self.weights = weights
        self._is_fitted = False

        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"Ensemble weights sum to {total:.4f}, normalizing to 1.0")
            self.weights = {k: v / total for k, v in self.weights.items()}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsemblePredictor":
        """Train all sub-models.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            self
        """
        for name, model in self.models.items():
            logger.info(f"Fitting ensemble sub-model: {name}")
            model.fit(X, y)
        self._is_fitted = True
        logger.info(f"Ensemble fitted with weights: {self.weights}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average prediction.

        Args:
            X: Feature matrix.

        Returns:
            Weighted average of sub-model predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before predicting")

        predictions = np.zeros(len(X))
        for name, model in self.models.items():
            weight = self.weights.get(name, 0.0)
            predictions += weight * model.predict(X)
        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """Weighted average feature importance from all sub-models.

        Returns:
            DataFrame with feature and importance columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before getting feature importance")

        importance_dfs = []
        for name, model in self.models.items():
            weight = self.weights.get(name, 0.0)
            fi = model.get_feature_importance()
            fi = fi.rename(columns={"importance": f"importance_{name}"})
            fi[f"weighted_{name}"] = fi[f"importance_{name}"] * weight
            importance_dfs.append(fi[["feature", f"weighted_{name}"]])

        # Merge all importance DataFrames on feature
        merged = importance_dfs[0]
        for df in importance_dfs[1:]:
            merged = merged.merge(df, on="feature", how="outer")

        # Sum weighted importances
        weighted_cols = [c for c in merged.columns if c.startswith("weighted_")]
        merged["importance"] = merged[weighted_cols].sum(axis=1)
        result = merged[["feature", "importance"]].copy()
        return result.sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from models.tree_models import RandomForestPredictor, XGBoostPredictor

    np.random.seed(42)
    n_samples, n_features = 500, 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randn(n_samples), name="target")

    # Create sub-models
    rf = RandomForestPredictor()
    xgb = XGBoostPredictor()

    # Create ensemble
    ensemble = EnsemblePredictor(
        models={"random_forest": rf, "xgboost": xgb},
        weights={"random_forest": 0.4, "xgboost": 0.6},
    )
    ensemble.fit(X, y)

    preds = ensemble.predict(X)
    print(f"Ensemble predictions shape: {preds.shape}")
    print(f"Predictions mean: {preds.mean():.6f}, std: {preds.std():.6f}")
    print(f"\nTop 5 features:\n{ensemble.get_feature_importance().head()}")

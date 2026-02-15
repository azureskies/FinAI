"""Baseline models for benchmark comparison."""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge


class BuyHoldBaseline:
    """Buy & Hold 0050 ETF as benchmark."""

    def calculate_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate buy-and-hold returns for benchmark.

        Args:
            price_data: DataFrame with 'close' column indexed by date.

        Returns:
            Cumulative return series.
        """
        if "close" not in price_data.columns:
            raise ValueError("price_data must contain a 'close' column")

        returns = price_data["close"].pct_change().fillna(0)
        cumulative = (1 + returns).cumprod() - 1
        logger.info(
            f"Buy & Hold return: {cumulative.iloc[-1]:.4f} "
            f"over {len(price_data)} trading days"
        )
        return cumulative


class RidgePredictor:
    """Ridge Regression as linear baseline."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.feature_names: list[str] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgePredictor":
        """Fit Ridge model.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            self
        """
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self._is_fitted = True
        logger.info(f"Ridge fitted with alpha={self.alpha}, features={len(self.feature_names)}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using Ridge model.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        return self.model.predict(X)

    def get_coefficients(self) -> pd.DataFrame:
        """Return feature coefficients for interpretability.

        Returns:
            DataFrame with feature names, coefficients, and absolute values.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting coefficients")
        coef_df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_,
            "abs_coefficient": np.abs(self.model.coef_),
        })
        return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance (absolute coefficients normalized).

        Returns:
            DataFrame with feature and importance columns.
        """
        coef_df = self.get_coefficients()
        total = coef_df["abs_coefficient"].sum()
        importance = coef_df["abs_coefficient"] / total if total > 0 else coef_df["abs_coefficient"]
        return pd.DataFrame({
            "feature": coef_df["feature"],
            "importance": importance,
        }).reset_index(drop=True)


if __name__ == "__main__":
    np.random.seed(42)
    n_samples, n_features = 500, 20

    # Test BuyHoldBaseline
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="B")
    prices = pd.DataFrame(
        {"close": 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_samples))},
        index=dates,
    )
    bh = BuyHoldBaseline()
    cum_ret = bh.calculate_returns(prices)
    print(f"Buy & Hold final return: {cum_ret.iloc[-1]:.4f}")

    # Test RidgePredictor
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randn(n_samples), name="target")

    ridge = RidgePredictor(alpha=1.0)
    ridge.fit(X, y)
    preds = ridge.predict(X)
    print(f"Ridge predictions shape: {preds.shape}")
    print(f"Top 5 coefficients:\n{ridge.get_coefficients().head()}")

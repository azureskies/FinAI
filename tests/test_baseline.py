"""Tests for baseline models: BuyHoldBaseline and RidgePredictor."""

import numpy as np
import pandas as pd
import pytest

from models.baseline import BuyHoldBaseline, RidgePredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Generate reproducible synthetic feature matrix and target."""
    np.random.seed(42)
    n_samples, n_features = 200, 10
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randn(n_samples), name="target")
    return X, y


@pytest.fixture
def price_data():
    """Generate reproducible synthetic price data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {"close": 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n))},
        index=dates,
    )
    return prices


# ---------------------------------------------------------------------------
# BuyHoldBaseline tests
# ---------------------------------------------------------------------------

class TestBuyHoldBaseline:
    def test_calculate_returns_shape(self, price_data):
        bh = BuyHoldBaseline()
        result = bh.calculate_returns(price_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_data)

    def test_calculate_returns_starts_at_zero(self, price_data):
        bh = BuyHoldBaseline()
        result = bh.calculate_returns(price_data)
        assert result.iloc[0] == 0.0

    def test_calculate_returns_monotonic_for_rising_prices(self):
        """If prices only go up, cumulative return should be monotonically increasing."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        prices = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)
        bh = BuyHoldBaseline()
        result = bh.calculate_returns(prices)
        diffs = result.diff().dropna()
        assert (diffs >= 0).all()

    def test_missing_close_column_raises(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        bh = BuyHoldBaseline()
        with pytest.raises(ValueError, match="close"):
            bh.calculate_returns(df)


# ---------------------------------------------------------------------------
# RidgePredictor tests
# ---------------------------------------------------------------------------

class TestRidgePredictor:
    def test_fit_returns_self(self, sample_data):
        X, y = sample_data
        ridge = RidgePredictor(alpha=1.0)
        result = ridge.fit(X, y)
        assert result is ridge

    def test_predict_shape(self, sample_data):
        X, y = sample_data
        ridge = RidgePredictor()
        ridge.fit(X, y)
        preds = ridge.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self, sample_data):
        X, _ = sample_data
        ridge = RidgePredictor()
        with pytest.raises(RuntimeError, match="fitted"):
            ridge.predict(X)

    def test_get_coefficients(self, sample_data):
        X, y = sample_data
        ridge = RidgePredictor()
        ridge.fit(X, y)
        coef_df = ridge.get_coefficients()
        assert set(coef_df.columns) == {"feature", "coefficient", "abs_coefficient"}
        assert len(coef_df) == X.shape[1]
        # Should be sorted descending by abs_coefficient
        assert coef_df["abs_coefficient"].is_monotonic_decreasing

    def test_get_coefficients_before_fit_raises(self):
        ridge = RidgePredictor()
        with pytest.raises(RuntimeError, match="fitted"):
            ridge.get_coefficients()

    def test_get_feature_importance(self, sample_data):
        X, y = sample_data
        ridge = RidgePredictor()
        ridge.fit(X, y)
        fi = ridge.get_feature_importance()
        assert set(fi.columns) == {"feature", "importance"}
        assert len(fi) == X.shape[1]
        # Importances should sum to ~1.0
        assert abs(fi["importance"].sum() - 1.0) < 1e-6

    def test_single_feature(self):
        """Edge case: single feature."""
        np.random.seed(42)
        X = pd.DataFrame({"feat_0": np.random.randn(50)})
        y = pd.Series(np.random.randn(50))
        ridge = RidgePredictor()
        ridge.fit(X, y)
        preds = ridge.predict(X)
        assert preds.shape == (50,)
        fi = ridge.get_feature_importance()
        assert len(fi) == 1

    def test_custom_alpha(self, sample_data):
        X, y = sample_data
        ridge = RidgePredictor(alpha=100.0)
        ridge.fit(X, y)
        assert ridge.alpha == 100.0
        preds = ridge.predict(X)
        assert preds.shape == (len(X),)

    def test_all_same_target(self, sample_data):
        """Edge case: target is constant."""
        X, _ = sample_data
        y_const = pd.Series(np.ones(len(X)))
        ridge = RidgePredictor()
        ridge.fit(X, y_const)
        preds = ridge.predict(X)
        # Predictions should be close to 1.0
        np.testing.assert_allclose(preds, 1.0, atol=0.1)

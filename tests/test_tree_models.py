"""Tests for tree-based models: RandomForestPredictor and XGBoostPredictor."""

import numpy as np
import pandas as pd
import pytest

from models.tree_models import RandomForestPredictor, XGBoostPredictor


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
    # Add a signal so models can learn something
    y = pd.Series(
        X["feat_0"] * 0.5 + X["feat_1"] * 0.3 + np.random.randn(n_samples) * 0.2,
        name="target",
    )
    return X, y


# ---------------------------------------------------------------------------
# RandomForestPredictor tests
# ---------------------------------------------------------------------------

class TestRandomForestPredictor:
    def test_fit_returns_self(self, sample_data):
        X, y = sample_data
        rf = RandomForestPredictor()
        result = rf.fit(X, y)
        assert result is rf

    def test_predict_shape(self, sample_data):
        X, y = sample_data
        rf = RandomForestPredictor()
        rf.fit(X, y)
        preds = rf.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self, sample_data):
        X, _ = sample_data
        rf = RandomForestPredictor()
        with pytest.raises(RuntimeError, match="fitted"):
            rf.predict(X)

    def test_get_feature_importance(self, sample_data):
        X, y = sample_data
        rf = RandomForestPredictor()
        rf.fit(X, y)
        fi = rf.get_feature_importance()
        assert set(fi.columns) == {"feature", "importance"}
        assert len(fi) == X.shape[1]
        # Should be sorted descending
        assert fi["importance"].is_monotonic_decreasing
        # Importances should sum to ~1.0
        assert abs(fi["importance"].sum() - 1.0) < 1e-6

    def test_feature_importance_before_fit_raises(self):
        rf = RandomForestPredictor()
        with pytest.raises(RuntimeError, match="fitted"):
            rf.get_feature_importance()

    def test_custom_params(self, sample_data):
        X, y = sample_data
        rf = RandomForestPredictor(params={"n_estimators": 50, "max_depth": 3})
        rf.fit(X, y)
        assert rf.model.n_estimators == 50
        assert rf.model.max_depth == 3

    def test_optimize_hyperparameters(self, sample_data):
        X, y = sample_data
        rf = RandomForestPredictor()
        best_params = rf.optimize_hyperparameters(X, y, n_trials=3)
        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params

    def test_single_feature(self):
        """Edge case: single feature."""
        np.random.seed(42)
        X = pd.DataFrame({"feat_0": np.random.randn(100)})
        y = pd.Series(np.random.randn(100))
        rf = RandomForestPredictor(params={"n_estimators": 10, "max_features": 1.0})
        rf.fit(X, y)
        preds = rf.predict(X)
        assert preds.shape == (100,)


# ---------------------------------------------------------------------------
# XGBoostPredictor tests
# ---------------------------------------------------------------------------

class TestXGBoostPredictor:
    def test_fit_returns_self(self, sample_data):
        X, y = sample_data
        xgb = XGBoostPredictor()
        result = xgb.fit(X, y)
        assert result is xgb

    def test_predict_shape(self, sample_data):
        X, y = sample_data
        xgb = XGBoostPredictor()
        xgb.fit(X, y)
        preds = xgb.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self, sample_data):
        X, _ = sample_data
        xgb = XGBoostPredictor()
        with pytest.raises(RuntimeError, match="fitted"):
            xgb.predict(X)

    def test_get_feature_importance(self, sample_data):
        X, y = sample_data
        xgb = XGBoostPredictor()
        xgb.fit(X, y)
        fi = xgb.get_feature_importance()
        assert set(fi.columns) == {"feature", "importance"}
        assert len(fi) == X.shape[1]
        # Should be sorted descending
        assert fi["importance"].is_monotonic_decreasing

    def test_feature_importance_before_fit_raises(self):
        xgb = XGBoostPredictor()
        with pytest.raises(RuntimeError, match="fitted"):
            xgb.get_feature_importance()

    def test_custom_params(self, sample_data):
        X, y = sample_data
        xgb = XGBoostPredictor(params={"n_estimators": 50, "max_depth": 3})
        xgb.fit(X, y)
        assert xgb.model.n_estimators == 50
        assert xgb.model.max_depth == 3

    def test_optimize_hyperparameters(self, sample_data):
        X, y = sample_data
        xgb = XGBoostPredictor()
        best_params = xgb.optimize_hyperparameters(X, y, n_trials=3)
        assert isinstance(best_params, dict)
        assert "learning_rate" in best_params
        assert "max_depth" in best_params

    def test_single_feature(self):
        """Edge case: single feature."""
        np.random.seed(42)
        X = pd.DataFrame({"feat_0": np.random.randn(100)})
        y = pd.Series(np.random.randn(100))
        xgb = XGBoostPredictor(params={"n_estimators": 10})
        xgb.fit(X, y)
        preds = xgb.predict(X)
        assert preds.shape == (100,)

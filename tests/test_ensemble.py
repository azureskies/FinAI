"""Tests for EnsemblePredictor."""

import numpy as np
import pandas as pd
import pytest

from models.ensemble import EnsemblePredictor
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
    y = pd.Series(
        X["feat_0"] * 0.5 + np.random.randn(n_samples) * 0.3,
        name="target",
    )
    return X, y


@pytest.fixture
def fitted_models(sample_data):
    """Return pre-fitted RF and XGBoost models."""
    X, y = sample_data
    rf = RandomForestPredictor(params={"n_estimators": 20, "max_depth": 5, "random_state": 42})
    rf.fit(X, y)
    xgb = XGBoostPredictor(params={"n_estimators": 20, "max_depth": 3, "random_state": 42})
    xgb.fit(X, y)
    return {"random_forest": rf, "xgboost": xgb}


# ---------------------------------------------------------------------------
# EnsemblePredictor tests
# ---------------------------------------------------------------------------

class TestEnsemblePredictor:
    def test_init_equal_weights(self, fitted_models):
        ensemble = EnsemblePredictor(models=fitted_models)
        assert len(ensemble.weights) == 2
        for w in ensemble.weights.values():
            assert abs(w - 0.5) < 1e-6

    def test_init_custom_weights(self, fitted_models):
        weights = {"random_forest": 0.4, "xgboost": 0.6}
        ensemble = EnsemblePredictor(models=fitted_models, weights=weights)
        assert abs(ensemble.weights["random_forest"] - 0.4) < 1e-6
        assert abs(ensemble.weights["xgboost"] - 0.6) < 1e-6

    def test_weight_normalization(self, fitted_models):
        """Weights that don't sum to 1.0 should be normalized."""
        weights = {"random_forest": 2.0, "xgboost": 3.0}
        ensemble = EnsemblePredictor(models=fitted_models, weights=weights)
        total = sum(ensemble.weights.values())
        assert abs(total - 1.0) < 1e-6
        assert abs(ensemble.weights["random_forest"] - 0.4) < 1e-6
        assert abs(ensemble.weights["xgboost"] - 0.6) < 1e-6

    def test_fit_returns_self(self, sample_data, fitted_models):
        X, y = sample_data
        ensemble = EnsemblePredictor(models=fitted_models)
        result = ensemble.fit(X, y)
        assert result is ensemble
        assert ensemble._is_fitted is True

    def test_predict_shape(self, sample_data, fitted_models):
        X, y = sample_data
        ensemble = EnsemblePredictor(models=fitted_models)
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self, sample_data, fitted_models):
        X, _ = sample_data
        ensemble = EnsemblePredictor(models=fitted_models)
        with pytest.raises(RuntimeError, match="fitted"):
            ensemble.predict(X)

    def test_predict_is_weighted_average(self, sample_data, fitted_models):
        """Verify predictions are the weighted average of sub-model predictions."""
        X, y = sample_data
        weights = {"random_forest": 0.4, "xgboost": 0.6}
        ensemble = EnsemblePredictor(models=fitted_models, weights=weights)
        ensemble.fit(X, y)

        ensemble_preds = ensemble.predict(X)
        rf_preds = fitted_models["random_forest"].predict(X)
        xgb_preds = fitted_models["xgboost"].predict(X)
        expected = 0.4 * rf_preds + 0.6 * xgb_preds

        np.testing.assert_allclose(ensemble_preds, expected, rtol=1e-6)

    def test_get_feature_importance(self, sample_data, fitted_models):
        X, y = sample_data
        ensemble = EnsemblePredictor(models=fitted_models)
        ensemble.fit(X, y)
        fi = ensemble.get_feature_importance()
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == X.shape[1]
        # Should be sorted descending
        assert fi["importance"].is_monotonic_decreasing

    def test_feature_importance_before_fit_raises(self, fitted_models):
        ensemble = EnsemblePredictor(models=fitted_models)
        with pytest.raises(RuntimeError, match="fitted"):
            ensemble.get_feature_importance()

    def test_single_model_ensemble(self, sample_data):
        """Edge case: ensemble with a single model."""
        X, y = sample_data
        rf = RandomForestPredictor(params={"n_estimators": 10, "random_state": 42})
        models = {"random_forest": rf}
        ensemble = EnsemblePredictor(models=models)
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        rf_preds = rf.predict(X)
        np.testing.assert_allclose(preds, rf_preds, rtol=1e-6)

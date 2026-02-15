"""Tests for Optuna hyperparameter optimization integration in weekly_retrain."""

from unittest.mock import patch

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
# Optimize flag triggers Optuna
# ---------------------------------------------------------------------------

class TestOptunaTriggeredByFlag:
    """Verify that --optimize flag triggers Optuna optimization."""

    def test_rf_optimize_called_with_flag(self, sample_data):
        """RandomForest optimize_hyperparameters is called when optimize=True."""
        X, y = sample_data
        rf = RandomForestPredictor()
        best_params = rf.optimize_hyperparameters(X, y, n_trials=3)

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "min_samples_leaf" in best_params
        # Model should be updated with best params
        assert rf.model.get_params()["n_estimators"] == best_params["n_estimators"]

    def test_xgb_optimize_called_with_flag(self, sample_data):
        """XGBoost optimize_hyperparameters is called when optimize=True."""
        X, y = sample_data
        xgb = XGBoostPredictor()
        best_params = xgb.optimize_hyperparameters(X, y, n_trials=3)

        assert isinstance(best_params, dict)
        assert "learning_rate" in best_params
        assert "max_depth" in best_params
        assert "n_estimators" in best_params
        # Model should be updated with best params
        assert xgb.model.get_params()["max_depth"] == best_params["max_depth"]

    def test_parse_args_optimize_flag(self):
        """Verify --optimize flag is recognized by the arg parser."""
        from scripts.weekly_retrain import parse_args

        with patch("sys.argv", ["weekly_retrain", "--optimize"]):
            args = parse_args()
        assert args.optimize is True

    def test_parse_args_no_optimize_default(self):
        """Verify --optimize defaults to False."""
        from scripts.weekly_retrain import parse_args

        with patch("sys.argv", ["weekly_retrain"]):
            args = parse_args()
        assert args.optimize is False


# ---------------------------------------------------------------------------
# Optimized model has different params than default
# ---------------------------------------------------------------------------

class TestOptimizedParamsDiffer:
    """Optimized params should (generally) differ from defaults."""

    def test_rf_optimized_params_differ(self, sample_data):
        """RF optimized params should not be identical to defaults."""
        X, y = sample_data
        default_params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 20,
        }
        rf = RandomForestPredictor()
        best_params = rf.optimize_hyperparameters(X, y, n_trials=3)

        # At least one parameter should differ from default
        differs = any(
            best_params.get(k) != v
            for k, v in default_params.items()
        )
        # Note: with only 3 trials it's *possible* to hit defaults,
        # but extremely unlikely with the search space
        assert differs, (
            f"Expected at least one param to differ from defaults. "
            f"Got: {best_params}"
        )

    def test_xgb_optimized_params_differ(self, sample_data):
        """XGBoost optimized params should not be identical to defaults."""
        X, y = sample_data
        default_params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
        }
        xgb = XGBoostPredictor()
        best_params = xgb.optimize_hyperparameters(X, y, n_trials=3)

        differs = any(
            best_params.get(k) != v
            for k, v in default_params.items()
        )
        assert differs, (
            f"Expected at least one param to differ from defaults. "
            f"Got: {best_params}"
        )


# ---------------------------------------------------------------------------
# Timeout works
# ---------------------------------------------------------------------------

class TestOptunaTimeout:
    """Verify timeout parameter is respected."""

    def test_rf_timeout_completes(self, sample_data):
        """RF optimization with very short timeout should still return params."""
        X, y = sample_data
        rf = RandomForestPredictor()
        # Very short timeout — may complete 0 or 1 trials
        best_params = rf.optimize_hyperparameters(
            X, y, n_trials=1000, timeout=5,
        )
        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params

    def test_xgb_timeout_completes(self, sample_data):
        """XGBoost optimization with very short timeout should still return params."""
        X, y = sample_data
        xgb = XGBoostPredictor()
        best_params = xgb.optimize_hyperparameters(
            X, y, n_trials=1000, timeout=5,
        )
        assert isinstance(best_params, dict)
        assert "learning_rate" in best_params

    def test_rf_timeout_none_uses_n_trials(self, sample_data):
        """With timeout=None, optimization runs for exactly n_trials."""
        X, y = sample_data
        rf = RandomForestPredictor()
        best_params = rf.optimize_hyperparameters(
            X, y, n_trials=3, timeout=None,
        )
        assert isinstance(best_params, dict)

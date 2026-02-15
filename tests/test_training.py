"""Tests for ModelTrainer pipeline."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from models.training import ModelTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trainer():
    """Create a ModelTrainer with default config."""
    return ModelTrainer(config_path="configs/model_params.yaml")


@pytest.fixture
def synthetic_data():
    """Generate reproducible synthetic dataset covering the config time range."""
    np.random.seed(42)
    # Create data from 2018 through 2024 to cover config time_split ranges
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="B")
    n = len(dates)
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
        index=dates,
    )
    y = pd.Series(
        X["feat_0"] * 0.3 + X["feat_1"] * 0.2 + np.random.randn(n) * 0.5,
        index=dates,
        name="target",
    )
    return X, y


@pytest.fixture
def price_df():
    """Generate synthetic price DataFrame with 'close' column."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {"close": 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, 100))},
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# create_target tests
# ---------------------------------------------------------------------------

class TestCreateTarget:
    def test_basic(self, trainer, price_df):
        target = trainer.create_target(price_df, days=5)
        assert isinstance(target, pd.Series)
        assert len(target) == len(price_df)
        # Last 5 values should be NaN (shifted forward)
        assert target.iloc[-5:].isna().all()
        # Earlier values should be valid
        assert target.iloc[:-5].notna().all()

    def test_default_days_from_config(self, trainer, price_df):
        target = trainer.create_target(price_df)
        # Config says target_days=5
        assert target.iloc[-5:].isna().all()
        assert target.iloc[:-5].notna().all()

    def test_custom_days(self, trainer, price_df):
        target = trainer.create_target(price_df, days=10)
        assert target.iloc[-10:].isna().all()
        assert target.iloc[:-10].notna().all()

    def test_missing_close_raises(self, trainer):
        df = pd.DataFrame({"open": [1, 2, 3]})
        with pytest.raises(ValueError, match="close"):
            trainer.create_target(df)


# ---------------------------------------------------------------------------
# time_split tests
# ---------------------------------------------------------------------------

class TestTimeSplit:
    def test_returns_three_splits(self, trainer, synthetic_data):
        X, _ = synthetic_data
        splits = trainer.time_split(X)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_splits_non_empty(self, trainer, synthetic_data):
        X, _ = synthetic_data
        splits = trainer.time_split(X)
        for name, df in splits.items():
            assert len(df) > 0, f"{name} split is empty"

    def test_splits_non_overlapping(self, trainer, synthetic_data):
        X, _ = synthetic_data
        splits = trainer.time_split(X)
        assert splits["train"].index.max() < splits["val"].index.min()
        assert splits["val"].index.max() < splits["test"].index.min()


# ---------------------------------------------------------------------------
# train_all_models tests
# ---------------------------------------------------------------------------

class TestTrainAllModels:
    def test_returns_expected_models(self, trainer, synthetic_data):
        X, y = synthetic_data
        n = len(X)
        split1, split2 = int(n * 0.6), int(n * 0.8)
        X_train, y_train = X.iloc[:split1], y.iloc[:split1]
        X_val, y_val = X.iloc[split1:split2], y.iloc[split1:split2]

        models = trainer.train_all_models(X_train, y_train, X_val, y_val)
        assert isinstance(models, dict)
        expected_keys = {"ridge", "random_forest", "xgboost", "ensemble"}
        assert set(models.keys()) == expected_keys

    def test_all_models_can_predict(self, trainer, synthetic_data):
        X, y = synthetic_data
        n = len(X)
        split1, split2 = int(n * 0.6), int(n * 0.8)
        X_train, y_train = X.iloc[:split1], y.iloc[:split1]
        X_val, y_val = X.iloc[split1:split2], y.iloc[split1:split2]
        X_test = X.iloc[split2:]

        models = trainer.train_all_models(X_train, y_train, X_val, y_val)
        for name, model in models.items():
            preds = model.predict(X_test)
            assert preds.shape == (len(X_test),), f"{name} prediction shape mismatch"


# ---------------------------------------------------------------------------
# compare_models tests
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_dataframe(self, trainer, synthetic_data):
        X, y = synthetic_data
        n = len(X)
        split1, split2 = int(n * 0.6), int(n * 0.8)
        X_train, y_train = X.iloc[:split1], y.iloc[:split1]
        X_val, y_val = X.iloc[split1:split2], y.iloc[split1:split2]
        X_test, y_test = X.iloc[split2:], y.iloc[split2:]

        models = trainer.train_all_models(X_train, y_train, X_val, y_val)
        result = trainer.compare_models(models, X_test, y_test)

        assert isinstance(result, pd.DataFrame)
        expected_cols = {"IC", "Rank_IC", "Sharpe_Top20", "MSE"}
        assert expected_cols == set(result.columns)
        assert len(result) == len(models)

    def test_mse_non_negative(self, trainer, synthetic_data):
        X, y = synthetic_data
        n = len(X)
        split1, split2 = int(n * 0.6), int(n * 0.8)
        X_train, y_train = X.iloc[:split1], y.iloc[:split1]
        X_val, y_val = X.iloc[split1:split2], y.iloc[split1:split2]
        X_test, y_test = X.iloc[split2:], y.iloc[split2:]

        models = trainer.train_all_models(X_train, y_train, X_val, y_val)
        result = trainer.compare_models(models, X_test, y_test)
        assert (result["MSE"] >= 0).all()


# ---------------------------------------------------------------------------
# save_model / load_model tests
# ---------------------------------------------------------------------------

class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, trainer, synthetic_data):
        X, y = synthetic_data
        n = len(X)
        split1 = int(n * 0.6)
        X_train, y_train = X.iloc[:split1], y.iloc[:split1]
        X_test = X.iloc[split1 : split1 + 50]

        from models.tree_models import XGBoostPredictor

        model = XGBoostPredictor(params={"n_estimators": 10, "random_state": 42})
        model.fit(X_train, y_train)
        original_preds = model.predict(X_test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            trainer.save_model(model, path)
            loaded = trainer.load_model(path)
            loaded_preds = loaded.predict(X_test)
            np.testing.assert_array_equal(original_preds, loaded_preds)
        finally:
            os.unlink(path)

    def test_save_creates_parent_dirs(self, trainer):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "model.pkl")
            trainer.save_model({"dummy": True}, path)
            assert os.path.exists(path)

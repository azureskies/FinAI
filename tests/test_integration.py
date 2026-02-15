"""End-to-end integration tests for the FinAI pipeline.

Tests the full flow: synthetic data -> cleaning -> features -> training
-> prediction -> backtest, verifying that each stage produces valid
output that the next stage can consume.

All tests run without external services (no yfinance, no Supabase).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult
from data.processors.cleaning import DataCleaner
from data.processors.features import FeatureEngine
from models.baseline import RidgePredictor
from models.ensemble import EnsemblePredictor
from models.tree_models import RandomForestPredictor, XGBoostPredictor


# ------------------------------------------------------------------ #
#  Full pipeline integration test
# ------------------------------------------------------------------ #

class TestFullPipeline:
    """End-to-end pipeline: clean -> features -> train -> backtest."""

    def test_pipeline_produces_valid_backtest_result(
        self,
        synthetic_price_df,
        synthetic_multi_stock_prices,
        synthetic_benchmark,
    ):
        """Run the entire pipeline on synthetic data and verify the result."""

        # Step 1: DataCleaner
        cleaner = DataCleaner()
        cleaned, report = cleaner.validate_price(synthetic_price_df)
        assert report["total_rows"] == len(synthetic_price_df)
        filled = cleaner.fill_missing(cleaned)
        assert filled.isna().sum().sum() == 0

        # Step 2: FeatureEngine (single stock for training features)
        engine = FeatureEngine()
        features = engine.compute_all(filled)
        features = features.dropna()
        assert len(features) > 50, "Need enough rows after NaN drop"

        # Step 3: Create target and split
        np.random.seed(42)
        y = pd.Series(
            features.iloc[:, 0] * 0.1 + np.random.randn(len(features)) * 0.5,
            index=features.index,
            name="target",
        )

        split = int(len(features) * 0.7)
        X_train, X_val = features.iloc[:split], features.iloc[split:]
        y_train, _y_val = y.iloc[:split], y.iloc[split:]

        # Step 4: Train models
        ridge = RidgePredictor(alpha=1.0)
        ridge.fit(X_train, y_train)

        rf = RandomForestPredictor(params={
            "n_estimators": 30, "max_depth": 5, "random_state": 42, "n_jobs": 1,
        })
        rf.fit(X_train, y_train)

        xgb = XGBoostPredictor(params={
            "n_estimators": 30, "max_depth": 4, "learning_rate": 0.1, "random_state": 42,
        })
        xgb.fit(X_train, y_train)

        ensemble = EnsemblePredictor(
            models={"random_forest": rf, "xgboost": xgb},
            weights={"random_forest": 0.4, "xgboost": 0.6},
        )
        ensemble._is_fitted = True

        # Verify all models produce finite predictions
        for name, model in [("ridge", ridge), ("rf", rf), ("xgb", xgb), ("ensemble", ensemble)]:
            preds = model.predict(X_val)
            assert len(preds) == len(X_val)
            assert np.all(np.isfinite(preds)), f"{name} produced non-finite predictions"

        # Step 5: Build predictions table for backtest
        prices = synthetic_multi_stock_prices
        stock_ids = prices["stock_id"].unique()
        pred_dates = sorted(prices["date"].unique())

        # Generate a prediction for each (monthly date, stock)
        monthly_dates = [pred_dates[0]]
        for d in pred_dates[1:]:
            if d.month != monthly_dates[-1].month:
                monthly_dates.append(d)

        pred_records = []
        for d in monthly_dates:
            for sid in stock_ids:
                pred_records.append({
                    "date": d,
                    "stock_id": sid,
                    "predicted_return": float(np.random.normal(0.001, 0.02)),
                })
        pred_df = pd.DataFrame(pred_records)

        # Step 6: Backtest
        bt_engine = BacktestEngine()
        result = bt_engine.run(
            predictions=pred_df,
            price_data=prices,
            benchmark_data=synthetic_benchmark,
            initial_capital=10_000_000,
        )

        # Verify BacktestResult structure
        assert isinstance(result, BacktestResult)
        assert not result.equity_curve.empty
        assert len(result.equity_curve) == len(pred_dates)
        assert result.metrics["initial_capital"] == 10_000_000
        assert result.metrics["final_value"] > 0
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert result.metrics["trading_days"] > 0

    def test_pipeline_models_are_consistent(self, synthetic_features):
        """Verify that re-training on the same data produces identical predictions."""
        np.random.seed(42)
        X = synthetic_features
        y = pd.Series(np.random.randn(len(X)), index=X.index)

        ridge1 = RidgePredictor(alpha=1.0)
        ridge1.fit(X, y)
        preds1 = ridge1.predict(X)

        ridge2 = RidgePredictor(alpha=1.0)
        ridge2.fit(X, y)
        preds2 = ridge2.predict(X)

        np.testing.assert_array_almost_equal(preds1, preds2)


# ------------------------------------------------------------------ #
#  Walk-forward integration test
# ------------------------------------------------------------------ #

class TestWalkForward:
    """Walk-forward backtest with rolling retrain on synthetic data."""

    def test_walk_forward_small_windows(
        self, synthetic_multi_stock_prices, synthetic_benchmark
    ):
        """Run walk_forward with small windows to verify the mechanism."""
        np.random.seed(42)
        prices = synthetic_multi_stock_prices
        dates = sorted(prices["date"].unique())
        stock_ids = prices["stock_id"].unique()
        n_features = 5

        # Build feature_data with target
        feat_records = []
        for d in dates:
            for sid in stock_ids:
                row = {"date": d, "stock_id": sid}
                for fi in range(n_features):
                    row[f"feat_{fi}"] = np.random.randn()
                row["target"] = np.random.normal(0.001, 0.02)
                feat_records.append(row)
        feature_data = pd.DataFrame(feat_records)

        model = RidgePredictor(alpha=1.0)
        engine = BacktestEngine()

        # Use small windows to complete multiple folds
        result = engine.walk_forward(
            feature_data=feature_data,
            price_data=prices,
            benchmark_data=synthetic_benchmark,
            model=model,
            train_window=40,
            test_window=20,
            target_col="target",
            initial_capital=10_000_000,
        )

        assert isinstance(result, BacktestResult)
        assert not result.equity_curve.empty
        assert result.metrics["trading_days"] > 0
        assert result.metrics["final_value"] > 0


# ------------------------------------------------------------------ #
#  Data cleaning + feature integration
# ------------------------------------------------------------------ #

class TestCleaningFeatureIntegration:
    """Verify that DataCleaner output feeds cleanly into FeatureEngine."""

    def test_dirty_data_produces_valid_features(self):
        """Inject data quality issues and confirm the pipeline still works."""
        np.random.seed(42)
        n = 200
        dates = pd.bdate_range("2023-01-01", periods=n)
        price = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame(
            {
                "open": price + np.random.randn(n) * 0.2,
                "high": price + abs(np.random.randn(n)) * 1.0,
                "low": price - abs(np.random.randn(n)) * 1.0,
                "close": price,
                "volume": np.random.randint(1000, 50000, size=n).astype(float),
            },
            index=dates,
        )

        # Inject issues
        df.iloc[5, df.columns.get_loc("close")] = np.nan
        df.iloc[10, df.columns.get_loc("volume")] = 0
        # Swap high/low
        df.iloc[15, df.columns.get_loc("high")] = df.iloc[15]["low"] - 1

        cleaner = DataCleaner()
        cleaned, report = cleaner.validate_price(df)
        filled = cleaner.fill_missing(cleaned)

        engine = FeatureEngine()
        features = engine.compute_all(filled)
        features_clean = features.dropna()

        assert len(features_clean) > 50
        assert features_clean.shape[1] > 10
        # No infinities
        assert np.all(np.isfinite(features_clean.values))


# ------------------------------------------------------------------ #
#  Multi-stock pipeline
# ------------------------------------------------------------------ #

class TestMultiStockPipeline:
    """Test processing multiple stocks through the feature pipeline."""

    def test_multi_stock_features(self):
        """Process each stock through FeatureEngine independently."""
        np.random.seed(42)
        n_days = 300
        n_stocks = 3
        dates = pd.bdate_range("2023-01-01", periods=n_days)

        engine = FeatureEngine()
        all_features = {}

        for i in range(n_stocks):
            sid = f"{2300 + i}"
            base = np.random.uniform(100, 500)
            rets = np.random.normal(0.0003, 0.015, n_days)
            closes = base * np.cumprod(1 + rets)
            stock_df = pd.DataFrame(
                {
                    "open": closes * (1 + np.random.normal(0, 0.003, n_days)),
                    "high": closes * 1.005,
                    "low": closes * 0.995,
                    "close": closes,
                    "volume": np.random.randint(1e6, 1e7, size=n_days).astype(float),
                },
                index=dates,
            )
            feats = engine.compute_technical(stock_df)
            feats = feats.dropna()
            all_features[sid] = feats
            assert len(feats) > 0, f"No features for stock {sid}"

        # Verify all stocks produced the same feature columns
        col_sets = [set(df.columns) for df in all_features.values()]
        assert len(set(frozenset(c) for c in col_sets)) == 1, "Feature columns differ across stocks"


# ------------------------------------------------------------------ #
#  Trained model fixture integration
# ------------------------------------------------------------------ #

class TestTrainedModelsFixture:
    """Verify that conftest trained_models fixture works end-to-end."""

    def test_all_models_predict(self, trained_models, synthetic_features):
        """Every model in the fixture should produce valid predictions."""
        X = synthetic_features
        for name, model in trained_models.items():
            preds = model.predict(X)
            assert len(preds) == len(X), f"{name}: wrong prediction count"
            assert np.all(np.isfinite(preds)), f"{name}: non-finite predictions"

    def test_ensemble_combines_sub_models(self, trained_models, synthetic_features):
        """Ensemble prediction should be between its sub-model predictions."""
        X = synthetic_features
        rf_preds = trained_models["random_forest"].predict(X)
        xgb_preds = trained_models["xgboost"].predict(X)
        ens_preds = trained_models["ensemble"].predict(X)

        # Weighted average => ensemble is between min and max of sub-models
        lower = np.minimum(rf_preds, xgb_preds)
        upper = np.maximum(rf_preds, xgb_preds)
        assert np.all(ens_preds >= lower - 1e-10)
        assert np.all(ens_preds <= upper + 1e-10)

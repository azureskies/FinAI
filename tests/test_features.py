"""Tests for data.processors.features.FeatureEngine."""

import numpy as np
import pandas as pd
import pytest

from data.processors.features import FeatureEngine


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def engine():
    """Return a FeatureEngine instance."""
    return FeatureEngine()


@pytest.fixture
def price_df():
    """Return a synthetic OHLCV DataFrame with enough rows for indicators."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": price + np.random.randn(n) * 0.2,
            "high": price + np.abs(np.random.randn(n)) * 1.0,
            "low": price - np.abs(np.random.randn(n)) * 1.0,
            "close": price,
            "volume": np.random.randint(1000, 50000, size=n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def fin_df():
    """Return a synthetic financial statement DataFrame."""
    dates = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"])
    return pd.DataFrame(
        {
            "eps": [2.5, 3.0, 2.8, 3.2],
            "bps": [45.0, 46.0, 47.0, 48.0],
            "revenue": [100e9, 110e9, 105e9, 120e9],
            "net_income": [20e9, 22e9, 21e9, 25e9],
            "total_assets": [500e9, 510e9, 520e9, 530e9],
            "total_liabilities": [200e9, 205e9, 210e9, 215e9],
            "dividend_yield": [0.03, 0.03, 0.03, 0.03],
        },
        index=dates,
    )


@pytest.fixture
def market_df(price_df):
    """Return a synthetic market index DataFrame aligned to price_df."""
    np.random.seed(99)
    n = len(price_df)
    mkt_price = 15000 + np.cumsum(np.random.randn(n) * 10)
    return pd.DataFrame({"close": mkt_price}, index=price_df.index)


# ------------------------------------------------------------------ #
#  compute_technical
# ------------------------------------------------------------------ #

class TestComputeTechnical:
    """Tests for FeatureEngine.compute_technical."""

    def test_adds_technical_columns(self, engine, price_df):
        """Should add many new technical indicator columns."""
        result = engine.compute_technical(price_df)
        new_cols = set(result.columns) - set(price_df.columns)
        assert len(new_cols) > 20

    def test_preserves_original_columns(self, engine, price_df):
        """Original OHLCV columns should remain."""
        result = engine.compute_technical(price_df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_same_row_count(self, engine, price_df):
        """Output should have the same number of rows as input."""
        result = engine.compute_technical(price_df)
        assert len(result) == len(price_df)

    def test_rsi_range(self, engine, price_df):
        """RSI values should be in [0, 100] where not NaN."""
        result = engine.compute_technical(price_df)
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_does_not_mutate_input(self, engine, price_df):
        """Input DataFrame should not be modified."""
        original = price_df.copy()
        engine.compute_technical(price_df)
        pd.testing.assert_frame_equal(price_df, original)

    def test_expected_indicators_present(self, engine, price_df):
        """Key indicators should be present in output."""
        result = engine.compute_technical(price_df)
        expected = [
            "rsi_14", "macd_line", "ema_12", "sma_20",
            "bb_upper", "bb_lower", "atr_14", "obv",
            "stoch_k", "stoch_d",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"


# ------------------------------------------------------------------ #
#  compute_fundamental
# ------------------------------------------------------------------ #

class TestComputeFundamental:
    """Tests for FeatureEngine.compute_fundamental."""

    def test_returns_fundamental_features(self, engine, price_df, fin_df):
        """Should produce valuation and quality features."""
        result = engine.compute_fundamental(price_df, fin_df)
        assert "pe_ratio" in result.columns
        assert "pb_ratio" in result.columns

    def test_empty_fin_df_returns_empty(self, engine, price_df):
        """Empty financial data should return empty DataFrame."""
        result = engine.compute_fundamental(price_df, pd.DataFrame())
        assert result.empty or result.dropna(how="all").empty

    def test_none_fin_df_returns_empty(self, engine, price_df):
        """None financial data should return empty DataFrame."""
        result = engine.compute_fundamental(price_df, None)
        assert len(result.columns) == 0

    def test_aligned_to_price_index(self, engine, price_df, fin_df):
        """Output index should match price DataFrame index."""
        result = engine.compute_fundamental(price_df, fin_df)
        pd.testing.assert_index_equal(result.index, price_df.index)

    def test_forward_fill_no_lookahead(self, engine, price_df, fin_df):
        """Fundamental data before the first fin_df date should be NaN (no lookahead)."""
        result = engine.compute_fundamental(price_df, fin_df)
        # Dates before 2023-03-31 should have NaN for pe_ratio
        early = result.loc[result.index < fin_df.index[0], "pe_ratio"]
        assert early.isna().all()


# ------------------------------------------------------------------ #
#  compute_market
# ------------------------------------------------------------------ #

class TestComputeMarket:
    """Tests for FeatureEngine.compute_market."""

    def test_returns_relative_strength(self, engine, price_df, market_df):
        """Should produce relative_strength_20 column."""
        result = engine.compute_market(price_df, market_df)
        assert "relative_strength_20" in result.columns

    def test_empty_market_df(self, engine, price_df):
        """Empty market data should return empty DataFrame."""
        result = engine.compute_market(price_df, pd.DataFrame())
        assert result.empty or result.dropna(how="all").empty

    def test_none_market_df(self, engine, price_df):
        """None market data should return empty DataFrame."""
        result = engine.compute_market(price_df, None)
        assert len(result.columns) == 0

    def test_aligned_to_stock_index(self, engine, price_df, market_df):
        """Output index should match stock DataFrame index."""
        result = engine.compute_market(price_df, market_df)
        pd.testing.assert_index_equal(result.index, price_df.index)

    def test_row_count_matches(self, engine, price_df, market_df):
        """Output row count should match input."""
        result = engine.compute_market(price_df, market_df)
        assert len(result) == len(price_df)


# ------------------------------------------------------------------ #
#  compute_all
# ------------------------------------------------------------------ #

class TestComputeAll:
    """Tests for FeatureEngine.compute_all."""

    def test_combines_all_features(self, engine, price_df, fin_df, market_df):
        """Should include technical, fundamental, and market features."""
        result = engine.compute_all(price_df, fin_df, market_df)
        assert "rsi_14" in result.columns
        assert "pe_ratio" in result.columns
        assert "relative_strength_20" in result.columns

    def test_market_cap_label_large(self, engine, price_df):
        """market_cap >= 100B should produce label 2."""
        result = engine.compute_all(price_df, market_cap=200_000_000_000)
        assert (result["market_cap_label"] == 2).all()

    def test_market_cap_label_mid(self, engine, price_df):
        """30B <= market_cap < 100B should produce label 1."""
        result = engine.compute_all(price_df, market_cap=50_000_000_000)
        assert (result["market_cap_label"] == 1).all()

    def test_market_cap_label_small(self, engine, price_df):
        """market_cap < 30B should produce label 0."""
        result = engine.compute_all(price_df, market_cap=10_000_000_000)
        assert (result["market_cap_label"] == 0).all()

    def test_no_raw_ohlcv_in_output(self, engine, price_df):
        """Raw OHLCV columns should not be in the final output."""
        result = engine.compute_all(price_df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col not in result.columns


# ------------------------------------------------------------------ #
#  remove_correlated
# ------------------------------------------------------------------ #

class TestRemoveCorrelated:
    """Tests for FeatureEngine.remove_correlated."""

    def test_removes_perfectly_correlated(self, engine):
        """Perfectly correlated columns should be dropped."""
        np.random.seed(0)
        x = np.random.randn(100)
        df = pd.DataFrame({"a": x, "b": x * 2, "c": np.random.randn(100)})
        result = engine.remove_correlated(df, threshold=0.95)
        # 'b' is perfectly correlated with 'a' -> should be dropped
        assert "a" in result.columns
        assert "b" not in result.columns
        assert "c" in result.columns

    def test_uncorrelated_kept(self, engine):
        """Uncorrelated columns should all be kept."""
        np.random.seed(1)
        df = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
                "c": np.random.randn(100),
            }
        )
        result = engine.remove_correlated(df, threshold=0.95)
        assert list(result.columns) == ["a", "b", "c"]

    def test_threshold_sensitivity(self, engine):
        """Lower threshold should drop more columns."""
        np.random.seed(2)
        x = np.random.randn(200)
        df = pd.DataFrame({
            "a": x,
            "b": x + np.random.randn(200) * 0.1,
            "c": np.random.randn(200),
        })
        strict = engine.remove_correlated(df, threshold=0.8)
        loose = engine.remove_correlated(df, threshold=0.99)
        assert len(strict.columns) <= len(loose.columns)

    def test_empty_dataframe(self, engine):
        """Empty DataFrame should return empty."""
        df = pd.DataFrame()
        result = engine.remove_correlated(df)
        assert result.empty

    def test_single_column(self, engine):
        """Single column DataFrame should remain unchanged."""
        df = pd.DataFrame({"a": np.random.randn(50)})
        result = engine.remove_correlated(df)
        assert list(result.columns) == ["a"]

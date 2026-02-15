"""Tests for data.collectors.price.PriceCollector."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from data.collectors.price import PriceCollector


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

_DEFAULT_CONFIG = {
    "price_sources": [
        {
            "name": "yfinance",
            "suffix": ".TW",
            "otc_suffix": ".TWO",
            "rate_limit": 0.0,  # no delay in tests
        }
    ],
    "data_quality": {
        "max_missing_rate": 0.05,
        "price_change_limit": 0.11,
    },
}


@pytest.fixture
def collector():
    """Return a PriceCollector with test config (no file I/O)."""
    return PriceCollector(config=_DEFAULT_CONFIG)


def _make_yf_dataframe(n: int = 50) -> pd.DataFrame:
    """Create a fake yfinance-style DataFrame."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    price = 500 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame(
        {
            "Open": price + np.random.randn(n) * 0.5,
            "High": price + np.abs(np.random.randn(n)) * 2,
            "Low": price - np.abs(np.random.randn(n)) * 2,
            "Close": price,
            "Adj Close": price * 0.98,
            "Volume": np.random.randint(10000, 500000, size=n),
        },
        index=dates,
    )


# ------------------------------------------------------------------ #
#  _build_ticker
# ------------------------------------------------------------------ #

class TestBuildTicker:
    """Tests for PriceCollector._build_ticker."""

    def test_twse_suffix(self, collector):
        assert collector._build_ticker("2330", "twse") == "2330.TW"

    def test_otc_suffix(self, collector):
        assert collector._build_ticker("6547", "otc") == "6547.TWO"

    def test_default_is_twse(self, collector):
        """Non-otc market should use the default TWSE suffix."""
        assert collector._build_ticker("2330", "other") == "2330.TW"


# ------------------------------------------------------------------ #
#  fetch
# ------------------------------------------------------------------ #

class TestFetch:
    """Tests for PriceCollector.fetch with mocked yfinance."""

    @patch("data.collectors.price.yf.download")
    def test_fetch_success(self, mock_download, collector):
        """Successful download should return a normalized DataFrame."""
        mock_download.return_value = _make_yf_dataframe()
        result = collector.fetch("2330", "2024-01-01", "2024-04-01")

        assert result is not None
        assert "close" in result.columns
        assert "stock_id" in result.columns
        assert result["stock_id"].iloc[0] == "2330"
        assert result.index.name == "date"

    @patch("data.collectors.price.yf.download")
    def test_fetch_empty_returns_none(self, mock_download, collector):
        """Empty download result should return None."""
        mock_download.return_value = pd.DataFrame()
        result = collector.fetch("9999", "2024-01-01", "2024-04-01")
        assert result is None

    @patch("data.collectors.price.yf.download")
    def test_fetch_none_returns_none(self, mock_download, collector):
        """None download result should return None."""
        mock_download.return_value = None
        result = collector.fetch("9999", "2024-01-01", "2024-04-01")
        assert result is None

    @patch("data.collectors.price.yf.download")
    def test_fetch_exception_returns_none(self, mock_download, collector):
        """Exception during download should return None."""
        mock_download.side_effect = Exception("Network error")
        result = collector.fetch("2330", "2024-01-01", "2024-04-01")
        assert result is None

    @patch("data.collectors.price.yf.download")
    def test_fetch_otc_market(self, mock_download, collector):
        """OTC market should pass the correct ticker to yfinance."""
        mock_download.return_value = _make_yf_dataframe()
        collector.fetch("6547", "2024-01-01", "2024-04-01", market="otc")
        args, kwargs = mock_download.call_args
        assert "6547.TWO" in args or kwargs.get("tickers", args[0]) == "6547.TWO"

    @patch("data.collectors.price.yf.download")
    def test_normalized_columns(self, mock_download, collector):
        """Output should have lowercase column names."""
        mock_download.return_value = _make_yf_dataframe()
        result = collector.fetch("2330", "2024-01-01", "2024-04-01")
        expected_cols = {"open", "high", "low", "close", "adj_close", "volume", "stock_id"}
        assert set(result.columns) == expected_cols


# ------------------------------------------------------------------ #
#  fetch_batch
# ------------------------------------------------------------------ #

class TestFetchBatch:
    """Tests for PriceCollector.fetch_batch."""

    @patch("data.collectors.price.yf.download")
    def test_batch_all_success(self, mock_download, collector):
        """All successful fetches should be in the result dict."""
        mock_download.return_value = _make_yf_dataframe()
        result = collector.fetch_batch(["2330", "2317"], "2024-01-01", "2024-04-01")
        assert len(result) == 2
        assert "2330" in result
        assert "2317" in result

    @patch("data.collectors.price.yf.download")
    def test_batch_partial_failure(self, mock_download, collector):
        """Failed fetches should be skipped; successful ones returned."""
        def side_effect(ticker, **kwargs):
            if "9999" in ticker:
                return pd.DataFrame()
            return _make_yf_dataframe()

        mock_download.side_effect = side_effect
        result = collector.fetch_batch(
            ["2330", "9999", "2317"], "2024-01-01", "2024-04-01"
        )
        assert "2330" in result
        assert "9999" not in result
        assert "2317" in result

    @patch("data.collectors.price.yf.download")
    def test_batch_empty_list(self, mock_download, collector):
        """Empty stock list should return empty dict."""
        result = collector.fetch_batch([], "2024-01-01", "2024-04-01")
        assert result == {}
        mock_download.assert_not_called()


# ------------------------------------------------------------------ #
#  compute_adjusted_ohlcv
# ------------------------------------------------------------------ #

class TestComputeAdjustedOhlcv:
    """Tests for PriceCollector.compute_adjusted_ohlcv."""

    def test_adds_adjusted_columns(self, collector):
        """Should add adj_open, adj_high, adj_low, adj_close, adj_volume."""
        df = pd.DataFrame(
            {
                "open": [100.0, 102.0],
                "high": [105.0, 107.0],
                "low": [98.0, 100.0],
                "close": [103.0, 105.0],
                "adj_close": [101.0, 103.0],
                "volume": [10000, 12000],
            }
        )
        result = collector.compute_adjusted_ohlcv(df)
        for col in ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]:
            assert col in result.columns

    def test_adj_volume_equals_volume(self, collector):
        """Adjusted volume should equal raw volume."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [98.0],
                "close": [103.0],
                "adj_close": [100.0],
                "volume": [50000],
            }
        )
        result = collector.compute_adjusted_ohlcv(df)
        assert result["adj_volume"].iloc[0] == 50000

    def test_ratio_applied_correctly(self, collector):
        """Adjusted prices should use the adj_close/close ratio."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [110.0],
                "low": [90.0],
                "close": [100.0],
                "adj_close": [50.0],  # 50% ratio
                "volume": [1000],
            }
        )
        result = collector.compute_adjusted_ohlcv(df)
        assert result["adj_open"].iloc[0] == 50.0
        assert result["adj_high"].iloc[0] == 55.0
        assert result["adj_low"].iloc[0] == 45.0

    def test_empty_dataframe(self, collector):
        """Empty DataFrame should return empty."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "adj_close", "volume"]
        )
        result = collector.compute_adjusted_ohlcv(df)
        assert result.empty

    def test_does_not_mutate_input(self, collector):
        """Input should not be modified."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [98.0],
                "close": [103.0],
                "adj_close": [101.0],
                "volume": [10000],
            }
        )
        original = df.copy()
        collector.compute_adjusted_ohlcv(df)
        pd.testing.assert_frame_equal(df, original)

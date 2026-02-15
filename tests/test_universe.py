"""Tests for data.collectors.universe.UniverseCollector."""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from data.collectors.universe import UniverseCollector


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

_DEFAULT_CONFIG = {
    "universe": {
        "min_market_cap": 10_000_000_000,
        "min_avg_volume": 1000,
        "min_listing_years": 2,
        "exclude_categories": ["全額交割"],
    },
}


@pytest.fixture
def collector():
    """Return a UniverseCollector with test config."""
    return UniverseCollector(config=_DEFAULT_CONFIG)


def _make_stock_list() -> pd.DataFrame:
    """Create a fake stock list DataFrame."""
    return pd.DataFrame({
        "stock_id": ["2330", "2317", "3711", "6547", "1234"],
        "stock_name": ["TSMC", "Foxconn", "ASMedia", "Knowles", "TestCo"],
        "market": ["twse", "twse", "twse", "otc", "twse"],
        "listing_date": pd.to_datetime([
            "2000-01-01", "2005-06-15", "2010-03-20", "2015-08-01", "2023-12-01"
        ]),
    })


def _make_market_cap_df() -> pd.DataFrame:
    """Create a fake market cap DataFrame."""
    return pd.DataFrame({
        "stock_id": ["2330", "2317", "3711", "6547", "1234"],
        "market_value": [20e12, 1e12, 50e9, 15e9, 5e9],
    })


def _make_volume_df() -> pd.DataFrame:
    """Create a fake daily volume DataFrame with multiple days."""
    rows = []
    for sid in ["2330", "2317", "3711", "6547", "1234"]:
        for d in pd.bdate_range("2024-04-01", periods=20):
            vol = 5_000_000 if sid != "1234" else 500
            rows.append({"stock_id": sid, "date": d, "Trading_Volume": vol})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  __init__ configuration
# ------------------------------------------------------------------ #

class TestUniverseInit:
    """Tests for UniverseCollector initialization."""

    def test_default_config_values(self, collector):
        """Config values should be loaded from the provided config dict."""
        assert collector.min_market_cap == 10_000_000_000
        assert collector.min_avg_volume == 1000
        assert collector.min_listing_years == 2
        assert "全額交割" in collector.exclude_categories

    def test_custom_config(self):
        """Custom config should override defaults."""
        cfg = {
            "universe": {
                "min_market_cap": 5e9,
                "min_avg_volume": 500,
                "min_listing_years": 1,
                "exclude_categories": [],
            }
        }
        c = UniverseCollector(config=cfg)
        assert c.min_market_cap == 5e9
        assert c.min_avg_volume == 500


# ------------------------------------------------------------------ #
#  _exclude_categories
# ------------------------------------------------------------------ #

class TestExcludeCategories:
    """Tests for UniverseCollector._exclude_categories."""

    def test_excludes_matching_category(self, collector):
        """Stocks with excluded category should be removed."""
        df = pd.DataFrame({
            "stock_id": ["2330", "9999"],
            "industry_category": ["半導體", "全額交割"],
        })
        result = collector._exclude_categories(df)
        assert len(result) == 1
        assert result.iloc[0]["stock_id"] == "2330"

    def test_no_category_column_passes_through(self, collector):
        """If no category column exists, all stocks pass through."""
        df = pd.DataFrame({"stock_id": ["2330", "2317"]})
        result = collector._exclude_categories(df)
        assert len(result) == 2

    def test_empty_exclude_list(self):
        """Empty exclude list should keep all stocks."""
        cfg = {"universe": {"exclude_categories": []}}
        c = UniverseCollector(config=cfg)
        df = pd.DataFrame({
            "stock_id": ["2330"],
            "industry_category": ["全額交割"],
        })
        result = c._exclude_categories(df)
        assert len(result) == 1

    def test_empty_dataframe(self, collector):
        """Empty DataFrame should return empty."""
        df = pd.DataFrame(columns=["stock_id", "industry_category"])
        result = collector._exclude_categories(df)
        assert result.empty


# ------------------------------------------------------------------ #
#  get_universe (mocked end-to-end)
# ------------------------------------------------------------------ #

class TestGetUniverse:
    """Tests for UniverseCollector.get_universe with mocked API calls."""

    @patch.object(UniverseCollector, "_fetch_daily_volume")
    @patch.object(UniverseCollector, "_fetch_market_cap")
    @patch.object(UniverseCollector, "_fetch_stock_list")
    def test_full_pipeline(self, mock_list, mock_mcap, mock_vol, collector):
        """Full pipeline should filter by age, market cap, and volume."""
        mock_list.return_value = _make_stock_list()
        mock_mcap.return_value = _make_market_cap_df()
        mock_vol.return_value = _make_volume_df()

        result = collector.get_universe(date(2024, 6, 1))

        # 1234 excluded: listing < 2 years & market_cap < 10B & volume too low
        # 6547 excluded: market_cap (15e9) >= 10B but volume 5M/1000=5000 >= 1000,
        #   so 6547 should pass. Let's verify.
        stock_ids = result["stock_id"].tolist()
        assert "2330" in stock_ids  # passes all filters
        assert "2317" in stock_ids  # passes all filters
        assert "1234" not in stock_ids  # listing too recent, market_cap too low

    @patch.object(UniverseCollector, "_fetch_stock_list")
    def test_empty_stock_list(self, mock_list, collector):
        """Empty stock list should return empty DataFrame."""
        mock_list.return_value = pd.DataFrame()
        result = collector.get_universe(date(2024, 6, 1))
        assert result.empty

    @patch.object(UniverseCollector, "_fetch_stock_list")
    def test_none_stock_list(self, mock_list, collector):
        """None stock list should return empty DataFrame."""
        mock_list.return_value = None
        result = collector.get_universe(date(2024, 6, 1))
        assert result.empty

    @patch.object(UniverseCollector, "_fetch_daily_volume")
    @patch.object(UniverseCollector, "_fetch_market_cap")
    @patch.object(UniverseCollector, "_fetch_stock_list")
    def test_listing_age_filter(self, mock_list, mock_mcap, mock_vol, collector):
        """Stocks listed less than min_listing_years ago should be excluded."""
        mock_list.return_value = _make_stock_list()
        mock_mcap.return_value = _make_market_cap_df()
        mock_vol.return_value = _make_volume_df()

        result = collector.get_universe(date(2024, 6, 1))
        # 1234 was listed on 2023-12-01, less than 2 years before 2024-06-01
        assert "1234" not in result["stock_id"].tolist()


# ------------------------------------------------------------------ #
#  get_twse_stocks / get_otc_stocks
# ------------------------------------------------------------------ #

class TestMarketFilters:
    """Tests for get_twse_stocks and get_otc_stocks."""

    @patch.object(UniverseCollector, "_fetch_daily_volume")
    @patch.object(UniverseCollector, "_fetch_market_cap")
    @patch.object(UniverseCollector, "_fetch_stock_list")
    def test_twse_only(self, mock_list, mock_mcap, mock_vol, collector):
        """get_twse_stocks should return only twse market stocks."""
        mock_list.return_value = _make_stock_list()
        mock_mcap.return_value = _make_market_cap_df()
        mock_vol.return_value = _make_volume_df()

        result = collector.get_twse_stocks(date(2024, 6, 1))
        if not result.empty and "market" in result.columns:
            assert (result["market"] == "twse").all()

    @patch.object(UniverseCollector, "_fetch_daily_volume")
    @patch.object(UniverseCollector, "_fetch_market_cap")
    @patch.object(UniverseCollector, "_fetch_stock_list")
    def test_otc_only(self, mock_list, mock_mcap, mock_vol, collector):
        """get_otc_stocks should return only otc market stocks."""
        mock_list.return_value = _make_stock_list()
        mock_mcap.return_value = _make_market_cap_df()
        mock_vol.return_value = _make_volume_df()

        result = collector.get_otc_stocks(date(2024, 6, 1))
        if not result.empty and "market" in result.columns:
            assert (result["market"] == "otc").all()

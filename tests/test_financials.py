"""Tests for data.collectors.financials.FinancialsCollector."""

from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from data.collectors.financials import (
    FinancialsCollector,
)


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

_DEFAULT_CONFIG = {
    "fundamental_sources": [
        {"name": "mops", "base_url": "https://mops.twse.com.tw"}
    ],
}


@pytest.fixture
def collector():
    """Return a FinancialsCollector with test config."""
    return FinancialsCollector(config=_DEFAULT_CONFIG)


# ------------------------------------------------------------------ #
#  latest_available_quarter — Lookahead bias protection
# ------------------------------------------------------------------ #

class TestLatestAvailableQuarter:
    """Tests for FinancialsCollector.latest_available_quarter."""

    def test_after_q1_deadline(self):
        """After May 15, Q1 of the current year should be available."""
        result = FinancialsCollector.latest_available_quarter(date(2024, 6, 1))
        assert result == (2024, 1)

    def test_after_q2_deadline(self):
        """After Aug 14, Q2 of the current year should be available."""
        result = FinancialsCollector.latest_available_quarter(date(2024, 9, 1))
        assert result == (2024, 2)

    def test_after_q3_deadline(self):
        """After Nov 14, Q3 of the current year should be available."""
        result = FinancialsCollector.latest_available_quarter(date(2024, 12, 1))
        assert result == (2024, 3)

    def test_after_q4_deadline(self):
        """After Mar 31, Q4 of the previous year should be available."""
        result = FinancialsCollector.latest_available_quarter(date(2024, 4, 1))
        assert result == (2023, 4)

    def test_before_q4_deadline(self):
        """Before Mar 31, only Q3 of the previous year is available."""
        result = FinancialsCollector.latest_available_quarter(date(2024, 3, 15))
        assert result == (2023, 3)

    def test_exact_q1_deadline_date(self):
        """On exactly May 15, Q1 should be available."""
        result = FinancialsCollector.latest_available_quarter(date(2024, 5, 15))
        assert result == (2024, 1)

    def test_exact_q4_deadline_date(self):
        """On exactly Mar 31, Q4 of previous year should be available."""
        result = FinancialsCollector.latest_available_quarter(date(2024, 3, 31))
        assert result == (2023, 4)

    def test_january_early(self):
        """Early January — only Q3 of previous year is available."""
        result = FinancialsCollector.latest_available_quarter(date(2024, 1, 15))
        assert result == (2023, 3)

    def test_most_recent_quarter_wins(self):
        """After Nov 14, Q3 should beat Q2, Q1, Q4(prev)."""
        result = FinancialsCollector.latest_available_quarter(date(2024, 11, 20))
        assert result == (2024, 3)

    def test_between_q1_and_q2_deadline(self):
        """Between May 15 and Aug 14, latest is Q1."""
        result = FinancialsCollector.latest_available_quarter(date(2024, 7, 1))
        assert result == (2024, 1)


# ------------------------------------------------------------------ #
#  latest_available_revenue_month
# ------------------------------------------------------------------ #

class TestLatestAvailableRevenueMonth:
    """Tests for FinancialsCollector.latest_available_revenue_month."""

    def test_after_deadline_day(self):
        """On/after the 10th, previous month's revenue is available."""
        result = FinancialsCollector.latest_available_revenue_month(date(2024, 6, 15))
        assert result == (2024, 5)

    def test_before_deadline_day(self):
        """Before the 10th, revenue from two months ago is available."""
        result = FinancialsCollector.latest_available_revenue_month(date(2024, 6, 5))
        assert result == (2024, 4)

    def test_on_deadline_day(self):
        """Exactly on the 10th, previous month's revenue is available."""
        result = FinancialsCollector.latest_available_revenue_month(date(2024, 6, 10))
        assert result == (2024, 5)

    def test_january_after_deadline(self):
        """January after 10th — December of previous year is available."""
        result = FinancialsCollector.latest_available_revenue_month(date(2024, 1, 15))
        assert result == (2023, 12)

    def test_january_before_deadline(self):
        """January before 10th — November of previous year is available."""
        result = FinancialsCollector.latest_available_revenue_month(date(2024, 1, 5))
        assert result == (2023, 11)

    def test_february_before_deadline(self):
        """February before 10th — December of previous year is available."""
        result = FinancialsCollector.latest_available_revenue_month(date(2024, 2, 5))
        assert result == (2023, 12)


# ------------------------------------------------------------------ #
#  _finmind_request (mocked)
# ------------------------------------------------------------------ #

class TestFinmindRequest:
    """Tests for FinancialsCollector._finmind_request with mocked HTTP."""

    @patch("data.collectors.financials.requests.get")
    def test_success(self, mock_get, collector):
        """Successful API call should return a DataFrame."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": 200,
            "data": [{"date": "2024-01-01", "value": 100}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = collector._finmind_request("TestDataset", {"data_id": "2330"})
        assert result is not None
        assert len(result) == 1

    @patch("data.collectors.financials.requests.get")
    def test_api_error_status(self, mock_get, collector):
        """Non-200 API status should return None."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": 400, "msg": "Bad request"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = collector._finmind_request("TestDataset", {"data_id": "2330"})
        assert result is None

    @patch("data.collectors.financials.requests.get")
    def test_network_error(self, mock_get, collector):
        """Network error should return None."""
        import requests
        mock_get.side_effect = requests.RequestException("timeout")
        result = collector._finmind_request("TestDataset", {"data_id": "2330"})
        assert result is None

    @patch("data.collectors.financials.requests.get")
    def test_empty_data(self, mock_get, collector):
        """Empty data array should return None."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": 200, "data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = collector._finmind_request("TestDataset", {"data_id": "2330"})
        assert result is None


# ------------------------------------------------------------------ #
#  fetch_quarterly (mocked)
# ------------------------------------------------------------------ #

class TestFetchQuarterly:
    """Tests for FinancialsCollector.fetch_quarterly with mocked API."""

    @patch.object(FinancialsCollector, "_finmind_request")
    def test_returns_reshaped_data(self, mock_req, collector):
        """Successful fetch should return reshaped quarterly data."""
        mock_req.return_value = pd.DataFrame({
            "date": ["2024-03-31", "2024-03-31", "2024-06-30", "2024-06-30"],
            "type": ["EPS", "ROE", "EPS", "ROE"],
            "value": [2.5, 0.15, 3.0, 0.18],
        })
        result = collector.fetch_quarterly("2330", "2023-01-01", "2024-06-30")
        assert result is not None
        assert "year" in result.columns
        assert "quarter" in result.columns

    @patch.object(FinancialsCollector, "_finmind_request")
    def test_returns_none_on_failure(self, mock_req, collector):
        """API failure should return None."""
        mock_req.return_value = None
        result = collector.fetch_quarterly("2330", "2023-01-01", "2024-06-30")
        assert result is None

"""
Taiwan stock financial data collector.

Fetches fundamental data from FinMind API (public endpoint) with
look-ahead bias protection for backtesting.
"""

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from loguru import logger

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data_sources.yaml"

# Taiwan quarterly report disclosure deadlines (month, day).
# After these dates, the corresponding quarter's data becomes "available".
QUARTERLY_DEADLINES: dict[int, tuple[int, int]] = {
    1: (5, 15),   # Q1 report due by May 15
    2: (8, 14),   # Q2 report due by Aug 14
    3: (11, 14),  # Q3 report due by Nov 14
    4: (3, 31),   # Q4 report due by Mar 31 of the *next* year
}

# Monthly revenue is disclosed before the 10th of the following month.
MONTHLY_REVENUE_DEADLINE_DAY = 10

FINMIND_BASE_URL = "https://api.finmindtrade.com/api/v4"


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FinancialsCollector:
    """Collect fundamental / financial data for Taiwan stocks."""

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or _load_config()
        fm_cfg = next(
            (s for s in cfg.get("fundamental_sources", []) if s["name"] == "mops"),
            {},
        )
        self.mops_base_url: str = fm_cfg.get("base_url", "https://mops.twse.com.tw")

        # FinMind token (optional, increases rate limit)
        self.finmind_token: str = os.environ.get("FINMIND_TOKEN", "")
        self.rate_limit: float = 1.0  # seconds between FinMind requests

    # ------------------------------------------------------------------
    # Look-ahead bias helpers
    # ------------------------------------------------------------------

    @staticmethod
    def latest_available_quarter(as_of: date) -> tuple[int, int]:
        """Return (year, quarter) of the most recent report available on *as_of*.

        Taiwan disclosure rules:
          Q1 -> available after May 15
          Q2 -> available after Aug 14
          Q3 -> available after Nov 14
          Q4 -> available after Mar 31 of the *following* year

        Example:
          as_of = 2024-06-01  -> latest available = (2024, Q1)
          as_of = 2024-03-30  -> latest available = (2023, Q3)
        """
        y = as_of.year

        # Check from the most recent quarter backwards
        # Q4 of previous year: available after Mar 31 of current year
        if as_of >= date(y, 3, 31):
            candidate_q4 = (y - 1, 4)
        else:
            candidate_q4 = None

        # Q1 of current year: available after May 15
        if as_of >= date(y, 5, 15):
            candidate_q1 = (y, 1)
        else:
            candidate_q1 = None

        # Q2 of current year: available after Aug 14
        if as_of >= date(y, 8, 14):
            candidate_q2 = (y, 2)
        else:
            candidate_q2 = None

        # Q3 of current year: available after Nov 14
        if as_of >= date(y, 11, 14):
            candidate_q3 = (y, 3)
        else:
            candidate_q3 = None

        # Pick the most recent available quarter
        for candidate in [candidate_q3, candidate_q2, candidate_q1, candidate_q4]:
            if candidate is not None:
                return candidate

        # If none matched (before Mar 31), fall back to Q3 of prev year
        return (y - 1, 3)

    @staticmethod
    def latest_available_revenue_month(as_of: date) -> tuple[int, int]:
        """Return (year, month) of the most recent revenue data available on *as_of*.

        Monthly revenue is disclosed before the 10th of the following month.
        """
        if as_of.day >= MONTHLY_REVENUE_DEADLINE_DAY:
            # Revenue for the previous month is available
            rev_month = as_of.month - 1
            rev_year = as_of.year
            if rev_month < 1:
                rev_month = 12
                rev_year -= 1
        else:
            # Revenue for two months ago
            rev_month = as_of.month - 2
            rev_year = as_of.year
            if rev_month < 1:
                rev_month += 12
                rev_year -= 1
        return (rev_year, rev_month)

    # ------------------------------------------------------------------
    # FinMind API helpers
    # ------------------------------------------------------------------

    def _finmind_request(self, dataset: str, params: dict) -> Optional[pd.DataFrame]:
        """Send a request to the FinMind API and return a DataFrame."""
        payload = {"dataset": dataset, **params}
        if self.finmind_token:
            payload["token"] = self.finmind_token

        try:
            resp = requests.get(
                f"{FINMIND_BASE_URL}/data", params=payload, timeout=30
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as e:
            logger.error("FinMind request failed for {}: {}", dataset, e)
            return None

        if body.get("status") != 200:
            logger.warning("FinMind returned status {}: {}", body.get("status"), body.get("msg"))
            return None

        data = body.get("data", [])
        if not data:
            logger.debug("FinMind returned empty data for {}", dataset)
            return None

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_quarterly(
        self,
        stock_id: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch quarterly financial statements (EPS, ROE, ROA, etc.) from FinMind.

        Args:
            stock_id: Taiwan stock code, e.g. "2330".
            start: Start date, e.g. "2022-01-01".
            end: End date, e.g. "2024-01-01".

        Returns:
            DataFrame with quarterly financial metrics, or None on failure.
        """
        logger.info("Fetching quarterly financials for {} ({} ~ {})", stock_id, start, end)

        df = self._finmind_request(
            "TaiwanStockFinancialStatements",
            {"data_id": stock_id, "start_date": start, "end_date": end},
        )
        if df is None:
            return None

        time.sleep(self.rate_limit)
        return self._reshape_quarterly(df, stock_id)

    def fetch_monthly_revenue(
        self,
        stock_id: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch monthly revenue data from FinMind.

        Returns:
            DataFrame with columns [stock_id, date, revenue, revenue_yoy, revenue_mom].
        """
        logger.info("Fetching monthly revenue for {} ({} ~ {})", stock_id, start, end)

        df = self._finmind_request(
            "TaiwanStockMonthRevenue",
            {"data_id": stock_id, "start_date": start, "end_date": end},
        )
        if df is None:
            return None

        time.sleep(self.rate_limit)
        return self._normalize_revenue(df, stock_id)

    def fetch_valuation(
        self,
        stock_id: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch daily valuation metrics (PE, PB, dividend yield) from FinMind.

        Returns:
            DataFrame with columns [stock_id, date, pe_ratio, pb_ratio, dividend_yield].
        """
        logger.info("Fetching valuation for {} ({} ~ {})", stock_id, start, end)

        df = self._finmind_request(
            "TaiwanStockPER",
            {"data_id": stock_id, "start_date": start, "end_date": end},
        )
        if df is None:
            return None

        time.sleep(self.rate_limit)
        return self._normalize_valuation(df, stock_id)

    def fetch_all(
        self,
        stock_id: str,
        start: str,
        end: str,
    ) -> dict[str, Optional[pd.DataFrame]]:
        """Fetch all financial data types for a single stock.

        Returns:
            Dict with keys: "quarterly", "revenue", "valuation".
        """
        return {
            "quarterly": self.fetch_quarterly(stock_id, start, end),
            "revenue": self.fetch_monthly_revenue(stock_id, start, end),
            "valuation": self.fetch_valuation(stock_id, start, end),
        }

    def get_available_financials(
        self,
        stock_id: str,
        as_of: date,
        start: str,
        end: str,
    ) -> dict[str, Optional[pd.DataFrame]]:
        """Fetch financial data and filter to only what is available on *as_of*.

        This prevents look-ahead bias in backtesting.
        """
        all_data = self.fetch_all(stock_id, start, end)

        # Filter quarterly data by disclosure deadline
        avail_year, avail_q = self.latest_available_quarter(as_of)
        if all_data["quarterly"] is not None and not all_data["quarterly"].empty:
            df_q = all_data["quarterly"]
            if "year" in df_q.columns and "quarter" in df_q.columns:
                mask = (df_q["year"] < avail_year) | (
                    (df_q["year"] == avail_year) & (df_q["quarter"] <= avail_q)
                )
                all_data["quarterly"] = df_q[mask]

        # Filter revenue by disclosure deadline
        avail_rev_year, avail_rev_month = self.latest_available_revenue_month(as_of)
        if all_data["revenue"] is not None and not all_data["revenue"].empty:
            df_r = all_data["revenue"]
            if "date" in df_r.columns:
                cutoff = f"{avail_rev_year}-{avail_rev_month:02d}-01"
                all_data["revenue"] = df_r[df_r["date"] <= cutoff]

        return all_data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reshape_quarterly(
        self, df: pd.DataFrame, stock_id: str
    ) -> Optional[pd.DataFrame]:
        """Reshape FinMind financial statements into wide format."""
        if df.empty:
            return None

        try:
            # FinMind returns long-format: date, type, value
            if "type" in df.columns and "value" in df.columns:
                pivot = df.pivot_table(
                    index="date", columns="type", values="value", aggfunc="first"
                )
                pivot = pivot.reset_index()
                pivot["stock_id"] = stock_id

                # Extract year and quarter from date
                pivot["date"] = pd.to_datetime(pivot["date"])
                pivot["year"] = pivot["date"].dt.year
                pivot["quarter"] = pivot["date"].dt.quarter
                return pivot
        except Exception as e:
            logger.error("Failed to reshape quarterly data for {}: {}", stock_id, e)

        return None

    def _normalize_revenue(
        self, df: pd.DataFrame, stock_id: str
    ) -> Optional[pd.DataFrame]:
        """Normalize monthly revenue data."""
        if df.empty:
            return None

        try:
            out = df.copy()
            out["stock_id"] = stock_id

            col_map = {
                "revenue": "revenue",
                "revenue_month": "revenue",
            }
            for old, new in col_map.items():
                if old in out.columns and new != old:
                    out = out.rename(columns={old: new})

            if "date" in out.columns:
                out["date"] = pd.to_datetime(out["date"])

            # Compute YoY and MoM if revenue exists
            if "revenue" in out.columns:
                out = out.sort_values("date")
                out["revenue_yoy"] = out["revenue"].pct_change(12)
                out["revenue_mom"] = out["revenue"].pct_change(1)

            return out
        except Exception as e:
            logger.error("Failed to normalize revenue for {}: {}", stock_id, e)
            return None

    def _normalize_valuation(
        self, df: pd.DataFrame, stock_id: str
    ) -> Optional[pd.DataFrame]:
        """Normalize PE / PB / dividend yield data."""
        if df.empty:
            return None

        try:
            out = df.copy()
            out["stock_id"] = stock_id

            col_map = {
                "PER": "pe_ratio",
                "PBR": "pb_ratio",
                "dividend_yield": "dividend_yield",
            }
            out = out.rename(columns={k: v for k, v in col_map.items() if k in out.columns})

            if "date" in out.columns:
                out["date"] = pd.to_datetime(out["date"])

            return out
        except Exception as e:
            logger.error("Failed to normalize valuation for {}: {}", stock_id, e)
            return None


if __name__ == "__main__":
    from datetime import date as _date

    collector = FinancialsCollector()

    # Show look-ahead bias logic
    test_date = _date(2024, 6, 1)
    y, q = FinancialsCollector.latest_available_quarter(test_date)
    print(f"As of {test_date}: latest available quarter = {y}Q{q}")

    ry, rm = FinancialsCollector.latest_available_revenue_month(test_date)
    print(f"As of {test_date}: latest available revenue = {ry}/{rm:02d}")

    # Fetch sample data
    data = collector.fetch_all("2330", "2023-01-01", "2024-06-01")
    for key, df in data.items():
        if df is not None:
            print(f"\n--- {key} ({len(df)} rows) ---")
            print(df.head())
        else:
            print(f"\n--- {key}: no data ---")

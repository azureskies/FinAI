"""
Taiwan stock universe (stock pool) management.

Provides filtered lists of eligible stocks based on market cap, volume,
listing age, and exclusion rules read from configs/data_sources.yaml.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from loguru import logger

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data_sources.yaml"

FINMIND_BASE_URL = "https://api.finmindtrade.com/api/v4"


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class UniverseCollector:
    """Manage the investable stock universe for Taiwan equities."""

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or _load_config()
        uni = cfg.get("universe", {})

        self.min_market_cap: float = uni.get("min_market_cap", 10_000_000_000)
        self.min_avg_volume: int = uni.get("min_avg_volume", 1000)
        self.min_listing_years: int = uni.get("min_listing_years", 2)
        self.exclude_categories: list[str] = uni.get("exclude_categories", [])

        self.finmind_token: str = os.environ.get("FINMIND_TOKEN", "")
        self.rate_limit: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_universe(self, as_of: date) -> pd.DataFrame:
        """Return the stock universe eligible on the given date.

        Steps:
          1. Fetch all listed TWSE/OTC stocks.
          2. Filter by listing age.
          3. Filter by market cap.
          4. Filter by average daily volume.
          5. Exclude special categories (full-delivery, disposition, etc.).

        Returns:
            DataFrame with columns [stock_id, stock_name, market, listing_date,
            market_cap, avg_volume] for stocks that pass all filters.
        """
        logger.info("Building stock universe as of {}", as_of)

        # Step 1: Fetch all listed stocks
        stock_list = self._fetch_stock_list()
        if stock_list is None or stock_list.empty:
            logger.error("Failed to fetch stock list")
            return pd.DataFrame()

        logger.info("Total stocks fetched: {}", len(stock_list))

        # Step 2: Filter by listing age (only if listing_date is available)
        if "listing_date" in stock_list.columns:
            stock_list["listing_date"] = pd.to_datetime(
                stock_list["listing_date"], errors="coerce"
            )
            has_date = stock_list["listing_date"].notna()
            if has_date.sum() > len(stock_list) * 0.5:
                # Majority have valid listing dates — apply filter
                cutoff_date = as_of - timedelta(days=self.min_listing_years * 365)
                before = len(stock_list)
                stock_list = stock_list[
                    stock_list["listing_date"].isna()
                    | (stock_list["listing_date"] <= pd.Timestamp(cutoff_date))
                ]
                logger.info(
                    "After listing age filter (>= {} years): {} -> {}",
                    self.min_listing_years,
                    before,
                    len(stock_list),
                )
            else:
                logger.info(
                    "Skipping listing age filter — only {}/{} stocks have valid dates",
                    has_date.sum(),
                    len(stock_list),
                )

        # Step 3: Exclude special categories
        stock_list = self._exclude_categories(stock_list)

        # Step 4: Filter by market cap
        stock_list = self._filter_market_cap(stock_list, as_of)

        # Step 5: Filter by volume
        stock_list = self._filter_volume(stock_list, as_of)

        logger.info("Final universe size: {} stocks", len(stock_list))
        return stock_list.reset_index(drop=True)

    def get_twse_stocks(self, as_of: date) -> pd.DataFrame:
        """Return only TWSE (listed) stocks from the universe."""
        universe = self.get_universe(as_of)
        if universe.empty or "market" not in universe.columns:
            return universe
        return universe[universe["market"] == "twse"].reset_index(drop=True)

    def get_otc_stocks(self, as_of: date) -> pd.DataFrame:
        """Return only OTC stocks from the universe."""
        universe = self.get_universe(as_of)
        if universe.empty or "market" not in universe.columns:
            return universe
        return universe[universe["market"] == "otc"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Data fetching helpers
    # ------------------------------------------------------------------

    def _finmind_request(self, dataset: str, params: dict) -> Optional[pd.DataFrame]:
        """Send a request to the FinMind API."""
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
            logger.warning(
                "FinMind returned status {}: {}", body.get("status"), body.get("msg")
            )
            return None

        data = body.get("data", [])
        if not data:
            return None

        return pd.DataFrame(data)

    def _fetch_stock_list(self) -> Optional[pd.DataFrame]:
        """Fetch all TWSE and OTC stocks. Tries FinMind first, falls back to TWSE/TPEX OpenAPI."""
        # Try FinMind first
        df = self._fetch_stock_list_finmind()
        if df is not None and not df.empty:
            return df

        # Fallback to TWSE/TPEX public API (no token required)
        logger.info("FinMind unavailable, falling back to TWSE/TPEX OpenAPI")
        return self._fetch_stock_list_twse()

    def _fetch_stock_list_finmind(self) -> Optional[pd.DataFrame]:
        """Fetch stock list from FinMind API."""
        logger.debug("Fetching stock list from FinMind")
        twse = self._finmind_request("TaiwanStockInfo", {})
        time.sleep(self.rate_limit)

        if twse is None:
            return None

        col_map = {
            "stock_id": "stock_id",
            "stock_name": "stock_name",
            "type": "market",
        }
        df = twse.rename(
            columns={k: v for k, v in col_map.items() if k in twse.columns}
        )

        # FinMind's 'date' field is the data update date, NOT listing date.
        # Drop it to avoid confusion; listing_date will be empty.
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        if "market" in df.columns:
            df["market"] = df["market"].map(
                lambda x: "twse" if "上市" in str(x) or "twse" in str(x).lower()
                else ("otc" if "上櫃" in str(x) or "otc" in str(x).lower() else str(x))
            )

        # Keep only common stocks (4-digit numeric stock_id)
        if "stock_id" in df.columns:
            df = df[df["stock_id"].str.match(r"^\d{4}$", na=False)]

        # Deduplicate — keep first occurrence per stock_id
        df = df.drop_duplicates(subset=["stock_id"], keep="first")

        # Add empty listing_date column for consistency
        if "listing_date" not in df.columns:
            df["listing_date"] = ""

        return df

    def _fetch_stock_list_twse(self) -> Optional[pd.DataFrame]:
        """Fetch stock list from TWSE/TPEX public OpenAPI (no token needed)."""
        rows: list[dict] = []

        # TWSE listed stocks
        try:
            resp = requests.get(
                "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
                timeout=30,
            )
            resp.raise_for_status()
            for item in resp.json():
                sid = item.get("公司代號", "").strip()
                if len(sid) == 4 and sid.isdigit():
                    rows.append({
                        "stock_id": sid,
                        "stock_name": item.get("公司簡稱", "").strip(),
                        "market": "twse",
                        "listing_date": item.get("上市日期", ""),
                    })
            logger.info("TWSE OpenAPI: fetched {} listed stocks", len(rows))
        except requests.RequestException as e:
            logger.error("TWSE OpenAPI failed: {}", e)

        time.sleep(0.5)

        # TPEX (OTC) stocks
        otc_count = 0
        try:
            resp = requests.get(
                "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis",
                timeout=30,
            )
            resp.raise_for_status()
            for item in resp.json():
                sid = item.get("SecuritiesCompanyCode", "").strip()
                if len(sid) == 4 and sid.isdigit():
                    rows.append({
                        "stock_id": sid,
                        "stock_name": item.get("CompanyName", "").strip(),
                        "market": "otc",
                        "listing_date": "",
                    })
                    otc_count += 1
            logger.info("TPEX OpenAPI: fetched {} OTC stocks", otc_count)
        except requests.RequestException as e:
            logger.error("TPEX OpenAPI failed: {}", e)

        if not rows:
            return None

        return pd.DataFrame(rows)

    def _fetch_market_cap(self, as_of: date) -> Optional[pd.DataFrame]:
        """Fetch market capitalization data from FinMind."""
        date_str = as_of.strftime("%Y-%m-%d")
        logger.debug("Fetching market cap data for {}", date_str)

        df = self._finmind_request(
            "TaiwanStockMarketValue",
            {"start_date": date_str, "end_date": date_str},
        )
        time.sleep(self.rate_limit)
        return df

    def _fetch_daily_volume(
        self, as_of: date, lookback_days: int = 60
    ) -> Optional[pd.DataFrame]:
        """Fetch daily trading volume for average volume calculation."""
        end_str = as_of.strftime("%Y-%m-%d")
        start = as_of - timedelta(days=lookback_days)
        start_str = start.strftime("%Y-%m-%d")
        logger.debug("Fetching volume data ({} ~ {})", start_str, end_str)

        df = self._finmind_request(
            "TaiwanStockDailyResults",
            {"start_date": start_str, "end_date": end_str},
        )
        time.sleep(self.rate_limit)
        return df

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    def _exclude_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove stocks in excluded categories."""
        if not self.exclude_categories or df.empty:
            return df

        # Check if there's a category / industry column
        cat_col = None
        for col in ["industry_category", "category", "type"]:
            if col in df.columns:
                cat_col = col
                break

        if cat_col is None:
            logger.debug("No category column found; skipping category exclusion")
            return df

        before = len(df)
        mask = ~df[cat_col].isin(self.exclude_categories)
        df = df[mask]
        logger.info(
            "After category exclusion: {} -> {} (excluded: {})",
            before,
            len(df),
            self.exclude_categories,
        )
        return df

    def _filter_market_cap(self, df: pd.DataFrame, as_of: date) -> pd.DataFrame:
        """Filter stocks by minimum market capitalization."""
        if df.empty:
            return df

        mcap_df = self._fetch_market_cap(as_of)
        if mcap_df is None or mcap_df.empty:
            logger.warning("No market cap data available; skipping market cap filter")
            return df

        # Merge market cap
        mcap_col = None
        for col in ["market_value", "MarketValue", "market_cap"]:
            if col in mcap_df.columns:
                mcap_col = col
                break

        if mcap_col is None:
            logger.warning("Cannot find market cap column in data")
            return df

        merge_df = mcap_df[["stock_id", mcap_col]].rename(
            columns={mcap_col: "market_cap"}
        )
        merge_df["market_cap"] = pd.to_numeric(merge_df["market_cap"], errors="coerce")

        before = len(df)
        df = df.merge(merge_df, on="stock_id", how="left")
        df = df[df["market_cap"] >= self.min_market_cap]
        logger.info(
            "After market cap filter (>= {:.0f}): {} -> {}",
            self.min_market_cap,
            before,
            len(df),
        )
        return df

    def _filter_volume(self, df: pd.DataFrame, as_of: date) -> pd.DataFrame:
        """Filter stocks by minimum average daily volume (in lots = 1000 shares)."""
        if df.empty:
            return df

        vol_df = self._fetch_daily_volume(as_of)
        if vol_df is None or vol_df.empty:
            logger.warning("No volume data available; skipping volume filter")
            return df

        vol_col = None
        for col in ["Trading_Volume", "Trading_Volume", "volume"]:
            if col in vol_df.columns:
                vol_col = col
                break

        if vol_col is None:
            logger.warning("Cannot find volume column in data")
            return df

        vol_df[vol_col] = pd.to_numeric(vol_df[vol_col], errors="coerce")

        # Compute average daily volume in lots (1 lot = 1000 shares)
        avg_vol = (
            vol_df.groupby("stock_id")[vol_col]
            .mean()
            .reset_index()
            .rename(columns={vol_col: "avg_volume"})
        )
        avg_vol["avg_volume"] = avg_vol["avg_volume"] / 1000  # convert shares to lots

        before = len(df)
        df = df.merge(avg_vol, on="stock_id", how="left")
        df = df[df["avg_volume"] >= self.min_avg_volume]
        logger.info(
            "After volume filter (>= {} lots/day): {} -> {}",
            self.min_avg_volume,
            before,
            len(df),
        )
        return df


if __name__ == "__main__":
    from datetime import date as _date

    collector = UniverseCollector()
    universe = collector.get_universe(_date(2024, 6, 1))
    if not universe.empty:
        print(f"Universe size: {len(universe)}")
        print(universe.head(20))
    else:
        print("Empty universe (check network / FinMind token)")

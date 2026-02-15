"""
Taiwan stock price data collector using yfinance.

Fetches daily OHLCV + adjusted close data for TWSE and OTC stocks.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import yfinance as yf
from loguru import logger

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "data_sources.yaml"


def _load_config() -> dict:
    """Load data_sources.yaml configuration."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PriceCollector:
    """Collect daily price data for Taiwan stocks via yfinance."""

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or _load_config()
        yf_cfg = next(
            (s for s in cfg.get("price_sources", []) if s["name"] == "yfinance"),
            {},
        )
        self.tw_suffix: str = yf_cfg.get("suffix", ".TW")
        self.otc_suffix: str = yf_cfg.get("otc_suffix", ".TWO")
        self.rate_limit: float = yf_cfg.get("rate_limit", 2.0)

        quality = cfg.get("data_quality", {})
        self.max_missing_rate: float = quality.get("max_missing_rate", 0.05)
        self.price_change_limit: float = quality.get("price_change_limit", 0.11)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        stock_id: str,
        start: str,
        end: str,
        *,
        market: str = "twse",
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data for a single stock.

        Args:
            stock_id: Taiwan stock code, e.g. "2330".
            start: Start date string, e.g. "2023-01-01".
            end: End date string, e.g. "2024-01-01".
            market: "twse" (listed) or "otc" (OTC).

        Returns:
            DataFrame with columns [open, high, low, close, adj_close, volume]
            indexed by date, or None on failure.
        """
        ticker = self._build_ticker(stock_id, market)
        logger.info("Fetching price for {} ({} ~ {})", ticker, start, end)

        try:
            data = yf.download(
                ticker, start=start, end=end, auto_adjust=False, progress=False
            )
        except Exception as e:
            logger.error("yfinance download failed for {}: {}", ticker, e)
            return None

        if data is None or data.empty:
            logger.warning("No data returned for {}", ticker)
            return None

        df = self._normalize(data, stock_id)
        df = self._validate(df, stock_id)
        return df

    def fetch_batch(
        self,
        stock_ids: list[str],
        start: str,
        end: str,
        *,
        market: str = "twse",
        market_map: Optional[dict[str, str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch price data for multiple stocks using yfinance batch download.

        Args:
            market_map: Optional {stock_id: "twse"|"otc"} mapping. If provided,
                        overrides the market parameter per stock.

        Returns:
            Mapping of stock_id -> DataFrame for successful fetches.
        """
        results: dict[str, pd.DataFrame] = {}

        # Build ticker list with correct suffixes
        ticker_to_sid: dict[str, str] = {}
        for sid in stock_ids:
            mkt = (market_map or {}).get(sid, market)
            ticker = self._build_ticker(sid, mkt)
            ticker_to_sid[ticker] = sid

        # Download in batches to avoid overloading yfinance
        batch_size = 50
        tickers = list(ticker_to_sid.keys())
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            batch_num = i // batch_size + 1
            logger.info("Batch download {}/{}: {} tickers", batch_num, total_batches, len(batch))

            try:
                data = yf.download(
                    batch, start=start, end=end, auto_adjust=False,
                    progress=False, group_by="ticker", threads=True,
                )
            except Exception as e:
                logger.error("Batch download failed: {}", e)
                time.sleep(self.rate_limit)
                continue

            if data is None or data.empty:
                logger.warning("Batch returned empty data")
                time.sleep(self.rate_limit)
                continue

            # Extract individual stock data
            for ticker in batch:
                sid = ticker_to_sid[ticker]
                try:
                    if len(batch) == 1:
                        stock_data = data
                    else:
                        stock_data = data[ticker] if ticker in data.columns.get_level_values(0) else pd.DataFrame()

                    if stock_data is not None and not stock_data.empty:
                        stock_data = stock_data.dropna(how="all")
                        if not stock_data.empty:
                            df = self._normalize(stock_data, sid)
                            df = self._validate(df, sid)
                            if not df.empty:
                                results[sid] = df
                except Exception as e:
                    logger.debug("{}: failed to extract from batch — {}", sid, e)

            time.sleep(self.rate_limit)

        logger.info(
            "Batch complete: {}/{} stocks fetched successfully", len(results), len(stock_ids)
        )
        return results

    def fetch_incremental(
        self,
        stock_id: str,
        last_date: str,
        *,
        market: str = "twse",
    ) -> Optional[pd.DataFrame]:
        """Fetch data from last_date+1 to today.

        Args:
            stock_id: Taiwan stock code, e.g. "2330".
            last_date: Last date already in DB, e.g. "2024-01-15".
            market: "twse" (listed) or "otc" (OTC).

        Returns:
            DataFrame with new rows, or None if no new data.
        """
        from datetime import date as _date, timedelta

        next_day = _date.fromisoformat(last_date) + timedelta(days=1)
        today = _date.today()
        if next_day > today:
            logger.debug("{}: already up to date (last_date={})", stock_id, last_date)
            return None

        return self.fetch(
            stock_id,
            str(next_day),
            str(today),
            market=market,
        )

    def compute_adjusted_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute adjusted OHLCV using the adj_close / close ratio.

        This back-adjusts historical prices for dividends and splits so that
        the relative price changes are preserved.
        """
        if df.empty:
            return df

        ratio = df["adj_close"] / df["close"]
        out = df.copy()
        for col in ["open", "high", "low", "close"]:
            out[f"adj_{col}"] = (df[col] * ratio).round(2)
        out["adj_volume"] = df["volume"]  # volume stays the same
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ticker(self, stock_id: str, market: str) -> str:
        suffix = self.otc_suffix if market == "otc" else self.tw_suffix
        return f"{stock_id}{suffix}"

    def _normalize(self, raw: pd.DataFrame, stock_id: str) -> pd.DataFrame:
        """Normalize yfinance output to a consistent schema."""
        df = raw.copy()

        # yfinance may return MultiIndex columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)

        expected = ["open", "high", "low", "close", "adj_close", "volume"]
        missing_cols = [c for c in expected if c not in df.columns]
        if missing_cols:
            logger.warning("Missing columns for {}: {}", stock_id, missing_cols)

        df = df[[c for c in expected if c in df.columns]]
        df.index.name = "date"
        df["stock_id"] = stock_id
        return df

    def _validate(self, df: pd.DataFrame, stock_id: str) -> pd.DataFrame:
        """Validate price data quality."""
        if df.empty:
            return df

        # --- Missing values ---
        missing_rate = df[["open", "high", "low", "close", "volume"]].isna().mean()
        for col, rate in missing_rate.items():
            if rate > self.max_missing_rate:
                logger.warning(
                    "{}: column '{}' has {:.1%} missing (threshold {:.1%})",
                    stock_id,
                    col,
                    rate,
                    self.max_missing_rate,
                )

        # --- Price logic: high >= low, high >= open, high >= close ---
        price_cols = ["open", "high", "low", "close"]
        valid_rows = df[price_cols].dropna()
        if not valid_rows.empty:
            bad_hl = valid_rows["high"] < valid_rows["low"]
            bad_ho = valid_rows["high"] < valid_rows["open"]
            bad_hc = valid_rows["high"] < valid_rows["close"]
            n_bad = (bad_hl | bad_ho | bad_hc).sum()
            if n_bad > 0:
                logger.warning(
                    "{}: {} rows have inconsistent OHLC prices", stock_id, n_bad
                )

        # --- Abnormal daily change ---
        if "close" in df.columns:
            pct = df["close"].pct_change().abs()
            extreme = pct[pct > self.price_change_limit]
            if not extreme.empty:
                logger.debug(
                    "{}: {} days with price change > {:.0%}",
                    stock_id,
                    len(extreme),
                    self.price_change_limit,
                )

        return df


if __name__ == "__main__":
    # Quick smoke test
    collector = PriceCollector()
    result = collector.fetch("2330", "2024-01-01", "2024-03-01")
    if result is not None:
        print(result.head())
        adjusted = collector.compute_adjusted_ohlcv(result)
        print(adjusted.head())
    else:
        print("No data returned for 2330")

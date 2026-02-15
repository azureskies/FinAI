"""Feature engineering module for Taiwan stock AI analysis platform.

Computes technical, fundamental, and market-relative features from raw OHLCV
and financial data. All computations are causal (no lookahead bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta
from loguru import logger


class FeatureEngine:
    """Compute ML features from raw price, financial, and market data.

    All features are computed causally — only using current and past data
    to avoid lookahead bias.
    """

    # ------------------------------------------------------------------ #
    #  Technical indicators (from OHLCV)
    # ------------------------------------------------------------------ #
    def compute_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
            Index should be DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Original columns plus ~40 technical feature columns.
        """
        df = df.copy()
        _o, h, lo, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

        # --- Momentum ---
        df["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()
        df["rsi_28"] = ta.momentum.RSIIndicator(c, window=28).rsi()

        macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        df["roc_5"] = ta.momentum.ROCIndicator(c, window=5).roc()
        df["roc_10"] = ta.momentum.ROCIndicator(c, window=10).roc()
        df["roc_20"] = ta.momentum.ROCIndicator(c, window=20).roc()

        df["momentum_5"] = c / c.shift(5) - 1
        df["momentum_10"] = c / c.shift(10) - 1
        df["momentum_20"] = c / c.shift(20) - 1

        # --- Trend ---
        df["ema_12"] = ta.trend.EMAIndicator(c, window=12).ema_indicator()
        df["ema_26"] = ta.trend.EMAIndicator(c, window=26).ema_indicator()
        df["ema_50"] = ta.trend.EMAIndicator(c, window=50).ema_indicator()

        df["sma_20"] = ta.trend.SMAIndicator(c, window=20).sma_indicator()
        df["sma_60"] = ta.trend.SMAIndicator(c, window=60).sma_indicator()
        df["sma_120"] = ta.trend.SMAIndicator(c, window=120).sma_indicator()

        df["adx_14"] = ta.trend.ADXIndicator(h, lo, c, window=14).adx()

        # --- Volatility ---
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma_20"]

        df["atr_14"] = ta.volatility.AverageTrueRange(h, lo, c, window=14).average_true_range()
        df["atr_28"] = ta.volatility.AverageTrueRange(h, lo, c, window=28).average_true_range()

        log_ret = np.log(c / c.shift(1))
        df["hist_volatility_20"] = log_ret.rolling(window=20).std() * np.sqrt(252)

        # --- Volume ---
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
        df["volume_sma_20"] = v.rolling(window=20).mean()
        df["volume_change"] = v / df["volume_sma_20"] - 1
        df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(h, lo, c, v, window=20).chaikin_money_flow()

        # --- Stochastic ---
        stoch = ta.momentum.StochasticOscillator(h, lo, c, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        stoch_rsi = ta.momentum.StochRSIIndicator(c, window=14, smooth1=3, smooth2=3)
        df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
        df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

        # --- Support / Resistance ---
        high_52w = h.rolling(window=252, min_periods=126).max()
        low_52w = lo.rolling(window=252, min_periods=126).min()
        df["pct_from_52w_high"] = c / high_52w - 1
        df["pct_from_52w_low"] = c / low_52w - 1

        logger.info(
            "Computed {} technical features ({} rows)",
            len([col for col in df.columns if col not in ["open", "high", "low", "close", "volume"]]),
            len(df),
        )
        return df

    # ------------------------------------------------------------------ #
    #  Fundamental features (from financial statements)
    # ------------------------------------------------------------------ #
    def compute_fundamental(
        self, price_df: pd.DataFrame, fin_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute fundamental features aligned to price dates.

        Financial data is forward-filled to avoid lookahead bias: each row
        only uses the most recent *published* financial data as of that date.

        Parameters
        ----------
        price_df : pd.DataFrame
            Daily price data with DatetimeIndex and at least a ``close`` column.
        fin_df : pd.DataFrame
            Financial statement data with DatetimeIndex (publication date) and
            columns such as: eps, bps, revenue, net_income, total_assets,
            total_liabilities, shares_outstanding, dividend_yield.

        Returns
        -------
        pd.DataFrame
            Fundamental feature columns aligned to price_df index.
        """
        result = pd.DataFrame(index=price_df.index)

        if fin_df is None or fin_df.empty:
            logger.warning("No financial data provided; skipping fundamental features")
            return result

        # Forward-fill financial data to daily frequency (no lookahead)
        fin_daily = fin_df.reindex(price_df.index, method="ffill")

        close = price_df["close"]

        # --- Valuation ---
        if "eps" in fin_daily.columns:
            result["pe_ratio"] = close / fin_daily["eps"].replace(0, np.nan)

        if "bps" in fin_daily.columns:
            result["pb_ratio"] = close / fin_daily["bps"].replace(0, np.nan)

        if "dividend_yield" in fin_daily.columns:
            result["dividend_yield"] = fin_daily["dividend_yield"]

        # --- Growth ---
        if "revenue" in fin_daily.columns:
            rev = fin_daily["revenue"]
            # YoY: compare to value ~252 trading days ago
            result["revenue_yoy"] = rev / rev.shift(252) - 1
            # MoM: compare to value ~21 trading days ago
            result["revenue_mom"] = rev / rev.shift(21) - 1

        if "eps" in fin_daily.columns:
            eps = fin_daily["eps"]
            result["eps_growth"] = eps / eps.shift(252) - 1

        # --- Quality ---
        if {"net_income", "total_assets"}.issubset(fin_daily.columns):
            result["roe"] = fin_daily["net_income"] / (
                fin_daily.get("shareholders_equity", fin_daily["total_assets"])
            ).replace(0, np.nan)
            result["roa"] = fin_daily["net_income"] / fin_daily["total_assets"].replace(0, np.nan)

        if {"total_liabilities", "total_assets"}.issubset(fin_daily.columns):
            result["debt_ratio"] = (
                fin_daily["total_liabilities"] / fin_daily["total_assets"].replace(0, np.nan)
            )

        n_feats = result.dropna(axis=1, how="all").shape[1]
        logger.info("Computed {} fundamental features", n_feats)
        return result

    # ------------------------------------------------------------------ #
    #  Market-relative features
    # ------------------------------------------------------------------ #
    def compute_market(
        self, stock_df: pd.DataFrame, market_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute market-relative features.

        Parameters
        ----------
        stock_df : pd.DataFrame
            Individual stock daily data with ``close`` column.
        market_df : pd.DataFrame
            Market index daily data with ``close`` column (e.g. TAIEX / 0050).

        Returns
        -------
        pd.DataFrame
            Market-relative feature columns aligned to stock_df index.
        """
        result = pd.DataFrame(index=stock_df.index)

        if market_df is None or market_df.empty:
            logger.warning("No market data provided; skipping market features")
            return result

        stock_ret = stock_df["close"].pct_change()
        market_ret = market_df["close"].reindex(stock_df.index).pct_change()

        # 20-day rolling relative strength
        result["relative_strength_20"] = (
            stock_ret.rolling(20).mean() / market_ret.rolling(20).mean().replace(0, np.nan)
        )

        logger.info("Computed market-relative features")
        return result

    # ------------------------------------------------------------------ #
    #  Combine all features
    # ------------------------------------------------------------------ #
    def compute_all(
        self,
        price_df: pd.DataFrame,
        fin_df: pd.DataFrame | None = None,
        market_df: pd.DataFrame | None = None,
        market_cap: float | None = None,
    ) -> pd.DataFrame:
        """Compute all features and merge into a single DataFrame.

        Parameters
        ----------
        price_df : pd.DataFrame
            OHLCV data with DatetimeIndex.
        fin_df : pd.DataFrame | None
            Financial statement data (optional).
        market_df : pd.DataFrame | None
            Market index data (optional).
        market_cap : float | None
            Market capitalisation in TWD. Used for market-cap label.

        Returns
        -------
        pd.DataFrame
            Merged feature matrix.
        """
        tech = self.compute_technical(price_df)

        # Start with technical features (exclude raw OHLCV)
        raw_cols = {"open", "high", "low", "close", "volume"}
        feat_cols = [c for c in tech.columns if c not in raw_cols]
        result = tech[feat_cols].copy()

        # Fundamental
        fund = self.compute_fundamental(price_df, fin_df)
        if not fund.empty:
            result = result.join(fund, how="left")

        # Market
        mkt = self.compute_market(price_df, market_df)
        if not mkt.empty:
            result = result.join(mkt, how="left")

        # Market-cap label
        if market_cap is not None:
            if market_cap >= 100_000_000_000:  # >= 100B TWD
                result["market_cap_label"] = 2  # large
            elif market_cap >= 30_000_000_000:  # >= 30B TWD
                result["market_cap_label"] = 1  # mid
            else:
                result["market_cap_label"] = 0  # small

        logger.info("Total features: {} | rows: {}", result.shape[1], result.shape[0])
        return result

    # ------------------------------------------------------------------ #
    #  Feature selection helpers
    # ------------------------------------------------------------------ #
    def remove_correlated(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """Remove highly correlated features.

        For each pair of features with |correlation| > threshold, the feature
        appearing later in column order is dropped.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix (numeric columns only).
        threshold : float
            Absolute correlation threshold (default 0.95).

        Returns
        -------
        pd.DataFrame
            Reduced feature matrix.
        """
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        if to_drop:
            logger.info(
                "Removing {} correlated features (threshold={}): {}",
                len(to_drop),
                threshold,
                to_drop,
            )
        return df.drop(columns=to_drop)


# ---------------------------------------------------------------------- #
#  Quick smoke test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("Running FeatureEngine smoke test...")

    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)

    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    price_df = pd.DataFrame(
        {
            "open": price + np.random.randn(n) * 0.2,
            "high": price + abs(np.random.randn(n)) * 1.0,
            "low": price - abs(np.random.randn(n)) * 1.0,
            "close": price,
            "volume": np.random.randint(1000, 50000, size=n).astype(float),
        },
        index=dates,
    )

    engine = FeatureEngine()

    # Technical only
    tech = engine.compute_technical(price_df)
    logger.info("Technical columns: {}", [c for c in tech.columns if c not in price_df.columns[:5]])

    # Full pipeline (no fin / market data)
    all_feats = engine.compute_all(price_df)
    logger.info("All features shape: {}", all_feats.shape)

    # Correlation filter
    reduced = engine.remove_correlated(all_feats.dropna())
    logger.info("After correlation filter: {}", reduced.shape)

    logger.info("Smoke test passed.")

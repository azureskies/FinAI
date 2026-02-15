"""Multi-factor scoring system for Taiwan stock ranking.

Scores all stocks using cross-sectional percentile ranking across
multiple factor groups, then computes a weighted composite score.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class StockScorer:
    """Multi-factor scoring system with cross-sectional percentile ranking."""

    FACTOR_WEIGHTS = {
        "momentum": 0.25,
        "trend": 0.20,
        "volatility": 0.15,
        "volume": 0.15,
        "ai_prediction": 0.25,
    }

    # Feature columns used for each factor group
    FACTOR_FEATURES = {
        "momentum": ["roc_20", "rsi_14"],
        "trend": ["macd_signal", "adx_14"],
        "volatility": ["bb_width", "atr_14"],  # inverse — lower is better
        "volume": ["volume_change", "cmf"],
    }

    # Factors where lower values are better (will be inverted)
    INVERSE_FACTORS = {"volatility"}

    def score_universe(
        self,
        features_df: pd.DataFrame,
        predictions_df: Optional[pd.DataFrame] = None,
        prices_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Score all stocks, return DataFrame with composite and factor scores.

        Parameters
        ----------
        features_df : pd.DataFrame
            Latest features per stock. Must have 'stock_id' column.
        predictions_df : pd.DataFrame | None
            Model predictions with 'stock_id' and 'predicted_return' columns.
        prices_df : pd.DataFrame | None
            Historical price data for risk metrics. Must have 'stock_id',
            'date', and 'close' columns.

        Returns
        -------
        pd.DataFrame
            Columns: stock_id, composite_score, momentum_score, trend_score,
            volatility_score, volume_score, ai_score, risk_level,
            max_drawdown, volatility_ann, win_rate
        """
        if features_df.empty:
            logger.warning("Empty features DataFrame — returning empty scores")
            return pd.DataFrame()

        result = features_df[["stock_id"]].copy()

        # Score each factor group
        for factor, cols in self.FACTOR_FEATURES.items():
            available = [c for c in cols if c in features_df.columns]
            if not available:
                result[f"{factor}_score"] = 50.0
                logger.warning("No columns for factor '{}', defaulting to 50", factor)
                continue

            group_scores = pd.DataFrame()
            for col in available:
                series = pd.to_numeric(features_df[col], errors="coerce")
                ranked = self._percentile_rank(series)
                if factor in self.INVERSE_FACTORS:
                    ranked = 100.0 - ranked
                group_scores[col] = ranked

            result[f"{factor}_score"] = group_scores.mean(axis=1)

        # AI prediction score
        if predictions_df is not None and not predictions_df.empty:
            pred_map = predictions_df.set_index("stock_id")["predicted_return"]
            matched = features_df["stock_id"].map(pred_map)
            result["ai_score"] = self._percentile_rank(matched)
        else:
            result["ai_score"] = 50.0

        # Composite score (weighted average)
        score_cols = {
            "momentum": "momentum_score",
            "trend": "trend_score",
            "volatility": "volatility_score",
            "volume": "volume_score",
            "ai_prediction": "ai_score",
        }
        composite = pd.Series(0.0, index=result.index)
        for factor, col in score_cols.items():
            weight = self.FACTOR_WEIGHTS[factor]
            composite += result[col].fillna(50.0) * weight
        result["composite_score"] = composite.round(2)

        # Risk metrics
        if prices_df is not None and not prices_df.empty:
            risk_data = []
            for sid in result["stock_id"]:
                metrics = self._compute_risk_metrics(
                    prices_df[prices_df["stock_id"] == sid]
                )
                metrics["stock_id"] = sid
                risk_data.append(metrics)

            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                result = result.merge(risk_df, on="stock_id", how="left")

        # Fill missing risk columns
        for col in ["max_drawdown", "volatility_ann", "win_rate"]:
            if col not in result.columns:
                result[col] = None

        # Classify risk level
        result["risk_level"] = result.apply(
            lambda row: self.classify_risk({
                "max_drawdown": row.get("max_drawdown"),
                "volatility_ann": row.get("volatility_ann"),
                "win_rate": row.get("win_rate"),
            }),
            axis=1,
        )

        logger.info(
            "Scored {} stocks — composite range: {:.1f} ~ {:.1f}",
            len(result),
            result["composite_score"].min(),
            result["composite_score"].max(),
        )
        return result

    def _percentile_rank(self, series: pd.Series) -> pd.Series:
        """Cross-sectional percentile rank -> 0-100."""
        valid = series.dropna()
        if valid.empty:
            return pd.Series(50.0, index=series.index)
        return series.rank(pct=True, na_option="keep").fillna(0.5) * 100

    def _compute_risk_metrics(self, prices_df: pd.DataFrame) -> dict:
        """Compute max_drawdown, volatility_annualized, win_rate."""
        result = {
            "max_drawdown": None,
            "volatility_ann": None,
            "win_rate": None,
        }

        if prices_df.empty or "close" not in prices_df.columns:
            return result

        close = pd.to_numeric(prices_df["close"], errors="coerce").dropna()
        if len(close) < 10:
            return result

        # Max drawdown
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax
        result["max_drawdown"] = round(float(drawdown.min()), 4)

        # Annualized volatility
        daily_ret = close.pct_change().dropna()
        if len(daily_ret) > 1:
            result["volatility_ann"] = round(
                float(daily_ret.std() * np.sqrt(252)), 4
            )

        # Win rate (% of positive daily returns)
        if len(daily_ret) > 0:
            result["win_rate"] = round(
                float((daily_ret > 0).sum() / len(daily_ret)), 4
            )

        return result

    def classify_risk(self, risk_metrics: dict) -> str:
        """Return '低風險' / '中風險' / '高風險'."""
        mdd = risk_metrics.get("max_drawdown")
        vol = risk_metrics.get("volatility_ann")

        if mdd is None or vol is None:
            return "中風險"

        # High risk: drawdown > 30% or volatility > 40%
        if mdd < -0.30 or vol > 0.40:
            return "高風險"
        # Low risk: drawdown > -15% and volatility < 25%
        if mdd > -0.15 and vol < 0.25:
            return "低風險"
        return "中風險"

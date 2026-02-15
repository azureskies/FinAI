"""Performance metrics for backtest evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


TRADING_DAYS_PER_YEAR = 252


class PerformanceMetrics:
    """Collection of risk-adjusted performance metrics."""

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Annualized Sharpe Ratio.

        Args:
            returns: Daily return series.
            risk_free_rate: Annual risk-free rate (default 2%).

        Returns:
            Annualized Sharpe ratio, or 0.0 if volatility is zero.
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf
        return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std())

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Sortino Ratio (penalizes only downside volatility).

        Args:
            returns: Daily return series.
            risk_free_rate: Annual risk-free rate.

        Returns:
            Annualized Sortino ratio, or 0.0 if no downside deviation.
        """
        if returns.empty:
            return 0.0
        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        excess = returns - daily_rf
        downside = excess[excess < 0]
        downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 0.0
        if downside_std == 0:
            return 0.0
        return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / downside_std)

    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        """Calmar Ratio = Annualized Return / |Max Drawdown|.

        Args:
            returns: Daily return series.

        Returns:
            Calmar ratio, or 0.0 if max drawdown is zero.
        """
        ann_ret = PerformanceMetrics.annualized_return(returns)
        mdd = PerformanceMetrics.max_drawdown(returns)
        if mdd == 0:
            return 0.0
        return float(ann_ret / abs(mdd))

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Maximum drawdown from peak (negative value).

        Args:
            returns: Daily return series.

        Returns:
            Maximum drawdown as a negative float (e.g. -0.20 for 20% drawdown).
        """
        if returns.empty:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """Percentage of positive return days.

        Args:
            returns: Daily return series.

        Returns:
            Win rate as a float between 0 and 1.
        """
        if returns.empty:
            return 0.0
        return float((returns > 0).sum() / len(returns))

    @staticmethod
    def information_coefficient(
        predictions: np.ndarray, actuals: np.ndarray
    ) -> float:
        """Spearman rank correlation between predictions and actual returns.

        Args:
            predictions: Predicted return values.
            actuals: Realized return values.

        Returns:
            Spearman rank correlation coefficient.
        """
        if len(predictions) < 2 or len(actuals) < 2:
            return 0.0
        corr, _ = stats.spearmanr(predictions, actuals)
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def annualized_return(returns: pd.Series) -> float:
        """Annualized compound return.

        Args:
            returns: Daily return series.

        Returns:
            Annualized return as a float.
        """
        if returns.empty:
            return 0.0
        total = (1 + returns).prod()
        n_years = len(returns) / TRADING_DAYS_PER_YEAR
        if n_years == 0:
            return 0.0
        return float(total ** (1 / n_years) - 1)

    @staticmethod
    def annualized_volatility(returns: pd.Series) -> float:
        """Annualized volatility (standard deviation).

        Args:
            returns: Daily return series.

        Returns:
            Annualized volatility as a float.
        """
        if returns.empty:
            return 0.0
        return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    @staticmethod
    def calculate_all(
        returns: pd.Series,
        predictions: np.ndarray | None = None,
        actuals: np.ndarray | None = None,
    ) -> dict:
        """Calculate all available performance metrics.

        Args:
            returns: Daily return series.
            predictions: Predicted values (optional, for IC calculation).
            actuals: Realized values (optional, for IC calculation).

        Returns:
            Dictionary of metric_name -> value.
        """
        m = PerformanceMetrics
        result = {
            "sharpe_ratio": m.sharpe_ratio(returns),
            "sortino_ratio": m.sortino_ratio(returns),
            "calmar_ratio": m.calmar_ratio(returns),
            "max_drawdown": m.max_drawdown(returns),
            "win_rate": m.win_rate(returns),
            "annualized_return": m.annualized_return(returns),
            "annualized_volatility": m.annualized_volatility(returns),
            "total_return": float((1 + returns).prod() - 1) if not returns.empty else 0.0,
            "trading_days": len(returns),
        }
        if predictions is not None and actuals is not None:
            result["information_coefficient"] = m.information_coefficient(predictions, actuals)
        logger.info("Metrics: Sharpe={:.3f}, MaxDD={:.2%}, AnnRet={:.2%}",
                     result["sharpe_ratio"], result["max_drawdown"],
                     result["annualized_return"])
        return result

    @staticmethod
    def t_test_vs_benchmark(
        strategy_returns: pd.Series, benchmark_returns: pd.Series
    ) -> dict:
        """Two-sample t-test: is strategy significantly better than benchmark?

        Args:
            strategy_returns: Daily strategy returns.
            benchmark_returns: Daily benchmark returns.

        Returns:
            Dict with t_statistic, p_value, and significant (at 5% level).
        """
        if strategy_returns.empty or benchmark_returns.empty:
            return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}

        t_stat, p_val = stats.ttest_ind(strategy_returns, benchmark_returns)
        # One-sided: strategy > benchmark
        p_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        significant = bool(p_one_sided < 0.05)

        logger.info(
            "T-test vs benchmark: t={:.3f}, p_one_sided={:.4f}, significant={}",
            t_stat, p_one_sided, significant,
        )
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_one_sided),
            "significant": significant,
        }


if __name__ == "__main__":
    np.random.seed(42)

    # Simulate daily returns
    n_days = 504  # ~2 years
    daily_ret = pd.Series(np.random.normal(0.0005, 0.015, n_days))
    bench_ret = pd.Series(np.random.normal(0.0003, 0.012, n_days))

    m = PerformanceMetrics()
    metrics = m.calculate_all(daily_ret)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nT-test vs benchmark:")
    t_result = m.t_test_vs_benchmark(daily_ret, bench_ret)
    for k, v in t_result.items():
        print(f"  {k}: {v}")

    # IC test
    preds = np.random.randn(100)
    actuals = preds * 0.3 + np.random.randn(100) * 0.7
    ic = m.information_coefficient(preds, actuals)
    print(f"\nInformation Coefficient: {ic:.4f}")

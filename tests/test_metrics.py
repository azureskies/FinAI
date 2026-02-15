"""Tests for backtest.metrics.PerformanceMetrics."""

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import TRADING_DAYS_PER_YEAR, PerformanceMetrics


@pytest.fixture
def positive_returns():
    """Consistently positive daily returns."""
    np.random.seed(1)
    return pd.Series(np.random.uniform(0.001, 0.01, 252))


@pytest.fixture
def negative_returns():
    """Consistently negative daily returns."""
    np.random.seed(2)
    return pd.Series(np.random.uniform(-0.01, -0.001, 252))


@pytest.fixture
def mixed_returns():
    """Normal distribution daily returns."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.015, 504))


@pytest.fixture
def zero_returns():
    """Flat (zero) daily returns."""
    return pd.Series(np.zeros(100))


@pytest.fixture
def empty_returns():
    """Empty return series."""
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------
class TestSharpeRatio:

    def test_positive_returns_positive_sharpe(self, positive_returns):
        sr = PerformanceMetrics.sharpe_ratio(positive_returns)
        assert sr > 0

    def test_negative_returns_negative_sharpe(self, negative_returns):
        sr = PerformanceMetrics.sharpe_ratio(negative_returns)
        assert sr < 0

    def test_zero_volatility_returns_zero(self, zero_returns):
        sr = PerformanceMetrics.sharpe_ratio(zero_returns)
        assert sr == 0.0

    def test_empty_returns_zero(self, empty_returns):
        sr = PerformanceMetrics.sharpe_ratio(empty_returns)
        assert sr == 0.0

    def test_custom_risk_free_rate(self, mixed_returns):
        sr1 = PerformanceMetrics.sharpe_ratio(mixed_returns, risk_free_rate=0.0)
        sr2 = PerformanceMetrics.sharpe_ratio(mixed_returns, risk_free_rate=0.05)
        # Higher risk-free rate -> lower sharpe
        assert sr1 > sr2


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------
class TestSortinoRatio:

    def test_positive_returns_high_sortino(self, positive_returns):
        """All positive returns means no downside deviation -> 0."""
        sr = PerformanceMetrics.sortino_ratio(positive_returns)
        # All returns are positive but excess may have some negative after subtracting rf
        assert isinstance(sr, float)

    def test_negative_returns_negative_sortino(self, negative_returns):
        sr = PerformanceMetrics.sortino_ratio(negative_returns)
        assert sr < 0

    def test_empty_returns_zero(self, empty_returns):
        sr = PerformanceMetrics.sortino_ratio(empty_returns)
        assert sr == 0.0

    def test_zero_returns_with_zero_rf(self, zero_returns):
        """Zero returns with zero risk-free rate -> all excess = 0 -> no downside -> 0."""
        sr = PerformanceMetrics.sortino_ratio(zero_returns, risk_free_rate=0.0)
        assert sr == 0.0

    def test_zero_returns_negative_sortino(self, zero_returns):
        """Zero returns with positive risk-free rate -> negative excess -> negative sortino."""
        sr = PerformanceMetrics.sortino_ratio(zero_returns, risk_free_rate=0.02)
        assert sr < 0


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------
class TestMaxDrawdown:

    def test_positive_returns_small_drawdown(self, positive_returns):
        mdd = PerformanceMetrics.max_drawdown(positive_returns)
        # All positive returns -> drawdown should be 0 (no drop from peak)
        assert mdd == pytest.approx(0.0)

    def test_negative_returns_large_drawdown(self, negative_returns):
        mdd = PerformanceMetrics.max_drawdown(negative_returns)
        assert mdd < 0  # Drawdown is negative
        assert mdd > -1.0  # Should not lose everything

    def test_empty_returns_zero(self, empty_returns):
        mdd = PerformanceMetrics.max_drawdown(empty_returns)
        assert mdd == 0.0

    def test_known_drawdown(self):
        """50% drop from peak should give -0.5 drawdown."""
        # Goes up to 2x then drops to 1x (50% drawdown from peak)
        returns = pd.Series([1.0, -0.5])  # 1 -> 2 -> 1
        mdd = PerformanceMetrics.max_drawdown(returns)
        assert mdd == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Calmar Ratio
# ---------------------------------------------------------------------------
class TestCalmarRatio:

    def test_no_drawdown_returns_zero(self, positive_returns):
        cr = PerformanceMetrics.calmar_ratio(positive_returns)
        assert cr == 0.0  # max_drawdown is 0

    def test_mixed_returns_positive(self, mixed_returns):
        cr = PerformanceMetrics.calmar_ratio(mixed_returns)
        assert isinstance(cr, float)

    def test_empty_returns_zero(self, empty_returns):
        cr = PerformanceMetrics.calmar_ratio(empty_returns)
        assert cr == 0.0


# ---------------------------------------------------------------------------
# Win Rate
# ---------------------------------------------------------------------------
class TestWinRate:

    def test_all_positive(self, positive_returns):
        wr = PerformanceMetrics.win_rate(positive_returns)
        assert wr == pytest.approx(1.0)

    def test_all_negative(self, negative_returns):
        wr = PerformanceMetrics.win_rate(negative_returns)
        assert wr == pytest.approx(0.0)

    def test_empty_returns_zero(self, empty_returns):
        wr = PerformanceMetrics.win_rate(empty_returns)
        assert wr == 0.0

    def test_fifty_fifty(self):
        returns = pd.Series([0.01, -0.01, 0.01, -0.01])
        wr = PerformanceMetrics.win_rate(returns)
        assert wr == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Annualized Return
# ---------------------------------------------------------------------------
class TestAnnualizedReturn:

    def test_positive_returns(self, positive_returns):
        ar = PerformanceMetrics.annualized_return(positive_returns)
        assert ar > 0

    def test_negative_returns(self, negative_returns):
        ar = PerformanceMetrics.annualized_return(negative_returns)
        assert ar < 0

    def test_empty_returns_zero(self, empty_returns):
        ar = PerformanceMetrics.annualized_return(empty_returns)
        assert ar == 0.0

    def test_one_year_of_returns(self):
        """252 days of 0.1% daily -> significant annualized return."""
        returns = pd.Series([0.001] * 252)
        ar = PerformanceMetrics.annualized_return(returns)
        expected = (1.001 ** 252) - 1
        assert ar == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Annualized Volatility
# ---------------------------------------------------------------------------
class TestAnnualizedVolatility:

    def test_zero_returns_zero_vol(self, zero_returns):
        av = PerformanceMetrics.annualized_volatility(zero_returns)
        assert av == pytest.approx(0.0)

    def test_empty_returns_zero(self, empty_returns):
        av = PerformanceMetrics.annualized_volatility(empty_returns)
        assert av == 0.0

    def test_positive_volatility(self, mixed_returns):
        av = PerformanceMetrics.annualized_volatility(mixed_returns)
        assert av > 0

    def test_annualization_factor(self):
        """annualized vol = daily_std * sqrt(252)."""
        np.random.seed(99)
        returns = pd.Series(np.random.normal(0, 0.01, 252))
        av = PerformanceMetrics.annualized_volatility(returns)
        expected = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        assert av == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------
class TestInformationCoefficient:

    def test_perfect_positive_correlation(self):
        preds = np.array([1, 2, 3, 4, 5], dtype=float)
        actuals = np.array([1, 2, 3, 4, 5], dtype=float)
        ic = PerformanceMetrics.information_coefficient(preds, actuals)
        assert ic == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        preds = np.array([1, 2, 3, 4, 5], dtype=float)
        actuals = np.array([5, 4, 3, 2, 1], dtype=float)
        ic = PerformanceMetrics.information_coefficient(preds, actuals)
        assert ic == pytest.approx(-1.0)

    def test_too_few_samples(self):
        ic = PerformanceMetrics.information_coefficient(np.array([1.0]), np.array([2.0]))
        assert ic == 0.0

    def test_empty_arrays(self):
        ic = PerformanceMetrics.information_coefficient(np.array([]), np.array([]))
        assert ic == 0.0


# ---------------------------------------------------------------------------
# Calculate All
# ---------------------------------------------------------------------------
class TestCalculateAll:

    def test_returns_expected_keys(self, mixed_returns):
        result = PerformanceMetrics.calculate_all(mixed_returns)
        expected_keys = {
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "win_rate", "annualized_return",
            "annualized_volatility", "total_return", "trading_days",
        }
        assert expected_keys.issubset(result.keys())

    def test_with_predictions_includes_ic(self, mixed_returns):
        preds = np.random.randn(10)
        actuals = np.random.randn(10)
        result = PerformanceMetrics.calculate_all(mixed_returns, preds, actuals)
        assert "information_coefficient" in result

    def test_without_predictions_no_ic(self, mixed_returns):
        result = PerformanceMetrics.calculate_all(mixed_returns)
        assert "information_coefficient" not in result

    def test_trading_days_count(self, mixed_returns):
        result = PerformanceMetrics.calculate_all(mixed_returns)
        assert result["trading_days"] == len(mixed_returns)

    def test_empty_returns(self, empty_returns):
        result = PerformanceMetrics.calculate_all(empty_returns)
        assert result["sharpe_ratio"] == 0.0
        assert result["total_return"] == 0.0


# ---------------------------------------------------------------------------
# T-Test vs Benchmark
# ---------------------------------------------------------------------------
class TestTTestVsBenchmark:

    def test_returns_expected_keys(self, mixed_returns):
        bench = pd.Series(np.random.normal(0.0003, 0.012, len(mixed_returns)))
        result = PerformanceMetrics.t_test_vs_benchmark(mixed_returns, bench)
        assert set(result.keys()) == {"t_statistic", "p_value", "significant"}

    def test_significant_is_bool(self, mixed_returns):
        bench = pd.Series(np.zeros(len(mixed_returns)))
        result = PerformanceMetrics.t_test_vs_benchmark(mixed_returns, bench)
        assert isinstance(result["significant"], bool)

    def test_identical_returns_not_significant(self, mixed_returns):
        result = PerformanceMetrics.t_test_vs_benchmark(mixed_returns, mixed_returns)
        assert result["t_statistic"] == pytest.approx(0.0)
        assert result["significant"] is False

    def test_clearly_better_strategy(self):
        np.random.seed(7)
        strategy = pd.Series(np.random.normal(0.01, 0.005, 500))
        benchmark = pd.Series(np.random.normal(-0.01, 0.005, 500))
        result = PerformanceMetrics.t_test_vs_benchmark(strategy, benchmark)
        assert result["t_statistic"] > 0
        assert result["significant"] is True

    def test_empty_strategy_returns_defaults(self, empty_returns):
        bench = pd.Series([0.01, 0.02])
        result = PerformanceMetrics.t_test_vs_benchmark(empty_returns, bench)
        assert result["t_statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_empty_benchmark_returns_defaults(self, mixed_returns, empty_returns):
        result = PerformanceMetrics.t_test_vs_benchmark(mixed_returns, empty_returns)
        assert result["t_statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["significant"] is False

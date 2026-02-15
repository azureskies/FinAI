"""Tests for backtest.engine.BacktestEngine and BacktestResult."""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine, BacktestResult


# ---------------------------------------------------------------------------
# Helpers to generate synthetic data
# ---------------------------------------------------------------------------

def make_price_data(n_days: int = 60, n_stocks: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic price data in flat format."""
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    records = []
    for i in range(n_stocks):
        sid = f"{2300 + i}"
        base = np.random.uniform(100, 500)
        rets = np.random.normal(0.0003, 0.015, n_days)
        closes = base * np.cumprod(1 + rets)
        for j, d in enumerate(dates):
            o = closes[j] * (1 + np.random.normal(0, 0.003))
            h = max(o, closes[j]) * 1.005
            lo = min(o, closes[j]) * 0.995
            records.append({
                "date": d, "stock_id": sid,
                "open": round(o, 2), "high": round(h, 2),
                "low": round(lo, 2), "close": round(closes[j], 2),
                "volume": int(np.random.uniform(1e6, 5e6)),
            })
    return pd.DataFrame(records)


def make_predictions(price_data: pd.DataFrame, freq: str = "monthly") -> pd.DataFrame:
    """Generate prediction rows on rebalance dates."""
    np.random.seed(99)
    dates = sorted(price_data["date"].unique())
    if freq == "monthly":
        pred_dates = [dates[0]]
        for i, d in enumerate(dates):
            if i > 0 and pd.Timestamp(d).month != pd.Timestamp(dates[i - 1]).month:
                pred_dates.append(d)
    else:
        pred_dates = dates

    stock_ids = price_data["stock_id"].unique()
    records = []
    for d in pred_dates:
        for sid in stock_ids:
            records.append({
                "date": d,
                "stock_id": sid,
                "predicted_return": np.random.normal(0.001, 0.02),
            })
    return pd.DataFrame(records)


def make_benchmark(n_days: int = 60, seed: int = 123) -> pd.DataFrame:
    """Generate benchmark data."""
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rets = np.random.normal(0.0003, 0.01, n_days)
    closes = 100.0 * np.cumprod(1 + rets)
    return pd.DataFrame({"date": dates, "close": closes})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return BacktestEngine()


@pytest.fixture
def price_data():
    return make_price_data(n_days=60, n_stocks=5)


@pytest.fixture
def benchmark_data():
    return make_benchmark(n_days=60)


@pytest.fixture
def predictions(price_data):
    return make_predictions(price_data)


# ---------------------------------------------------------------------------
# BacktestEngine.run basic tests
# ---------------------------------------------------------------------------
class TestBacktestEngineRun:

    def test_returns_backtest_result(self, engine, predictions, price_data, benchmark_data):
        result = engine.run(predictions, price_data, benchmark_data)
        assert isinstance(result, BacktestResult)

    def test_result_has_equity_curve(self, engine, predictions, price_data, benchmark_data):
        result = engine.run(predictions, price_data, benchmark_data)
        assert isinstance(result.equity_curve, pd.Series)
        assert len(result.equity_curve) > 0

    def test_result_has_daily_returns(self, engine, predictions, price_data, benchmark_data):
        result = engine.run(predictions, price_data, benchmark_data)
        assert isinstance(result.daily_returns, pd.Series)
        assert len(result.daily_returns) == len(result.equity_curve)

    def test_result_has_metrics_dict(self, engine, predictions, price_data, benchmark_data):
        result = engine.run(predictions, price_data, benchmark_data)
        assert isinstance(result.metrics, dict)
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "total_return" in result.metrics

    def test_initial_capital_preserved_in_metrics(self, engine, predictions, price_data, benchmark_data):
        capital = 5_000_000
        result = engine.run(predictions, price_data, benchmark_data, initial_capital=capital)
        assert result.metrics["initial_capital"] == capital

    def test_equity_curve_starts_near_initial_capital(self, engine, predictions, price_data, benchmark_data):
        capital = 10_000_000
        result = engine.run(predictions, price_data, benchmark_data, initial_capital=capital)
        assert result.equity_curve.iloc[0] == pytest.approx(capital, rel=0.01)

    def test_benchmark_returns_aligned(self, engine, predictions, price_data, benchmark_data):
        result = engine.run(predictions, price_data, benchmark_data)
        assert isinstance(result.benchmark_returns, pd.Series)
        assert len(result.benchmark_returns) == len(result.equity_curve)

    def test_trades_dataframe(self, engine, predictions, price_data, benchmark_data):
        result = engine.run(predictions, price_data, benchmark_data)
        assert isinstance(result.trades, pd.DataFrame)

    def test_positions_dataframe(self, engine, predictions, price_data, benchmark_data):
        result = engine.run(predictions, price_data, benchmark_data)
        assert isinstance(result.positions, pd.DataFrame)

    def test_t_test_in_metrics(self, engine, predictions, price_data, benchmark_data):
        result = engine.run(predictions, price_data, benchmark_data)
        assert "t_test" in result.metrics
        t_test = result.metrics["t_test"]
        assert "t_statistic" in t_test
        assert "p_value" in t_test
        assert "significant" in t_test


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestBacktestEdgeCases:

    def test_empty_predictions(self, engine, price_data, benchmark_data):
        """No predictions -> no trades, equity stays at initial capital."""
        empty_preds = pd.DataFrame(columns=["date", "stock_id", "predicted_return"])
        result = engine.run(empty_preds, price_data, benchmark_data, initial_capital=10_000_000)
        assert isinstance(result, BacktestResult)
        # No trades should occur
        assert result.trades.empty or len(result.trades) == 0
        # Equity should remain at initial capital
        assert result.equity_curve.iloc[-1] == pytest.approx(10_000_000, rel=0.001)

    def test_single_stock(self, engine, benchmark_data):
        """Backtest with only one stock."""
        price_data = make_price_data(n_days=60, n_stocks=1)
        preds = make_predictions(price_data)
        result = engine.run(preds, price_data, benchmark_data)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_short_period(self, engine):
        """Very short backtest period (5 days)."""
        price_data = make_price_data(n_days=5, n_stocks=3)
        bench = make_benchmark(n_days=5)
        preds = make_predictions(price_data)
        result = engine.run(preds, price_data, bench)
        assert isinstance(result, BacktestResult)

    def test_custom_initial_capital(self, engine, predictions, price_data, benchmark_data):
        r1 = engine.run(predictions, price_data, benchmark_data, initial_capital=1_000_000)
        r2 = engine.run(predictions, price_data, benchmark_data, initial_capital=50_000_000)
        assert r1.metrics["initial_capital"] == 1_000_000
        assert r2.metrics["initial_capital"] == 50_000_000


# ---------------------------------------------------------------------------
# Rebalance date selection
# ---------------------------------------------------------------------------
class TestRebalanceDates:

    def test_monthly_rebalance(self, engine):
        dates = pd.bdate_range("2024-01-01", periods=120)
        rebal = engine._get_rebalance_dates(list(dates))
        # Should have roughly one date per month
        assert 3 <= len(rebal) <= 8

    def test_daily_rebalance(self):
        engine = BacktestEngine()
        engine.rebalance_freq = "daily"
        dates = pd.bdate_range("2024-01-01", periods=20)
        rebal = engine._get_rebalance_dates(list(dates))
        assert len(rebal) == 20

    def test_weekly_rebalance(self):
        engine = BacktestEngine()
        engine.rebalance_freq = "weekly"
        dates = pd.bdate_range("2024-01-01", periods=60)
        rebal = engine._get_rebalance_dates(list(dates))
        # ~12 weeks + first date
        assert 10 <= len(rebal) <= 15

    def test_empty_dates(self, engine):
        rebal = engine._get_rebalance_dates([])
        assert rebal == set()


# ---------------------------------------------------------------------------
# BacktestResult structure
# ---------------------------------------------------------------------------
class TestBacktestResult:

    def test_dataclass_fields(self):
        result = BacktestResult(
            equity_curve=pd.Series([100, 101]),
            daily_returns=pd.Series([0.0, 0.01]),
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={"sharpe_ratio": 1.5},
            benchmark_returns=pd.Series([0.0, 0.005]),
        )
        assert result.equity_curve.iloc[0] == 100
        assert result.metrics["sharpe_ratio"] == 1.5
        assert result.config == {}  # default

    def test_config_field(self):
        result = BacktestResult(
            equity_curve=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={},
            benchmark_returns=pd.Series(dtype=float),
            config={"test": True},
        )
        assert result.config["test"] is True


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------
class TestSummary:

    def test_summary_contains_key_info(self, engine, predictions, price_data, benchmark_data):
        result = engine.run(predictions, price_data, benchmark_data)
        summary = engine.summary(result)
        assert "BACKTEST SUMMARY" in summary
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary
        assert "Total Return" in summary
        assert "Win Rate" in summary

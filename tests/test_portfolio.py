"""Tests for backtest.portfolio.PortfolioManager."""

import numpy as np
import pandas as pd
import pytest

from backtest.costs import TransactionCosts
from backtest.portfolio import PortfolioManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pm_equal():
    return PortfolioManager(method="equal_weight", max_positions=5, max_single_weight=0.30)


@pytest.fixture
def pm_inverse():
    return PortfolioManager(method="inverse_volatility", max_positions=5, max_single_weight=0.30)


@pytest.fixture
def sample_predictions():
    """10 stocks with scores."""
    return pd.DataFrame({
        "stock_id": [f"{2300 + i}" for i in range(10)],
        "predicted_return": np.random.uniform(-0.02, 0.02, 10),
        "score": list(range(10, 0, -1)),  # 10 down to 1
    })


@pytest.fixture
def sample_prices():
    return pd.DataFrame({
        "stock_id": [f"{2300 + i}" for i in range(10)],
        "close": [100.0 + i * 10 for i in range(10)],
    })


@pytest.fixture
def sample_volatilities():
    return pd.Series(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05],
        index=[f"{2300 + i}" for i in range(10)],
    )


# ---------------------------------------------------------------------------
# Equal Weight
# ---------------------------------------------------------------------------
class TestEqualWeight:

    def test_weights_sum_to_one(self, pm_equal):
        stocks = ["A", "B", "C"]
        weights = pm_equal.equal_weight(stocks)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_all_weights_equal(self, pm_equal):
        stocks = ["A", "B", "C", "D"]
        weights = pm_equal.equal_weight(stocks)
        expected = 1.0 / 4
        for w in weights.values():
            assert w == pytest.approx(expected)

    def test_single_stock(self, pm_equal):
        weights = pm_equal.equal_weight(["ONLY"])
        assert weights == {"ONLY": pytest.approx(1.0)}

    def test_empty_list(self, pm_equal):
        weights = pm_equal.equal_weight([])
        assert weights == {}


# ---------------------------------------------------------------------------
# Inverse Volatility
# ---------------------------------------------------------------------------
class TestInverseVolatility:

    def test_weights_sum_to_one(self, pm_inverse, sample_volatilities):
        stocks = list(sample_volatilities.index[:5])
        weights = pm_inverse.inverse_volatility(stocks, sample_volatilities)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_lower_vol_gets_higher_weight(self, pm_inverse):
        vols = pd.Series({"A": 0.01, "B": 0.05})
        weights = pm_inverse.inverse_volatility(["A", "B"], vols)
        assert weights["A"] > weights["B"]

    def test_missing_vol_falls_back(self, pm_inverse):
        """Stocks without vol data should fallback to equal weight."""
        vols = pd.Series({"X": 0.02})
        weights = pm_inverse.inverse_volatility(["A", "B"], vols)
        # No overlap -> fallback to equal weight
        assert len(weights) == 2
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_partial_vol_data(self, pm_inverse):
        """Only stocks with vol data get inverse vol weights."""
        vols = pd.Series({"A": 0.01, "B": 0.02})
        weights = pm_inverse.inverse_volatility(["A", "B", "C"], vols)
        # C has no vol, only A and B should appear
        assert "C" not in weights
        assert sum(weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Allocate (top-level method)
# ---------------------------------------------------------------------------
class TestAllocate:

    def test_respects_max_positions(self, pm_equal, sample_predictions, sample_prices):
        weights = pm_equal.allocate(sample_predictions, sample_prices)
        assert len(weights) <= pm_equal.max_positions

    def test_selects_top_scores(self, pm_equal, sample_predictions, sample_prices):
        weights = pm_equal.allocate(sample_predictions, sample_prices)
        # Top 5 by score are stock_ids 2300..2304 (scores 10..6)
        selected = set(weights.keys())
        expected_top = {f"{2300 + i}" for i in range(5)}
        assert selected == expected_top

    def test_weights_sum_to_one_after_capping(self, pm_equal, sample_predictions, sample_prices):
        weights = pm_equal.allocate(sample_predictions, sample_prices)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_empty_predictions(self, pm_equal, sample_prices):
        empty_preds = pd.DataFrame(columns=["stock_id", "predicted_return", "score"])
        weights = pm_equal.allocate(empty_preds, sample_prices)
        assert weights == {}

    def test_inverse_vol_method(self, pm_inverse, sample_predictions, sample_prices, sample_volatilities):
        weights = pm_inverse.allocate(sample_predictions, sample_prices, volatilities=sample_volatilities)
        assert len(weights) > 0
        assert sum(weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Weight Capping
# ---------------------------------------------------------------------------
class TestCapWeights:

    def test_capping_renormalizes(self):
        pm = PortfolioManager(max_single_weight=0.25)
        raw = {"A": 0.5, "B": 0.3, "C": 0.2}
        capped = pm._cap_weights(raw)
        assert sum(capped.values()) == pytest.approx(1.0)
        # A was 0.5, should be capped to 0.25 before renormalization
        assert capped["A"] <= capped["B"] or capped["A"] == pytest.approx(capped["B"], abs=0.01)

    def test_no_capping_needed(self):
        pm = PortfolioManager(max_single_weight=0.50)
        raw = {"A": 0.3, "B": 0.3, "C": 0.4}
        capped = pm._cap_weights(raw)
        assert sum(capped.values()) == pytest.approx(1.0)

    def test_empty_weights(self):
        pm = PortfolioManager(max_single_weight=0.10)
        capped = pm._cap_weights({})
        assert capped == {}


# ---------------------------------------------------------------------------
# Rebalance
# ---------------------------------------------------------------------------
class TestRebalance:

    def test_buy_from_empty_portfolio(self, pm_equal):
        target = {"A": 0.5, "B": 0.5}
        prices = {"A": 100.0, "B": 200.0}
        costs = TransactionCosts()
        trades = pm_equal.rebalance(
            current_weights={},
            target_weights=target,
            prices=prices,
            capital=1_000_000,
            costs=costs,
        )
        for sid, trade in trades.items():
            assert trade["action"] == "buy"
            assert trade["shares"] > 0
            assert trade["cost"] > 0

    def test_sell_all_positions(self, pm_equal):
        current = {"A": 0.5, "B": 0.5}
        prices = {"A": 100.0, "B": 200.0}
        costs = TransactionCosts()
        trades = pm_equal.rebalance(
            current_weights=current,
            target_weights={},
            prices=prices,
            capital=1_000_000,
            costs=costs,
        )
        for sid, trade in trades.items():
            assert trade["action"] == "sell"

    def test_no_trade_when_same_weights(self, pm_equal):
        weights = {"A": 0.5, "B": 0.5}
        prices = {"A": 100.0, "B": 200.0}
        costs = TransactionCosts()
        trades = pm_equal.rebalance(
            current_weights=weights,
            target_weights=weights,
            prices=prices,
            capital=1_000_000,
            costs=costs,
        )
        assert len(trades) == 0

    def test_invalid_price_skipped(self, pm_equal):
        """Stocks with price <= 0 should be skipped."""
        costs = TransactionCosts()
        trades = pm_equal.rebalance(
            current_weights={},
            target_weights={"A": 1.0},
            prices={"A": 0.0},
            capital=1_000_000,
            costs=costs,
        )
        assert "A" not in trades

    def test_lot_size_rounding(self, pm_equal):
        """Shares should be rounded to multiples of 1000 (Taiwan lot)."""
        costs = TransactionCosts()
        trades = pm_equal.rebalance(
            current_weights={},
            target_weights={"A": 1.0},
            prices={"A": 100.0},
            capital=1_000_000,
            costs=costs,
        )
        if "A" in trades:
            assert trades["A"]["shares"] % 1000 == 0

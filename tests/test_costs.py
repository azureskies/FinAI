"""Tests for backtest.costs.TransactionCosts."""

import pytest

from backtest.costs import TransactionCosts


class TestTransactionCostsDefaults:
    """Test default configuration values."""

    def test_default_commission_rate(self):
        tc = TransactionCosts()
        assert tc.commission_rate == pytest.approx(0.001425)

    def test_default_tax_rate(self):
        tc = TransactionCosts()
        assert tc.tax_rate == pytest.approx(0.003)

    def test_default_min_commission(self):
        tc = TransactionCosts()
        assert tc.min_commission == pytest.approx(20.0)

    def test_default_slippage(self):
        tc = TransactionCosts()
        assert tc.slippage == pytest.approx(0.001)


class TestCalculateBuy:
    """Test calculate() for buy trades."""

    def test_buy_no_tax(self):
        """Buy side should never incur transaction tax."""
        tc = TransactionCosts()
        cost = tc.calculate(price=100.0, shares=1000, is_buy=True)
        trade_value = 100.0 * 1000
        expected_commission = trade_value * 0.001425  # 142.5
        expected_slippage = trade_value * 0.001  # 100.0
        assert cost == pytest.approx(expected_commission + expected_slippage)

    def test_buy_commission_components(self):
        """Verify commission and slippage individually."""
        tc = TransactionCosts()
        price, shares = 600.0, 1000
        trade_value = price * shares
        cost = tc.calculate(price, shares, is_buy=True)
        commission = max(trade_value * tc.commission_rate, tc.min_commission)
        slippage = trade_value * tc.slippage
        assert cost == pytest.approx(commission + slippage)

    def test_buy_min_commission_applied(self):
        """Small trade should use min commission (TWD 20)."""
        tc = TransactionCosts()
        # trade_value = 10 * 1 = 10, commission = 10 * 0.001425 = 0.01425 < 20
        cost = tc.calculate(price=10.0, shares=1, is_buy=True)
        expected = tc.min_commission + 10.0 * tc.slippage
        assert cost == pytest.approx(expected)


class TestCalculateSell:
    """Test calculate() for sell trades."""

    def test_sell_includes_tax(self):
        """Sell side should include transaction tax."""
        tc = TransactionCosts()
        price, shares = 100.0, 1000
        trade_value = price * shares
        cost = tc.calculate(price, shares, is_buy=False)
        commission = max(trade_value * tc.commission_rate, tc.min_commission)
        tax = trade_value * tc.tax_rate
        slippage = trade_value * tc.slippage
        assert cost == pytest.approx(commission + tax + slippage)

    def test_sell_more_expensive_than_buy(self):
        """Sell cost > buy cost because of tax."""
        tc = TransactionCosts()
        buy = tc.calculate(500.0, 1000, is_buy=True)
        sell = tc.calculate(500.0, 1000, is_buy=False)
        assert sell > buy

    def test_sell_tax_amount(self):
        """Exact tax amount = trade_value * 0.003."""
        tc = TransactionCosts()
        price, shares = 200.0, 2000
        sell_cost = tc.calculate(price, shares, is_buy=False)
        buy_cost = tc.calculate(price, shares, is_buy=True)
        # Difference should be exactly the tax
        expected_tax = price * shares * tc.tax_rate
        assert sell_cost - buy_cost == pytest.approx(expected_tax)


class TestCalculateRoundTrip:
    """Test calculate_round_trip()."""

    def test_round_trip_equals_buy_plus_sell(self):
        tc = TransactionCosts()
        price, shares = 300.0, 5000
        rt = tc.calculate_round_trip(price, shares)
        buy = tc.calculate(price, shares, is_buy=True)
        sell = tc.calculate(price, shares, is_buy=False)
        assert rt == pytest.approx(buy + sell)

    def test_round_trip_typical_rate(self):
        """Round trip cost should be roughly 0.5-0.6% for normal trade sizes."""
        tc = TransactionCosts()
        price, shares = 600.0, 1000
        trade_value = price * shares
        rt = tc.calculate_round_trip(price, shares)
        pct = rt / trade_value
        assert 0.004 < pct < 0.008


class TestCustomRates:
    """Test with custom fee parameters."""

    def test_zero_commission(self):
        tc = TransactionCosts(commission_rate=0.0, min_commission=0.0)
        cost = tc.calculate(100.0, 1000, is_buy=True)
        # Only slippage
        assert cost == pytest.approx(100.0 * 1000 * tc.slippage)

    def test_zero_slippage(self):
        tc = TransactionCosts(slippage=0.0)
        cost = tc.calculate(100.0, 1000, is_buy=True)
        expected = max(100_000 * 0.001425, 20.0)
        assert cost == pytest.approx(expected)

    def test_custom_all_rates(self):
        tc = TransactionCosts(
            commission_rate=0.002,
            tax_rate=0.005,
            min_commission=50.0,
            slippage=0.002,
        )
        cost = tc.calculate(500.0, 100, is_buy=False)
        tv = 500.0 * 100
        commission = max(tv * 0.002, 50.0)
        tax = tv * 0.005
        slip = tv * 0.002
        assert cost == pytest.approx(commission + tax + slip)


class TestEdgeCases:
    """Edge case scenarios."""

    def test_zero_shares(self):
        tc = TransactionCosts()
        cost = tc.calculate(100.0, 0, is_buy=True)
        # trade_value=0, commission=max(0,20)=20, slip=0
        assert cost == pytest.approx(tc.min_commission)

    def test_zero_price(self):
        tc = TransactionCosts()
        cost = tc.calculate(0.0, 1000, is_buy=True)
        assert cost == pytest.approx(tc.min_commission)

    def test_large_trade(self):
        """Large trade should not trigger min commission."""
        tc = TransactionCosts()
        price, shares = 1000.0, 100_000
        tv = price * shares
        cost = tc.calculate(price, shares, is_buy=True)
        assert cost == pytest.approx(tv * tc.commission_rate + tv * tc.slippage)

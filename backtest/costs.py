"""Transaction cost calculator for Taiwan stock market."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class TransactionCosts:
    """Model Taiwan stock transaction costs.

    Default rates:
        - Broker commission: 0.1425% each way (negotiable, often discounted).
        - Securities transaction tax: 0.3% on sell side only.
        - Minimum commission per trade: TWD 20.
        - Slippage estimate: 0.1% (large cap default).
    """

    commission_rate: float = 0.001425  # 0.1425%
    tax_rate: float = 0.003            # 0.3% sell only
    min_commission: float = 20         # TWD 20
    slippage: float = 0.001            # 0.1% default

    def calculate(self, price: float, shares: int, is_buy: bool) -> float:
        """Calculate total cost for a single trade.

        Args:
            price: Execution price per share.
            shares: Number of shares traded.
            is_buy: True for buy, False for sell.

        Returns:
            Total cost in TWD (commission + tax + slippage).
        """
        trade_value = price * shares

        # Commission (both sides)
        commission = max(trade_value * self.commission_rate, self.min_commission)

        # Tax (sell only)
        tax = trade_value * self.tax_rate if not is_buy else 0.0

        # Slippage
        slip = trade_value * self.slippage

        total = commission + tax + slip
        logger.debug(
            "Trade cost: {} {} shares @ {:.2f} => commission={:.2f}, tax={:.2f}, "
            "slippage={:.2f}, total={:.2f}",
            "BUY" if is_buy else "SELL",
            shares,
            price,
            commission,
            tax,
            slip,
            total,
        )
        return total

    def calculate_round_trip(self, price: float, shares: int) -> float:
        """Calculate total round-trip cost (buy + sell).

        Args:
            price: Price per share (assumed same for buy and sell).
            shares: Number of shares.

        Returns:
            Total round-trip cost in TWD.
        """
        buy_cost = self.calculate(price, shares, is_buy=True)
        sell_cost = self.calculate(price, shares, is_buy=False)
        return buy_cost + sell_cost


if __name__ == "__main__":
    costs = TransactionCosts()

    # Example: buy 1000 shares of TSMC at TWD 600
    price, shares = 600.0, 1000
    buy = costs.calculate(price, shares, is_buy=True)
    sell = costs.calculate(price, shares, is_buy=False)
    rt = costs.calculate_round_trip(price, shares)

    print(f"Trade value: TWD {price * shares:,.0f}")
    print(f"Buy cost:  TWD {buy:,.2f}")
    print(f"Sell cost: TWD {sell:,.2f}")
    print(f"Round-trip: TWD {rt:,.2f} ({rt / (price * shares) * 100:.3f}%)")

"""Portfolio allocation and rebalancing logic."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtest.costs import TransactionCosts


class PortfolioManager:
    """Manage portfolio construction and rebalancing.

    Supports equal-weight and inverse-volatility allocation methods.
    """

    def __init__(
        self,
        method: str = "equal_weight",
        max_positions: int = 20,
        max_single_weight: float = 0.10,
    ) -> None:
        self.method = method
        self.max_positions = max_positions
        self.max_single_weight = max_single_weight
        logger.info(
            "PortfolioManager: method={}, max_positions={}, max_weight={:.0%}",
            method, max_positions, max_single_weight,
        )

    def allocate(
        self,
        predictions: pd.DataFrame,
        current_prices: pd.DataFrame,
        volatilities: Optional[pd.Series] = None,
    ) -> dict[str, float]:
        """Allocate portfolio weights based on model predictions.

        Args:
            predictions: DataFrame with columns ['stock_id', 'predicted_return', 'score'].
                         Must be pre-sorted or will be sorted by score descending.
            current_prices: DataFrame with columns ['stock_id', 'close'].
            volatilities: Series indexed by stock_id with rolling volatility
                          (required for inverse_volatility method).

        Returns:
            Dict of {stock_id: weight} where weights sum to 1.0.
        """
        # Sort by score descending, take top N
        ranked = predictions.sort_values("score", ascending=False)
        top_stocks = ranked["stock_id"].head(self.max_positions).tolist()

        if len(top_stocks) == 0:
            logger.warning("No stocks selected for allocation")
            return {}

        if self.method == "inverse_volatility" and volatilities is not None:
            weights = self.inverse_volatility(top_stocks, volatilities)
        else:
            weights = self.equal_weight(top_stocks)

        # Enforce max single weight cap
        weights = self._cap_weights(weights)

        logger.info(
            "Allocated {} positions, method={}, max_weight={:.2%}",
            len(weights), self.method, max(weights.values()) if weights else 0,
        )
        return weights

    def equal_weight(self, top_stocks: list[str]) -> dict[str, float]:
        """Equal weight allocation across selected stocks.

        Args:
            top_stocks: List of stock IDs to include.

        Returns:
            Dict of {stock_id: weight}.
        """
        if not top_stocks:
            return {}
        w = 1.0 / len(top_stocks)
        return {sid: w for sid in top_stocks}

    def inverse_volatility(
        self, top_stocks: list[str], volatilities: pd.Series
    ) -> dict[str, float]:
        """Inverse volatility weighting (lower vol -> higher weight).

        Args:
            top_stocks: List of stock IDs.
            volatilities: Series indexed by stock_id.

        Returns:
            Dict of {stock_id: weight}.
        """
        # Filter to stocks that have volatility data
        available = [s for s in top_stocks if s in volatilities.index]
        if not available:
            logger.warning("No volatility data available, falling back to equal weight")
            return self.equal_weight(top_stocks)

        vols = volatilities.loc[available]
        # Replace zero/nan with median to avoid division issues
        median_vol = vols.median()
        vols = vols.replace(0, median_vol).fillna(median_vol)

        inv_vol = 1.0 / vols
        total = inv_vol.sum()
        weights = {sid: float(inv_vol[sid] / total) for sid in available}
        return weights

    def rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        prices: dict[str, float],
        capital: float,
        costs: TransactionCosts,
    ) -> dict[str, dict]:
        """Calculate trades needed to rebalance from current to target weights.

        Args:
            current_weights: Current portfolio weights {stock_id: weight}.
            target_weights: Target portfolio weights {stock_id: weight}.
            prices: Current prices {stock_id: price}.
            capital: Total portfolio value in TWD.
            costs: TransactionCosts instance for cost calculation.

        Returns:
            Dict of {stock_id: {'action': 'buy'|'sell', 'shares': int, 'cost': float}}.
        """
        all_stocks = set(current_weights) | set(target_weights)
        trades: dict[str, dict] = {}

        for sid in all_stocks:
            cur_w = current_weights.get(sid, 0.0)
            tgt_w = target_weights.get(sid, 0.0)
            diff_w = tgt_w - cur_w

            if abs(diff_w) < 1e-6:
                continue

            price = prices.get(sid, 0.0)
            if price <= 0:
                logger.warning("Invalid price for {}, skipping", sid)
                continue

            trade_value = abs(diff_w) * capital
            # Round to lot size (1000 shares in Taiwan market)
            shares = int(trade_value / price / 1000) * 1000
            if shares <= 0:
                continue

            is_buy = diff_w > 0
            cost = costs.calculate(price, shares, is_buy=is_buy)

            trades[sid] = {
                "action": "buy" if is_buy else "sell",
                "shares": shares,
                "cost": cost,
            }

        total_cost = sum(t["cost"] for t in trades.values())
        logger.info(
            "Rebalance: {} trades, total cost TWD {:.0f} ({:.3%} of capital)",
            len(trades), total_cost, total_cost / capital if capital > 0 else 0,
        )
        return trades

    def _cap_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Enforce max single weight constraint and renormalize.

        Args:
            weights: Raw weights.

        Returns:
            Capped and renormalized weights summing to 1.0.
        """
        capped = {k: min(v, self.max_single_weight) for k, v in weights.items()}
        total = sum(capped.values())
        if total == 0:
            return capped
        return {k: v / total for k, v in capped.items()}


if __name__ == "__main__":
    np.random.seed(42)

    # Simulated predictions
    n_stocks = 50
    stock_ids = [f"{2300 + i}" for i in range(n_stocks)]
    preds = pd.DataFrame({
        "stock_id": stock_ids,
        "predicted_return": np.random.normal(0.001, 0.02, n_stocks),
        "score": np.random.uniform(0, 1, n_stocks),
    })
    prices_df = pd.DataFrame({
        "stock_id": stock_ids,
        "close": np.random.uniform(50, 800, n_stocks),
    })
    vols = pd.Series(
        np.random.uniform(0.01, 0.05, n_stocks), index=stock_ids
    )

    # Test equal weight
    pm = PortfolioManager(method="equal_weight", max_positions=10)
    w_eq = pm.allocate(preds, prices_df)
    print(f"Equal weight: {len(w_eq)} positions, sum={sum(w_eq.values()):.4f}")
    for sid, wt in list(w_eq.items())[:3]:
        print(f"  {sid}: {wt:.4f}")

    # Test inverse volatility
    pm_iv = PortfolioManager(method="inverse_volatility", max_positions=10)
    w_iv = pm_iv.allocate(preds, prices_df, volatilities=vols)
    print(f"\nInverse vol: {len(w_iv)} positions, sum={sum(w_iv.values()):.4f}")
    for sid, wt in list(w_iv.items())[:3]:
        print(f"  {sid}: {wt:.4f}")

    # Test rebalance
    tc = TransactionCosts()
    price_dict = {sid: float(prices_df.loc[prices_df["stock_id"] == sid, "close"].iloc[0])
                  for sid in w_eq}
    trades = pm.rebalance(
        current_weights={},
        target_weights=w_eq,
        prices=price_dict,
        capital=10_000_000,
        costs=tc,
    )
    print(f"\nRebalance trades: {len(trades)}")
    for sid, t in list(trades.items())[:3]:
        print(f"  {sid}: {t}")

"""Backtest engine with walk-forward validation support."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from backtest.costs import TransactionCosts
from backtest.metrics import PerformanceMetrics
from backtest.portfolio import PortfolioManager


_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "backtest_config.yaml"


class Predictor(Protocol):
    """Protocol for models used in walk-forward backtesting."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class BacktestResult:
    """Container for backtest output."""

    equity_curve: pd.Series           # Daily portfolio value
    daily_returns: pd.Series          # Daily returns
    positions: pd.DataFrame           # Historical positions
    trades: pd.DataFrame              # Trade log
    metrics: dict                     # Performance metrics
    benchmark_returns: pd.Series      # Benchmark (0050) returns
    config: dict = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtest engine for Taiwan stocks.

    Reads configuration from configs/backtest_config.yaml and supports
    both static-prediction backtests and walk-forward validation with
    rolling model retraining.
    """

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG) -> None:
        self.config = self._load_config(config_path)

        tc_cfg = self.config.get("transaction_costs", {})
        slip_cfg = self.config.get("slippage", {})
        self.costs = TransactionCosts(
            commission_rate=tc_cfg.get("commission_rate", 0.001425),
            tax_rate=tc_cfg.get("tax_rate", 0.003),
            min_commission=tc_cfg.get("min_commission", 20),
            slippage=slip_cfg.get("large_cap", 0.001),
        )

        port_cfg = self.config.get("portfolio", {})
        self.portfolio = PortfolioManager(
            method=port_cfg.get("method", "equal_weight"),
            max_positions=port_cfg.get("max_positions", 20),
            max_single_weight=port_cfg.get("max_single_weight", 0.10),
        )
        self.rebalance_freq = port_cfg.get("rebalance_frequency", "monthly")
        self.metrics_calc = PerformanceMetrics()

        logger.info("BacktestEngine initialized: rebalance={}", self.rebalance_freq)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        predictions: pd.DataFrame,
        price_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        initial_capital: float = 10_000_000,
    ) -> BacktestResult:
        """Run backtest simulation on pre-computed predictions.

        Args:
            predictions: DataFrame ['date', 'stock_id', 'predicted_return'].
            price_data: DataFrame ['date', 'stock_id', 'open', 'high', 'low', 'close', 'volume'].
            benchmark_data: DataFrame ['date', 'close'] for benchmark (0050).
            initial_capital: Starting capital in TWD.

        Returns:
            BacktestResult with full simulation output.
        """
        predictions = predictions.copy()
        price_data = price_data.copy()
        predictions["date"] = pd.to_datetime(predictions["date"])
        price_data["date"] = pd.to_datetime(price_data["date"])

        all_dates = sorted(price_data["date"].unique())
        rebalance_dates = self._get_rebalance_dates(all_dates)

        capital = initial_capital
        current_positions: dict[str, int] = {}   # stock_id -> shares
        current_weights: dict[str, float] = {}

        equity_records: list[dict] = []
        trade_records: list[dict] = []
        position_records: list[dict] = []

        for i, date in enumerate(all_dates):
            date_prices = price_data[price_data["date"] == date]
            price_map = dict(zip(date_prices["stock_id"], date_prices["close"]))

            # Rebalance on designated dates
            if date in rebalance_dates:
                date_preds = predictions[predictions["date"] == date]
                if not date_preds.empty:
                    # Execute at T+1 open price to avoid lookahead bias
                    next_date = all_dates[i + 1] if i + 1 < len(all_dates) else None
                    result = self._execute_rebalance(
                        date, next_date, date_preds, price_data,
                        current_positions, current_weights, capital,
                    )
                    current_positions = result["positions"]
                    current_weights = result["weights"]
                    capital = result["remaining_cash"]
                    trade_records.extend(result["trades"])

            # Mark-to-market: calculate portfolio value
            portfolio_value = capital
            for sid, shares in current_positions.items():
                px = price_map.get(sid, 0.0)
                portfolio_value += px * shares

            equity_records.append({"date": date, "value": portfolio_value})

            # Snapshot positions
            for sid, shares in current_positions.items():
                px = price_map.get(sid, 0.0)
                position_records.append({
                    "date": date,
                    "stock_id": sid,
                    "shares": shares,
                    "price": px,
                    "market_value": px * shares,
                    "weight": (px * shares) / portfolio_value if portfolio_value > 0 else 0,
                })

        # Build result dataframes
        equity_df = pd.DataFrame(equity_records).set_index("date")["value"]
        daily_ret = equity_df.pct_change().fillna(0)

        trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
            columns=["date", "stock_id", "action", "shares", "price", "cost"]
        )
        positions_df = pd.DataFrame(position_records) if position_records else pd.DataFrame(
            columns=["date", "stock_id", "shares", "price", "market_value", "weight"]
        )

        # Benchmark returns
        bench = benchmark_data.copy()
        bench["date"] = pd.to_datetime(bench["date"])
        bench = bench.set_index("date").sort_index()
        bench_ret = bench["close"].pct_change().fillna(0)
        # Align to backtest date range
        bench_ret = bench_ret.reindex(equity_df.index).fillna(0)

        # Metrics
        metrics = self.metrics_calc.calculate_all(daily_ret)
        t_test = self.metrics_calc.t_test_vs_benchmark(daily_ret, bench_ret)
        metrics["t_test"] = t_test

        total_costs = trades_df["cost"].sum() if not trades_df.empty else 0
        metrics["total_trading_costs"] = float(total_costs)
        metrics["num_trades"] = len(trades_df)
        metrics["initial_capital"] = initial_capital
        metrics["final_value"] = float(equity_df.iloc[-1]) if not equity_df.empty else initial_capital

        logger.info(
            "Backtest complete: {} days, final={:,.0f}, return={:.2%}",
            len(equity_df), metrics["final_value"],
            metrics.get("total_return", 0),
        )

        return BacktestResult(
            equity_curve=equity_df,
            daily_returns=daily_ret,
            positions=positions_df,
            trades=trades_df,
            metrics=metrics,
            benchmark_returns=bench_ret,
            config=self.config,
        )

    def walk_forward(
        self,
        feature_data: pd.DataFrame,
        price_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        model: Predictor,
        train_window: int = 756,
        test_window: int = 63,
        target_col: str = "target",
        initial_capital: float = 10_000_000,
    ) -> BacktestResult:
        """Walk-forward backtest with rolling model retraining.

        For each window:
        1. Train model on train_window days.
        2. Predict on test_window days.
        3. Run backtest on test_window.
        4. Slide forward by test_window.
        5. Concatenate all test periods for final result.

        Args:
            feature_data: DataFrame with ['date', 'stock_id', target_col, ...feature columns].
            price_data: DataFrame with ['date', 'stock_id', 'open', 'close', ...].
            benchmark_data: DataFrame with ['date', 'close'] for benchmark.
            model: Object implementing fit(X, y) and predict(X).
            train_window: Number of days for training.
            test_window: Number of days for testing (rebalance period).
            target_col: Name of the target column in feature_data.
            initial_capital: Starting capital in TWD.

        Returns:
            BacktestResult aggregated from all test windows.
        """
        feature_data = feature_data.copy()
        feature_data["date"] = pd.to_datetime(feature_data["date"])

        dates = sorted(feature_data["date"].unique())
        total_dates = len(dates)

        all_predictions: list[pd.DataFrame] = []
        fold = 0

        start_idx = train_window
        while start_idx + test_window <= total_dates:
            fold += 1
            train_dates = dates[start_idx - train_window: start_idx]
            test_dates = dates[start_idx: start_idx + test_window]

            logger.info(
                "Walk-forward fold {}: train {} ~ {}, test {} ~ {}",
                fold,
                train_dates[0].strftime("%Y-%m-%d"),
                train_dates[-1].strftime("%Y-%m-%d"),
                test_dates[0].strftime("%Y-%m-%d"),
                test_dates[-1].strftime("%Y-%m-%d"),
            )

            # Split data
            train_mask = feature_data["date"].isin(train_dates)
            test_mask = feature_data["date"].isin(test_dates)

            train_df = feature_data[train_mask]
            test_df = feature_data[test_mask]

            # Identify feature columns (exclude meta columns)
            meta_cols = {"date", "stock_id", target_col}
            feature_cols = [c for c in feature_data.columns if c not in meta_cols]

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]

            # Train and predict
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            fold_preds = test_df[["date", "stock_id"]].copy()
            fold_preds["predicted_return"] = preds
            fold_preds["score"] = preds  # Use prediction as score
            all_predictions.append(fold_preds)

            start_idx += test_window

        if not all_predictions:
            logger.warning("No walk-forward folds completed")
            return BacktestResult(
                equity_curve=pd.Series(dtype=float),
                daily_returns=pd.Series(dtype=float),
                positions=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics={},
                benchmark_returns=pd.Series(dtype=float),
                config=self.config,
            )

        combined_preds = pd.concat(all_predictions, ignore_index=True)
        logger.info(
            "Walk-forward complete: {} folds, {} prediction rows",
            fold, len(combined_preds),
        )

        return self.run(combined_preds, price_data, benchmark_data, initial_capital)

    def summary(self, result: BacktestResult) -> str:
        """Generate human-readable summary of backtest results.

        Args:
            result: BacktestResult from run() or walk_forward().

        Returns:
            Formatted summary string.
        """
        m = result.metrics
        lines = [
            "=" * 60,
            "BACKTEST SUMMARY",
            "=" * 60,
            f"Period: {result.equity_curve.index[0]} ~ {result.equity_curve.index[-1]}"
            if not result.equity_curve.empty else "Period: N/A",
            f"Trading Days: {m.get('trading_days', 0)}",
            f"Initial Capital: TWD {m.get('initial_capital', 0):,.0f}",
            f"Final Value:     TWD {m.get('final_value', 0):,.0f}",
            "",
            "--- Returns ---",
            f"Total Return:      {m.get('total_return', 0):+.2%}",
            f"Annualized Return: {m.get('annualized_return', 0):+.2%}",
            f"Annualized Vol:    {m.get('annualized_volatility', 0):.2%}",
            "",
            "--- Risk-Adjusted ---",
            f"Sharpe Ratio:  {m.get('sharpe_ratio', 0):.3f}",
            f"Sortino Ratio: {m.get('sortino_ratio', 0):.3f}",
            f"Calmar Ratio:  {m.get('calmar_ratio', 0):.3f}",
            f"Max Drawdown:  {m.get('max_drawdown', 0):.2%}",
            "",
            "--- Trading ---",
            f"Win Rate:       {m.get('win_rate', 0):.2%}",
            f"Total Trades:   {m.get('num_trades', 0)}",
            f"Trading Costs:  TWD {m.get('total_trading_costs', 0):,.0f}",
        ]

        t_test = m.get("t_test", {})
        if t_test:
            lines.extend([
                "",
                "--- Statistical Test vs Benchmark ---",
                f"T-statistic: {t_test.get('t_statistic', 0):.3f}",
                f"P-value:     {t_test.get('p_value', 0):.4f}",
                f"Significant: {t_test.get('significant', False)}",
            ])

        if "information_coefficient" in m:
            lines.append(f"\nInformation Coefficient: {m['information_coefficient']:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_rebalance(
        self,
        signal_date: Any,
        execution_date: Any | None,
        predictions: pd.DataFrame,
        price_data: pd.DataFrame,
        current_positions: dict[str, int],
        current_weights: dict[str, float],
        cash: float,
    ) -> dict:
        """Execute a single rebalance event.

        Predictions are generated on signal_date; trades execute at
        execution_date open prices to avoid lookahead bias.

        Returns:
            Dict with 'positions', 'weights', 'remaining_cash', 'trades'.
        """
        # Prepare predictions for allocator
        pred_df = predictions.copy()
        if "score" not in pred_df.columns:
            pred_df["score"] = pred_df["predicted_return"]

        # Get execution prices (T+1 open)
        if execution_date is not None:
            exec_prices = price_data[price_data["date"] == execution_date]
        else:
            # Fallback to signal date close if no next date
            exec_prices = price_data[price_data["date"] == signal_date]

        if exec_prices.empty:
            logger.warning("No price data for execution date {}", execution_date)
            return {
                "positions": current_positions,
                "weights": current_weights,
                "remaining_cash": cash,
                "trades": [],
            }

        price_col = "open" if "open" in exec_prices.columns and execution_date is not None else "close"
        price_map = dict(zip(exec_prices["stock_id"], exec_prices[price_col]))

        # Calculate total portfolio value
        portfolio_value = cash
        for sid, shares in current_positions.items():
            px = price_map.get(sid, 0.0)
            portfolio_value += px * shares

        # Allocate target weights
        target_weights = self.portfolio.allocate(pred_df, exec_prices)

        # Calculate trades
        trades = self.portfolio.rebalance(
            current_weights, target_weights, price_map, portfolio_value, self.costs,
        )

        # Execute trades
        new_positions = dict(current_positions)
        trade_records = []
        total_cash = cash

        # Sell first to free up cash
        for sid, trade in sorted(trades.items(), key=lambda x: x[1]["action"] != "sell"):
            action = trade["action"]
            shares = trade["shares"]
            cost = trade["cost"]
            price = price_map.get(sid, 0.0)

            if action == "sell":
                cur_shares = new_positions.get(sid, 0)
                sell_shares = min(shares, cur_shares)
                if sell_shares <= 0:
                    continue
                total_cash += price * sell_shares - cost
                new_positions[sid] = cur_shares - sell_shares
                if new_positions[sid] <= 0:
                    del new_positions[sid]
            else:  # buy
                buy_cost = price * shares + cost
                if buy_cost > total_cash:
                    # Reduce shares to fit available cash
                    affordable = int((total_cash - cost) / price / 1000) * 1000
                    if affordable <= 0:
                        continue
                    shares = affordable
                    buy_cost = price * shares + cost
                total_cash -= buy_cost
                new_positions[sid] = new_positions.get(sid, 0) + shares

            trade_records.append({
                "date": execution_date or signal_date,
                "stock_id": sid,
                "action": action,
                "shares": shares,
                "price": price,
                "cost": cost,
            })

        # Recalculate actual weights
        total_value = total_cash
        for sid, shares in new_positions.items():
            px = price_map.get(sid, 0.0)
            total_value += px * shares

        new_weights = {}
        for sid, shares in new_positions.items():
            px = price_map.get(sid, 0.0)
            new_weights[sid] = (px * shares) / total_value if total_value > 0 else 0

        return {
            "positions": new_positions,
            "weights": new_weights,
            "remaining_cash": total_cash,
            "trades": trade_records,
        }

    def _get_rebalance_dates(self, all_dates: list) -> set:
        """Determine rebalance dates from the full date list.

        Args:
            all_dates: Sorted list of trading dates.

        Returns:
            Set of dates on which to rebalance.
        """
        if not all_dates:
            return set()

        dates = pd.DatetimeIndex(all_dates)

        if self.rebalance_freq == "daily":
            return set(all_dates)

        if self.rebalance_freq == "weekly":
            # Rebalance on each Monday (or first trading day of the week)
            rebal = set()
            for d in dates:
                # Monday = 0
                if d.weekday() == 0:
                    rebal.add(d)
            # Ensure first date is included
            rebal.add(dates[0])
            return rebal

        # Default: monthly (first trading day of each month)
        rebal = set()
        seen_months: set[tuple[int, int]] = set()
        for d in dates:
            ym = (d.year, d.month)
            if ym not in seen_months:
                seen_months.add(ym)
                rebal.add(d)
        return rebal

    @staticmethod
    def _load_config(path: str | Path) -> dict:
        """Load backtest configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config not found at {}, using defaults", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("Loaded backtest config from {}", path)
        return cfg


if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data for smoke test
    n_days = 252
    n_stocks = 30
    dates = pd.bdate_range("2023-01-01", periods=n_days)

    # Price data
    price_records = []
    for sid_i in range(n_stocks):
        sid = f"{2300 + sid_i}"
        base = np.random.uniform(100, 600)
        returns = np.random.normal(0.0003, 0.02, n_days)
        closes = base * np.cumprod(1 + returns)
        for j, d in enumerate(dates):
            o = closes[j] * (1 + np.random.normal(0, 0.005))
            h = max(o, closes[j]) * (1 + abs(np.random.normal(0, 0.005)))
            lo = min(o, closes[j]) * (1 - abs(np.random.normal(0, 0.005)))
            price_records.append({
                "date": d, "stock_id": sid,
                "open": round(o, 2), "high": round(h, 2),
                "low": round(lo, 2), "close": round(closes[j], 2),
                "volume": int(np.random.uniform(1e6, 1e7)),
            })
    price_df = pd.DataFrame(price_records)

    # Predictions (monthly)
    pred_records = []
    monthly_dates = [dates[0]] + [d for i, d in enumerate(dates) if i > 0 and d.month != dates[i - 1].month]
    for d in monthly_dates:
        for sid_i in range(n_stocks):
            sid = f"{2300 + sid_i}"
            pred_records.append({
                "date": d, "stock_id": sid,
                "predicted_return": np.random.normal(0.001, 0.02),
            })
    pred_df = pd.DataFrame(pred_records)

    # Benchmark
    bench_returns = np.random.normal(0.0003, 0.01, n_days)
    bench_close = 100 * np.cumprod(1 + bench_returns)
    bench_df = pd.DataFrame({"date": dates, "close": bench_close})

    # Run backtest
    engine = BacktestEngine()
    result = engine.run(pred_df, price_df, bench_df)
    print(engine.summary(result))

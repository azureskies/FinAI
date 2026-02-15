"""Monthly performance report generation.

Loads predictions and actual returns for the past month, calculates
performance metrics, and compares against the 0050 benchmark.

Usage:
    python -m scripts.monthly_report
    python -m scripts.monthly_report --dry-run
    python -m scripts.monthly_report --month 2025-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from data.collectors.price import PriceCollector
from data.loaders.supabase import SupabaseLoader

_BACKTEST_CONFIG_PATH = "configs/backtest_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly performance report")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate report without saving to Supabase",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Report month in YYYY-MM format (default: previous month)",
    )
    return parser.parse_args()


def _load_backtest_config() -> dict:
    with open(_BACKTEST_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_month_range(month_str: str | None) -> tuple[str, str]:
    """Return (start_date, end_date) for the target month."""
    if month_str:
        year, mon = month_str.split("-")
        start = date(int(year), int(mon), 1)
    else:
        today = date.today()
        first_of_this_month = today.replace(day=1)
        start = (first_of_this_month - timedelta(days=1)).replace(day=1)

    # End of month
    if start.month == 12:
        end = date(start.year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(start.year, start.month + 1, 1) - timedelta(days=1)

    return str(start), str(end)


def _compute_metrics(returns: pd.Series) -> dict:
    """Compute standard performance metrics from a return series."""
    if returns.empty or len(returns) < 2:
        return {}

    trading_days = 252
    n = len(returns)

    total_return = float((1 + returns).prod() - 1)
    ann_return = float((1 + total_return) ** (trading_days / max(n, 1)) - 1)
    ann_vol = float(returns.std() * np.sqrt(trading_days))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_std = float(downside.std() * np.sqrt(trading_days)) if len(downside) > 0 else 0.0
    sortino = ann_return / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative / peak - 1)
    max_dd = float(drawdown.min())

    win_rate = float((returns > 0).mean())

    return {
        "total_return": round(total_return, 4),
        "annualized_return": round(ann_return, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "trading_days": n,
    }


def main() -> None:
    args = parse_args()
    config = _load_backtest_config()

    start_date, end_date = _get_month_range(args.month)
    logger.info("=== Monthly Report ===")
    logger.info("Period: {} to {}", start_date, end_date)

    db = SupabaseLoader()
    collector = PriceCollector()

    # Pipeline tracking and alerting
    tracker = None
    alert_mgr = None
    run_id = None

    if not args.dry_run:
        from monitoring.alerts import AlertLevel, AlertManager
        from monitoring.tracker import PipelineTracker

        tracker = PipelineTracker(db)
        alert_mgr = AlertManager()
        run_id = tracker.start_run(
            "monthly_report", metadata={"period": f"{start_date}~{end_date}"}
        )

    try:
        # -------------------------------------------------------------- #
        # 1. Load predictions and actual returns
        # -------------------------------------------------------------- #
        logger.info("Step 1: Loading predictions and actual prices...")

        # Get all predictions in the month range
        predictions_resp = (
            db.client.table("predictions")
            .select("date, stock_id, predicted_return, score, model_version")
            .gte("date", start_date)
            .lte("date", end_date)
            .order("date")
            .execute()
        )
        if not predictions_resp.data:
            logger.warning("No predictions found for this period")
            sys.exit(0)

        preds_df = pd.DataFrame(predictions_resp.data)
        preds_df["date"] = pd.to_datetime(preds_df["date"])
        stock_ids = preds_df["stock_id"].unique().tolist()

        # Get actual prices for the period (plus buffer for forward returns)
        buffer_end = str(date.fromisoformat(end_date) + timedelta(days=10))
        prices_df = db.get_prices(stock_ids, start_date, buffer_end)

        if prices_df.empty:
            logger.warning("No price data found for prediction stocks")
            sys.exit(0)

        # -------------------------------------------------------------- #
        # 2. Calculate performance metrics
        # -------------------------------------------------------------- #
        logger.info("Step 2: Calculating performance metrics...")

        # Compute actual forward returns
        prices_df = prices_df.sort_values(["stock_id", "date"])
        prices_df["actual_return"] = (
            prices_df.groupby("stock_id")["close"]
            .transform(lambda s: s.shift(-5) / s - 1)
        )

        # Merge predictions with actuals
        merged = preds_df.merge(
            prices_df[["date", "stock_id", "actual_return"]],
            on=["date", "stock_id"],
            how="inner",
        )
        merged = merged.dropna(subset=["actual_return"])

        if merged.empty:
            logger.warning("No matched prediction-actual pairs")
            sys.exit(0)

        # Information Coefficient (rank correlation per date)
        from scipy.stats import spearmanr

        ic_by_date = []
        for dt, group in merged.groupby("date"):
            if len(group) >= 5:
                ic, _ = spearmanr(group["predicted_return"], group["actual_return"])
                if not np.isnan(ic):
                    ic_by_date.append({"date": dt, "ic": ic})

        ic_df = pd.DataFrame(ic_by_date) if ic_by_date else pd.DataFrame(columns=["date", "ic"])
        mean_ic = float(ic_df["ic"].mean()) if not ic_df.empty else 0.0
        ic_ir = float(ic_df["ic"].mean() / ic_df["ic"].std()) if not ic_df.empty and ic_df["ic"].std() > 0 else 0.0

        # Simulate top-N portfolio returns (long top 5 by score each period)
        portfolio_returns = []
        for dt, group in merged.groupby("date"):
            top_n = group.nlargest(5, "score")
            avg_ret = top_n["actual_return"].mean()
            portfolio_returns.append({"date": dt, "return": avg_ret})

        port_df = pd.DataFrame(portfolio_returns)
        port_metrics = _compute_metrics(pd.Series(port_df["return"].values)) if not port_df.empty else {}

        # -------------------------------------------------------------- #
        # 3. Compare vs benchmark (0050)
        # -------------------------------------------------------------- #
        logger.info("Step 3: Comparing vs benchmark (0050)...")
        benchmark_symbol = config.get("benchmark", {}).get("symbol", "0050")

        bench_price = collector.fetch(benchmark_symbol, start_date, buffer_end)
        bench_metrics = {}
        if bench_price is not None and not bench_price.empty:
            bench_returns = bench_price["close"].pct_change().dropna()
            bench_metrics = _compute_metrics(bench_returns)

        # -------------------------------------------------------------- #
        # 4. Generate summary
        # -------------------------------------------------------------- #
        logger.info("Step 4: Generating summary...")

        report = {
            "period_start": start_date,
            "period_end": end_date,
            "prediction_count": len(preds_df),
            "matched_count": len(merged),
            "unique_stocks": len(stock_ids),
            "mean_ic": round(mean_ic, 4),
            "ic_ir": round(ic_ir, 4),
            "portfolio_metrics": port_metrics,
            "benchmark_metrics": bench_metrics,
        }

        # Excess return
        port_ret = port_metrics.get("total_return", 0.0)
        bench_ret = bench_metrics.get("total_return", 0.0)
        report["excess_return"] = round(port_ret - bench_ret, 4)

        logger.info("--- Report Summary ---")
        logger.info("Period: {} ~ {}", start_date, end_date)
        logger.info("Mean IC: {:.4f} | IC IR: {:.4f}", mean_ic, ic_ir)
        logger.info("Portfolio return: {:.2%}", port_ret)
        logger.info("Benchmark return: {:.2%}", bench_ret)
        logger.info("Excess return: {:.2%}", report["excess_return"])
        if port_metrics:
            logger.info("Sharpe: {:.2f} | Max DD: {:.2%}",
                         port_metrics.get("sharpe_ratio", 0), port_metrics.get("max_drawdown", 0))

        # -------------------------------------------------------------- #
        # 5. Save to Supabase
        # -------------------------------------------------------------- #
        if args.dry_run:
            logger.info("[DRY RUN] Would save report to Supabase")
        else:
            logger.info("Step 5: Saving report to Supabase...")
            db.save_backtest({
                "run_date": str(date.today()),
                "model_type": "monthly_report",
                "period_start": start_date,
                "period_end": end_date,
                "metrics": report,
                "config": config,
            })

        logger.info("Monthly report completed successfully")

        # Record pipeline success
        if tracker and run_id:
            tracker.finish_run(run_id, status="success", metrics=report)
        if alert_mgr:
            from monitoring.alerts import AlertLevel

            sharpe = port_metrics.get("sharpe_ratio", 0)
            max_dd = port_metrics.get("max_drawdown", 0)
            alert_mgr.send_alert(
                level=AlertLevel.INFO,
                title="Monthly report completed",
                message=(
                    f"Period: {start_date}~{end_date}\n"
                    f"Return: {port_ret:.2%} vs Benchmark: {bench_ret:.2%} "
                    f"(Excess: {report['excess_return']:.2%})\n"
                    f"Sharpe: {sharpe:.2f}, MaxDD: {max_dd:.2%}, IC: {mean_ic:.4f}"
                ),
            )

    except Exception as exc:
        logger.error("Monthly report pipeline failed: {}", exc)
        if tracker and run_id:
            tracker.finish_run(run_id, status="failed", error=str(exc))
        if alert_mgr:
            from monitoring.alerts import AlertLevel
            alert_mgr.send_alert(
                level=AlertLevel.CRITICAL,
                title="Monthly report FAILED",
                message=str(exc),
            )
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Run all health checks and send alerts for failures.

Usage:
    python -m scripts.health_check
    python -m scripts.health_check --alert
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from data.loaders import DatabaseLoader
from monitoring.alerts import AlertLevel, AlertManager
from monitoring.health import CheckStatus, HealthChecker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinAI health checks")
    parser.add_argument(
        "--alert",
        action="store_true",
        help="Send alerts for any failures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=== FinAI Health Check ===")
    db = DatabaseLoader()
    if hasattr(db, "init_schema"):
        db.init_schema()
    checker = HealthChecker(db)
    report = checker.run_all_checks()

    # Print summary
    logger.info("Overall status: {}", report.overall_status.value)
    for check in report.checks:
        logger.info("  [{}] {}: {}", check.status.value, check.name, check.message)

    # Send alerts if requested
    if args.alert and report.overall_status != CheckStatus.OK:
        alert_mgr = AlertManager()
        level = (
            AlertLevel.CRITICAL
            if report.overall_status == CheckStatus.CRITICAL
            else AlertLevel.WARNING
        )

        failed_checks = [c for c in report.checks if c.status != CheckStatus.OK]
        body = "\n".join(
            f"- [{c.status.value}] {c.name}: {c.message}" for c in failed_checks
        )

        alert_mgr.send_alert(
            level=level,
            title=f"Health check: {report.overall_status.value}",
            message=body,
        )

    if report.overall_status == CheckStatus.CRITICAL:
        sys.exit(1)

    logger.info("Health check completed")


if __name__ == "__main__":
    main()

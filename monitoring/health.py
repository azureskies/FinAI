"""Health checker for FinAI pipeline components.

Verifies data freshness, model health, prediction coverage,
and feature completeness. Returns a unified HealthReport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

import pandas as pd
import yaml
from loguru import logger

from data.loaders import DatabaseLoader

_CONFIG_PATH = "configs/monitoring.yaml"


class CheckStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    status: CheckStatus
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class HealthReport:
    """Aggregated health report from all checks."""

    timestamp: str
    overall_status: CheckStatus
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class HealthChecker:
    """Run health checks against the FinAI platform."""

    def __init__(
        self,
        db: DatabaseLoader,
        config: Optional[dict] = None,
    ) -> None:
        self.db = db
        if config is not None:
            self.config = config
        else:
            self.config = self._load_config()
        self.thresholds = self.config.get("thresholds", {})

    @staticmethod
    def _load_config() -> dict:
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Monitoring config not found at {}", _CONFIG_PATH)
            return {}

    # ------------------------------------------------------------------ #
    #  Individual checks
    # ------------------------------------------------------------------ #

    def check_data_freshness(self) -> CheckResult:
        """Verify latest price data is from most recent trading day."""
        max_age_hours = self.thresholds.get("max_data_age_hours", 48)

        try:
            resp = (
                self.db.client.table("stock_prices")
                .select("date")
                .order("date", desc=True)
                .limit(1)
                .execute()
            )
            if not resp.data:
                return CheckResult(
                    name="data_freshness",
                    status=CheckStatus.CRITICAL,
                    message="No price data found in database",
                )

            latest_date = pd.Timestamp(resp.data[0]["date"])
            age_hours = (datetime.now() - latest_date).total_seconds() / 3600

            if age_hours > max_age_hours:
                return CheckResult(
                    name="data_freshness",
                    status=CheckStatus.WARNING,
                    message=f"Latest price data is {age_hours:.0f}h old (threshold: {max_age_hours}h)",
                    details={"latest_date": str(latest_date.date()), "age_hours": round(age_hours, 1)},
                )

            return CheckResult(
                name="data_freshness",
                status=CheckStatus.OK,
                message=f"Price data up to date ({latest_date.date()})",
                details={"latest_date": str(latest_date.date()), "age_hours": round(age_hours, 1)},
            )
        except Exception as exc:
            return CheckResult(
                name="data_freshness",
                status=CheckStatus.CRITICAL,
                message=f"Failed to check data freshness: {exc}",
            )

    def check_model_health(self) -> CheckResult:
        """Verify active model IC > 0 and model is not stale."""
        min_ic = self.thresholds.get("min_ic", 0.0)

        try:
            resp = (
                self.db.client.table("model_versions")
                .select("id, model_type, metrics, is_active, created_at")
                .eq("is_active", True)
                .execute()
            )
            if not resp.data:
                return CheckResult(
                    name="model_health",
                    status=CheckStatus.WARNING,
                    message="No active models found",
                )

            issues = []
            for model in resp.data:
                metrics = model.get("metrics", {})
                val_ic = metrics.get("val_ic", 0.0)
                if val_ic <= min_ic:
                    issues.append(
                        f"{model['model_type']}: IC={val_ic:.4f} <= {min_ic}"
                    )

            if issues:
                return CheckResult(
                    name="model_health",
                    status=CheckStatus.WARNING,
                    message=f"Model IC below threshold: {'; '.join(issues)}",
                    details={"active_models": len(resp.data), "issues": issues},
                )

            return CheckResult(
                name="model_health",
                status=CheckStatus.OK,
                message=f"{len(resp.data)} active model(s) healthy",
                details={"active_models": len(resp.data)},
            )
        except Exception as exc:
            return CheckResult(
                name="model_health",
                status=CheckStatus.CRITICAL,
                message=f"Failed to check model health: {exc}",
            )

    def check_prediction_coverage(self) -> CheckResult:
        """Verify predictions exist for all universe stocks."""
        try:
            # Get latest predictions
            preds_df = self.db.get_latest_predictions()
            if preds_df.empty:
                return CheckResult(
                    name="prediction_coverage",
                    status=CheckStatus.WARNING,
                    message="No predictions found",
                )

            # Get universe stock count
            resp = (
                self.db.client.table("stock_prices")
                .select("stock_id")
                .execute()
            )
            universe_ids = {r["stock_id"] for r in resp.data} if resp.data else set()
            predicted_ids = set(preds_df["stock_id"].unique())

            if not universe_ids:
                return CheckResult(
                    name="prediction_coverage",
                    status=CheckStatus.WARNING,
                    message="No stocks in universe to compare",
                    details={"predicted_count": len(predicted_ids)},
                )

            coverage = len(predicted_ids & universe_ids) / len(universe_ids)
            missing = universe_ids - predicted_ids

            if coverage < 0.8:
                return CheckResult(
                    name="prediction_coverage",
                    status=CheckStatus.WARNING,
                    message=f"Low prediction coverage: {coverage:.0%} ({len(missing)} stocks missing)",
                    details={"coverage": round(coverage, 4), "missing_count": len(missing)},
                )

            return CheckResult(
                name="prediction_coverage",
                status=CheckStatus.OK,
                message=f"Prediction coverage: {coverage:.0%}",
                details={"coverage": round(coverage, 4), "predicted_count": len(predicted_ids)},
            )
        except Exception as exc:
            return CheckResult(
                name="prediction_coverage",
                status=CheckStatus.CRITICAL,
                message=f"Failed to check prediction coverage: {exc}",
            )

    def check_feature_completeness(self) -> CheckResult:
        """Verify no excess NaN in features."""
        max_missing_rate = self.thresholds.get("max_missing_rate", 0.05)

        try:
            # Sample recent features
            today = date.today()
            start = str(today - timedelta(days=7))

            resp = (
                self.db.client.table("stock_features")
                .select("features")
                .gte("date", start)
                .limit(100)
                .execute()
            )
            if not resp.data:
                return CheckResult(
                    name="feature_completeness",
                    status=CheckStatus.WARNING,
                    message="No recent feature data found",
                )

            # Check for null values in features JSONB
            total_values = 0
            null_count = 0
            for row in resp.data:
                features = row.get("features", {})
                for val in features.values():
                    total_values += 1
                    if val is None:
                        null_count += 1

            if total_values == 0:
                return CheckResult(
                    name="feature_completeness",
                    status=CheckStatus.WARNING,
                    message="No feature values found in recent data",
                )

            missing_rate = null_count / total_values

            if missing_rate > max_missing_rate:
                return CheckResult(
                    name="feature_completeness",
                    status=CheckStatus.WARNING,
                    message=f"High missing rate: {missing_rate:.1%} (threshold: {max_missing_rate:.1%})",
                    details={"missing_rate": round(missing_rate, 4), "total_values": total_values},
                )

            return CheckResult(
                name="feature_completeness",
                status=CheckStatus.OK,
                message=f"Feature completeness: {1 - missing_rate:.1%}",
                details={"missing_rate": round(missing_rate, 4), "total_values": total_values},
            )
        except Exception as exc:
            return CheckResult(
                name="feature_completeness",
                status=CheckStatus.CRITICAL,
                message=f"Failed to check feature completeness: {exc}",
            )

    # ------------------------------------------------------------------ #
    #  Run all checks
    # ------------------------------------------------------------------ #

    def run_all_checks(self) -> HealthReport:
        """Run all enabled health checks and return aggregated report."""
        enabled = self.config.get("enabled_checks", [
            "data_freshness",
            "model_health",
            "prediction_coverage",
            "feature_completeness",
        ])

        check_map = {
            "data_freshness": self.check_data_freshness,
            "model_health": self.check_model_health,
            "prediction_coverage": self.check_prediction_coverage,
            "feature_completeness": self.check_feature_completeness,
        }

        results: list[CheckResult] = []
        for name in enabled:
            if name in check_map:
                logger.info("Running check: {}", name)
                result = check_map[name]()
                results.append(result)
                logger.info("  {} -> {}: {}", name, result.status.value, result.message)

        # Determine overall status
        statuses = [r.status for r in results]
        if CheckStatus.CRITICAL in statuses:
            overall = CheckStatus.CRITICAL
        elif CheckStatus.WARNING in statuses:
            overall = CheckStatus.WARNING
        else:
            overall = CheckStatus.OK

        return HealthReport(
            timestamp=datetime.now().isoformat(),
            overall_status=overall,
            checks=results,
        )

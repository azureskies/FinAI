"""Tests for pipeline instrumentation with monitoring system.

Verifies that scripts use PipelineTracker and AlertManager correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from monitoring.alerts import AlertLevel, AlertManager
from monitoring.tracker import PipelineTracker


# ------------------------------------------------------------------ #
#  PipelineTracker integration
# ------------------------------------------------------------------ #

class TestPipelineTrackerIntegration:
    """Verify PipelineTracker is used correctly in pipeline scripts."""

    @pytest.fixture()
    def mock_db(self):
        db = MagicMock()
        insert_resp = MagicMock()
        insert_resp.data = [{"id": "run-test-001"}]
        db.client.table.return_value.insert.return_value.execute.return_value = insert_resp
        return db

    def test_tracker_start_and_finish_success(self, mock_db):
        """Tracker should record start and successful finish."""
        tracker = PipelineTracker(mock_db)

        run_id = tracker.start_run("daily_update", metadata={"target_date": "2024-06-01"})
        assert len(run_id) == 36  # UUID format

        tracker.finish_run(run_id, status="success", metrics={"stocks": 20})

        # Verify update was called with correct status
        update_call = mock_db.client.table.return_value.update.call_args[0][0]
        assert update_call["status"] == "success"
        assert update_call["metrics"] == {"stocks": 20}

    def test_tracker_start_and_finish_failure(self, mock_db):
        """Tracker should record start and failed finish with error."""
        tracker = PipelineTracker(mock_db)

        run_id = tracker.start_run("weekly_retrain")
        tracker.finish_run(run_id, status="failed", error="Connection timeout")

        update_call = mock_db.client.table.return_value.update.call_args[0][0]
        assert update_call["status"] == "failed"
        assert update_call["error"] == "Connection timeout"


# ------------------------------------------------------------------ #
#  AlertManager integration
# ------------------------------------------------------------------ #

class TestAlertIntegration:
    """Verify alert triggers on success and failure."""

    def test_info_alert_on_success(self):
        """INFO alert should be sent on pipeline success."""
        config = {"alert_channels": {}}
        mgr = AlertManager(config=config)

        # No channels configured, but the method should still run without error
        result = mgr.send_alert(
            level=AlertLevel.INFO,
            title="Daily update completed",
            message="Stocks: 20, Predictions: 20",
        )
        # No channels, returns False
        assert result is False

    def test_critical_alert_on_failure(self):
        """CRITICAL alert should be sent on pipeline failure."""
        config = {"alert_channels": {}}
        mgr = AlertManager(config=config)

        result = mgr.send_alert(
            level=AlertLevel.CRITICAL,
            title="Daily update FAILED",
            message="Connection timeout",
        )
        assert result is False

    @patch("monitoring.alerts.smtplib.SMTP")
    def test_alert_sends_email_on_failure(self, mock_smtp):
        """Email alert should be sent when pipeline fails."""
        config = {
            "alert_channels": {
                "email": {
                    "enabled": True,
                    "type": "email",
                    "smtp_host": "smtp.test.com",
                    "smtp_port": 587,
                    "username": "test@test.com",
                    "password": "pass",
                    "from_addr": "test@test.com",
                    "to_addrs": ["admin@test.com"],
                },
            }
        }
        mgr = AlertManager(config=config)

        result = mgr.send_alert(
            level=AlertLevel.CRITICAL,
            title="Weekly retrain FAILED",
            message="No feature data found",
        )
        assert result is True
        mock_smtp.assert_called_once()

    @patch("monitoring.alerts.requests")
    def test_alert_sends_webhook_on_promotion(self, mock_requests):
        """Webhook alert should be sent when model is promoted."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        config = {
            "alert_channels": {
                "webhook": {
                    "enabled": True,
                    "type": "webhook",
                    "url": "https://hooks.test.com/test",
                },
            }
        }
        mgr = AlertManager(config=config)

        result = mgr.send_alert(
            level=AlertLevel.WARNING,
            title="Model promoted",
            message="ensemble -> v2 (ic=0.0523)",
            channel="webhook",
        )
        assert result is True


# ------------------------------------------------------------------ #
#  Script instrumentation patterns
# ------------------------------------------------------------------ #

class TestScriptInstrumentation:
    """Verify the instrumentation pattern used by pipeline scripts."""

    @pytest.fixture()
    def mock_db(self):
        db = MagicMock()
        insert_resp = MagicMock()
        insert_resp.data = [{"id": "run-001"}]
        db.client.table.return_value.insert.return_value.execute.return_value = insert_resp
        return db

    def test_daily_update_instrumentation_pattern(self, mock_db):
        """Simulate the daily_update instrumentation flow."""
        tracker = PipelineTracker(mock_db)
        alert_mgr = AlertManager(config={"alert_channels": {}})

        # Start
        run_id = tracker.start_run("daily_update", metadata={"target_date": "2024-06-01"})

        # Simulate success
        summary = {"stocks_processed": 18, "predictions": 18}
        tracker.finish_run(run_id, status="success", metrics=summary)
        alert_mgr.send_alert(
            AlertLevel.INFO, "Daily update completed", "Stocks: 18, Predictions: 18"
        )

        # Verify tracking calls
        assert mock_db.client.table.return_value.insert.called
        assert mock_db.client.table.return_value.update.called

    def test_daily_update_failure_pattern(self, mock_db):
        """Simulate the daily_update failure flow."""
        tracker = PipelineTracker(mock_db)
        alert_mgr = AlertManager(config={"alert_channels": {}})

        run_id = tracker.start_run("daily_update")

        # Simulate failure
        tracker.finish_run(run_id, status="failed", error="No price data fetched")
        alert_mgr.send_alert(
            AlertLevel.CRITICAL, "Daily update FAILED", "No price data fetched"
        )

        update_call = mock_db.client.table.return_value.update.call_args[0][0]
        assert update_call["status"] == "failed"
        assert "No price data" in update_call["error"]

    def test_weekly_retrain_promotion_pattern(self, mock_db):
        """Simulate the weekly_retrain promotion alert flow."""
        tracker = PipelineTracker(mock_db)
        alert_mgr = AlertManager(config={"alert_channels": {}})

        run_id = tracker.start_run("weekly_retrain", metadata={"optimize": True})

        # Simulate success with promotion
        metrics = {"models_trained": ["ridge", "xgboost"], "promotions": ["xgboost -> v2"]}
        tracker.finish_run(run_id, status="success", metrics=metrics)

        # Promotion alert (WARNING)
        alert_mgr.send_alert(
            AlertLevel.WARNING, "Model promoted", "xgboost -> v2 (ic=0.0523)"
        )
        # Completion alert (INFO)
        alert_mgr.send_alert(
            AlertLevel.INFO, "Weekly retrain completed", "Models: ridge, xgboost"
        )

        update_call = mock_db.client.table.return_value.update.call_args[0][0]
        assert update_call["status"] == "success"

    def test_monthly_report_instrumentation_pattern(self, mock_db):
        """Simulate the monthly_report instrumentation flow."""
        tracker = PipelineTracker(mock_db)
        alert_mgr = AlertManager(config={"alert_channels": {}})

        run_id = tracker.start_run("monthly_report", metadata={"period": "2024-05-01~2024-05-31"})

        # Simulate success
        report = {"mean_ic": 0.045, "excess_return": 0.02, "sharpe_ratio": 1.5}
        tracker.finish_run(run_id, status="success", metrics=report)
        alert_mgr.send_alert(
            AlertLevel.INFO,
            "Monthly report completed",
            "Return: 5.2% vs Benchmark: 3.2% (Excess: 2.0%)\nSharpe: 1.50, IC: 0.045",
        )

        update_call = mock_db.client.table.return_value.update.call_args[0][0]
        assert update_call["status"] == "success"
        assert update_call["metrics"]["mean_ic"] == 0.045

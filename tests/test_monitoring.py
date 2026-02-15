"""Tests for the monitoring and alerting system."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_db
from api.main import app
from monitoring.alerts import AlertLevel, AlertManager
from monitoring.health import CheckStatus, HealthChecker, HealthReport
from monitoring.tracker import PipelineTracker


# ------------------------------------------------------------------ #
#  HealthChecker
# ------------------------------------------------------------------ #

class TestHealthChecker:
    @pytest.fixture()
    def mock_db(self):
        db = MagicMock()
        return db

    @pytest.fixture()
    def checker(self, mock_db):
        config = {
            "thresholds": {
                "max_data_age_hours": 48,
                "min_ic": 0.0,
                "max_missing_rate": 0.05,
            },
            "enabled_checks": [
                "data_freshness",
                "model_health",
                "prediction_coverage",
                "feature_completeness",
            ],
        }
        return HealthChecker(db=mock_db, config=config)

    def test_check_data_freshness_ok(self, checker, mock_db):
        """Fresh data should return OK."""
        today = datetime.now().strftime("%Y-%m-%d")
        resp = MagicMock()
        resp.data = [{"date": today}]
        (
            mock_db.client.table.return_value
            .select.return_value
            .order.return_value
            .limit.return_value
            .execute.return_value
        ) = resp

        result = checker.check_data_freshness()
        assert result.status == CheckStatus.OK
        assert "up to date" in result.message

    def test_check_data_freshness_stale(self, checker, mock_db):
        """Stale data should return WARNING."""
        old_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        resp = MagicMock()
        resp.data = [{"date": old_date}]
        (
            mock_db.client.table.return_value
            .select.return_value
            .order.return_value
            .limit.return_value
            .execute.return_value
        ) = resp

        result = checker.check_data_freshness()
        assert result.status == CheckStatus.WARNING
        assert "old" in result.message

    def test_check_data_freshness_no_data(self, checker, mock_db):
        """No data should return CRITICAL."""
        resp = MagicMock()
        resp.data = []
        (
            mock_db.client.table.return_value
            .select.return_value
            .order.return_value
            .limit.return_value
            .execute.return_value
        ) = resp

        result = checker.check_data_freshness()
        assert result.status == CheckStatus.CRITICAL

    def test_check_model_health_ok(self, checker, mock_db):
        """Active model with IC > 0 should return OK."""
        resp = MagicMock()
        resp.data = [
            {
                "id": "m-001",
                "model_type": "ensemble",
                "metrics": {"val_ic": 0.05},
                "is_active": True,
                "created_at": datetime.now().isoformat(),
            }
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .eq.return_value
            .execute.return_value
        ) = resp

        result = checker.check_model_health()
        assert result.status == CheckStatus.OK

    def test_check_model_health_low_ic(self, checker, mock_db):
        """Active model with IC <= 0 should return WARNING."""
        resp = MagicMock()
        resp.data = [
            {
                "id": "m-001",
                "model_type": "ensemble",
                "metrics": {"val_ic": -0.01},
                "is_active": True,
                "created_at": datetime.now().isoformat(),
            }
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .eq.return_value
            .execute.return_value
        ) = resp

        result = checker.check_model_health()
        assert result.status == CheckStatus.WARNING

    def test_check_model_health_no_active(self, checker, mock_db):
        """No active model should return WARNING."""
        resp = MagicMock()
        resp.data = []
        (
            mock_db.client.table.return_value
            .select.return_value
            .eq.return_value
            .execute.return_value
        ) = resp

        result = checker.check_model_health()
        assert result.status == CheckStatus.WARNING

    def test_check_prediction_coverage_ok(self, checker, mock_db):
        """Full coverage should return OK."""
        mock_db.get_latest_predictions.return_value = pd.DataFrame([
            {"stock_id": "2330", "predicted_return": 0.05, "score": 0.8},
            {"stock_id": "2317", "predicted_return": 0.03, "score": 0.6},
        ])
        stock_resp = MagicMock()
        stock_resp.data = [{"stock_id": "2330"}, {"stock_id": "2317"}]
        (
            mock_db.client.table.return_value
            .select.return_value
            .execute.return_value
        ) = stock_resp

        result = checker.check_prediction_coverage()
        assert result.status == CheckStatus.OK

    def test_check_prediction_coverage_low(self, checker, mock_db):
        """Low coverage should return WARNING."""
        mock_db.get_latest_predictions.return_value = pd.DataFrame([
            {"stock_id": "2330", "predicted_return": 0.05, "score": 0.8},
        ])
        stock_resp = MagicMock()
        stock_resp.data = [
            {"stock_id": "2330"}, {"stock_id": "2317"},
            {"stock_id": "2454"}, {"stock_id": "2308"},
            {"stock_id": "2881"}, {"stock_id": "2882"},
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .execute.return_value
        ) = stock_resp

        result = checker.check_prediction_coverage()
        assert result.status == CheckStatus.WARNING

    def test_check_feature_completeness_ok(self, checker, mock_db):
        """Low missing rate should return OK."""
        resp = MagicMock()
        resp.data = [
            {"features": {"rsi_14": 55.0, "ma_20": 120.0}},
            {"features": {"rsi_14": 60.0, "ma_20": 121.0}},
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .gte.return_value
            .limit.return_value
            .execute.return_value
        ) = resp

        result = checker.check_feature_completeness()
        assert result.status == CheckStatus.OK

    def test_check_feature_completeness_high_missing(self, checker, mock_db):
        """High missing rate should return WARNING."""
        resp = MagicMock()
        resp.data = [
            {"features": {"rsi_14": None, "ma_20": None, "vol": None}},
            {"features": {"rsi_14": None, "ma_20": 121.0, "vol": None}},
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .gte.return_value
            .limit.return_value
            .execute.return_value
        ) = resp

        result = checker.check_feature_completeness()
        assert result.status == CheckStatus.WARNING

    def test_run_all_checks(self, checker, mock_db):
        """run_all_checks should return a HealthReport."""
        # Setup data_freshness -> OK
        today = datetime.now().strftime("%Y-%m-%d")
        freshness_resp = MagicMock()
        freshness_resp.data = [{"date": today}]

        # Setup model_health -> OK
        model_resp = MagicMock()
        model_resp.data = [
            {"id": "m-001", "model_type": "ensemble", "metrics": {"val_ic": 0.05},
             "is_active": True, "created_at": datetime.now().isoformat()}
        ]

        # Setup prediction_coverage -> OK
        mock_db.get_latest_predictions.return_value = pd.DataFrame([
            {"stock_id": "2330", "predicted_return": 0.05, "score": 0.8},
        ])
        stock_resp = MagicMock()
        stock_resp.data = [{"stock_id": "2330"}]

        # Setup feature_completeness -> OK
        feat_resp = MagicMock()
        feat_resp.data = [{"features": {"rsi_14": 55.0}}]

        def table_router(name):
            mock_table = MagicMock()
            if name == "stock_prices":
                # data_freshness uses order().limit()
                mock_table.select.return_value.order.return_value.limit.return_value.execute.return_value = freshness_resp
                # prediction_coverage uses select().execute()
                mock_table.select.return_value.execute.return_value = stock_resp
            elif name == "model_versions":
                mock_table.select.return_value.eq.return_value.execute.return_value = model_resp
            elif name == "stock_features":
                mock_table.select.return_value.gte.return_value.limit.return_value.execute.return_value = feat_resp
            return mock_table

        mock_db.client.table.side_effect = table_router

        report = checker.run_all_checks()
        assert isinstance(report, HealthReport)
        assert len(report.checks) == 4
        assert report.overall_status == CheckStatus.OK

    def test_report_to_dict(self):
        """HealthReport.to_dict should return serializable dict."""
        from monitoring.health import CheckResult

        report = HealthReport(
            timestamp="2024-01-01T00:00:00",
            overall_status=CheckStatus.OK,
            checks=[
                CheckResult(name="test", status=CheckStatus.OK, message="good"),
            ],
        )
        d = report.to_dict()
        assert d["overall_status"] == "ok"
        assert len(d["checks"]) == 1
        assert d["checks"][0]["name"] == "test"


# ------------------------------------------------------------------ #
#  AlertManager
# ------------------------------------------------------------------ #

class TestAlertManager:
    def test_alert_level_ordering(self):
        assert AlertLevel.INFO < AlertLevel.WARNING < AlertLevel.CRITICAL

    def test_send_alert_no_channels(self):
        """No channels configured returns False."""
        mgr = AlertManager(config={"alert_channels": {}})
        result = mgr.send_alert(AlertLevel.INFO, "test", "body")
        assert result is False

    def test_send_alert_disabled_channel(self):
        """Disabled channel should be skipped."""
        config = {
            "alert_channels": {
                "email": {"enabled": False, "type": "email"},
            }
        }
        mgr = AlertManager(config=config)
        result = mgr.send_alert(AlertLevel.WARNING, "test", "body", channel="email")
        assert result is False

    @patch("monitoring.alerts.smtplib.SMTP")
    def test_send_email_success(self, mock_smtp):
        """Email send should call SMTP correctly."""
        config = {
            "alert_channels": {
                "email": {
                    "enabled": True,
                    "type": "email",
                    "smtp_host": "smtp.test.com",
                    "smtp_port": 587,
                    "username": "user@test.com",
                    "password": "pass",
                    "from_addr": "user@test.com",
                    "to_addrs": ["admin@test.com"],
                },
            }
        }
        mgr = AlertManager(config=config)
        result = mgr.send_alert(
            AlertLevel.CRITICAL, "Test Alert", "Something broke", channel="email"
        )
        assert result is True
        mock_smtp.assert_called_once_with("smtp.test.com", 587)

    @patch("monitoring.alerts.requests")
    def test_send_webhook_success(self, mock_requests):
        """Webhook send should POST to URL."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        config = {
            "alert_channels": {
                "webhook": {
                    "enabled": True,
                    "type": "webhook",
                    "url": "https://hooks.example.com/test",
                },
            }
        }
        mgr = AlertManager(config=config)
        result = mgr.send_alert(
            AlertLevel.WARNING, "Test", "Webhook test", channel="webhook"
        )
        assert result is True
        mock_requests.post.assert_called_once()

    def test_send_alert_unknown_channel(self):
        """Unknown channel type should return False."""
        config = {
            "alert_channels": {
                "sms": {"enabled": True, "type": "sms"},
            }
        }
        mgr = AlertManager(config=config)
        result = mgr.send_alert(AlertLevel.INFO, "test", "body", channel="sms")
        assert result is False


# ------------------------------------------------------------------ #
#  PipelineTracker
# ------------------------------------------------------------------ #

class TestPipelineTracker:
    @pytest.fixture()
    def mock_db(self):
        return MagicMock()

    @pytest.fixture()
    def tracker(self, mock_db):
        return PipelineTracker(db=mock_db)

    def test_start_run(self, tracker, mock_db):
        """start_run should insert a record and return a UUID."""
        run_id = tracker.start_run("daily_update")
        # run_id is now generated internally as UUID
        assert len(run_id) == 36  # UUID format
        mock_db.client.table.assert_called_with("pipeline_runs")

    def test_finish_run(self, tracker, mock_db):
        """finish_run should update the record."""
        tracker.finish_run("run-001", status="success", metrics={"ic": 0.05})
        mock_db.client.table.assert_called_with("pipeline_runs")
        mock_db.client.table.return_value.update.assert_called_once()

    def test_finish_run_with_error(self, tracker, mock_db):
        """finish_run with error should include error field."""
        tracker.finish_run("run-001", status="failed", error="Connection timeout")
        call_args = mock_db.client.table.return_value.update.call_args[0][0]
        assert call_args["status"] == "failed"
        assert call_args["error"] == "Connection timeout"

    def test_get_recent_runs(self, tracker, mock_db):
        """get_recent_runs should query pipeline_runs table."""
        resp = MagicMock()
        resp.data = [
            {"id": "run-001", "pipeline_name": "daily_update", "status": "success"},
            {"id": "run-002", "pipeline_name": "daily_update", "status": "failed"},
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .eq.return_value
            .order.return_value
            .limit.return_value
            .execute.return_value
        ) = resp

        runs = tracker.get_recent_runs("daily_update", n=5)
        assert len(runs) == 2
        assert runs[0]["id"] == "run-001"


# ------------------------------------------------------------------ #
#  Monitoring API endpoints
# ------------------------------------------------------------------ #

class TestMonitoringAPI:
    @pytest.fixture()
    def mock_db(self):
        db = MagicMock()
        return db

    @pytest.fixture()
    def client(self, mock_db):
        app.dependency_overrides[get_db] = lambda: mock_db
        yield TestClient(app)
        app.dependency_overrides.clear()

    @pytest.fixture()
    def client_no_db(self):
        app.dependency_overrides[get_db] = lambda: None
        yield TestClient(app)
        app.dependency_overrides.clear()

    def test_health_no_db(self, client_no_db):
        resp = client_no_db.get("/api/monitoring/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_status"] == "unknown"
        assert "not configured" in data["message"]

    def test_health_with_db(self, client, mock_db):
        """Health endpoint should run checks and return report."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Setup all table queries
        freshness_resp = MagicMock()
        freshness_resp.data = [{"date": today}]

        model_resp = MagicMock()
        model_resp.data = [
            {"id": "m-001", "model_type": "ensemble", "metrics": {"val_ic": 0.05},
             "is_active": True, "created_at": datetime.now().isoformat()}
        ]

        mock_db.get_latest_predictions.return_value = pd.DataFrame([
            {"stock_id": "2330", "predicted_return": 0.05, "score": 0.8},
        ])
        stock_resp = MagicMock()
        stock_resp.data = [{"stock_id": "2330"}]

        feat_resp = MagicMock()
        feat_resp.data = [{"features": {"rsi_14": 55.0}}]

        def table_router(name):
            mt = MagicMock()
            if name == "stock_prices":
                mt.select.return_value.order.return_value.limit.return_value.execute.return_value = freshness_resp
                mt.select.return_value.execute.return_value = stock_resp
            elif name == "model_versions":
                mt.select.return_value.eq.return_value.execute.return_value = model_resp
            elif name == "stock_features":
                mt.select.return_value.gte.return_value.limit.return_value.execute.return_value = feat_resp
            return mt

        mock_db.client.table.side_effect = table_router

        resp = client.get("/api/monitoring/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_status"] in ("ok", "warning", "critical")
        assert len(data["checks"]) > 0

    def test_pipeline_runs_no_db(self, client_no_db):
        resp = client_no_db.get("/api/monitoring/pipeline-runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"] == []
        assert "not configured" in data["message"]

    def test_pipeline_runs_with_db(self, client, mock_db):
        """Pipeline runs endpoint should return run records."""
        resp_mock = MagicMock()
        resp_mock.data = [
            {
                "id": "run-001",
                "pipeline_name": "daily_update",
                "start_time": "2024-06-01T18:00:00",
                "end_time": "2024-06-01T18:05:00",
                "status": "success",
                "metrics": {"rows_updated": 100},
                "error": None,
            }
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .order.return_value
            .limit.return_value
            .execute.return_value
        ) = resp_mock

        resp = client.get("/api/monitoring/pipeline-runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["runs"]) == 1
        assert data["runs"][0]["pipeline_name"] == "daily_update"

    def test_pipeline_runs_with_filter(self, client, mock_db):
        """Pipeline runs should support pipeline_name filter."""
        resp_mock = MagicMock()
        resp_mock.data = []
        (
            mock_db.client.table.return_value
            .select.return_value
            .order.return_value
            .limit.return_value
            .eq.return_value
            .execute.return_value
        ) = resp_mock

        resp = client.get("/api/monitoring/pipeline-runs?pipeline_name=weekly_retrain")
        assert resp.status_code == 200

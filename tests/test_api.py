"""Tests for the FastAPI backend API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_db
from api.main import app


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture()
def mock_db():
    """Create a mock SupabaseLoader with pre-configured responses."""
    db = MagicMock()

    # stock_prices — select stock_id
    stock_resp = MagicMock()
    stock_resp.data = [
        {"stock_id": "2330.TW"},
        {"stock_id": "2317.TW"},
        {"stock_id": "2330.TW"},  # duplicate to test dedup
    ]
    db.client.table.return_value.select.return_value.execute.return_value = stock_resp

    # get_prices
    db.get_prices.return_value = pd.DataFrame([
        {
            "date": pd.Timestamp("2024-01-02"),
            "stock_id": "2330.TW",
            "open": 580.0, "high": 585.0, "low": 578.0,
            "close": 583.0, "volume": 30000000, "adj_close": 583.0,
        },
    ])

    # get_features
    db.get_features.return_value = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-02"), "stock_id": "2330.TW", "rsi_14": 55.2},
    ])

    # get_latest_predictions
    db.get_latest_predictions.return_value = pd.DataFrame([
        {
            "date": pd.Timestamp("2024-06-01"),
            "stock_id": "2330.TW",
            "predicted_return": 0.05,
            "score": 0.85,
            "model_version": "v1",
        },
        {
            "date": pd.Timestamp("2024-06-01"),
            "stock_id": "2317.TW",
            "predicted_return": 0.03,
            "score": 0.72,
            "model_version": "v1",
        },
    ])

    # get_backtest_history
    db.get_backtest_history.return_value = [
        {
            "id": "bt-001",
            "run_date": "2024-06-01",
            "model_type": "ensemble",
            "period_start": "2023-01-01",
            "period_end": "2024-01-01",
            "metrics": {"sharpe": 1.2},
            "config": {},
        }
    ]

    return db


@pytest.fixture()
def client(mock_db):
    """TestClient with mocked database dependency."""
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def client_no_db():
    """TestClient with no database (simulates Supabase not configured)."""
    app.dependency_overrides[get_db] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


# ------------------------------------------------------------------ #
#  Health check
# ------------------------------------------------------------------ #

class TestHealthCheck:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ------------------------------------------------------------------ #
#  Stocks
# ------------------------------------------------------------------ #

class TestStocks:
    def test_list_stocks(self, client):
        resp = client.get("/api/stocks")
        assert resp.status_code == 200
        data = resp.json()
        ids = [s["stock_id"] for s in data["stocks"]]
        assert "2330.TW" in ids
        assert "2317.TW" in ids
        # Should be deduplicated
        assert len(ids) == 2

    def test_list_stocks_no_db(self, client_no_db):
        resp = client_no_db.get("/api/stocks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stocks"] == []
        assert "not configured" in data["message"]

    def test_get_prices(self, client):
        resp = client.get("/api/stocks/2330.TW/prices")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["prices"]) == 1
        assert data["prices"][0]["close"] == 583.0

    def test_get_prices_no_db(self, client_no_db):
        resp = client_no_db.get("/api/stocks/2330.TW/prices")
        assert resp.status_code == 200
        assert resp.json()["prices"] == []

    def test_get_features(self, client):
        resp = client.get("/api/stocks/2330.TW/features")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["features"]) == 1
        assert data["features"][0]["features"]["rsi_14"] == 55.2

    def test_get_predictions(self, client):
        resp = client.get("/api/stocks/2330.TW/predictions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["score"] == 0.85

    def test_get_predictions_not_found(self, client):
        resp = client.get("/api/stocks/9999.TW/predictions")
        assert resp.status_code == 200
        assert resp.json()["predictions"] == []


# ------------------------------------------------------------------ #
#  Backtest
# ------------------------------------------------------------------ #

class TestBacktest:
    def test_list_results(self, client):
        resp = client.get("/api/backtest/results")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "bt-001"

    def test_list_results_no_db(self, client_no_db):
        resp = client_no_db.get("/api/backtest/results")
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_get_result_by_id(self, client, mock_db):
        detail_resp = MagicMock()
        detail_resp.data = [
            {
                "id": "bt-001",
                "run_date": "2024-06-01",
                "model_type": "ensemble",
                "period_start": "2023-01-01",
                "period_end": "2024-01-01",
                "metrics": {"sharpe": 1.2},
                "config": {},
            }
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .eq.return_value
            .execute.return_value
        ) = detail_resp

        resp = client.get("/api/backtest/bt-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "bt-001"

    def test_get_result_not_found(self, client, mock_db):
        empty_resp = MagicMock()
        empty_resp.data = []
        (
            mock_db.client.table.return_value
            .select.return_value
            .eq.return_value
            .execute.return_value
        ) = empty_resp

        resp = client.get("/api/backtest/nonexistent")
        assert resp.status_code == 404

    def test_run_backtest(self, client):
        resp = client.post(
            "/api/backtest/run",
            json={"model_type": "ensemble", "period_start": "2023-01-01"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_run_backtest_walk_forward(self, client):
        resp = client.post(
            "/api/backtest/run",
            json={
                "model_type": "ensemble",
                "mode": "walk_forward",
                "period_start": "2023-01-01",
                "period_end": "2024-01-01",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert "ensemble" in data["message"]

    def test_run_backtest_with_capital(self, client):
        resp = client.post(
            "/api/backtest/run",
            json={
                "model_type": "ridge",
                "initial_capital": 5_000_000,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_run_backtest_no_db(self, client_no_db):
        resp = client_no_db.post(
            "/api/backtest/run", json={"model_type": "ensemble"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_run_backtest_invalid_mode(self, client):
        resp = client.post(
            "/api/backtest/run",
            json={"model_type": "ensemble", "mode": "invalid"},
        )
        assert resp.status_code == 422

    @patch("api.routers.backtest._execute_backtest")
    def test_execute_backtest_called(self, mock_exec, client):
        resp = client.post(
            "/api/backtest/run",
            json={"model_type": "xgboost", "mode": "run"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"
        # Background task was scheduled (TestClient runs them synchronously)
        mock_exec.assert_called_once()


# ------------------------------------------------------------------ #
#  Models
# ------------------------------------------------------------------ #

class TestModels:
    def test_list_models(self, client, mock_db):
        model_resp = MagicMock()
        model_resp.data = [
            {
                "id": "m-001",
                "model_type": "ensemble",
                "metrics": {"rmse": 0.02},
                "is_active": True,
            }
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .order.return_value
            .execute.return_value
        ) = model_resp

        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1

    def test_list_models_no_db(self, client_no_db):
        resp = client_no_db.get("/api/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == []

    def test_active_models(self, client, mock_db):
        active_resp = MagicMock()
        active_resp.data = [
            {"id": "m-001", "model_type": "ensemble", "is_active": True}
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .eq.return_value
            .execute.return_value
        ) = active_resp

        resp = client.get("/api/models/active")
        assert resp.status_code == 200
        assert len(resp.json()["models"]) == 1

    def test_model_metrics(self, client, mock_db):
        metrics_resp = MagicMock()
        metrics_resp.data = [
            {"id": "m-001", "model_type": "ensemble", "metrics": {"rmse": 0.02}}
        ]
        (
            mock_db.client.table.return_value
            .select.return_value
            .eq.return_value
            .execute.return_value
        ) = metrics_resp

        resp = client.get("/api/models/m-001/metrics")
        assert resp.status_code == 200
        assert resp.json()["metrics"]["rmse"] == 0.02

    def test_model_metrics_not_found(self, client, mock_db):
        empty_resp = MagicMock()
        empty_resp.data = []
        (
            mock_db.client.table.return_value
            .select.return_value
            .eq.return_value
            .execute.return_value
        ) = empty_resp

        resp = client.get("/api/models/nonexistent/metrics")
        assert resp.status_code == 404


# ------------------------------------------------------------------ #
#  Dashboard
# ------------------------------------------------------------------ #

class TestDashboard:
    def test_summary(self, client, mock_db):
        # Setup chained mock for stock_prices select
        price_resp = MagicMock()
        price_resp.data = [
            {"stock_id": "2330.TW"},
            {"stock_id": "2317.TW"},
        ]

        active_resp = MagicMock()
        active_resp.data = [{"id": "m-001"}]

        bt_resp = MagicMock()
        bt_resp.data = [{"id": "bt-001"}, {"id": "bt-002"}]

        # Configure table routing
        def table_router(name):
            mock_table = MagicMock()
            if name == "stock_prices":
                mock_table.select.return_value.execute.return_value = price_resp
            elif name == "model_versions":
                mock_table.select.return_value.eq.return_value.execute.return_value = active_resp
            elif name == "backtest_results":
                mock_table.select.return_value.execute.return_value = bt_resp
            return mock_table

        mock_db.client.table.side_effect = table_router

        resp = client.get("/api/dashboard/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stocks_count"] == 2
        assert data["predictions_count"] == 2
        assert data["active_models"] == 1
        assert data["backtest_runs"] == 2

    def test_summary_no_db(self, client_no_db):
        resp = client_no_db.get("/api/dashboard/summary")
        assert resp.status_code == 200
        assert resp.json()["stocks_count"] == 0
        assert "not configured" in resp.json()["message"]

    def test_top_picks(self, client):
        resp = client.get("/api/dashboard/top-picks?n=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["picks"]) == 2
        # Should be sorted by score desc (from get_latest_predictions)
        assert data["picks"][0]["stock_id"] == "2330.TW"
        assert data["picks"][0]["score"] == 0.85

    def test_top_picks_no_db(self, client_no_db):
        resp = client_no_db.get("/api/dashboard/top-picks")
        assert resp.status_code == 200
        assert resp.json()["picks"] == []

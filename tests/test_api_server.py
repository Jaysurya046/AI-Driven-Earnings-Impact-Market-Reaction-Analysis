"""Tests for API server state, concurrency, security, and endpoints."""
import asyncio
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api_server import app, state_manager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    # Reset global state before each test
    asyncio.run(state_manager.reset())
    yield
    asyncio.run(state_manager.reset())


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["analysis_loaded"] is False
    assert data["is_analyzing"] is False


def test_endpoints_require_analysis(client):
    for endpoint in ["/sector-analysis", "/backtest", "/predictions/AAPL", "/data/events", "/metrics"]:
        res = client.get(endpoint)
        assert res.status_code == 400, f"Endpoint {endpoint} should return 400 when analysis not loaded"


def test_concurrency_lock_rejects_simultaneous_runs(client):
    # Simulate an ongoing analysis
    async def _lock_state():
        async with state_manager._lock:
            state_manager.is_analyzing = True
            state_manager.current_task_id = "task-1234"

    asyncio.run(_lock_state())

    response = client.post("/analyze", json={"tickers": ["AAPL"]})
    assert response.status_code == 409
    assert "already running" in response.json()["detail"]


def test_prediction_with_populated_state(client, event_analysis_df):
    from src.feature_engineering import FeatureEngineer
    from src.ml_models import ModelTrainer

    features = FeatureEngineer.create_features(event_analysis_df)
    cols = FeatureEngineer.get_feature_columns(features)

    # Train a model with balanced classes
    n_feats = len(cols)
    X_tr = np.tile(np.linspace(1, 10, n_feats), (4, 1))
    y_tr = np.array([0, 0, 1, 1])
    X_te = np.tile(np.linspace(2, 8, n_feats), (2, 1))
    y_te = np.array([0, 1])

    trainer = ModelTrainer()
    trainer.train_all_models(X_tr, X_te, y_tr, y_te)
    for m in trainer.models:
        trainer.feature_columns[m] = cols

    async def _populate():
        async with state_manager._lock:
            state_manager.event_analysis = event_analysis_df
            state_manager.features = features
            state_manager.model_trainer = trainer
            state_manager.model_comparison = pd.DataFrame([
                {'Model': 'Logistic Regression', 'F1-Score': 0.8}
            ])
            state_manager.pipeline = object()

    asyncio.run(_populate())

    response = client.get("/predictions/AAA")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAA"
    assert "prediction" in data

    events_resp = client.get("/data/events")
    assert events_resp.status_code == 200
    assert events_resp.json()["count"] > 0

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert metrics_resp.json()["total_events"] > 0

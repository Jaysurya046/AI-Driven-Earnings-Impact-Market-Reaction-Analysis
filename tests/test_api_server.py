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


def test_enforce_single_worker_environment(monkeypatch):
    """Verify single-worker enforcement guard catches multi-worker configurations"""
    import sys
    from api_server import enforce_single_worker_environment

    # Safe single worker
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.setattr(sys, "argv", ["api_server.py"])
    enforce_single_worker_environment()  # Should not raise

    # Disallowed via WEB_CONCURRENCY
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with pytest.raises(RuntimeError, match="requires single-worker execution"):
        enforce_single_worker_environment()

    # Disallowed via CLI flag
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.setattr(sys, "argv", ["api_server.py", "--workers", "2"])
    with pytest.raises(RuntimeError, match="requires single-worker execution"):
        enforce_single_worker_environment()


def test_pipeline_run_for_api(monkeypatch):
    """Verify EarningsAnalysisPipeline.run_for_api provides unified single orchestration entry point"""
    from main import EarningsAnalysisPipeline

    pipeline = EarningsAnalysisPipeline({
        'lookback_years': 1,
        'event_window': 3,
        'save_intermediate': False,
        'models_to_train': ['logistic_regression']
    })

    # Mock data collection and balanced event analysis
    sample_prices = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100),
        'Open': 100.0, 'High': 105.0, 'Low': 95.0, 'Close': 102.0, 'Adj Close': 102.0,
        'Volume': 1000000, 'Ticker': 'AAA'
    })
    sample_earnings = pd.DataFrame({
        'Date': [pd.Timestamp('2023-02-15')],
        'EPS': [1.5],
        'Surprise_Percentage': [4.0],
        'Ticker': ['AAA']
    })

    sample_events = pd.DataFrame([
        {
            'Ticker': f'T{i}',
            'EarningsDate': pd.Timestamp(f'2023-{(i % 6) + 1:02d}-15'),
            'SurprisePercentage': 2.0 if i % 2 == 0 else -2.0,
            'PostEventReturn': 3.0 if i % 2 == 0 else -3.0,
            'PreEventReturn': 1.0,
            'AbnormalReturnCAR': 2.0 if i % 2 == 0 else -2.0,
            'PreEventVolatility': 1.5,
            'PostEventVolatility': 2.0,
            'EstimationVolatility': 1.2,
            'VolumeSpikeRatio': 1.8,
            'PreEventVolume': 1000000,
            'PostEventVolume': 1800000,
            'EstimationAvgVolume': 1000000,
            'PreEventAvgVolume': 1000000,
            'DataPointsInWindow': 11,
            'Sector': 'Technology'
        }
        for i in range(20)
    ])

    monkeypatch.setattr(pipeline, "step_1_collect_data", lambda tickers: (sample_prices, sample_earnings))
    monkeypatch.setattr(pipeline, "step_2_event_analysis", lambda p, e: sample_events)

    results = pipeline.run_for_api(["AAA"])
    assert 'pipeline' in results
    assert 'price_data' in results
    assert 'event_analysis' in results
    assert 'features' in results
    assert 'model_trainer' in results
    assert 'model_comparison' in results
    assert 'report' in results


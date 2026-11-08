"""
Tests for LSTM Networks and Hybrid Ensemble Predictor
Verifies availability checks, lazy imports, and regression fixes in prediction pipelines.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.lstm_models import LSTMPredictor, HybridEnsemblePredictor


def test_lstm_predictor_is_available():
    """Verify is_available returns boolean without unhandled exceptions"""
    available = LSTMPredictor.is_available()
    assert isinstance(available, bool)


def test_lstm_uninstalled_raises_runtime_error(monkeypatch):
    """Verify LSTMPredictor raises clear error when TensorFlow is requested but unavailable"""
    predictor = LSTMPredictor()
    monkeypatch.setattr(predictor, "_check_tensorflow", MagicMock(side_effect=RuntimeError("TensorFlow is required")))

    with pytest.raises(RuntimeError, match="TensorFlow is required"):
        predictor.build_model((10, 5), n_features=5)


def test_hybrid_ensemble_scalers_regression():
    """
    Regression test for HybridEnsemblePredictor.predict bug where
    getattr(model_name, 'scalers', ...) was called on a string.
    """
    ensemble = HybridEnsemblePredictor()

    # Create dummy trained state
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8]])

    scaler = StandardScaler()
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler.fit(X_train)

    mock_lstm = MagicMock()
    mock_lstm.predict_proba.return_value = np.array([0.65, 0.75])

    ensemble.traditional_models = {"Random Forest": mock_model}
    ensemble.scalers = {"Random Forest": scaler}
    ensemble.lstm_predictor = mock_lstm
    ensemble.weights = {"lstm": 0.5, "traditional": 0.5}

    X_test = np.array([[2.0, 3.0], [4.0, 5.0]])

    # Call predict — must not raise AttributeError on string 'Random Forest'
    preds = ensemble.predict(X_test)
    assert preds is not None
    assert len(preds) == 2
    # Verify weighted ensemble: 0.5 * 0.65 + 0.5 * 0.7 = 0.675
    assert preds[0] == pytest.approx(0.675, abs=1e-3)


def test_hybrid_ensemble_uninitialized():
    """Verify predict returns None when ensemble has not been trained yet"""
    ensemble = HybridEnsemblePredictor()
    assert ensemble.predict(np.array([[1.0, 2.0]])) is None


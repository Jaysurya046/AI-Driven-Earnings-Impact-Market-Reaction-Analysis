"""Tests for cross-validation with confidence intervals and unscaled tree models."""
import numpy as np
import pytest

from src.config import load_config
from src.ml_models import ModelTrainer


def test_xgboost_scaling_disabled_by_default():
    trainer = ModelTrainer()
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = np.array([0, 0, 1, 1])

    model, scaler = trainer.train_xgboost(X, y)
    assert scaler is None  # Tree models must NOT scale features by default


def test_cross_validation_with_confidence_intervals():
    trainer = ModelTrainer()
    # 20 samples with 2 classes
    np.random.seed(42)
    X = np.random.randn(20, 4)
    y = np.array([0, 1] * 10)

    from sklearn.linear_model import LogisticRegression
    factory = lambda: LogisticRegression(max_iter=200, random_state=42)

    cv_res = trainer.evaluate_cv(factory, X, y, cv=3, scale=True, model_name="LR_Test", time_series=False)

    assert cv_res['Model'] == "LR_Test"
    assert cv_res['CV_Folds'] == 3
    assert 0.0 <= cv_res['Accuracy_Mean'] <= 1.0
    assert isinstance(cv_res['Accuracy_CI95'], tuple)
    assert len(cv_res['Accuracy_CI95']) == 2
    ci_low, ci_high = cv_res['Accuracy_CI95']
    assert 0.0 <= ci_low <= ci_high <= 1.0


def test_cross_validate_all_models():
    trainer = ModelTrainer()
    np.random.seed(42)
    X = np.random.randn(24, 3)
    y = np.array([0, 1] * 12)

    cv_df = trainer.cross_validate_all_models(X, y, cv=3, time_series=False)
    assert len(cv_df) == 3
    models_evaluated = cv_df['Model'].tolist()
    assert 'Logistic Regression' in models_evaluated
    assert 'Random Forest' in models_evaluated
    assert 'XGBoost' in models_evaluated
    assert 'Accuracy_Mean' in cv_df.columns
    assert 'F1_CI95' in cv_df.columns


def test_expanded_universe_tickers():
    cfg = load_config()
    tickers = cfg.get('tickers')
    assert len(tickers) >= 20
    assert 'AAPL' in tickers
    assert 'UNH' in tickers
    assert 'XOM' in tickers


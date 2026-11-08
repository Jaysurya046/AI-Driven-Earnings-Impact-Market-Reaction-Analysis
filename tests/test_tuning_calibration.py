"""Tests for CV hyperparameter tuning and probability calibration."""
import numpy as np
import pandas as pd
import pytest

from src.ml_models import ModelTrainer


def test_tune_hyperparameters():
    trainer = ModelTrainer()
    np.random.seed(42)
    X = np.random.randn(30, 4)
    y = np.array([0, 1] * 15)

    best_lr, params_lr = trainer.tune_hyperparameters(
        "Logistic Regression",
        X,
        y,
        cv=3,
        param_grid={'C': [0.1, 1.0]},
        time_series=False
    )
    assert best_lr is not None
    assert 'C' in params_lr
    assert params_lr['C'] in [0.1, 1.0]

    best_rf, params_rf = trainer.tune_hyperparameters(
        "Random Forest",
        X,
        y,
        cv=2,
        param_grid={'n_estimators': [10, 20]},
        time_series=False
    )
    assert best_rf is not None
    assert 'n_estimators' in params_rf


def test_calibrate_probabilities_and_predict():
    trainer = ModelTrainer()
    np.random.seed(42)
    # 40 train, 20 val, 10 test
    X_train = np.random.randn(40, 3)
    y_train = np.array([0, 1] * 20)
    X_val = np.random.randn(20, 3)
    y_val = np.array([0, 1] * 10)
    X_test = np.random.randn(10, 3)

    # Train base Random Forest
    rf = trainer.train_random_forest(X_train, y_train, n_estimators=20)
    trainer.models['Random Forest'] = rf
    trainer.scalers['Random Forest'] = None
    trainer.feature_columns['Random Forest'] = ['f1', 'f2', 'f3']

    # Calibrate probabilities
    calibrator = trainer.calibrate_probabilities('Random Forest', X_val, y_val, method='sigmoid')
    assert calibrator is not None

    # Predict calibrated probabilities on array
    probs_arr = trainer.predict_proba_calibrated('Random Forest', X_test)
    assert len(probs_arr) == 10
    assert (probs_arr >= 0.0).all() and (probs_arr <= 1.0).all()

    # Predict calibrated probabilities on DataFrame
    df_test = pd.DataFrame(X_test, columns=['f1', 'f2', 'f3'])
    probs_df = trainer.predict_proba_calibrated('Random Forest', df_test)
    assert np.allclose(probs_arr, probs_df)

"""Tests for unified config loading merging config.ini and environment variables."""
import os
from pathlib import Path
import pytest

from src.config import load_config, DEFAULT_CONFIG


def test_load_config_from_ini(tmp_path):
    ini_content = """
[DATA_COLLECTION]
lookback_years = 3
tickers = TSLA, NVDA

[EVENT_ANALYSIS]
event_window = 7
estimation_window = 90

[FEATURE_ENGINEERING]
label_threshold = 1.5

[MACHINE_LEARNING]
test_size = 0.25
random_state = 123
models_to_train = random_forest, xgboost
"""
    ini_file = tmp_path / "test_config.ini"
    ini_file.write_text(ini_content)

    cfg = load_config(config_file=ini_file)
    assert cfg['lookback_years'] == 3
    assert cfg['tickers'] == ['TSLA', 'NVDA']
    assert cfg['event_window'] == 7
    assert cfg['estimation_window'] == 90
    assert cfg['label_threshold'] == 1.5
    assert cfg['test_size'] == 0.25
    assert cfg['random_state'] == 123
    assert cfg['models_to_train'] == ['random_forest', 'xgboost']


def test_env_var_overrides_ini(tmp_path, monkeypatch):
    ini_content = """
[DATA_COLLECTION]
lookback_years = 2
tickers = AAPL
"""
    ini_file = tmp_path / "test_config.ini"
    ini_file.write_text(ini_content)

    monkeypatch.setenv("LOOKBACK_YEARS", "10")
    monkeypatch.setenv("TICKERS", "MSFT, GOOGL, AMZN")
    monkeypatch.setenv("EVENT_WINDOW", "3")

    cfg = load_config(config_file=ini_file)
    assert cfg['lookback_years'] == 10
    assert cfg['tickers'] == ['MSFT', 'GOOGL', 'AMZN']
    assert cfg['event_window'] == 3


def test_fallback_to_defaults(tmp_path):
    non_existent = tmp_path / "does_not_exist.ini"
    cfg = load_config(config_file=non_existent)
    assert cfg['lookback_years'] == DEFAULT_CONFIG['lookback_years']
    assert cfg['event_window'] == DEFAULT_CONFIG['event_window']
    assert cfg.get('non_existent_key', 42) == 42


def test_sensitive_keys_masked_in_repr():
    cfg = load_config()
    cfg['api_secret_key'] = 'super-secret-token'
    rep = repr(cfg)
    assert 'super-secret-token' not in rep
    assert '***' in rep


def test_models_to_train_configurable():
    """Verify ModelTrainer.train_all_models honors models_to_train filtering"""
    from src.ml_models import ModelTrainer
    import numpy as np

    trainer = ModelTrainer()
    X_train = np.random.randn(30, 5)
    y_train = np.random.choice([0, 1], size=30)
    X_test = np.random.randn(10, 5)
    y_test = np.random.choice([0, 1], size=10)

    # Train only random_forest
    res_rf = trainer.train_all_models(X_train, X_test, y_train, y_test, models_to_train=['random_forest'])
    assert len(res_rf) == 1
    assert res_rf.iloc[0]['Model'] == 'Random Forest'
    assert 'Random Forest' in trainer.models
    assert 'Logistic Regression' not in trainer.models

    # Train logistic_regression and xgboost
    trainer2 = ModelTrainer()
    res_two = trainer2.train_all_models(
        X_train, X_test, y_train, y_test,
        models_to_train=['logistic_regression', 'xgboost']
    )
    assert len(res_two) == 2
    models_trained = set(res_two['Model'].tolist())
    assert models_trained == {'Logistic Regression', 'XGBoost'}


def test_retraining_scheduler_consumes_app_config(tmp_path):
    """Verify RetrainingScheduler consumes canonical AppConfig and scopes JSON to cadence"""
    from retraining_scheduler import RetrainingScheduler
    from src.config import AppConfig

    custom_app_cfg = AppConfig(
        tickers=['NVDA', 'AMD'],
        lookback_years=3,
        event_window=4
    )
    scheduler_json = tmp_path / "scheduler_cadence.json"
    scheduler = RetrainingScheduler(config_path=str(scheduler_json), app_config=custom_app_cfg)

    assert scheduler.app_config.tickers == ['NVDA', 'AMD']
    assert scheduler.app_config.lookback_years == 3
    assert scheduler.app_config.event_window == 4
    # Scheduler JSON should only contain schedule, triggers, notification
    assert "schedule" in scheduler.config
    assert "retraining_triggers" in scheduler.config
    assert "pipeline_config" not in scheduler.config



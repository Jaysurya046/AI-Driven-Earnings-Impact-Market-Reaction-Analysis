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

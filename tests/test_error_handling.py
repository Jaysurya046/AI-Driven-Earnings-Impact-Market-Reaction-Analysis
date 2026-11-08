"""Tests for data shape validation and specific error handling."""
import numpy as np
import pandas as pd
import pytest

from src.data_collection import DataCollector
from src.utils import create_performance_report


def test_validate_collected_data_success(price_data, earnings_data):
    # Should complete without error on valid data
    DataCollector.validate_collected_data(price_data, earnings_data)


def test_validate_collected_data_empty_price_raises():
    empty_price = pd.DataFrame()
    earnings = pd.DataFrame({'Date': [pd.Timestamp("2024-01-01")], 'Ticker': ['AAA']})
    with pytest.raises(ValueError, match="Price data collection failed: DataFrame is empty"):
        DataCollector.validate_collected_data(empty_price, earnings)


def test_validate_collected_data_empty_earnings_raises(price_data):
    empty_earnings = pd.DataFrame()
    with pytest.raises(ValueError, match="Earnings data collection failed: DataFrame is empty"):
        DataCollector.validate_collected_data(price_data, empty_earnings)


def test_validate_collected_data_missing_columns_raises(price_data, earnings_data):
    bad_price = price_data.drop(columns=['Adj Close'])
    with pytest.raises(ValueError, match="Price DataFrame is missing required columns"):
        DataCollector.validate_collected_data(bad_price, earnings_data)

    bad_earnings = earnings_data.drop(columns=['Ticker'])
    with pytest.raises(ValueError, match="Earnings DataFrame is missing required columns"):
        DataCollector.validate_collected_data(price_data, bad_earnings)


def test_validate_collected_data_no_ticker_overlap_raises(price_data):
    diff_earnings = pd.DataFrame({
        'Date': [pd.Timestamp("2024-01-01")],
        'Ticker': ['NON_EXISTENT']
    })
    with pytest.raises(ValueError, match="No ticker overlap"):
        DataCollector.validate_collected_data(price_data, diff_earnings)


def test_performance_report_handles_invalid_auc_gracefully():
    # When single class present, roc_auc_score throws ValueError
    actual = np.array([1, 1, 1, 1])
    pred = np.array([1, 1, 1, 1])
    probs = np.array([0.9, 0.8, 0.95, 0.85])

    report = create_performance_report(actual, pred, probabilities=probs)
    assert 'accuracy' in report
    assert report['accuracy'] == 1.0
    # auc_roc should be omitted or handled cleanly without raising an unhandled exception
    assert 'auc_roc' not in report or report['auc_roc'] is not None

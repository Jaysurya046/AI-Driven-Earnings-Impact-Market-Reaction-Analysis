"""Tests for utility functions and pandas 2.1+ modern APIs."""
import numpy as np
import pandas as pd
import pytest

from src.utils import (
    handle_missing_values,
    filter_by_date_range,
    calculate_rolling_metrics,
    normalize_features,
    detect_outliers,
)


def test_handle_missing_values_forward_fill():
    df = pd.DataFrame({'a': [1.0, np.nan, np.nan, 4.0]})
    filled = handle_missing_values(df, strategy='forward_fill')
    assert filled['a'].tolist() == [1.0, 1.0, 1.0, 4.0]


def test_handle_missing_values_backward_fill():
    df = pd.DataFrame({'a': [1.0, np.nan, np.nan, 4.0]})
    filled = handle_missing_values(df, strategy='backward_fill')
    assert filled['a'].tolist() == [1.0, 4.0, 4.0, 4.0]


def test_handle_missing_values_mean_and_drop():
    df = pd.DataFrame({'a': [2.0, np.nan, 4.0]})
    filled_mean = handle_missing_values(df, strategy='mean')
    assert filled_mean['a'].iloc[1] == 3.0

    dropped = handle_missing_values(df, strategy='drop')
    assert len(dropped) == 2


def test_normalize_features():
    X = np.array([[10.0, 0.0], [20.0, 50.0], [30.0, 100.0]])
    norm = normalize_features(X)
    assert norm.min() == pytest.approx(0.0, abs=1e-6)
    assert norm.max() == pytest.approx(1.0, abs=1e-6)
    assert norm[1, 0] == pytest.approx(0.5, abs=1e-6)


def test_detect_outliers():
    df = pd.DataFrame({'val': [10.0, 10.2, 9.8, 10.1, 100.0]})
    outliers = detect_outliers(df, columns=['val'], method='iqr')
    assert outliers.iloc[-1] is True or outliers.iloc[-1] == True
    assert outliers.iloc[0] is False or outliers.iloc[0] == False


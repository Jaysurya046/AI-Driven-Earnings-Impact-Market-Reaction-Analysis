"""
Tests for shared chart data preparation functions in src/chart_data.py
Verifies statistical computations, data reshaping, and surface decoupling.
"""

import numpy as np
import pandas as pd
import pytest

from src.chart_data import (
    prepare_surprise_vs_reaction_data,
    prepare_model_comparison_data,
    prepare_volatility_analysis_data,
    prepare_volume_analysis_data,
    prepare_feature_importance_data,
)


@pytest.fixture
def sample_event_analysis():
    return pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        'SurprisePercentage': [5.0, -3.0, 2.5, -4.0],
        'PostEventReturn': [3.0, -2.0, 1.5, -3.5],
        'PreEventVolatility': [1.2, 1.5, 1.1, 1.8],
        'PostEventVolatility': [2.5, 2.2, 2.0, 2.8],
        'EstimationVolatility': [1.0, 1.2, 1.0, 1.4],
        'PreEventVolume': [1000000, 2000000, 1500000, 3000000],
        'PostEventVolume': [2500000, 3000000, 2000000, 6000000],
        'VolumeSpikeRatio': [2.5, 1.5, 1.33, 2.0],
    })


def test_prepare_surprise_vs_reaction_data(sample_event_analysis):
    data = prepare_surprise_vs_reaction_data(sample_event_analysis)

    assert data['has_trend'] is True
    assert len(data['surprises']) == 4
    assert len(data['returns']) == 4
    assert len(data['colors']) == 4
    # Green for positive return, red for negative
    assert data['colors'] == ['green', 'red', 'green', 'red']
    assert data['slope'] > 0  # Positive surprise correlates with positive return
    assert len(data['trend_x']) == 100
    assert len(data['trend_y']) == 100


def test_prepare_surprise_vs_reaction_empty():
    empty_res = prepare_surprise_vs_reaction_data(pd.DataFrame())
    assert empty_res['has_trend'] is False
    assert len(empty_res['surprises']) == 0


def test_prepare_model_comparison_data():
    df = pd.DataFrame([
        {'Model': 'Logistic Regression', 'Accuracy': 0.70, 'Precision': 0.68, 'Recall': 0.72, 'F1-Score': 0.70, 'AUC-ROC': 0.75},
        {'Model': 'Random Forest', 'Accuracy': 0.80, 'Precision': 0.78, 'Recall': 0.82, 'F1-Score': 0.80, 'AUC-ROC': 0.85},
    ])

    data = prepare_model_comparison_data(df)
    assert data['models'] == ['Logistic Regression', 'Random Forest']
    assert 'F1-Score' in data['metrics']
    assert data['best_model']['Model'] == 'Random Forest'
    assert len(data['melted_df']) == 10  # 2 models * 5 metrics


def test_prepare_volatility_analysis_data(sample_event_analysis):
    data = prepare_volatility_analysis_data(sample_event_analysis)
    assert len(data['tickers']) == 4
    assert data['mean_post'] > data['mean_pre']
    assert not data['melted_df'].empty
    assert set(data['melted_df']['Period'].unique()) == {'Pre-Event', 'Post-Event', 'Estimation Period'}


def test_prepare_volume_analysis_data(sample_event_analysis):
    data = prepare_volume_analysis_data(sample_event_analysis)
    assert data['avg_spike'] > 1.0
    assert len(data['spike_ratio']) == 4


def test_prepare_feature_importance_data():
    df = pd.DataFrame({
        'Feature': ['f1', 'f2', 'f3', 'f4'],
        'Importance': [0.1, 0.4, 0.2, 0.3]
    })
    data = prepare_feature_importance_data(df, top_n=2)
    assert data['features'] == ['f2', 'f4']
    assert len(data['importances']) == 2


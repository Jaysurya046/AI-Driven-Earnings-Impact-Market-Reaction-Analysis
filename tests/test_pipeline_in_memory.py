"""Tests for running pipeline steps purely in memory without CSV file coupling."""
import pandas as pd
import pytest

from main import EarningsAnalysisPipeline


def test_pipeline_runs_in_memory(tmp_path, price_data):
    # Create multi-event earnings data with alternating reactions
    dates_aaa = price_data.loc[price_data["Ticker"] == "AAA", "Date"].iloc[100:106].tolist()
    dates_bbb = price_data.loc[price_data["Ticker"] == "BBB", "Date"].iloc[100:106].tolist()
    
    records = []
    for i, d in enumerate(dates_aaa):
        records.append({'Date': d, 'Ticker': 'AAA', 'EPS': 1.0 + i * 0.1, 'Surprise_Percentage': 5.0 if i % 2 == 0 else -5.0})
    for i, d in enumerate(dates_bbb):
        records.append({'Date': d, 'Ticker': 'BBB', 'EPS': 1.0 + i * 0.1, 'Surprise_Percentage': -5.0 if i % 2 == 0 else 5.0})
    earnings_df = pd.DataFrame(records)

    cfg = {
        'data_dir': tmp_path / 'data',
        'models_dir': tmp_path / 'models',
        'viz_dir': tmp_path / 'viz',
        'save_intermediate': False,  # Disable CSV disk persistence
        'lookback_years': 1,
        'event_window': 5,
        'estimation_window': 60,
        'label_threshold': 0.0,
        'test_size': 0.33,
        'random_state': 42
    }
    pipeline = EarningsAnalysisPipeline(cfg)

    # Pass in-memory DataFrames directly between steps
    ea = pipeline.step_2_event_analysis(price_data=price_data, earnings_data=earnings_df)
    assert isinstance(ea, pd.DataFrame)
    assert len(ea) > 0

    features, labels = pipeline.step_3_feature_engineering(event_analysis=ea)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(labels, pd.Series)

    mc = pipeline.step_4_train_models(features=features, labels=labels)
    assert isinstance(mc, pd.DataFrame)
    assert 'Model' in mc.columns

    report = pipeline.step_6_insights_report(event_analysis=ea, model_comparison=mc)
    assert isinstance(report, str)
    assert "EARNINGS IMPACT ANALYSIS" in report

    # Verify intermediate CSV files were NOT created on disk because save_intermediate was False
    assert not (tmp_path / 'data' / 'event_analysis.csv').exists()
    assert not (tmp_path / 'data' / 'engineered_features.csv').exists()
    assert not (tmp_path / 'data' / 'model_comparison.csv').exists()

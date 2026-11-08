"""Tests for centralized feature definitions and explicit column returns."""
import pandas as pd
import pytest

from src.feature_engineering import (
    FeatureEngineer,
    SAFE_FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    METADATA_COLUMNS,
)


def test_get_feature_columns_returns_canonical_list():
    cols = FeatureEngineer.get_feature_columns()
    assert isinstance(cols, list)
    assert cols == SAFE_FEATURE_COLUMNS
    for leaky in LEAKY_COLUMNS:
        assert leaky not in cols
    for meta in METADATA_COLUMNS:
        assert meta not in cols


def test_create_features_returns_explicit_column_names(event_analysis_df):
    features, cols = FeatureEngineer.create_features(event_analysis_df, return_feature_names=True)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(cols, list)
    assert len(cols) > 0
    # Every column in cols must be present in features
    for c in cols:
        assert c in features.columns
    # Must match df.attrs
    assert features.attrs['feature_columns'] == cols


def test_create_features_backward_compatibility(event_analysis_df):
    features = FeatureEngineer.create_features(event_analysis_df)
    assert isinstance(features, pd.DataFrame)
    assert 'feature_columns' in features.attrs
    cols = FeatureEngineer.get_feature_columns(features)
    assert set(cols) == set(features.attrs['feature_columns'])

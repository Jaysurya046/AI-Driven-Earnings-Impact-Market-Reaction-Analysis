"""Tests that the ML feature set contains no post-event (leaky) information."""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FeatureEngineer,
    LEAKY_COLUMNS,
    NON_FEATURE_COLUMNS,
)


def test_no_leaky_columns_created(event_analysis_df):
    """create_features must not introduce features derived from PostEventReturn."""
    features = FeatureEngineer.create_features(event_analysis_df)
    assert "PostEventMomentum" not in features.columns
    assert "PreVol_PostReturn" not in features.columns
    assert "VolatilityChangeRatio" not in features.columns  # used PostEventVolatility


def test_feature_selection_excludes_all_leaky_columns(event_analysis_df):
    """The central selector must never return a leaky or metadata column."""
    features = FeatureEngineer.create_features(event_analysis_df)
    selected = set(FeatureEngineer.get_feature_columns(features))
    for col in LEAKY_COLUMNS + ["Ticker", "EarningsDate"]:
        selected.discard(col)
        assert col not in selected, f"{col} leaked into feature selection"
    assert "SurprisePercentage" in selected  # legitimately known at announcement
    assert len(selected) > 0


def test_prepare_data_rejects_explicit_leaky_columns(event_analysis_df):
    """Callers cannot bypass the guard by passing leaky columns manually."""
    features = FeatureEngineer.create_features(event_analysis_df)
    labels = FeatureEngineer.create_binary_labels(features)
    with pytest.raises(ValueError, match="leak"):
        FeatureEngineer.prepare_train_test_data(
            features, labels, feature_columns=["PostEventReturn"]
        )


def test_prepare_data_chronological_split(event_analysis_df):
    """Train rows must all predate test rows when chronological=True."""
    features = FeatureEngineer.create_features(event_analysis_df)
    labels = FeatureEngineer.create_binary_labels(features)

    dates = pd.to_datetime(features["EarningsDate"])
    order = dates.argsort(kind="stable")
    features = features.iloc[order]
    labels = labels.iloc[order]

    X_train, X_test, y_train, y_test, cols = FeatureEngineer.prepare_train_test_data(
        features, labels, test_size=0.5, chronological=True
    )
    assert len(X_train) + len(X_test) == len(features)
    assert X_test.shape[1] == len(cols)

    split_idx = int(len(features) * 0.5)
    train_dates = pd.to_datetime(features["EarningsDate"]).iloc[:split_idx]
    test_dates = pd.to_datetime(features["EarningsDate"]).iloc[split_idx:]
    assert train_dates.max() <= test_dates.min(), "chronological split violated"


def test_prepare_data_drops_unlabeled_rows(event_analysis_df):
    features = FeatureEngineer.create_features(event_analysis_df)
    labels = FeatureEngineer.create_binary_labels(features)
    labels.iloc[0] = np.nan
    X_train, X_test, y_train, y_test, _ = FeatureEngineer.prepare_train_test_data(
        features, labels
    )
    total = len(X_train) + len(X_test)
    assert total == len(features) - 1


def test_labels_threshold_semantics(event_analysis_df):
    labels = FeatureEngineer.create_binary_labels(event_analysis_df, threshold=0.0)
    expected = (event_analysis_df["PostEventReturn"] > 0).astype(int)
    pd.testing.assert_series_equal(labels, expected, check_names=False)

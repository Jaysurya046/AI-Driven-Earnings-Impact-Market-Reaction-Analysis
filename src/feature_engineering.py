"""
Feature Engineering Module
Creates ML features from event analysis results
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Columns that are identifiers, metadata, or leak the future outcome
# (derived from data at/after the earnings announcement). These must never
# be used as model input features; they live here so every module shares
# one definition instead of duplicating exclusion lists.
LEAKY_COLUMNS = [
    'PostEventReturn',            # the target itself
    'PostEventVolatility',        # measured after the announcement
    'PostEventAvgVolume',         # measured after the announcement
    'VolumeSpikeRatio',           # post-event volume / estimation volume
    'AbnormalReturnCAR',          # CAR computed over the post window
]

METADATA_COLUMNS = [
    'Ticker',
    'EarningsDate',
    'EPS',
    'Surprise_Percentage',
    'DataPointsInWindow',
    'DataPointsInEstimation',
]

# Backwards-compatible alias used by older call sites
NON_FEATURE_COLUMNS = LEAKY_COLUMNS + METADATA_COLUMNS


class FeatureEngineer:
    """Creates machine learning features from earnings event data"""

    @staticmethod
    def get_feature_columns(features_df: pd.DataFrame) -> List[str]:
        """
        Return the numeric columns that are safe to use as model inputs.

        Single source of truth: excludes identifiers/metadata and any column
        derived from post-announcement data (see LEAKY_COLUMNS), plus the
        label column if present.

        Args:
            features_df: DataFrame produced by create_features()

        Returns:
            Ordered list of feature column names
        """
        excluded = set(NON_FEATURE_COLUMNS) | {'Label'}
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in excluded]

    @staticmethod
    def create_features(event_analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature set for ML models.

        Leakage policy: only information available *before* the earnings
        announcement may enter a feature. Any quantity computed from prices
        or volumes strictly after the announcement date is excluded via
        NON_FEATURE_COLUMNS (enforced by get_feature_columns()).

        Args:
            event_analysis_df: DataFrame from EventAnalyzer.analyze_multiple_events()

        Returns:
            DataFrame with engineered features
        """
        df = event_analysis_df.copy()

        # 1. Earnings Surprise Features (known at/before announcement)
        df['EarningsSurpriseAbs'] = np.abs(df['SurprisePercentage'])
        df['EarningsSurpriseDirection'] = np.sign(df['SurprisePercentage'])

        # 2. Pre-event Features
        df['PreEventVolatility'] = df['PreEventVolatility'].fillna(0)
        df['PreEventReturn'] = df['PreEventReturn'].fillna(0)

        # 3. Volatility Features
        # NOTE: Post-event volatility ratios are intentionally NOT created
        # here; they leak the outcome window and were removed.
        df['PreVsEstimationVolatilityRatio'] = np.where(
            df['EstimationVolatility'] > 0,
            df['PreEventVolatility'] / df['EstimationVolatility'],
            1.0
        )

        # 4. Volume Features (pre-event only)
        df['PreEventVolumeRatio'] = np.where(
            df['EstimationAvgVolume'].fillna(0) > 0,
            df['PreEventAvgVolume'] / df['EstimationAvgVolume'],
            1.0
        )
        df['PreEventVolumeRatio'] = df['PreEventVolumeRatio'].fillna(1.0)

        # 5. Momentum Features (pre-event only; PostEventMomentum removed)
        df['PreEventMomentum'] = df['PreEventReturn']

        # 6. Derived Features (pre-event only; PreVol_PostReturn removed)
        df['SurpriseVolatilityInteraction'] = (
            df['EarningsSurpriseAbs'] * df['PreVsEstimationVolatilityRatio']
        )

        # 7. Data Quality
        df['SufficientDataPoints'] = (df['DataPointsInWindow'] >= 8).astype(int)

        return df
    
    @staticmethod
    def create_labels(
        event_analysis_df: pd.DataFrame,
        positive_threshold: float = 1.0,
        negative_threshold: float = -1.0
    ) -> pd.Series:
        """
        Create classification labels for market reaction
        
        Args:
            event_analysis_df: DataFrame from EventAnalyzer
            positive_threshold: Return % threshold for positive reaction
            negative_threshold: Return % threshold for negative reaction
            
        Returns:
            Series with labels: 1 (positive), 0 (neutral), -1 (negative)
        """
        returns = event_analysis_df['PostEventReturn']
        
        labels = pd.Series(0, index=returns.index)  # Default to neutral
        labels[returns > positive_threshold] = 1     # Positive reaction
        labels[returns < negative_threshold] = -1    # Negative reaction
        
        return labels
    
    @staticmethod
    def create_binary_labels(
        event_analysis_df: pd.DataFrame,
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Create binary classification labels
        
        Args:
            event_analysis_df: DataFrame from EventAnalyzer
            threshold: Return % threshold for positive/negative
            
        Returns:
            Series with binary labels: 1 (positive), 0 (negative)
        """
        returns = event_analysis_df['PostEventReturn']
        return (returns > threshold).astype(int)
    
    @staticmethod
    def prepare_train_test_data(
        features_df: pd.DataFrame,
        labels: pd.Series,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        chronological: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and labels for ML training.

        By default the split is *chronological* (train on the oldest events,
        test on the most recent ones) so the models are evaluated the way
        they will be used in production. A random split leaks future
        information into the training set for time-series data and must only
        be used with ``chronological=False`` for experiments.

        Args:
            features_df: DataFrame with engineered features
            labels: Series with classification labels
            feature_columns: Feature columns to use (None = safe numeric
                columns via get_feature_columns(), which excludes leaky ones)
            test_size: Fraction of rows held out for testing
            random_state: Random seed (used only when chronological=False)
            chronological: Whether to split by time instead of randomly

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_columns)
        """
        # Select features from a single source of truth
        if feature_columns is None:
            feature_columns = FeatureEngineer.get_feature_columns(features_df)
        else:
            # Guard against callers passing leaky columns explicitly
            blocked = [c for c in feature_columns if c in LEAKY_COLUMNS]
            if blocked:
                raise ValueError(
                    f"Feature columns leak post-event information: {blocked}. "
                    "Remove them or rely on the default selection."
                )

        # Keep rows with valid labels, ordered by event date when possible
        data = features_df.loc[labels.notna()].copy()
        y = labels.loc[data.index].astype(int).values

        if chronological and 'EarningsDate' in data.columns:
            order = pd.to_datetime(data['EarningsDate']).argsort(kind='stable')
            data = data.iloc[order]
            y = y[order]

        X = data[feature_columns].fillna(0).values

        if chronological:
            split_idx = max(1, int(len(X) * (1 - test_size)))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state
            )

        return X_train, X_test, y_train, y_test, feature_columns
    
    @staticmethod
    def get_feature_importance_df(
        model,
        feature_columns: list
    ) -> pd.DataFrame:
        """
        Extract feature importance from trained model
        
        Args:
            model: Trained sklearn model
            feature_columns: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df

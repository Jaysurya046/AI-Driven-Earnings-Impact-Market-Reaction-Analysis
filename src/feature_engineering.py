"""
Feature Engineering Module
Creates ML features from event analysis results
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates machine learning features from earnings event data"""
    
    @staticmethod
    def create_features(event_analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature set for ML models
        
        Args:
            event_analysis_df: DataFrame from EventAnalyzer.analyze_multiple_events()
            
        Returns:
            DataFrame with engineered features
        """
        df = event_analysis_df.copy()
        
        # 1. Earnings Surprise Features
        df['EarningsSurpriseAbs'] = np.abs(df['SurprisePercentage'])
        df['EarningsSurpriseDirection'] = np.sign(df['SurprisePercentage'])
        
        # 2. Pre-event Features
        df['PreEventVolatility'] = df['PreEventVolatility'].fillna(0)
        df['PreEventReturn'] = df['PreEventReturn'].fillna(0)
        
        # 3. Volatility Features
        df['VolatilityChangeRatio'] = np.where(
            df['EstimationVolatility'] > 0,
            df['PostEventVolatility'] / df['EstimationVolatility'],
            1.0
        )
        df['VolatilityIncrease'] = (df['PostEventVolatility'] - df['EstimationVolatility']).fillna(0)
        
        # 4. Volume Features
        df['VolumeSpikeRatio'] = df['VolumeSpikeRatio'].fillna(1.0)
        df['VolumeAnomalous'] = (df['VolumeSpikeRatio'] > 1.5).astype(int)
        
        # 5. Momentum Features
        df['PreEventMomentum'] = df['PreEventReturn']
        df['PostEventMomentum'] = df['PostEventReturn']
        
        # 6. Abnormal Returns
        df['AbnormalReturnCAR'] = df['AbnormalReturnCAR'].fillna(0)
        
        # 7. Derived Features
        df['PreVol_PostReturn'] = df['PreEventVolatility'] * np.sign(df['PostEventReturn'])
        df['SurpriseVolatilityInteraction'] = df['EarningsSurpriseAbs'] * df['VolatilityChangeRatio']
        
        # 8. Data Quality
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
        feature_columns: list = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and labels for ML training
        
        Args:
            features_df: DataFrame with engineered features
            labels: Series with classification labels
            feature_columns: List of feature columns to use (None = all numeric)
            test_size: Fraction for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Select features
        if feature_columns is None:
            feature_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove label-like columns
            feature_columns = [col for col in feature_columns if col not in 
                             ['PostEventReturn', 'PreEventReturn', 'EPS']]
        
        X = features_df[feature_columns].fillna(0).values
        y = labels.values
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
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

"""
Utility Functions Module
Helper functions for data processing and analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def filter_by_date_range(df: pd.DataFrame, start_date: str, end_date: str, date_column: str = 'Date') -> pd.DataFrame:
    """
    Filter DataFrame by date range
    
    Args:
        df: Input DataFrame
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        date_column: Name of date column
        
    Returns:
        Filtered DataFrame
    """
    df[date_column] = pd.to_datetime(df[date_column])
    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    return df[mask]


def calculate_rolling_metrics(
    df: pd.DataFrame,
    price_column: str = 'Adj Close',
    window: int = 30
) -> pd.DataFrame:
    """
    Calculate rolling volatility and returns
    
    Args:
        df: DataFrame with price data
        price_column: Name of price column
        window: Rolling window size
        
    Returns:
        DataFrame with rolling metrics
    """
    df['Returns'] = df[price_column].pct_change() * 100
    df['RollingVolatility'] = df['Returns'].rolling(window=window).std()
    df['RollingMean'] = df['Returns'].rolling(window=window).mean()
    
    return df


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Normalize features to [0, 1] range
    
    Args:
        X: Feature array
        
    Returns:
        Normalized features
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
    
    return X_normalized


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'forward_fill',
    numeric_columns: list = None
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame
    
    Args:
        df: Input DataFrame
        strategy: 'forward_fill', 'backward_fill', 'drop', or 'mean'
        numeric_columns: Columns to process (None = all numeric)
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'forward_fill':
        df[numeric_columns] = df[numeric_columns].ffill()
    elif strategy == 'backward_fill':
        df[numeric_columns] = df[numeric_columns].bfill()
    elif strategy == 'mean':
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    elif strategy == 'drop':
        df = df.dropna(subset=numeric_columns)
    
    return df


def detect_outliers(
    df: pd.DataFrame,
    columns: list,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.Series:
    """
    Detect outliers in DataFrame
    
    Args:
        df: Input DataFrame
        columns: Columns to check
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean Series marking outliers
    """
    outliers = pd.Series([False] * len(df), index=df.index)
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[col] < Q1 - threshold * IQR) | (df[col] > Q3 + threshold * IQR)
        elif method == 'zscore':
            from scipy import stats
            outlier_mask = np.abs(stats.zscore(df[col].dropna())) > threshold
        
        outliers = outliers | outlier_mask
    
    return outliers


def create_performance_report(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray = None
) -> dict:
    """
    Create comprehensive performance report
    
    Args:
        actual: Actual labels
        predicted: Predicted labels
        probabilities: Prediction probabilities (optional)
        
    Returns:
        Dictionary with performance metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )
    
    report = {
        'accuracy': accuracy_score(actual, predicted),
        'precision': precision_score(actual, predicted, zero_division=0),
        'recall': recall_score(actual, predicted, zero_division=0),
        'f1': f1_score(actual, predicted, zero_division=0),
        'confusion_matrix': confusion_matrix(actual, predicted).tolist()
    }
    
    if probabilities is not None:
        try:
            report['auc_roc'] = roc_auc_score(actual, probabilities)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not calculate AUC-ROC in report: {e}")
    
    return report


def get_business_day(date: pd.Timestamp, days_offset: int = 1) -> pd.Timestamp:
    """
    Get business day (trading day) offset from a date
    
    Args:
        date: Reference date
        days_offset: Number of business days to offset
        
    Returns:
        Adjusted date
    """
    offset = pd.tseries.offsets.BDay(days_offset)
    return date + offset


def export_analysis_summary(
    event_analysis: pd.DataFrame,
    features: pd.DataFrame,
    model_results: dict,
    output_path: str
) -> None:
    """
    Export complete analysis summary to Excel
    
    Args:
        event_analysis: Event analysis results
        features: Engineered features
        model_results: Model evaluation results
        output_path: Path to save Excel file
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            event_analysis.to_excel(writer, sheet_name='Event Analysis', index=False)
            features.to_excel(writer, sheet_name='Features', index=False)
            
            # Model results summary
            model_summary = []
            for model_name, results in model_results.items():
                model_summary.append({
                    'Model': model_name,
                    'Accuracy': results.get('Accuracy', np.nan),
                    'Precision': results.get('Precision', np.nan),
                    'Recall': results.get('Recall', np.nan),
                    'F1-Score': results.get('F1-Score', np.nan)
                })
            
            pd.DataFrame(model_summary).to_excel(writer, sheet_name='Model Results', index=False)
        
        logger.info(f"Analysis summary exported to {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting summary: {str(e)}")

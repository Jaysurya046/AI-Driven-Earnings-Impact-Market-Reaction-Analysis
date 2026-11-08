"""
Shared Chart Data Preparation Module
Provides pure data-shaping and statistical preparation functions for analytical charts.
Decouples data preparation from surface-specific rendering (Matplotlib, Plotly, Notebooks).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def prepare_surprise_vs_reaction_data(event_analysis_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare data and linear trend parameters for Surprise vs Reaction scatter plots.

    Args:
        event_analysis_df: DataFrame containing 'SurprisePercentage' and 'PostEventReturn'

    Returns:
        Dictionary with cleaned points, colors, regression line parameters, and correlation.
    """
    if event_analysis_df is None or event_analysis_df.empty:
        return {
            'cleaned_df': pd.DataFrame(),
            'surprises': np.array([]),
            'returns': np.array([]),
            'colors': [],
            'has_trend': False,
            'slope': 0.0,
            'intercept': 0.0,
            'correlation': 0.0,
            'r_squared': 0.0,
            'trend_x': np.array([]),
            'trend_y': np.array([]),
        }

    valid_mask = event_analysis_df['SurprisePercentage'].notna() & event_analysis_df['PostEventReturn'].notna()
    cleaned = event_analysis_df[valid_mask].copy()

    if cleaned.empty:
        return {
            'cleaned_df': cleaned,
            'surprises': np.array([]),
            'returns': np.array([]),
            'colors': [],
            'has_trend': False,
            'slope': 0.0,
            'intercept': 0.0,
            'correlation': 0.0,
            'r_squared': 0.0,
            'trend_x': np.array([]),
            'trend_y': np.array([]),
        }

    surprises = cleaned['SurprisePercentage'].to_numpy(dtype=float)
    returns = cleaned['PostEventReturn'].to_numpy(dtype=float)
    colors = ['green' if r > 0 else 'red' for r in returns]

    has_trend = bool(len(surprises) >= 2 and np.std(surprises) > 1e-9)
    slope, intercept, r_squared = 0.0, 0.0, 0.0
    trend_x, trend_y = np.array([]), np.array([])

    if has_trend:
        z = np.polyfit(surprises, returns, 1)
        slope = float(z[0])
        intercept = float(z[1])
        p = np.poly1d(z)
        trend_x = np.linspace(surprises.min(), surprises.max(), 100)
        trend_y = p(trend_x)

        corr_matrix = np.corrcoef(surprises, returns)
        corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        r_squared = float(corr ** 2)
    else:
        corr = 0.0

    return {
        'cleaned_df': cleaned,
        'surprises': surprises,
        'returns': returns,
        'colors': colors,
        'has_trend': has_trend,
        'slope': slope,
        'intercept': intercept,
        'correlation': corr,
        'r_squared': r_squared,
        'trend_x': trend_x,
        'trend_y': trend_y,
    }


def prepare_model_comparison_data(
    model_comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Prepare standardized metrics matrix and rankings from model comparison results.

    Args:
        model_comparison_df: DataFrame from ModelTrainer.train_all_models()
        metrics: Optional list of metrics to include

    Returns:
        Dictionary with models, metrics, 2D value matrix, best model record, and melted DataFrame.
    """
    if model_comparison_df is None or model_comparison_df.empty:
        return {
            'models': [],
            'metrics': [],
            'matrix': {},
            'best_model': None,
            'display_df': pd.DataFrame(),
            'melted_df': pd.DataFrame(),
        }

    default_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    if 'AUC-ROC' in model_comparison_df.columns and model_comparison_df['AUC-ROC'].notna().any():
        default_metrics.append('AUC-ROC')

    selected_metrics = [m for m in (metrics or default_metrics) if m in model_comparison_df.columns]

    models = model_comparison_df['Model'].tolist()
    matrix = {m: model_comparison_df[m].tolist() for m in selected_metrics}

    best_idx = model_comparison_df['F1-Score'].idxmax() if 'F1-Score' in model_comparison_df else 0
    best_model = model_comparison_df.loc[best_idx].to_dict()

    melted = model_comparison_df.melt(
        id_vars=['Model'],
        value_vars=selected_metrics,
        var_name='Metric',
        value_name='Score'
    )

    display_df = model_comparison_df[['Model'] + selected_metrics].copy()
    for col in selected_metrics:
        display_df[col] = display_df[col].round(4)

    return {
        'models': models,
        'metrics': selected_metrics,
        'matrix': matrix,
        'best_model': best_model,
        'display_df': display_df,
        'melted_df': melted,
    }


def prepare_volatility_analysis_data(event_analysis_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare volatility comparisons across estimation, pre-event, and post-event periods.

    Args:
        event_analysis_df: DataFrame with event analysis records

    Returns:
        Dictionary with period labels, values per event, mean shifts, and melted DataFrame.
    """
    if event_analysis_df is None or event_analysis_df.empty:
        return {
            'tickers': [],
            'estimation': np.array([]),
            'pre_event': np.array([]),
            'post_event': np.array([]),
            'mean_estimation': 0.0,
            'mean_pre': 0.0,
            'mean_post': 0.0,
            'melted_df': pd.DataFrame(),
        }

    estimation = event_analysis_df['EstimationVolatility'].to_numpy(dtype=float) if 'EstimationVolatility' in event_analysis_df else np.zeros(len(event_analysis_df))
    pre_event = event_analysis_df['PreEventVolatility'].to_numpy(dtype=float) if 'PreEventVolatility' in event_analysis_df else np.zeros(len(event_analysis_df))
    post_event = event_analysis_df['PostEventVolatility'].to_numpy(dtype=float) if 'PostEventVolatility' in event_analysis_df else np.zeros(len(event_analysis_df))

    # Standardized melted DataFrame for Box/Violin plots
    melt_cols = [c for c in ['PreEventVolatility', 'PostEventVolatility', 'EstimationVolatility'] if c in event_analysis_df.columns]
    id_vars = ['Ticker'] if 'Ticker' in event_analysis_df.columns else []

    if melt_cols and id_vars:
        melted = event_analysis_df[id_vars + melt_cols].melt(
            id_vars=id_vars,
            value_vars=melt_cols,
            var_name='Period',
            value_name='Volatility'
        )
        label_map = {
            'PreEventVolatility': 'Pre-Event',
            'PostEventVolatility': 'Post-Event',
            'EstimationVolatility': 'Estimation Period'
        }
        melted['Period'] = melted['Period'].map(label_map)
    else:
        melted = pd.DataFrame()

    return {
        'tickers': event_analysis_df['Ticker'].tolist() if 'Ticker' in event_analysis_df.columns else [],
        'estimation': estimation,
        'pre_event': pre_event,
        'post_event': post_event,
        'mean_estimation': float(np.nanmean(estimation)) if len(estimation) else 0.0,
        'mean_pre': float(np.nanmean(pre_event)) if len(pre_event) else 0.0,
        'mean_post': float(np.nanmean(post_event)) if len(post_event) else 0.0,
        'melted_df': melted,
    }


def prepare_volume_analysis_data(event_analysis_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare volume spike and pre/post volume distribution metrics.

    Args:
        event_analysis_df: DataFrame with pre- and post-event volume columns

    Returns:
        Dictionary with pre/post volumes, volume spike ratios, and average surges.
    """
    if event_analysis_df is None or event_analysis_df.empty:
        return {
            'pre_volume': np.array([]),
            'post_volume': np.array([]),
            'spike_ratio': np.array([]),
            'avg_spike': 1.0,
            'percent_spiked': 0.0,
        }

    pre = event_analysis_df['PreEventVolume'].to_numpy(dtype=float) if 'PreEventVolume' in event_analysis_df else np.zeros(len(event_analysis_df))
    post = event_analysis_df['PostEventVolume'].to_numpy(dtype=float) if 'PostEventVolume' in event_analysis_df else np.zeros(len(event_analysis_df))

    if 'VolumeSpikeRatio' in event_analysis_df:
        spikes = event_analysis_df['VolumeSpikeRatio'].to_numpy(dtype=float)
    else:
        spikes = np.where(pre > 0, post / pre, 1.0)

    avg_spike = float(np.nanmean(spikes)) if len(spikes) else 1.0
    percent_spiked = float((spikes > 1.5).mean() * 100.0) if len(spikes) else 0.0

    return {
        'pre_volume': pre,
        'post_volume': post,
        'spike_ratio': spikes,
        'avg_spike': avg_spike,
        'percent_spiked': percent_spiked,
    }


def prepare_feature_importance_data(
    importance_df: pd.DataFrame,
    top_n: int = 15
) -> Dict[str, Any]:
    """
    Prepare sorted feature importance data for ranking charts.

    Args:
        importance_df: DataFrame with 'Feature' and 'Importance'
        top_n: Number of top features to retain

    Returns:
        Dictionary with sorted feature names, importance scores, and sliced DataFrame.
    """
    if importance_df is None or importance_df.empty:
        return {'features': [], 'importances': np.array([]), 'top_df': pd.DataFrame()}

    col = 'Importance' if 'Importance' in importance_df.columns else importance_df.columns[-1]
    sorted_df = importance_df.sort_values(col, ascending=False).head(top_n)

    return {
        'features': sorted_df['Feature'].tolist() if 'Feature' in sorted_df.columns else [f"f{i}" for i in range(len(sorted_df))],
        'importances': sorted_df[col].to_numpy(dtype=float),
        'top_df': sorted_df,
    }


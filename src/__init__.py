"""
AI-Driven Earnings Impact & Market Reaction Analysis
Package initialization exposing primary analysis and modeling classes.
"""

from src.config import AppConfig, load_config, mask_secrets
from src.data_collection import DataCollector, validate_collected_data
from src.event_analysis import EventAnalyzer
from src.feature_engineering import (
    LEAKY_COLUMNS,
    METADATA_COLUMNS,
    NON_FEATURE_COLUMNS,
    SAFE_FEATURE_COLUMNS,
    FeatureEngineer,
)
from src.ml_models import ModelTrainer
from src.backtesting import Backtester
from src.visualizations import Visualizer
from src.sector_comparison import SectorAnalyzer
from src.sentiment_analysis import SentimentAnalyzer
from src.chart_data import (
    prepare_surprise_vs_reaction_data,
    prepare_model_comparison_data,
    prepare_volatility_analysis_data,
    prepare_volume_analysis_data,
    prepare_feature_importance_data,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "AppConfig",
    "load_config",
    "mask_secrets",
    "DataCollector",
    "validate_collected_data",
    "EventAnalyzer",
    "FeatureEngineer",
    "SAFE_FEATURE_COLUMNS",
    "NON_FEATURE_COLUMNS",
    "LEAKY_COLUMNS",
    "METADATA_COLUMNS",
    "ModelTrainer",
    "Backtester",
    "Visualizer",
    "SectorAnalyzer",
    "SentimentAnalyzer",
    "prepare_surprise_vs_reaction_data",
    "prepare_model_comparison_data",
    "prepare_volatility_analysis_data",
    "prepare_volume_analysis_data",
    "prepare_feature_importance_data",
]

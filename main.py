"""
Main Analysis Pipeline
Orchestrates the complete earnings impact analysis workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List

from src.data_collection import DataCollector
from src.event_analysis import EventAnalyzer
from src.feature_engineering import FeatureEngineer
from src.ml_models import ModelTrainer
from src.visualizations import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarningsAnalysisPipeline:
    """Main orchestration class for the complete analysis workflow"""
    
    def __init__(self, config: dict):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary with settings
        """
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.models_dir = Path(config.get('models_dir', 'models'))
        self.viz_dir = Path(config.get('viz_dir', 'visualizations'))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_collector = DataCollector(lookback_years=config.get('lookback_years', 5))
        self.event_analyzer = EventAnalyzer(
            event_window=config.get('event_window', 5),
            estimation_window=config.get('estimation_window', 60)
        )
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.visualizer = Visualizer()
        
        # Data storage
        self.price_data = None
        self.earnings_data = None
        self.event_analysis = None
        self.features = None
        self.labels = None
    
    def step_1_collect_data(self, tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Step 1: Collect historical price and earnings data
        
        Args:
            tickers: List of stock tickers
            
        Returns:
            Tuple of (price_data, earnings_data)
        """
        logger.info("=" * 80)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("=" * 80)
        
        self.price_data, self.earnings_data = self.data_collector.fetch_multiple_stocks(tickers)
        
        # Save raw data
        self.data_collector.save_to_csv(
            self.price_data,
            self.data_dir / 'raw_price_data.csv'
        )
        self.data_collector.save_to_csv(
            self.earnings_data,
            self.data_dir / 'raw_earnings_data.csv'
        )
        
        logger.info(f"Collected {len(self.price_data)} price records for {len(tickers)} stocks")
        logger.info(f"Collected {len(self.earnings_data)} earnings records")
        
        return self.price_data, self.earnings_data
    
    def step_2_event_analysis(self) -> pd.DataFrame:
        """
        Step 2: Analyze price movements around earnings events
        
        Returns:
            DataFrame with event analysis results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: EVENT-BASED ANALYSIS")
        logger.info("=" * 80)
        
        self.event_analysis = self.event_analyzer.analyze_multiple_events(
            self.price_data,
            self.earnings_data
        )
        
        # Save event analysis
        self.event_analysis.to_csv(
            self.data_dir / 'event_analysis.csv',
            index=False
        )
        
        logger.info(f"Analyzed {len(self.event_analysis)} earnings events")
        logger.info(f"\nSummary Statistics:")
        logger.info(f"  Avg Pre-Event Return: {self.event_analysis['PreEventReturn'].mean():.2f}%")
        logger.info(f"  Avg Post-Event Return: {self.event_analysis['PostEventReturn'].mean():.2f}%")
        logger.info(f"  Avg CAR: {self.event_analysis['AbnormalReturnCAR'].mean():.2f}%")
        logger.info(f"  Avg Volume Spike Ratio: {self.event_analysis['VolumeSpikeRatio'].mean():.2f}x")
        
        return self.event_analysis
    
    def step_3_feature_engineering(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Step 3: Engineer features for ML models
        
        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        self.features = self.feature_engineer.create_features(self.event_analysis)
        self.labels = self.feature_engineer.create_binary_labels(
            self.event_analysis,
            threshold=self.config.get('label_threshold', 0.0)
        )
        
        # Save features
        self.features.to_csv(
            self.data_dir / 'engineered_features.csv',
            index=False
        )
        
        logger.info(f"Created {len(self.features.columns) - len(self.event_analysis.columns)} new features")
        logger.info(f"Feature set shape: {self.features.shape}")
        logger.info(f"\nLabel Distribution:")
        logger.info(f"  Positive reactions: {(self.labels == 1).sum()} ({(self.labels == 1).sum() / len(self.labels) * 100:.1f}%)")
        logger.info(f"  Negative reactions: {(self.labels == 0).sum()} ({(self.labels == 0).sum() / len(self.labels) * 100:.1f}%)")
        
        return self.features, self.labels
    
    def step_4_train_models(self) -> pd.DataFrame:
        """
        Step 4: Train ML models
        
        Returns:
            DataFrame with model comparison results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: MACHINE LEARNING MODELING")
        logger.info("=" * 80)
        
        # Prepare training data (feature list comes from the single source of
        # truth in FeatureEngineer, which excludes leaky post-event columns)
        X_train, X_test, y_train, y_test, feature_columns = \
            self.feature_engineer.prepare_train_test_data(
                self.features,
                self.labels,
                test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42),
                chronological=True
            )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Features used: {len(feature_columns)}")
        
        # Train models
        model_comparison = self.model_trainer.train_all_models(X_train, X_test, y_train, y_test)
        
        # Save comparison
        model_comparison.to_csv(
            self.data_dir / 'model_comparison.csv',
            index=False
        )

        # Save trained models with metadata
        for model_name in self.model_trainer.models:
            clean_name = model_name.lower().replace(' ', '_')
            self.model_trainer.save_model(
                model_name=model_name,
                filepath=self.models_dir / f"{clean_name}.pkl",
                feature_columns=feature_columns
            )
        
        logger.info("\nModel Comparison Results:")
        logger.info(model_comparison.to_string())
        
        return model_comparison
    
    def step_5_visualizations(self) -> None:
        """
        Step 5: Generate visualizations
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: VISUALIZATIONS & DASHBOARDS")
        logger.info("=" * 80)
        
        # Plot 1: Surprise vs Reaction
        self.visualizer.plot_surprise_vs_reaction(
            self.event_analysis,
            save_path=self.viz_dir / '01_surprise_vs_reaction.png'
        )
        
        # Plot 2: Model Comparison
        model_comparison = pd.read_csv(self.data_dir / 'model_comparison.csv')
        self.visualizer.plot_model_comparison(
            model_comparison,
            save_path=self.viz_dir / '02_model_comparison.png'
        )
        
        # Plot 3: Volatility Analysis
        self.visualizer.plot_volatility_analysis(
            self.event_analysis,
            save_path=self.viz_dir / '03_volatility_analysis.png'
        )
        
        # Plot 4: Volume Analysis
        self.visualizer.plot_volume_analysis(
            self.event_analysis,
            save_path=self.viz_dir / '04_volume_analysis.png'
        )
        
        # Plot 5: Feature Importance (for best model)
        best_model_name = model_comparison.loc[model_comparison['F1-Score'].idxmax(), 'Model']
        best_model = self.model_trainer.models[best_model_name]
        scaler = self.model_trainer.scalers[best_model_name]
        
        # Use the centralized safe feature list (no duplicated exclusion lists)
        feature_columns = self.feature_engineer.get_feature_columns(self.features)
        
        importance_df = self.feature_engineer.get_feature_importance_df(best_model, feature_columns)
        
        if not importance_df.empty:
            self.visualizer.plot_feature_importance(
                importance_df,
                top_n=15,
                save_path=self.viz_dir / f'05_feature_importance_{best_model_name}.png'
            )
        
        logger.info(f"Visualizations saved to {self.viz_dir}")
    
    def step_6_insights_report(self) -> str:
        """
        Step 6: Generate insights report
        
        Returns:
            Report string with key insights
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: INSIGHTS & REPORTING")
        logger.info("=" * 80)
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("EARNINGS IMPACT ANALYSIS - FINAL REPORT")
        report.append("=" * 80)
        
        # Data Summary
        report.append("\n1. DATA SUMMARY")
        report.append(f"   - Total Earnings Events Analyzed: {len(self.event_analysis)}")
        report.append(f"   - Unique Stocks: {self.event_analysis['Ticker'].nunique()}")
        report.append(f"   - Date Range: {self.event_analysis['EarningsDate'].min()} to {self.event_analysis['EarningsDate'].max()}")
        
        # Return Analysis
        report.append("\n2. RETURN ANALYSIS")
        report.append(f"   - Avg Pre-Event Return: {self.event_analysis['PreEventReturn'].mean():.2f}%")
        report.append(f"   - Avg Post-Event Return: {self.event_analysis['PostEventReturn'].mean():.2f}%")
        report.append(f"   - Avg Cumulative Abnormal Return: {self.event_analysis['AbnormalReturnCAR'].mean():.2f}%")
        
        # Volatility Insights
        report.append("\n3. VOLATILITY INSIGHTS")
        report.append(f"   - Avg Pre-Event Volatility: {self.event_analysis['PreEventVolatility'].mean():.2f}%")
        report.append(f"   - Avg Post-Event Volatility: {self.event_analysis['PostEventVolatility'].mean():.2f}%")
        report.append(f"   - Volatility Change: {(self.event_analysis['PostEventVolatility'].mean() - self.event_analysis['EstimationVolatility'].mean()):.2f}%")
        
        # Volume Insights
        report.append("\n4. VOLUME INSIGHTS")
        report.append(f"   - Avg Volume Spike Ratio: {self.event_analysis['VolumeSpikeRatio'].mean():.2f}x")
        report.append(f"   - Events with Significant Volume Spike (>1.5x): {(self.event_analysis['VolumeSpikeRatio'] > 1.5).sum()}")
        
        # Earnings Surprise Correlation
        correlation = self.event_analysis['SurprisePercentage'].corr(self.event_analysis['PostEventReturn'])
        report.append("\n5. EARNINGS SURPRISE CORRELATION")
        report.append(f"   - Correlation (Surprise vs Post-Event Return): {correlation:.3f}")
        
        # Model Performance
        model_comparison = pd.read_csv(self.data_dir / 'model_comparison.csv')
        best_model = model_comparison.loc[model_comparison['F1-Score'].idxmax()]
        report.append("\n6. MODEL PERFORMANCE")
        report.append(f"   - Best Model: {best_model['Model']}")
        report.append(f"   - Accuracy: {best_model['Accuracy']:.4f}")
        report.append(f"   - Precision: {best_model['Precision']:.4f}")
        report.append(f"   - Recall: {best_model['Recall']:.4f}")
        report.append(f"   - F1-Score: {best_model['F1-Score']:.4f}")
        
        # Key Findings
        report.append("\n7. KEY FINDINGS")
        pos_reactions = (self.event_analysis['PostEventReturn'] > 0).sum()
        report.append(f"   - Positive post-event reactions: {pos_reactions} ({pos_reactions/len(self.event_analysis)*100:.1f}%)")
        report.append(f"   - Average surprise magnitude: {self.event_analysis['SurprisePercentage'].abs().mean():.2f}%")
        
        # Recommendations
        report.append("\n8. RECOMMENDATIONS")
        report.append("   - Focus on stocks with high earnings surprise magnitudes")
        report.append("   - Monitor volume spikes as a potential predictive indicator")
        report.append("   - Use ensemble methods for better prediction accuracy")
        report.append("   - Consider sector-specific analysis for improved insights")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        with open(self.data_dir / 'final_report.txt', 'w') as f:
            f.write(report_text)
        
        logger.info(report_text)
        
        return report_text
    
    def run_complete_pipeline(self, tickers: List[str]) -> None:
        """
        Run the complete analysis pipeline
        
        Args:
            tickers: List of stock tickers to analyze
        """
        try:
            logger.info("STARTING EARNINGS IMPACT ANALYSIS PIPELINE")
            logger.info(f"Analyzing: {', '.join(tickers)}")
            
            # Execute all steps
            self.step_1_collect_data(tickers)
            self.step_2_event_analysis()
            self.step_3_feature_engineering()
            self.step_4_train_models()
            self.step_5_visualizations()
            self.step_6_insights_report()
            
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    # Configuration
    config = {
        'lookback_years': 5,
        'event_window': 5,
        'estimation_window': 60,
        'label_threshold': 0.0,
        'data_dir': 'data',
        'models_dir': 'models',
        'viz_dir': 'visualizations'
    }
    
    # List of stocks to analyze (modify as needed)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    
    # Run pipeline
    pipeline = EarningsAnalysisPipeline(config)
    pipeline.run_complete_pipeline(tickers)

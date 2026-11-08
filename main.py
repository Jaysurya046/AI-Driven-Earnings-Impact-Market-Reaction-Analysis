"""
Main Analysis Pipeline
Orchestrates the complete earnings impact analysis workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

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
    
    def __init__(self, config: Any):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary or AppConfig instance
        """
        if hasattr(config, 'to_dict'):
            self.config = config.to_dict()
        else:
            self.config = dict(config)

        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.models_dir = Path(self.config.get('models_dir', 'models'))
        self.viz_dir = Path(self.config.get('viz_dir', 'visualizations'))
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_collector = DataCollector(
            lookback_years=self.config.get('lookback_years', 5),
            max_workers=self.config.get('max_workers', 5)
        )
        self.event_analyzer = EventAnalyzer(
            event_window=self.config.get('event_window', 5),
            estimation_window=self.config.get('estimation_window', 60)
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
        self.model_comparison = None
        self.save_intermediate = config.get('save_intermediate', True)
    
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
        
        # Validate data shapes and schema immediately after collection
        self.data_collector.validate_collected_data(self.price_data, self.earnings_data)
        
        # Save raw data if intermediate saving enabled
        if self.save_intermediate:
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
    
    def step_2_event_analysis(
        self,
        price_data: Optional[pd.DataFrame] = None,
        earnings_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Step 2: Analyze price movements around earnings events
        
        Args:
            price_data: Optional in-memory price DataFrame
            earnings_data: Optional in-memory earnings DataFrame
            
        Returns:
            DataFrame with event analysis results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: EVENT-BASED ANALYSIS")
        logger.info("=" * 80)
        
        prices = price_data if price_data is not None else self.price_data
        earnings = earnings_data if earnings_data is not None else self.earnings_data
        
        if prices is None or earnings is None:
            raise ValueError("Price data and earnings data must be provided or collected first.")
        
        self.price_data = prices
        self.earnings_data = earnings
        
        self.event_analysis = self.event_analyzer.analyze_multiple_events(
            prices,
            earnings
        )
        
        if self.save_intermediate:
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
    
    def step_3_feature_engineering(
        self,
        event_analysis: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Step 3: Engineer features for ML models
        
        Args:
            event_analysis: Optional in-memory event analysis DataFrame
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        ea = event_analysis if event_analysis is not None else self.event_analysis
        if ea is None:
            raise ValueError("Event analysis data must be provided or computed first.")
        self.event_analysis = ea
        
        self.features = self.feature_engineer.create_features(ea)
        self.labels = self.feature_engineer.create_binary_labels(
            ea,
            threshold=self.config.get('label_threshold', 0.0)
        )
        
        if self.save_intermediate:
            self.features.to_csv(
                self.data_dir / 'engineered_features.csv',
                index=False
            )
        
        logger.info(f"Created {len(self.features.columns) - len(ea.columns)} new features")
        logger.info(f"Feature set shape: {self.features.shape}")
        logger.info(f"\nLabel Distribution:")
        logger.info(f"  Positive reactions: {(self.labels == 1).sum()} ({(self.labels == 1).sum() / len(self.labels) * 100:.1f}%)")
        logger.info(f"  Negative reactions: {(self.labels == 0).sum()} ({(self.labels == 0).sum() / len(self.labels) * 100:.1f}%)")
        
        return self.features, self.labels
    
    def step_4_train_models(
        self,
        features: Optional[pd.DataFrame] = None,
        labels: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Step 4: Train ML models
        
        Args:
            features: Optional in-memory features DataFrame
            labels: Optional in-memory labels Series
            
        Returns:
            DataFrame with model comparison results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: MACHINE LEARNING MODELING")
        logger.info("=" * 80)
        
        feats = features if features is not None else self.features
        lbls = labels if labels is not None else self.labels
        
        if feats is None or lbls is None:
            raise ValueError("Features and labels must be provided or engineered first.")
        self.features = feats
        self.labels = lbls
        
        # Prepare training data (feature list comes from the single source of
        # truth in FeatureEngineer, which excludes leaky post-event columns)
        X_train, X_test, y_train, y_test, feature_columns = \
            self.feature_engineer.prepare_train_test_data(
                feats,
                lbls,
                test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42),
                chronological=True
            )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Features used: {len(feature_columns)}")
        
        # Train configured models in memory
        models_to_train = self.config.get('models_to_train')
        model_comparison = self.model_trainer.train_all_models(
            X_train, X_test, y_train, y_test, models_to_train=models_to_train
        )

        # Optional deep learning LSTM training
        if self.config.get('enable_lstm', False):
            try:
                from src.lstm_models import LSTMPredictor
                if LSTMPredictor.is_available():
                    logger.info("Training optional deep learning LSTM predictor...")
                    lstm = LSTMPredictor(sequence_length=10, lstm_units=50)
                    lstm.train(X_train, y_train, epochs=20, batch_size=32)
                    lstm_metrics = lstm.evaluate(X_test, y_test)
                    if lstm_metrics:
                        lstm_row = pd.DataFrame([{
                            'Model': 'LSTM',
                            'Accuracy': lstm_metrics.get('Accuracy', np.nan),
                            'Precision': lstm_metrics.get('Precision', np.nan),
                            'Recall': lstm_metrics.get('Recall', np.nan),
                            'F1-Score': lstm_metrics.get('F1-Score', np.nan),
                            'AUC-ROC': lstm_metrics.get('AUC-ROC', np.nan)
                        }])
                        model_comparison = pd.concat([model_comparison, lstm_row], ignore_index=True)
                else:
                    logger.warning("enable_lstm is True, but TensorFlow is not installed. Install via: pip install .[deeplearning]")
            except Exception as e:
                logger.warning(f"Optional LSTM training failed: {e}")

        self.model_comparison = model_comparison
        
        if self.save_intermediate:
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
    
    def step_5_visualizations(
        self,
        event_analysis: Optional[pd.DataFrame] = None,
        model_comparison: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Step 5: Generate visualizations using in-memory DataFrames
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: VISUALIZATIONS & DASHBOARDS")
        logger.info("=" * 80)
        
        ea = event_analysis if event_analysis is not None else self.event_analysis
        mc = model_comparison if model_comparison is not None else self.model_comparison
        
        if mc is None and (self.data_dir / 'model_comparison.csv').exists():
            mc = pd.read_csv(self.data_dir / 'model_comparison.csv')
            
        if ea is None or mc is None:
            logger.warning("Missing event_analysis or model_comparison for visualizations. Skipping.")
            return
        
        # Plot 1: Surprise vs Reaction
        self.visualizer.plot_surprise_vs_reaction(
            ea,
            save_path=self.viz_dir / '01_surprise_vs_reaction.png'
        )
        
        # Plot 2: Model Comparison (uses in-memory DataFrame)
        self.visualizer.plot_model_comparison(
            mc,
            save_path=self.viz_dir / '02_model_comparison.png'
        )
        
        # Plot 3: Volatility Analysis
        self.visualizer.plot_volatility_analysis(
            ea,
            save_path=self.viz_dir / '03_volatility_analysis.png'
        )
        
        # Plot 4: Volume Analysis
        self.visualizer.plot_volume_analysis(
            ea,
            save_path=self.viz_dir / '04_volume_analysis.png'
        )
        
        # Plot 5: Feature Importance (for best model)
        best_model_name = mc.loc[mc['F1-Score'].idxmax(), 'Model']
        if best_model_name in self.model_trainer.models:
            best_model = self.model_trainer.models[best_model_name]
            feature_columns = self.feature_engineer.get_feature_columns(self.features)
            importance_df = self.feature_engineer.get_feature_importance_df(best_model, feature_columns)
            
            if not importance_df.empty:
                self.visualizer.plot_feature_importance(
                    importance_df,
                    top_n=15,
                    save_path=self.viz_dir / f'05_feature_importance_{best_model_name}.png'
                )
        
        logger.info(f"Visualizations saved to {self.viz_dir}")
    
    def step_6_insights_report(
        self,
        event_analysis: Optional[pd.DataFrame] = None,
        model_comparison: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Step 6: Generate insights report using in-memory DataFrames
        
        Returns:
            Report string with key insights
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: INSIGHTS & REPORTING")
        logger.info("=" * 80)
        
        ea = event_analysis if event_analysis is not None else self.event_analysis
        mc = model_comparison if model_comparison is not None else self.model_comparison
        
        if mc is None and (self.data_dir / 'model_comparison.csv').exists():
            mc = pd.read_csv(self.data_dir / 'model_comparison.csv')
            
        if ea is None or mc is None:
            raise ValueError("Missing event_analysis or model_comparison for report generation.")
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("EARNINGS IMPACT ANALYSIS - FINAL REPORT")
        report.append("=" * 80)
        
        # Data Summary
        report.append("\n1. DATA SUMMARY")
        report.append(f"   - Total Earnings Events Analyzed: {len(ea)}")
        report.append(f"   - Unique Stocks: {ea['Ticker'].nunique()}")
        report.append(f"   - Date Range: {ea['EarningsDate'].min()} to {ea['EarningsDate'].max()}")
        
        # Return Analysis
        report.append("\n2. RETURN ANALYSIS")
        report.append(f"   - Avg Pre-Event Return: {ea['PreEventReturn'].mean():.2f}%")
        report.append(f"   - Avg Post-Event Return: {ea['PostEventReturn'].mean():.2f}%")
        report.append(f"   - Avg Cumulative Abnormal Return: {ea['AbnormalReturnCAR'].mean():.2f}%")
        
        # Volatility Insights
        report.append("\n3. VOLATILITY INSIGHTS")
        report.append(f"   - Avg Pre-Event Volatility: {ea['PreEventVolatility'].mean():.2f}%")
        report.append(f"   - Avg Post-Event Volatility: {ea['PostEventVolatility'].mean():.2f}%")
        report.append(f"   - Volatility Change: {(ea['PostEventVolatility'].mean() - ea['EstimationVolatility'].mean()):.2f}%")
        
        # Volume Insights
        report.append("\n4. VOLUME INSIGHTS")
        report.append(f"   - Avg Volume Spike Ratio: {ea['VolumeSpikeRatio'].mean():.2f}x")
        report.append(f"   - Events with Significant Volume Spike (>1.5x): {(ea['VolumeSpikeRatio'] > 1.5).sum()}")
        
        # Earnings Surprise Correlation
        correlation = ea['SurprisePercentage'].corr(ea['PostEventReturn']) if len(ea) > 1 else 0.0
        report.append("\n5. EARNINGS SURPRISE CORRELATION")
        report.append(f"   - Correlation (Surprise vs Post-Event Return): {correlation:.3f}")
        
        # Model Performance (using in-memory comparison DataFrame)
        best_model = mc.loc[mc['F1-Score'].idxmax()]
        report.append("\n6. MODEL PERFORMANCE")
        report.append(f"   - Best Model: {best_model['Model']}")
        report.append(f"   - Accuracy: {best_model['Accuracy']:.4f}")
        report.append(f"   - Precision: {best_model['Precision']:.4f}")
        report.append(f"   - Recall: {best_model['Recall']:.4f}")
        report.append(f"   - F1-Score: {best_model['F1-Score']:.4f}")
        
        # Key Findings
        report.append("\n7. KEY FINDINGS")
        pos_reactions = (ea['PostEventReturn'] > 0).sum()
        report.append(f"   - Positive post-event reactions: {pos_reactions} ({pos_reactions/len(ea)*100:.1f}%)")
        report.append(f"   - Average surprise magnitude: {ea['SurprisePercentage'].abs().mean():.2f}%")
        
        # Recommendations
        report.append("\n8. RECOMMENDATIONS")
        report.append("   - Focus on stocks with high earnings surprise magnitudes")
        report.append("   - Monitor volume spikes as a potential predictive indicator")
        report.append("   - Use ensemble methods for better prediction accuracy")
        report.append("   - Consider sector-specific analysis for improved insights")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if self.save_intermediate:
            with open(self.data_dir / 'final_report.txt', 'w') as f:
                f.write(report_text)
        
        logger.info(report_text)
        return report_text
    
    def run_pipeline(
        self,
        tickers: List[str],
        generate_visualizations: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Execute unified analysis pipeline and return in-memory artifacts.

        Args:
            tickers: List of stock tickers to analyze
            generate_visualizations: Whether to render and save visualization plots
            generate_report: Whether to synthesize insights report

        Returns:
            Dictionary containing pipeline instance and generated artifacts
        """
        logger.info("STARTING EARNINGS IMPACT ANALYSIS PIPELINE")
        logger.info(f"Analyzing: {', '.join(tickers)}")

        # Execute all steps passing in-memory DataFrames
        prices, earnings = self.step_1_collect_data(tickers)
        event_analysis = self.step_2_event_analysis(prices, earnings)
        features, labels = self.step_3_feature_engineering(event_analysis)
        model_comparison = self.step_4_train_models(features, labels)

        if generate_visualizations:
            self.step_5_visualizations(event_analysis, model_comparison)

        report_text = None
        if generate_report:
            report_text = self.step_6_insights_report(event_analysis, model_comparison)

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return {
            'pipeline': self,
            'price_data': prices,
            'earnings_data': earnings,
            'event_analysis': event_analysis,
            'features': features,
            'labels': labels,
            'model_trainer': self.model_trainer,
            'model_comparison': model_comparison,
            'report': report_text
        }

    def run_for_api(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Execute pipeline tailored for API server usage (omits filesystem plotting).

        Args:
            tickers: List of tickers to analyze

        Returns:
            Dictionary with DataFrames and models required by API state manager
        """
        return self.run_pipeline(tickers, generate_visualizations=False, generate_report=True)

    def run_complete_pipeline(self, tickers: List[str]) -> None:
        """
        Run the complete analysis pipeline (CLI mode).
        
        Args:
            tickers: List of stock tickers to analyze
        """
        try:
            self.run_pipeline(tickers, generate_visualizations=True, generate_report=True)
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    from src.config import load_config
    
    # Load configuration from config.ini, environment, and defaults
    config = load_config()
    tickers = config.get('tickers', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA'])
    
    # Run pipeline
    pipeline = EarningsAnalysisPipeline(config.to_dict())
    pipeline.run_complete_pipeline(tickers)

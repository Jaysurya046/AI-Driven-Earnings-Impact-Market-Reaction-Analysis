"""
Quick Start Example
Demonstrates how to use the earnings analysis system
"""

from main import EarningsAnalysisPipeline

# Define configuration
config = {
    'lookback_years': 5,
    'event_window': 5,
    'estimation_window': 60,
    'label_threshold': 0.0,
    'data_dir': 'data',
    'models_dir': 'models',
    'viz_dir': 'visualizations'
}

# Define tickers to analyze
# You can modify this list based on your interest
tickers = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL', # Google/Alphabet
    'AMZN',  # Amazon
    'TSLA',  # Tesla
]

# Initialize and run pipeline
if __name__ == "__main__":
    pipeline = EarningsAnalysisPipeline(config)
    
    # Run complete analysis
    pipeline.run_complete_pipeline(tickers)
    
    # Access results programmatically
    # print(pipeline.event_analysis)
    # print(pipeline.features)
    # print(pipeline.model_trainer.results)

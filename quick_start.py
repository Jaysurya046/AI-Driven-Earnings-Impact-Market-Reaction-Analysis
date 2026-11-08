"""
Quick Start Example
Demonstrates how to use the earnings analysis system
"""

from src import load_config
from main import EarningsAnalysisPipeline

# Define tickers to analyze
tickers = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL', # Google/Alphabet
    'AMZN',  # Amazon
    'TSLA',  # Tesla
]

if __name__ == "__main__":
    # Load unified configuration (config.ini, .env, and system environment variables)
    config = load_config()
    pipeline = EarningsAnalysisPipeline(config.to_dict())
    
    # Run complete analysis
    pipeline.run_complete_pipeline(tickers)

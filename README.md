# AI-Driven Earnings Impact & Market Reaction Analysis

A comprehensive data-driven system that analyzes stock price reactions around quarterly earnings announcements and uses machine learning to predict post-earnings market movement.

## Project Overview

This system:
- **Collects** historical stock price and earnings data using Yahoo Finance
- **Analyzes** price movements in event windows around earnings dates
- **Engineers** ML features from market behavior patterns
- **Trains** classification models to predict market reactions
- **Visualizes** results through comprehensive dashboards
- **Generates** actionable insights for investment strategies

## Project Structure

```
├── .github/workflows/
│   └── ci.yml                   # Automated CI workflow (linting, typing, pytest)
├── src/                          # Standardized core package
│   ├── __init__.py              # Package init and unified exports
│   ├── config.py                # Unified config loader (config.ini + .env)
│   ├── data_collection.py       # Yahoo Finance ingestion with retry & validation
│   ├── event_analysis.py        # Trading-day event study & CAR calculation
│   ├── feature_engineering.py   # Strict pre-announcement feature creation
│   ├── ml_models.py             # Classification, CV, tuning & calibration
│   ├── backtesting.py           # Realistic backtesting (slippage & commission)
│   ├── sector_comparison.py     # Sector-level impact analysis
│   ├── sentiment_analysis.py    # News sentiment extraction around earnings
│   ├── streamlit_app.py         # Interactive multi-tab web dashboard
│   ├── utils.py                 # Modern pandas utilities & helper functions
│   └── visualizations.py        # Chart generation and reporting
│
├── tests/                        # Comprehensive test suite (unit & integration)
├── data/                         # Cached market data and analysis outputs
├── models/                       # Model artifacts (.pkl) and companion metadata (.json)
├── visualizations/               # Generated analytical figures
├── notebooks/                    # Interactive research notebooks
│
├── api_server.py                 # Thread-safe FastAPI server with async workers
├── main.py                       # Orchestrated in-memory analysis pipeline
├── retraining_scheduler.py       # Scheduled retraining daemon
├── pyproject.toml                # PEP 517/621 packaging & test config
├── config.ini                    # Core configuration file
├── requirements.txt              # Primary project dependencies
├── requirements-lock.txt         # Pinned reproducible dependency lockfile
└── README.md                     # Project documentation
```

## Installation

### Prerequisites
- Python 3.9+ (Python 3.10–3.13 supported)
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   cd "AI-Driven Earnings Impact & Market Reaction Analysis"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install package & dependencies**
   ```bash
   # Standard install
   pip install -r requirements.txt

   # Editable package installation with dev tools
   pip install -e ".[dev]"
   ```

4. **Run Automated Test Suite**
   ```bash
   python -m pytest
   ```

## Quick Start

### Basic Usage

Run the complete analysis pipeline:

```bash
python quick_start.py
```

Or with custom configuration:

```bash
python main.py
```

### Example: Analyze Specific Stocks

```python
from main import EarningsAnalysisPipeline

config = {
    'lookback_years': 5,
    'event_window': 5,
    'estimation_window': 60,
    'label_threshold': 0.0,
    'data_dir': 'data',
    'models_dir': 'models',
    'viz_dir': 'visualizations'
}

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

pipeline = EarningsAnalysisPipeline(config)
pipeline.run_complete_pipeline(tickers)
```

## Key Features

### 1. Data Collection
- **Yahoo Finance Integration**: Fetches historical OHLCV data
- **Earnings Dates**: Retrieves quarterly earnings announcement dates
- **EPS Data**: Collects reported vs expected EPS for surprise calculation

### 2. Event-Based Analysis
- **Event Window**: Analyzes ±5 trading days around earnings (configurable)
- **Returns Calculation**: Pre-event vs post-event returns
- **Abnormal Returns**: Mean-adjusted abnormal return model
- **CAR (CAR)**: Cumulative abnormal returns across event window
- **Volatility Analysis**: Pre/post earnings volatility changes
- **Volume Analysis**: Volume spike detection and measurement

### 3. Feature Engineering
Features created include:
- Earnings surprise percentage and direction
- Pre-event volatility and returns
- Volatility change ratio
- Volume spike indicators
- Price momentum indicators
- Derived interaction features

### 4. Machine Learning Models

**Three Classification Models:**
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting with state-of-the-art performance

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- AUC-ROC Score

### 5. Visualizations

Generated dashboards include:
- **Earnings Surprise vs Market Reaction**: Scatter plot with trend line
- **Model Comparison**: Performance metrics across all models
- **Volatility Analysis**: Pre/during/post event volatility
- **Volume Analysis**: Trading volume around earnings
- **Feature Importance**: Top predictive features
- **Confusion Matrices**: Model prediction accuracy breakdown

### 6. Insights & Reporting

Automatic generation of:
- Executive summary with key statistics
- Correlation analysis between surprise and returns
- Best-performing model identification
- Actionable investment recommendations

## Configuration

Edit `config.ini` to customize:

```ini
[DATA_COLLECTION]
lookback_years = 5
tickers = AAPL, MSFT, GOOGL, AMZN, TSLA

[EVENT_ANALYSIS]
event_window = 5
estimation_window = 60

[MACHINE_LEARNING]
test_size = 0.2
random_state = 42
```

## Usage Examples

### Example 1: Full Pipeline with Default Settings
```python
from main import EarningsAnalysisPipeline

config = {
    'lookback_years': 5,
    'event_window': 5,
    'estimation_window': 60,
    'label_threshold': 0.0,
    'data_dir': 'data',
    'models_dir': 'models',
    'viz_dir': 'visualizations'
}

pipeline = EarningsAnalysisPipeline(config)
pipeline.run_complete_pipeline(['AAPL', 'MSFT', 'GOOGL'])
```

### Example 2: Step-by-Step Analysis
```python
pipeline = EarningsAnalysisPipeline(config)

# Step 1: Collect data
price_data, earnings_data = pipeline.step_1_collect_data(['AAPL', 'MSFT'])

# Step 2: Event analysis
event_analysis = pipeline.step_2_event_analysis()

# Step 3: Features
features, labels = pipeline.step_3_feature_engineering()

# Step 4: Models
model_comparison = pipeline.step_4_train_models()

# Step 5: Visualizations
pipeline.step_5_visualizations()

# Step 6: Report
report = pipeline.step_6_insights_report()
```

### Example 3: Custom Data Analysis
```python
from src.data_collection import DataCollector
from src.event_analysis import EventAnalyzer

collector = DataCollector(lookback_years=3)
price_data, earnings_data = collector.fetch_multiple_stocks(['TSLA', 'NVDA'])

analyzer = EventAnalyzer(event_window=3, estimation_window=90)
results = analyzer.analyze_multiple_events(price_data, earnings_data)

print(results[['Ticker', 'EarningsDate', 'PostEventReturn', 'AbnormalReturnCAR']])
```

## Output Files

After running the pipeline, you'll find:

### Data Files (`data/`)
- `raw_price_data.csv` - Historical price data
- `raw_earnings_data.csv` - Earnings dates and EPS data
- `event_analysis.csv` - Returns and metrics around earnings
- `engineered_features.csv` - ML features and labels
- `model_comparison.csv` - Model performance metrics
- `final_report.txt` - Analysis report

### Visualizations (`visualizations/`)
- `01_surprise_vs_reaction.png` - Earnings surprise correlation
- `02_model_comparison.png` - Model performance bars
- `03_volatility_analysis.png` - Volatility around earnings
- `04_volume_analysis.png` - Volume changes
- `05_feature_importance_*.png` - Top predictive features

### Models (`models/`)
- `logistic_regression.pkl` - Trained model
- `random_forest.pkl` - Trained model
- `xgboost.pkl` - Trained model

## Key Insights You'll Discover

1. **Earnings Surprise Correlation**: How EPS surprises impact stock prices
2. **Volatility Patterns**: Risk changes around earnings announcements
3. **Volume Spikes**: Trading activity patterns during events
4. **Market Mispricing**: Stocks with predictable post-earnings reactions
5. **Model Accuracy**: Which ML approach best predicts market reactions

## Customization

### Modify Event Window
```python
analyzer = EventAnalyzer(event_window=7)  # ±7 days instead of ±5
```

### Change Classification Threshold
```python
labels = feature_engineer.create_binary_labels(event_analysis, threshold=1.5)
```

### Train with Different Features
```python
feature_columns = ['EarningsSurpriseAbs', 'VolatilityChangeRatio', 'VolumeAnomalous']
X_train, X_test, y_train, y_test = feature_engineer.prepare_train_test_data(
    features, labels, feature_columns=feature_columns
)
```

## Requirements

See `requirements.txt` for complete list. Key packages:
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **yfinance**: Yahoo Finance API
- **scikit-learn**: ML models and utilities
- **xgboost**: Gradient boosting
- **matplotlib/seaborn**: Visualization

## Troubleshooting

### Issue: No earnings data found
**Solution**: Some stocks may not have earnings data on Yahoo Finance. Try other tickers.

### Issue: Insufficient data for event window
**Solution**: Increase `estimation_window` or use stocks with longer trading histories.

### Issue: Model performance is poor
**Solution**: 
- Try different event windows
- Adjust label threshold
- Use more historical data
- Test with different ticker groups

## Next Steps for Enhancement

1. **Sentiment Analysis**: Incorporate news sentiment around earnings
2. **Sector Analysis**: Compare reactions across different sectors
3. **Deep Learning**: Use LSTM networks for time-series prediction
4. **Real-time Prediction**: Deploy models for live market predictions
5. **Power BI Integration**: Connect to Power BI for interactive dashboards
6. **Database Storage**: Migrate to SQL database for scalability

## API Reference

### DataCollector
```python
collector = DataCollector(lookback_years=5)
price_data = collector.fetch_stock_price_data('AAPL')
earnings_data = collector.fetch_earnings_dates('AAPL')
price_df, earnings_df = collector.fetch_multiple_stocks(['AAPL', 'MSFT'])
```

### EventAnalyzer
```python
analyzer = EventAnalyzer(event_window=5, estimation_window=60)
results = analyzer.analyze_event_window(price_data, earnings_date, 'AAPL')
analysis_df = analyzer.analyze_multiple_events(price_data, earnings_data)
```

### FeatureEngineer
```python
engineer = FeatureEngineer()
features = engineer.create_features(event_analysis)
labels = engineer.create_binary_labels(event_analysis, threshold=0.0)
X_train, X_test, y_train, y_test = engineer.prepare_train_test_data(features, labels)
```

### ModelTrainer
```python
trainer = ModelTrainer()
comparison = trainer.train_all_models(X_train, X_test, y_train, y_test)
trainer.save_model('Random Forest', 'models/rf_model.pkl')
```

### Visualizer
```python
viz = Visualizer()
viz.plot_surprise_vs_reaction(event_analysis)
viz.plot_model_comparison(model_comparison)
viz.plot_feature_importance(importance_df, top_n=15)
```

## Contributing

To extend the system:

1. Add new features in `feature_engineering.py`
2. Implement new models in `ml_models.py`
3. Create new visualizations in `visualizations.py`
4. Update `main.py` to integrate new components

## License

This project is provided as-is for educational and research purposes.

## Support

For issues, questions, or suggestions:
1. Check troubleshooting section
2. Review configuration settings
3. Verify data availability
4. Check log messages for detailed error information

## Disclaimer

This analysis is for educational purposes. Past performance does not guarantee future results. Always conduct your own due diligence before making investment decisions. The predictions from ML models should not be used as sole basis for trading decisions.

---

**Created**: February 2026
**Python Version**: 3.8+
**Last Updated**: February 2026

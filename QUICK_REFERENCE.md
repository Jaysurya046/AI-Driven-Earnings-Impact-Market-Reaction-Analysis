# Quick Reference Cheat Sheet

## Installation (One Command)
```bash
pip install -r requirements.txt
```

## Running the System

### Quick Start (2 minutes)
```bash
python quick_start.py
```

### Full Analysis
```bash
python main.py
```

### Interactive Notebook
```bash
jupyter notebook notebooks/earnings_analysis.ipynb
```

---

## Common Tasks

### Analyze Specific Stocks
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

### Get Event Analysis Results
```python
from src.data_collection import DataCollector
from src.event_analysis import EventAnalyzer

collector = DataCollector(lookback_years=5)
price_data, earnings_data = collector.fetch_multiple_stocks(['AAPL', 'MSFT'])

analyzer = EventAnalyzer(event_window=5, estimation_window=60)
results = analyzer.analyze_multiple_events(price_data, earnings_data)

print(results[['Ticker', 'PostEventReturn', 'AbnormalReturnCAR']])
```

### Train Models
```python
from src.feature_engineering import FeatureEngineer
from src.ml_models import ModelTrainer

# Prepare data
engineer = FeatureEngineer()
features = engineer.create_features(event_analysis)
labels = engineer.create_binary_labels(event_analysis)

X_train, X_test, y_train, y_test = engineer.prepare_train_test_data(
    features, labels
)

# Train models
trainer = ModelTrainer()
comparison = trainer.train_all_models(X_train, X_test, y_train, y_test)
print(comparison)
```

### Create Visualizations
```python
from src.visualizations import Visualizer

viz = Visualizer()
viz.plot_surprise_vs_reaction(event_analysis)
viz.plot_model_comparison(model_comparison)
viz.plot_feature_importance(importance_df, top_n=15)
```

---

## Key Classes & Methods

### DataCollector
```python
from src.data_collection import DataCollector

collector = DataCollector(lookback_years=5)
price_data = collector.fetch_stock_price_data('AAPL')
earnings_data = collector.fetch_earnings_dates('AAPL')
price_df, earnings_df = collector.fetch_multiple_stocks(['AAPL', 'MSFT'])
```

### EventAnalyzer
```python
from src.event_analysis import EventAnalyzer

analyzer = EventAnalyzer(event_window=5, estimation_window=60)
results = analyzer.analyze_event_window(price_data, earnings_date, 'AAPL')
analysis_df = analyzer.analyze_multiple_events(price_data, earnings_data)
```

### FeatureEngineer
```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_features(event_analysis)
labels = engineer.create_binary_labels(event_analysis, threshold=0.0)
X_train, X_test, y_train, y_test = engineer.prepare_train_test_data(
    features, labels, test_size=0.2
)
```

### ModelTrainer
```python
from src.ml_models import ModelTrainer

trainer = ModelTrainer()
comparison = trainer.train_all_models(X_train, X_test, y_train, y_test)
trainer.save_model('Random Forest', 'models/rf.pkl')
trainer.load_model('models/rf.pkl', 'Random Forest')
```

### Visualizer
```python
from src.visualizations import Visualizer

viz = Visualizer()
viz.plot_surprise_vs_reaction(event_analysis, save_path='viz/surprise.png')
viz.plot_model_comparison(model_comparison)
viz.plot_feature_importance(importance_df, top_n=15)
viz.plot_confusion_matrix(cm, model_name='XGBoost')
```

---

## Data Formats

### Price Data (raw_price_data.csv)
```
Date,Open,High,Low,Close,Adj Close,Volume,Ticker
2023-01-03,150.93,151.93,150.15,150.63,150.63,15648900,AAPL
```

### Earnings Data (raw_earnings_data.csv)
```
Date,EPS,Surprise_Percentage,Ticker
2023-01-27,1.72,0.0,AAPL
```

### Event Analysis (event_analysis.csv)
```
Ticker,EarningsDate,PreEventReturn,PostEventReturn,AbnormalReturnCAR,PreEventVolatility,...
AAPL,2023-01-27,1.23,2.45,1.87,23.4,...
```

### Features (engineered_features.csv)
```
Ticker,EarningsDate,EarningsSurpriseAbs,VolatilityChangeRatio,VolumeSpikeRatio,...
AAPL,2023-01-27,0.0,1.23,1.45,...
```

---

## Configuration File (config.ini)

```ini
[DATA_COLLECTION]
lookback_years = 5
tickers = AAPL, MSFT, GOOGL, AMZN, TSLA

[EVENT_ANALYSIS]
event_window = 5
estimation_window = 60

[FEATURE_ENGINEERING]
label_threshold = 0.0

[MACHINE_LEARNING]
test_size = 0.2
random_state = 42

[DIRECTORIES]
data_dir = data
models_dir = models
viz_dir = visualizations
```

---

## Output Files Created

| File | Location | Purpose |
|------|----------|---------|
| raw_price_data.csv | data/ | Historical prices |
| raw_earnings_data.csv | data/ | Earnings dates & EPS |
| event_analysis.csv | data/ | Analysis results |
| engineered_features.csv | data/ | ML features |
| model_comparison.csv | data/ | Model metrics |
| final_report.txt | data/ | Summary report |
| logistic_regression.pkl | models/ | Trained model |
| random_forest.pkl | models/ | Trained model |
| xgboost.pkl | models/ | Trained model |
| 01_surprise_vs_reaction.png | visualizations/ | Chart |
| 02_model_comparison.png | visualizations/ | Chart |
| 03_volatility_analysis.png | visualizations/ | Chart |
| 04_volume_analysis.png | visualizations/ | Chart |
| 05_feature_importance_*.png | visualizations/ | Chart |

---

## Key Metrics Explained

### Returns
- **PreEventReturn**: Return from day -5 to day -1
- **PostEventReturn**: Return from day +1 to day +5
- Formula: `(Price_final - Price_initial) / Price_initial * 100`

### Abnormal Returns
- **AbnormalReturnCAR**: Cumulative abnormal return
- Formula: `(1 + AR1/100) × (1 + AR2/100) × ... - 1`
- Shows unexpected market reaction

### Volatility
- **VolatilityChangeRatio**: PostVolatility / EstimationVolatility
- Shows increase in risk
- Value > 1 = increased volatility

### Volume
- **VolumeSpikeRatio**: PostVolume / EstimationVolume
- Shows trading activity increase
- Value > 1.5 = significant spike

---

## Model Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | % correct predictions |
| Precision | TP/(TP+FP) | % positive predictions correct |
| Recall | TP/(TP+FN) | % actual positives found |
| F1-Score | 2×(P×R)/(P+R) | Harmonic mean of P&R |
| AUC-ROC | Area under curve | Overall discrimination ability |

**Legend**: TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No earnings data | Use different stock (try AAPL, MSFT) |
| Insufficient data | Reduce event_window or increase lookback_years |
| Import error | Run `pip install -r requirements.txt` |
| Low accuracy | Try different threshold or more stocks |
| Memory error | Reduce number of stocks or years |
| Plot not showing | Add `%matplotlib inline` in Jupyter |

---

## Performance Tips

### Speed Up Analysis
```python
# Use fewer stocks
tickers = ['AAPL', 'MSFT']  # Instead of 10 stocks

# Reduce lookback period
config['lookback_years'] = 2  # Instead of 5

# Reduce event window
config['event_window'] = 3  # Instead of 5
```

### Improve Model Accuracy
```python
# Use more stocks (20+)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', ...]

# Increase data period
config['lookback_years'] = 5

# Engineer more features
# Add custom features to FeatureEngineer class
```

---

## File Locations Quick Access

```
Project Root: c:\Users\jayas\OneDrive\Desktop\
             AI-Driven Earnings Impact & Market Reaction Analysis\

Python Files: src/
Data Output: data/
Models: models/
Visualizations: visualizations/
Notebooks: notebooks/
Documentation: README.md, GETTING_STARTED.md, ARCHITECTURE.md
Config: config.ini
Main Script: main.py
Quick Start: quick_start.py
```

---

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick analysis
python quick_start.py

# Run full analysis
python main.py

# Open Jupyter notebook
jupyter notebook notebooks/earnings_analysis.ipynb

# Check Python version
python --version

# List installed packages
pip list

# Update packages
pip install --upgrade -r requirements.txt
```

---

## Learning Path

1. **Start**: Run `python quick_start.py`
2. **Explore**: Check outputs in data/, visualizations/
3. **Understand**: Read GETTING_STARTED.md
4. **Learn**: Open and run Jupyter notebook
5. **Deep Dive**: Read ARCHITECTURE.md
6. **Customize**: Modify config.ini and parameters
7. **Extend**: Add new features or models
8. **Deploy**: Use trained models for predictions

---

## Important Concepts

### Event Window
- Definition: ±N trading days around earnings
- Default: ±5 days (configurable)
- Purpose: Analyze price reaction to announcement

### Abnormal Returns
- Definition: Actual return minus expected return
- Method: Mean-adjusted model
- Interpretation: Unexpected market reaction

### CAR (Cumulative Abnormal Return)
- Definition: Product of (1 + abnormal return)
- Purpose: Total unexpected return over period
- Interpretation: Net market surprise effect

### Feature Importance
- Definition: How much each feature helps predictions
- Source: Model-specific (coefficients, importances)
- Usage: Understand what drives market reactions

---

## Getting Help

1. **Configuration Issues**: Check config.ini and GETTING_STARTED.md
2. **Code Questions**: Review docstrings and comments in src/
3. **Methodology**: Read ARCHITECTURE.md for mathematical details
4. **Examples**: Check quick_start.py and notebooks/
5. **API Reference**: See README.md for complete documentation

---

## Project Statistics

- **Total Python Code**: ~1,500 lines
- **Documentation**: ~1,200 lines
- **Modules**: 6 core analysis modules
- **Classes**: 6 main classes + utilities
- **Features Engineered**: 15+ per event
- **Models Trained**: 3 (LogReg, RF, XGB)
- **Visualizations**: 7 types
- **Test Stocks**: 5-10 recommended
- **Execution Time**: 1-2 minutes (5 stocks, 5 years)

---

**For more information, see the complete documentation files!**

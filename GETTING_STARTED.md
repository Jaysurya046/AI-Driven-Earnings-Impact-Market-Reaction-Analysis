# Getting Started Guide

## Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Quick Analysis
```bash
python quick_start.py
```

This will analyze 5 major tech stocks and generate all outputs.

---

## Complete Setup (10 minutes)

### Step 1: Configure Your Analysis
Edit `config.ini` to customize:
- Stock tickers to analyze
- Event window (±5 trading days is default)
- Historical lookback period
- Output directories

### Step 2: Run Full Pipeline
```bash
python main.py
```

The pipeline will automatically:
1. Download historical data
2. Fetch earnings announcement dates
3. Analyze price movements
4. Engineer features
5. Train ML models
6. Generate visualizations
7. Create comprehensive report

### Step 3: Explore Results
All outputs saved in:
- `data/` - Raw and processed data files
- `models/` - Trained model files
- `visualizations/` - Generated charts
- `notebooks/` - Interactive Jupyter notebook

---

## Using the Interactive Notebook

### Open and Run Notebook
```bash
jupyter notebook notebooks/earnings_analysis.ipynb
```

The notebook allows you to:
- Run analysis step-by-step
- Modify parameters interactively
- Explore data with custom visualizations
- Train models with different settings
- Understand the methodology

### Key Notebook Sections
1. **Import Libraries** - Load all required packages
2. **Fetch Data** - Download stock prices and earnings data
3. **Event Analysis** - Calculate returns and abnormal returns
4. **Feature Engineering** - Create ML features
5. **Model Training** - Train classification models
6. **Evaluation** - Compare model performance
7. **Visualization** - Create insights and charts

---

## Understanding the System

### Architecture Overview

```
Input Data (Yahoo Finance)
         |
         v
DataCollector (data_collection.py)
         |
         v
EventAnalyzer (event_analysis.py)
    - Calculate returns
    - Compute abnormal returns
    - Measure volatility changes
         |
         v
FeatureEngineer (feature_engineering.py)
    - Create feature set
    - Generate labels
    - Prepare for ML
         |
         v
ModelTrainer (ml_models.py)
    - Train 3 models
    - Evaluate performance
    - Compare results
         |
         v
Visualizer (visualizations.py)
    - Create dashboards
    - Generate charts
    - Export insights
         |
         v
Output (data/, models/, visualizations/)
```

### Core Concepts

**Event Window**: ±5 trading days around earnings announcement
- Day -5 to Day -1: Pre-event period
- Day 0: Earnings announcement day
- Day +1 to Day +5: Post-event period

**Abnormal Returns**: Actual returns minus expected returns
- Measures unexpected market reaction
- Uses 60-day estimation window

**Features**: Input variables for ML models
- Earnings surprise %
- Volatility changes
- Volume spikes
- Price momentum
- Interaction terms

**Labels**: Target variable (positive/negative reaction)
- 1 = Post-event return > threshold
- 0 = Post-event return ≤ threshold

---

## Customization Examples

### Example 1: Analyze Different Stocks
```python
from main import EarningsAnalysisPipeline

config = {
    'lookback_years': 3,
    'event_window': 3,
    'estimation_window': 60,
    'label_threshold': 1.0,  # Only very positive reactions
    'data_dir': 'data',
    'models_dir': 'models',
    'viz_dir': 'visualizations'
}

tickers = ['JPM', 'GS', 'BAC']  # Banking stocks

pipeline = EarningsAnalysisPipeline(config)
pipeline.run_complete_pipeline(tickers)
```

### Example 2: Longer Event Window
```python
from src.event_analysis import EventAnalyzer

analyzer = EventAnalyzer(event_window=10, estimation_window=90)
results = analyzer.analyze_multiple_events(price_data, earnings_data)
```

### Example 3: Custom Features
```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_features(event_analysis)

# Add custom feature
features['CustomFeature'] = (
    features['EarningsSurpriseAbs'] * 
    features['VolatilityChangeRatio']
)
```

### Example 4: Different ML Models
```python
from src.ml_models import ModelTrainer

trainer = ModelTrainer()

# Train only specific models
trainer.train_logistic_regression(X_train, y_train)
trainer.train_random_forest(X_train, y_train, n_estimators=200)
```

---

## Understanding the Outputs

### Data Files

**raw_price_data.csv**
- Historical OHLCV data
- 250+ trading days per stock per year
- Used for analysis and feature calculation

**raw_earnings_data.csv**
- Earnings announcement dates
- Reported vs expected EPS
- Used to identify event windows

**event_analysis.csv**
- Returns before and after earnings
- Cumulative abnormal returns (CAR)
- Volatility and volume metrics
- Ready for ML modeling

**engineered_features.csv**
- All ML features
- Labels (0 = negative, 1 = positive)
- Ready for model training

**model_comparison.csv**
- Performance metrics for all 3 models
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC scores

### Visualization Files

**01_surprise_vs_reaction.png**
- Scatter plot of earnings surprise vs post-event returns
- Shows market efficiency
- Trend line indicates pattern strength

**02_model_comparison.png**
- Bar charts comparing model performance
- Helps identify best model
- Shows metric trade-offs

**03_volatility_analysis.png**
- Pre/during/post event volatility
- Shows risk increase
- Indicates event significance

**04_volume_analysis.png**
- Volume spikes around earnings
- Identifies trading activity patterns
- Measures market engagement

**05_feature_importance_*.png**
- Top predictive features
- Shows what drives predictions
- Helps interpret model decisions

### Final Report

**final_report.txt**
- Executive summary
- Key statistics
- Findings and conclusions
- Investment recommendations

---

## Troubleshooting

### Issue: "No data found" for a stock
**Cause**: Stock may not have earnings data on Yahoo Finance
**Solution**: Use different tickers (AAPL, MSFT, GOOGL work well)

### Issue: "Insufficient data for event window"
**Cause**: Not enough historical data before/after event
**Solution**: 
- Increase estimation_window
- Decrease event_window
- Use more recent stocks with longer history

### Issue: Low model accuracy
**Cause**: Market reactions may be unpredictable or random
**Solution**:
- Try different thresholds
- Use different stocks
- Increase feature engineering
- Try ensemble methods

### Issue: "Import error" when running
**Cause**: Missing dependencies
**Solution**:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Plots not showing in Jupyter
**Cause**: Matplotlib backend not configured
**Solution**: Add at start of notebook:
```python
%matplotlib inline
```

---

## Performance Tips

### For Faster Analysis
- Reduce `lookback_years` (e.g., 2 instead of 5)
- Reduce number of stocks
- Use `quick_start.py` instead of `main.py`

### For Better Results
- Use 5+ years of data
- Include 20+ stocks from different sectors
- Increase `estimation_window` (e.g., 90 days)
- Engineer domain-specific features

### For Production Use
- Save trained models: `trainer.save_model('Random Forest', 'model.pkl')`
- Export data to database (not just CSV)
- Implement real-time monitoring
- Set up automated retraining

---

## Next Steps

1. **Run Analysis**: Execute `python quick_start.py`
2. **Explore Results**: Check `data/` and `visualizations/` folders
3. **Review Notebook**: Open and run `notebooks/earnings_analysis.ipynb`
4. **Customize**: Modify `config.ini` with your stocks
5. **Extend**: Add custom features or models

---

## Resources

### Documentation
- See [README.md](README.md) for complete documentation
- Check comments in source code for implementation details
- Review this guide for getting started

### Key Files
- [main.py](main.py) - Main analysis pipeline
- [quick_start.py](quick_start.py) - Simple example
- [config.ini](config.ini) - Configuration settings
- [notebooks/earnings_analysis.ipynb](notebooks/earnings_analysis.ipynb) - Interactive notebook

### Data Sources
- **Prices**: Yahoo Finance API (via yfinance)
- **Earnings**: Yahoo Finance earnings dates
- **Features**: Calculated from price/volume data

---

## Support & Help

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Review the README.md for detailed API docs
3. Check source code comments for implementation details
4. Verify all dependencies are installed
5. Ensure valid stock tickers are used

---

## Tips for Best Results

1. **Use Quality Data**
   - Prefer large-cap stocks (have more data)
   - Use stocks with consistent earnings announcements
   - Verify data completeness

2. **Understand the Metrics**
   - CAR = cumulative abnormal returns
   - Volatility = std dev of daily returns
   - Volume spike = post-event volume / estimation volume

3. **Interpret Results**
   - Model accuracy reflects market efficiency
   - Feature importance shows market drivers
   - Visualizations reveal patterns

4. **Validate Findings**
   - Test on out-of-sample data
   - Compare across multiple stocks
   - Consider sector-specific effects

---

**Happy analyzing!** 📊📈

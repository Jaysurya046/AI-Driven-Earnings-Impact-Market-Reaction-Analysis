# Quick Reference Cheat Sheet

## Installation & Setup

```bash
# Production install
pip install -r requirements.txt

# Editable package install with development & testing dependencies
pip install -e ".[dev]"

# Configure environment variables & secrets
cp .env.example .env

# Run all 68 automated tests
python -m pytest
```

---

## Running the Platform

### End-to-End Pipeline (CLI)
```bash
python main.py
```

### Interactive Streamlit Dashboard
```bash
streamlit run src/streamlit_app.py
```

### High-Throughput REST API (FastAPI)
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
# Interactive documentation: http://localhost:8000/docs
```

### Automated Retraining Scheduler
```bash
# Run one immediate retraining pass
python retraining_scheduler.py --run-now

# Run continuous background daemon
python retraining_scheduler.py --continuous
```

---

## Core Classes & Modern API Signatures

### 1. Unified Configuration (`src.config`)
```python
from src import load_config, mask_secrets

# Loads from .env, env vars, config.ini, and defaults
config = load_config(env_file=".env")
masked = mask_secrets(config.to_dict())  # Automatically masks secrets
print(config['tickers'])
```

### 2. Data Collection & Shape Validation (`src.data_collection`)
```python
from src import DataCollector

collector = DataCollector(lookback_years=5, max_retries=3)
price_df, earnings_df = collector.fetch_multiple_stocks(['AAPL', 'NVDA', 'MSFT'])

# Validate shapes and required schemas
DataCollector.validate_collected_data(price_df, earnings_df)
```

### 3. Trading-Day Event Studies (`src.event_analysis`)
```python
from src import EventAnalyzer

analyzer = EventAnalyzer(event_window=5, estimation_window=60)
# Uses exact trading-day index offsets (handles weekends & after-hours)
event_df = analyzer.analyze_multiple_events(price_df, earnings_df)
```

### 4. Zero-Leakage Feature Engineering (`src.feature_engineering`)
```python
from src import FeatureEngineer, SAFE_FEATURE_COLUMNS

engineer = FeatureEngineer()
# Extracts pre-announcement features only (zero look-ahead bias)
features_df, feature_cols = engineer.create_features(event_df, return_feature_names=True)
labels = engineer.create_binary_labels(event_df, threshold=0.0)

# Chronological time-series train/test split
X_train, X_test, y_train, y_test, cols = engineer.prepare_train_test_data(
    features_df, labels, chronological=True
)
```

### 5. Machine Learning, CV & Metadata Persistence (`src.ml_models`)
```python
from src import ModelTrainer

trainer = ModelTrainer()

# Cross-validation with 95% confidence intervals
cv_res = trainer.evaluate_cv(trainer.models['random_forest'], X_train, y_train, cv=5)
print("95% CI Accuracy:", cv_res['accuracy_ci_95'])

# Hyperparameter optimization (GridSearchCV within CV folds)
best_model, best_params = trainer.tune_hyperparameters('xgboost', X_train, y_train, cv=3)

# Probability calibration (CalibratedClassifierCV)
calibrated_model = trainer.calibrate_probabilities(best_model, X_test, y_test)

# Saves model weights (.pkl) and companion metadata (.json)
trainer.save_model(
    'xgboost',
    'models/xgboost.pkl',
    feature_columns=feature_cols,
    metadata={'best_params': best_params}
)
```

### 6. Realistic Backtesting with Market Frictions (`src.backtesting`)
```python
from src import Backtester

backtester = Backtester(
    initial_capital=100000.0,
    commission_rate=0.001,      # 0.1% broker commission
    slippage_bps=5.0,           # 5 basis points bid-ask slippage
    next_day_open=True          # Next-day open execution timing
)
perf = backtester.run_momentum_strategy(event_df)
print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
```

### 7. Sector Comparison & Sentiment Analysis
```python
from src import SectorAnalyzer, SentimentAnalyzer

# Sector comparisons
sector_analyzer = SectorAnalyzer()
sector_summary = sector_analyzer.aggregate_by_sector(event_df)

# News sentiment extraction
sentiment_analyzer = SentimentAnalyzer(method='textblob')
score = sentiment_analyzer.analyze_text("Company reports strong Q3 beat and raises guidance")
```

---

## REST API Endpoints Quick Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Healthcheck and model readiness probe |
| `POST` | `/analyze` | Offload background pipeline execution (thread-safe) |
| `POST` | `/predict` | Predict post-earnings probability from pre-announcement metrics |
| `GET` | `/metrics` | Retrieve model evaluation metrics & confidence intervals |
| `GET` | `/models` | List loaded models and serialized feature columns |
| `GET` | `/status` | Check state manager progress and active worker tasks |

---

## Key Output Artifacts

| Artifact | Location | Description |
|---|---|---|
| Trained Models | `models/*.pkl` | Serialized estimator weights |
| Model Metadata | `models/*.json` | Feature columns, scalers, and library versions |
| Raw Prices | `data/raw_price_data.csv` | OHLCV historical price series |
| Raw Earnings | `data/raw_earnings_data.csv` | Announcement dates & EPS surprises |
| Event Metrics | `data/event_analysis.csv` | Pre/post returns, abnormal returns, CAR |
| Feature Set | `data/engineered_features.csv` | Pre-event features & binary classification labels |
| Visualizations | `visualizations/*.png` | High-res charts (scatter, model comparison, feature importance) |
| Summary Report | `data/final_report.txt` | Executive summary and quantitative findings |

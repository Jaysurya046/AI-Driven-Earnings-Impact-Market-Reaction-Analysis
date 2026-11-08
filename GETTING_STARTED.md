# Getting Started Guide

## Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
# Production dependencies
pip install -r requirements.txt

# Or editable install with test & dev dependencies
pip install -e ".[dev]"
```

### Step 2: Configure Secrets & Environment
```bash
cp .env.example .env
```
Customize `API_SECRET_KEY`, ticker lists, or port preferences in `.env` if desired.

### Step 3: Verify the Test Suite
```bash
python -m pytest
```
Runs all 68 automated unit, integration, and security tests to ensure the platform is functioning correctly.

### Step 4: Run Sanity Analysis
```bash
python quick_start.py
```
This runs a rapid in-memory analysis on 5 liquid stocks and verifies data ingestion, event studies, and model training.

---

## Complete Setup & Operational Workflows

### 1. Configure Your Quantitative Run
Customize settings either in `.env` or in `config.ini`:
- `tickers`: Stock universe to analyze (defaults to 20 liquid US large caps).
- `event_window`: Trading days around announcement (default: ±5 trading days).
- `estimation_window`: Historical baseline window (default: 60 trading days).
- `lookback_years`: Historical price history length (default: 5 years).

### 2. Run Full Quantitative Pipeline (CLI)
```bash
python main.py
```
The orchestrated pipeline will execute in-memory:
1. **Data Collection**: Ingests historical OHLCV and quarterly earnings via Yahoo Finance with MultiIndex normalization and exponential backoff retry logic.
2. **Event Study**: Maps earnings dates to trading days and computes Cumulative Abnormal Returns (CAR) against historical mean returns.
3. **Feature Engineering**: Calculates pure pre-announcement features (`SAFE_FEATURE_COLUMNS`) and performs a chronological time-series split.
4. **Model Training & Evaluation**: Trains Logistic Regression, Random Forest, and unscaled XGBoost; runs CV with 95% confidence intervals; and serializes `.pkl` models and `.json` companion metadata.
5. **Backtesting**: Simulates momentum and mean-reversion strategies under realistic commission friction and slippage basis points.
6. **Reporting**: Generates summary statistics and high-resolution charts in `visualizations/`.

### 3. Launch Interactive Streamlit Research Dashboard
```bash
streamlit run src/streamlit_app.py
```
Explore the interactive UI across multiple analytical tabs:
- **Overview Dashboard**: Earnings surprise correlations, distribution plots, and CAR charts.
- **ML Model Performance**: ROC curves, F1-scores, and feature importance rankings.
- **Sector Analysis**: GICS sector-level comparisons and t-test statistics.
- **Quantitative Backtest**: Strategy portfolio equity curves and drawdown metrics.

### 4. Start the High-Throughput REST API
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive OpenAPI documentation at `http://localhost:8000/docs`.
- Real-time prediction: `POST /predict`
- Background asynchronous pipeline run: `POST /analyze`
- System state inspection: `GET /status`

---

## Core Methodological Concepts

### 1. Trading-Day Exact Event Studies
- Event window: $[-5, +5]$ trading days around earnings releases.
- If an earnings report occurs after hours or over a weekend, the announcement day ($t=0$) is mapped to the immediate subsequent market session.
- Pre-event baseline window: $[-60, -10]$ trading days computes expected baseline return and volatility.

### 2. Zero-Leakage Feature Isolation
- Model inputs use only data known prior to the announcement date.
- Post-event returns (`PostEventMomentum`, `PreVol_PostReturn`) are strictly excluded to ensure realistic evaluation without look-ahead bias.

### 3. Chronological Time-Series Partitioning
- Avoids random train-test splits that cause future information leakage.
- Earliest 80% of events form the training set; latest 20% form the out-of-sample test set.

### 4. Realistic Backtest Frictions
- **Broker Commissions**: Configurable fee percentage (e.g. 0.1%).
- **Market Slippage**: Basis-point execution penalty (e.g. 5 bps) reflecting bid-ask spread costs.
- **Next-Day Execution**: Orders execute at the opening price of the next trading day.

---

## Programmatic Usage Examples

### Example 1: In-Memory Pipeline Execution
```python
from src.config import load_config
from main import EarningsAnalysisPipeline

config = load_config()
pipeline = EarningsAnalysisPipeline(config.to_dict())

# In-memory execution without mandatory disk round-trips
price_df, earnings_df = pipeline.step_1_collect_data(['AAPL', 'MSFT', 'GOOGL'])
event_df = pipeline.step_2_event_analysis(price_df, earnings_df)
features_df, labels = pipeline.step_3_feature_engineering(event_df)
model_comparison = pipeline.step_4_train_models(features_df, labels)
```

### Example 2: Model Training with Cross-Validation & Metadata Saving
```python
from src.ml_models import ModelTrainer

trainer = ModelTrainer()
# Cross-validation with 95% confidence intervals
cv_report = trainer.evaluate_cv(trainer.models['random_forest'], X_train, y_train, cv=5)
print("95% CI Accuracy:", cv_report['accuracy_ci_95'])

# Save model pickle with companion metadata JSON
trainer.save_model(
    'random_forest',
    'models/rf_model.pkl',
    feature_columns=feature_cols,
    metadata={'cv_report': cv_report}
)
```

### Example 3: Backtesting with Slippage & Commissions
```python
from src.backtesting import Backtester

backtester = Backtester(
    initial_capital=100000.0,
    commission_rate=0.001,      # 10 bps broker commission
    slippage_bps=5.0,           # 5 bps slippage
    next_day_open=True
)
results = backtester.run_momentum_strategy(event_df)
print(f"Total Return: {results['total_return']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

---

## Troubleshooting Guide

### Issue: "Yahoo Finance MultiIndex column format error"
- **Resolution**: Fully resolved in `DataCollector` via automatic multi-level column flattening and timezone normalization.

### Issue: "Data leakage or artificially inflated test accuracy"
- **Resolution**: Fixed in `src/feature_engineering.py`. All post-event features are blocked via `SAFE_FEATURE_COLUMNS` and evaluated strictly on chronologically held-out test splits.

### Issue: "FastAPI JSON serialization error on NaN floats"
- **Resolution**: Handled in `api_server.py` via `_safe_float()`, which sanitizes IEEE 754 `NaN` values before response delivery.

### Issue: "Concurrent API calls collision"
- **Resolution**: Handled by `AnalysisStateManager` using `asyncio.Lock()`, queuing parallel requests safely.

---

## Running the Automated Test Suite

```bash
# Run all 68 tests
python -m pytest

# Run with verbose output and short tracebacks
python -m pytest -v --tb=short

# Run specific domain test
python -m pytest tests/test_feature_leakage.py
```

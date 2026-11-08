# AI-Driven Earnings Impact & Market Reaction Analysis

A quantitative, production-grade system that analyzes equity price behavior around quarterly corporate earnings announcements and leverages machine learning to predict post-earnings market reactions.

[![CI](https://github.com/Jaysurya046/ai-earnings-impact-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Jaysurya046/ai-earnings-impact-analysis/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## System Capabilities

- **Resilient Market Data Ingestion**: Yahoo Finance connector with MultiIndex column normalization, timezone handling, future earnings date filtering, and exponential backoff retry logic.
- **Trading-Day Event Studies**: Exact calendar-to-trading-day alignment for event windows ($[-5, +5]$ days) and estimation windows (60 days), calculating Cumulative Abnormal Returns (CAR) using market-adjusted benchmarks.
- **Zero-Leakage Feature Engineering**: Strict pre-announcement feature boundaries (`SAFE_FEATURE_COLUMNS`) ensuring no look-ahead contamination from post-announcement returns or volume.
- **Rigorous ML Evaluation**: TimeSeriesSplit chronological validation to prevent look-ahead bias, cross-validation with 95% confidence intervals, tree-model scaling safety, and hyperparameter tuning (`GridSearchCV`) with probability calibration (`CalibratedClassifierCV`).
- **Realistic Backtesting**: Quantitative simulation of momentum and mean-reversion strategies accounting for broker commissions, bid-ask slippage (basis points), and next-day open execution timing.
- **Thread-Safe REST API**: FastAPI server with async concurrency locks (`AnalysisStateManager`), non-blocking background workers, sanitized JSON float serialization, and secured CORS.
- **Standardized Packaging & Hygiene**: PEP 517/621 `pyproject.toml`, reproducible `requirements-lock.txt`, secret management with `.env.example`, and automated GitHub Actions CI.

---

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
│   ├── lstm_models.py           # Deep learning LSTM sequence predictors
│   ├── utils.py                 # Modern pandas utilities & helper functions
│   └── visualizations.py        # Chart generation and reporting
│
├── tests/                        # Comprehensive test suite (68 unit & integration tests)
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
├── .env.example                  # Environment configuration & secrets template
├── requirements.txt              # Primary project dependencies
├── requirements-lock.txt         # Pinned reproducible dependency lockfile
└── README.md                     # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+ (Python 3.10–3.13 recommended)
- Git & pip

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Jaysurya046/ai-earnings-impact-analysis.git
cd "ai-earnings-impact-analysis"

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# Production install
pip install -r requirements.txt

# Or editable install with development & testing tools
pip install -e ".[dev]"
```

### 3. Environment Secrets Configuration
```bash
# Copy secrets template to active .env
cp .env.example .env
```
Edit `.env` to specify your `API_SECRET_KEY`, preferred port, or custom ticker list.

### 4. Run Test Suite
```bash
python -m pytest
```
All 68 tests across data collection, event analysis, feature isolation, ML training, backtesting, and API concurrency will run.

---

## Running the Platform

### 1. End-to-End Analysis Pipeline (CLI)
Run the full in-memory quantitative pipeline:
```bash
# Run with defaults or config.ini overrides
python main.py
```
To run a fast programmatic trial on selected equities:
```python
from src.config import load_config
from main import EarningsAnalysisPipeline

config = load_config()
pipeline = EarningsAnalysisPipeline(config.to_dict())

# Runs fully in-memory without mandatory disk round-trips
price_df, earnings_df = pipeline.step_1_collect_data(['AAPL', 'MSFT', 'NVDA'])
event_df = pipeline.step_2_event_analysis(price_df, earnings_df)
features_df, labels = pipeline.step_3_feature_engineering(event_df)
comparison = pipeline.step_4_train_models(features_df, labels)
```

### 2. Interactive Streamlit Dashboard
Launch the multi-tab exploratory research dashboard:
```bash
streamlit run src/streamlit_app.py
```
Features:
- **Overview Dashboard**: Surprise distribution vs CAR reactions and cumulative returns.
- **ML Performance**: Accuracy, F1, ROC-AUC comparisons and feature importance rankings.
- **Sector Comparison**: Sector-by-sector reaction heatmaps and t-test statistics.
- **Backtesting**: Portfolio equity curves under realistic commission & slippage friction.

### 3. REST API Server (FastAPI)
Start the high-throughput, thread-safe asynchronous prediction server:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at `http://localhost:8000/docs`.

Key Endpoints:
- `GET /health`: Service health and model status
- `POST /analyze`: Asynchronous background pipeline execution
- `POST /predict`: Real-time post-earnings movement probability prediction
- `GET /metrics`: Current model evaluation metrics and confidence intervals
- `GET /status`: State manager progress and running background job status

---

## Core Modules & API Reference

### Data Collection (`src.data_collection`)
```python
from src import DataCollector

collector = DataCollector(lookback_years=5, max_retries=3)
price_df, earnings_df = collector.fetch_multiple_stocks(['AAPL', 'NVDA', 'GOOGL'])
DataCollector.validate_collected_data(price_df, earnings_df)
```

### Event Study Analysis (`src.event_analysis`)
```python
from src import EventAnalyzer

analyzer = EventAnalyzer(event_window=5, estimation_window=60)
# Uses trading-day offsets and mean-adjusted abnormal returns
analysis_df = analyzer.analyze_multiple_events(price_df, earnings_df)
```

### Feature Engineering (`src.feature_engineering`)
```python
from src import FeatureEngineer, SAFE_FEATURE_COLUMNS

engineer = FeatureEngineer()
# Extracts only pre-announcement information (zero post-event leakage)
features_df, feature_cols = engineer.create_features(analysis_df, return_feature_names=True)
labels = engineer.create_binary_labels(analysis_df, threshold=0.0)

# Chronological time-series split
X_train, X_test, y_train, y_test, cols = engineer.prepare_train_test_data(
    features_df, labels, chronological=True
)
```

### Machine Learning (`src.ml_models`)
```python
from src import ModelTrainer

trainer = ModelTrainer()
# Cross-validation with 95% confidence intervals
cv_results = trainer.evaluate_cv(trainer.models['random_forest'], X_train, y_train, cv=5)

# Hyperparameter optimization within CV folds
best_xgb, best_params = trainer.tune_hyperparameters('xgboost', X_train, y_train, cv=3)

# Saves model pickle and companion .json metadata
trainer.save_model('xgboost', 'models/xgboost.pkl', feature_columns=cols)
```

### Realistic Backtesting (`src.backtesting`)
```python
from src import Backtester

backtester = Backtester(
    initial_capital=100000.0,
    commission_rate=0.001,      # 0.1% broker commission per trade
    slippage_bps=5.0,           # 5 basis points market slippage
    next_day_open=True          # Execute at next-day open price
)
results = backtester.run_momentum_strategy(analysis_df)
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}, Max Drawdown: {results['max_drawdown']:.2%}")
```

---

## Verification & Testing

The project maintains 100% test pass rate across 15 automated test suites:
- `test_feature_leakage.py`: Proves post-event returns are never used as model inputs.
- `test_event_analysis.py`: Verifies trading-day indexing and CAR event window summation.
- `test_data_collection.py`: Checks MultiIndex flattening, future date filters, and retry logic.
- `test_model_metadata.py`: Confirms `.json` metadata serialization and persistence.
- `test_config.py`: Validates hierarchical merging of `config.ini`, `.env`, and OS env vars.
- `test_feature_definitions.py`: Asserts central `SAFE_FEATURE_COLUMNS` integrity.
- `test_api_server.py`: Verifies async concurrency locks, background workers, and `/predict`.
- `test_error_handling.py`: Tests specific exception handling and data validation gates.
- `test_utils.py`: Validates modern `.ffill()` / `.bfill()` pandas operations.
- `test_pipeline_in_memory.py`: Confirms zero-IO in-memory DataFrame passing.
- `test_cv_and_scaling.py`: Validates 95% CI cross-validation and unscaled tree models.
- `test_tuning_calibration.py`: Tests `GridSearchCV` hyperparameter tuning and calibration.
- `test_backtest_realism.py`: Tests slippage, commission friction, and next-day execution.
- `test_packaging.py`: Confirms PEP 517/621 compliance and top-level package exports.
- `test_secrets_management.py`: Verifies `.env` loading, secret masking, and `.gitignore` rules.

To execute tests:
```bash
python -m pytest -v --tb=short
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

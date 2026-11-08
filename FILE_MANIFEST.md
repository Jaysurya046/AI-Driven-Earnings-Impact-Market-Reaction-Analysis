# Project File Manifest

## Complete Project File Structure

```
AI-Driven Earnings Impact & Market Reaction Analysis/
│
├── .github/workflows/
│   └── ci.yml                             - GitHub Actions matrix CI/CD workflow (Python 3.10-3.13)
│
├── [DOCUMENTATION FILES]
│   ├── README.md                          - Master project overview, features, quickstart & reference
│   ├── ARCHITECTURE.md                    - Deep technical architecture, data flow & module design
│   ├── COMPLETION_SUMMARY.md              - Complete status report of all 18 production improvements
│   ├── ENHANCEMENT_SUMMARY.md             - Four-phase enhancement history & implementation roadmap
│   ├── GETTING_STARTED.md                 - Step-by-step setup, CLI, dashboard, and API usage guide
│   ├── QUICK_REFERENCE.md                 - Developer cheat sheet for common operations & APIs
│   └── FILE_MANIFEST.md                   - Comprehensive repository file directory & descriptions
│
├── [CORE PYTHON PACKAGE - src/]
│   ├── __init__.py                        - Standard package initialization exposing primary domain APIs
│   ├── config.py                          - Unified config loader (config.ini + .env + OS env overrides)
│   ├── data_collection.py                 - Yahoo Finance data fetcher with MultiIndex handling & retries
│   ├── event_analysis.py                  - Trading-day event studies, abnormal returns & CAR computation
│   ├── feature_engineering.py             - Pre-announcement feature isolation (SAFE_FEATURE_COLUMNS)
│   ├── ml_models.py                       - Model training, CV with 95% CI, tuning & companion .json metadata
│   ├── backtesting.py                     - Realistic backtesting engine (commission, slippage, next-day open)
│   ├── sector_comparison.py               - Multi-sector GICS aggregation & t-test statistics
│   ├── sentiment_analysis.py              - Earnings news sentiment extraction (TextBlob / VADER)
│   ├── streamlit_app.py                   - Interactive multi-tab exploratory research dashboard
│   ├── lstm_models.py                     - Deep learning LSTM sequential reaction predictors
│   ├── utils.py                           - Modern pandas helpers (.ffill/.bfill), outlier & metric tools
│   └── visualizations.py                  - High-resolution publication charting & reporting
│
├── [AUTOMATED TEST SUITE - tests/]
│   ├── test_api_server.py                 - FastAPI concurrency locks, background tasks & /predict tests
│   ├── test_backtest_realism.py           - Friction testing (broker commission & slippage bps)
│   ├── test_config.py                     - Config precedence, type conversion & fallbacks
│   ├── test_cv_and_scaling.py             - 95% CI cross-validation & unscaled tree model tests
│   ├── test_data_collection.py            - MultiIndex flattening, timezone normalization & API retries
│   ├── test_error_handling.py             - Typed error handlers & post-collection shape validation
│   ├── test_event_analysis.py             - Trading-day calendar mapping & CAR calculation tests
│   ├── test_feature_definitions.py        - Canonical SAFE_FEATURE_COLUMNS integrity tests
│   ├── test_feature_leakage.py            - Proves zero post-event return leakage into model inputs
│   ├── test_model_metadata.py             - Companion .json metadata persistence tests
│   ├── test_packaging.py                  - PEP 517/621 packaging & top-level export validation
│   ├── test_pipeline_in_memory.py         - In-memory DataFrame bus integration test
│   ├── test_repo_hygiene.py               - GitHub Actions CI, lockfiles & doc integrity tests
│   ├── test_sector_and_sentiment.py       - Sector comparison & sentiment pipeline tests
│   ├── test_secrets_management.py         - .env.example, dotenv loader & secret masking tests
│   ├── test_tuning_calibration.py         - GridSearchCV hyperparameter tuning & probability calibration
│   └── test_utils.py                      - Modernized pandas API (.ffill/.bfill) tests
│
├── [MAIN EXECUTABLE SCRIPTS]
│   ├── main.py                            - Orchestrated in-memory quantitative analysis pipeline
│   ├── api_server.py                      - High-throughput thread-safe FastAPI REST prediction server
│   ├── retraining_scheduler.py            - Automated cron-style background model retraining daemon
│   └── quick_start.py                     - Rapid 5-stock demonstration script
│
├── [BUILD & CONFIGURATION]
│   ├── LICENSE                            - Standard MIT open-source license agreement
│   ├── pyproject.toml                     - PEP 517/PEP 621 packaging, setuptools discovery & test config
│   ├── config.ini                         - Default baseline configuration settings
│   ├── .env.example                       - Environment configuration & secrets template
│   ├── requirements.txt                   - Core and development dependency requirements
│   ├── requirements-lock.txt              - Pinned reproducible production dependency lockfile
│   └── .gitignore                         - Environment variable protection & build cache exclusion
│
├── [JUPYTER NOTEBOOKS - notebooks/]
│   └── earnings_analysis.ipynb            - Interactive exploratory research notebook
│
└── [GENERATED DIRECTORIES]
    ├── data/                              - Cached market data & optional intermediate CSV snapshots
    ├── models/                            - Trained model weights (.pkl) & companion metadata (.json)
    └── visualizations/                    - High-resolution analytical charts & summary reports
```

---

## Detailed Component Specifications

### 1. Core Python Package (`src/`)

| File | Primary Classes / Functions | Responsibility |
|---|---|---|
| `__init__.py` | Top-level exports | Exposes core domain classes for direct `import src` usage. |
| `config.py` | `AppConfig`, `load_config`, `mask_secrets` | Hierarchical configuration loader merging `.env`, env vars, `config.ini`, and defaults. |
| `data_collection.py` | `DataCollector`, `validate_collected_data` | Robust Yahoo Finance ingestion with MultiIndex handling, future date filtering, and retries. |
| `event_analysis.py` | `EventAnalyzer` | Exact trading-day event studies, abnormal returns, and CAR calculation. |
| `feature_engineering.py` | `FeatureEngineer`, `SAFE_FEATURE_COLUMNS` | Zero-leakage pre-announcement feature creation and chronological time-series splitting. |
| `ml_models.py` | `ModelTrainer` | Classifier training, tree-model unscaling, CV with 95% CI, tuning, calibration, and metadata saving. |
| `backtesting.py` | `Backtester` | Strategy simulations with realistic broker commissions, slippage basis points, and next-day open execution. |
| `sector_comparison.py` | `SectorAnalyzer` | Sector-level aggregation and statistical hypothesis testing (t-tests). |
| `sentiment_analysis.py` | `SentimentAnalyzer` | News sentiment score extraction around earnings announcement windows. |
| `streamlit_app.py` | Streamlit Dashboard | Multi-tab interactive UI for data exploration, model inspection, and backtesting. |
| `lstm_models.py` | `LSTMPredictor`, `HybridEnsemblePredictor` | Deep learning time-series sequence modeling for post-earnings price trends. |
| `utils.py` | Pandas helper utilities | Modern `.ffill()` / `.bfill()`, outlier identification, and performance report generation. |
| `visualizations.py` | `Visualizer` | Matplotlib/Seaborn charting for scatter plots, event windows, and confusion matrices. |

---

### 2. Services & Entry Points

| File | Runtime Command | Purpose |
|---|---|---|
| `main.py` | `python main.py` | End-to-end in-memory pipeline execution with optional intermediate CSV export. |
| `api_server.py` | `uvicorn api_server:app --reload` | Thread-safe FastAPI server with background task execution and concurrency locks. |
| `retraining_scheduler.py` | `python retraining_scheduler.py --continuous` | Automated background retraining daemon. |
| `quick_start.py` | `python quick_start.py` | Fast 5-stock sanity execution. |
| `pyproject.toml` | `pip install -e ".[dev]"` | Standardized package installation and pytest configuration. |
| `requirements-lock.txt` | `pip install -r requirements-lock.txt` | Reproducible deployment across production environments. |
| `.github/workflows/ci.yml` | Triggered on Git Push / PR | Automated GitHub Actions CI across Python 3.10, 3.11, 3.12, and 3.13. |

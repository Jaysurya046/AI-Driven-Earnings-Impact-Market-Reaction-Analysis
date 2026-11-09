# Technical Architecture & Implementation Guide

## System Architecture

### High-Level Overview

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                  EARNINGS IMPACT & MARKET REACTION ANALYSIS PLATFORM                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  ┌───────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │   Configuration &     │  │   Data Ingestion &      │  │    Event Study Engine   │  │
│  │   Secrets Management  │  │   Hardening Layer       │  │    (Trading-Day Exact)  │  │
│  ├───────────────────────┤  ├─────────────────────────┤  ├─────────────────────────┤  │
│  │ • Unified load_config │  │ • yfinance connector    │  │ • Trading-day offsets   │  │
│  │ • config.ini + .env   │  │ • MultiIndex flattening │  │ • Estimation (-60:-10)  │  │
│  │ • OS Env Overrides    │  │ • Timezone normalization│  │ • Event window [-5:+5]  │  │
│  │ • Sensitive Masking   │  │ • Exponential backoff   │  │ • Mean-adjusted AR      │  │
│  │ • .env.example / git  │  │ • Shape validation gates│  │ • CAR calculation       │  │
│  └───────────────────────┘  └─────────────────────────┘  └─────────────────────────┘  │
│              │                           │                            │                │
│              └───────────────────────────┼────────────────────────────┘                │
│                                          v                                             │
│                       ┌─────────────────────────────────────┐                          │
│                       │   In-Memory Pipeline DataFrame Bus  │                          │
│                       │   (Optional CSV File Checkpointing) │                          │
│                       └─────────────────────────────────────┘                          │
│                                          │                                             │
│         ┌────────────────────────────────┼────────────────────────────────┐            │
│         v                                v                                v            │
│  ┌───────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │  Feature Engineering  │  │   Machine Learning &    │  │   Realistic Quantitative│  │
│  │  (Zero Data Leakage)  │  │   Model Calibration     │  │   Backtesting Engine    │  │
│  ├───────────────────────┤  ├─────────────────────────┤  ├─────────────────────────┤  │
│  │ • SAFE_FEATURE_COLUMNS│  │ • Logistic Regression   │  │ • Commission modeling   │  │
│  │ • Strict pre-event    │  │ • Random Forest         │  │ • Slippage friction(bps)│  │
│  │ • No post-return leak │  │ • XGBoost (unscaled)    │  │ • Next-day open exec    │  │
│  │ • TimeSeriesSplit     │  │ • 95% CI Cross-Val      │  │ • Momentum & Reversion  │  │
│  │ • Chronological order │  │ • Hyperparameter tuning │  │ • Sharpe / Max Drawdown │  │
│  │ • Feature importance  │  │ • Probability Calib     │  │ • Portfolio equity curve│  │
│  └───────────────────────┘  └─────────────────────────┘  └─────────────────────────┘  │
│                                          │                                             │
│         ┌────────────────────────────────┴────────────────────────────────┐            │
│         v                                                                 v            │
│  ┌───────────────────────────────┐                             ┌───────────────────┐  │
│  │   Thread-Safe REST API        │                             │  Visualization &  │  │
│  │   (FastAPI + Async Workers)   │                             │  Analytics UI     │  │
│  ├───────────────────────────────┤                             ├───────────────────┤  │
│  │ • AnalysisStateManager        │                             │ • Streamlit tabs  │  │
│  │ • asyncio.Lock state guard    │                             │ • Matplotlib / SNS│  │
│  │ • Background worker tasks     │                             │ • Interactive     │  │
│  │ • Sanitized JSON floats       │                             │   Scatter/Heatmaps│  │
│  │ • Secured CORS origins        │                             │ • Automated report│  │
│  └───────────────────────────────┘                             └───────────────────┘  │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                            Artifact & Metadata Store                             │  │
│  │ • Serialized Models (.pkl)        • Companion Metadata (.json)                   │  │
│  │ • Feature Columns Registry        • Pinned Dependencies (requirements-lock.txt)  │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Module Architecture

### 1. Unified Configuration & Secrets (`src/config.py`)

**Purpose**: Centralize configuration management with strict priority inheritance and secrets masking.

**Order of Precedence**:
1. Environment variables (`APP_*` or bare names) — highest priority.
2. `.env` file variables (loaded via `python-dotenv`).
3. `config.ini` settings.
4. Hardcoded `DEFAULT_CONFIG` fallbacks — lowest priority.

**Key API**:
```python
class AppConfig:
    def get(key: str, default: Any = None) -> Any
    def to_dict() -> Dict[str, Any]
    def __repr__() -> str  # Automatically masks secret tokens

def load_config(config_file=None, env_file=None, env_prefix="APP_") -> AppConfig
def mask_secrets(config_dict: Dict[str, Any]) -> Dict[str, Any]
```

---

### 2. Data Collection & Hardening (`src/data_collection.py`)

**Purpose**: Resilient market data ingestion from Yahoo Finance with automated anomaly handling.

**Key Classes & Functions**:
```python
class DataCollector:
    def __init__(lookback_years: int = 5, max_retries: int = 3, retry_delay: float = 1.0)
    def fetch_stock_price_data(ticker: str) -> pd.DataFrame
    def fetch_earnings_dates(ticker: str) -> pd.DataFrame
    def fetch_multiple_stocks(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]
    
    @staticmethod
    def validate_collected_data(price_df, earnings_df, min_price_records=10, min_earnings_records=1) -> None
```

**Robustness Features**:
- **MultiIndex Flattening**: Seamlessly extracts OHLCV columns whether returned as single or MultiIndex data.
- **Date Normalization**: Strips timezone offsets (`tz_localize(None)`) to ensure perfect alignment with market holidays and weekend earnings reports.
- **Future Date Filtering**: Discards future announcement estimates where actual EPS data has not yet occurred.
- **Exponential Backoff**: Retries failed network calls with geometric delay increments.
- **Immediate Validation**: Checks non-emptiness, required schemas, and ticker overlap before passing DataFrames downstream.

---

### 3. Event Study Analysis (`src/event_analysis.py`)

**Purpose**: Quantify equity market reaction using trading-day event studies and Cumulative Abnormal Returns (CAR).

**Key Classes**:
```python
class EventAnalyzer:
    def __init__(event_window: int = 5, estimation_window: int = 60)
    def analyze_event_window(price_data: pd.DataFrame, earnings_date: datetime, ticker: str) -> Dict
    def analyze_multiple_events(price_data: pd.DataFrame, earnings_data: pd.DataFrame) -> pd.DataFrame
```

**Methodology**:
1. **Trading-Day Indexing**: Uses exact trading days rather than calendar intervals. If an earnings release occurs on a weekend or post-market, the announcement event day ($t=0$) is mapped to the immediate subsequent trading session.
2. **Estimation Window**: Historical window ($[-60, -10]$ trading days) computes baseline expected return $\bar{R}_{\text{est}}$ and historical volatility $\sigma_{\text{est}}$.
3. **Abnormal Return ($AR_t$)**:
   $$\text{AR}_t = R_t - \bar{R}_{\text{est}}$$
4. **Cumulative Abnormal Return (CAR)**:
   $$\text{CAR} = \sum_{t=-w}^{+w} \text{AR}_t$$
5. **Volatility & Volume Shifts**: Computes pre-event vs post-event standard deviations and volume spike ratios.

---

### 4. Feature Engineering & Leakage Isolation (`src/feature_engineering.py`)

**Purpose**: Generate predictive ML features strictly from pre-announcement data, preventing look-ahead bias.

**Single Source of Truth**:
- `SAFE_FEATURE_COLUMNS`: Canonical list of approved model input features.
- `LEAKY_COLUMNS`: Prohibited columns derived from post-announcement data (`PostEventReturn`, `PostEventVolatility`, `PostEventVolume`).
- `NON_FEATURE_COLUMNS`: Identifiers and metadata stripped before training (`Date`, `Ticker`, `Sector`).

**Feature Set**:
1. **Earnings Surprise**: `EarningsSurpriseAbs`, `EarningsSurpriseDirection`, `SurprisePercentage`.
2. **Pre-Event Volatility**: `PreEventVolatility`, `VolatilityRatio` ($\sigma_{\text{pre}} / \sigma_{\text{est}}$).
3. **Volume Dynamics**: `VolumeSpikeRatio` ($\text{Vol}_{\text{pre}} / \text{Vol}_{\text{est}}$), `VolumeAnomalous`.
4. **Pre-Event Momentum**: `PreEventMomentum` (return over days $[-5, -1]$), `PriceTrend_20D`.
5. **Safe Interactions**: `SurpriseVolatilityInteraction`, `PreEventVol_SurpriseInteraction`.

**Chronological Splitting**:
```python
# Partitions data strictly in chronological order (earliest 80% train, latest 20% test)
X_train, X_test, y_train, y_test, cols = feature_engineer.prepare_train_test_data(
    features_df, labels, chronological=True
)
```

---

### 5. Machine Learning, Calibration & Metadata (`src/ml_models.py`)

**Purpose**: Model training, cross-validation with confidence intervals, hyperparameter tuning, probability calibration, and artifact persistence.

**Key Capabilities**:
- **Selective Feature Scaling**: Automatically disables scaling for tree models (`RandomForest`, `XGBoost`) while retaining `StandardScaler` for `LogisticRegression`.
- **Cross-Validation with 95% CI**:
  $$\text{CI}_{95\%} = \bar{s} \pm 1.96 \times \frac{\text{Std}(s)}{\sqrt{k}}$$
- **Hyperparameter Optimization**: `tune_hyperparameters` performs `GridSearchCV` strictly inside CV folds.
- **Probability Calibration**: `calibrate_probabilities` applies `CalibratedClassifierCV` (isotonic or sigmoid) to align predicted probabilities with empirical event frequencies.
- **Metadata Serialization**: In addition to pickled weights (`.pkl`), creates companion `.json` files storing:
  - Feature column names in exact training order
  - Standard scaling parameters
  - Training timestamp & Python/library versions
  - Accuracy, F1, and ROC-AUC evaluation metrics

---

### 6. Realistic Backtesting Engine (`src/backtesting.py`)

**Purpose**: Simulate trading strategies accounting for market frictions.

**Realistic Execution Parameters**:
- `commission_rate`: Fractional fee per trade (e.g. 0.001 = 10 bps).
- `slippage_bps`: Bid-ask spread penalty (e.g. 5 bps deducted on execution).
- `next_day_open`: Simulates realistic order fills on the day following announcement.

**Strategies**:
- **Momentum Strategy**: Goes long on positive surprise/CAR, short on negative.
- **Mean-Reversion Strategy**: Bets against extreme initial event reactions.
- **Metrics**: Total Return, Annualized Sharpe Ratio, Maximum Drawdown, Win Rate, and Profit Factor.

---

### 7. Thread-Safe REST API Server (`api_server.py`)

**Purpose**: High-throughput FastAPI service for real-time inference and asynchronous pipeline execution.

**Architecture**:
```
Client Request
      │
      ▼
FastAPI Router (CORS & Auth Guard)
      │
      ├── GET /predict ──────► Reads Model Artifacts & Cached Scalers (Fast In-Memory)
      │
      └── POST /analyze ─────► AnalysisStateManager (asyncio.Lock Guard)
                                    │
                                    └── BackgroundTasks (Non-blocking async worker)
```

- **Thread-Safe State**: `AnalysisStateManager` guards concurrent pipeline runs with `asyncio.Lock()`.
- **Sanitized JSON**: Implements `_safe_float()` converting IEEE 754 `NaN`/`Infinity` floats into JSON-compliant values.

---

## Data Flow Architecture (In-Memory & Storage)

```
[Yahoo Finance API]
        │
        ▼ (HTTP + Backoff Retry)
[DataCollector] ───► DataFrame (Prices & Dates)
        │
        ▼ (In-Memory DataFrame Bus)
[EventAnalyzer] ───► DataFrame (Event Windows & CAR)
        │
        ▼ (In-Memory DataFrame Bus)
[FeatureEngineer] ─► DataFrame (Safe Pre-Event Features & Binary Labels)
        │
        ▼ (In-Memory DataFrame Bus)
[ModelTrainer] ────► Model Pickles (.pkl) + Companion Metadata (.json)
        │
        ▼ (In-Memory DataFrame Bus)
[Backtester] ──────► Strategy Performance & Equity Curves
        │
        ▼
[Visualizer] ──────► High-Res Artifacts (PNG / PDF)
```

*(Note: Setting `save_intermediate=True` writes CSV snapshots to `data/` at each pipeline step for auditing.)*

---

## Packaging & Testing Specifications

- **Package Standard**: PEP 517 / PEP 621 compliant `pyproject.toml` with `setuptools>=61.0`.
- **Test Suite**: 68 automated unit, integration, and regression tests executed via `pytest`.
- **Continuous Integration**: `.github/workflows/ci.yml` matrix pipeline across Python 3.10, 3.11, 3.12, and 3.13.
- **Dependency Pinning**: Reproducible production deployments guaranteed via `requirements-lock.txt`.

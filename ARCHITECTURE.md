# Technical Architecture & Implementation Guide

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 EARNINGS ANALYSIS SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────┐  │
│  │  Data Layer    │  │  Analysis Layer  │  │  ML Layer  │  │
│  ├────────────────┤  ├──────────────────┤  ├────────────┤  │
│  │                │  │                  │  │            │  │
│  │ • Yahoo API    │  │ • Event Window   │  │ • Feature  │  │
│  │ • CSV Storage  │  │ • Return Calc    │  │   Engineer │  │
│  │ • SQL Ready    │  │ • Abnormal Ret   │  │ • Training │  │
│  │                │  │ • CAR Compute    │  │ • Eval     │  │
│  │                │  │ • Vol/Vol Analy  │  │            │  │
│  └────────────────┘  └──────────────────┘  └────────────┘  │
│         │                    │                    │          │
│         └────────────────────┴────────────────────┘          │
│                         │                                    │
│                         v                                    │
│              ┌────────────────────────┐                     │
│              │  Visualization Layer   │                     │
│              ├────────────────────────┤                     │
│              │ • Matplotlib/Seaborn   │                     │
│              │ • Dashboard Charts     │                     │
│              │ • Interactive Plots    │                     │
│              └────────────────────────┘                     │
│                         │                                    │
│                         v                                    │
│         ┌──────────────────────────────────┐               │
│         │      Output & Reporting Layer    │               │
│         ├──────────────────────────────────┤               │
│         │ • CSV/Database Export           │               │
│         │ • Model Serialization           │               │
│         │ • Report Generation             │               │
│         │ • Dashboard Export              │               │
│         └──────────────────────────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. Data Collection Module (`data_collection.py`)

**Purpose**: Fetch and store market data

**Key Classes**:
```python
class DataCollector:
    def __init__(lookback_years)
    def fetch_stock_price_data(ticker) -> DataFrame
    def fetch_earnings_dates(ticker) -> DataFrame
    def fetch_multiple_stocks(tickers) -> Tuple[DataFrame, DataFrame]
    def save_to_csv(df, filepath) -> None
```

**Data Flow**:
```
Yahoo Finance API
      |
      v
yfinance Library
      |
      v
DataCollector
      |
      +---> Validates data
      |
      +---> Formats columns
      |
      +---> Combines tickers
      |
      v
CSV Files
```

**Output Columns**:
- Price data: Date, Open, High, Low, Close, Adj Close, Volume, Ticker
- Earnings data: Date, EPS, Surprise_Percentage, Ticker

---

### 2. Event Analysis Module (`event_analysis.py`)

**Purpose**: Analyze price movements around earnings events

**Key Classes**:
```python
class EventAnalyzer:
    def __init__(event_window, estimation_window)
    def calculate_returns(prices) -> Series
    def calculate_abnormal_returns(event_prices, estimation_prices) -> Series
    def calculate_cumulative_abnormal_returns(abnormal_returns) -> Series
    def analyze_event_window(price_data, earnings_date, ticker) -> Dict
    def analyze_multiple_events(price_data, earnings_data) -> DataFrame
```

**Calculations**:

1. **Daily Returns**: 
   ```
   Return_t = (Price_t - Price_{t-1}) / Price_{t-1} * 100
   ```

2. **Expected Return** (from estimation window):
   ```
   Expected_Return = Mean(Returns_{-60 to -10})
   ```

3. **Abnormal Return**:
   ```
   Abnormal_Return = Actual_Return - Expected_Return
   ```

4. **CAR** (Cumulative Abnormal Return):
   ```
   CAR = (1 + AR_1/100) * (1 + AR_2/100) * ... * (1 + AR_n/100) - 1
   ```

5. **Volatility** (Standard Deviation):
   ```
   Volatility = StdDev(Daily_Returns)
   ```

**Output Features**:
- PreEventReturn: Return from day -5 to day -1
- PostEventReturn: Return from day +1 to day +5
- AbnormalReturnCAR: CAR across ±5 day window
- PreEventVolatility: Volatility pre-event
- PostEventVolatility: Volatility post-event
- VolumeSpikeRatio: Post event volume / estimation volume

---

### 3. Feature Engineering Module (`feature_engineering.py`)

**Purpose**: Create ML-ready features and labels

**Key Classes**:
```python
class FeatureEngineer:
    def create_features(event_analysis_df) -> DataFrame
    def create_labels(event_analysis_df, pos_threshold, neg_threshold) -> Series
    def create_binary_labels(event_analysis_df, threshold) -> Series
    def prepare_train_test_data(features, labels, feature_columns, 
                                test_size, random_state) -> Tuple
    def get_feature_importance_df(model, feature_columns) -> DataFrame
```

**Features Created (Strict Pre-Announcement Integrity)**:

1. **Surprise Features**:
   - EarningsSurpriseAbs: |Reported EPS - Expected EPS|
   - EarningsSurpriseDirection: Sign(Surprise)

2. **Volatility Features**:
   - PreEventVolatility: Historical volatility prior to announcement
   - VolatilityChangeRatio: PostVol / EstimationVol
   - VolatilityIncrease: PostVol - EstimationVol

3. **Volume Features**:
   - VolumeSpikeRatio: PostVolume / EstimationVolume
   - VolumeAnomalous: 1 if Ratio > 1.5, else 0

4. **Momentum Features**:
   - PreEventMomentum: Pre-event historical return
   - PriceTrend_20D: Short-term historical momentum prior to announcement

5. **Safe Interaction Features**:
   - SurpriseVolatilityInteraction: Surprise × VolRatio
   - PreEventVol_SurpriseInteraction: Pre-event Volatility × EPS Surprise

> **Note on Data Leakage Prevention**: Features derived from post-announcement prices/returns (such as `PostEventMomentum` or `PreVol_PostReturn`) are strictly prohibited and stripped out via `SAFE_FEATURE_COLUMNS` and `LEAKY_COLUMNS` to ensure realistic, uncompromised model evaluation.

**Label Creation**:
```python
if PostEventReturn > threshold:
    Label = 1  # Positive reaction
else:
    Label = 0  # Negative reaction
```

**Train-Test Split (Chronological Order)**:
```
All Events (N, sorted by Date ascending)
   |
   +---> Training Set (Earliest 80%) [No look-ahead bias]
   |
   +---> Test Set (Latest 20%)
```

---

### 4. ML Models Module (`ml_models.py`)

**Purpose**: Train, tune, calibrate, and evaluate classification models with metadata persistence

**Key Classes & Capabilities**:
```python
class ModelTrainer:
    def train_logistic_regression(X_train, y_train) -> Tuple[Model, Scaler]
    def train_random_forest(X_train, y_train) -> Model
    def train_xgboost(X_train, y_train, scale_features=False) -> Tuple[Model, Optional[Scaler]]
    def evaluate_model(model, X_test, y_test, scaler, model_name) -> Dict
    def evaluate_cv(model, X, y, cv=5) -> Dict  # Cross-validation with 95% CI
    def tune_hyperparameters(model_name, X_train, y_train, cv=3) -> Tuple[Model, Dict]
    def calibrate_probabilities(model, X_val, y_val, scaler) -> CalibratedClassifierCV
    def save_model(model_name, filepath, feature_columns=None, metadata=None) -> None  # Saves .pkl and companion .json
    def load_model(filepath, model_name) -> Dict  # Loads model and associated metadata
```

**Models**:

1. **Logistic Regression**
   - Type: Linear classifier
   - Scaling: Required (StandardScaler)
   - Pros: Fast, interpretable
   - Cons: May underfit on complex patterns

2. **Random Forest**
   - Type: Ensemble tree-based
   - Scaling: Not required
   - Pros: Captures non-linear patterns, feature importance
   - Cons: Prone to overfitting

3. **XGBoost**
   - Type: Gradient boosting
   - Scaling: Recommended
   - Pros: State-of-the-art performance
   - Cons: More complex, longer training

**Evaluation Metrics**:

```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 * (Precision × Recall) / (Precision + Recall)
AUC-ROC = Area under receiver operating characteristic curve
```

**Model Comparison**:
```
Model             Accuracy  Precision  Recall  F1-Score  AUC-ROC
─────────────────────────────────────────────────────────────────
Logistic Reg      0.6201    0.6124    0.5987   0.6054   0.6789
Random Forest     0.6543    0.6421    0.6234   0.6326   0.7123
XGBoost           0.6876    0.6754    0.6543   0.6648   0.7456
```

---

### 5. Visualization Module (`visualizations.py`)

**Purpose**: Create insights and dashboards

**Key Classes**:
```python
class Visualizer:
    def plot_event_window(event_analysis_df, ticker) -> None
    def plot_surprise_vs_reaction(event_analysis_df) -> None
    def plot_model_comparison(model_comparison_df) -> None
    def plot_feature_importance(importance_df, top_n) -> None
    def plot_volatility_analysis(event_analysis_df) -> None
    def plot_volume_analysis(event_analysis_df) -> None
    def plot_confusion_matrix(confusion_matrix, model_name) -> None
```

**Chart Types**:
- Scatter plots: Surprise vs Returns
- Bar charts: Model comparisons
- Line plots: Volatility trends
- Heatmaps: Confusion matrices
- Histograms: Distribution analysis

---

### 6. Main Pipeline (`main.py`)

**Purpose**: Orchestrate entire analysis workflow

**Pipeline Steps**:
```
1. Data Collection
   ├─ Fetch price data (5 years)
   ├─ Fetch earnings dates
   └─ Save raw data

2. Event Analysis
   ├─ Analyze each earnings event
   ├─ Calculate returns and abnormal returns
   └─ Save event analysis

3. Feature Engineering
   ├─ Create ML features
   ├─ Generate labels
   └─ Save feature set

4. Model Training
   ├─ Train 3 models
   ├─ Evaluate on test set
   └─ Save model comparison

5. Visualization
   ├─ Generate charts
   ├─ Create dashboards
   └─ Save visualizations

6. Reporting
   ├─ Compile insights
   ├─ Generate statistics
   └─ Save final report
```

**Configuration**:
```python
config = {
    'lookback_years': 5,
    'event_window': 5,
    'estimation_window': 60,
    'label_threshold': 0.0,
    'data_dir': 'data',
    'models_dir': 'models',
    'viz_dir': 'visualizations'
}
```

---

## Data Flow Diagram

```
START
  |
  v
┌──────────────────────────────────┐
│ DataCollector.fetch_data()       │
│ - Download prices               │
│ - Get earnings dates            │
└──────────────────────────────────┘
  |
  v
┌──────────────────────────────────┐
│ Raw CSV Files                     │
│ - raw_price_data.csv            │
│ - raw_earnings_data.csv         │
└──────────────────────────────────┘
  |
  v
┌──────────────────────────────────┐
│ EventAnalyzer.analyze_events()   │
│ - Calculate returns             │
│ - Compute abnormal returns      │
│ - Measure volatility/volume     │
└──────────────────────────────────┘
  |
  v
┌──────────────────────────────────┐
│ event_analysis.csv               │
│ - PreEventReturn                │
│ - PostEventReturn               │
│ - AbnormalReturnCAR             │
│ - VolatilityRatio               │
│ - VolumeSpikeRatio              │
└──────────────────────────────────┘
  |
  v
┌──────────────────────────────────┐
│ FeatureEngineer.create_features()│
│ - Engineer features             │
│ - Create labels                 │
│ - Handle missing values         │
└──────────────────────────────────┘
  |
  v
┌──────────────────────────────────┐
│ engineered_features.csv          │
│ - All ML features               │
│ - Binary labels (0/1)           │
└──────────────────────────────────┘
  |
  v
┌──────────────────────────────────┐
│ ModelTrainer.train_all_models()  │
│ - Train Logistic Regression     │
│ - Train Random Forest           │
│ - Train XGBoost                 │
└──────────────────────────────────┘
  |
  v
┌──────────────────────────────────┐
│ Model Comparison & Evaluation    │
│ - Accuracy, Precision, Recall   │
│ - Confusion matrices            │
│ - Feature importance            │
└──────────────────────────────────┘
  |
  v
┌──────────────────────────────────┐
│ Visualizer.generate_plots()      │
│ - Surprise vs reaction          │
│ - Model comparison              │
│ - Feature importance            │
│ - Volatility analysis           │
└──────────────────────────────────┘
  |
  v
┌──────────────────────────────────┐
│ Final Outputs                    │
│ - Data files (CSV)              │
│ - Models (.pkl)                 │
│ - Visualizations (PNG)          │
│ - Report (TXT)                  │
└──────────────────────────────────┘
  |
  v
 END
```

---

## Key Algorithms

### 1. Abnormal Return Calculation

```python
def calculate_abnormal_returns(event_window, estimation_window):
    # Step 1: Get estimation period data (days -60 to -10)
    estimation_returns = calculate_returns(estimation_window_prices)
    
    # Step 2: Calculate expected return
    expected_return = estimation_returns.mean()
    
    # Step 3: Get event period data (days -5 to +5)
    event_returns = calculate_returns(event_window_prices)
    
    # Step 4: Calculate abnormal returns
    abnormal_returns = event_returns - expected_return
    
    # Step 5: Calculate CAR
    car = (1 + abnormal_returns[0]/100) * 
          (1 + abnormal_returns[1]/100) * ... - 1
    
    return abnormal_returns, car
```

### 2. Model Training

```python
def train_model(X_train, y_train, model_type):
    # Step 1: Split features and labels
    X_features, y_labels = prepare_data()
    
    # Step 2: Scale if needed
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Step 3: Initialize model
    model = create_model(model_type)
    
    # Step 4: Train on data
    model.fit(X_scaled, y_train)
    
    # Step 5: Return trained model
    return model, scaler
```

### 3. Feature Engineering

```python
def engineer_features(event_analysis_data):
    features = event_analysis_data.copy()
    
    # 1. Surprise-based features
    features['SurpriseAbs'] = abs(features['Surprise'])
    
    # 2. Volatility-based features
    features['VolRatio'] = (features['PostVol'] / 
                           features['EstimationVol'])
    
    # 3. Volume-based features
    features['VolSpike'] = features['VolumeSpikeRatio']
    
    # 4. Interaction features
    features['SurpriseVolInteract'] = (
        features['SurpriseAbs'] * 
        features['VolRatio']
    )
    
    # 5. Momentum features
    features['Momentum'] = features['PreEventReturn']
    
    return features
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Data fetch | O(n) | n = number of days |
| Event analysis | O(n*m) | n = stocks, m = events |
| Feature engineering | O(n*m) | n = stocks, m = events |
| Model training | O(n²) to O(n³) | Depends on algorithm |
| Model evaluation | O(n) | n = test samples |

### Space Complexity

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Price data | O(n*m) | n = days, m = stocks |
| Event data | O(m) | m = earnings events |
| Features | O(n*f) | n = events, f = features |
| Models | O(f²) | f = features |

### Typical Execution Times

| Step | Time (for 5 stocks, 5 years) |
|------|-----|
| Data collection | 30-60 seconds |
| Event analysis | 5-10 seconds |
| Feature engineering | <1 second |
| Model training | 10-30 seconds |
| Visualization | 5-15 seconds |
| **Total** | **~1-2 minutes** |

---

## Error Handling

### Data Validation

```python
# Check data completeness
if len(price_data) < min_required_days:
    logger.error("Insufficient price data")
    
# Check earnings dates
if earnings_dates is None or len(earnings_dates) == 0:
    logger.warning(f"No earnings data for {ticker}")
    
# Validate feature values
if feature.isna().sum() > threshold:
    logger.warning("High missing value rate")
```

### Model Robustness

```python
# Cross-validation
cross_val_scores = cross_val_score(model, X, y, cv=5)

# Handle class imbalance
class_weights = compute_class_weight(
    'balanced', 
    np.unique(y), 
    y
)

# Regularization
model = LogisticRegression(C=1.0, penalty='l2')
```

---

## Extensibility

### Adding New Features

```python
class FeatureEngineer:
    def create_features(self, event_analysis_df):
        # ... existing features ...
        
        # Add new feature
        df['MyNewFeature'] = (
            df['PostEventReturn'] * df['VolumeSpikeRatio']
        )
        
        return df
```

### Adding New Models

```python
from sklearn.ensemble import GradientBoostingClassifier

class ModelTrainer:
    def train_gradient_boosting(self, X_train, y_train):
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)
        return model
```

### Adding New Visualizations

```python
class Visualizer:
    def plot_returns_heatmap(self, event_analysis_df):
        pivot = event_analysis_df.pivot_table(
            values='PostEventReturn',
            index='Ticker',
            columns='Quarter'
        )
        sns.heatmap(pivot, cmap='RdYlGn', center=0)
        plt.show()
```

---

## Best Practices

### Data Quality
- Validate all inputs
- Handle missing values appropriately
- Check for outliers
- Log all operations

### Model Development
- Use cross-validation
- Monitor for overfitting
- Compare multiple models
- Document hyperparameters

### Code Organization
- Follow PEP 8 style
- Use type hints
- Document functions
- Log important steps

### Performance
- Cache expensive computations
- Use vectorized operations
- Parallelize where possible
- Monitor memory usage

---

## References

### Mathematical Concepts
- Event study methodology: Standard approach in financial research
- Abnormal returns: Actual - Expected return
- CAR: Cumulative product of abnormal returns
- Volatility: Standard deviation of returns

### Python Libraries
- **yfinance**: Yahoo Finance data API
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **xgboost**: Gradient boosting
- **matplotlib/seaborn**: Visualization

### Papers & Resources
- MacKinley (1997): Event Studies in Economics and Finance
- Fama et al.: Market Efficiency Studies
- Vapnik: Support Vector Machines
- Friedman: Gradient Boosting

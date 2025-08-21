# Forecasting Algorithms Implementation

## Algorithm Overview

Time Series Pro implements 9 different forecasting algorithms with graceful fallbacks for optional dependencies:

### Always Available (Core Dependencies)
1. **ARIMA** - Statistical time series model
2. **Linear Regression** - Time-based feature regression
3. **Moving Average** - Simple statistical baseline
4. **Random Forest** - Ensemble method with lag features

### Optional Dependencies
5. **Prophet** - Facebook's forecasting library
6. **LightGBM** - Gradient boosting framework  
7. **XGBoost** - Extreme gradient boosting
8. **SARIMAX** - Seasonal ARIMA with external variables
9. **LSTM** - Neural network (with fallback to MLP)

## Implementation Architecture

### ForecastingEngine Class (`forecasting.py:54-910`)

**Central Orchestrator**: All algorithms implemented as methods in single class
- Consistent interface: `train_*()` methods
- Shared utilities: `_prepare_data()`, `_calculate_metrics()`
- Graceful error handling throughout

**Key Methods**:
- `_prepare_data()`: 80/20 train/test split
- `_calculate_metrics()`: Standardized performance metrics
- `train_*()`: Individual algorithm implementations

## Algorithm Implementations

### 1. ARIMA (`train_arima`, lines 90-141)

**Statistical Time Series Model**
```python
model = ARIMA(train_data[self.target_column], order=order)
fitted_model = model.fit()
```

**Features**:
- Configurable order parameters (p,d,q)
- Automatic differencing for stationarity
- Confidence interval estimation (±5%)
- Default order: (1,1,1)

**Use Cases**: Stationary time series, simple trend patterns

### 2. Linear Regression (`train_linear_regression`, lines 143-213)

**Time-Based Feature Engineering**
```python
features['time_idx'] = range(len(data))
features['day_of_week'] = data.index.dayofweek
features['month'] = data.index.month
features['day_of_month'] = data.index.day
```

**Features**:
- Temporal features (day, week, month, time index)
- Simple linear trend modeling
- Fast training and prediction
- Confidence intervals (±5%)

**Use Cases**: Linear trends, basic seasonality patterns

### 3. Moving Average (`train_moving_average`, lines 215-263)

**Statistical Baseline**
```python
ma_values = train_data[self.target_column].rolling(window=window).mean()
last_ma_value = ma_values.iloc[-1]
```

**Features**:
- Configurable window size (default: 7)
- Simple trend following
- Constant future predictions
- Wide confidence intervals (±10%)

**Use Cases**: Baseline comparison, simple trend following

### 4. Random Forest (`train_random_forest`, lines 265-383)

**Advanced Feature Engineering**
```python
# Lag features
for lag in range(1, n_lags + 1):
    features[f'lag_{lag}'] = data[target_col].shift(lag)

# Rolling statistics  
for window in [3, 7, 14]:
    features[f'rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
```

**Features**:
- Lag features (1-5 periods)
- Rolling statistics (mean, std)
- Time-based features (hour, day, month, quarter)
- Ensemble predictions with 100 trees
- Iterative future forecasting

**Use Cases**: Non-linear patterns, feature-rich datasets

### 5. Prophet (`train_prophet`, lines 385-463)

**Facebook's Production Forecasting System**
```python
model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_mode='additive',
    n_changepoints=15,
    mcmc_samples=0  # MAP estimation for speed
)
```

**Features**:
- Automatic seasonality detection
- Trend changepoint detection
- Holiday effects (configurable)
- Uncertainty quantification
- Suppressed logging for clean output

**Use Cases**: Business time series, seasonal patterns, holiday effects

### 6. LightGBM (`train_lightgbm`, lines 465-598)

**High-Performance Gradient Boosting**
```python
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': num_leaves,
    'learning_rate': learning_rate,
    'verbose': -1
}
```

**Features**:
- Extended lag features (1-7 periods)
- Multiple rolling windows (3,7,14,30)
- Difference features for stationarity
- Early stopping for overfitting prevention
- Iterative multi-step forecasting

**Use Cases**: Complex patterns, large datasets, high accuracy requirements

### 7. XGBoost (`train_xgboost`, lines 600-724)

**Extreme Gradient Boosting**
```python
model = xgb.XGBRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=42,
    n_jobs=-1
)
```

**Features**:
- Similar feature engineering to LightGBM
- Regularization for overfitting prevention
- Parallel processing (`n_jobs=-1`)
- Robust error handling for small datasets

**Use Cases**: High-accuracy forecasting, complex non-linear relationships

### 8. SARIMAX (`train_sarimax`, lines 726-786)

**Seasonal ARIMA with External Variables**
```python
model = SARIMAX(
    train_data[self.target_column], 
    order=order, 
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
```

**Features**:
- Seasonal patterns modeling
- Configurable seasonal periods
- Stationarity/invertibility constraints relaxed
- Default seasonal order: (1,1,1,12)

**Use Cases**: Strong seasonal patterns, monthly/yearly data

### 9. LSTM (`train_lstm`, lines 788-899)

**Neural Network Time Series Model**
```python
# Fallback to MLPRegressor for compatibility
model = MLPRegressor(
    hidden_layer_sizes=(hidden_units, hidden_units//2),
    max_iter=epochs,
    random_state=42,
    early_stopping=True
)
```

**Features**:
- Sequence-based modeling
- MinMaxScaler normalization
- Fallback to MLP when LSTM unavailable
- Configurable sequence length (default: 30)
- Multi-step iterative forecasting

**Use Cases**: Complex sequential patterns, long-term dependencies

## Standardized Interface

### Method Signature Pattern
```python
def train_algorithm(self, param1=default1, param2=default2, forecast_periods=30):
    """Train [Algorithm] model"""
    # 1. Data preparation
    train_data, test_data = self._prepare_data()
    
    # 2. Model-specific training
    model = AlgorithmClass(**params)
    model.fit(X_train, y_train)
    
    # 3. Test predictions
    test_predictions = model.predict(X_test)
    metrics = self._calculate_metrics(y_test, test_predictions)
    
    # 4. Future forecasting
    future_forecast = generate_forecast(forecast_periods)
    
    # 5. Standardized return
    return {
        'metrics': metrics,
        'parameters': {...},
        'training_samples': len(train_data),
        'test_samples': len(test_data),
        'forecast_data': forecast_data
    }
```

### Standardized Metrics (`_calculate_metrics`, lines 73-88)
```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
r2 = r2_score(y_true, y_pred)
```

**Metrics Calculated**:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error (handles zero values)
- **R²**: Coefficient of determination

### Standardized Forecast Data Structure
```python
forecast_data = {
    'historical': {
        'dates': ['2023-01-01', ...],
        'values': [100.5, ...]
    },
    'test': {
        'dates': ['2023-12-01', ...],
        'actual': [95.2, ...],
        'predicted': [94.8, ...]
    },
    'forecast': {
        'dates': ['2024-01-01', ...],
        'values': [98.5, ...],
        'confidence_lower': [93.1, ...],
        'confidence_upper': [103.9, ...]
    }
}
```

## Error Handling & Fallbacks

### Dependency Availability Flags
```python
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
```

### Graceful Error Handling
- Import errors: Skip unavailable algorithms
- Training failures: Return detailed error messages
- Data validation: Check minimum requirements
- Fallback implementations: LSTM → MLP, NHITS → LSTM

## Feature Engineering Patterns

### Time-Based Features
- Hour, day of week, month, quarter, day of year
- Week of year, day of month
- Time index (sequential numbering)

### Lag Features
- Previous values: lag_1, lag_2, ..., lag_n
- Configurable lookback window
- Automatic handling of missing values

### Rolling Statistics
- Rolling mean, std, min, max
- Multiple window sizes (3, 7, 14, 30 days)
- Forward-fill for missing values

### Difference Features
- First difference: `data.diff(1)`
- Seasonal difference: `data.diff(7)`
- Stationarity improvement

## Performance Considerations

### Training Speed (Fastest to Slowest)
1. Moving Average (instant)
2. Linear Regression (seconds)
3. ARIMA (seconds to minutes)
4. Random Forest (minutes)
5. LightGBM/XGBoost (minutes)
6. Prophet (minutes)
7. SARIMAX (minutes to hours)
8. LSTM (hours)

### Memory Usage
- Lightweight: ARIMA, Linear Regression, Moving Average
- Moderate: Random Forest, Prophet
- Heavy: LightGBM, XGBoost, LSTM (during training)

### Accuracy Trade-offs
- Baseline: Moving Average, Linear Regression
- Good: ARIMA, Random Forest, Prophet
- Best: LightGBM, XGBoost (with proper tuning)
- Variable: LSTM (data-dependent)
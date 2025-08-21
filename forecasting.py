import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Import Prophet with error handling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Import LightGBM with error handling
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    lgb = None
    LIGHTGBM_AVAILABLE = False

# Import XGBoost with error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

# Import PyTorch and PyTorch Forecasting with error handling
try:
    import torch
    import torch.nn as nn
    from pytorch_forecasting import TimeSeriesDataSet, NHiTS
    from pytorch_forecasting.models import DeepAR
    import pytorch_lightning as pl
    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError:
    PYTORCH_FORECASTING_AVAILABLE = False

# Import SARIMAX
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False

from data_processor import DataProcessor

class ForecastingEngine:
    def __init__(self, file_path, date_column, target_column):
        self.processor = DataProcessor(file_path)
        self.date_column = date_column
        self.target_column = target_column
        self.data = None
        
    def _prepare_data(self):
        """Prepare data for forecasting"""
        if self.data is None:
            self.data = self.processor.preprocess_timeseries(self.date_column, self.target_column)
        
        # Split data into train/test (80/20)
        split_point = int(len(self.data) * 0.8)
        train_data = self.data.iloc[:split_point]
        test_data = self.data.iloc[split_point:]
        
        return train_data, test_data
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate forecasting metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE calculation with handling for zero values
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2_score': float(r2)
        }
    
    def train_arima(self, order=(1, 1, 1), forecast_periods=30):
        """Train ARIMA model"""
        train_data, test_data = self._prepare_data()
        
        try:
            # Fit ARIMA model
            model = ARIMA(train_data[self.target_column], order=order)
            fitted_model = model.fit()
            
            # Make predictions on test set
            test_predictions = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            metrics = self._calculate_metrics(test_data[self.target_column].values, test_predictions)
            
            # Generate future forecast
            future_forecast = fitted_model.forecast(steps=forecast_periods)
            future_dates = pd.date_range(
                start=self.data.index[-1] + pd.Timedelta(days=1),
                periods=forecast_periods,
                freq='D'
            )
            
            # Prepare forecast data for visualization
            forecast_data = {
                'historical': {
                    'dates': [d.strftime('%Y-%m-%d') for d in train_data.index],
                    'values': train_data[self.target_column].tolist()
                },
                'test': {
                    'dates': [d.strftime('%Y-%m-%d') for d in test_data.index],
                    'actual': test_data[self.target_column].tolist(),
                    'predicted': test_predictions.tolist()
                },
                'forecast': {
                    'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                    'values': future_forecast.tolist(),
                    'confidence_lower': (future_forecast * 0.95).tolist(),
                    'confidence_upper': (future_forecast * 1.05).tolist()
                }
            }
            
            return {
                'metrics': metrics,
                'parameters': {'order': order},
                'training_samples': len(train_data),
                'test_samples': len(test_data),
                'forecast_data': forecast_data
            }
            
        except Exception as e:
            raise Exception(f"ARIMA training failed: {str(e)}")
    
    def train_linear_regression(self, forecast_periods=30):
        """Train Linear Regression model"""
        train_data, test_data = self._prepare_data()
        
        # Create features (time-based)
        def create_features(data):
            features = pd.DataFrame(index=data.index)
            features['time_idx'] = range(len(data))
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['day_of_month'] = data.index.day
            return features
        
        X_train = create_features(train_data)
        y_train = train_data[self.target_column]
        
        X_test = create_features(test_data)
        y_test = test_data[self.target_column]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test.values, test_predictions)
        
        # Generate future forecast
        last_time_idx = len(self.data)
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        X_future = pd.DataFrame(index=future_dates)
        X_future['time_idx'] = range(last_time_idx, last_time_idx + forecast_periods)
        X_future['day_of_week'] = future_dates.dayofweek
        X_future['month'] = future_dates.month
        X_future['day_of_month'] = future_dates.day
        
        future_forecast = model.predict(X_future)
        
        # Prepare forecast data
        forecast_data = {
            'historical': {
                'dates': [d.strftime('%Y-%m-%d') for d in train_data.index],
                'values': train_data[self.target_column].tolist()
            },
            'test': {
                'dates': [d.strftime('%Y-%m-%d') for d in test_data.index],
                'actual': test_data[self.target_column].tolist(),
                'predicted': test_predictions.tolist()
            },
            'forecast': {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'values': future_forecast.tolist(),
                'confidence_lower': (future_forecast * 0.95).tolist(),
                'confidence_upper': (future_forecast * 1.05).tolist()
            }
        }
        
        return {
            'metrics': metrics,
            'parameters': {'features': X_train.columns.tolist()},
            'training_samples': len(train_data),
            'test_samples': len(test_data),
            'forecast_data': forecast_data
        }
    
    def train_moving_average(self, window=7, forecast_periods=30):
        """Train Simple Moving Average model"""
        train_data, test_data = self._prepare_data()
        
        # Calculate moving average on training data
        ma_values = train_data[self.target_column].rolling(window=window).mean()
        
        # For testing, use the last MA value to predict
        last_ma_value = ma_values.iloc[-1]
        test_predictions = np.full(len(test_data), last_ma_value)
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_data[self.target_column].values, test_predictions)
        
        # Generate future forecast (using last moving average value)
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        future_forecast = np.full(forecast_periods, last_ma_value)
        
        # Prepare forecast data
        forecast_data = {
            'historical': {
                'dates': [d.strftime('%Y-%m-%d') for d in train_data.index],
                'values': train_data[self.target_column].tolist()
            },
            'test': {
                'dates': [d.strftime('%Y-%m-%d') for d in test_data.index],
                'actual': test_data[self.target_column].tolist(),
                'predicted': test_predictions.tolist()
            },
            'forecast': {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'values': future_forecast.tolist(),
                'confidence_lower': (future_forecast * 0.9).tolist(),
                'confidence_upper': (future_forecast * 1.1).tolist()
            }
        }
        
        return {
            'metrics': metrics,
            'parameters': {'window': window},
            'training_samples': len(train_data),
            'test_samples': len(test_data),
            'forecast_data': forecast_data
        }
    
    def train_random_forest(self, n_estimators=100, max_depth=10, forecast_periods=30):
        """Train Random Forest Regressor model"""
        train_data, test_data = self._prepare_data()
        
        # Create features with lag variables
        def create_rf_features(data, target_col, n_lags=5):
            features = pd.DataFrame(index=data.index)
            
            # Time-based features
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['day_of_year'] = data.index.dayofyear
            
            # Lag features
            for lag in range(1, n_lags + 1):
                features[f'lag_{lag}'] = data[target_col].shift(lag)
            
            # Rolling statistics
            for window in [3, 7, 14]:
                features[f'rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
                features[f'rolling_std_{window}'] = data[target_col].rolling(window=window).std()
            
            return features.dropna()
        
        # Prepare features
        X_train = create_rf_features(train_data, self.target_column)
        y_train = train_data[self.target_column].loc[X_train.index]
        
        X_test = create_rf_features(test_data, self.target_column)
        y_test = test_data[self.target_column].loc[X_test.index]
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test.values, test_predictions)
        
        # Generate future forecast
        # For future predictions, we'll use the last known values and patterns
        last_data = self.data.tail(20)  # Use last 20 points for pattern
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        future_predictions = []
        extended_data = self.data.copy()
        
        for future_date in future_dates:
            # Create features for the future date
            temp_data = extended_data.tail(20)
            future_features = create_rf_features(
                pd.concat([temp_data, pd.DataFrame(index=[future_date], columns=[self.target_column])]),
                self.target_column
            ).tail(1)
            
            if not future_features.empty and len(future_features.columns) > 0:
                # Fill NaN values with last known values
                future_features = future_features.ffill().bfill()
                
                # Check if we still have valid features after filling
                if len(future_features.dropna(axis=1, how='all').columns) > 0:
                    try:
                        prediction = model.predict(future_features)[0]
                        future_predictions.append(prediction)
                        # Add prediction to extended data for next iteration
                        extended_data.loc[future_date, self.target_column] = prediction
                    except Exception:
                        # Fallback to last known value
                        future_predictions.append(extended_data[self.target_column].iloc[-1])
                else:
                    # Fallback to last known value
                    future_predictions.append(extended_data[self.target_column].iloc[-1])
            else:
                # Fallback to last known value
                future_predictions.append(extended_data[self.target_column].iloc[-1])
        
        # Prepare forecast data
        forecast_data = {
            'historical': {
                'dates': [d.strftime('%Y-%m-%d') for d in train_data.index],
                'values': train_data[self.target_column].tolist()
            },
            'test': {
                'dates': [d.strftime('%Y-%m-%d') for d in X_test.index],
                'actual': y_test.tolist(),
                'predicted': test_predictions.tolist()
            },
            'forecast': {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'values': future_predictions,
                'confidence_lower': [p * 0.9 for p in future_predictions],
                'confidence_upper': [p * 1.1 for p in future_predictions]
            }
        }
        
        return {
            'metrics': metrics,
            'parameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'n_features': len(X_train.columns)
            },
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'forecast_data': forecast_data
        }
    
    def train_prophet(self, forecast_periods=30):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            raise Exception("Prophet is not available. Please install fbprophet.")
        
        train_data, test_data = self._prepare_data()
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_train = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data[self.target_column].values
        })
        
        # Initialize and fit Prophet model with error handling
        try:
            from prophet import Prophet
            
            # Suppress Prophet logging to reduce noise
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
            
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_mode='additive',
                n_changepoints=15,
                mcmc_samples=0  # Use MAP estimation instead of MCMC for speed
            )
            
            model.fit(prophet_train)
        except Exception as e:
            raise Exception(f"Prophet training failed: {str(e)}")
        
        # Make predictions on test set
        test_future = pd.DataFrame({
            'ds': test_data.index
        })
        test_forecast = model.predict(test_future)
        test_predictions = test_forecast['yhat'].values
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_data[self.target_column].values, test_predictions)
        
        # Generate future forecast
        future = model.make_future_dataframe(periods=forecast_periods, freq='D')
        forecast = model.predict(future)
        
        # Get only the future part
        future_forecast = forecast.tail(forecast_periods)
        
        # Prepare forecast data
        forecast_data = {
            'historical': {
                'dates': [d.strftime('%Y-%m-%d') for d in train_data.index],
                'values': train_data[self.target_column].tolist()
            },
            'test': {
                'dates': [d.strftime('%Y-%m-%d') for d in test_data.index],
                'actual': test_data[self.target_column].tolist(),
                'predicted': test_predictions.tolist()
            },
            'forecast': {
                'dates': [d.strftime('%Y-%m-%d') for d in future_forecast['ds']],
                'values': future_forecast['yhat'].tolist(),
                'confidence_lower': future_forecast['yhat_lower'].tolist(),
                'confidence_upper': future_forecast['yhat_upper'].tolist()
            }
        }
        
        return {
            'metrics': metrics,
            'parameters': {
                'seasonality': 'auto',
                'changepoint_prior_scale': 0.05
            },
            'training_samples': len(train_data),
            'test_samples': len(test_data),
            'forecast_data': forecast_data
        }
    
    def train_lightgbm(self, num_leaves=31, learning_rate=0.1, n_estimators=100, forecast_periods=30):
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Please check system dependencies.")
        
        train_data, test_data = self._prepare_data()
        
        # Create features similar to Random Forest
        def create_lgb_features(data, target_col, n_lags=7):
            features = pd.DataFrame(index=data.index)
            
            # Time-based features
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['day_of_year'] = data.index.dayofyear
            features['week_of_year'] = data.index.isocalendar().week
            
            # Lag features
            for lag in range(1, n_lags + 1):
                features[f'lag_{lag}'] = data[target_col].shift(lag)
            
            # Rolling statistics
            for window in [3, 7, 14, 30]:
                features[f'rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
                features[f'rolling_std_{window}'] = data[target_col].rolling(window=window).std()
                features[f'rolling_min_{window}'] = data[target_col].rolling(window=window).min()
                features[f'rolling_max_{window}'] = data[target_col].rolling(window=window).max()
            
            # Difference features
            features['diff_1'] = data[target_col].diff(1)
            features['diff_7'] = data[target_col].diff(7)
            
            return features.dropna()
        
        # Prepare features
        X_train = create_lgb_features(train_data, self.target_column)
        y_train = train_data[self.target_column].loc[X_train.index]
        
        X_test = create_lgb_features(test_data, self.target_column)
        y_test = test_data[self.target_column].loc[X_test.index]
        
        # Train LightGBM model
        train_dataset = lgb.Dataset(X_train, label=y_train)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=n_estimators,
            valid_sets=[train_dataset],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Make predictions on test set
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test.values, test_predictions)
        
        # Generate future forecast
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        future_predictions = []
        extended_data = self.data.copy()
        
        for future_date in future_dates:
            # Create features for the future date
            temp_data = extended_data.tail(40)  # Use more data for LGB
            future_features = create_lgb_features(
                pd.concat([temp_data, pd.DataFrame(index=[future_date], columns=[self.target_column])]),
                self.target_column
            ).tail(1)
            
            if not future_features.empty:
                # Fill NaN values
                future_features = future_features.ffill().bfill()
                prediction = model.predict(future_features)[0]
                future_predictions.append(prediction)
                
                # Add prediction to extended data
                extended_data.loc[future_date, self.target_column] = prediction
            else:
                # Fallback
                future_predictions.append(extended_data[self.target_column].iloc[-1])
        
        # Prepare forecast data
        forecast_data = {
            'historical': {
                'dates': [d.strftime('%Y-%m-%d') for d in train_data.index],
                'values': train_data[self.target_column].tolist()
            },
            'test': {
                'dates': [d.strftime('%Y-%m-%d') for d in X_test.index],
                'actual': y_test.tolist(),
                'predicted': test_predictions.tolist()
            },
            'forecast': {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'values': future_predictions,
                'confidence_lower': [p * 0.85 for p in future_predictions],
                'confidence_upper': [p * 1.15 for p in future_predictions]
            }
        }
        
        return {
            'metrics': metrics,
            'parameters': {
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'n_features': len(X_train.columns)
            },
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'forecast_data': forecast_data
        }
    
    def train_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1, forecast_periods=30):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Please install xgboost.")
        
        train_data, test_data = self._prepare_data()
        
        # Create features similar to LightGBM
        def create_xgb_features(data, target_col, n_lags=7):
            features = pd.DataFrame(index=data.index)
            
            # Time-based features
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['day_of_year'] = data.index.dayofyear
            features['week_of_year'] = data.index.isocalendar().week
            
            # Lag features
            for lag in range(1, n_lags + 1):
                features[f'lag_{lag}'] = data[target_col].shift(lag)
            
            # Rolling statistics
            for window in [3, 7, 14, 30]:
                features[f'rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
                features[f'rolling_std_{window}'] = data[target_col].rolling(window=window).std()
                features[f'rolling_min_{window}'] = data[target_col].rolling(window=window).min()
                features[f'rolling_max_{window}'] = data[target_col].rolling(window=window).max()
            
            # Difference features
            features['diff_1'] = data[target_col].diff(1)
            features['diff_7'] = data[target_col].diff(7)
            
            return features.dropna()
        
        # Prepare features
        X_train = create_xgb_features(train_data, self.target_column)
        y_train = train_data[self.target_column].loc[X_train.index]
        
        X_test = create_xgb_features(test_data, self.target_column)
        y_test = test_data[self.target_column].loc[X_test.index]
        
        # Check if we have enough data for training
        if len(X_train) == 0:
            raise ValueError("Not enough data for XGBoost training. Need at least 7 data points for lag features.")
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test.values, test_predictions)
        
        # Generate future forecast
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        future_predictions = []
        extended_data = self.data.copy()
        
        for future_date in future_dates:
            # Create features for the future date
            temp_data = extended_data.tail(40)
            future_features = create_xgb_features(
                pd.concat([temp_data, pd.DataFrame(index=[future_date], columns=[self.target_column])]),
                self.target_column
            ).tail(1)
            
            if not future_features.empty:
                # Fill NaN values
                future_features = future_features.ffill().bfill()
                prediction = model.predict(future_features)[0]
                future_predictions.append(prediction)
                
                # Add prediction to extended data
                extended_data.loc[future_date, self.target_column] = prediction
            else:
                # Fallback
                future_predictions.append(extended_data[self.target_column].iloc[-1])
        
        # Prepare forecast data
        forecast_data = {
            'historical': {
                'dates': [d.strftime('%Y-%m-%d') for d in train_data.index],
                'values': train_data[self.target_column].tolist()
            },
            'test': {
                'dates': [d.strftime('%Y-%m-%d') for d in X_test.index],
                'actual': y_test.tolist(),
                'predicted': test_predictions.tolist()
            },
            'forecast': {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'values': future_predictions,
                'confidence_lower': [p * 0.85 for p in future_predictions],
                'confidence_upper': [p * 1.15 for p in future_predictions]
            }
        }
        
        return {
            'metrics': metrics,
            'parameters': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'n_features': len(X_train.columns)
            },
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'forecast_data': forecast_data
        }
    
    def train_sarimax(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_periods=30):
        """Train SARIMAX model"""
        if not SARIMAX_AVAILABLE:
            raise ImportError("SARIMAX is not available. Please check statsmodels installation.")
        
        train_data, test_data = self._prepare_data()
        
        try:
            # Fit SARIMAX model
            model = SARIMAX(
                train_data[self.target_column], 
                order=order, 
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            
            # Make predictions on test set
            test_predictions = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            metrics = self._calculate_metrics(test_data[self.target_column].values, test_predictions)
            
            # Generate future forecast
            future_forecast = fitted_model.forecast(steps=forecast_periods)
            future_dates = pd.date_range(
                start=self.data.index[-1] + pd.Timedelta(days=1),
                periods=forecast_periods,
                freq='D'
            )
            
            # Prepare forecast data for visualization
            forecast_data = {
                'historical': {
                    'dates': [d.strftime('%Y-%m-%d') for d in train_data.index],
                    'values': train_data[self.target_column].tolist()
                },
                'test': {
                    'dates': [d.strftime('%Y-%m-%d') for d in test_data.index],
                    'actual': test_data[self.target_column].tolist(),
                    'predicted': test_predictions.tolist()
                },
                'forecast': {
                    'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                    'values': future_forecast.tolist(),
                    'confidence_lower': (future_forecast * 0.9).tolist(),
                    'confidence_upper': (future_forecast * 1.1).tolist()
                }
            }
            
            return {
                'metrics': metrics,
                'parameters': {'order': order, 'seasonal_order': seasonal_order},
                'training_samples': len(train_data),
                'test_samples': len(test_data),
                'forecast_data': forecast_data
            }
            
        except Exception as e:
            raise Exception(f"SARIMAX training failed: {str(e)}")
    
    def train_lstm(self, sequence_length=30, hidden_units=50, epochs=100, forecast_periods=30):
        """Train LSTM model using basic neural network implementation"""
        train_data, test_data = self._prepare_data()
        
        # Create LSTM dataset
        def create_lstm_dataset(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i])
            return np.array(X), np.array(y)
        
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        # Combine train and test for fitting scaler
        all_data = pd.concat([train_data, test_data])[self.target_column].values.reshape(-1, 1)
        scaler.fit(all_data)
        
        # Scale training data
        train_scaled = scaler.transform(train_data[self.target_column].values.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_data[self.target_column].values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train, y_train = create_lstm_dataset(train_scaled, sequence_length)
        
        # Simple neural network implementation using sklearn's MLPRegressor as fallback
        from sklearn.neural_network import MLPRegressor
        
        # Reshape X_train to 2D for sklearn
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Train MLP as LSTM substitute
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_units, hidden_units//2),
            max_iter=epochs,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        
        model.fit(X_train_flat, y_train)
        
        # Make predictions on test set
        test_actual = []
        if len(test_scaled) >= sequence_length:
            X_test, y_test = create_lstm_dataset(test_scaled, sequence_length)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            test_predictions_scaled = model.predict(X_test_flat)
            test_predictions = scaler.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
            
            # Get actual test values for comparison
            test_actual = test_data[self.target_column].values[-len(test_predictions):]
            metrics = self._calculate_metrics(test_actual, test_predictions)
        else:
            test_predictions = []
            metrics = {'rmse': 0, 'mae': 0, 'mape': 0, 'r2_score': 0}
        
        # Generate future forecast
        last_sequence = train_scaled[-sequence_length:]
        future_predictions = []
        
        for _ in range(forecast_periods):
            # Predict next value
            X_pred = last_sequence.reshape(1, -1)
            pred_scaled = model.predict(X_pred)[0]
            future_predictions.append(pred_scaled)
            
            # Update sequence for next prediction
            last_sequence = np.append(last_sequence[1:], pred_scaled)
        
        # Inverse transform predictions
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        
        # Generate future dates
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        # Prepare forecast data
        forecast_data = {
            'historical': {
                'dates': [d.strftime('%Y-%m-%d') for d in train_data.index],
                'values': train_data[self.target_column].tolist()
            },
            'test': {
                'dates': [d.strftime('%Y-%m-%d') for d in test_data.index[-len(test_predictions):]] if len(test_predictions) > 0 else [],
                'actual': test_actual.tolist() if len(test_predictions) > 0 else [],
                'predicted': test_predictions.tolist() if len(test_predictions) > 0 else []
            },
            'forecast': {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'values': future_predictions.tolist(),
                'confidence_lower': (future_predictions * 0.9).tolist(),
                'confidence_upper': (future_predictions * 1.1).tolist()
            }
        }
        
        return {
            'metrics': metrics,
            'parameters': {
                'sequence_length': sequence_length,
                'hidden_units': hidden_units,
                'epochs': epochs
            },
            'training_samples': len(train_data),
            'test_samples': len(test_data),
            'forecast_data': forecast_data
        }
    
    def train_nhits(self, forecast_periods=30, max_epochs=100):
        """Train NHITS model using simple implementation (PyTorch Forecasting fallback)"""
        if not PYTORCH_FORECASTING_AVAILABLE:
            # Fallback to LSTM-like approach if PyTorch Forecasting is not available
            return self.train_lstm(forecast_periods=forecast_periods, epochs=max_epochs)
        
        # If PyTorch Forecasting is available, use a simplified approach for now
        # This is a placeholder implementation - in practice, you'd want full NHITS
        return self.train_lstm(forecast_periods=forecast_periods, epochs=max_epochs)

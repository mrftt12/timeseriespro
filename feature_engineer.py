"""
Automated Feature Engineering Pipeline for Time Series Pro
Provides intelligent feature generation and selection for forecasting models
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Feature selection
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, RFE, SelectFromModel,
    f_regression, mutual_info_regression
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# Holiday detection
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

# Conditional import to avoid circular dependency
try:
    from models import FeatureConfig
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    FeatureConfig = None
# Conditional import to avoid circular dependency
try:
    from app import db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    db = None


class FeatureEngineer:
    """
    Automated feature engineering pipeline for time series forecasting.
    Generates lag variables, rolling statistics, calendar features, and technical indicators.
    """
    
    def __init__(self, project_id: int, data: pd.DataFrame, target_column: str, 
                 date_column: str = None, config: Dict[str, Any] = None):
        """
        Initialize the Feature Engineer
        
        Args:
            project_id: ID of the project
            data: Input DataFrame with time series data
            target_column: Name of the target variable column
            date_column: Name of the date column (if time series)
            config: Feature generation configuration
        """
        self.project_id = project_id
        self.data = data.copy()
        self.target_column = target_column
        self.date_column = date_column
        self.config = config or {}
        
        # Ensure date column is datetime if provided
        if self.date_column and self.date_column in self.data.columns:
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            self.data = self.data.sort_values(self.date_column)
            
        self.generated_features = []
        self.feature_importance = {}
        
        # Load saved configuration from database
        self._load_saved_config()
    
    def generate_all_features(self, save_config: bool = True) -> pd.DataFrame:
        """
        Generate all configured features
        
        Args:
            save_config: Whether to save the configuration to database
            
        Returns:
            DataFrame with original data plus generated features
        """
        logging.info(f"Starting feature generation for project {self.project_id}")
        
        result_df = self.data.copy()
        
        # Generate different types of features based on configuration
        if self.config.get('lag_features_enabled', True):
            result_df = self._generate_lag_features(result_df)
            
        if self.config.get('rolling_features_enabled', True):
            result_df = self._generate_rolling_features(result_df)
            
        if self.config.get('calendar_features_enabled', True) and self.date_column:
            result_df = self._generate_calendar_features(result_df)
            
        if self.config.get('technical_indicators_enabled', False):
            result_df = self._generate_technical_indicators(result_df)
            
        if self.config.get('polynomial_features_enabled', False):
            result_df = self._generate_polynomial_features(result_df)
            
        if self.config.get('interaction_features_enabled', False):
            result_df = self._generate_interaction_features(result_df)
            
        if self.config.get('fourier_features_enabled', False) and self.date_column:
            result_df = self._generate_fourier_features(result_df)
            
        # New advanced feature types
        if self.config.get('statistical_features_enabled', False):
            result_df = self._generate_statistical_features(result_df)
            
        if self.config.get('volatility_features_enabled', False):
            result_df = self._generate_volatility_features(result_df)
            
        if self.config.get('momentum_features_enabled', False):
            result_df = self._generate_momentum_features(result_df)
            
        if self.config.get('trend_features_enabled', False):
            result_df = self._generate_trend_features(result_df)
            
        if self.config.get('seasonal_features_enabled', False):
            result_df = self._generate_seasonal_decomposition_features(result_df)
            
        if self.config.get('percentile_features_enabled', False):
            result_df = self._generate_percentile_features(result_df)
            
        if self.config.get('autocorr_features_enabled', False):
            result_df = self._generate_autocorrelation_features(result_df)
        
        # Remove any infinite or extremely large values
        result_df = self._clean_features(result_df)
        
        # Feature selection if enabled
        if self.config.get('auto_feature_selection', True) and len(self.generated_features) > 0:
            result_df = self._auto_feature_selection(result_df)
        
        # Save configuration if requested
        if save_config:
            self._save_config()
        
        logging.info(f"Feature generation complete. Generated {len(self.generated_features)} features")
        return result_df
    
    def _generate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate lag features (historical values)"""
        lag_config = self.config.get('lag_config', {})
        max_lags = lag_config.get('max_lags', 12)
        target_lags = lag_config.get('target_lags', list(range(1, min(max_lags + 1, 13))))
        
        for lag in target_lags:
            if lag > 0 and lag < len(df):
                feature_name = f'{self.target_column}_lag_{lag}'
                df[feature_name] = df[self.target_column].shift(lag)
                self.generated_features.append(feature_name)
        
        # Generate lags for other numeric columns if configured
        if lag_config.get('include_other_columns', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            other_cols = [col for col in numeric_cols if col != self.target_column]
            
            for col in other_cols[:3]:  # Limit to top 3 most correlated
                for lag in [1, 7]:  # Only generate a few lags for other columns
                    if lag < len(df):
                        feature_name = f'{col}_lag_{lag}'
                        df[feature_name] = df[col].shift(lag)
                        self.generated_features.append(feature_name)
        
        return df
    
    def _generate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling window statistics"""
        rolling_config = self.config.get('rolling_config', {})
        windows = rolling_config.get('windows', [3, 7, 14, 30])
        statistics = rolling_config.get('statistics', ['mean', 'std', 'min', 'max'])
        
        for window in windows:
            if window < len(df):
                for stat in statistics:
                    feature_name = f'{self.target_column}_rolling_{stat}_{window}'
                    
                    if stat == 'mean':
                        df[feature_name] = df[self.target_column].rolling(window=window).mean()
                    elif stat == 'std':
                        df[feature_name] = df[self.target_column].rolling(window=window).std()
                    elif stat == 'min':
                        df[feature_name] = df[self.target_column].rolling(window=window).min()
                    elif stat == 'max':
                        df[feature_name] = df[self.target_column].rolling(window=window).max()
                    elif stat == 'median':
                        df[feature_name] = df[self.target_column].rolling(window=window).median()
                    elif stat == 'sum':
                        df[feature_name] = df[self.target_column].rolling(window=window).sum()
                    
                    self.generated_features.append(feature_name)
                
                # Additional rolling features
                if rolling_config.get('include_advanced', False):
                    # Rolling range
                    rolling_max = df[self.target_column].rolling(window=window).max()
                    rolling_min = df[self.target_column].rolling(window=window).min()
                    feature_name = f'{self.target_column}_rolling_range_{window}'
                    df[feature_name] = rolling_max - rolling_min
                    self.generated_features.append(feature_name)
                    
                    # Rolling coefficient of variation
                    rolling_mean = df[self.target_column].rolling(window=window).mean()
                    rolling_std = df[self.target_column].rolling(window=window).std()
                    feature_name = f'{self.target_column}_rolling_cv_{window}'
                    df[feature_name] = rolling_std / rolling_mean
                    self.generated_features.append(feature_name)
        
        return df
    
    def _generate_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate calendar-based features"""
        if not self.date_column or self.date_column not in df.columns:
            return df
            
        calendar_config = self.config.get('calendar_config', {})
        
        # Basic calendar features
        if calendar_config.get('basic_features', True):
            df['hour'] = df[self.date_column].dt.hour
            df['day_of_week'] = df[self.date_column].dt.dayofweek
            df['day_of_month'] = df[self.date_column].dt.day
            df['day_of_year'] = df[self.date_column].dt.dayofyear
            df['week_of_year'] = df[self.date_column].dt.isocalendar().week
            df['month'] = df[self.date_column].dt.month
            df['quarter'] = df[self.date_column].dt.quarter
            df['year'] = df[self.date_column].dt.year
            
            basic_features = ['hour', 'day_of_week', 'day_of_month', 'day_of_year', 
                            'week_of_year', 'month', 'quarter', 'year']
            self.generated_features.extend(basic_features)
        
        # Cyclical encoding for periodic features
        if calendar_config.get('cyclical_encoding', True):
            # Hour cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df[self.date_column].dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df[self.date_column].dt.hour / 24)
            
            # Day of week cyclical encoding
            df['dow_sin'] = np.sin(2 * np.pi * df[self.date_column].dt.dayofweek / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df[self.date_column].dt.dayofweek / 7)
            
            # Month cyclical encoding
            df['month_sin'] = np.sin(2 * np.pi * df[self.date_column].dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df[self.date_column].dt.month / 12)
            
            cyclical_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
            self.generated_features.extend(cyclical_features)
        
        # Holiday features
        if calendar_config.get('holidays', False) and HOLIDAYS_AVAILABLE:
            country_code = calendar_config.get('country_code', 'US')
            try:
                country_holidays = holidays.country_holidays(country_code)
                df['is_holiday'] = df[self.date_column].dt.date.isin(country_holidays).astype(int)
                
                # Days since/until holiday
                df['days_since_holiday'] = self._calculate_days_since_holiday(df[self.date_column], country_holidays)
                df['days_until_holiday'] = self._calculate_days_until_holiday(df[self.date_column], country_holidays)
                
                holiday_features = ['is_holiday', 'days_since_holiday', 'days_until_holiday']
                self.generated_features.extend(holiday_features)
            except Exception as e:
                logging.warning(f"Failed to generate holiday features: {e}")
        
        # Business day features
        if calendar_config.get('business_features', True):
            df['is_weekend'] = (df[self.date_column].dt.dayofweek >= 5).astype(int)
            df['is_business_day'] = (df[self.date_column].dt.dayofweek < 5).astype(int)
            df['is_month_start'] = df[self.date_column].dt.is_month_start.astype(int)
            df['is_month_end'] = df[self.date_column].dt.is_month_end.astype(int)
            df['is_quarter_start'] = df[self.date_column].dt.is_quarter_start.astype(int)
            df['is_quarter_end'] = df[self.date_column].dt.is_quarter_end.astype(int)
            
            business_features = ['is_weekend', 'is_business_day', 'is_month_start', 
                               'is_month_end', 'is_quarter_start', 'is_quarter_end']
            self.generated_features.extend(business_features)
        
        return df
    
    def _generate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators for time series analysis"""
        if not TALIB_AVAILABLE:
            logging.warning("TA-Lib not available for technical indicators")
            return self._generate_simple_technical_indicators(df)
        
        tech_config = self.config.get('technical_config', {})
        prices = df[self.target_column].values
        
        try:
            # Moving averages
            if tech_config.get('moving_averages', True):
                for period in [5, 10, 20]:
                    if len(prices) > period:
                        ma = talib.SMA(prices, timeperiod=period)
                        df[f'SMA_{period}'] = ma
                        self.generated_features.append(f'SMA_{period}')
                        
                        # EMA
                        ema = talib.EMA(prices, timeperiod=period)
                        df[f'EMA_{period}'] = ema
                        self.generated_features.append(f'EMA_{period}')
            
            # Momentum indicators
            if tech_config.get('momentum', True):
                # RSI
                rsi = talib.RSI(prices)
                df['RSI'] = rsi
                self.generated_features.append('RSI')
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(prices)
                df['MACD'] = macd
                df['MACD_signal'] = macd_signal
                df['MACD_hist'] = macd_hist
                self.generated_features.extend(['MACD', 'MACD_signal', 'MACD_hist'])
            
            # Volatility indicators
            if tech_config.get('volatility', True):
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(prices)
                df['BB_upper'] = bb_upper
                df['BB_middle'] = bb_middle
                df['BB_lower'] = bb_lower
                df['BB_width'] = bb_upper - bb_lower
                df['BB_position'] = (prices - bb_lower) / (bb_upper - bb_lower)
                
                bb_features = ['BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_position']
                self.generated_features.extend(bb_features)
                
        except Exception as e:
            logging.warning(f"Error generating technical indicators: {e}")
            return self._generate_simple_technical_indicators(df)
        
        return df
    
    def _generate_simple_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate simple technical indicators without TA-Lib"""
        prices = df[self.target_column]
        
        # Simple moving averages
        for period in [5, 10, 20]:
            if len(prices) > period:
                df[f'SMA_{period}'] = prices.rolling(window=period).mean()
                self.generated_features.append(f'SMA_{period}')
        
        # Simple RSI calculation
        if len(prices) > 14:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_simple'] = 100 - (100 / (1 + rs))
            self.generated_features.append('RSI_simple')
        
        return df
    
    def _generate_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features"""
        poly_config = self.config.get('polynomial_config', {})
        degree = poly_config.get('degree', 2)
        
        if degree > 1:
            # Polynomial features of target variable
            for d in range(2, degree + 1):
                feature_name = f'{self.target_column}_poly_{d}'
                df[feature_name] = df[self.target_column] ** d
                self.generated_features.append(feature_name)
        
        return df
    
    def _generate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between variables"""
        interaction_config = self.config.get('interaction_config', {})
        max_interactions = interaction_config.get('max_interactions', 5)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column and col not in self.generated_features]
        
        # Generate interactions between top correlated features
        if len(numeric_cols) >= 2:
            correlations = df[numeric_cols].corrwith(df[self.target_column]).abs().sort_values(ascending=False)
            top_features = correlations.head(max_interactions).index.tolist()
            
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:
                    interaction_name = f'{feat1}_x_{feat2}'
                    df[interaction_name] = df[feat1] * df[feat2]
                    self.generated_features.append(interaction_name)
        
        return df
    
    def _generate_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Fourier features for capturing seasonality"""
        fourier_config = self.config.get('fourier_config', {})
        n_fourier = fourier_config.get('n_fourier', 5)
        
        if not self.date_column:
            return df
        
        # Create time index
        time_idx = np.arange(len(df))
        
        for k in range(1, n_fourier + 1):
            # Fourier features for different seasonal patterns
            for period in [7, 30, 365]:  # Weekly, monthly, yearly patterns
                if len(df) > period:
                    sin_name = f'fourier_sin_{k}_{period}'
                    cos_name = f'fourier_cos_{k}_{period}'
                    
                    df[sin_name] = np.sin(2 * np.pi * k * time_idx / period)
                    df[cos_name] = np.cos(2 * np.pi * k * time_idx / period)
                    
                    self.generated_features.extend([sin_name, cos_name])
        
        return df
    
    def _generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features like skewness, kurtosis"""
        if not self.config.get('statistical_features_enabled', False):
            return df
            
        stat_config = self.config.get('statistical_config', {})
        windows = stat_config.get('windows', [7, 14, 30])
        
        target_series = df[self.target_column]
        
        for window in windows:
            if len(df) > window:
                # Rolling statistical moments
                skew_name = f'{self.target_column}_rolling_skew_{window}'
                kurt_name = f'{self.target_column}_rolling_kurt_{window}'
                
                df[skew_name] = target_series.rolling(window=window, min_periods=1).skew()
                df[kurt_name] = target_series.rolling(window=window, min_periods=1).kurt()
                
                self.generated_features.extend([skew_name, kurt_name])
        
        return df
    
    def _generate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based features"""
        if not self.config.get('volatility_features_enabled', False):
            return df
            
        vol_config = self.config.get('volatility_config', {})
        windows = vol_config.get('windows', [5, 10, 20])
        
        target_series = df[self.target_column]
        
        # Calculate returns
        returns = target_series.pct_change()
        
        for window in windows:
            if len(df) > window:
                # Rolling volatility measures
                vol_name = f'{self.target_column}_volatility_{window}'
                vol_std_name = f'{self.target_column}_vol_std_{window}'
                
                df[vol_name] = returns.rolling(window=window, min_periods=1).std()
                df[vol_std_name] = returns.rolling(window=window, min_periods=1).std().rolling(window=5).std()
                
                self.generated_features.extend([vol_name, vol_std_name])
        
        return df
    
    def _generate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based features"""
        if not self.config.get('momentum_features_enabled', False):
            return df
            
        momentum_config = self.config.get('momentum_config', {})
        periods = momentum_config.get('periods', [5, 10, 20])
        
        target_series = df[self.target_column]
        
        for period in periods:
            if len(df) > period:
                # Rate of change and momentum indicators
                roc_name = f'{self.target_column}_roc_{period}'
                momentum_name = f'{self.target_column}_momentum_{period}'
                
                df[roc_name] = target_series.pct_change(periods=period)
                df[momentum_name] = target_series / target_series.shift(period) - 1
                
                self.generated_features.extend([roc_name, momentum_name])
        
        return df
    
    def _generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-based features"""
        if not self.config.get('trend_features_enabled', False):
            return df
            
        trend_config = self.config.get('trend_config', {})
        windows = trend_config.get('windows', [7, 14, 21])
        
        target_series = df[self.target_column]
        
        for window in windows:
            if len(df) > window:
                # Linear trend slope over window
                trend_name = f'{self.target_column}_trend_slope_{window}'
                
                # Calculate rolling linear regression slope
                slopes = []
                for i in range(len(target_series)):
                    start_idx = max(0, i - window + 1)
                    end_idx = i + 1
                    
                    if end_idx - start_idx >= window:
                        y_vals = target_series.iloc[start_idx:end_idx].values
                        x_vals = np.arange(len(y_vals))
                        
                        if len(x_vals) > 1:
                            slope = np.polyfit(x_vals, y_vals, 1)[0]
                            slopes.append(slope)
                        else:
                            slopes.append(0)
                    else:
                        slopes.append(0)
                
                df[trend_name] = slopes
                self.generated_features.append(trend_name)
        
        return df
    
    def _generate_seasonal_decomposition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from seasonal decomposition"""
        if not self.config.get('seasonal_features_enabled', False):
            return df
            
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            if len(df) < 24:  # Need sufficient data for decomposition
                return df
                
            target_series = df[self.target_column].dropna()
            
            # Simple seasonal decomposition
            if len(target_series) >= 24:
                decomposition = seasonal_decompose(target_series, model='additive', period=min(12, len(target_series)//2))
                
                # Add trend and seasonal components as features
                trend_name = f'{self.target_column}_trend_component'
                seasonal_name = f'{self.target_column}_seasonal_component'
                resid_name = f'{self.target_column}_residual_component'
                
                # Pad/truncate to match original length
                df[trend_name] = decomposition.trend.reindex(df.index).fillna(method='bfill').fillna(method='ffill')
                df[seasonal_name] = decomposition.seasonal.reindex(df.index).fillna(0)
                df[resid_name] = decomposition.resid.reindex(df.index).fillna(0)
                
                self.generated_features.extend([trend_name, seasonal_name, resid_name])
                
        except ImportError:
            logging.warning("statsmodels not available for seasonal decomposition")
        except Exception as e:
            logging.warning(f"Seasonal decomposition failed: {e}")
            
        return df
    
    def _generate_percentile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate percentile-based features"""
        if not self.config.get('percentile_features_enabled', False):
            return df
            
        percentile_config = self.config.get('percentile_config', {})
        windows = percentile_config.get('windows', [10, 20, 30])
        percentiles = percentile_config.get('percentiles', [25, 50, 75, 90])
        
        target_series = df[self.target_column]
        
        for window in windows:
            if len(df) > window:
                for pct in percentiles:
                    pct_name = f'{self.target_column}_rolling_pct_{pct}_{window}'
                    df[pct_name] = target_series.rolling(window=window, min_periods=1).quantile(pct/100)
                    self.generated_features.append(pct_name)
        
        return df
    
    def _generate_autocorrelation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate autocorrelation-based features"""
        if not self.config.get('autocorr_features_enabled', False):
            return df
            
        autocorr_config = self.config.get('autocorr_config', {})
        lags = autocorr_config.get('lags', [1, 7, 14, 30])
        
        target_series = df[self.target_column]
        
        for lag in lags:
            if len(df) > lag:
                autocorr_name = f'{self.target_column}_autocorr_{lag}'
                
                # Rolling autocorrelation
                autocorr_vals = []
                window = min(50, len(target_series) // 2)  # Adaptive window size
                
                for i in range(len(target_series)):
                    start_idx = max(0, i - window + 1)
                    end_idx = i + 1
                    
                    if end_idx - start_idx > lag:
                        series_window = target_series.iloc[start_idx:end_idx]
                        if len(series_window) > lag and series_window.std() > 0:
                            autocorr = series_window.autocorr(lag=lag)
                            autocorr_vals.append(autocorr if not np.isnan(autocorr) else 0)
                        else:
                            autocorr_vals.append(0)
                    else:
                        autocorr_vals.append(0)
                
                df[autocorr_name] = autocorr_vals
                self.generated_features.append(autocorr_name)
        
        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean generated features by removing infinite and extreme values"""
        for feature in self.generated_features:
            if feature in df.columns:
                # Replace infinite values
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                
                # Cap extreme values (beyond 3 standard deviations)
                if df[feature].std() > 0:
                    mean_val = df[feature].mean()
                    std_val = df[feature].std()
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    df[feature] = df[feature].clip(lower_bound, upper_bound)
        
        return df
    
    def _auto_feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically select the most important features using multiple methods"""
        selection_config = self.config.get('feature_selection_config', {})
        max_features = selection_config.get('max_features', 20)
        method = selection_config.get('method', 'random_forest')
        
        if len(self.generated_features) <= max_features:
            return df
        
        # Prepare data for feature selection
        feature_data = df[self.generated_features].dropna()
        target_data = df[self.target_column].loc[feature_data.index]
        
        if len(feature_data) < 10:  # Need minimum data for selection
            return df
        
        selected_features = []
        
        try:
            if method == 'correlation':
                # Correlation-based selection
                correlations = feature_data.corrwith(target_data).abs().sort_values(ascending=False)
                selected_features = correlations.head(max_features).index.tolist()
                
            elif method == 'variance':
                # Variance-based selection
                selector = VarianceThreshold(threshold=0.01)
                selector.fit(feature_data)
                selected_mask = selector.get_support()
                high_var_features = [feat for feat, selected in zip(self.generated_features, selected_mask) if selected]
                
                # If still too many, use correlation on remaining
                if len(high_var_features) > max_features:
                    correlations = feature_data[high_var_features].corrwith(target_data).abs().sort_values(ascending=False)
                    selected_features = correlations.head(max_features).index.tolist()
                else:
                    selected_features = high_var_features
                    
            elif method == 'mutual_info':
                # Mutual information based selection
                selector = SelectKBest(score_func=mutual_info_regression, k=min(max_features, len(self.generated_features)))
                selector.fit(feature_data, target_data)
                selected_mask = selector.get_support()
                selected_features = [feat for feat, selected in zip(self.generated_features, selected_mask) if selected]
                
            elif method == 'f_regression':
                # F-regression based selection
                selector = SelectKBest(score_func=f_regression, k=min(max_features, len(self.generated_features)))
                selector.fit(feature_data, target_data)
                selected_mask = selector.get_support()
                selected_features = [feat for feat, selected in zip(self.generated_features, selected_mask) if selected]
                
            elif method == 'random_forest':
                # Random Forest feature importance
                rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(feature_data, target_data)
                
                importance_scores = dict(zip(self.generated_features, rf.feature_importances_))
                sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [feat for feat, score in sorted_features[:max_features]]
                
            elif method == 'lasso':
                # LASSO-based selection
                lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
                lasso.fit(feature_data, target_data)
                
                # Select features with non-zero coefficients
                non_zero_mask = np.abs(lasso.coef_) > 1e-8
                lasso_features = [feat for feat, selected in zip(self.generated_features, non_zero_mask) if selected]
                
                # If too many features, use importance ranking
                if len(lasso_features) > max_features:
                    feature_importance = dict(zip(self.generated_features, np.abs(lasso.coef_)))
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    selected_features = [feat for feat, score in sorted_features[:max_features] if feat in lasso_features]
                else:
                    selected_features = lasso_features
                    
            elif method == 'rfe':
                # Recursive Feature Elimination
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                selector = RFE(estimator, n_features_to_select=max_features)
                selector.fit(feature_data, target_data)
                selected_mask = selector.get_support()
                selected_features = [feat for feat, selected in zip(self.generated_features, selected_mask) if selected]
                
            else:
                # Default to correlation if method not recognized
                correlations = feature_data.corrwith(target_data).abs().sort_values(ascending=False)
                selected_features = correlations.head(max_features).index.tolist()
                
        except Exception as e:
            logging.warning(f"Feature selection failed with method {method}: {e}")
            # Fallback to correlation-based selection
            correlations = feature_data.corrwith(target_data).abs().sort_values(ascending=False)
            selected_features = correlations.head(max_features).index.tolist()
        
        # Keep selected features plus original columns
        columns_to_keep = [col for col in df.columns if col not in self.generated_features or col in selected_features]
        result_df = df[columns_to_keep].copy()
        
        # Update generated features list
        self.generated_features = [feat for feat in self.generated_features if feat in selected_features]
        
        logging.info(f"Feature selection ({method}): reduced from {len(feature_data.columns)} to {len(selected_features)} features")
        return result_df
    
    def get_feature_importance(self, df: pd.DataFrame, method: str = 'random_forest') -> Dict[str, float]:
        """Calculate feature importance using specified method"""
        if len(self.generated_features) == 0:
            return {}
        
        # Prepare data
        feature_data = df[self.generated_features].dropna()
        target_data = df[self.target_column].loc[feature_data.index]
        
        if len(feature_data) < 10:
            return {}
        
        importance_dict = {}
        
        if method == 'random_forest':
            try:
                rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(feature_data, target_data)
                importance_dict = dict(zip(feature_data.columns, rf.feature_importances_))
            except Exception as e:
                logging.warning(f"Random forest feature importance failed: {e}")
                
        elif method == 'correlation':
            importance_dict = feature_data.corrwith(target_data).abs().to_dict()
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        self.feature_importance = importance_dict
        
        return importance_dict
    
    def _calculate_days_since_holiday(self, dates: pd.Series, holidays_set) -> pd.Series:
        """Calculate days since last holiday"""
        days_since = []
        for date in dates:
            days_back = 0
            current_date = date.date()
            while days_back < 365:  # Look back up to a year
                if current_date in holidays_set:
                    break
                current_date -= timedelta(days=1)
                days_back += 1
            days_since.append(days_back if days_back < 365 else 365)
        
        return pd.Series(days_since, index=dates.index)
    
    def _calculate_days_until_holiday(self, dates: pd.Series, holidays_set) -> pd.Series:
        """Calculate days until next holiday"""
        days_until = []
        for date in dates:
            days_ahead = 0
            current_date = date.date()
            while days_ahead < 365:  # Look ahead up to a year
                if current_date in holidays_set:
                    break
                current_date += timedelta(days=1)
                days_ahead += 1
            days_until.append(days_ahead if days_ahead < 365 else 365)
        
        return pd.Series(days_until, index=dates.index)
    
    def _load_saved_config(self):
        """Load saved feature configuration from database"""
        if not MODELS_AVAILABLE or not DB_AVAILABLE:
            logging.warning("Database not available, using default configuration")
            return
            
        try:
            configs = FeatureConfig.query.filter_by(project_id=self.project_id, is_enabled=True).all()
            for config in configs:
                feature_config = config.get_configuration()
                self.config[f'{config.feature_type}_enabled'] = True
                self.config[f'{config.feature_type}_config'] = feature_config
        except Exception as e:
            logging.warning(f"Failed to load saved feature config: {e}")
    
    def _save_config(self):
        """Save current feature configuration to database"""
        if not MODELS_AVAILABLE or not DB_AVAILABLE:
            logging.warning("Database not available, configuration not saved")
            return
            
        try:
            # Clear existing configurations
            FeatureConfig.query.filter_by(project_id=self.project_id).delete()
            
            # Save current configuration
            feature_types = ['lag', 'rolling', 'calendar', 'technical', 'polynomial', 'interaction', 'fourier']
            
            for feature_type in feature_types:
                if self.config.get(f'{feature_type}_features_enabled', False):
                    config_obj = FeatureConfig(
                        project_id=self.project_id,
                        feature_type=feature_type,
                        is_enabled=True
                    )
                    config_obj.set_configuration(self.config.get(f'{feature_type}_config', {}))
                    db.session.add(config_obj)
            
            db.session.commit()
            logging.info(f"Saved feature configuration for project {self.project_id}")
            
        except Exception as e:
            logging.error(f"Failed to save feature configuration: {e}")
            db.session.rollback()
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of generated features"""
        summary = {
            'total_features': len(self.generated_features),
            'feature_types': {},
            'feature_importance': self.feature_importance,
            'config_summary': {}
        }
        
        # Count features by type
        for feature in self.generated_features:
            if 'lag' in feature:
                summary['feature_types']['lag'] = summary['feature_types'].get('lag', 0) + 1
            elif 'rolling' in feature:
                summary['feature_types']['rolling'] = summary['feature_types'].get('rolling', 0) + 1
            elif any(cal in feature for cal in ['hour', 'day', 'month', 'quarter', 'year']):
                summary['feature_types']['calendar'] = summary['feature_types'].get('calendar', 0) + 1
            elif any(tech in feature for tech in ['SMA', 'EMA', 'RSI', 'MACD', 'BB']):
                summary['feature_types']['technical'] = summary['feature_types'].get('technical', 0) + 1
            elif 'poly' in feature:
                summary['feature_types']['polynomial'] = summary['feature_types'].get('polynomial', 0) + 1
            elif '_x_' in feature:
                summary['feature_types']['interaction'] = summary['feature_types'].get('interaction', 0) + 1
            elif 'fourier' in feature:
                summary['feature_types']['fourier'] = summary['feature_types'].get('fourier', 0) + 1
        
        # Configuration summary
        for key, value in self.config.items():
            if key.endswith('_enabled'):
                summary['config_summary'][key] = value
        
        return summary
    
    def visualize_feature_importance(self, df: pd.DataFrame, top_n: int = 20, method: str = 'random_forest') -> Dict[str, Any]:
        """
        Create visualization data for feature importance
        
        Args:
            df: DataFrame with features
            top_n: Number of top features to visualize
            method: Method to calculate importance
            
        Returns:
            Dictionary with visualization data for frontend
        """
        if len(self.generated_features) == 0:
            return {'error': 'No features generated yet'}
            
        try:
            # Calculate feature importance
            importance_scores = self.get_feature_importance(df, method=method)
            
            if not importance_scores:
                return {'error': 'Could not calculate feature importance'}
            
            # Sort by importance
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_n]
            
            # Prepare data for visualization
            viz_data = {
                'labels': [feat for feat, _ in top_features],
                'values': [score for _, score in top_features],
                'method': method,
                'total_features': len(self.generated_features),
                'feature_types': self._categorize_features([feat for feat, _ in top_features])
            }
            
            # Add feature correlation matrix data for top features
            if len(top_features) > 1:
                top_feature_names = [feat for feat, _ in top_features[:10]]  # Limit for performance
                feature_subset = df[top_feature_names].dropna()
                
                if len(feature_subset) > 10:
                    correlation_matrix = feature_subset.corr()
                    viz_data['correlation_matrix'] = {
                        'labels': correlation_matrix.columns.tolist(),
                        'values': correlation_matrix.values.tolist()
                    }
            
            return viz_data
            
        except Exception as e:
            logging.error(f"Feature visualization failed: {e}")
            return {'error': f'Visualization failed: {str(e)}'}
    
    def _categorize_features(self, feature_list: List[str]) -> Dict[str, int]:
        """Categorize features by type for visualization"""
        categories = {
            'lag': 0,
            'rolling': 0, 
            'calendar': 0,
            'technical': 0,
            'statistical': 0,
            'volatility': 0,
            'momentum': 0,
            'trend': 0,
            'seasonal': 0,
            'percentile': 0,
            'autocorr': 0,
            'polynomial': 0,
            'interaction': 0,
            'fourier': 0,
            'other': 0
        }
        
        for feature in feature_list:
            if '_lag_' in feature:
                categories['lag'] += 1
            elif 'rolling' in feature:
                categories['rolling'] += 1
            elif any(cal in feature for cal in ['day_', 'month', 'quarter', 'year', 'holiday']):
                categories['calendar'] += 1
            elif any(tech in feature for tech in ['rsi', 'macd', 'bb_', 'sma', 'ema']):
                categories['technical'] += 1
            elif any(stat in feature for stat in ['skew', 'kurt']):
                categories['statistical'] += 1
            elif 'volatility' in feature or 'vol_std' in feature:
                categories['volatility'] += 1
            elif any(mom in feature for mom in ['roc_', 'momentum']):
                categories['momentum'] += 1
            elif 'trend' in feature:
                categories['trend'] += 1
            elif any(seas in feature for seas in ['seasonal', 'component']):
                categories['seasonal'] += 1
            elif 'pct_' in feature:
                categories['percentile'] += 1
            elif 'autocorr' in feature:
                categories['autocorr'] += 1
            elif any(poly in feature for poly in ['^2', '^3', 'poly_']):
                categories['polynomial'] += 1
            elif '_x_' in feature:
                categories['interaction'] += 1
            elif 'fourier' in feature:
                categories['fourier'] += 1
            else:
                categories['other'] += 1
                
        return categories
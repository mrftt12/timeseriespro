import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        """Load data from CSV or Excel file"""
        try:
            if self.file_path.endswith('.csv'):
                self.data = pd.read_csv(self.file_path)
            elif self.file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format")
            
            return self.data
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_timeseries(self, date_column, target_column):
        """Preprocess time series data"""
        if self.data is None:
            self.load_data()
        
        df = self.data.copy()
        
        # Convert date column to datetime
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            raise ValueError(f"Could not convert {date_column} to datetime: {str(e)}")
        
        # Check if target column exists and is numeric
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} not found in data")
        
        try:
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        except Exception as e:
            raise ValueError(f"Could not convert {target_column} to numeric: {str(e)}")
        
        # Sort by date
        df = df.sort_values(date_column)
        
        # Handle missing values
        missing_count = df[target_column].isnull().sum()
        if missing_count > 0:
            logging.warning(f"Found {missing_count} missing values in target column. Using forward fill.")
            df[target_column] = df[target_column].fillna(method='ffill')
            df[target_column] = df[target_column].fillna(method='bfill')
        
        # Remove any remaining missing values
        df = df.dropna(subset=[date_column, target_column])
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after preprocessing")
        
        # Set date as index
        df.set_index(date_column, inplace=True)
        
        return df
    
    def validate_timeseries(self, df, date_column, target_column):
        """Validate time series data quality"""
        issues = []
        recommendations = []
        
        # Check data length
        if len(df) < 30:
            issues.append("Dataset is very small (< 30 observations)")
            recommendations.append("Consider collecting more data for better forecasting accuracy")
        
        # Check for duplicated dates
        if df.index.duplicated().any():
            issues.append("Duplicate dates found in the dataset")
            recommendations.append("Remove or aggregate duplicate dates")
        
        # Check for large gaps in dates
        date_diff = df.index.to_series().diff()
        median_diff = date_diff.median()
        large_gaps = date_diff > median_diff * 3
        
        if large_gaps.any():
            issues.append(f"Found {large_gaps.sum()} large gaps in the time series")
            recommendations.append("Consider filling gaps or using appropriate interpolation")
        
        # Check for outliers (simple z-score method)
        z_scores = np.abs((df[target_column] - df[target_column].mean()) / df[target_column].std())
        outliers = z_scores > 3
        
        if outliers.any():
            issues.append(f"Found {outliers.sum()} potential outliers")
            recommendations.append("Review outliers and consider appropriate treatment")
        
        # Check for trend and seasonality
        rolling_mean = df[target_column].rolling(window=min(12, len(df)//4)).mean()
        if not rolling_mean.dropna().empty:
            trend_change = abs(rolling_mean.iloc[-1] - rolling_mean.iloc[0]) / rolling_mean.iloc[0]
            if trend_change > 0.5:
                recommendations.append("Strong trend detected - consider trend-aware models")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'data_quality_score': max(0, 100 - len(issues) * 15)
        }
    
    def get_data_summary(self, df, target_column):
        """Get summary statistics for the dataset"""
        summary = {
            'count': len(df),
            'mean': df[target_column].mean(),
            'std': df[target_column].std(),
            'min': df[target_column].min(),
            'max': df[target_column].max(),
            'start_date': df.index.min().strftime('%Y-%m-%d'),
            'end_date': df.index.max().strftime('%Y-%m-%d'),
            'frequency': self._infer_frequency(df)
        }
        
        return summary
    
    def _infer_frequency(self, df):
        """Infer the frequency of the time series"""
        if len(df) < 2:
            return "Unknown"
        
        date_diff = df.index.to_series().diff().dropna()
        mode_diff = date_diff.mode()[0] if not date_diff.mode().empty else date_diff.median()
        
        if mode_diff <= pd.Timedelta(days=1):
            return "Daily"
        elif mode_diff <= pd.Timedelta(days=7):
            return "Weekly"
        elif mode_diff <= pd.Timedelta(days=31):
            return "Monthly"
        elif mode_diff <= pd.Timedelta(days=92):
            return "Quarterly"
        elif mode_diff <= pd.Timedelta(days=366):
            return "Yearly"
        else:
            return "Unknown"

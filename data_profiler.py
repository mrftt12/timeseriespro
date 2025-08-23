"""
Comprehensive Data Profiling Engine for Time Series Pro
Provides statistical analysis, quality assessment, and actionable recommendations
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro

# Time series analysis imports
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Machine learning imports for outlier detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from data_processor import DataProcessor

# Database imports - handle circular import gracefully
try:
    from models import DataProfile
    from app import db
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


class DataProfiler:
    """
    Comprehensive data profiling engine that analyzes datasets and provides
    statistical insights, quality metrics, and actionable recommendations.
    """
    
    def __init__(self, project_id: int, file_path: str, date_column: str = None, target_column: str = None):
        """
        Initialize the DataProfiler
        
        Args:
            project_id: ID of the project this profile belongs to
            file_path: Path to the dataset file
            date_column: Name of the date/time column
            target_column: Name of the target variable column
        """
        self.project_id = project_id
        self.file_path = file_path
        self.date_column = date_column
        self.target_column = target_column
        self.data = None
        self.profile_results = {}
        self.processor = DataProcessor(file_path)
        
    def analyze_dataset(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis
        
        Args:
            force_refresh: If True, recalculate even if cached results exist
            
        Returns:
            Dictionary containing complete profile analysis
        """
        if not force_refresh:
            # Try to load cached results
            cached_profile = self._load_cached_profile()
            if cached_profile:
                return cached_profile
        
        logging.info(f"Starting comprehensive data profiling for project {self.project_id}")
        
        # Load and prepare data
        self._load_data()
        
        # Perform all analyses
        results = {
            'basic_statistics': self._calculate_basic_statistics(),
            'column_analysis': self._analyze_all_columns(),
            'data_quality': self._assess_data_quality(),
            'outlier_analysis': self._comprehensive_outlier_analysis(),
            'correlation_analysis': self._analyze_correlations(),
            'time_series_analysis': self._analyze_time_series_properties(),
            'distribution_analysis': self._analyze_distributions(),
            'missing_data_analysis': self._analyze_missing_data(),
            'recommendations': self._generate_recommendations(),
            'quality_score': 0.0,  # Will be calculated based on all factors
            'profile_metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'dataset_shape': self.data.shape,
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024
            }
        }
        
        # Calculate overall quality score
        results['quality_score'] = self._calculate_quality_score(results)
        
        # Cache results
        self._cache_profile_results(results)
        
        return results
    
    def _load_data(self):
        """Load and preprocess the dataset"""
        try:
            self.data = self.processor.load_data()
            logging.info(f"Loaded dataset with shape: {self.data.shape}")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def _calculate_basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistical measures for the dataset"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        # Exclude boolean from numeric
        boolean_cols = self.data.select_dtypes(include=['bool']).columns
        numeric_cols = numeric_cols.difference(boolean_cols)
        
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
        
        return {
            'dataset_shape': {
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'numeric_columns': len(numeric_cols),
                'boolean_columns': len(boolean_cols),
                'categorical_columns': len(categorical_cols),
                'datetime_columns': len(datetime_cols)
            },
            'column_types': {
                'numeric': numeric_cols.tolist(),
                'boolean': boolean_cols.tolist(),
                'categorical': categorical_cols.tolist(), 
                'datetime': datetime_cols.tolist()
            },
            'memory_usage': {
                'total_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024,
                'per_column': {col: self.data[col].memory_usage(deep=True) / 1024 / 1024 
                              for col in self.data.columns}
            }
        }
    
    def _analyze_all_columns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze each column individually"""
        column_profiles = {}
        
        for column in self.data.columns:
            column_profiles[column] = self._analyze_single_column(column)
        
        return column_profiles
    
    def _analyze_single_column(self, column: str) -> Dict[str, Any]:
        """Analyze a single column in detail"""
        series = self.data[column]
        
        profile = {
            'data_type': str(series.dtype),
            'non_null_count': series.count(),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100,
        }
        
        # Numeric column analysis (but not boolean)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            profile.update(self._analyze_numeric_column(series))
        
        # Boolean column analysis
        elif pd.api.types.is_bool_dtype(series):
            profile.update(self._analyze_boolean_column(series))
        
        # Categorical column analysis
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            profile.update(self._analyze_categorical_column(series))
        
        # Datetime column analysis
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile.update(self._analyze_datetime_column(series))
        
        return profile
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column properties"""
        numeric_series = series.dropna()
        
        if len(numeric_series) == 0:
            return {'error': 'No valid numeric data'}
        
        profile = {
            'statistics': {
                'mean': float(numeric_series.mean()),
                'median': float(numeric_series.median()),
                'mode': float(numeric_series.mode().iloc[0]) if not numeric_series.mode().empty else None,
                'std': float(numeric_series.std()),
                'var': float(numeric_series.var()),
                'min': float(numeric_series.min()),
                'max': float(numeric_series.max()),
                'range': float(numeric_series.max() - numeric_series.min()),
                'q25': float(numeric_series.quantile(0.25)),
                'q75': float(numeric_series.quantile(0.75)),
                'iqr': float(numeric_series.quantile(0.75) - numeric_series.quantile(0.25)),
                'skewness': float(numeric_series.skew()),
                'kurtosis': float(numeric_series.kurtosis())
            },
            'distribution_tests': self._test_distribution(numeric_series),
            'outliers': self._detect_outliers_single_column(numeric_series),
            'zeros_count': int((numeric_series == 0).sum()),
            'negative_count': int((numeric_series < 0).sum()),
            'positive_count': int((numeric_series > 0).sum())
        }
        
        return profile
    
    def _analyze_boolean_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze boolean column properties"""
        boolean_series = series.dropna()
        
        if len(boolean_series) == 0:
            return {'error': 'No valid boolean data'}
        
        true_count = boolean_series.sum()
        false_count = len(boolean_series) - true_count
        
        profile = {
            'true_count': int(true_count),
            'false_count': int(false_count),
            'true_percentage': float((true_count / len(boolean_series)) * 100),
            'false_percentage': float((false_count / len(boolean_series)) * 100),
            'balance_ratio': float(min(true_count, false_count) / max(true_count, false_count)) if max(true_count, false_count) > 0 else 0
        }
        
        return profile
    
    def _analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical column properties"""
        value_counts = series.value_counts()
        
        profile = {
            'value_counts': value_counts.head(20).to_dict(),  # Top 20 values
            'most_frequent': value_counts.index[0] if not value_counts.empty else None,
            'most_frequent_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
            'cardinality': len(value_counts),
            'cardinality_ratio': len(value_counts) / len(series.dropna()) if len(series.dropna()) > 0 else 0,
            'min_length': int(series.dropna().astype(str).str.len().min()) if not series.dropna().empty else 0,
            'max_length': int(series.dropna().astype(str).str.len().max()) if not series.dropna().empty else 0,
            'mean_length': float(series.dropna().astype(str).str.len().mean()) if not series.dropna().empty else 0
        }
        
        return profile
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column properties"""
        datetime_series = series.dropna()
        
        if len(datetime_series) == 0:
            return {'error': 'No valid datetime data'}
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(datetime_series):
            try:
                datetime_series = pd.to_datetime(datetime_series)
            except:
                return {'error': 'Cannot convert to datetime'}
        
        profile = {
            'min_date': datetime_series.min().isoformat(),
            'max_date': datetime_series.max().isoformat(),
            'date_range_days': (datetime_series.max() - datetime_series.min()).days,
            'frequency_analysis': self._analyze_datetime_frequency(datetime_series),
            'gaps_analysis': self._analyze_datetime_gaps(datetime_series)
        }
        
        return profile
    
    def _analyze_datetime_frequency(self, datetime_series: pd.Series) -> Dict[str, Any]:
        """Analyze the frequency pattern of datetime data"""
        if len(datetime_series) < 2:
            return {'inferred_frequency': 'unknown', 'reason': 'insufficient_data'}
        
        # Calculate differences between consecutive dates
        sorted_dates = datetime_series.sort_values()
        date_diffs = sorted_dates.diff().dropna()
        
        # Find the most common difference
        mode_diff = date_diffs.mode().iloc[0] if not date_diffs.mode().empty else date_diffs.median()
        
        # Infer frequency
        if mode_diff <= pd.Timedelta(hours=1):
            frequency = 'hourly'
        elif mode_diff <= pd.Timedelta(days=1):
            frequency = 'daily'
        elif mode_diff <= pd.Timedelta(days=7):
            frequency = 'weekly'
        elif mode_diff <= pd.Timedelta(days=31):
            frequency = 'monthly'
        elif mode_diff <= pd.Timedelta(days=92):
            frequency = 'quarterly'
        elif mode_diff <= pd.Timedelta(days=366):
            frequency = 'yearly'
        else:
            frequency = 'irregular'
        
        return {
            'inferred_frequency': frequency,
            'mode_difference': str(mode_diff),
            'median_difference': str(date_diffs.median()),
            'std_difference': str(date_diffs.std()),
            'regularity_score': self._calculate_datetime_regularity(date_diffs)
        }
    
    def _analyze_datetime_gaps(self, datetime_series: pd.Series) -> Dict[str, Any]:
        """Analyze gaps in datetime data"""
        sorted_dates = datetime_series.sort_values()
        date_diffs = sorted_dates.diff().dropna()
        
        if len(date_diffs) == 0:
            return {'total_gaps': 0}
        
        median_diff = date_diffs.median()
        large_gaps = date_diffs > median_diff * 3
        
        return {
            'total_gaps': int(large_gaps.sum()),
            'largest_gap': str(date_diffs.max()),
            'gaps_over_3x_median': int(large_gaps.sum()),
            'gap_locations': sorted_dates[large_gaps].dt.strftime('%Y-%m-%d').tolist()[:10]  # First 10 gaps
        }
    
    def _calculate_datetime_regularity(self, date_diffs: pd.Series) -> float:
        """Calculate how regular the datetime intervals are (0-1 score)"""
        if len(date_diffs) == 0:
            return 0.0
        
        # Calculate coefficient of variation for date differences
        cv = date_diffs.std() / date_diffs.mean() if date_diffs.mean() > pd.Timedelta(0) else float('inf')
        
        # Convert to 0-1 score (lower CV = higher regularity)
        regularity_score = max(0, 1 - min(cv, 1))
        
        return float(regularity_score)
    
    def _test_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Test the distribution of a numeric series"""
        tests = {}
        
        try:
            # Shapiro-Wilk test for normality (good for small samples)
            if len(series) <= 5000:  # Shapiro-Wilk has sample size limitations
                shapiro_stat, shapiro_p = shapiro(series)
                tests['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
        except Exception as e:
            tests['shapiro_wilk'] = {'error': str(e)}
        
        try:
            # Jarque-Bera test for normality
            jb_stat, jb_p = jarque_bera(series)
            tests['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'is_normal': jb_p > 0.05
            }
        except Exception as e:
            tests['jarque_bera'] = {'error': str(e)}
        
        try:
            # D'Agostino-Pearson test for normality
            dp_stat, dp_p = normaltest(series)
            tests['dagostino_pearson'] = {
                'statistic': float(dp_stat),
                'p_value': float(dp_p),
                'is_normal': dp_p > 0.05
            }
        except Exception as e:
            tests['dagostino_pearson'] = {'error': str(e)}
        
        return tests
    
    def _detect_outliers_single_column(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers in a single numeric column using multiple methods"""
        outliers = {}
        
        # Z-score method
        z_scores = np.abs((series - series.mean()) / series.std())
        z_outliers = z_scores > 3
        outliers['z_score'] = {
            'count': int(z_outliers.sum()),
            'percentage': float((z_outliers.sum() / len(series)) * 100),
            'indices': series[z_outliers].index.tolist()[:20]  # First 20 outliers
        }
        
        # IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))
        outliers['iqr'] = {
            'count': int(iqr_outliers.sum()),
            'percentage': float((iqr_outliers.sum() / len(series)) * 100),
            'indices': series[iqr_outliers].index.tolist()[:20]
        }
        
        # Modified Z-score (more robust)
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else np.zeros_like(series)
        modified_z_outliers = np.abs(modified_z_scores) > 3.5
        outliers['modified_z_score'] = {
            'count': int(modified_z_outliers.sum()),
            'percentage': float((modified_z_outliers.sum() / len(series)) * 100),
            'indices': series[modified_z_outliers].index.tolist()[:20]
        }
        
        return outliers
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess overall data quality"""
        issues = []
        recommendations = []
        quality_factors = {}
        
        # Missing data assessment
        total_missing = self.data.isnull().sum().sum()
        total_cells = len(self.data) * len(self.data.columns)
        missing_percentage = (total_missing / total_cells) * 100
        
        quality_factors['missing_data'] = {
            'total_missing': int(total_missing),
            'missing_percentage': float(missing_percentage),
            'score': max(0, 100 - missing_percentage * 2)  # Penalty for missing data
        }
        
        if missing_percentage > 10:
            issues.append(f"High missing data rate: {missing_percentage:.1f}%")
            recommendations.append("Consider imputation strategies or data collection improvements")
        
        # Duplicate rows assessment
        duplicate_count = self.data.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(self.data)) * 100
        
        quality_factors['duplicates'] = {
            'count': int(duplicate_count),
            'percentage': float(duplicate_percentage),
            'score': max(0, 100 - duplicate_percentage * 5)
        }
        
        if duplicate_percentage > 5:
            issues.append(f"High duplicate rate: {duplicate_percentage:.1f}%")
            recommendations.append("Remove or investigate duplicate records")
        
        # Data type consistency
        mixed_type_columns = []
        for col in self.data.select_dtypes(include=['object']).columns:
            if self.data[col].apply(lambda x: type(x).__name__).nunique() > 1:
                mixed_type_columns.append(col)
        
        quality_factors['type_consistency'] = {
            'mixed_type_columns': mixed_type_columns,
            'score': max(0, 100 - len(mixed_type_columns) * 10)
        }
        
        if mixed_type_columns:
            issues.append(f"Mixed data types in columns: {mixed_type_columns}")
            recommendations.append("Standardize data types for consistent analysis")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'quality_factors': quality_factors,
            'overall_score': np.mean([factor['score'] for factor in quality_factors.values()])
        }
    
    def _comprehensive_outlier_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive outlier analysis across all numeric columns"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        # Exclude boolean columns from outlier analysis
        boolean_cols = self.data.select_dtypes(include=['bool']).columns
        numeric_cols = numeric_cols.difference(boolean_cols)
        
        if len(numeric_cols) == 0:
            return {'error': 'No numeric columns for outlier analysis'}
        
        outlier_summary = {
            'methods_available': ['z_score', 'iqr', 'modified_z_score'],
            'column_outliers': {},
            'overall_outlier_statistics': {}
        }
        
        # Add advanced methods if sklearn is available
        if SKLEARN_AVAILABLE:
            outlier_summary['methods_available'].extend(['isolation_forest', 'local_outlier_factor'])
            outlier_summary.update(self._advanced_outlier_detection(numeric_cols))
        
        # Analyze outliers for each numeric column
        for col in numeric_cols:
            series = self.data[col].dropna()
            if len(series) > 0:
                outlier_summary['column_outliers'][col] = self._detect_outliers_single_column(series)
        
        # Calculate overall outlier statistics
        total_outliers_z = sum([
            outlier_summary['column_outliers'][col]['z_score']['count'] 
            for col in outlier_summary['column_outliers']
        ])
        total_outliers_iqr = sum([
            outlier_summary['column_outliers'][col]['iqr']['count'] 
            for col in outlier_summary['column_outliers']
        ])
        
        outlier_summary['overall_outlier_statistics'] = {
            'total_outliers_z_score': total_outliers_z,
            'total_outliers_iqr': total_outliers_iqr,
            'outlier_percentage_z': (total_outliers_z / len(self.data)) * 100,
            'outlier_percentage_iqr': (total_outliers_iqr / len(self.data)) * 100
        }
        
        return outlier_summary
    
    def _advanced_outlier_detection(self, numeric_cols: List[str]) -> Dict[str, Any]:
        """Advanced outlier detection using machine learning methods"""
        if not SKLEARN_AVAILABLE:
            return {}
        
        # Prepare data for ML-based outlier detection
        numeric_data = self.data[numeric_cols].dropna()
        
        if len(numeric_data) < 10:  # Need minimum data for ML methods
            return {'advanced_methods': 'insufficient_data'}
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        advanced_outliers = {}
        
        try:
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_outliers = iso_forest.fit_predict(scaled_data) == -1
            
            advanced_outliers['isolation_forest'] = {
                'count': int(iso_outliers.sum()),
                'percentage': float((iso_outliers.sum() / len(numeric_data)) * 100),
                'indices': numeric_data[iso_outliers].index.tolist()[:20]
            }
        except Exception as e:
            advanced_outliers['isolation_forest'] = {'error': str(e)}
        
        try:
            # Local Outlier Factor
            lof = LocalOutlierFactor(n_neighbors=min(20, len(numeric_data)-1))
            lof_outliers = lof.fit_predict(scaled_data) == -1
            
            advanced_outliers['local_outlier_factor'] = {
                'count': int(lof_outliers.sum()),
                'percentage': float((lof_outliers.sum() / len(numeric_data)) * 100),
                'indices': numeric_data[lof_outliers].index.tolist()[:20]
            }
        except Exception as e:
            advanced_outliers['local_outlier_factor'] = {'error': str(e)}
        
        return {'advanced_outlier_methods': advanced_outliers}
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between numeric variables"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        # Exclude boolean columns from correlation analysis
        boolean_cols = self.data.select_dtypes(include=['bool']).columns
        numeric_cols = numeric_cols.difference(boolean_cols)
        
        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        correlation_matrix = self.data[numeric_cols].corr()
        
        # Find high correlations (excluding self-correlations)
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if not pd.isna(corr_value) and abs(corr_value) > 0.7:
                    high_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations,
            'correlation_with_target': self._target_correlations(numeric_cols) if self.target_column else None
        }
    
    def _target_correlations(self, numeric_cols: List[str]) -> Dict[str, float]:
        """Calculate correlations with target column"""
        if not self.target_column or self.target_column not in self.data.columns:
            return {}
        
        target_correlations = {}
        for col in numeric_cols:
            if col != self.target_column:
                corr = self.data[col].corr(self.data[self.target_column])
                if not pd.isna(corr):
                    target_correlations[col] = float(corr)
        
        # Sort by absolute correlation value
        return dict(sorted(target_correlations.items(), key=lambda x: abs(x[1]), reverse=True))
    
    def _analyze_time_series_properties(self) -> Dict[str, Any]:
        """Analyze time series specific properties"""
        if not self.date_column or not self.target_column:
            return {'error': 'Date and target columns required for time series analysis'}
        
        if self.date_column not in self.data.columns or self.target_column not in self.data.columns:
            return {'error': 'Date or target column not found in dataset'}
        
        # Prepare time series data
        try:
            ts_data = self.data[[self.date_column, self.target_column]].copy()
            ts_data[self.date_column] = pd.to_datetime(ts_data[self.date_column])
            ts_data = ts_data.sort_values(self.date_column)
            ts_data.set_index(self.date_column, inplace=True)
            ts_data = ts_data.dropna()
        except Exception as e:
            return {'error': f'Failed to prepare time series data: {str(e)}'}
        
        if len(ts_data) < 10:
            return {'error': 'Insufficient time series data'}
        
        time_series_analysis = {}
        
        # Stationarity tests
        if STATSMODELS_AVAILABLE:
            time_series_analysis['stationarity'] = self._test_stationarity(ts_data[self.target_column])
        
        # Trend analysis
        time_series_analysis['trend'] = self._analyze_trend(ts_data[self.target_column])
        
        # Seasonality analysis
        if len(ts_data) >= 24:  # Need sufficient data for seasonality
            time_series_analysis['seasonality'] = self._analyze_seasonality(ts_data[self.target_column])
        
        # Autocorrelation analysis
        if STATSMODELS_AVAILABLE and len(ts_data) >= 10:
            time_series_analysis['autocorrelation'] = self._analyze_autocorrelation(ts_data[self.target_column])
        
        return time_series_analysis
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Test time series stationarity using ADF and KPSS tests"""
        stationarity_results = {}
        
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(series.dropna())
            stationarity_results['adf_test'] = {
                'statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                'is_stationary': adf_result[1] < 0.05
            }
        except Exception as e:
            stationarity_results['adf_test'] = {'error': str(e)}
        
        try:
            # KPSS test
            kpss_result = kpss(series.dropna())
            stationarity_results['kpss_test'] = {
                'statistic': float(kpss_result[0]),
                'p_value': float(kpss_result[1]),
                'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                'is_stationary': kpss_result[1] > 0.05
            }
        except Exception as e:
            stationarity_results['kpss_test'] = {'error': str(e)}
        
        return stationarity_results
    
    def _analyze_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze trend in time series"""
        # Simple trend analysis using linear regression
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        valid_idx = ~np.isnan(y)
        x_clean = x[valid_idx]
        y_clean = y[valid_idx]
        
        if len(x_clean) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat',
            'trend_strength': 'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.3 else 'weak'
        }
    
    def _analyze_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze seasonality in time series"""
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not available'}
        
        try:
            # Seasonal decomposition
            decomposition = seasonal_decompose(series.dropna(), model='additive', period=min(12, len(series)//4))
            
            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            seasonal_strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0
            
            return {
                'seasonal_strength': float(seasonal_strength),
                'has_seasonality': seasonal_strength > 0.1,
                'seasonal_pattern': 'strong' if seasonal_strength > 0.3 else 'moderate' if seasonal_strength > 0.1 else 'weak'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_autocorrelation(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze autocorrelation structure"""
        try:
            # Ljung-Box test for autocorrelation
            lb_stat, lb_p = acorr_ljungbox(series.dropna(), lags=min(10, len(series)//4), return_df=False)
            
            return {
                'ljung_box_statistic': float(lb_stat.iloc[-1]) if hasattr(lb_stat, 'iloc') else float(lb_stat),
                'ljung_box_p_value': float(lb_p.iloc[-1]) if hasattr(lb_p, 'iloc') else float(lb_p),
                'has_autocorrelation': (float(lb_p.iloc[-1]) if hasattr(lb_p, 'iloc') else float(lb_p)) < 0.05
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze distributions of numeric columns"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        # Exclude boolean columns from distribution analysis
        boolean_cols = self.data.select_dtypes(include=['bool']).columns
        numeric_cols = numeric_cols.difference(boolean_cols)
        
        if len(numeric_cols) == 0:
            return {'error': 'No numeric columns for distribution analysis'}
        
        distributions = {}
        
        for col in numeric_cols:
            series = self.data[col].dropna()
            if len(series) > 0:
                distributions[col] = {
                    'histogram_data': self._calculate_histogram(series),
                    'distribution_shape': self._describe_distribution_shape(series),
                    'normality_assessment': self._assess_normality(series)
                }
        
        return distributions
    
    def _calculate_histogram(self, series: pd.Series, bins: int = 20) -> Dict[str, Any]:
        """Calculate histogram data for a series"""
        hist, bin_edges = np.histogram(series, bins=bins)
        
        return {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_centers': ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
        }
    
    def _describe_distribution_shape(self, series: pd.Series) -> Dict[str, Any]:
        """Describe the shape characteristics of a distribution"""
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        # Interpret skewness
        if abs(skewness) < 0.5:
            skew_desc = 'approximately symmetric'
        elif skewness > 0.5:
            skew_desc = 'right-skewed (positive skew)'
        else:
            skew_desc = 'left-skewed (negative skew)'
        
        # Interpret kurtosis
        if kurtosis < 1:
            kurt_desc = 'platykurtic (thin tails)'
        elif kurtosis > 3:
            kurt_desc = 'leptokurtic (heavy tails)'
        else:
            kurt_desc = 'mesokurtic (normal-like tails)'
        
        return {
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'skewness_description': skew_desc,
            'kurtosis_description': kurt_desc
        }
    
    def _assess_normality(self, series: pd.Series) -> Dict[str, Any]:
        """Assess how close a distribution is to normal"""
        normality_tests = self._test_distribution(series)
        
        # Count how many tests suggest normality
        normal_votes = 0
        total_tests = 0
        
        for test_name, test_result in normality_tests.items():
            if 'error' not in test_result and 'is_normal' in test_result:
                total_tests += 1
                if test_result['is_normal']:
                    normal_votes += 1
        
        normality_score = normal_votes / total_tests if total_tests > 0 else 0
        
        return {
            'normality_score': normality_score,
            'assessment': 'likely normal' if normality_score > 0.5 else 'likely non-normal',
            'test_results': normality_tests
        }
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_analysis = {
            'total_missing_cells': int(self.data.isnull().sum().sum()),
            'missing_percentage': float((self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100),
            'columns_with_missing': {},
            'missing_patterns': {}
        }
        
        # Analyze missing data by column
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                missing_analysis['columns_with_missing'][col] = {
                    'count': int(missing_count),
                    'percentage': float((missing_count / len(self.data)) * 100)
                }
        
        # Analyze missing data patterns (combinations of missing columns)
        if len(missing_analysis['columns_with_missing']) > 0:
            missing_patterns = self.data[list(missing_analysis['columns_with_missing'].keys())].isnull()
            pattern_counts = missing_patterns.value_counts()
            
            missing_analysis['missing_patterns'] = {
                str(pattern): int(count) for pattern, count in pattern_counts.head(10).items()
            }
        
        return missing_analysis
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Recommendations based on missing data
        total_missing_pct = (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        if total_missing_pct > 5:
            recommendations.append({
                'category': 'data_quality',
                'priority': 'high' if total_missing_pct > 20 else 'medium',
                'issue': f'High missing data rate: {total_missing_pct:.1f}%',
                'recommendation': 'Consider imputation strategies or investigate data collection process',
                'action': 'Implement missing value treatment before modeling'
            })
        
        # Recommendations based on outliers
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        # Exclude boolean columns from outlier recommendations
        boolean_cols = self.data.select_dtypes(include=['bool']).columns
        numeric_cols = numeric_cols.difference(boolean_cols)
        if len(numeric_cols) > 0:
            total_outliers = 0
            for col in numeric_cols:
                series = self.data[col].dropna()
                if len(series) > 0:
                    z_scores = np.abs((series - series.mean()) / series.std())
                    total_outliers += (z_scores > 3).sum()
            
            outlier_pct = (total_outliers / len(self.data)) * 100
            if outlier_pct > 5:
                recommendations.append({
                    'category': 'outliers',
                    'priority': 'medium',
                    'issue': f'High outlier rate: {outlier_pct:.1f}%',
                    'recommendation': 'Review outliers and consider appropriate treatment',
                    'action': 'Investigate outlier causes and apply suitable treatment methods'
                })
        
        # Recommendations based on data size
        if len(self.data) < 100:
            recommendations.append({
                'category': 'sample_size',
                'priority': 'high',
                'issue': f'Small dataset: {len(self.data)} rows',
                'recommendation': 'Collect more data for better model performance',
                'action': 'Consider data augmentation or collecting additional samples'
            })
        
        # Recommendations for time series
        if self.date_column and self.target_column:
            if self.date_column in self.data.columns and self.target_column in self.data.columns:
                try:
                    ts_data = self.data[[self.date_column, self.target_column]].copy()
                    ts_data[self.date_column] = pd.to_datetime(ts_data[self.date_column])
                    
                    # Check for irregular frequency
                    date_diffs = ts_data[self.date_column].diff().dropna()
                    cv = date_diffs.std() / date_diffs.mean() if date_diffs.mean() > pd.Timedelta(0) else float('inf')
                    
                    if cv > 0.3:  # High coefficient of variation indicates irregular frequency
                        recommendations.append({
                            'category': 'time_series',
                            'priority': 'medium',
                            'issue': 'Irregular time series frequency detected',
                            'recommendation': 'Consider resampling or interpolation to regular frequency',
                            'action': 'Use resampling methods to create regular time intervals'
                        })
                except:
                    pass
        
        return recommendations
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        scores = []
        
        # Missing data score
        if 'missing_data_analysis' in results:
            missing_pct = results['missing_data_analysis']['missing_percentage']
            missing_score = max(0, 100 - missing_pct * 2)
            scores.append(missing_score)
        
        # Data quality score
        if 'data_quality' in results and 'overall_score' in results['data_quality']:
            scores.append(results['data_quality']['overall_score'])
        
        # Outlier score
        if 'outlier_analysis' in results and 'overall_outlier_statistics' in results['outlier_analysis']:
            outlier_pct = results['outlier_analysis']['overall_outlier_statistics'].get('outlier_percentage_iqr', 0)
            outlier_score = max(0, 100 - outlier_pct * 3)
            scores.append(outlier_score)
        
        # Sample size score
        sample_size_score = min(100, len(self.data) / 10)  # Score based on sample size
        scores.append(sample_size_score)
        
        return float(np.mean(scores)) if scores else 50.0
    
    def _load_cached_profile(self) -> Optional[Dict[str, Any]]:
        """Load cached profile results from database"""
        if not DATABASE_AVAILABLE:
            return None
            
        try:
            profiles = DataProfile.query.filter_by(project_id=self.project_id).all()
            if not profiles:
                return None
            
            # Combine all profiles into a comprehensive result
            cached_result = {
                'basic_statistics': {},
                'column_analysis': {},
                'profile_metadata': {
                    'generated_at': max(profile.created_at for profile in profiles).isoformat(),
                    'cached': True
                }
            }
            
            for profile in profiles:
                cached_result['column_analysis'][profile.column_name] = {
                    'data_type': profile.data_type,
                    'missing_count': profile.missing_count,
                    'missing_percentage': profile.missing_percentage,
                    'outlier_count': profile.outlier_count,
                    'statistical_summary': profile.get_statistical_summary(),
                    'quality_score': profile.quality_score,
                    'recommendations': profile.get_recommendations()
                }
            
            return cached_result
        except Exception as e:
            logging.warning(f"Failed to load cached profile: {e}")
            return None
    
    def _cache_profile_results(self, results: Dict[str, Any]):
        """Cache profile results in database"""
        if not DATABASE_AVAILABLE:
            return
            
        try:
            # Clear existing profiles for this project
            DataProfile.query.filter_by(project_id=self.project_id).delete()
            
            # Store column-level profiles
            if 'column_analysis' in results:
                for column_name, column_data in results['column_analysis'].items():
                    profile = DataProfile(
                        project_id=self.project_id,
                        column_name=column_name,
                        data_type=column_data.get('data_type', 'unknown'),
                        missing_count=column_data.get('null_count', 0),
                        missing_percentage=column_data.get('null_percentage', 0.0),
                        outlier_count=self._get_outlier_count(column_data),
                        quality_score=results.get('quality_score', 0.0)
                    )
                    
                    # Set JSON fields
                    if 'statistics' in column_data:
                        profile.set_statistical_summary(column_data['statistics'])
                    
                    profile.set_recommendations(results.get('recommendations', []))
                    
                    db.session.add(profile)
            
            db.session.commit()
            logging.info(f"Cached profile results for project {self.project_id}")
            
        except Exception as e:
            logging.error(f"Failed to cache profile results: {e}")
            db.session.rollback()
    
    def _get_outlier_count(self, column_data: Dict[str, Any]) -> int:
        """Extract outlier count from column data"""
        if 'outliers' in column_data and 'iqr' in column_data['outliers']:
            return column_data['outliers']['iqr']['count']
        return 0
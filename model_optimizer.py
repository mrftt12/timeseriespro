"""
Hyperparameter Optimization Framework for Time Series Pro
Provides Bayesian optimization using Optuna for all forecasting algorithms
"""

import optuna
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Core imports
from forecasting import ForecastingEngine, LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE, PROPHET_AVAILABLE, SARIMAX_AVAILABLE, PYTORCH_FORECASTING_AVAILABLE

# Database integration
try:
    from app import db
    from models import OptimizationExperiment
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    db = None
    OptimizationExperiment = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Hyperparameter optimization framework using Optuna for Bayesian optimization
    Supports all forecasting algorithms with configurable objectives and parallel trials
    """
    
    def __init__(self, project_id: int, data: pd.DataFrame, target_column: str, 
                 date_column: str = None, objective: str = 'rmse', test_size: float = 0.2):
        """
        Initialize the model optimizer
        
        Args:
            project_id: Project identifier for tracking experiments
            data: Time series data
            target_column: Name of the target variable column
            date_column: Name of the date column (optional)
            objective: Optimization objective ('rmse', 'mae', 'mape', 'r2')
            test_size: Fraction of data to use for validation
        """
        self.project_id = project_id
        self.data = data.copy()
        self.target_column = target_column
        self.date_column = date_column
        self.objective = objective.lower()
        self.test_size = test_size
        
        # Split data for validation
        split_idx = int(len(data) * (1 - test_size))
        self.train_data = data.iloc[:split_idx].copy()
        self.test_data = data.iloc[split_idx:].copy()
        
        # Store file path for ForecastingEngine creation in individual trials
        self.file_path = None  # Will be set per trial if needed
        
        # Available algorithms
        self.available_algorithms = {
            'arima': True,
            'linear_regression': True,
            'moving_average': True,
            'random_forest': True,
            'prophet': PROPHET_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE,
            'sarimax': SARIMAX_AVAILABLE,
            'lstm': PYTORCH_FORECASTING_AVAILABLE,
            'nhits': PYTORCH_FORECASTING_AVAILABLE
        }
        
        # Objective functions
        self.objective_functions = {
            'rmse': self._calculate_rmse,
            'mae': self._calculate_mae,
            'mape': self._calculate_mape,
            'r2': self._calculate_r2
        }
        
        logger.info(f"ModelOptimizer initialized for project {project_id}")
        logger.info(f"Available algorithms: {[k for k, v in self.available_algorithms.items() if v]}")
    
    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² Score (return negative for minimization)"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return -r2  # Negative for minimization
    
    def _train_and_predict(self, algorithm: str, params: Dict[str, Any]) -> np.ndarray:
        """Train model and make predictions on test data"""
        if algorithm == 'random_forest':
            return self._train_random_forest(params)
        elif algorithm == 'lightgbm':
            return self._train_lightgbm(params)
        elif algorithm == 'xgboost':
            return self._train_xgboost(params)
        elif algorithm == 'arima':
            return self._train_arima(params)
        elif algorithm == 'moving_average':
            return self._train_moving_average(params)
        else:
            raise ValueError(f"Algorithm {algorithm} not implemented for optimization")
    
    def _train_random_forest(self, params: Dict[str, Any]) -> np.ndarray:
        """Train Random Forest model"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare features (simple lag features for now)
        X_train = self._create_features(self.train_data)
        y_train = self.train_data[self.target_column].iloc[1:]  # Skip first row due to lag
        
        X_test = self._create_features(self.test_data)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        return model.predict(X_test)
    
    def _train_lightgbm(self, params: Dict[str, Any]) -> np.ndarray:
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM not available")
        
        import lightgbm as lgb
        
        # Prepare features
        X_train = self._create_features(self.train_data)
        y_train = self.train_data[self.target_column].iloc[1:]
        
        X_test = self._create_features(self.test_data)
        
        # Train model
        model = lgb.LGBMRegressor(
            num_leaves=params['num_leaves'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        return model.predict(X_test)
    
    def _train_xgboost(self, params: Dict[str, Any]) -> np.ndarray:
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost not available")
        
        import xgboost as xgb
        
        # Prepare features
        X_train = self._create_features(self.train_data)
        y_train = self.train_data[self.target_column].iloc[1:]
        
        X_test = self._create_features(self.test_data)
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        return model.predict(X_test)
    
    def _train_arima(self, params: Dict[str, Any]) -> np.ndarray:
        """Train ARIMA model"""
        from statsmodels.tsa.arima.model import ARIMA
        
        train_values = self.train_data[self.target_column].values
        test_length = len(self.test_data)
        
        # Train ARIMA model
        model = ARIMA(train_values, order=params['order'])
        fitted_model = model.fit()
        
        # Make predictions
        forecast = fitted_model.forecast(steps=test_length)
        return forecast
    
    def _train_moving_average(self, params: Dict[str, Any]) -> np.ndarray:
        """Train Moving Average model"""
        window = params['window']
        train_values = self.train_data[self.target_column].values
        test_length = len(self.test_data)
        
        # Calculate moving average
        ma = np.convolve(train_values, np.ones(window)/window, mode='valid')
        last_ma = ma[-1] if len(ma) > 0 else train_values.mean()
        
        # Return constant prediction (simple MA)
        return np.full(test_length, last_ma)
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create simple lag features for ML models"""
        features = pd.DataFrame()
        
        # Add lag feature
        features['lag_1'] = data[self.target_column].shift(1)
        
        # Add rolling mean
        features['rolling_mean_3'] = data[self.target_column].rolling(window=3).mean()
        
        # Drop first row with NaN values
        features = features.iloc[1:].ffill()
        
        return features
    
    def _evaluate_model(self, algorithm: str, params: Dict[str, Any]) -> float:
        """
        Evaluate a model with given parameters on validation data
        
        Args:
            algorithm: Algorithm name
            params: Hyperparameters dictionary
            
        Returns:
            Objective function value
        """
        try:
            # Use direct model training instead of ForecastingEngine for optimization
            predictions = self._train_and_predict(algorithm, params)
            
            # Get actual values (skip first row to align with predictions)
            actual = self.test_data[self.target_column].iloc[1:].values
            
            # Align lengths if needed
            min_len = min(len(predictions), len(actual))
            predictions = predictions[:min_len]
            actual = actual[:min_len]
            
            # Calculate objective
            objective_func = self.objective_functions[self.objective]
            score = objective_func(actual, predictions)
            
            return score
            
        except Exception as e:
            logger.warning(f"Trial failed for {algorithm} with params {params}: {str(e)}")
            return float('inf')  # Return worst possible score for failed trials
    
    def optimize_arima(self, n_trials: int = 50, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize ARIMA model hyperparameters
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results dictionary
        """
        def objective(trial):
            # Define search space for ARIMA parameters
            p = trial.suggest_int('p', 0, 5)
            d = trial.suggest_int('d', 0, 2) 
            q = trial.suggest_int('q', 0, 5)
            
            params = {'order': (p, d, q)}
            return self._evaluate_model('arima', params)
        
        return self._run_optimization('arima', objective, n_trials, timeout)
    
    def optimize_random_forest(self, n_trials: int = 100, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters
        
        Args:
            n_trials: Number of optimization trials  
            timeout: Timeout in seconds
            
        Returns:
            Optimization results dictionary
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20)
            }
            return self._evaluate_model('random_forest', params)
        
        return self._run_optimization('random_forest', objective, n_trials, timeout)
    
    def optimize_lightgbm(self, n_trials: int = 100, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results dictionary
        """
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not available")
        
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500)
            }
            return self._evaluate_model('lightgbm', params)
        
        return self._run_optimization('lightgbm', objective, n_trials, timeout)
    
    def optimize_xgboost(self, n_trials: int = 100, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results dictionary
        """
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not available")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            }
            return self._evaluate_model('xgboost', params)
        
        return self._run_optimization('xgboost', objective, n_trials, timeout)
    
    def optimize_sarimax(self, n_trials: int = 50, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize SARIMAX model hyperparameters
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results dictionary
        """
        if not SARIMAX_AVAILABLE:
            raise ValueError("SARIMAX is not available")
        
        def objective(trial):
            # ARIMA order parameters
            p = trial.suggest_int('p', 0, 3)
            d = trial.suggest_int('d', 0, 2)
            q = trial.suggest_int('q', 0, 3)
            
            # Seasonal parameters
            seasonal_p = trial.suggest_int('seasonal_p', 0, 2)
            seasonal_d = trial.suggest_int('seasonal_d', 0, 1)
            seasonal_q = trial.suggest_int('seasonal_q', 0, 2)
            s = trial.suggest_categorical('s', [12, 4, 7])  # Monthly, quarterly, weekly
            
            params = {
                'order': (p, d, q),
                'seasonal_order': (seasonal_p, seasonal_d, seasonal_q, s)
            }
            return self._evaluate_model('sarimax', params)
        
        return self._run_optimization('sarimax', objective, n_trials, timeout)
    
    def optimize_lstm(self, n_trials: int = 50, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize LSTM hyperparameters
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results dictionary
        """
        if not PYTORCH_FORECASTING_AVAILABLE:
            raise ValueError("PyTorch Forecasting is not available")
        
        def objective(trial):
            params = {
                'sequence_length': trial.suggest_int('sequence_length', 10, 60),
                'hidden_units': trial.suggest_categorical('hidden_units', [32, 64, 128, 256]),
                'epochs': trial.suggest_int('epochs', 20, 150)
            }
            return self._evaluate_model('lstm', params)
        
        return self._run_optimization('lstm', objective, n_trials, timeout)
    
    def optimize_moving_average(self, n_trials: int = 20, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize Moving Average window size
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results dictionary
        """
        def objective(trial):
            params = {
                'window': trial.suggest_int('window', 3, 30)
            }
            return self._evaluate_model('moving_average', params)
        
        return self._run_optimization('moving_average', objective, n_trials, timeout)
    
    def _run_optimization(self, algorithm: str, objective_func: Callable, 
                         n_trials: int, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Optuna optimization for a specific algorithm
        
        Args:
            algorithm: Algorithm name
            objective_func: Optuna objective function
            n_trials: Number of trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results dictionary
        """
        logger.info(f"Starting optimization for {algorithm} with {n_trials} trials")
        
        # Create study
        study_name = f"optimize_{algorithm}_project_{self.project_id}_{int(time.time())}"
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Store experiment in database if available
        experiment_id = None
        if DB_AVAILABLE:
            try:
                experiment = OptimizationExperiment(
                    project_id=self.project_id,
                    algorithm_type=algorithm,
                    n_trials=n_trials,
                    status='running'
                )
                db.session.add(experiment)
                db.session.commit()
                experiment_id = experiment.id
                logger.info(f"Created optimization experiment {experiment_id}")
            except Exception as e:
                logger.warning(f"Failed to create experiment record: {e}")
        
        start_time = time.time()
        
        try:
            # Run optimization
            study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
            
            # Get results
            best_trial = study.best_trial
            best_params = best_trial.params
            best_value = best_trial.value
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Compile results
            results = {
                'algorithm': algorithm,
                'best_params': best_params,
                'best_score': best_value,
                'objective': self.objective,
                'n_trials': len(study.trials),
                'duration_seconds': duration,
                'experiment_id': experiment_id,
                'study_name': study_name,
                'trials_data': [
                    {
                        'number': trial.number,
                        'value': trial.value,
                        'params': trial.params,
                        'state': trial.state.name
                    } for trial in study.trials
                ]
            }
            
            # Update database if available
            if DB_AVAILABLE and experiment_id:
                try:
                    experiment = OptimizationExperiment.query.get(experiment_id)
                    if experiment:
                        experiment.best_parameters = json.dumps(best_params)
                        experiment.best_score = best_value
                        experiment.status = 'completed'
                        experiment.trials_data = json.dumps(results['trials_data'])
                        experiment.updated_at = datetime.utcnow()
                        db.session.commit()
                        logger.info(f"Updated experiment {experiment_id} with results")
                except Exception as e:
                    logger.warning(f"Failed to update experiment record: {e}")
            
            logger.info(f"Optimization completed for {algorithm}")
            logger.info(f"Best score: {best_value:.4f}, Best params: {best_params}")
            
            return results
            
        except Exception as e:
            # Update experiment status on failure
            if DB_AVAILABLE and experiment_id:
                try:
                    experiment = OptimizationExperiment.query.get(experiment_id)
                    if experiment:
                        experiment.status = 'failed'
                        experiment.error_message = str(e)
                        experiment.completed_at = datetime.utcnow()
                        db.session.commit()
                except Exception as db_e:
                    logger.warning(f"Failed to update failed experiment: {db_e}")
            
            logger.error(f"Optimization failed for {algorithm}: {str(e)}")
            raise
    
    def run_optimization(self, algorithm: str, n_trials: int = 100, 
                        timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Generic optimization runner for any supported algorithm
        
        Args:
            algorithm: Algorithm name
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Optimization results dictionary
        """
        if algorithm not in self.available_algorithms:
            raise ValueError(f"Algorithm '{algorithm}' is not supported")
        
        if not self.available_algorithms[algorithm]:
            raise ValueError(f"Algorithm '{algorithm}' is not available (missing dependencies)")
        
        # Route to specific optimizer
        optimizers = {
            'arima': self.optimize_arima,
            'linear_regression': lambda n, t: {'algorithm': 'linear_regression', 'note': 'No hyperparameters to optimize'},
            'moving_average': self.optimize_moving_average,
            'random_forest': self.optimize_random_forest,
            'prophet': lambda n, t: {'algorithm': 'prophet', 'note': 'Prophet uses automatic parameter selection'},
            'lightgbm': self.optimize_lightgbm,
            'xgboost': self.optimize_xgboost,
            'sarimax': self.optimize_sarimax,
            'lstm': self.optimize_lstm,
            'nhits': lambda n, t: {'algorithm': 'nhits', 'note': 'NHiTS optimization not implemented yet'}
        }
        
        return optimizers[algorithm](n_trials, timeout)
    
    def get_optimization_progress(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get optimization progress for an experiment
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Progress information dictionary
        """
        if not DB_AVAILABLE:
            return {'error': 'Database not available'}
        
        try:
            experiment = OptimizationExperiment.query.get(experiment_id)
            if not experiment:
                return {'error': 'Experiment not found'}
            
            progress = {
                'experiment_id': experiment_id,
                'algorithm': experiment.algorithm,
                'status': experiment.status,
                'objective': experiment.objective,
                'n_trials_planned': experiment.n_trials,
                'n_trials_completed': experiment.n_trials_completed or 0,
                'best_score': experiment.best_score,
                'best_params': json.loads(experiment.best_params) if experiment.best_params else None,
                'duration_seconds': experiment.duration_seconds,
                'started_at': experiment.started_at.isoformat() if experiment.started_at else None,
                'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None,
                'error_message': experiment.error_message
            }
            
            # Calculate progress percentage
            if experiment.n_trials and experiment.n_trials_completed:
                progress['progress_percentage'] = min(100, (experiment.n_trials_completed / experiment.n_trials) * 100)
            else:
                progress['progress_percentage'] = 0
            
            return progress
            
        except Exception as e:
            logger.error(f"Failed to get optimization progress: {e}")
            return {'error': str(e)}
    
    def get_optimization_results(self, project_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get optimization results for a project
        
        Args:
            project_id: Project identifier
            limit: Maximum number of results to return
            
        Returns:
            List of optimization results
        """
        if not DB_AVAILABLE:
            return []
        
        try:
            experiments = OptimizationExperiment.query.filter_by(project_id=project_id)\
                                                    .order_by(OptimizationExperiment.created_at.desc())\
                                                    .limit(limit)\
                                                    .all()
            
            results = []
            for exp in experiments:
                result = {
                    'experiment_id': exp.id,
                    'algorithm': exp.algorithm,
                    'objective': exp.objective,
                    'status': exp.status,
                    'best_score': exp.best_score,
                    'best_params': json.loads(exp.best_params) if exp.best_params else None,
                    'n_trials': exp.n_trials,
                    'n_trials_completed': exp.n_trials_completed,
                    'duration_seconds': exp.duration_seconds,
                    'created_at': exp.created_at.isoformat(),
                    'completed_at': exp.completed_at.isoformat() if exp.completed_at else None
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get optimization results: {e}")
            return []
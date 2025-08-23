"""
Celery tasks for hyperparameter optimization operations
Part of Epic #2: Advanced Data Science Features
"""

from celery import current_task
from celery_app import celery_app
import pandas as pd
from datetime import datetime
import traceback

# Flask app context
from app import app, db
from models import Project, OptimizationExperiment
from model_optimizer import ModelOptimizer
from feature_engineer import FeatureEngineer
from data_processor import DataProcessor


@celery_app.task(bind=True)
def run_optimization_async(self, project_id: int, optimization_config: dict):
    """Asynchronous hyperparameter optimization"""
    with app.app_context():
        try:
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Initializing optimization', 'progress': 5})
            
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            if not project.dataset_path or not project.target_column:
                raise ValueError("Dataset or target column not configured")
            
            # Extract configuration
            algorithm_type = optimization_config.get('algorithm_type', 'random_forest')
            n_trials = optimization_config.get('n_trials', 100)
            optimization_metric = optimization_config.get('metric', 'rmse')
            use_feature_engineering = optimization_config.get('use_feature_engineering', True)
            
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Loading and preparing data', 'progress': 15})
            
            # Load dataset
            processor = DataProcessor(project.dataset_path)
            df = processor.load_data()
            df = processor.preprocess_timeseries(project.date_column, project.target_column)
            
            # Generate features if requested
            if use_feature_engineering:
                self.update_state(state='PROGRESS', meta={'step': 'Generating features', 'progress': 25})
                engineer = FeatureEngineer(project_id)
                df = engineer.generate_features(df)
            
            # Prepare data
            X = df.drop(columns=[project.target_column])
            y = df[project.target_column]
            
            # Split data
            split_point = int(len(df) * 0.8)
            X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
            
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Starting optimization', 'progress': 35})
            
            # Initialize optimizer
            optimizer = ModelOptimizer(project_id)
            
            # Create progress callback to update task status
            def update_progress_callback(trial_number, current_best):
                progress = 35 + (trial_number / n_trials) * 55  # 35% to 90%
                self.update_state(
                    state='PROGRESS', 
                    meta={
                        'step': f'Running optimization trial {trial_number}/{n_trials}',
                        'progress': progress,
                        'current_best_score': current_best,
                        'trials_completed': trial_number
                    }
                )
            
            # Run optimization
            result = optimizer.optimize(
                algorithm_type=algorithm_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_trials=n_trials,
                optimization_metric=optimization_metric
            )
            
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Finalizing optimization', 'progress': 95})
            
            return {
                'status': 'completed',
                'project_id': project_id,
                'experiment_id': result.get('experiment_id'),
                'algorithm_type': algorithm_type,
                'best_params': result['best_params'],
                'best_score': result['best_score'],
                'n_trials_completed': result['n_trials'],
                'optimization_metric': optimization_metric,
                'feature_engineering_used': use_feature_engineering,
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            error_msg = f"Hyperparameter optimization failed: {str(exc)}"
            self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': traceback.format_exc()})
            raise


@celery_app.task(bind=True)
def run_multi_algorithm_optimization_async(self, project_id: int, algorithms: list, optimization_config: dict):
    """Run optimization experiments for multiple algorithms"""
    with app.app_context():
        try:
            self.update_state(state='PROGRESS', meta={'step': 'Initializing multi-algorithm optimization', 'progress': 5})
            
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            results = []
            total_algorithms = len(algorithms)
            n_trials_per_algo = optimization_config.get('n_trials', 50)
            
            for i, algorithm_type in enumerate(algorithms):
                algo_progress = 10 + (i / total_algorithms) * 80
                
                self.update_state(
                    state='PROGRESS', 
                    meta={
                        'step': f'Optimizing {algorithm_type} ({i+1}/{total_algorithms})',
                        'progress': algo_progress
                    }
                )
                
                try:
                    # Create algorithm-specific config
                    algo_config = optimization_config.copy()
                    algo_config['algorithm_type'] = algorithm_type
                    
                    # Run optimization for this algorithm
                    algo_result = run_optimization_async.apply_async(
                        args=[project_id, algo_config]
                    ).get(propagate=True)
                    
                    results.append({
                        'algorithm_type': algorithm_type,
                        'success': True,
                        'experiment_id': algo_result['experiment_id'],
                        'best_score': algo_result['best_score'],
                        'best_params': algo_result['best_params']
                    })
                    
                except Exception as e:
                    results.append({
                        'algorithm_type': algorithm_type,
                        'success': False,
                        'error': str(e)
                    })
            
            self.update_state(state='PROGRESS', meta={'step': 'Analyzing results', 'progress': 95})
            
            # Find best overall result
            successful_results = [r for r in results if r['success']]
            best_overall = None
            
            if successful_results:
                best_overall = min(successful_results, key=lambda x: x['best_score'])
            
            return {
                'status': 'completed',
                'project_id': project_id,
                'total_algorithms': total_algorithms,
                'successful_experiments': len(successful_results),
                'failed_experiments': len([r for r in results if not r['success']]),
                'results': results,
                'best_overall': best_overall,
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            error_msg = f"Multi-algorithm optimization failed: {str(exc)}"
            self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': traceback.format_exc()})
            raise


@celery_app.task(bind=True)
def feature_selection_optimization_async(self, project_id: int, feature_selection_config: dict):
    """Optimize feature selection for better model performance"""
    with app.app_context():
        try:
            self.update_state(state='PROGRESS', meta={'step': 'Starting feature selection optimization', 'progress': 10})
            
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            # Load and prepare data
            processor = DataProcessor(project.dataset_path)
            df = processor.load_data()
            df = processor.preprocess_timeseries(project.date_column, project.target_column)
            
            self.update_state(state='PROGRESS', meta={'step': 'Generating features', 'progress': 30})
            
            # Generate features
            engineer = FeatureEngineer(project_id)
            features_df = engineer.generate_features(df)
            
            X = features_df.drop(columns=[project.target_column])
            y = features_df[project.target_column]
            
            self.update_state(state='PROGRESS', meta={'step': 'Running feature selection methods', 'progress': 50})
            
            # Apply different feature selection methods
            selection_methods = feature_selection_config.get('methods', ['variance', 'univariate', 'rfe', 'lasso'])
            selected_features = engineer.optimize_feature_selection(X, y, selection_methods)
            
            self.update_state(state='PROGRESS', meta={'step': 'Calculating feature importance', 'progress': 70})
            
            # Calculate feature importance
            feature_importance = engineer.get_feature_importance(X, y)
            
            self.update_state(state='PROGRESS', meta={'step': 'Testing selected features', 'progress': 85})
            
            # Test model performance with selected features
            if selected_features:
                X_selected = X[selected_features]
                
                # Simple cross-validation test
                from sklearn.model_selection import cross_val_score
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import make_scorer
                import numpy as np
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Test with all features
                scores_all = cross_val_score(model, X.fillna(0), y, cv=5, scoring='neg_root_mean_squared_error')
                rmse_all_features = -scores_all.mean()
                
                # Test with selected features
                scores_selected = cross_val_score(model, X_selected.fillna(0), y, cv=5, scoring='neg_root_mean_squared_error')
                rmse_selected_features = -scores_selected.mean()
                
                improvement = (rmse_all_features - rmse_selected_features) / rmse_all_features * 100
            else:
                rmse_all_features = None
                rmse_selected_features = None
                improvement = 0
            
            return {
                'status': 'completed',
                'project_id': project_id,
                'original_feature_count': len(X.columns),
                'selected_feature_count': len(selected_features),
                'selected_features': selected_features,
                'feature_importance': dict(list(feature_importance.items())[:20]),  # Top 20
                'selection_methods_used': selection_methods,
                'performance_comparison': {
                    'rmse_all_features': rmse_all_features,
                    'rmse_selected_features': rmse_selected_features,
                    'improvement_percentage': improvement
                },
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            error_msg = f"Feature selection optimization failed: {str(exc)}"
            self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': traceback.format_exc()})
            raise


@celery_app.task(bind=True)
def hyperparameter_sensitivity_analysis_async(self, experiment_id: int):
    """Analyze hyperparameter sensitivity for an optimization experiment"""
    with app.app_context():
        try:
            self.update_state(state='PROGRESS', meta={'step': 'Loading experiment data', 'progress': 20})
            
            experiment = OptimizationExperiment.query.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            trials_data = experiment.get_trials_data()
            if not trials_data:
                raise ValueError("No trials data available for analysis")
            
            self.update_state(state='PROGRESS', meta={'step': 'Analyzing parameter sensitivity', 'progress': 50})
            
            # Extract parameter values and scores
            param_analysis = {}
            all_params = set()
            
            # Collect all parameter names
            for trial in trials_data:
                if 'params' in trial and isinstance(trial['params'], dict):
                    all_params.update(trial['params'].keys())
            
            # Analyze each parameter
            for param_name in all_params:
                param_values = []
                param_scores = []
                
                for trial in trials_data:
                    if ('params' in trial and 'value' in trial and 
                        param_name in trial['params'] and 
                        trial['value'] is not None and 
                        trial['value'] != float('inf')):
                        param_values.append(trial['params'][param_name])
                        param_scores.append(trial['value'])
                
                if len(param_values) > 5:  # Need sufficient data points
                    # Calculate correlation between parameter and score
                    try:
                        correlation = np.corrcoef(param_values, param_scores)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                    except:
                        correlation = 0.0
                    
                    param_analysis[param_name] = {
                        'correlation_with_score': float(correlation),
                        'value_range': {
                            'min': min(param_values),
                            'max': max(param_values),
                            'mean': float(np.mean(param_values)),
                            'std': float(np.std(param_values))
                        },
                        'sensitivity_score': abs(correlation),  # Higher absolute correlation = more sensitive
                        'data_points': len(param_values)
                    }
            
            self.update_state(state='PROGRESS', meta={'step': 'Ranking parameters', 'progress': 80})
            
            # Rank parameters by sensitivity
            sorted_params = sorted(
                param_analysis.items(), 
                key=lambda x: x[1]['sensitivity_score'], 
                reverse=True
            )
            
            # Generate insights
            insights = []
            
            if sorted_params:
                most_sensitive = sorted_params[0]
                insights.append(f"Most sensitive parameter: {most_sensitive[0]} (correlation: {most_sensitive[1]['correlation_with_score']:.3f})")
                
                if len(sorted_params) > 1:
                    least_sensitive = sorted_params[-1]
                    insights.append(f"Least sensitive parameter: {least_sensitive[0]} (correlation: {least_sensitive[1]['correlation_with_score']:.3f})")
                
                # Identify highly sensitive parameters
                highly_sensitive = [p for p, data in sorted_params if data['sensitivity_score'] > 0.3]
                if highly_sensitive:
                    insights.append(f"Highly sensitive parameters (>0.3): {', '.join(highly_sensitive)}")
            
            return {
                'status': 'completed',
                'experiment_id': experiment_id,
                'algorithm_type': experiment.algorithm_type,
                'total_trials_analyzed': len(trials_data),
                'parameter_analysis': param_analysis,
                'parameter_ranking': [(name, data['sensitivity_score']) for name, data in sorted_params],
                'insights': insights,
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            error_msg = f"Sensitivity analysis failed: {str(exc)}"
            self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': traceback.format_exc()})
            raise


@celery_app.task(bind=True)
def optimization_convergence_analysis_async(self, experiment_id: int):
    """Analyze convergence behavior of optimization experiment"""
    with app.app_context():
        try:
            self.update_state(state='PROGRESS', meta={'step': 'Loading experiment', 'progress': 20})
            
            experiment = OptimizationExperiment.query.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            trials_data = experiment.get_trials_data()
            if not trials_data:
                raise ValueError("No trials data available")
            
            self.update_state(state='PROGRESS', meta={'step': 'Analyzing convergence', 'progress': 60})
            
            # Extract scores and sort by trial order
            valid_trials = [t for t in trials_data if 'value' in t and t['value'] is not None and t['value'] != float('inf')]
            valid_trials.sort(key=lambda x: x.get('trial_number', 0))
            
            scores = [trial['value'] for trial in valid_trials]
            trial_numbers = [trial.get('trial_number', i) for i, trial in enumerate(valid_trials)]
            
            if len(scores) < 10:
                raise ValueError("Insufficient trials for convergence analysis")
            
            # Calculate best scores so far (convergence curve)
            best_scores = []
            current_best = float('inf')
            
            for score in scores:
                if score < current_best:
                    current_best = score
                best_scores.append(current_best)
            
            # Analyze convergence properties
            final_best = best_scores[-1]
            improvement_after_half = best_scores[len(best_scores)//2] - final_best
            total_improvement = best_scores[0] - final_best
            
            convergence_rate = improvement_after_half / total_improvement if total_improvement > 0 else 0
            
            # Find when 90% of improvement was achieved
            target_score = best_scores[0] - 0.9 * total_improvement
            convergence_90_trial = None
            
            for i, score in enumerate(best_scores):
                if score <= target_score:
                    convergence_90_trial = trial_numbers[i]
                    break
            
            # Calculate plateau detection
            plateau_threshold = final_best * 0.01  # 1% improvement threshold
            plateau_start = None
            
            for i in range(len(best_scores) - 10, 0, -1):  # Look backwards
                if best_scores[i-1] - best_scores[i] > plateau_threshold:
                    plateau_start = trial_numbers[i]
                    break
            
            return {
                'status': 'completed',
                'experiment_id': experiment_id,
                'total_trials': len(scores),
                'convergence_analysis': {
                    'initial_best': best_scores[0],
                    'final_best': final_best,
                    'total_improvement': total_improvement,
                    'convergence_rate': convergence_rate,
                    'convergence_90_percent_trial': convergence_90_trial,
                    'plateau_started_at_trial': plateau_start,
                    'best_scores_history': best_scores,
                    'trial_numbers': trial_numbers
                },
                'recommendations': self._generate_convergence_recommendations(
                    convergence_rate, convergence_90_trial, plateau_start, len(scores)
                ),
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            error_msg = f"Convergence analysis failed: {str(exc)}"
            self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': traceback.format_exc()})
            raise
    
    def _generate_convergence_recommendations(self, convergence_rate: float, 
                                           convergence_90_trial: int, 
                                           plateau_start: int, 
                                           total_trials: int) -> List[str]:
        """Generate recommendations based on convergence analysis"""
        recommendations = []
        
        if convergence_rate < 0.1:
            recommendations.append("Fast convergence detected. Consider increasing exploration with more diverse search strategies.")
        elif convergence_rate > 0.7:
            recommendations.append("Slow convergence detected. Consider refining search space or increasing exploitation.")
        
        if convergence_90_trial and convergence_90_trial < total_trials * 0.3:
            recommendations.append("90% improvement achieved early. Future optimizations could use fewer trials.")
        
        if plateau_start and plateau_start < total_trials * 0.5:
            recommendations.append("Optimization plateaued early. Consider expanding search space or using different algorithms.")
        
        if not recommendations:
            recommendations.append("Optimization showed healthy convergence behavior.")
        
        return recommendations
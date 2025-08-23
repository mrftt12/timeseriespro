"""
Advanced API Routes for Data Science Features
Provides REST endpoints for data profiling, feature engineering, and optimization
"""

from flask import request, jsonify, render_template
from datetime import datetime
import logging
import json
import numpy as np
from typing import Dict, Any

from app import app, db
from models import Project, DataProfile, FeatureConfig, OptimizationExperiment, ExternalDataSource
from data_profiler import DataProfiler
from feature_engineer import FeatureEngineer
from data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@app.route('/api/projects/<int:project_id>/profile', methods=['POST'])
def generate_data_profile(project_id: int):
    """Generate comprehensive data profile for a project"""
    try:
        project = Project.query.get_or_404(project_id)
        
        if not project.dataset_path:
            return jsonify({'error': 'No dataset found for this project'}), 400
        
        # Get parameters from request (handle both JSON and form data)
        force_refresh = False
        if request.is_json and request.json:
            force_refresh = request.json.get('force_refresh', False)
        elif request.form:
            force_refresh = request.form.get('force_refresh', 'false').lower() == 'true'
        
        # Initialize data profiler
        profiler = DataProfiler(
            project_id=project_id,
            file_path=project.dataset_path,
            date_column=project.date_column,
            target_column=project.target_column
        )
        
        # Generate profile
        profile_results = profiler.analyze_dataset(force_refresh=force_refresh)
        
        # Convert numpy types to native Python types for JSON serialization
        profile_results = convert_numpy_types(profile_results)
        
        return jsonify({
            'status': 'success',
            'project_id': project_id,
            'profile': profile_results,
            'generated_at': datetime.utcnow().isoformat()
        })
        
    except FileNotFoundError:
        return jsonify({'error': 'Dataset file not found'}), 404
    except ValueError as e:
        return jsonify({'error': f'Data validation error: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Profile generation failed for project {project_id}: {str(e)}")
        return jsonify({'error': f'Profile generation failed: {str(e)}'}), 500


@app.route('/api/projects/<int:project_id>/profile', methods=['GET'])
def get_data_profile(project_id: int):
    """Get existing data profile for a project"""
    try:
        # Try to get cached profile from database
        profiles = DataProfile.query.filter_by(project_id=project_id).all()
        
        if not profiles:
            return jsonify({'error': 'No profile found. Generate one first.'}), 404
        
        # Combine profiles into result
        profile_data = {
            'project_id': project_id,
            'columns': {},
            'summary': {
                'total_columns': len(profiles),
                'last_updated': max(profile.created_at for profile in profiles).isoformat()
            }
        }
        
        for profile in profiles:
            profile_data['columns'][profile.column_name] = {
                'data_type': profile.data_type,
                'missing_count': profile.missing_count,
                'missing_percentage': profile.missing_percentage,
                'outlier_count': profile.outlier_count,
                'quality_score': profile.quality_score,
                'statistical_summary': profile.get_statistical_summary(),
                'recommendations': profile.get_recommendations()
            }
        
        return jsonify(profile_data)
        
    except Exception as e:
        logger.error(f"Failed to get profile for project {project_id}: {str(e)}")
        return jsonify({'error': f'Failed to retrieve profile: {str(e)}'}), 500


@app.route('/api/projects/<int:project_id>/profile/summary', methods=['GET'])
def get_profile_summary(project_id: int):
    """Get summary of data profile for quick overview"""
    try:
        profiles = DataProfile.query.filter_by(project_id=project_id).all()
        
        if not profiles:
            return jsonify({'error': 'No profile found'}), 404
        
        # Calculate summary statistics
        total_missing = sum(p.missing_count or 0 for p in profiles)
        total_outliers = sum(p.outlier_count or 0 for p in profiles)
        avg_quality_score = sum(p.quality_score or 0 for p in profiles) / len(profiles)
        
        summary = {
            'project_id': project_id,
            'total_columns': len(profiles),
            'total_missing_values': total_missing,
            'total_outliers': total_outliers,
            'average_quality_score': round(avg_quality_score, 2),
            'column_types': {},
            'last_updated': max(p.created_at for p in profiles).isoformat()
        }
        
        # Count column types
        for profile in profiles:
            data_type = profile.data_type or 'unknown'
            summary['column_types'][data_type] = summary['column_types'].get(data_type, 0) + 1
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Failed to get profile summary for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/features/config', methods=['GET'])
def get_feature_config(project_id: int):
    """Get feature engineering configuration for a project"""
    try:
        configs = FeatureConfig.query.filter_by(project_id=project_id).all()
        
        config_data = {
            'project_id': project_id,
            'configurations': {},
            'enabled_features': []
        }
        
        for config in configs:
            config_data['configurations'][config.feature_type] = {
                'enabled': config.is_enabled,
                'configuration': config.get_configuration(),
                'created_at': config.created_at.isoformat()
            }
            
            if config.is_enabled:
                config_data['enabled_features'].append(config.feature_type)
        
        return jsonify(config_data)
        
    except Exception as e:
        logger.error(f"Failed to get feature config for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/features/config', methods=['POST'])
def update_feature_config(project_id: int):
    """Update feature engineering configuration"""
    try:
        project = Project.query.get_or_404(project_id)
        config_data = request.get_json()
        
        if not config_data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # Clear existing configurations
        FeatureConfig.query.filter_by(project_id=project_id).delete()
        
        # Add new configurations
        for feature_type, settings in config_data.items():
            if feature_type in ['lag', 'rolling', 'calendar', 'technical', 'polynomial', 'interaction', 'fourier']:
                feature_config = FeatureConfig(
                    project_id=project_id,
                    feature_type=feature_type,
                    is_enabled=settings.get('enabled', False)
                )
                feature_config.set_configuration(settings.get('config', {}))
                db.session.add(feature_config)
        
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Feature configuration updated',
            'project_id': project_id
        })
        
    except Exception as e:
        logger.error(f"Failed to update feature config for project {project_id}: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/features/generate', methods=['POST'])
def generate_features(project_id: int):
    """Generate features for a project based on configuration"""
    try:
        project = Project.query.get_or_404(project_id)
        
        if not project.dataset_path or not project.target_column:
            return jsonify({'error': 'Project must have dataset and target column configured'}), 400
        
        # Load data
        processor = DataProcessor(project.dataset_path)
        data = processor.load_data()
        
        # Get feature generation parameters
        request_data = request.get_json() or {}
        custom_config = request_data.get('config', {})
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(
            project_id=project_id,
            data=data,
            target_column=project.target_column,
            date_column=project.date_column,
            config=custom_config
        )
        
        # Generate features
        enhanced_data = feature_engineer.generate_all_features(save_config=True)
        
        # Get feature importance
        importance = feature_engineer.get_feature_importance(enhanced_data)
        
        # Get summary
        feature_summary = feature_engineer.get_feature_summary()
        
        return jsonify({
            'status': 'success',
            'project_id': project_id,
            'feature_summary': feature_summary,
            'feature_importance': importance,
            'data_shape': {
                'original_columns': len(data.columns),
                'enhanced_columns': len(enhanced_data.columns),
                'new_features': len(feature_engineer.generated_features)
            },
            'generated_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Feature generation failed for project {project_id}: {str(e)}")
        return jsonify({'error': f'Feature generation failed: {str(e)}'}), 500


@app.route('/api/projects/<int:project_id>/features/importance', methods=['GET'])
def get_feature_importance(project_id: int):
    """Get feature importance analysis for generated features"""
    try:
        project = Project.query.get_or_404(project_id)
        
        # Get method from query parameters
        method = request.args.get('method', 'random_forest')
        
        if not project.dataset_path or not project.target_column:
            return jsonify({'error': 'Project configuration incomplete'}), 400
        
        # Load data and generate features
        processor = DataProcessor(project.dataset_path)
        data = processor.load_data()
        
        feature_engineer = FeatureEngineer(
            project_id=project_id,
            data=data,
            target_column=project.target_column,
            date_column=project.date_column
        )
        
        enhanced_data = feature_engineer.generate_all_features(save_config=False)
        importance = feature_engineer.get_feature_importance(enhanced_data, method=method)
        
        return jsonify({
            'project_id': project_id,
            'method': method,
            'feature_importance': importance,
            'total_features': len(importance),
            'generated_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/features/types', methods=['GET'])
def get_available_feature_types():
    """Get available feature types and their configurations"""
    feature_types = {
        'lag': {
            'name': 'Lag Features',
            'description': 'Historical values of the target variable',
            'parameters': {
                'max_lags': {'type': 'integer', 'default': 12, 'min': 1, 'max': 50},
                'include_other_columns': {'type': 'boolean', 'default': False}
            }
        },
        'rolling': {
            'name': 'Rolling Statistics',
            'description': 'Moving window statistics',
            'parameters': {
                'windows': {'type': 'array', 'default': [3, 7, 14, 30]},
                'statistics': {'type': 'array', 'default': ['mean', 'std', 'min', 'max']},
                'include_advanced': {'type': 'boolean', 'default': False}
            }
        },
        'calendar': {
            'name': 'Calendar Features',
            'description': 'Date and time-based features',
            'parameters': {
                'basic_features': {'type': 'boolean', 'default': True},
                'cyclical_encoding': {'type': 'boolean', 'default': True},
                'holidays': {'type': 'boolean', 'default': False},
                'business_features': {'type': 'boolean', 'default': True},
                'country_code': {'type': 'string', 'default': 'US'}
            }
        },
        'technical': {
            'name': 'Technical Indicators',
            'description': 'Financial and time series technical indicators',
            'parameters': {
                'moving_averages': {'type': 'boolean', 'default': True},
                'momentum': {'type': 'boolean', 'default': True},
                'volatility': {'type': 'boolean', 'default': True}
            }
        },
        'polynomial': {
            'name': 'Polynomial Features',
            'description': 'Polynomial transformations',
            'parameters': {
                'degree': {'type': 'integer', 'default': 2, 'min': 2, 'max': 5}
            }
        },
        'interaction': {
            'name': 'Interaction Features',
            'description': 'Feature interactions and combinations',
            'parameters': {
                'max_interactions': {'type': 'integer', 'default': 5, 'min': 2, 'max': 20}
            }
        },
        'fourier': {
            'name': 'Fourier Features',
            'description': 'Fourier transform for seasonality',
            'parameters': {
                'n_fourier': {'type': 'integer', 'default': 5, 'min': 1, 'max': 15}
            }
        }
    }
    
    return jsonify(feature_types)


@app.route('/api/projects/<int:project_id>/features/visualization', methods=['GET'])
def get_feature_visualization(project_id: int):
    """Get feature importance visualization data"""
    try:
        project = Project.query.get_or_404(project_id)
        
        if not project.dataset_path or not project.target_column:
            return jsonify({'error': 'Project configuration incomplete'}), 400
        
        # Get visualization parameters
        top_n = request.args.get('top_n', 20, type=int)
        method = request.args.get('method', 'random_forest')
        
        # Load data and generate features
        processor = DataProcessor(project.dataset_path)
        data = processor.load_data()
        
        feature_engineer = FeatureEngineer(
            project_id=project_id,
            data=data,
            target_column=project.target_column,
            date_column=project.date_column
        )
        
        # Generate features first
        enhanced_data = feature_engineer.generate_all_features(save_config=False)
        
        # Get visualization data
        viz_data = feature_engineer.visualize_feature_importance(
            enhanced_data, 
            top_n=top_n, 
            method=method
        )
        
        return jsonify({
            'status': 'success',
            'visualization': viz_data,
            'project_id': project_id,
            'generated_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to generate feature visualization for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/optimize/<string:algorithm>', methods=['POST'])
def start_optimization(project_id: int, algorithm: str):
    """Start hyperparameter optimization for a specific algorithm"""
    try:
        project = Project.query.get_or_404(project_id)
        
        if not project.dataset_path or not project.target_column:
            return jsonify({'error': 'Project configuration incomplete'}), 400
        
        # Get optimization parameters
        request_data = request.get_json() or {}
        n_trials = request_data.get('n_trials', 50)
        timeout = request_data.get('timeout')
        objective = request_data.get('objective', 'rmse')
        test_size = request_data.get('test_size', 0.2)
        
        # Load data
        from data_processor import DataProcessor
        processor = DataProcessor(project.dataset_path)
        data = processor.load_data()
        
        # Initialize optimizer
        from model_optimizer import ModelOptimizer
        optimizer = ModelOptimizer(
            project_id=project_id,
            data=data,
            target_column=project.target_column,
            date_column=project.date_column,
            objective=objective,
            test_size=test_size
        )
        
        # Start optimization (this will run in background)
        try:
            result = optimizer.run_optimization(algorithm, n_trials, timeout)
            
            return jsonify({
                'status': 'success',
                'message': f'Optimization completed for {algorithm}',
                'result': result
            })
            
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
    except Exception as e:
        logger.error(f"Failed to start optimization: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/optimize/<int:experiment_id>/status', methods=['GET'])
def get_optimization_status(project_id: int, experiment_id: int):
    """Get optimization progress status"""
    try:
        # Initialize a dummy optimizer to access progress method
        from model_optimizer import ModelOptimizer
        import pandas as pd
        
        # Create minimal dummy data for method access
        dummy_data = pd.DataFrame({'value': [1, 2, 3]})
        optimizer = ModelOptimizer(project_id, dummy_data, 'value')
        
        progress = optimizer.get_optimization_progress(experiment_id)
        
        return jsonify({
            'status': 'success',
            'progress': progress
        })
        
    except Exception as e:
        logger.error(f"Failed to get optimization status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/optimize/results', methods=['GET'])
def get_optimization_results_api(project_id: int):
    """Get optimization results for a project"""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        # Initialize a dummy optimizer to access results method
        from model_optimizer import ModelOptimizer
        import pandas as pd
        
        # Create minimal dummy data for method access
        dummy_data = pd.DataFrame({'value': [1, 2, 3]})
        optimizer = ModelOptimizer(project_id, dummy_data, 'value')
        
        results = optimizer.get_optimization_results(project_id, limit)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Failed to get optimization results: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/optimize/algorithms', methods=['GET'])
def get_available_algorithms(project_id: int):
    """Get list of algorithms available for optimization"""
    try:
        # Initialize optimizer to check available algorithms
        from model_optimizer import ModelOptimizer
        import pandas as pd
        
        # Create minimal dummy data for method access
        dummy_data = pd.DataFrame({'value': [1, 2, 3]})
        optimizer = ModelOptimizer(project_id, dummy_data, 'value')
        
        available = {alg: available for alg, available in optimizer.available_algorithms.items() if available}
        
        algorithms_info = {
            'arima': {
                'name': 'ARIMA',
                'description': 'AutoRegressive Integrated Moving Average',
                'hyperparams': ['p (0-5)', 'd (0-2)', 'q (0-5)'],
                'recommended_trials': 50
            },
            'random_forest': {
                'name': 'Random Forest',
                'description': 'Ensemble of decision trees',
                'hyperparams': ['n_estimators (10-500)', 'max_depth (3-20)'],
                'recommended_trials': 100
            },
            'lightgbm': {
                'name': 'LightGBM',
                'description': 'Gradient boosting framework',
                'hyperparams': ['num_leaves (10-300)', 'learning_rate (0.01-0.3)', 'n_estimators (50-500)'],
                'recommended_trials': 100
            },
            'xgboost': {
                'name': 'XGBoost',
                'description': 'Extreme gradient boosting',
                'hyperparams': ['n_estimators (50-500)', 'max_depth (3-15)', 'learning_rate (0.01-0.3)'],
                'recommended_trials': 100
            },
            'sarimax': {
                'name': 'SARIMAX',
                'description': 'Seasonal AutoRegressive Integrated Moving Average',
                'hyperparams': ['p, d, q (0-3)', 'seasonal_p, seasonal_d, seasonal_q (0-2)', 'seasonality (4,7,12)'],
                'recommended_trials': 50
            },
            'lstm': {
                'name': 'LSTM',
                'description': 'Long Short-Term Memory neural network',
                'hyperparams': ['sequence_length (10-60)', 'hidden_units (32-256)', 'epochs (20-150)'],
                'recommended_trials': 50
            },
            'moving_average': {
                'name': 'Moving Average',
                'description': 'Simple moving average',
                'hyperparams': ['window (3-30)'],
                'recommended_trials': 20
            }
        }
        
        result = {}
        for alg in available:
            if alg in algorithms_info:
                result[alg] = algorithms_info[alg]
                result[alg]['available'] = True
            else:
                result[alg] = {
                    'name': alg.replace('_', ' ').title(),
                    'available': True,
                    'note': 'No hyperparameters to optimize' if alg in ['linear_regression', 'prophet'] else 'Optimization not implemented'
                }
        
        return jsonify({
            'status': 'success',
            'algorithms': result,
            'total_available': len(available)
        })
        
    except Exception as e:
        logger.error(f"Failed to get available algorithms: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/optimization/experiments', methods=['GET'])
def get_optimization_experiments(project_id: int):
    """Get hyperparameter optimization experiments for a project"""
    try:
        experiments = OptimizationExperiment.query.filter_by(project_id=project_id)\
                                                  .order_by(OptimizationExperiment.created_at.desc())\
                                                  .all()
        
        experiment_data = []
        for exp in experiments:
            experiment_data.append({
                'id': exp.id,
                'algorithm_type': exp.algorithm_type,
                'status': exp.status,
                'n_trials': exp.n_trials,
                'best_score': exp.best_score,
                'best_parameters': exp.get_best_parameters(),
                'created_at': exp.created_at.isoformat(),
                'updated_at': exp.updated_at.isoformat() if exp.updated_at else None
            })
        
        return jsonify({
            'project_id': project_id,
            'experiments': experiment_data,
            'total_experiments': len(experiment_data)
        })
        
    except Exception as e:
        logger.error(f"Failed to get optimization experiments for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/external-data', methods=['GET'])
def get_external_data_sources(project_id: int):
    """Get external data sources configured for a project"""
    try:
        sources = ExternalDataSource.query.filter_by(project_id=project_id).all()
        
        source_data = []
        for source in sources:
            source_data.append({
                'id': source.id,
                'source_type': source.source_type,
                'is_active': source.is_active,
                'last_sync': source.last_sync.isoformat() if source.last_sync else None,
                'configuration': source.get_api_configuration(),
                'data_mapping': source.get_data_mapping(),
                'created_at': source.created_at.isoformat()
            })
        
        return jsonify({
            'project_id': project_id,
            'external_sources': source_data,
            'total_sources': len(source_data)
        })
        
    except Exception as e:
        logger.error(f"Failed to get external data sources for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/recommendations', methods=['GET'])
def get_project_recommendations(project_id: int):
    """Get AI-powered recommendations for improving model performance"""
    try:
        project = Project.query.get_or_404(project_id)
        
        recommendations = []
        
        # Check if data profiling exists
        profile_count = DataProfile.query.filter_by(project_id=project_id).count()
        if profile_count == 0:
            recommendations.append({
                'type': 'data_profiling',
                'priority': 'high',
                'title': 'Generate Data Profile',
                'description': 'Run comprehensive data analysis to understand data quality and characteristics',
                'action': 'POST /api/projects/{}/profile'.format(project_id),
                'estimated_time': '2-5 minutes'
            })
        
        # Check feature engineering configuration
        feature_count = FeatureConfig.query.filter_by(project_id=project_id, is_enabled=True).count()
        if feature_count == 0:
            recommendations.append({
                'type': 'feature_engineering',
                'priority': 'high',
                'title': 'Configure Feature Engineering',
                'description': 'Enable automated feature generation to improve model accuracy',
                'action': 'POST /api/projects/{}/features/config'.format(project_id),
                'estimated_time': '1-2 minutes'
            })
        
        # Check if optimization experiments exist
        opt_count = OptimizationExperiment.query.filter_by(project_id=project_id).count()
        if opt_count == 0 and project.models:  # Has models but no optimization
            recommendations.append({
                'type': 'hyperparameter_optimization',
                'priority': 'medium',
                'title': 'Optimize Model Parameters',
                'description': 'Use Bayesian optimization to find best hyperparameters automatically',
                'action': 'POST /api/projects/{}/optimization/start'.format(project_id),
                'estimated_time': '10-30 minutes'
            })
        
        # Check external data integration
        external_count = ExternalDataSource.query.filter_by(project_id=project_id, is_active=True).count()
        if external_count == 0 and project.date_column:
            recommendations.append({
                'type': 'external_data',
                'priority': 'low',
                'title': 'Add External Data',
                'description': 'Integrate holiday, weather, or economic data to enhance predictions',
                'action': 'POST /api/projects/{}/external-data'.format(project_id),
                'estimated_time': '5-10 minutes'
            })
        
        return jsonify({
            'project_id': project_id,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'generated_at': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/system/capabilities', methods=['GET'])
def get_system_capabilities():
    """Get current system capabilities and available features"""
    from forecasting import (LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE, 
                           PROPHET_AVAILABLE, PYTORCH_FORECASTING_AVAILABLE)
    from feature_engineer import TALIB_AVAILABLE, HOLIDAYS_AVAILABLE
    from data_profiler import SKLEARN_AVAILABLE, STATSMODELS_AVAILABLE
    
    capabilities = {
        'core_algorithms': {
            'arima': True,
            'linear_regression': True,
            'moving_average': True,
            'random_forest': True
        },
        'advanced_algorithms': {
            'prophet': PROPHET_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE,
            'lstm': PYTORCH_FORECASTING_AVAILABLE
        },
        'data_profiling': {
            'basic_statistics': True,
            'advanced_outlier_detection': SKLEARN_AVAILABLE,
            'time_series_analysis': STATSMODELS_AVAILABLE
        },
        'feature_engineering': {
            'basic_features': True,
            'technical_indicators': TALIB_AVAILABLE,
            'holiday_features': HOLIDAYS_AVAILABLE
        },
        'optimization': {
            'hyperparameter_tuning': True,  # Will implement with Optuna
            'feature_selection': True,
            'ensemble_methods': True  # Phase 2
        },
        'external_data': {
            'holiday_data': HOLIDAYS_AVAILABLE,
            'weather_data': False,  # Will implement in Phase 1
            'economic_data': False  # Will implement in Phase 1
        }
    }
    
    return jsonify({
        'capabilities': capabilities,
        'version': '2.0.0-alpha',
        'epic': 'add-ds-features',
        'last_updated': datetime.utcnow().isoformat()
    })


# Ensemble Methods API Endpoints (Phase 2)

@app.route('/api/projects/<int:project_id>/ensemble/candidates', methods=['GET'])
def get_ensemble_candidates(project_id: int):
    """Get available models for ensemble creation"""
    try:
        project = Project.query.get_or_404(project_id)
        
        from ensemble_methods import EnsembleMethods
        ensemble = EnsembleMethods(project_id)
        
        available_models = ensemble.get_available_models()
        
        # Check if we have enough models for ensemble
        ensemble_ready = len(available_models) >= 2
        diverse_models = ensemble.select_diverse_models() if ensemble_ready else []
        
        return jsonify({
            'project_id': project_id,
            'available_models': available_models,
            'total_models': len(available_models),
            'ensemble_ready': ensemble_ready,
            'recommended_models': diverse_models,
            'minimum_models_required': 2
        })
        
    except Exception as e:
        logger.error(f"Failed to get ensemble candidates for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/ensemble/train', methods=['POST'])
def train_ensemble(project_id: int):
    """Train ensemble model"""
    try:
        project = Project.query.get_or_404(project_id)
        
        request_data = request.get_json() or {}
        ensemble_type = request_data.get('ensemble_type', 'voting')
        model_ids = request_data.get('model_ids')
        
        # Validate ensemble type
        valid_types = ['voting', 'stacking', 'blending']
        if ensemble_type not in valid_types:
            return jsonify({'error': f'Invalid ensemble type. Must be one of: {valid_types}'}), 400
        
        from ensemble_methods import EnsembleMethods
        ensemble = EnsembleMethods(project_id)
        
        # Auto-select models if not provided
        if not model_ids:
            model_ids = ensemble.select_diverse_models()
        
        if len(model_ids) < 2:
            return jsonify({'error': 'Need at least 2 models for ensemble'}), 400
        
        # Train the specified ensemble type
        if ensemble_type == 'voting':
            weights = request_data.get('weights')
            result = ensemble.voting_ensemble(model_ids=model_ids, weights=weights)
        elif ensemble_type == 'stacking':
            meta_learner = request_data.get('meta_learner', 'linear')
            result = ensemble.stacking_ensemble(model_ids=model_ids, meta_learner=meta_learner)
        elif ensemble_type == 'blending':
            holdout_ratio = request_data.get('holdout_ratio', 0.2)
            result = ensemble.blending_ensemble(model_ids=model_ids, holdout_ratio=holdout_ratio)
        
        # Save ensemble configuration
        from ensemble_methods import save_ensemble_config
        config_id = save_ensemble_config(
            project_id, 
            ensemble_type, 
            result['model_ids'], 
            result
        )
        
        result['config_id'] = config_id
        
        return jsonify({
            'status': 'success',
            'message': f'{ensemble_type.title()} ensemble trained successfully',
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Failed to train ensemble for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/ensemble/compare', methods=['POST'])
def compare_ensembles(project_id: int):
    """Compare multiple ensemble methods"""
    try:
        project = Project.query.get_or_404(project_id)
        
        request_data = request.get_json() or {}
        model_ids = request_data.get('model_ids')
        ensemble_types = request_data.get('ensemble_types', ['voting', 'stacking', 'blending'])
        
        from ensemble_methods import EnsembleMethods
        ensemble = EnsembleMethods(project_id)
        
        # Auto-select models if not provided
        if not model_ids:
            model_ids = ensemble.select_diverse_models()
        
        if len(model_ids) < 2:
            return jsonify({'error': 'Need at least 2 models for ensemble comparison'}), 400
        
        results = []
        
        # Train each requested ensemble type
        for ensemble_type in ensemble_types:
            try:
                if ensemble_type == 'voting':
                    result = ensemble.voting_ensemble(model_ids=model_ids)
                elif ensemble_type == 'stacking':
                    result = ensemble.stacking_ensemble(model_ids=model_ids)
                elif ensemble_type == 'blending':
                    result = ensemble.blending_ensemble(model_ids=model_ids)
                else:
                    continue
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to train {ensemble_type} ensemble: {str(e)}")
                continue
        
        if not results:
            return jsonify({'error': 'No ensemble methods completed successfully'}), 500
        
        # Compare ensemble performance
        comparison = ensemble.compare_ensemble_performance(results)
        
        return jsonify({
            'project_id': project_id,
            'ensemble_results': results,
            'comparison': comparison,
            'models_used': model_ids
        })
        
    except Exception as e:
        logger.error(f"Failed to compare ensembles for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/ensemble/configs', methods=['GET'])
def get_ensemble_configs(project_id: int):
    """Get saved ensemble configurations for a project"""
    try:
        # Query ensemble configurations from optimization_experiment table
        ensemble_configs = OptimizationExperiment.query.filter(
            OptimizationExperiment.project_id == project_id,
            OptimizationExperiment.algorithm_type.like('ensemble_%')
        ).order_by(OptimizationExperiment.created_at.desc()).all()
        
        configs = []
        for config in ensemble_configs:
            try:
                base_models = json.loads(config.search_space).get('base_models', [])
                meta_config = json.loads(config.best_parameters) if config.best_parameters else {}
                
                config_data = {
                    'id': config.id,
                    'ensemble_type': config.algorithm_type.replace('ensemble_', ''),
                    'base_models': base_models,
                    'performance': meta_config.get('performance', {}),
                    'created_at': config.created_at.isoformat(),
                    'status': config.status
                }
                configs.append(config_data)
                
            except Exception as e:
                logger.warning(f"Error parsing ensemble config {config.id}: {str(e)}")
                continue
        
        return jsonify({
            'project_id': project_id,
            'ensemble_configs': configs,
            'total_configs': len(configs)
        })
        
    except Exception as e:
        logger.error(f"Failed to get ensemble configs for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<int:project_id>/ensemble/performance', methods=['GET'])
def get_ensemble_performance(project_id: int):
    """Get ensemble performance comparison with individual models"""
    try:
        project = Project.query.get_or_404(project_id)
        
        # Get all individual model performances
        models = ModelResult.query.filter_by(project_id=project_id).all()
        individual_performances = []
        
        for model in models:
            individual_performances.append({
                'model_id': model.id,
                'model_type': model.model_type,
                'rmse': model.rmse,
                'mae': model.mae,
                'r2': model.r2_score,
                'created_at': model.created_at.isoformat()
            })
        
        # Get ensemble configurations and their performances
        ensemble_configs = OptimizationExperiment.query.filter(
            OptimizationExperiment.project_id == project_id,
            OptimizationExperiment.algorithm_type.like('ensemble_%')
        ).all()
        
        ensemble_performances = []
        for config in ensemble_configs:
            try:
                meta_config = json.loads(config.best_parameters) if config.best_parameters else {}
                performance = meta_config.get('performance', {})
                
                if performance:
                    ensemble_performances.append({
                        'config_id': config.id,
                        'ensemble_type': config.algorithm_type.replace('ensemble_', ''),
                        'rmse': performance.get('rmse'),
                        'mae': performance.get('mae'),
                        'r2': performance.get('r2'),
                        'created_at': config.created_at.isoformat()
                    })
                    
            except Exception as e:
                logger.warning(f"Error parsing ensemble performance {config.id}: {str(e)}")
                continue
        
        # Find best performers
        best_individual = min(individual_performances, key=lambda x: x['rmse']) if individual_performances else None
        best_ensemble = min(ensemble_performances, key=lambda x: x['rmse']) if ensemble_performances else None
        
        # Calculate improvement
        improvement = None
        if best_individual and best_ensemble:
            rmse_improvement = (best_individual['rmse'] - best_ensemble['rmse']) / best_individual['rmse'] * 100
            improvement = {
                'rmse_improvement_percent': rmse_improvement,
                'best_individual_rmse': best_individual['rmse'],
                'best_ensemble_rmse': best_ensemble['rmse']
            }
        
        return jsonify({
            'project_id': project_id,
            'individual_performances': individual_performances,
            'ensemble_performances': ensemble_performances,
            'best_individual': best_individual,
            'best_ensemble': best_ensemble,
            'performance_improvement': improvement
        })
        
    except Exception as e:
        logger.error(f"Failed to get ensemble performance for project {project_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Error handlers for advanced API routes
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error)}), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500
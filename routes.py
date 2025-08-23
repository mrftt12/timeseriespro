import os
import pandas as pd
from flask import render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import io
import base64
import numpy as np
import logging

from app import app, db
from models import Project, ModelResult, DataProfile, FeatureConfig, OptimizationExperiment
from data_processor import DataProcessor
from forecasting import ForecastingEngine, LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE, SARIMAX_AVAILABLE, PYTORCH_FORECASTING_AVAILABLE

# Import new advanced data science modules
from data_profiler import DataProfiler
from feature_engineer import FeatureEngineer
from model_optimizer import ModelOptimizer

# Configure logging
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def dashboard():
    projects = Project.query.order_by(Project.updated_at.desc()).all()
    
    # Get summary statistics
    total_projects = Project.query.count()
    total_models = ModelResult.query.count()
    
    # Get recent activity
    recent_models = ModelResult.query.join(Project).order_by(ModelResult.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html', 
                         projects=projects,
                         total_projects=total_projects,
                         total_models=total_models,
                         recent_models=recent_models)

@app.route('/upload', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        project_name = request.form.get('project_name', '').strip()
        project_description = request.form.get('project_description', '').strip()
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if not project_name:
            flash('Project name is required', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                
                # Create uploads directory if it doesn't exist
                upload_dir = app.config['UPLOAD_FOLDER']
                if not os.path.exists(upload_dir):
                    os.makedirs(upload_dir)
                
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                
                # Create new project
                project = Project(
                    name=project_name,
                    description=project_description,
                    dataset_filename=filename,
                    dataset_path=file_path
                )
                
                db.session.add(project)
                db.session.commit()
                
                flash('Dataset uploaded successfully!', 'success')
                return redirect(url_for('project_detail', id=project.id))
                
            except Exception as e:
                flash(f'Error uploading file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload CSV or Excel files only.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/project/<int:id>')
def project_detail(id):
    project = Project.query.get_or_404(id)
    
    try:
        # Load and preview dataset
        processor = DataProcessor(project.dataset_path)
        df = processor.load_data()
        
        # Get basic dataset info
        dataset_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'preview': df.head(10).to_dict('records')
        }
        
        # Convert dtypes to strings for JSON serialization
        dataset_info['dtypes'] = {k: str(v) for k, v in dataset_info['dtypes'].items()}
        
        # Get existing models for this project
        models = ModelResult.query.filter_by(project_id=id).order_by(ModelResult.created_at.desc()).all()
        
        # Check for existing data profiles
        existing_profiles = DataProfile.query.filter_by(project_id=id).count()
        has_profile = existing_profiles > 0
        
        return render_template('project.html', 
                             project=project, 
                             dataset_info=dataset_info,
                             models=models,
                             has_profile=has_profile,
                             lightgbm_available=LIGHTGBM_AVAILABLE,
                             xgboost_available=XGBOOST_AVAILABLE,
                             sarimax_available=SARIMAX_AVAILABLE,
                             pytorch_forecasting_available=PYTORCH_FORECASTING_AVAILABLE)
                             
    except Exception as e:
        flash(f'Error loading dataset: {str(e)}', 'error')
        return render_template('project.html', 
                             project=project, 
                             dataset_info=None,
                             models=[])

@app.route('/project/<int:id>/configure', methods=['POST'])
def configure_project(id):
    project = Project.query.get_or_404(id)
    
    date_column = request.form.get('date_column')
    target_column = request.form.get('target_column')
    
    if not date_column or not target_column:
        flash('Both date and target columns must be selected', 'error')
        return redirect(url_for('project_detail', id=id))
    
    project.date_column = date_column
    project.target_column = target_column
    project.updated_at = datetime.utcnow()
    
    db.session.commit()
    flash('Project configuration updated!', 'success')
    
    return redirect(url_for('project_detail', id=id))

@app.route('/project/<int:id>/train', methods=['POST'])
def train_model(id):
    project = Project.query.get_or_404(id)
    
    if not project.date_column or not project.target_column:
        flash('Please configure date and target columns first', 'error')
        return redirect(url_for('project_detail', id=id))
    
    model_type = request.form.get('model_type')
    model_name = request.form.get('model_name', f'{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    try:
        # Initialize forecasting engine
        engine = ForecastingEngine(project.dataset_path, project.date_column, project.target_column)
        
        # Train model based on type
        if model_type == 'arima':
            result = engine.train_arima()
        elif model_type == 'linear_regression':
            result = engine.train_linear_regression()
        elif model_type == 'moving_average':
            window = int(request.form.get('ma_window', 7))
            result = engine.train_moving_average(window)
        elif model_type == 'random_forest':
            n_estimators = int(request.form.get('rf_n_estimators', 100))
            max_depth = int(request.form.get('rf_max_depth', 10))
            result = engine.train_random_forest(n_estimators, max_depth)
        elif model_type == 'prophet':
            result = engine.train_prophet()
        elif model_type == 'lightgbm':
            num_leaves = int(request.form.get('lgb_num_leaves', 31))
            learning_rate = float(request.form.get('lgb_learning_rate', 0.1))
            n_estimators = int(request.form.get('lgb_n_estimators', 100))
            result = engine.train_lightgbm(num_leaves, learning_rate, n_estimators)
        elif model_type == 'xgboost':
            n_estimators = int(request.form.get('xgb_n_estimators', 100))
            max_depth = int(request.form.get('xgb_max_depth', 6))
            learning_rate = float(request.form.get('xgb_learning_rate', 0.1))
            result = engine.train_xgboost(n_estimators, max_depth, learning_rate)
        elif model_type == 'sarimax':
            order = tuple(map(int, request.form.get('sarimax_order', '1,1,1').split(',')))
            seasonal_order = tuple(map(int, request.form.get('sarimax_seasonal_order', '1,1,1,12').split(',')))
            result = engine.train_sarimax(order, seasonal_order)
        elif model_type == 'lstm':
            sequence_length = int(request.form.get('lstm_sequence_length', 30))
            hidden_units = int(request.form.get('lstm_hidden_units', 50))
            epochs = int(request.form.get('lstm_epochs', 100))
            result = engine.train_lstm(sequence_length, hidden_units, epochs)
        elif model_type == 'nhits':
            max_epochs = int(request.form.get('nhits_epochs', 100))
            result = engine.train_nhits(max_epochs=max_epochs)
        else:
            flash('Invalid model type', 'error')
            return redirect(url_for('project_detail', id=id))
        
        # Save model result
        model_result = ModelResult(
            project_id=id,
            model_name=model_name,
            model_type=model_type,
            rmse=result['metrics']['rmse'],
            mae=result['metrics']['mae'],
            mape=result['metrics']['mape'],
            r2_score=result['metrics'].get('r2_score'),
            training_samples=result['training_samples'],
            test_samples=result['test_samples']
        )
        
        model_result.set_parameters(result['parameters'])
        model_result.set_forecast_data(result['forecast_data'])
        
        db.session.add(model_result)
        db.session.commit()
        
        flash(f'Model {model_name} trained successfully!', 'success')
        
    except Exception as e:
        flash(f'Error training model: {str(e)}', 'error')
    
    return redirect(url_for('project_detail', id=id))

@app.route('/model/<int:id>/chart')
def model_chart(id):
    model = ModelResult.query.get_or_404(id)
    forecast_data = model.get_forecast_data()
    
    return jsonify(forecast_data)

@app.route('/compare')
def compare_models():
    projects = Project.query.order_by(Project.name).all()
    selected_project_id = request.args.get('project_id', type=int)
    
    models = []
    if selected_project_id:
        models = ModelResult.query.filter_by(project_id=selected_project_id).order_by(ModelResult.created_at.desc()).all()
    
    return render_template('compare.html', projects=projects, models=models, selected_project_id=selected_project_id)

@app.route('/model/<int:id>/export')
def export_model(id):
    model = ModelResult.query.get_or_404(id)
    forecast_data = model.get_forecast_data()
    
    # Create DataFrame from forecast data
    df = pd.DataFrame(forecast_data.get('forecast', []))
    
    # Create CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    # Convert to bytes
    output_bytes = io.BytesIO()
    output_bytes.write(output.getvalue().encode('utf-8'))
    output_bytes.seek(0)
    
    filename = f"{model.model_name}_forecast.csv"
    
    return send_file(
        output_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

@app.route('/project/<int:id>/delete', methods=['POST'])
def delete_project(id):
    project = Project.query.get_or_404(id)
    
    try:
        # Delete associated file
        if project.dataset_path and os.path.exists(project.dataset_path):
            os.remove(project.dataset_path)
        
        # Delete from database (models will be deleted due to cascade)
        db.session.delete(project)
        db.session.commit()
        
        flash('Project deleted successfully!', 'success')
        
    except Exception as e:
        flash(f'Error deleting project: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/model/<int:id>/delete', methods=['POST'])
def delete_model(id):
    model = ModelResult.query.get_or_404(id)
    project_id = model.project_id
    
    try:
        db.session.delete(model)
        db.session.commit()
        flash('Model deleted successfully!', 'success')
        
    except Exception as e:
        flash(f'Error deleting model: {str(e)}', 'error')
    
    return redirect(url_for('project_detail', id=project_id))

# ===== ADVANCED DATA SCIENCE FEATURES API ENDPOINTS =====
# Epic #2: Advanced Data Science Features

@app.route('/api/projects/<int:project_id>/profile', methods=['POST'])
def generate_data_profile_api(project_id):
    """Generate comprehensive data profile for project"""
    project = Project.query.get_or_404(project_id)
    
    if not project.dataset_path or not os.path.exists(project.dataset_path):
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        # Get force refresh parameter
        force_refresh = request.json.get('force_refresh', False) if request.json else False
        
        # Create profiler instance
        profiler = DataProfiler(
            project_id=project.id,
            file_path=project.dataset_path,
            date_column=project.date_column,
            target_column=project.target_column
        )
        
        # Generate comprehensive profile
        profile_results = profiler.analyze_dataset(force_refresh=force_refresh)
        
        # Convert numpy types for JSON serialization
        profile_results = convert_numpy_types(profile_results)
        
        return jsonify({
            'success': True,
            'profile': profile_results,
            'message': 'Data profile generated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate data profile: {str(e)}'}), 500

@app.route('/api/projects/<int:project_id>/profile', methods=['GET'])
def get_data_profile_api(project_id):
    """Retrieve existing data profile for project"""
    project = Project.query.get_or_404(project_id)
    
    try:
        # Try to load cached profile first
        profiler = DataProfiler(
            project_id=project.id,
            file_path=project.dataset_path,
            date_column=project.date_column,
            target_column=project.target_column
        )
        
        cached_profile = profiler._load_cached_profile()
        
        if cached_profile:
            return jsonify({
                'success': True,
                'profile': cached_profile,
                'cached': True
            })
        else:
            # No cached profile found, suggest generating one
            return jsonify({
                'success': False,
                'message': 'No profile found for this project. Generate one first.',
                'suggestion': 'POST to /api/projects/{}/profile to generate'.format(project_id)
            }), 404
            
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve data profile: {str(e)}'}), 500

@app.route('/api/projects/<int:project_id>/profile/refresh', methods=['POST'])
def refresh_data_profile_api(project_id):
    """Force refresh data profile for project"""
    project = Project.query.get_or_404(project_id)
    
    if not project.dataset_path or not os.path.exists(project.dataset_path):
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        # Create profiler instance and force refresh
        profiler = DataProfiler(
            project_id=project.id,
            file_path=project.dataset_path,
            date_column=project.date_column,
            target_column=project.target_column
        )
        
        # Force regenerate profile
        profile_results = profiler.analyze_dataset(force_refresh=True)
        
        return jsonify({
            'success': True,
            'profile': profile_results,
            'message': 'Data profile refreshed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to refresh data profile: {str(e)}'}), 500

@app.route('/api/projects/<int:project_id>/features/config', methods=['POST'])
def configure_features_api(project_id):
    """Configure feature engineering pipeline"""
    project = Project.query.get_or_404(project_id)
    
    try:
        config_data = request.get_json()
        if not config_data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # Load data for feature engineer
        if not project.dataset_path or not project.target_column:
            return jsonify({'error': 'Project must have dataset and target column configured'}), 400
            
        processor = DataProcessor(project.dataset_path)
        data = processor.load_data()
        
        engineer = FeatureEngineer(
            project_id=project_id,
            data=data,
            target_column=project.target_column,
            date_column=project.date_column
        )
        
        # Save each feature type configuration
        for feature_type, settings in config_data.items():
            if feature_type.endswith('_enabled'):
                continue  # Skip enable flags
            
            feature_name = feature_type.replace('_config', '')
            is_enabled = config_data.get(f'{feature_name}_enabled', False)
            
            # Configure the feature type in the engineer
            engineer.config[f'{feature_name}_enabled'] = is_enabled
            if is_enabled and isinstance(settings, dict):
                engineer.config[f'{feature_name}_config'] = settings
        
        return jsonify({
            'success': True,
            'message': 'Feature configuration saved successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to save feature configuration: {str(e)}'}), 500

@app.route('/api/projects/<int:project_id>/features/generate', methods=['POST'])
def generate_features_api(project_id):
    """Generate features based on configuration"""
    project = Project.query.get_or_404(project_id)
    
    if not project.dataset_path or not os.path.exists(project.dataset_path):
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        # Load dataset
        processor = DataProcessor(project.dataset_path)
        df = processor.load_data()
        
        # Preprocess if needed
        if project.date_column and project.target_column:
            df = processor.preprocess_timeseries(project.date_column, project.target_column)
        
        # Generate features
        # Load data for feature engineer
        if not project.dataset_path or not project.target_column:
            return jsonify({'error': 'Project must have dataset and target column configured'}), 400
            
        processor = DataProcessor(project.dataset_path)
        data = processor.load_data()
        
        engineer = FeatureEngineer(
            project_id=project_id,
            data=data,
            target_column=project.target_column,
            date_column=project.date_column
        )
        df_with_features = engineer.generate_all_features()
        features_df = df_with_features
        
        # Get feature summary
        feature_summary = engineer.get_feature_summary()
        
        return jsonify({
            'success': True,
            'original_columns': len(df.columns),
            'generated_columns': len(features_df.columns),
            'new_features': len(features_df.columns) - len(df.columns),
            'feature_summary': feature_summary,
            'message': f'Generated {len(features_df.columns) - len(df.columns)} new features'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate features: {str(e)}'}), 500

@app.route('/api/projects/<int:project_id>/features/selection', methods=['POST'])
def feature_selection_api(project_id):
    """Perform automated feature selection"""
    project = Project.query.get_or_404(project_id)
    
    if not project.dataset_path or not project.target_column:
        return jsonify({'error': 'Dataset or target column not configured'}), 400
    
    try:
        data = request.get_json()
        selection_methods = data.get('methods', ['variance', 'univariate', 'rfe', 'lasso'])
        
        # Load and prepare dataset
        processor = DataProcessor(project.dataset_path)
        df = processor.load_data()
        df = processor.preprocess_timeseries(project.date_column, project.target_column)
        
        # Generate features
        # Load data for feature engineer
        if not project.dataset_path or not project.target_column:
            return jsonify({'error': 'Project must have dataset and target column configured'}), 400
            
        processor = DataProcessor(project.dataset_path)
        data = processor.load_data()
        
        engineer = FeatureEngineer(
            project_id=project_id,
            data=data,
            target_column=project.target_column,
            date_column=project.date_column
        )
        df_with_features = engineer.generate_all_features()
        features_df = df_with_features
        
        # Prepare data for feature selection
        X = features_df.drop(columns=[project.target_column])
        y = features_df[project.target_column]
        
        # Perform feature selection
        selected_features = engineer.optimize_feature_selection(X, y, selection_methods)
        
        # Get feature importance
        feature_importance = engineer.get_feature_importance(X, y)
        
        return jsonify({
            'success': True,
            'selected_features': selected_features,
            'feature_importance': feature_importance,
            'total_features': len(X.columns),
            'selected_count': len(selected_features),
            'selection_methods': selection_methods
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to perform feature selection: {str(e)}'}), 500

@app.route('/api/projects/<int:project_id>/optimize', methods=['POST'])
def start_optimization_api(project_id):
    """Start hyperparameter optimization experiment"""
    project = Project.query.get_or_404(project_id)
    
    if not project.dataset_path or not project.target_column:
        return jsonify({'error': 'Dataset or target column not configured'}), 400
    
    try:
        data = request.get_json()
        algorithm_type = data.get('algorithm_type', 'random_forest')
        n_trials = data.get('n_trials', 50)
        optimization_metric = data.get('metric', 'rmse')
        use_feature_engineering = data.get('use_feature_engineering', True)
        
        # Load and prepare dataset
        processor = DataProcessor(project.dataset_path)
        df = processor.load_data()
        df = processor.preprocess_timeseries(project.date_column, project.target_column)
        
        # Generate features if requested
        if use_feature_engineering:
            # Load data for feature engineer
            if not project.dataset_path or not project.target_column:
                return jsonify({'error': 'Project must have dataset and target column configured'}), 400
                
            processor_fe = DataProcessor(project.dataset_path)
            data_fe = processor_fe.load_data()
            
            engineer = FeatureEngineer(
                project_id=project_id,
                data=data_fe,
                target_column=project.target_column,
                date_column=project.date_column
            )
            df = engineer.generate_all_features()
        
        # Prepare data
        X = df.drop(columns=[project.target_column])
        y = df[project.target_column]
        
        # Split data
        split_point = int(len(df) * 0.8)
        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
        
        # Initialize optimizer with DataFrame data
        optimizer = ModelOptimizer(
            project_id=project_id,
            data=df,
            target_column=project.target_column,
            date_column=project.date_column,
            objective=optimization_metric
        )
        
        # Run optimization using algorithm-specific method
        if algorithm_type == 'random_forest':
            result = optimizer.optimize_random_forest(n_trials=n_trials)
        elif algorithm_type == 'lightgbm':
            result = optimizer.optimize_lightgbm(n_trials=n_trials)
        elif algorithm_type == 'xgboost':
            result = optimizer.optimize_xgboost(n_trials=n_trials)
        elif algorithm_type == 'arima':
            result = optimizer.optimize_arima(n_trials=n_trials)
        elif algorithm_type == 'sarimax':
            result = optimizer.optimize_sarimax(n_trials=n_trials)
        elif algorithm_type == 'lstm':
            result = optimizer.optimize_lstm(n_trials=n_trials)
        elif algorithm_type == 'moving_average':
            result = optimizer.optimize_moving_average(n_trials=n_trials)
        else:
            return jsonify({'error': f'Optimization not supported for {algorithm_type}'}), 400
        
        return jsonify({
            'success': True,
            'experiment_id': result.get('experiment_id'),
            'best_params': result['best_params'],
            'best_score': result['best_score'],
            'n_trials': result['n_trials'],
            'message': f'Optimization completed with best {optimization_metric}: {result["best_score"]:.4f}'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start optimization: {str(e)}'}), 500

@app.route('/api/optimization/<int:experiment_id>/status', methods=['GET'])
def get_optimization_status_api(experiment_id):
    """Get optimization experiment status and progress"""
    try:
        optimizer = ModelOptimizer()
        progress = optimizer.get_optimization_progress(experiment_id)
        
        if not progress:
            return jsonify({'error': 'Experiment not found'}), 404
        
        return jsonify({
            'success': True,
            'progress': progress
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get optimization status: {str(e)}'}), 500

@app.route('/api/projects/<int:project_id>/train-optimized', methods=['POST'])
def train_optimized_model_api(project_id):
    """Train model using optimized hyperparameters"""
    project = Project.query.get_or_404(project_id)
    
    try:
        data = request.get_json()
        experiment_id = data.get('experiment_id')
        model_name = data.get('model_name', f'optimized_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        if not experiment_id:
            return jsonify({'error': 'Experiment ID required'}), 400
        
        # Get optimization experiment
        experiment = OptimizationExperiment.query.get(experiment_id)
        if not experiment:
            return jsonify({'error': 'Optimization experiment not found'}), 404
        
        # Load dataset and prepare features
        processor = DataProcessor(project.dataset_path)
        df = processor.load_data()
        df = processor.preprocess_timeseries(project.date_column, project.target_column)
        
        # Generate features
        # Load data for feature engineer
        if not project.dataset_path or not project.target_column:
            return jsonify({'error': 'Project must have dataset and target column configured'}), 400
            
        processor = DataProcessor(project.dataset_path)
        data = processor.load_data()
        
        engineer = FeatureEngineer(
            project_id=project_id,
            data=data,
            target_column=project.target_column,
            date_column=project.date_column
        )
        df = engineer.generate_all_features()
        
        # Train model with optimized parameters
        algorithm_type = experiment.algorithm_type
        best_params = experiment.get_best_parameters()
        
        # Use existing forecasting engine with optimized parameters
        engine = ForecastingEngine(project.dataset_path, project.date_column, project.target_column)
        
        # Train the model based on algorithm type
        if algorithm_type == 'random_forest':
            result = engine.train_random_forest(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', 10)
            )
        elif algorithm_type == 'lightgbm':
            result = engine.train_lightgbm(
                num_leaves=best_params.get('num_leaves', 31),
                learning_rate=best_params.get('learning_rate', 0.1),
                n_estimators=best_params.get('n_estimators', 100)
            )
        else:
            return jsonify({'error': f'Training with {algorithm_type} not yet implemented'}), 400
        
        # Save model result with optimization info
        model_result = ModelResult(
            project_id=project_id,
            model_name=model_name,
            model_type=f'{algorithm_type}_optimized',
            optimization_experiment_id=experiment_id,
            rmse=result['metrics']['rmse'],
            mae=result['metrics']['mae'],
            mape=result['metrics']['mape'],
            r2_score=result['metrics'].get('r2_score'),
            training_samples=result['training_samples'],
            test_samples=result['test_samples']
        )
        
        model_result.set_parameters({**result['parameters'], **best_params})
        model_result.set_forecast_data(result['forecast_data'])
        model_result.set_advanced_metrics({'optimization_score': experiment.best_score})
        
        db.session.add(model_result)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'model_id': model_result.id,
            'model_name': model_name,
            'metrics': result['metrics'],
            'optimization_score': experiment.best_score,
            'message': 'Optimized model trained successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to train optimized model: {str(e)}'}), 500

@app.route('/api/projects/<int:project_id>/experiments', methods=['GET'])
def list_experiments_api(project_id):
    """List all optimization experiments for a project"""
    try:
        experiments = OptimizationExperiment.query.filter_by(project_id=project_id).order_by(
            OptimizationExperiment.created_at.desc()
        ).all()
        
        experiments_data = []
        for exp in experiments:
            experiments_data.append({
                'id': exp.id,
                'algorithm_type': exp.algorithm_type,
                'status': exp.status,
                'best_score': exp.best_score,
                'n_trials': exp.n_trials,
                'created_at': exp.created_at.isoformat(),
                'updated_at': exp.updated_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'experiments': experiments_data,
            'total': len(experiments_data)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to list experiments: {str(e)}'}), 500

@app.route('/api/experiments/compare', methods=['POST'])
def compare_experiments_api():
    """Compare multiple optimization experiments"""
    try:
        data = request.get_json()
        experiment_ids = data.get('experiment_ids', [])
        
        if not experiment_ids:
            return jsonify({'error': 'No experiment IDs provided'}), 400
        
        optimizer = ModelOptimizer()
        comparison = optimizer.compare_optimization_results(experiment_ids)
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to compare experiments: {str(e)}'}), 500


@app.route('/api/projects/<int:project_id>/profile/sweetviz-report', methods=['POST'])
def generate_sweetviz_report(project_id):
    """Generate Sweetviz report for project data"""
    project = Project.query.get_or_404(project_id)
    
    try:
        # Check if project has dataset
        if not project.dataset_path or not os.path.exists(project.dataset_path):
            return jsonify({'error': 'Dataset not found'}), 404
            
        # Import matplotlib and set non-GUI backend before importing sweetviz
        import matplotlib
        matplotlib.use('Agg')  # Set non-interactive backend
        import sweetviz as sv
        
        # Load dataset
        processor = DataProcessor(project.dataset_path)
        df = processor.load_data()
        
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.getcwd(), 'static', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"sweetviz_report_project_{project_id}_{timestamp}.html"
        report_path = os.path.join(reports_dir, report_filename)
        
        # Generate Sweetviz report
        print(f"Generating Sweetviz report for project {project_id}")
        my_report = sv.analyze(df)
        my_report.show_html(filepath=report_path, open_browser=False)
        
        # Return URL path for the report
        report_url = f"/static/reports/{report_filename}"
        
        return jsonify({
            'success': True,
            'message': 'Sweetviz report generated successfully',
            'report_url': report_url,
            'report_filename': report_filename
        })
        
    except Exception as e:
        print(f"Error generating Sweetviz report: {str(e)}")
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500

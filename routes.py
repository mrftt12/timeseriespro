import os
import pandas as pd
from flask import render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import io
import base64

from app import app, db
from models import Project, ModelResult
from data_processor import DataProcessor
from forecasting import ForecastingEngine, LIGHTGBM_AVAILABLE

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

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
        
        return render_template('project.html', 
                             project=project, 
                             dataset_info=dataset_info,
                             models=models,
                             lightgbm_available=LIGHTGBM_AVAILABLE)
                             
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

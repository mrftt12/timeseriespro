"""
Celery tasks for data processing operations
Part of Epic #2: Advanced Data Science Features
"""

from celery import current_task
from celery_app import celery_app
import pandas as pd
from datetime import datetime
import traceback

# Flask app context
from app import app, db
from models import Project, DataProfile
from data_profiler import DataProfiler
from external_data import ExternalDataConnector
from data_processor import DataProcessor


@celery_app.task(bind=True)
def generate_data_profile_async(self, project_id: int):
    """Asynchronous data profiling task"""
    with app.app_context():
        try:
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Loading project', 'progress': 10})
            
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            if not project.dataset_path:
                raise ValueError("Dataset path not configured")
            
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Loading dataset', 'progress': 20})
            
            # Load dataset
            processor = DataProcessor(project.dataset_path)
            df = processor.load_data()
            
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Generating profile', 'progress': 50})
            
            # Generate profile
            profiler = DataProfiler(project_id)
            profile_results = profiler.generate_profile(df, project.target_column)
            
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Saving results', 'progress': 90})
            
            return {
                'status': 'completed',
                'project_id': project_id,
                'profile_summary': {
                    'columns_analyzed': len(profile_results.get('column_profiles', {})),
                    'overall_quality_score': profile_results.get('data_quality_summary', {}).get('overall_quality_score', 0),
                    'issues_found': len(profile_results.get('recommendations', [])),
                },
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            error_msg = f"Data profiling failed: {str(exc)}"
            self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': traceback.format_exc()})
            raise


@celery_app.task(bind=True)
def sync_external_data_async(self, project_id: int, source_id: int, date_range: dict = None):
    """Asynchronous external data synchronization"""
    with app.app_context():
        try:
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Initializing connector', 'progress': 10})
            
            connector = ExternalDataConnector(project_id)
            
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Fetching external data', 'progress': 30})
            
            # Sync data
            result = connector.sync_external_data(source_id, date_range)
            
            # Update task status
            self.update_state(state='PROGRESS', meta={'step': 'Processing data', 'progress': 80})
            
            if result['success']:
                return {
                    'status': 'completed',
                    'project_id': project_id,
                    'source_id': source_id,
                    'records_fetched': result['records_fetched'],
                    'completed_at': datetime.utcnow().isoformat()
                }
            else:
                raise Exception(result['error'])
                
        except Exception as exc:
            error_msg = f"External data sync failed: {str(exc)}"
            self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': traceback.format_exc()})
            raise


@celery_app.task(bind=True)
def validate_dataset_async(self, project_id: int):
    """Asynchronous dataset validation"""
    with app.app_context():
        try:
            self.update_state(state='PROGRESS', meta={'step': 'Loading project', 'progress': 10})
            
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            self.update_state(state='PROGRESS', meta={'step': 'Loading dataset', 'progress': 30})
            
            processor = DataProcessor(project.dataset_path)
            df = processor.load_data()
            
            self.update_state(state='PROGRESS', meta={'step': 'Running validations', 'progress': 60})
            
            # Basic validations
            validations = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_data': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024*1024),
                'validation_passed': True,
                'issues': []
            }
            
            # Check for issues
            if validations['missing_percentage'] > 30:
                validations['issues'].append('High percentage of missing data (>30%)')
                validations['validation_passed'] = False
            
            if validations['duplicate_rows'] > len(df) * 0.1:
                validations['issues'].append('High number of duplicate rows (>10%)')
                validations['validation_passed'] = False
            
            if len(df) < 50:
                validations['issues'].append('Dataset too small for reliable modeling (<50 rows)')
                validations['validation_passed'] = False
            
            self.update_state(state='PROGRESS', meta={'step': 'Finalizing validation', 'progress': 90})
            
            return {
                'status': 'completed',
                'project_id': project_id,
                'validation_results': validations,
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            error_msg = f"Dataset validation failed: {str(exc)}"
            self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': traceback.format_exc()})
            raise


@celery_app.task(bind=True)
def preprocess_dataset_async(self, project_id: int, preprocessing_config: dict):
    """Asynchronous dataset preprocessing"""
    with app.app_context():
        try:
            self.update_state(state='PROGRESS', meta={'step': 'Loading project', 'progress': 10})
            
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            self.update_state(state='PROGRESS', meta={'step': 'Loading dataset', 'progress': 20})
            
            processor = DataProcessor(project.dataset_path)
            df = processor.load_data()
            original_shape = df.shape
            
            self.update_state(state='PROGRESS', meta={'step': 'Applying preprocessing', 'progress': 50})
            
            # Apply preprocessing steps
            if preprocessing_config.get('remove_duplicates', False):
                df = df.drop_duplicates()
            
            if preprocessing_config.get('handle_missing', False):
                missing_strategy = preprocessing_config.get('missing_strategy', 'drop')
                if missing_strategy == 'drop':
                    df = df.dropna()
                elif missing_strategy == 'fill_mean':
                    numeric_columns = df.select_dtypes(include=['number']).columns
                    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
                elif missing_strategy == 'fill_median':
                    numeric_columns = df.select_dtypes(include=['number']).columns
                    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            
            if preprocessing_config.get('remove_outliers', False):
                # Simple outlier removal using IQR
                numeric_columns = df.select_dtypes(include=['number']).columns
                for col in numeric_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            self.update_state(state='PROGRESS', meta={'step': 'Finalizing preprocessing', 'progress': 90})
            
            final_shape = df.shape
            
            return {
                'status': 'completed',
                'project_id': project_id,
                'preprocessing_results': {
                    'original_shape': original_shape,
                    'final_shape': final_shape,
                    'rows_removed': original_shape[0] - final_shape[0],
                    'preprocessing_config': preprocessing_config
                },
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            error_msg = f"Dataset preprocessing failed: {str(exc)}"
            self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': traceback.format_exc()})
            raise


@celery_app.task(bind=True)
def export_data_async(self, project_id: int, export_format: str = 'csv', include_features: bool = False):
    """Asynchronous data export"""
    with app.app_context():
        try:
            self.update_state(state='PROGRESS', meta={'step': 'Loading project', 'progress': 10})
            
            project = Project.query.get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            self.update_state(state='PROGRESS', meta={'step': 'Loading dataset', 'progress': 30})
            
            processor = DataProcessor(project.dataset_path)
            df = processor.load_data()
            
            # Include features if requested
            if include_features:
                self.update_state(state='PROGRESS', meta={'step': 'Generating features', 'progress': 50})
                from feature_engineer import FeatureEngineer
                engineer = FeatureEngineer(project_id)
                df = engineer.generate_features(df)
            
            self.update_state(state='PROGRESS', meta={'step': 'Exporting data', 'progress': 80})
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"project_{project_id}_data_{timestamp}.{export_format}"
            
            # For now, just return metadata (in practice, you'd save to file system or cloud storage)
            export_info = {
                'filename': filename,
                'format': export_format,
                'rows': len(df),
                'columns': len(df.columns),
                'file_size_mb': df.memory_usage(deep=True).sum() / (1024*1024),
                'include_features': include_features
            }
            
            return {
                'status': 'completed',
                'project_id': project_id,
                'export_info': export_info,
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as exc:
            error_msg = f"Data export failed: {str(exc)}"
            self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': traceback.format_exc()})
            raise
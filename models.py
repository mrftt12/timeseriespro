from datetime import datetime
from app import db
import json

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Dataset information
    dataset_filename = db.Column(db.String(255))
    dataset_path = db.Column(db.String(500))
    date_column = db.Column(db.String(100))
    target_column = db.Column(db.String(100))
    
    # Preprocessing settings
    preprocessing_config = db.Column(db.Text)  # JSON string
    
    # Model results
    models = db.relationship('ModelResult', backref='project', lazy=True, cascade='all, delete-orphan')
    
    def get_preprocessing_config(self):
        if self.preprocessing_config:
            return json.loads(self.preprocessing_config)
        return {}
    
    def set_preprocessing_config(self, config):
        self.preprocessing_config = json.dumps(config)

class ModelResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    
    model_name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # ARIMA, LinearRegression, MovingAverage
    
    # Model parameters
    parameters = db.Column(db.Text)  # JSON string
    
    # Performance metrics
    rmse = db.Column(db.Float)
    mae = db.Column(db.Float)  
    mape = db.Column(db.Float)
    r2_score = db.Column(db.Float)
    
    # Training information
    training_samples = db.Column(db.Integer)
    test_samples = db.Column(db.Integer)
    
    # Forecast data
    forecast_data = db.Column(db.Text)  # JSON string containing forecast results
    
    # Advanced features for Epic #2
    optimization_experiment_id = db.Column(db.Integer, db.ForeignKey('optimization_experiment.id'))
    ensemble_config = db.Column(db.Text)  # JSON ensemble configuration
    advanced_metrics = db.Column(db.Text)  # JSON additional metrics
    diagnostic_data = db.Column(db.Text)  # JSON residual analysis, etc.
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship for optimization experiments
    optimization_experiment = db.relationship('OptimizationExperiment', backref=db.backref('model_results', lazy=True))
    
    def get_parameters(self):
        if self.parameters:
            return json.loads(self.parameters)
        return {}
    
    def set_parameters(self, params):
        self.parameters = json.dumps(params)
    
    def get_forecast_data(self):
        if self.forecast_data:
            return json.loads(self.forecast_data)
        return {}
    
    def set_forecast_data(self, data):
        self.forecast_data = json.dumps(data)
    
    def get_ensemble_config(self):
        if self.ensemble_config:
            return json.loads(self.ensemble_config)
        return {}
    
    def set_ensemble_config(self, config):
        self.ensemble_config = json.dumps(config)
    
    def get_advanced_metrics(self):
        if self.advanced_metrics:
            return json.loads(self.advanced_metrics)
        return {}
    
    def set_advanced_metrics(self, metrics):
        self.advanced_metrics = json.dumps(metrics)
    
    def get_diagnostic_data(self):
        if self.diagnostic_data:
            return json.loads(self.diagnostic_data)
        return {}
    
    def set_diagnostic_data(self, data):
        self.diagnostic_data = json.dumps(data)


# Advanced Data Science Models for Epic #2

class DataProfile(db.Model):
    """Data profiles and quality metrics for comprehensive data analysis"""
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    column_name = db.Column(db.String(100), nullable=False)
    data_type = db.Column(db.String(50))
    missing_count = db.Column(db.Integer)
    missing_percentage = db.Column(db.Float)
    outlier_count = db.Column(db.Integer)
    statistical_summary = db.Column(db.Text)  # JSON
    quality_score = db.Column(db.Float)
    recommendations = db.Column(db.Text)  # JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    project = db.relationship('Project', backref=db.backref('data_profiles', lazy=True, cascade='all, delete-orphan'))
    
    def get_statistical_summary(self):
        if self.statistical_summary:
            return json.loads(self.statistical_summary)
        return {}
    
    def set_statistical_summary(self, summary):
        self.statistical_summary = json.dumps(summary)
    
    def get_recommendations(self):
        if self.recommendations:
            return json.loads(self.recommendations)
        return []
    
    def set_recommendations(self, recommendations):
        self.recommendations = json.dumps(recommendations)


class FeatureConfig(db.Model):
    """Feature engineering configurations"""
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    feature_type = db.Column(db.String(50), nullable=False)  # lag, rolling, calendar, technical
    configuration = db.Column(db.Text)  # JSON with feature parameters
    is_enabled = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    project = db.relationship('Project', backref=db.backref('feature_configs', lazy=True, cascade='all, delete-orphan'))
    
    def get_configuration(self):
        if self.configuration:
            return json.loads(self.configuration)
        return {}
    
    def set_configuration(self, config):
        self.configuration = json.dumps(config)


class PreprocessingPipeline(db.Model):
    """Advanced preprocessing pipelines"""
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    pipeline_name = db.Column(db.String(100), nullable=False)
    steps = db.Column(db.Text)  # JSON array of processing steps
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    project = db.relationship('Project', backref=db.backref('preprocessing_pipelines', lazy=True, cascade='all, delete-orphan'))
    
    def get_steps(self):
        if self.steps:
            return json.loads(self.steps)
        return []
    
    def set_steps(self, steps):
        self.steps = json.dumps(steps)


class OptimizationExperiment(db.Model):
    """Hyperparameter optimization experiments"""
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    algorithm_type = db.Column(db.String(50), nullable=False)
    search_space = db.Column(db.Text)  # JSON parameter space
    best_parameters = db.Column(db.Text)  # JSON best params
    trials_data = db.Column(db.Text)  # JSON optimization history
    status = db.Column(db.String(20), default='pending')  # running, completed, failed
    n_trials = db.Column(db.Integer, default=100)
    best_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    project = db.relationship('Project', backref=db.backref('optimization_experiments', lazy=True, cascade='all, delete-orphan'))
    
    def get_search_space(self):
        if self.search_space:
            return json.loads(self.search_space)
        return {}
    
    def set_search_space(self, space):
        self.search_space = json.dumps(space)
    
    def get_best_parameters(self):
        if self.best_parameters:
            return json.loads(self.best_parameters)
        return {}
    
    def set_best_parameters(self, params):
        self.best_parameters = json.dumps(params)
    
    def get_trials_data(self):
        if self.trials_data:
            return json.loads(self.trials_data)
        return []
    
    def set_trials_data(self, trials):
        self.trials_data = json.dumps(trials)


class ExternalDataSource(db.Model):
    """External data integrations"""
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    source_type = db.Column(db.String(50), nullable=False)  # holidays, weather, economic
    api_configuration = db.Column(db.Text)  # JSON API config
    data_mapping = db.Column(db.Text)  # JSON field mappings
    last_sync = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    project = db.relationship('Project', backref=db.backref('external_data_sources', lazy=True, cascade='all, delete-orphan'))
    
    def get_api_configuration(self):
        if self.api_configuration:
            return json.loads(self.api_configuration)
        return {}
    
    def set_api_configuration(self, config):
        self.api_configuration = json.dumps(config)
    
    def get_data_mapping(self):
        if self.data_mapping:
            return json.loads(self.data_mapping)
        return {}
    
    def set_data_mapping(self, mapping):
        self.data_mapping = json.dumps(mapping)


# Extended Model Results for Advanced Features
# Add new columns to ModelResult via ALTER TABLE statements in migration

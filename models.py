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
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
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

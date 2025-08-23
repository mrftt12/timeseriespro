"""
Unit tests for Time Series Pro database models - Phase 1
Tests all new models, relationships, and JSON serialization functionality
"""

import unittest
import json
import tempfile
import os
from datetime import datetime

from app import app, db
from models import (
    Project, ModelResult, DataProfile, FeatureConfig,
    PreprocessingPipeline, OptimizationExperiment, ExternalDataSource
)


class TestDatabaseModels(unittest.TestCase):
    """Test cases for all database models and relationships"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database"""
        cls.db_fd, cls.db_path = tempfile.mkstemp()
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{cls.db_path}'
        app.config['TESTING'] = True
        
    def setUp(self):
        """Set up test fixtures before each test"""
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        db.create_all()
        
        # Create a test project for relationship testing
        self.test_project = Project(
            name="Test Project",
            description="A test project for unit testing",
            date_column="date",
            target_column="value"
        )
        db.session.add(self.test_project)
        db.session.commit()
        
    def tearDown(self):
        """Clean up after each test"""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test database"""
        os.close(cls.db_fd)
        os.unlink(cls.db_path)
    
    def test_project_model(self):
        """Test Project model basic functionality"""
        project = Project(
            name="Sample Project",
            description="Test description",
            dataset_filename="test.csv",
            date_column="timestamp",
            target_column="sales"
        )
        
        # Test preprocessing config JSON methods
        config = {"normalize": True, "handle_outliers": "remove"}
        project.set_preprocessing_config(config)
        
        db.session.add(project)
        db.session.commit()
        
        # Verify the project was saved
        saved_project = Project.query.filter_by(name="Sample Project").first()
        self.assertIsNotNone(saved_project)
        self.assertEqual(saved_project.description, "Test description")
        self.assertEqual(saved_project.dataset_filename, "test.csv")
        self.assertEqual(saved_project.date_column, "timestamp")
        self.assertEqual(saved_project.target_column, "sales")
        
        # Test JSON serialization
        retrieved_config = saved_project.get_preprocessing_config()
        self.assertEqual(retrieved_config["normalize"], True)
        self.assertEqual(retrieved_config["handle_outliers"], "remove")
    
    def test_model_result_basic(self):
        """Test ModelResult model basic functionality"""
        model_result = ModelResult(
            project_id=self.test_project.id,
            model_name="Test ARIMA",
            model_type="ARIMA",
            rmse=12.5,
            mae=8.3,
            mape=0.15,
            r2_score=0.85,
            training_samples=100,
            test_samples=25
        )
        
        # Test JSON methods
        params = {"p": 2, "d": 1, "q": 2}
        model_result.set_parameters(params)
        
        forecast_data = {
            "historical": {"dates": ["2023-01-01"], "values": [100]},
            "forecast": {"dates": ["2024-01-01"], "values": [105]}
        }
        model_result.set_forecast_data(forecast_data)
        
        ensemble_config = {"models": ["arima", "prophet"], "weights": [0.6, 0.4]}
        model_result.set_ensemble_config(ensemble_config)
        
        advanced_metrics = {"aic": 150.2, "bic": 158.7}
        model_result.set_advanced_metrics(advanced_metrics)
        
        diagnostic_data = {"residuals": [1.2, -0.8, 0.5], "ljung_box_p": 0.24}
        model_result.set_diagnostic_data(diagnostic_data)
        
        db.session.add(model_result)
        db.session.commit()
        
        # Verify the model was saved
        saved_model = ModelResult.query.filter_by(model_name="Test ARIMA").first()
        self.assertIsNotNone(saved_model)
        self.assertEqual(saved_model.model_type, "ARIMA")
        self.assertEqual(saved_model.rmse, 12.5)
        self.assertEqual(saved_model.project_id, self.test_project.id)
        
        # Test JSON deserialization
        self.assertEqual(saved_model.get_parameters()["p"], 2)
        self.assertEqual(saved_model.get_forecast_data()["historical"]["values"][0], 100)
        self.assertEqual(saved_model.get_ensemble_config()["weights"][0], 0.6)
        self.assertEqual(saved_model.get_advanced_metrics()["aic"], 150.2)
        self.assertEqual(saved_model.get_diagnostic_data()["ljung_box_p"], 0.24)
    
    def test_data_profile_model(self):
        """Test DataProfile model and relationships"""
        data_profile = DataProfile(
            project_id=self.test_project.id,
            column_name="sales",
            data_type="float64",
            missing_count=5,
            missing_percentage=0.05,
            outlier_count=3,
            quality_score=0.92
        )
        
        # Test JSON methods
        summary = {"mean": 1250.5, "std": 340.2, "min": 500, "max": 2500}
        data_profile.set_statistical_summary(summary)
        
        recommendations = [
            {"type": "outlier_treatment", "action": "cap_at_percentile", "params": {"percentile": 95}},
            {"type": "missing_value", "action": "interpolate", "params": {"method": "linear"}}
        ]
        data_profile.set_recommendations(recommendations)
        
        db.session.add(data_profile)
        db.session.commit()
        
        # Verify the profile was saved
        saved_profile = DataProfile.query.filter_by(column_name="sales").first()
        self.assertIsNotNone(saved_profile)
        self.assertEqual(saved_profile.project_id, self.test_project.id)
        self.assertEqual(saved_profile.data_type, "float64")
        self.assertEqual(saved_profile.quality_score, 0.92)
        
        # Test relationship
        self.assertEqual(saved_profile.project.name, "Test Project")
        
        # Test JSON methods
        retrieved_summary = saved_profile.get_statistical_summary()
        self.assertEqual(retrieved_summary["mean"], 1250.5)
        
        retrieved_recommendations = saved_profile.get_recommendations()
        self.assertEqual(len(retrieved_recommendations), 2)
        self.assertEqual(retrieved_recommendations[0]["type"], "outlier_treatment")
    
    def test_feature_config_model(self):
        """Test FeatureConfig model and relationships"""
        feature_config = FeatureConfig(
            project_id=self.test_project.id,
            feature_type="lag",
            is_enabled=True
        )
        
        # Test configuration JSON
        config = {
            "lags": [1, 7, 30],
            "target_column": "sales",
            "naming_convention": "sales_lag_{lag}"
        }
        feature_config.set_configuration(config)
        
        db.session.add(feature_config)
        db.session.commit()
        
        # Verify the feature config was saved
        saved_feature = FeatureConfig.query.filter_by(feature_type="lag").first()
        self.assertIsNotNone(saved_feature)
        self.assertEqual(saved_feature.project_id, self.test_project.id)
        self.assertTrue(saved_feature.is_enabled)
        
        # Test relationship
        self.assertEqual(saved_feature.project.name, "Test Project")
        
        # Test JSON configuration
        retrieved_config = saved_feature.get_configuration()
        self.assertEqual(retrieved_config["lags"], [1, 7, 30])
        self.assertEqual(retrieved_config["target_column"], "sales")
    
    def test_preprocessing_pipeline_model(self):
        """Test PreprocessingPipeline model and relationships"""
        pipeline = PreprocessingPipeline(
            project_id=self.test_project.id,
            pipeline_name="Standard Preprocessing",
            is_active=True
        )
        
        # Test steps JSON
        steps = [
            {"step": "remove_outliers", "method": "iqr", "factor": 1.5},
            {"step": "fill_missing", "method": "interpolate", "order": 1},
            {"step": "normalize", "method": "minmax", "feature_range": [0, 1]}
        ]
        pipeline.set_steps(steps)
        
        db.session.add(pipeline)
        db.session.commit()
        
        # Verify the pipeline was saved
        saved_pipeline = PreprocessingPipeline.query.filter_by(pipeline_name="Standard Preprocessing").first()
        self.assertIsNotNone(saved_pipeline)
        self.assertEqual(saved_pipeline.project_id, self.test_project.id)
        self.assertTrue(saved_pipeline.is_active)
        
        # Test relationship
        self.assertEqual(saved_pipeline.project.name, "Test Project")
        
        # Test JSON steps
        retrieved_steps = saved_pipeline.get_steps()
        self.assertEqual(len(retrieved_steps), 3)
        self.assertEqual(retrieved_steps[0]["step"], "remove_outliers")
        self.assertEqual(retrieved_steps[1]["method"], "interpolate")
    
    def test_optimization_experiment_model(self):
        """Test OptimizationExperiment model and relationships"""
        experiment = OptimizationExperiment(
            project_id=self.test_project.id,
            algorithm_type="ARIMA",
            status="completed",
            n_trials=50,
            best_score=0.85
        )
        
        # Test JSON methods
        search_space = {
            "p": {"type": "int", "low": 0, "high": 5},
            "d": {"type": "int", "low": 0, "high": 2},
            "q": {"type": "int", "low": 0, "high": 5}
        }
        experiment.set_search_space(search_space)
        
        best_params = {"p": 2, "d": 1, "q": 2}
        experiment.set_best_parameters(best_params)
        
        trials_data = [
            {"trial": 1, "params": {"p": 1, "d": 1, "q": 1}, "score": 0.75},
            {"trial": 2, "params": {"p": 2, "d": 1, "q": 2}, "score": 0.85}
        ]
        experiment.set_trials_data(trials_data)
        
        db.session.add(experiment)
        db.session.commit()
        
        # Verify the experiment was saved
        saved_experiment = OptimizationExperiment.query.filter_by(algorithm_type="ARIMA").first()
        self.assertIsNotNone(saved_experiment)
        self.assertEqual(saved_experiment.project_id, self.test_project.id)
        self.assertEqual(saved_experiment.status, "completed")
        self.assertEqual(saved_experiment.best_score, 0.85)
        
        # Test relationship
        self.assertEqual(saved_experiment.project.name, "Test Project")
        
        # Test JSON methods
        retrieved_space = saved_experiment.get_search_space()
        self.assertEqual(retrieved_space["p"]["high"], 5)
        
        retrieved_params = saved_experiment.get_best_parameters()
        self.assertEqual(retrieved_params["p"], 2)
        
        retrieved_trials = saved_experiment.get_trials_data()
        self.assertEqual(len(retrieved_trials), 2)
        self.assertEqual(retrieved_trials[1]["score"], 0.85)
    
    def test_external_data_source_model(self):
        """Test ExternalDataSource model and relationships"""
        data_source = ExternalDataSource(
            project_id=self.test_project.id,
            source_type="weather",
            last_sync=datetime.utcnow(),
            is_active=True
        )
        
        # Test JSON methods
        api_config = {
            "api_key": "test_key_123",
            "base_url": "https://api.weather.com",
            "endpoints": {
                "historical": "/v1/historical",
                "forecast": "/v1/forecast"
            }
        }
        data_source.set_api_configuration(api_config)
        
        data_mapping = {
            "temperature": "temp_celsius",
            "humidity": "humidity_percent",
            "precipitation": "precip_mm"
        }
        data_source.set_data_mapping(data_mapping)
        
        db.session.add(data_source)
        db.session.commit()
        
        # Verify the data source was saved
        saved_source = ExternalDataSource.query.filter_by(source_type="weather").first()
        self.assertIsNotNone(saved_source)
        self.assertEqual(saved_source.project_id, self.test_project.id)
        self.assertTrue(saved_source.is_active)
        
        # Test relationship
        self.assertEqual(saved_source.project.name, "Test Project")
        
        # Test JSON methods
        retrieved_config = saved_source.get_api_configuration()
        self.assertEqual(retrieved_config["api_key"], "test_key_123")
        self.assertEqual(retrieved_config["base_url"], "https://api.weather.com")
        
        retrieved_mapping = saved_source.get_data_mapping()
        self.assertEqual(retrieved_mapping["temperature"], "temp_celsius")
    
    def test_model_result_optimization_relationship(self):
        """Test the relationship between ModelResult and OptimizationExperiment"""
        # Create an optimization experiment
        experiment = OptimizationExperiment(
            project_id=self.test_project.id,
            algorithm_type="Prophet",
            status="completed",
            best_score=0.90
        )
        db.session.add(experiment)
        db.session.commit()
        
        # Create a model result linked to the experiment
        model_result = ModelResult(
            project_id=self.test_project.id,
            model_name="Optimized Prophet",
            model_type="Prophet",
            optimization_experiment_id=experiment.id,
            rmse=10.2,
            mae=7.5
        )
        db.session.add(model_result)
        db.session.commit()
        
        # Test the relationship
        saved_model = ModelResult.query.filter_by(model_name="Optimized Prophet").first()
        self.assertIsNotNone(saved_model.optimization_experiment)
        self.assertEqual(saved_model.optimization_experiment.algorithm_type, "Prophet")
        self.assertEqual(saved_model.optimization_experiment.best_score, 0.90)
        
        # Test backref
        saved_experiment = OptimizationExperiment.query.filter_by(algorithm_type="Prophet").first()
        self.assertEqual(len(saved_experiment.model_results), 1)
        self.assertEqual(saved_experiment.model_results[0].model_name, "Optimized Prophet")
    
    def test_project_relationships(self):
        """Test all relationships from Project to other models"""
        # Add various related models
        data_profile = DataProfile(project_id=self.test_project.id, column_name="test_col")
        feature_config = FeatureConfig(project_id=self.test_project.id, feature_type="test_feature")
        pipeline = PreprocessingPipeline(project_id=self.test_project.id, pipeline_name="test_pipeline")
        experiment = OptimizationExperiment(project_id=self.test_project.id, algorithm_type="test_algo")
        data_source = ExternalDataSource(project_id=self.test_project.id, source_type="test_source")
        model_result = ModelResult(
            project_id=self.test_project.id,
            model_name="Test Model",
            model_type="TestType"
        )
        
        db.session.add_all([data_profile, feature_config, pipeline, experiment, data_source, model_result])
        db.session.commit()
        
        # Test all relationships
        project = Project.query.get(self.test_project.id)
        self.assertEqual(len(project.data_profiles), 1)
        self.assertEqual(len(project.feature_configs), 1)
        self.assertEqual(len(project.preprocessing_pipelines), 1)
        self.assertEqual(len(project.optimization_experiments), 1)
        self.assertEqual(len(project.external_data_sources), 1)
        self.assertEqual(len(project.models), 1)
        
        # Verify cascade delete behavior
        db.session.delete(project)
        db.session.commit()
        
        # All related objects should be deleted
        self.assertEqual(DataProfile.query.count(), 0)
        self.assertEqual(FeatureConfig.query.count(), 0)
        self.assertEqual(PreprocessingPipeline.query.count(), 0)
        self.assertEqual(OptimizationExperiment.query.count(), 0)
        self.assertEqual(ExternalDataSource.query.count(), 0)
        self.assertEqual(ModelResult.query.count(), 0)


if __name__ == '__main__':
    unittest.main()
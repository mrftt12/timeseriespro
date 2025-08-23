"""
Backward Compatibility Test for Time Series Pro
Ensures all existing functionality still works after database schema evolution
"""

import unittest
import tempfile
import os
import json
from datetime import datetime

from app import app, db
from models import Project, ModelResult


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility of existing functionality"""
    
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
    
    def test_existing_project_functionality(self):
        """Test that existing Project model functionality still works"""
        # Create a project exactly as before migration
        project = Project(
            name="Legacy Test Project",
            description="Testing backward compatibility",
            dataset_filename="test_data.csv",
            dataset_path="/uploads/test_data.csv",
            date_column="timestamp",
            target_column="sales"
        )
        
        # Test preprocessing config (this was already JSON before migration)
        preprocessing_config = {
            "remove_outliers": True,
            "normalize": "minmax",
            "fill_missing": "interpolate"
        }
        project.set_preprocessing_config(preprocessing_config)
        
        db.session.add(project)
        db.session.commit()
        
        # Verify the project was saved and can be retrieved
        saved_project = Project.query.filter_by(name="Legacy Test Project").first()
        self.assertIsNotNone(saved_project)
        self.assertEqual(saved_project.description, "Testing backward compatibility")
        self.assertEqual(saved_project.dataset_filename, "test_data.csv")
        self.assertEqual(saved_project.date_column, "timestamp")
        self.assertEqual(saved_project.target_column, "sales")
        
        # Test that preprocessing config JSON methods still work
        retrieved_config = saved_project.get_preprocessing_config()
        self.assertEqual(retrieved_config["remove_outliers"], True)
        self.assertEqual(retrieved_config["normalize"], "minmax")
        self.assertEqual(retrieved_config["fill_missing"], "interpolate")
        
        print("✓ Existing Project functionality is fully compatible")
    
    def test_existing_model_result_functionality(self):
        """Test that existing ModelResult functionality still works"""
        # Create a project first
        project = Project(name="Test Project", description="For model results")
        db.session.add(project)
        db.session.commit()
        
        # Create a model result exactly as before migration
        model_result = ModelResult(
            project_id=project.id,
            model_name="Legacy ARIMA Model",
            model_type="ARIMA",
            rmse=15.6,
            mae=12.3,
            mape=0.08,
            r2_score=0.92,
            training_samples=80,
            test_samples=20
        )
        
        # Test existing JSON methods for parameters
        parameters = {
            "p": 2,
            "d": 1, 
            "q": 2,
            "seasonal": False,
            "trend": "add"
        }
        model_result.set_parameters(parameters)
        
        # Test existing JSON methods for forecast data
        forecast_data = {
            "historical": {
                "dates": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "values": [100, 105, 98]
            },
            "test": {
                "dates": ["2023-01-04", "2023-01-05"],
                "actual": [102, 107],
                "predicted": [101, 108]
            },
            "forecast": {
                "dates": ["2023-01-06", "2023-01-07"],
                "values": [110, 112],
                "confidence_lower": [105, 107],
                "confidence_upper": [115, 117]
            }
        }
        model_result.set_forecast_data(forecast_data)
        
        db.session.add(model_result)
        db.session.commit()
        
        # Verify the model result was saved and can be retrieved
        saved_model = ModelResult.query.filter_by(model_name="Legacy ARIMA Model").first()
        self.assertIsNotNone(saved_model)
        self.assertEqual(saved_model.model_type, "ARIMA")
        self.assertEqual(saved_model.rmse, 15.6)
        self.assertEqual(saved_model.mae, 12.3)
        self.assertEqual(saved_model.project_id, project.id)
        
        # Test that existing JSON methods still work
        retrieved_params = saved_model.get_parameters()
        self.assertEqual(retrieved_params["p"], 2)
        self.assertEqual(retrieved_params["d"], 1)
        self.assertEqual(retrieved_params["seasonal"], False)
        
        retrieved_forecast = saved_model.get_forecast_data()
        self.assertEqual(len(retrieved_forecast["historical"]["dates"]), 3)
        self.assertEqual(retrieved_forecast["historical"]["values"][0], 100)
        self.assertEqual(retrieved_forecast["test"]["actual"][1], 107)
        self.assertEqual(retrieved_forecast["forecast"]["values"][0], 110)
        
        print("✓ Existing ModelResult functionality is fully compatible")
    
    def test_existing_relationships_still_work(self):
        """Test that existing Project-ModelResult relationships still work"""
        # Create a project
        project = Project(name="Relationship Test Project")
        db.session.add(project)
        db.session.commit()
        
        # Create multiple model results for the project
        models_data = [
            {"name": "ARIMA Model", "type": "ARIMA", "rmse": 12.5},
            {"name": "Linear Model", "type": "LinearRegression", "rmse": 18.2},
            {"name": "Prophet Model", "type": "Prophet", "rmse": 10.8}
        ]
        
        for model_data in models_data:
            model_result = ModelResult(
                project_id=project.id,
                model_name=model_data["name"],
                model_type=model_data["type"],
                rmse=model_data["rmse"]
            )
            db.session.add(model_result)
        
        db.session.commit()
        
        # Test the relationship from project to models
        saved_project = Project.query.filter_by(name="Relationship Test Project").first()
        self.assertEqual(len(saved_project.models), 3)
        
        # Verify each model can access its project
        for model in saved_project.models:
            self.assertEqual(model.project.name, "Relationship Test Project")
            self.assertIn(model.model_type, ["ARIMA", "LinearRegression", "Prophet"])
        
        # Test that deleting project cascades to models (existing behavior)
        model_count_before = ModelResult.query.count()
        self.assertEqual(model_count_before, 3)
        
        db.session.delete(saved_project)
        db.session.commit()
        
        model_count_after = ModelResult.query.count()
        self.assertEqual(model_count_after, 0)  # All models should be deleted
        
        print("✓ Existing Project-ModelResult relationships work correctly")
    
    def test_new_columns_are_optional(self):
        """Test that new columns in ModelResult don't break existing functionality"""
        # Create a project
        project = Project(name="New Columns Test")
        db.session.add(project)
        db.session.commit()
        
        # Create a model result without using any new columns
        model_result = ModelResult(
            project_id=project.id,
            model_name="Basic Model",
            model_type="ARIMA",
            rmse=14.2,
            mae=11.5
        )
        
        db.session.add(model_result)
        db.session.commit()
        
        # Verify the model was saved successfully
        saved_model = ModelResult.query.filter_by(model_name="Basic Model").first()
        self.assertIsNotNone(saved_model)
        
        # Verify new columns exist but are null/empty
        self.assertIsNone(saved_model.optimization_experiment_id)
        self.assertIsNone(saved_model.ensemble_config)
        self.assertIsNone(saved_model.advanced_metrics)
        self.assertIsNone(saved_model.diagnostic_data)
        
        # Test that new JSON methods return empty defaults
        self.assertEqual(saved_model.get_ensemble_config(), {})
        self.assertEqual(saved_model.get_advanced_metrics(), {})
        self.assertEqual(saved_model.get_diagnostic_data(), {})
        
        print("✓ New columns are optional and don't break existing functionality")
    
    def test_database_schema_integrity(self):
        """Test that the database schema is still consistent"""
        # Verify all expected tables exist using inspector
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        # Original tables should still exist
        self.assertIn('project', tables)
        self.assertIn('model_result', tables)
        
        # New tables should also exist
        expected_new_tables = [
            'data_profile',
            'feature_config', 
            'preprocessing_pipeline',
            'optimization_experiment',
            'external_data_source'
        ]
        
        for table in expected_new_tables:
            self.assertIn(table, tables)
        
        print("✓ Database schema has all expected tables")
    
    def test_json_serialization_backwards_compatible(self):
        """Test that JSON serialization methods are backwards compatible"""
        # Test with data that would have been stored before migration
        legacy_preprocessing_config = {
            "handle_missing": "drop",
            "outlier_detection": "iqr",
            "normalization": "standard"
        }
        
        legacy_model_params = {
            "algorithm": "ARIMA",
            "p": 1,
            "d": 1,
            "q": 1,
            "confidence_interval": 0.95
        }
        
        legacy_forecast_data = {
            "dates": ["2023-01-01", "2023-01-02"],
            "values": [100, 105],
            "predictions": [98, 103]
        }
        
        # Create project and model with legacy data
        project = Project(name="Legacy JSON Test")
        project.set_preprocessing_config(legacy_preprocessing_config)
        db.session.add(project)
        db.session.commit()
        
        model_result = ModelResult(
            project_id=project.id,
            model_name="Legacy JSON Model",
            model_type="ARIMA"
        )
        model_result.set_parameters(legacy_model_params)
        model_result.set_forecast_data(legacy_forecast_data)
        db.session.add(model_result)
        db.session.commit()
        
        # Verify data can be retrieved correctly
        saved_project = Project.query.filter_by(name="Legacy JSON Test").first()
        retrieved_config = saved_project.get_preprocessing_config()
        self.assertEqual(retrieved_config["handle_missing"], "drop")
        
        saved_model = ModelResult.query.filter_by(model_name="Legacy JSON Model").first()
        retrieved_params = saved_model.get_parameters()
        self.assertEqual(retrieved_params["p"], 1)
        
        retrieved_forecast = saved_model.get_forecast_data()
        self.assertEqual(retrieved_forecast["values"][0], 100)
        
        print("✓ JSON serialization is backwards compatible")


if __name__ == '__main__':
    unittest.main(verbosity=2)
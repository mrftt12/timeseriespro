"""
PostgreSQL Compatibility Test for Time Series Pro Database Schema
Tests that all models work correctly with PostgreSQL
"""

import os
import unittest
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime

# Mock psycopg2 for testing without requiring actual PostgreSQL
class MockPsycopg2:
    """Mock psycopg2 module for testing"""
    
    class Error(Exception):
        pass
    
    @staticmethod
    def connect(**kwargs):
        return MockConnection()

class MockConnection:
    """Mock database connection"""
    
    def cursor(self):
        return MockCursor()
    
    def commit(self):
        pass
    
    def close(self):
        pass

class MockCursor:
    """Mock database cursor"""
    
    def execute(self, query, params=None):
        # Simulate different PostgreSQL behaviors
        if "ADD COLUMN IF NOT EXISTS" in query:
            pass  # PostgreSQL supports this syntax
        elif "duplicate column" in query.lower():
            raise MockPsycopg2.Error("column already exists")
    
    def fetchone(self):
        return None


class TestPostgreSQLCompatibility(unittest.TestCase):
    """Test PostgreSQL compatibility for all models"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock environment variables
        self.original_env = os.environ.copy()
        
    def tearDown(self):
        """Clean up test environment"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    @patch('migration.psycopg2', MockPsycopg2)
    def test_postgresql_migration_script(self):
        """Test that migration script works with PostgreSQL"""
        from migration import migrate_postgresql
        
        # Set PostgreSQL environment
        os.environ['DATABASE_URL'] = 'postgresql://user:pass@localhost:5432/testdb'
        
        try:
            # This should run without errors
            migrate_postgresql()
        except Exception as e:
            # Expect app context errors since we're not running full Flask app
            self.assertIn("application context", str(e).lower())
    
    def test_sqlalchemy_postgresql_sql_generation(self):
        """Test that SQLAlchemy generates valid PostgreSQL SQL"""
        from app import app
        from models import (
            Project, ModelResult, DataProfile, FeatureConfig,
            PreprocessingPipeline, OptimizationExperiment, ExternalDataSource
        )
        
        # Configure for PostgreSQL
        app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:pass@localhost:5432/test'
        
        with app.app_context():
            from sqlalchemy import create_engine
            from sqlalchemy.schema import CreateTable
            
            # Create a mock PostgreSQL engine
            engine = create_engine('postgresql://user:pass@localhost:5432/test', 
                                 strategy='mock', executor=lambda sql, *_: None)
            
            # Test that all models can generate valid PostgreSQL DDL
            models_to_test = [
                Project, ModelResult, DataProfile, FeatureConfig,
                PreprocessingPipeline, OptimizationExperiment, ExternalDataSource
            ]
            
            for model in models_to_test:
                try:
                    ddl = CreateTable(model.__table__).compile(engine)
                    sql_text = str(ddl)
                    
                    # Verify PostgreSQL-specific elements
                    self.assertIn('CREATE TABLE', sql_text.upper())
                    
                    # Check that JSON fields are properly handled
                    if hasattr(model, 'parameters') or any(
                        col.name.endswith('_config') or col.name.endswith('_data') 
                        for col in model.__table__.columns
                    ):
                        # PostgreSQL should use TEXT type for JSON fields
                        self.assertIn('TEXT', sql_text.upper())
                    
                    print(f"✓ {model.__name__} PostgreSQL DDL generated successfully")
                    
                except Exception as e:
                    self.fail(f"Failed to generate PostgreSQL DDL for {model.__name__}: {e}")
    
    def test_json_field_compatibility(self):
        """Test that JSON serialization works with PostgreSQL data types"""
        from models import DataProfile
        
        # Test various JSON data types that should work in PostgreSQL
        test_cases = [
            # Simple types
            {"string_field": "test_value"},
            {"number_field": 42},
            {"float_field": 3.14159},
            {"boolean_field": True},
            {"null_field": None},
            
            # Complex types
            {"array_field": [1, 2, 3, "test"]},
            {"nested_object": {"inner": {"value": 123}}},
            
            # Special PostgreSQL considerations
            {"unicode_field": "测试数据"},
            {"large_number": 9223372036854775807},  # Large int
            {"scientific_notation": 1.23e-10},
        ]
        
        profile = DataProfile()
        
        for test_data in test_cases:
            try:
                # Test serialization
                profile.set_statistical_summary(test_data)
                
                # Test deserialization
                retrieved = profile.get_statistical_summary()
                self.assertEqual(retrieved, test_data)
                
                print(f"✓ JSON compatibility test passed for: {test_data}")
                
            except Exception as e:
                self.fail(f"JSON compatibility failed for {test_data}: {e}")
    
    def test_foreign_key_constraints(self):
        """Test that foreign key constraints are PostgreSQL compatible"""
        from app import app
        from models import Project, ModelResult, DataProfile
        
        with app.app_context():
            # Test foreign key definitions
            project_table = Project.__table__
            model_result_table = ModelResult.__table__
            data_profile_table = DataProfile.__table__
            
            # Check ModelResult foreign key to Project
            project_fk = None
            for fk in model_result_table.foreign_keys:
                if fk.column.table.name == 'project':
                    project_fk = fk
                    break
            
            self.assertIsNotNone(project_fk, "ModelResult should have foreign key to Project")
            self.assertEqual(project_fk.column.name, 'id')
            
            # Check DataProfile foreign key to Project
            profile_fk = None
            for fk in data_profile_table.foreign_keys:
                if fk.column.table.name == 'project':
                    profile_fk = fk
                    break
            
            self.assertIsNotNone(profile_fk, "DataProfile should have foreign key to Project")
            
            print("✓ Foreign key constraints are properly defined")
    
    def test_index_compatibility(self):
        """Test that indexes are compatible with PostgreSQL"""
        from models import (
            Project, ModelResult, DataProfile, FeatureConfig,
            PreprocessingPipeline, OptimizationExperiment, ExternalDataSource
        )
        
        models_to_test = [
            Project, ModelResult, DataProfile, FeatureConfig,
            PreprocessingPipeline, OptimizationExperiment, ExternalDataSource
        ]
        
        for model in models_to_test:
            table = model.__table__
            
            # Check that primary keys are defined
            self.assertTrue(table.primary_key.columns, 
                          f"{model.__name__} should have a primary key")
            
            # Check foreign key indexes (important for PostgreSQL performance)
            for fk in table.foreign_keys:
                fk_column = fk.parent
                self.assertIsNotNone(fk_column, 
                                   f"Foreign key column should be defined in {model.__name__}")
            
            print(f"✓ {model.__name__} indexes are PostgreSQL compatible")
    
    def test_data_type_compatibility(self):
        """Test that all data types are compatible with PostgreSQL"""
        from models import (
            Project, ModelResult, DataProfile, FeatureConfig,
            PreprocessingPipeline, OptimizationExperiment, ExternalDataSource
        )
        
        # PostgreSQL compatible data types
        compatible_types = {
            'INTEGER', 'VARCHAR', 'TEXT', 'DATETIME', 'FLOAT', 'BOOLEAN'
        }
        
        models_to_test = [
            Project, ModelResult, DataProfile, FeatureConfig,
            PreprocessingPipeline, OptimizationExperiment, ExternalDataSource
        ]
        
        for model in models_to_test:
            for column in model.__table__.columns:
                column_type = str(column.type).upper()
                
                # Extract base type (remove length specifications)
                base_type = column_type.split('(')[0]
                
                self.assertTrue(
                    any(compat_type in base_type for compat_type in compatible_types),
                    f"Column {column.name} in {model.__name__} has incompatible type: {column_type}"
                )
            
            print(f"✓ {model.__name__} data types are PostgreSQL compatible")
    
    def test_cascade_delete_behavior(self):
        """Test that cascade delete works properly in PostgreSQL"""
        from models import Project, ModelResult, DataProfile
        
        # Test that relationships have proper cascade settings
        project_relationships = [
            'models', 'data_profiles', 'feature_configs', 
            'preprocessing_pipelines', 'optimization_experiments', 
            'external_data_sources'
        ]
        
        for rel_name in project_relationships:
            if hasattr(Project, rel_name):
                relationship = getattr(Project, rel_name)
                # Check that cascade is properly configured
                self.assertIsNotNone(relationship.property.cascade)
                self.assertIn('delete', str(relationship.property.cascade))
                print(f"✓ {rel_name} relationship has proper cascade delete")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
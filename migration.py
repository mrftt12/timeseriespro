#!/usr/bin/env python3
"""
Database Migration Script for Time Series Pro - Phase 1
Adds advanced data science tables while maintaining backward compatibility
"""

import os
import sqlite3
import logging
from datetime import datetime
from app import app, db
from models import (
    Project, ModelResult, DataProfile, FeatureConfig, 
    PreprocessingPipeline, OptimizationExperiment, ExternalDataSource
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_database(db_path):
    """Create a backup of the existing database"""
    backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(db_path):
        import shutil
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    return None

def check_table_exists(cursor, table_name):
    """Check if a table exists in the database"""
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (table_name,))
    return cursor.fetchone() is not None

def add_column_if_not_exists(cursor, table_name, column_definition):
    """Add a column to a table if it doesn't already exist"""
    try:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")
        logger.info(f"Added column to {table_name}: {column_definition}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            logger.info(f"Column already exists in {table_name}: {column_definition}")
        else:
            raise

def migrate_sqlite():
    """Migrate SQLite database"""
    db_path = "instance/forecasting.db"
    
    # Create backup
    backup_path = backup_database(db_path)
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Add new columns to existing ModelResult table
        logger.info("Updating ModelResult table with new columns...")
        add_column_if_not_exists(cursor, "model_result", 
                                "optimization_experiment_id INTEGER REFERENCES optimization_experiment(id)")
        add_column_if_not_exists(cursor, "model_result", "ensemble_config TEXT")
        add_column_if_not_exists(cursor, "model_result", "advanced_metrics TEXT") 
        add_column_if_not_exists(cursor, "model_result", "diagnostic_data TEXT")
        
        # Create new tables if they don't exist
        new_tables = [
            ('data_profile', DataProfile),
            ('feature_config', FeatureConfig), 
            ('preprocessing_pipeline', PreprocessingPipeline),
            ('optimization_experiment', OptimizationExperiment),
            ('external_data_source', ExternalDataSource)
        ]
        
        for table_name, model_class in new_tables:
            if not check_table_exists(cursor, table_name):
                logger.info(f"Creating new table: {table_name}")
                # Will be created by SQLAlchemy below
            else:
                logger.info(f"Table already exists: {table_name}")
        
        conn.commit()
        conn.close()
        
        # Use SQLAlchemy to create any missing tables with proper constraints
        with app.app_context():
            db.create_all()
            logger.info("SQLAlchemy tables created/updated successfully")
        
        logger.info("SQLite migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if backup_path and os.path.exists(backup_path):
            logger.info(f"Restore from backup: {backup_path}")
        raise

def migrate_postgresql():
    """Migrate PostgreSQL database"""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url or not database_url.startswith('postgresql'):
        logger.info("No PostgreSQL database URL found, skipping PostgreSQL migration")
        return
    
    try:
        import psycopg2
        from urllib.parse import urlparse
        
        # Parse the database URL
        url = urlparse(database_url)
        conn = psycopg2.connect(
            host=url.hostname,
            port=url.port,
            user=url.username,
            password=url.password,
            database=url.path[1:]  # Remove leading slash
        )
        cursor = conn.cursor()
        
        # Add new columns to existing ModelResult table
        logger.info("Updating ModelResult table with new columns...")
        try:
            cursor.execute("""
                ALTER TABLE model_result 
                ADD COLUMN IF NOT EXISTS optimization_experiment_id INTEGER 
                REFERENCES optimization_experiment(id)
            """)
            cursor.execute("ALTER TABLE model_result ADD COLUMN IF NOT EXISTS ensemble_config TEXT")
            cursor.execute("ALTER TABLE model_result ADD COLUMN IF NOT EXISTS advanced_metrics TEXT")
            cursor.execute("ALTER TABLE model_result ADD COLUMN IF NOT EXISTS diagnostic_data TEXT")
        except psycopg2.Error as e:
            logger.info(f"Columns may already exist: {e}")
        
        conn.commit()
        conn.close()
        
        # Use SQLAlchemy to create any missing tables
        with app.app_context():
            db.create_all()
            logger.info("PostgreSQL tables created/updated successfully")
        
        logger.info("PostgreSQL migration completed successfully!")
        
    except ImportError:
        logger.warning("psycopg2 not installed, skipping PostgreSQL migration")
    except Exception as e:
        logger.error(f"PostgreSQL migration failed: {e}")
        raise

def verify_migration():
    """Verify that the migration was successful"""
    with app.app_context():
        try:
            # Test that all models can be imported and have their tables
            models_to_test = [Project, ModelResult, DataProfile, FeatureConfig, 
                            PreprocessingPipeline, OptimizationExperiment, ExternalDataSource]
            
            for model in models_to_test:
                # Try to query the table (will fail if table doesn't exist)
                db.session.query(model).first()
                logger.info(f"✓ {model.__name__} table verified")
            
            # Test relationships
            logger.info("Testing relationships...")
            
            # Create a test project
            test_project = Project(name="Migration Test", description="Test project for migration verification")
            db.session.add(test_project)
            db.session.flush()  # Get the ID without committing
            
            # Test each relationship
            test_profile = DataProfile(project_id=test_project.id, column_name="test_column")
            test_feature = FeatureConfig(project_id=test_project.id, feature_type="test_feature")
            test_pipeline = PreprocessingPipeline(project_id=test_project.id, pipeline_name="test_pipeline")
            test_experiment = OptimizationExperiment(project_id=test_project.id, algorithm_type="test_algo")
            test_source = ExternalDataSource(project_id=test_project.id, source_type="test_source")
            
            db.session.add_all([test_profile, test_feature, test_pipeline, test_experiment, test_source])
            db.session.flush()
            
            # Test JSON serialization methods
            test_profile.set_statistical_summary({"mean": 42.0})
            test_feature.set_configuration({"param1": "value1"})
            test_pipeline.set_steps([{"step": "normalize"}])
            test_experiment.set_search_space({"param": [1, 2, 3]})
            test_source.set_api_configuration({"api_key": "test"})
            
            # Verify JSON retrieval
            assert test_profile.get_statistical_summary()["mean"] == 42.0
            assert test_feature.get_configuration()["param1"] == "value1"
            assert test_pipeline.get_steps()[0]["step"] == "normalize"
            assert test_experiment.get_search_space()["param"] == [1, 2, 3]
            assert test_source.get_api_configuration()["api_key"] == "test"
            
            # Clean up test data
            db.session.rollback()
            
            logger.info("✓ All relationships and JSON serialization working correctly")
            logger.info("🎉 Migration verification completed successfully!")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Migration verification failed: {e}")
            raise

def main():
    """Run the complete migration process"""
    logger.info("Starting database migration for Time Series Pro Phase 1...")
    
    try:
        # Migrate SQLite (development database)
        migrate_sqlite()
        
        # Migrate PostgreSQL (production database)
        migrate_postgresql()
        
        # Verify the migration
        verify_migration()
        
        logger.info("✅ Database migration completed successfully!")
        logger.info("All Phase 1 tables and relationships are now available.")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    main()
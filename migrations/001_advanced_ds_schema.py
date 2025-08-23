"""
Database Migration: Advanced Data Science Features Schema
Migration ID: 001
Created: 2025-08-22
Epic: #2 Advanced Data Science Features

This migration adds the foundational database schema to support advanced data science features
including data profiling, feature engineering, hyperparameter optimization, and external data integration.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDSMigration:
    def __init__(self, database_url=None):
        self.database_url = database_url or os.environ.get("DATABASE_URL", "sqlite:///instance/forecasting.db")
        self.engine = create_engine(self.database_url)
    
    def up(self):
        """Apply migration - add new tables for advanced DS features"""
        try:
            with self.engine.connect() as conn:
                # Data profiling table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS data_profile (
                        id INTEGER PRIMARY KEY,
                        project_id INTEGER NOT NULL,
                        column_name VARCHAR(100) NOT NULL,
                        data_type VARCHAR(50),
                        missing_count INTEGER DEFAULT 0,
                        missing_percentage FLOAT DEFAULT 0.0,
                        outlier_count INTEGER DEFAULT 0,
                        statistical_summary TEXT,
                        quality_score FLOAT,
                        recommendations TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
                    )
                """))
                
                # Feature engineering configuration
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS feature_config (
                        id INTEGER PRIMARY KEY,
                        project_id INTEGER NOT NULL,
                        feature_type VARCHAR(50) NOT NULL,
                        configuration TEXT NOT NULL,
                        is_enabled BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
                    )
                """))
                
                # Preprocessing pipelines
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS preprocessing_pipeline (
                        id INTEGER PRIMARY KEY,
                        project_id INTEGER NOT NULL,
                        pipeline_name VARCHAR(100) NOT NULL,
                        steps TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
                    )
                """))
                
                # Hyperparameter optimization experiments
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS optimization_experiment (
                        id INTEGER PRIMARY KEY,
                        project_id INTEGER NOT NULL,
                        algorithm_type VARCHAR(50) NOT NULL,
                        search_space TEXT,
                        best_parameters TEXT,
                        best_value FLOAT,
                        trials_data TEXT,
                        status VARCHAR(20) DEFAULT 'pending',
                        n_trials INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
                    )
                """))
                
                # External data sources
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS external_data_source (
                        id INTEGER PRIMARY KEY,
                        project_id INTEGER NOT NULL,
                        source_type VARCHAR(50) NOT NULL,
                        api_configuration TEXT,
                        data_mapping TEXT,
                        last_sync TIMESTAMP,
                        sync_frequency VARCHAR(20) DEFAULT 'daily',
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
                    )
                """))
                
                # Ensemble configurations (Phase 2 preparation)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ensemble_config (
                        id INTEGER PRIMARY KEY,
                        project_id INTEGER NOT NULL,
                        ensemble_type VARCHAR(20) NOT NULL,
                        base_models TEXT,
                        meta_config TEXT,
                        performance_metrics TEXT,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (project_id) REFERENCES project(id) ON DELETE CASCADE
                    )
                """))
                
                # Extend model_result table with advanced fields
                try:
                    conn.execute(text("ALTER TABLE model_result ADD COLUMN optimization_experiment_id INTEGER"))
                except SQLAlchemyError:
                    logger.info("Column optimization_experiment_id already exists")
                
                try:
                    conn.execute(text("ALTER TABLE model_result ADD COLUMN ensemble_config TEXT"))
                except SQLAlchemyError:
                    logger.info("Column ensemble_config already exists")
                
                try:
                    conn.execute(text("ALTER TABLE model_result ADD COLUMN advanced_metrics TEXT"))
                except SQLAlchemyError:
                    logger.info("Column advanced_metrics already exists")
                
                try:
                    conn.execute(text("ALTER TABLE model_result ADD COLUMN diagnostic_data TEXT"))
                except SQLAlchemyError:
                    logger.info("Column diagnostic_data already exists")
                
                # Create indexes for performance
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_data_profile_project ON data_profile(project_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_feature_config_project ON feature_config(project_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_optimization_project ON optimization_experiment(project_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_external_data_project ON external_data_source(project_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ensemble_config_project ON ensemble_config(project_id)"))
                
                conn.commit()
                logger.info("✅ Advanced DS schema migration applied successfully")
                
        except Exception as e:
            logger.error(f"❌ Migration failed: {str(e)}")
            raise
    
    def down(self):
        """Rollback migration - remove advanced DS tables"""
        try:
            with self.engine.connect() as conn:
                # Remove added columns from model_result
                # Note: SQLite doesn't support DROP COLUMN, so this would require table recreation
                logger.warning("Rollback not fully implemented for SQLite - added columns will remain")
                
                # Drop tables in reverse order
                conn.execute(text("DROP TABLE IF EXISTS ensemble_config"))
                conn.execute(text("DROP TABLE IF EXISTS external_data_source"))
                conn.execute(text("DROP TABLE IF EXISTS optimization_experiment"))
                conn.execute(text("DROP TABLE IF EXISTS preprocessing_pipeline"))
                conn.execute(text("DROP TABLE IF EXISTS feature_config"))
                conn.execute(text("DROP TABLE IF EXISTS data_profile"))
                
                conn.commit()
                logger.info("✅ Advanced DS schema rollback completed")
                
        except Exception as e:
            logger.error(f"❌ Rollback failed: {str(e)}")
            raise

def run_migration():
    """Run the migration"""
    migration = AdvancedDSMigration()
    migration.up()

if __name__ == "__main__":
    run_migration()
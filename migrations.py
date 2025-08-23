"""
Database migrations for Advanced Data Science Features
Part of Epic #2: Time Series Pro Advanced Data Science Features
"""

from app import app, db
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

def run_migrations():
    """Run all pending database migrations"""
    with app.app_context():
        try:
            # Check if migration tracking table exists
            _create_migration_table()
            
            # Run all migrations
            migrations = [
                ('001_add_advanced_model_columns', _migration_001_add_advanced_model_columns),
                ('002_create_data_profile_table', _migration_002_create_data_profile_table),
                ('003_create_feature_config_table', _migration_003_create_feature_config_table),
                ('004_create_preprocessing_pipeline_table', _migration_004_create_preprocessing_pipeline_table),
                ('005_create_optimization_experiment_table', _migration_005_create_optimization_experiment_table),
                ('006_create_external_data_source_table', _migration_006_create_external_data_source_table),
            ]
            
            for migration_name, migration_func in migrations:
                if not _is_migration_applied(migration_name):
                    logger.info(f"Running migration: {migration_name}")
                    migration_func()
                    _mark_migration_applied(migration_name)
                    logger.info(f"Migration {migration_name} completed successfully")
                else:
                    logger.debug(f"Migration {migration_name} already applied, skipping")
            
            db.session.commit()
            logger.info("All migrations completed successfully")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Migration failed: {str(e)}")
            raise

def _create_migration_table():
    """Create table to track applied migrations"""
    try:
        db.session.execute(text("""
            CREATE TABLE IF NOT EXISTS migration_history (
                id INTEGER PRIMARY KEY,
                migration_name VARCHAR(255) UNIQUE NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to create migration table: {str(e)}")
        raise

def _is_migration_applied(migration_name: str) -> bool:
    """Check if a migration has already been applied"""
    try:
        result = db.session.execute(
            text("SELECT COUNT(*) FROM migration_history WHERE migration_name = :name"),
            {"name": migration_name}
        ).fetchone()
        return result[0] > 0
    except Exception:
        return False

def _mark_migration_applied(migration_name: str):
    """Mark a migration as applied"""
    try:
        db.session.execute(
            text("INSERT INTO migration_history (migration_name) VALUES (:name)"),
            {"name": migration_name}
        )
    except Exception as e:
        logger.error(f"Failed to mark migration as applied: {str(e)}")
        raise

# Individual Migration Functions

def _migration_001_add_advanced_model_columns():
    """Add advanced columns to model_result table"""
    try:
        # Check if columns already exist before adding them
        columns_to_add = [
            ("optimization_experiment_id", "INTEGER"),
            ("ensemble_config", "TEXT"),
            ("advanced_metrics", "TEXT"),
            ("diagnostic_data", "TEXT")
        ]
        
        for column_name, column_type in columns_to_add:
            try:
                db.session.execute(text(f"""
                    ALTER TABLE model_result ADD COLUMN {column_name} {column_type}
                """))
            except Exception as e:
                if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                    logger.debug(f"Column {column_name} already exists, skipping")
                else:
                    raise
        
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Migration 001 failed: {str(e)}")
        raise

def _migration_002_create_data_profile_table():
    """Create data_profile table"""
    try:
        db.session.execute(text("""
            CREATE TABLE IF NOT EXISTS data_profile (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                column_name VARCHAR(100) NOT NULL,
                data_type VARCHAR(50),
                missing_count INTEGER,
                missing_percentage REAL,
                outlier_count INTEGER,
                statistical_summary TEXT,
                quality_score REAL,
                recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES project (id)
            )
        """))
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Migration 002 failed: {str(e)}")
        raise

def _migration_003_create_feature_config_table():
    """Create feature_config table"""
    try:
        db.session.execute(text("""
            CREATE TABLE IF NOT EXISTS feature_config (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                feature_type VARCHAR(50) NOT NULL,
                configuration TEXT,
                is_enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES project (id)
            )
        """))
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Migration 003 failed: {str(e)}")
        raise

def _migration_004_create_preprocessing_pipeline_table():
    """Create preprocessing_pipeline table"""
    try:
        db.session.execute(text("""
            CREATE TABLE IF NOT EXISTS preprocessing_pipeline (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                pipeline_name VARCHAR(100) NOT NULL,
                steps TEXT,
                is_active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES project (id)
            )
        """))
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Migration 004 failed: {str(e)}")
        raise

def _migration_005_create_optimization_experiment_table():
    """Create optimization_experiment table"""
    try:
        db.session.execute(text("""
            CREATE TABLE IF NOT EXISTS optimization_experiment (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                algorithm_type VARCHAR(50) NOT NULL,
                search_space TEXT,
                best_parameters TEXT,
                trials_data TEXT,
                status VARCHAR(20) DEFAULT 'pending',
                n_trials INTEGER DEFAULT 100,
                best_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES project (id)
            )
        """))
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Migration 005 failed: {str(e)}")
        raise

def _migration_006_create_external_data_source_table():
    """Create external_data_source table"""
    try:
        db.session.execute(text("""
            CREATE TABLE IF NOT EXISTS external_data_source (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL,
                source_type VARCHAR(50) NOT NULL,
                api_configuration TEXT,
                data_mapping TEXT,
                last_sync TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES project (id)
            )
        """))
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Migration 006 failed: {str(e)}")
        raise

def check_database_health():
    """Check database health and report status"""
    with app.app_context():
        try:
            # Check if all required tables exist
            required_tables = [
                'project',
                'model_result',
                'data_profile',
                'feature_config',
                'preprocessing_pipeline',
                'optimization_experiment',
                'external_data_source',
                'migration_history'
            ]
            
            health_status = {
                'database_accessible': True,
                'tables_exist': {},
                'total_projects': 0,
                'total_models': 0,
                'migration_status': 'unknown'
            }
            
            # Check table existence
            for table_name in required_tables:
                try:
                    result = db.session.execute(
                        text(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")
                    ).fetchone()
                    health_status['tables_exist'][table_name] = True
                except Exception:
                    health_status['tables_exist'][table_name] = False
            
            # Get basic statistics
            try:
                result = db.session.execute(text("SELECT COUNT(*) FROM project")).fetchone()
                health_status['total_projects'] = result[0]
                
                result = db.session.execute(text("SELECT COUNT(*) FROM model_result")).fetchone()
                health_status['total_models'] = result[0]
            except Exception:
                pass
            
            # Check migration status
            try:
                result = db.session.execute(
                    text("SELECT COUNT(*) FROM migration_history")
                ).fetchone()
                health_status['migration_status'] = f"{result[0]} migrations applied"
            except Exception:
                health_status['migration_status'] = 'migration table not found'
            
            all_tables_exist = all(health_status['tables_exist'].values())
            health_status['overall_status'] = 'healthy' if all_tables_exist else 'needs_migration'
            
            return health_status
            
        except Exception as e:
            return {
                'database_accessible': False,
                'error': str(e),
                'overall_status': 'error'
            }

def reset_database():
    """Reset database - WARNING: This will delete all data!"""
    with app.app_context():
        try:
            logger.warning("RESETTING DATABASE - ALL DATA WILL BE LOST")
            
            # Drop all tables
            tables_to_drop = [
                'migration_history',
                'external_data_source',
                'optimization_experiment',
                'preprocessing_pipeline',
                'feature_config',
                'data_profile',
                'model_result',
                'project'
            ]
            
            for table_name in tables_to_drop:
                try:
                    db.session.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                except Exception as e:
                    logger.warning(f"Could not drop table {table_name}: {str(e)}")
            
            db.session.commit()
            
            # Recreate all tables
            db.create_all()
            
            # Run migrations
            run_migrations()
            
            logger.info("Database reset completed successfully")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Database reset failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Run migrations when script is executed directly
    run_migrations()
    
    # Print database health status
    status = check_database_health()
    print(f"Database Status: {status['overall_status']}")
    print(f"Projects: {status['total_projects']}")
    print(f"Models: {status['total_models']}")
    print(f"Migration Status: {status['migration_status']}")
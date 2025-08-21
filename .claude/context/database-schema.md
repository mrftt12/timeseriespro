# Database Schema & Models

## Database Configuration

### Connection Details
- **Development**: SQLite (`instance/forecasting.db`)
- **Production**: PostgreSQL via `DATABASE_URL` environment variable
- **ORM**: SQLAlchemy 2.0.41 with DeclarativeBase
- **Auto-creation**: Tables created on application startup

### Connection Pool Settings
```python
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,      # Recycle connections every 5 minutes
    "pool_pre_ping": True,    # Validate connections before use
}
```

## Table Schema

### Projects Table

**Purpose**: Store project metadata and dataset information

```sql
CREATE TABLE project (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now')),
    
    -- Dataset Information
    dataset_filename VARCHAR(255),
    dataset_path VARCHAR(500),
    date_column VARCHAR(100),
    target_column VARCHAR(100),
    
    -- Preprocessing Configuration
    preprocessing_config TEXT  -- JSON string
);
```

**Key Relationships**:
- One-to-many with `ModelResult` (cascade delete)

**JSON Fields**:
- `preprocessing_config`: Stores data preprocessing settings as JSON

### ModelResult Table

**Purpose**: Store trained model results, parameters, and forecasts

```sql
CREATE TABLE model_result (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES project(id),
    
    -- Model Identification
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- ARIMA, LinearRegression, etc.
    
    -- Model Configuration
    parameters TEXT,  -- JSON string
    
    -- Performance Metrics
    rmse FLOAT,
    mae FLOAT,
    mape FLOAT,
    r2_score FLOAT,
    
    -- Training Statistics
    training_samples INTEGER,
    test_samples INTEGER,
    
    -- Forecast Results
    forecast_data TEXT,  -- JSON string
    
    created_at DATETIME DEFAULT (datetime('now'))
);
```

**Key Relationships**:
- Many-to-one with `Project` via `project_id`

**JSON Fields**:
- `parameters`: Model-specific configuration (ARIMA order, RF n_estimators, etc.)
- `forecast_data`: Complete forecast results for visualization

## Model Classes

### Project Model (`models.py:5-31`)

```python
class Project(db.Model):
    # Primary identification
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    
    # Timestamps with auto-update
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Dataset metadata
    dataset_filename = db.Column(db.String(255))
    dataset_path = db.Column(db.String(500))
    date_column = db.Column(db.String(100))
    target_column = db.Column(db.String(100))
    
    # JSON configuration
    preprocessing_config = db.Column(db.Text)
    
    # Relationships
    models = db.relationship('ModelResult', backref='project', lazy=True, cascade='all, delete-orphan')
```

**Helper Methods**:
- `get_preprocessing_config()`: Parse JSON config
- `set_preprocessing_config(config)`: Serialize config to JSON

### ModelResult Model (`models.py:32-72`)

```python
class ModelResult(db.Model):
    # Primary identification
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    
    # Model identification
    model_name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    
    # Configuration and results
    parameters = db.Column(db.Text)  # JSON
    forecast_data = db.Column(db.Text)  # JSON
    
    # Performance metrics
    rmse = db.Column(db.Float)
    mae = db.Column(db.Float)
    mape = db.Column(db.Float)
    r2_score = db.Column(db.Float)
    
    # Training metadata
    training_samples = db.Column(db.Integer)
    test_samples = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

**Helper Methods**:
- `get_parameters()`: Parse JSON parameters
- `set_parameters(params)`: Serialize parameters to JSON
- `get_forecast_data()`: Parse JSON forecast data
- `set_forecast_data(data)`: Serialize forecast data to JSON

## JSON Data Structures

### Preprocessing Config
```python
{
    "missing_value_strategy": "forward_fill",
    "outlier_detection": True,
    "outlier_threshold": 3.0,
    "seasonality_detection": "auto",
    "data_quality_checks": True
}
```

### Model Parameters Examples

**ARIMA**:
```python
{
    "order": [1, 1, 1],
    "seasonal_order": [1, 1, 1, 12],
    "trend": "c"
}
```

**Random Forest**:
```python
{
    "n_estimators": 100,
    "max_depth": 10,
    "n_features": 8,
    "random_state": 42
}
```

### Forecast Data Structure
```python
{
    "historical": {
        "dates": ["2023-01-01", "2023-01-02", ...],
        "values": [100.5, 102.3, ...]
    },
    "test": {
        "dates": ["2023-12-01", "2023-12-02", ...],
        "actual": [95.2, 97.1, ...],
        "predicted": [94.8, 96.9, ...]
    },
    "forecast": {
        "dates": ["2024-01-01", "2024-01-02", ...],
        "values": [98.5, 99.2, ...],
        "confidence_lower": [93.1, 94.0, ...],
        "confidence_upper": [103.9, 104.4, ...]
    }
}
```

## Query Patterns

### Common Queries

**Get Project with All Models**:
```python
project = Project.query.get(project_id)
models = project.models  # Relationship loaded automatically
```

**Recent Projects Dashboard**:
```python
projects = Project.query.order_by(Project.updated_at.desc()).all()
```

**Model Performance Comparison**:
```python
models = ModelResult.query.filter_by(project_id=project_id).order_by(ModelResult.rmse.asc()).all()
```

**Recent Activity**:
```python
recent_models = (ModelResult.query
                .join(Project)
                .order_by(ModelResult.created_at.desc())
                .limit(5)
                .all())
```

## Data Lifecycle

### Project Creation
1. User uploads dataset → File saved to `uploads/`
2. Project record created with dataset metadata
3. `preprocessing_config` initialized as empty JSON

### Model Training
1. ForecastingEngine processes data
2. ModelResult created with:
   - Performance metrics (RMSE, MAE, MAPE, R²)
   - Model parameters (JSON)
   - Complete forecast data (JSON)
3. Project `updated_at` timestamp refreshed

### Data Cleanup
- Cascade delete: Deleting project removes all associated ModelResults
- File cleanup: Manual (uploaded files not auto-deleted)
- No soft delete implemented

## Performance Considerations

### Indexing Strategy
- Primary keys: Auto-indexed
- Foreign keys: Indexed by SQLAlchemy
- No additional indexes defined (suitable for single-user application)

### JSON Storage
- Pros: Flexible schema, easy serialization
- Cons: No SQL querying of JSON content, larger storage footprint
- Alternative: Consider separate tables for structured parameters

### Query Optimization
- Relationship loading: Lazy loading by default
- Join queries: Used for dashboard aggregations
- No pagination: Suitable for single-user scenario

## Migration Strategy

### Schema Evolution
- Development: Automatic table creation
- Production: Manual migration scripts needed
- JSON backward compatibility: Handle missing keys gracefully

### Data Export/Import
- Projects: Standard SQLAlchemy model serialization
- Files: Manual handling of `uploads/` directory
- Database backup: SQLite file copy or PostgreSQL dump
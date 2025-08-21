# Time Series Pro - Application Architecture

## High-Level Architecture

Time Series Pro follows a traditional Model-View-Controller (MVC) pattern implemented with Flask:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Frontend       │    │   Flask Routes   │    │   Business Logic    │
│   (Templates +   │◄──►│   (routes.py)    │◄──►│   (ForecastingEngine│
│    Static Assets)│    │                  │    │    DataProcessor)   │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────────┐
                       │   Database       │    │   File System       │
                       │   (SQLite)       │    │   (uploads/)        │
                       │   - Projects     │    │   - CSV/Excel files │
                       │   - ModelResults │    │   - User datasets   │
                       └──────────────────┘    └─────────────────────┘
```

## Request Flow

### 1. Data Upload & Project Creation
```
User Upload → routes.py:upload_dataset() → DataProcessor.load_data() → Database (Project) → Project Detail View
```

### 2. Model Training
```
Project Page → routes.py:train_model() → ForecastingEngine.train_*() → Database (ModelResult) → Results Display
```

### 3. Model Comparison
```
Compare Page → routes.py:compare_models() → Database Query → Chart.js Visualization
```

## Core Components

### 1. Web Layer (`app.py`, `routes.py`, `main.py`)
- **app.py**: Flask application initialization, database setup, configuration
- **routes.py**: HTTP route handlers, request/response processing, template rendering
- **main.py**: Development server entry point

### 2. Business Logic Layer
- **ForecastingEngine**: Central orchestrator for all ML models
- **DataProcessor**: Data validation, preprocessing, quality assessment
- **Models**: SQLAlchemy ORM models for data persistence

### 3. Data Layer
- **SQLite Database**: Project metadata, model results, configurations
- **File System**: User-uploaded datasets, temporary files

### 4. Frontend Layer
- **Jinja2 Templates**: Server-side HTML rendering
- **Chart.js**: Interactive time series visualizations
- **Bootstrap**: UI styling and responsive design

## Data Flow Architecture

### Project Workflow
1. **Upload**: User uploads CSV/Excel → DataProcessor validates → Project created
2. **Configure**: User selects date/target columns → Data preprocessing
3. **Train**: User selects algorithm → ForecastingEngine trains model → Results stored
4. **Compare**: User views multiple models → Database aggregation → Chart visualization

### Model Training Pipeline
```
Raw Data → DataProcessor → 80/20 Split → Algorithm Training → Metrics Calculation → Results Storage
```

Each model follows standardized pipeline:
- Data validation and preprocessing
- Train/test split (80/20)
- Model-specific training
- Standardized metrics (RMSE, MAE, MAPE, R²)
- Future forecast generation
- JSON serialization for storage

## Key Design Decisions

### 1. Modular Forecasting Architecture
Each algorithm is implemented as a separate method in `ForecastingEngine`, enabling:
- Easy addition of new models
- Consistent interface across algorithms
- Independent algorithm evolution

### 2. Graceful Dependency Management
Optional ML libraries (Prophet, LightGBM, XGBoost, PyTorch) use availability flags:
```python
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
```

### 3. JSON-Based Configuration Storage
Complex model parameters and results stored as JSON in database:
- Flexible parameter storage
- Easy serialization/deserialization
- Forward compatibility with new model types

### 4. Project-Based Organization
Users organize work by projects containing:
- Dataset metadata
- Multiple model results
- Comparison capabilities
- Historical tracking

## Database Design

### Projects Table
- Metadata: name, description, timestamps
- Dataset info: filename, path, column mappings
- Configuration: preprocessing settings (JSON)

### ModelResults Table
- Model metadata: name, type, parameters (JSON)
- Performance metrics: RMSE, MAE, MAPE, R²
- Training info: sample counts, timestamps
- Forecast data: complete results (JSON)

## Security Considerations

- File upload validation (allowed extensions, secure filenames)
- SQL injection prevention via SQLAlchemy ORM
- No user authentication (single-user application)
- Local file storage only (no cloud integration)

## Scalability Characteristics

- **Current**: Single-user, local SQLite database
- **File Storage**: Local uploads directory
- **Memory**: In-memory model training (no persistence)
- **Concurrency**: Single-threaded Flask development server

## Extension Points

- **New Algorithms**: Add to `ForecastingEngine` with consistent interface
- **Data Sources**: Extend `DataProcessor` for new file types
- **Visualizations**: Add Chart.js configurations
- **Metrics**: Extend standardized metrics calculation
- **Export**: Add model/forecast export capabilities
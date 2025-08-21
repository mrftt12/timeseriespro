# Technology Stack & Dependencies

## Core Framework
- **Flask 3.1.1**: Web framework
- **SQLAlchemy 2.0.41**: Database ORM
- **Werkzeug 3.1.3**: WSGI utilities

## Machine Learning Libraries

### Core ML Stack
- **scikit-learn 1.6.1**: Basic ML algorithms, preprocessing, metrics
- **pandas 2.2.3**: Data manipulation and analysis
- **numpy 2.2.6**: Numerical computing

### Time Series Specific
- **statsmodels 0.14.4**: Statistical models (ARIMA, SARIMAX)
- **prophet 1.1.6**: Facebook's forecasting library (optional)

### Advanced ML Models
- **lightgbm 4.6.0**: Gradient boosting framework (optional)
- **xgboost 2.0.0**: Extreme gradient boosting (optional)
- **torch 2.0.0**: PyTorch for deep learning (optional)
- **pytorch-lightning 2.0.0**: High-level PyTorch wrapper (optional)
- **pytorch-forecasting 1.0.0**: Time series models (optional)

## Data Processing
- **openpyxl 3.1.5**: Excel file handling
- **holidays 0.74**: Holiday detection for seasonality

## Production & Deployment
- **gunicorn 23.0.0**: WSGI HTTP Server
- **psycopg2-binary 2.9.10**: PostgreSQL adapter (for production DB)

## Validation & Security
- **email-validator 2.2.0**: Email validation utilities

## Dependency Management

### Optional Dependencies Strategy
The application uses graceful degradation for optional ML libraries:

```python
# Example pattern used throughout
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    lgb = None
    LIGHTGBM_AVAILABLE = False
```

### Available Algorithms by Dependency
- **Always Available**: Linear Regression, Moving Average, Random Forest, ARIMA
- **Prophet Required**: Prophet forecasting
- **LightGBM Required**: LightGBM models
- **XGBoost Required**: XGBoost models
- **SARIMAX Required**: SARIMAX (seasonal ARIMA)
- **PyTorch Stack Required**: LSTM, NHITS (deep learning models)

## Frontend Technologies

### JavaScript Libraries
- **Chart.js**: Time series visualization
- **Bootstrap**: UI framework
- **jQuery**: DOM manipulation (implied from templates)

### Template Engine
- **Jinja2**: Server-side HTML templating

## Database Configuration

### Development
- **SQLite**: `instance/forecasting.db`
- **Auto-migration**: Tables created on startup

### Production Ready
- **PostgreSQL**: Via psycopg2-binary
- **Connection Pooling**: Built-in SQLAlchemy configuration

## Build & Package Management

### Package Definition
- **pyproject.toml**: Modern Python packaging
- **uv.lock**: Dependency lock file (uv package manager)

### Python Version
- **Requires**: Python >=3.11

## Development Tools

### Server Configuration
- **Development**: `python main.py` (Flask dev server)
- **Production**: `gunicorn -c gunicorn.conf.py main:app`

### File Handling
- **Upload Directory**: `uploads/`
- **Max File Size**: 16MB
- **Supported Formats**: CSV, Excel (.xlsx, .xls)

## Key Configuration

### Flask Settings
```python
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///forecasting.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### Security
- **Secret Key**: Environment variable or dev default
- **File Upload**: Secure filename handling
- **WSGI Proxy**: X-Forwarded headers support

## Logging & Monitoring
- **Python Logging**: DEBUG level configured
- **Prophet Logging**: Suppressed to reduce noise
- **Error Handling**: Comprehensive try/catch blocks

## Future Considerations

### Scalability Dependencies
- **Redis**: For caching and session management
- **Celery**: For background task processing
- **Docker**: For containerized deployment

### Additional ML Libraries
- **TensorFlow**: Alternative deep learning framework
- **Plotly**: Interactive visualization alternative
- **MLflow**: Model lifecycle management
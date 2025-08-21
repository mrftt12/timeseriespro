# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Time Series Pro is a Flask-based web application for time series forecasting. It provides multiple ML algorithms including ARIMA, Prophet, LightGBM, XGBoost, LSTM, and traditional statistical methods. Users can upload datasets, train models, compare results, and visualize forecasts through a web interface.

## Development Commands

### Running the Application
```bash
# Development server
python main.py

# Production server with Gunicorn
gunicorn -c gunicorn.conf.py main:app
```

### Dependency Management
```bash
# Install dependencies using uv (preferred)
uv sync

# Or using pip with pyproject.toml
pip install -e .
```

### Database Operations
The application uses SQLite by default with automatic table creation on startup. Database files are stored in `instance/forecasting.db`.

## Architecture Overview

### Core Components

**ForecastingEngine** (`forecasting.py`)
- Central orchestrator for all forecasting models
- Handles data preprocessing, model training, and prediction generation  
- Supports 9 different algorithms: ARIMA, Linear Regression, Moving Average, Random Forest, Prophet, LightGBM, XGBoost, SARIMAX, LSTM, NHITS
- Graceful fallbacks when optional dependencies are unavailable
- Consistent interface with `train_*()` methods returning standardized results

**DataProcessor** (`data_processor.py`)  
- Time series data validation and preprocessing
- Handles CSV/Excel file loading with error handling
- Automatic data quality assessment and recommendations
- Missing value imputation and outlier detection

**Database Models** (`models.py`)
- `Project`: Stores dataset metadata and preprocessing configurations
- `ModelResult`: Stores trained model parameters, metrics, and forecast data
- JSON serialization for complex configuration storage

**Web Interface** (`routes.py`, `app.py`)
- Project-based workflow for organizing forecasting experiments
- Model comparison dashboard with interactive charts
- File upload handling with security validation

### Key Architecture Decisions

**Modular Model Implementation**: Each forecasting algorithm is implemented as a separate method in `ForecastingEngine`, allowing easy addition of new models while maintaining consistent interfaces.

**Graceful Dependency Handling**: Optional ML libraries (Prophet, LightGBM, XGBoost, PyTorch) are imported with try/catch blocks and availability flags, preventing crashes when dependencies are missing.

**Standardized Data Flow**: All models follow the same pattern:
1. Data preprocessing via `_prepare_data()` 
2. 80/20 train/test split
3. Model training and validation
4. Future forecast generation
5. Standardized metrics calculation (RMSE, MAE, MAPE, R²)

**JSON-based Storage**: Model parameters and forecast results are serialized to JSON in the database, providing flexibility for varying model configurations.

**Chart.js Integration**: Frontend uses Chart.js for interactive time series visualizations with historical data, test predictions, and future forecasts with confidence intervals.

## Working with Models

### Adding New Forecasting Algorithms
1. Add dependency to `pyproject.toml` with optional import in `forecasting.py`
2. Create `train_[model_name]()` method following existing patterns
3. Return standardized result dictionary with metrics and forecast_data
4. Add model option to frontend forms and routing

### Data Requirements
- CSV or Excel files with date and numeric target columns
- Minimum 30 data points recommended for reliable forecasting
- Automatic handling of missing values and data type conversion

### Model Results Structure
All models return consistent format:
```python
{
    'metrics': {'rmse': float, 'mae': float, 'mape': float, 'r2_score': float},
    'parameters': {...},  # Model-specific parameters
    'training_samples': int,
    'test_samples': int, 
    'forecast_data': {
        'historical': {'dates': [], 'values': []},
        'test': {'dates': [], 'actual': [], 'predicted': []},
        'forecast': {'dates': [], 'values': [], 'confidence_lower': [], 'confidence_upper': []}
    }
}
```

## File Organization

- **Core Logic**: `forecasting.py`, `data_processor.py`, `models.py`
- **Web Layer**: `app.py`, `routes.py`, `main.py`
- **Frontend**: `templates/`, `static/css/`, `static/js/`
- **Data Storage**: `uploads/` (user files), `instance/` (database)
- **Configuration**: `pyproject.toml`, `gunicorn.conf.py`
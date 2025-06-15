# TimeSeries Forecasting Platform

## Overview

A Flask-based web application for time series data analysis and forecasting. The platform allows users to upload datasets, train various forecasting models (ARIMA, Linear Regression), and compare model performance through an intuitive web interface.

## System Architecture

The application follows a traditional Flask MVC architecture with the following components:

**Backend Framework**: Flask with SQLAlchemy ORM
- Flask serves as the web framework handling HTTP requests and responses
- SQLAlchemy provides database abstraction and ORM capabilities
- Gunicorn serves as the WSGI server for production deployment

**Database**: SQLite (development) with PostgreSQL support
- SQLite used as default for development and testing
- PostgreSQL packages included for production deployments
- Database schema managed through SQLAlchemy models

**Frontend**: Server-side rendered templates with Bootstrap
- Jinja2 templating engine for dynamic HTML generation
- Bootstrap 5 for responsive UI components
- Plotly.js for interactive data visualizations
- Font Awesome for icons

## Key Components

### Data Processing Engine (`data_processor.py`)
- Handles CSV and Excel file uploads
- Performs data preprocessing and validation
- Converts date columns to datetime format
- Validates numeric target columns for forecasting

### Forecasting Engine (`forecasting.py`)
- Implements multiple forecasting algorithms:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - Linear Regression
  - Moving Average (planned)
- Calculates performance metrics (RMSE, MAE, MAPE, R²)
- Handles train/test data splitting (80/20)

### Database Models (`models.py`)
- **Project**: Stores project metadata, dataset information, and preprocessing configurations
- **ModelResult**: Stores trained model parameters and performance metrics
- JSON serialization for complex configuration storage

### Web Routes (`routes.py`)
- Dashboard for project overview and statistics
- Dataset upload and project creation
- Model training and comparison interfaces
- RESTful API endpoints for AJAX interactions

## Data Flow

1. **Data Upload**: Users upload CSV/Excel files through the web interface
2. **Preprocessing**: DataProcessor validates and cleans the time series data
3. **Model Training**: ForecastingEngine trains multiple models on the processed data
4. **Evaluation**: Models are evaluated using standard forecasting metrics
5. **Storage**: Results are persisted in the database for comparison and analysis
6. **Visualization**: Performance metrics and forecasts are displayed through interactive charts

## External Dependencies

### Python Libraries
- **pandas/numpy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning algorithms and metrics
- **statsmodels**: Statistical models including ARIMA
- **openpyxl**: Excel file processing
- **psycopg2-binary**: PostgreSQL database adapter

### Frontend Libraries (CDN)
- **Bootstrap 5**: UI framework and components
- **Font Awesome**: Icon library
- **Plotly.js**: Interactive charting library

### Infrastructure
- **Gunicorn**: Production WSGI server
- **PostgreSQL**: Production database (Nix package)
- **OpenSSL**: Secure connections and encryption

## Deployment Strategy

**Development Environment**:
- Flask development server with debug mode enabled
- SQLite database for rapid prototyping
- Hot reload enabled for development workflow

**Production Environment**:
- Gunicorn WSGI server with autoscaling deployment target
- PostgreSQL database for production data persistence
- Proxy fix middleware for proper request handling behind reverse proxy
- Environment-based configuration for secrets and database URLs

**File Upload Handling**:
- 16MB maximum file size limit
- Secure filename handling to prevent path traversal
- Dedicated uploads directory for dataset storage

**Security Considerations**:
- Session secret key from environment variables
- File upload validation and sanitization
- SQL injection prevention through SQLAlchemy ORM

## Changelog
- June 15, 2025: Initial setup with Flask application structure
- June 15, 2025: Added PostgreSQL database for production-grade data persistence
- June 15, 2025: Updated dependency specifications to user-requested versions (numpy>=2.2.6, pandas>=2.2.3, lightgbm>=4.6.0, scikit-learn>=1.6.1, prophet>=1.1.6)
- June 15, 2025: Fixed Prophet model timeout issues with optimized settings and error handling
- June 15, 2025: Fixed Random Forest training errors with improved feature validation and pandas compatibility
- June 15, 2025: Made LightGBM conditionally available (disabled due to system dependencies but prevents application crashes)

## User Preferences

Preferred communication style: Simple, everyday language.
# Codebase Structure Analysis

## Directory Layout

```
timeseriespro/
├── main.py              # Flask application entry point
├── app.py               # Secondary application file
├── routes.py            # Web routes and API endpoints
├── forecasting.py       # Core forecasting algorithms
├── models.py            # Database models
├── data_processor.py    # Data processing utilities
├── gunicorn.conf.py     # Production server configuration
├── pyproject.toml       # Project dependencies and config
├── uv.lock             # Lock file for dependencies
├── replit.md           # Deployment notes
├── instance/           # Instance-specific files
│   └── forecasting.db  # SQLite database
├── static/             # Static web assets
│   ├── css/
│   │   └── style.css   # Main stylesheet
│   └── js/
│       ├── main.js     # Core JavaScript
│       └── charts.js   # Chart visualization logic
├── templates/          # Jinja2 HTML templates
│   ├── base.html       # Base template
│   ├── dashboard.html  # Main dashboard
│   ├── project.html    # Project management
│   ├── upload.html     # Data upload form
│   └── compare.html    # Model comparison
├── uploads/            # User uploaded files
└── tmp/                # Temporary files
```

## Core Modules

### main.py / app.py
- Flask application initialization
- Configuration management
- Database setup
- Route registration

### forecasting.py
- ARIMA model implementation
- Prophet forecasting
- Linear regression
- Model evaluation metrics
- Forecast data processing

### routes.py
- Web route handlers
- API endpoints
- File upload processing
- Response rendering

### models.py
- SQLAlchemy database models
- Project model
- Forecast model
- Data relationships

### data_processor.py
- CSV file validation
- Data cleaning and preprocessing
- Time series data formatting
- Error handling for data issues

## Frontend Architecture

### Templates
- **base.html**: Common layout and navigation
- **dashboard.html**: Main application interface
- **project.html**: Project management interface
- **upload.html**: Data upload form
- **compare.html**: Model comparison results

### Static Assets
- **style.css**: Application styling
- **main.js**: Core JavaScript functionality
- **charts.js**: Chart.js integration for visualizations

## Database Schema

### Projects Table
- Stores project metadata
- Links to associated forecasts
- User organization

### Forecasts Table
- Individual forecast results
- Model parameters
- Performance metrics
- Generated predictions

## Dependencies (pyproject.toml)
Key libraries:
- Flask for web framework
- SQLAlchemy for database ORM
- pandas for data manipulation
- scikit-learn for machine learning
- Prophet for time series forecasting
- Chart.js for frontend visualizations

## Development Patterns
- MVC architecture with Flask
- Database migrations through SQLAlchemy
- Template inheritance for UI consistency
- AJAX for dynamic content updates
- RESTful API design patterns
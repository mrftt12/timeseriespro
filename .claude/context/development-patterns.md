# Development Patterns & Best Practices

## Code Organization Patterns

### Separation of Concerns

**Business Logic Layer** (`forecasting.py`, `data_processor.py`)
- Pure Python classes with no Flask dependencies
- Stateless operations with clear input/output contracts
- Comprehensive error handling and logging
- Testable in isolation

**Web Layer** (`routes.py`, `app.py`)
- Thin controllers that orchestrate business logic
- Request/response handling only
- Template rendering and data serialization
- Flash messages for user feedback

**Data Layer** (`models.py`)
- SQLAlchemy ORM models with helper methods
- JSON serialization/deserialization utilities
- Clear relationship definitions
- Database constraints and validation

### Error Handling Patterns

#### Graceful Degradation
```python
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    lgb = None
    LIGHTGBM_AVAILABLE = False
```

#### Comprehensive Exception Handling
```python
try:
    # Core operation
    result = engine.train_model()
except ValueError as e:
    flash(f'Data validation error: {str(e)}', 'error')
except Exception as e:
    flash(f'Training failed: {str(e)}', 'error')
```

#### User-Friendly Error Messages
- Technical errors converted to actionable user messages
- Flash message categories for appropriate styling
- Detailed logging for debugging while showing simplified errors to users

## Data Processing Patterns

### Standardized Pipeline Architecture
Every forecasting algorithm follows consistent pipeline:

1. **Data Preparation** (`_prepare_data()`)
   - 80/20 train/test split
   - Consistent data validation
   - Index handling for time series

2. **Feature Engineering** (algorithm-specific)
   - Time-based features (day, month, quarter)
   - Lag variables for autocorrelation
   - Rolling statistics (mean, std, min, max)
   - Difference features for stationarity

3. **Model Training**
   - Algorithm-specific implementation
   - Hyperparameter handling
   - Cross-validation where appropriate

4. **Prediction & Evaluation**
   - Test set predictions
   - Standardized metrics calculation
   - Future forecast generation
   - Confidence interval estimation

5. **Result Packaging**
   - Consistent return format
   - JSON-serializable data structures
   - Visualization-ready format

### JSON Serialization Pattern
```python
# Model parameters
def get_parameters(self):
    if self.parameters:
        return json.loads(self.parameters)
    return {}

def set_parameters(self, params):
    self.parameters = json.dumps(params)
```

## Database Patterns

### Relationship Management
```python
# One-to-many with cascade delete
models = db.relationship('ModelResult', backref='project', lazy=True, cascade='all, delete-orphan')
```

### Timestamp Automation
```python
created_at = db.Column(db.DateTime, default=datetime.utcnow)
updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### Flexible Configuration Storage
- JSON fields for model parameters
- Backward compatibility handling
- Schema evolution support

## Frontend Patterns

### Template Inheritance
```html
<!-- base.html -->
{% block content %}{% endblock %}
{% block scripts %}{% endblock %}

<!-- project.html -->
{% extends "base.html" %}
{% block content %}...{% endblock %}
```

### Chart Abstraction
```javascript
// Reusable chart utilities
const ChartUtils = {
    defaultConfig: {...},
    colors: {...}
};

// Specialized chart functions
function createForecastChart(data, containerId) {
    // Consistent chart creation pattern
}
```

### Form Handling Pattern
```python
@app.route('/endpoint', methods=['GET', 'POST'])
def handler():
    if request.method == 'POST':
        # Validate input
        # Process data
        # Flash message
        # Redirect on success
    return render_template(...)
```

## Configuration Management

### Environment-Based Configuration
```python
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///forecasting.db")
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
```

### Feature Flags via Availability Checking
```python
# Dynamic UI based on available features
lightgbm_available=LIGHTGBM_AVAILABLE,
xgboost_available=XGBOOST_AVAILABLE
```

## Validation Patterns

### File Upload Validation
```python
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

### Data Validation
```python
# Check required fields
if not project.date_column or not project.target_column:
    flash('Please configure date and target columns first', 'error')
    return redirect(url_for('project_detail', id=id))
```

### Parameter Validation
```python
# Type conversion with defaults
window = int(request.form.get('ma_window', 7))
learning_rate = float(request.form.get('lgb_learning_rate', 0.1))
```

## Testing Patterns

### Algorithm Testing Structure
```python
def test_algorithm_consistency():
    """Test that algorithm returns expected result structure"""
    # Setup test data
    # Train model
    # Validate result format
    # Check metric ranges
```

### Data Processing Testing
```python
def test_data_preprocessing():
    """Test data validation and preprocessing"""
    # Test with various data formats
    # Validate error handling
    # Check edge cases
```

## Performance Patterns

### Lazy Loading
```python
# Database relationships loaded on demand
models = db.relationship('ModelResult', backref='project', lazy=True)
```

### Efficient Data Transfer
```python
# JSON serialization for complex structures
forecast_data = json.dumps(forecast_results)
```

### Memory Management
```python
# Clear model references after training
del model  # Explicit cleanup for large models
```

## Security Patterns

### Secure File Handling
```python
filename = secure_filename(file.filename)
file_path = os.path.join(upload_dir, filename)
```

### SQL Injection Prevention
```python
# Always use ORM queries
project = Project.query.get_or_404(id)
models = ModelResult.query.filter_by(project_id=project_id).all()
```

### Input Sanitization
```python
project_name = request.form.get('project_name', '').strip()
if not project_name:
    flash('Project name is required', 'error')
```

## Extension Patterns

### Adding New Algorithms
1. **Import with Availability Flag**
   ```python
   try:
       import new_algorithm
       NEW_ALGORITHM_AVAILABLE = True
   except ImportError:
       NEW_ALGORITHM_AVAILABLE = False
   ```

2. **Implement Training Method**
   ```python
   def train_new_algorithm(self, param1=default, forecast_periods=30):
       # Follow standardized pattern
       train_data, test_data = self._prepare_data()
       # Algorithm-specific implementation
       return standardized_result
   ```

3. **Add Route Handler**
   ```python
   elif model_type == 'new_algorithm':
       param1 = request.form.get('param1', default_value)
       result = engine.train_new_algorithm(param1)
   ```

4. **Update Frontend**
   ```html
   {% if new_algorithm_available %}
   <option value="new_algorithm">New Algorithm</option>
   {% endif %}
   ```

### Adding New Visualizations
1. **Create Chart Function** (`charts.js`)
2. **Add Data Processing** (backend)
3. **Update Templates** (frontend)
4. **Add Route Handling** (if needed)

### Database Schema Evolution
1. **Backward Compatible Changes**
   - New nullable columns
   - JSON field extensions
   - Additional tables

2. **Migration Strategy**
   - Development: Auto-creation
   - Production: Manual migration scripts
   - JSON compatibility handling

## Deployment Patterns

### Development
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
```

### Production
```python
# gunicorn.conf.py configuration
bind = "0.0.0.0:8000"
workers = 2
```

### Environment Configuration
- Development: SQLite database
- Production: PostgreSQL via environment variables
- Logging: Configurable levels
- Secret management: Environment variables

## Monitoring & Observability

### Logging Strategy
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Suppress noisy libraries
logging.getLogger('prophet').setLevel(logging.WARNING)
```

### Error Tracking
- Comprehensive exception handling
- User-friendly error messages
- Technical details in logs
- Flash message categorization

### Performance Monitoring
- Training time tracking (implicit)
- Model performance metrics storage
- Database query optimization opportunities
- Memory usage awareness
# Frontend Architecture & API Endpoints

## Frontend Technology Stack

### Core Libraries
- **Bootstrap 5.3.0**: Responsive UI framework
- **Font Awesome 6.4.0**: Icon library
- **Plotly.js 2.26.0**: Interactive charting library
- **Vanilla JavaScript**: Custom functionality (no jQuery dependency)

### Template Engine
- **Jinja2**: Server-side templating with Flask integration
- **Template Inheritance**: Base template with blocks for extensibility
- **Context Variables**: Flask route data passed to templates

## API Endpoints

### Core Application Routes

#### Dashboard (`/`)
- **Method**: GET
- **Handler**: `dashboard()` in `routes.py:20-35`
- **Purpose**: Main landing page with project overview
- **Data**:
  - All projects (ordered by `updated_at`)
  - Summary statistics (total projects, total models)
  - Recent activity (last 5 model results)
- **Template**: `dashboard.html`

#### Upload Dataset (`/upload`)
- **Methods**: GET, POST
- **Handler**: `upload_dataset()` in `routes.py:37-89`
- **Purpose**: Create new project with dataset upload
- **POST Data**:
  - `file`: CSV/Excel file
  - `project_name`: Required string
  - `project_description`: Optional string
- **Security**: 
  - File extension validation
  - Secure filename handling
  - 16MB max file size
- **Response**: Redirect to project detail on success

#### Project Detail (`/project/<int:id>`)
- **Method**: GET
- **Handler**: `project_detail()` in `routes.py:91-129`
- **Purpose**: Project dashboard with dataset preview and models
- **Data**:
  - Project metadata
  - Dataset preview (10 rows, column info, dtypes)
  - All trained models for project
  - Algorithm availability flags
- **Template**: `project.html`

#### Configure Project (`/project/<int:id>/configure`)
- **Method**: POST
- **Handler**: `configure_project()` in `routes.py:131-149`
- **Purpose**: Set date and target columns for time series
- **POST Data**:
  - `date_column`: Column name for datetime
  - `target_column`: Column name for forecast target
- **Response**: Redirect to project detail

#### Train Model (`/project/<int:id>/train`)
- **Method**: POST
- **Handler**: `train_model()` in `routes.py:151-233`
- **Purpose**: Train forecasting model
- **POST Data**:
  - `model_type`: Algorithm selection (arima, prophet, etc.)
  - `model_name`: Optional custom name
  - Algorithm-specific parameters (dynamic)
- **Response**: Redirect to project detail with results

### API Endpoint Patterns

#### Model Training Parameters by Algorithm

**ARIMA**:
- No additional parameters (uses defaults)

**Linear Regression**:
- No additional parameters

**Moving Average**:
- `ma_window`: Window size (default: 7)

**Random Forest**:
- `rf_n_estimators`: Number of trees (default: 100)
- `rf_max_depth`: Tree depth (default: 10)

**Prophet**:
- No additional parameters

**LightGBM**:
- `lgb_num_leaves`: Leaf count (default: 31)
- `lgb_learning_rate`: Learning rate (default: 0.1)
- `lgb_n_estimators`: Boosting rounds (default: 100)

**XGBoost**:
- `xgb_n_estimators`: Estimators (default: 100)
- `xgb_max_depth`: Tree depth (default: 6)
- `xgb_learning_rate`: Learning rate (default: 0.1)

**SARIMAX**:
- `sarimax_order`: ARIMA order as "p,d,q" (default: "1,1,1")
- `sarimax_seasonal_order`: Seasonal order as "P,D,Q,s" (default: "1,1,1,12")

**LSTM**:
- `lstm_sequence_length`: Lookback window (default: 30)
- `lstm_hidden_units`: Hidden layer size (default: 50)
- `lstm_epochs`: Training epochs (default: 100)

**NHITS**:
- `nhits_epochs`: Training epochs (default: 100)

## Frontend Architecture

### Template Structure

#### Base Template (`templates/base.html`)
```html
<!DOCTYPE html>
<html>
<head>
    <!-- Bootstrap, Font Awesome, Plotly.js -->
    <!-- Custom CSS -->
</head>
<body>
    <nav><!-- Navigation bar --></nav>
    <div><!-- Flash messages --></div>
    <main>{% block content %}{% endblock %}</main>
    <script><!-- Bootstrap, custom JS --></script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

#### Navigation Features
- Active link highlighting based on `request.endpoint`
- Responsive navbar with mobile collapse
- Icon integration with Font Awesome
- Consistent branding

#### Flash Message System
- Server-side Flask flash messages
- Bootstrap alert styling
- Auto-dismissible with close buttons
- Category-based styling (success, error, warning)

### JavaScript Architecture

#### Chart Utilities (`static/js/charts.js`)

**ChartUtils Object**:
- Default Plotly.js configuration
- Color palette management
- Reusable layout templates

**Chart Functions**:
- `createForecastChart(data, containerId)`: Main time series visualization
- `createMetricsComparisonChart(models, containerId)`: Bar chart comparison
- `createDecompositionChart(data, containerId)`: Trend/seasonal decomposition
- `createResidualChart(data, containerId)`: Residual analysis scatter plot
- `createPerformanceRadarChart(models, containerId)`: Performance radar chart

#### Chart Data Structure
```javascript
{
    historical: {
        dates: ['2023-01-01', ...],
        values: [100.5, ...]
    },
    test: {
        dates: ['2023-12-01', ...],
        actual: [95.2, ...],
        predicted: [94.8, ...]
    },
    forecast: {
        dates: ['2024-01-01', ...],
        values: [98.5, ...],
        confidence_lower: [93.1, ...],
        confidence_upper: [103.9, ...]
    }
}
```

### Responsive Design

#### Breakpoints
- Mobile-first Bootstrap approach
- Collapsible navigation for mobile
- Responsive charts with Plotly.js
- Fluid container layout

#### Chart Responsiveness
- `responsive: true` in Plotly config
- Dynamic sizing based on container
- Mobile-optimized hover interactions
- Export functionality (PNG/SVG)

### Error Handling

#### Frontend Validation
- HTML5 form validation
- File type restrictions
- Required field indicators
- Bootstrap validation styling

#### Backend Error Display
- Flask flash messages for user feedback
- Exception handling in route handlers
- Graceful degradation for missing data
- Algorithm availability checks

## Data Flow

### Project Creation Workflow
1. User uploads file → `upload_dataset()` POST
2. File validation and storage
3. Database record creation
4. Redirect to project detail page

### Model Training Workflow
1. User configures columns → `configure_project()` POST
2. User selects algorithm and parameters → `train_model()` POST  
3. `ForecastingEngine` processes data
4. Results stored in database
5. Page refresh displays new model

### Chart Rendering Workflow
1. Server renders template with model data
2. JavaScript parses forecast JSON
3. Plotly.js renders interactive chart
4. User can interact, zoom, export

## Security Considerations

### File Upload Security
- Extension whitelist: CSV, Excel only
- `secure_filename()` for path traversal protection
- File size limits (16MB)
- Upload directory isolation

### Input Validation
- HTML form validation
- Server-side parameter validation
- SQL injection prevention via ORM
- XSS protection via Jinja2 auto-escaping

### Session Management
- Flask session cookies
- No user authentication (single-user app)
- CSRF protection not implemented (consider for multi-user)

## Performance Considerations

### Frontend Performance
- CDN-hosted libraries (Bootstrap, Plotly.js)
- Minimal custom JavaScript
- On-demand chart rendering
- No unnecessary AJAX requests

### Data Transfer
- JSON serialization for model results
- Compressed chart data structure
- Minimal data preview (10 rows)
- Lazy loading of chart data

### Browser Compatibility
- Modern browser requirements (ES6+)
- Plotly.js compatibility requirements
- Bootstrap 5.x compatibility
- No Internet Explorer support

## Extension Points

### Adding New Chart Types
1. Create function in `charts.js`
2. Add to `window` exports
3. Call from template scripts
4. Add corresponding backend data structure

### New API Endpoints
1. Add route handler in `routes.py`
2. Create corresponding template
3. Update navigation if needed
4. Add form validation

### Enhanced Interactivity
- AJAX model training (avoid page refresh)
- Real-time training progress
- Model comparison interface
- Advanced parameter tuning UI
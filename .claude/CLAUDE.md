# Project: Time Series Pro

## Repository Information
- Repository: timeseriespro
- Type: Python Flask application for time series forecasting
- Primary Language: Python
- Framework: Flask
- Database: SQLite

## Current Project State
This is a time series forecasting application with:
- Data upload and processing capabilities
- Multiple forecasting models (ARIMA, Prophet, Linear Regression)
- Web interface for visualization
- Project management and comparison features

## Key Files
- `main.py` - Main Flask application
- `forecasting.py` - Core forecasting logic and models
- `routes.py` - Web routes and API endpoints
- `models.py` - Database models
- `data_processor.py` - Data processing utilities
- `templates/` - HTML templates
- `static/` - CSS and JavaScript assets

## PM System Rules

### Context Preservation
- Always read existing context from `.claude/context/` before starting work
- Update context files when making significant changes
- Maintain epic and task state in `.claude/epics/`

### Workflow Discipline
1. **No Vibe Coding**: Every change must trace back to a specification
2. **PRD First**: Start with Product Requirements Document
3. **Epic Planning**: Convert PRDs to technical implementation plans
4. **Task Decomposition**: Break epics into concrete, actionable tasks
5. **GitHub Sync**: Push tasks as issues for transparency

### Agent Coordination
- Use specialized agents for complex tasks
- Maintain progress updates in task files
- Sync updates to GitHub issues
- Coordinate parallel work through Git commits

### Code Quality Standards
- Follow existing code conventions
- Run tests after changes
- Update documentation
- Maintain security best practices

## Commands Available
- `/pm:prd-new` - Create new Product Requirements Document
- `/pm:prd-parse` - Convert PRD to implementation epic
- `/pm:epic-decompose` - Break epic into tasks
- `/pm:epic-sync` - Push to GitHub
- `/pm:issue-start` - Begin work on issue
- `/pm:next` - Get next priority task
- `/pm:status` - Project dashboard

## Development Guidelines
- Prefer editing existing files over creating new ones
- Never commit secrets or API keys
- Use type hints in Python code
- Follow Flask best practices
- Test changes before deployment

## Current Focus
The project appears to be in active development with modifications to core files. The PM system will help organize future feature development and maintain code quality.
# Dockerfile for TimeSeriesPro Flask App

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Create venv and install dependencies
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# Ensure venv is used by default
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Expose port (default Flask port)
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app
CMD ["python", "main.py"]

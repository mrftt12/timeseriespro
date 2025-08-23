# Dockerfile for TimeSeriesPro Flask App

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads instance static/reports

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_ENV=production
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application with Gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]

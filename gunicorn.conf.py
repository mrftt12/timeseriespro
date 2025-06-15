# Gunicorn configuration for TimeSeries Forecasting Platform
import os

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# Timeouts - Increased for ML model training
timeout = 600  # 10 minutes for model training
keepalive = 2
graceful_timeout = 600

# Restart workers after this many requests, with up to 50 more requests
# to avoid thundering herd.
max_requests = 1200
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'forecasting_platform'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None

# Reload application when code changes (development)
reload = True
reload_engine = 'auto'
"""
Celery configuration and task definitions for Time Series Pro
Part of Epic #2: Advanced Data Science Features - Asynchronous Processing
"""

import os
from celery import Celery
from celery.signals import worker_ready
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery instance
def create_celery_app():
    """Create and configure Celery application"""
    celery_app = Celery(
        'timeseries_pro',
        broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
        backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
        include=[
            'tasks.data_tasks',
            'tasks.model_tasks',
            'tasks.optimization_tasks'
        ]
    )
    
    # Configure Celery
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=30 * 60,  # 30 minutes
        task_soft_time_limit=25 * 60,  # 25 minutes
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        result_expires=3600,  # 1 hour
        task_routes={
            'tasks.data_tasks.*': {'queue': 'data_processing'},
            'tasks.model_tasks.*': {'queue': 'model_training'},
            'tasks.optimization_tasks.*': {'queue': 'optimization'},
        },
        task_default_queue='default',
        task_create_missing_queues=True
    )
    
    return celery_app

# Create the Celery app instance
celery_app = create_celery_app()

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handler called when worker is ready"""
    logger.info("Celery worker is ready and waiting for tasks")

if __name__ == '__main__':
    # Start Celery worker
    celery_app.start()
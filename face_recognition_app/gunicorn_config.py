"""
Gunicorn configuration for Face Recognition Application with Uvicorn workers.

This configuration enables multi-process architecture for better CPU utilization
and stability. Each worker is a separate process with its own memory space.

Usage:
  # Production with multiple workers:
  gunicorn face_app.asgi:application -c gunicorn_config.py

  # Or with specific worker count:
  gunicorn face_app.asgi:application -c gunicorn_config.py -w 4

Benefits:
  - Multiple processes = Better CPU core utilization
  - Process isolation = One crash doesn't affect others
  - Automatic worker restart on failure
  - Graceful reload for zero-downtime deployments
"""

import os
import multiprocessing

# =============================================================================
# WORKER CONFIGURATION
# =============================================================================

# Worker class - use Uvicorn for ASGI support
worker_class = 'uvicorn.workers.UvicornWorker'

# Number of worker processes
# Rule of thumb: (2 * CPU cores) + 1 for I/O bound apps
# For CPU-intensive face recognition, use CPU cores count
# Set via GUNICORN_WORKERS env var or default to 4
_cpu_count = multiprocessing.cpu_count()
workers = int(os.environ.get('GUNICORN_WORKERS', min(_cpu_count, 4)))

# Threads per worker
# For async workers like Uvicorn, threads are not used the same way
# Keep at 1 since Uvicorn handles concurrency with async
threads = 1

# Worker connections - max simultaneous clients per worker
worker_connections = int(os.environ.get('GUNICORN_WORKER_CONNECTIONS', 1000))


# =============================================================================
# BINDING CONFIGURATION
# =============================================================================

# Bind address
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8002')


# =============================================================================
# TIMEOUT CONFIGURATION
# =============================================================================

# Worker timeout (seconds) - kill worker if no response
timeout = int(os.environ.get('GUNICORN_TIMEOUT', 120))

# Keep-alive for connections
keepalive = int(os.environ.get('GUNICORN_KEEPALIVE', 120))

# Graceful timeout for worker shutdown
graceful_timeout = int(os.environ.get('GUNICORN_GRACEFUL_TIMEOUT', 30))


# =============================================================================
# PROCESS MANAGEMENT
# =============================================================================

# Preload application code before forking workers
# Reduces memory via copy-on-write, but shared FaceAnalysis won't work across processes
# Set to False to ensure each process gets its own FaceAnalysis instance
preload_app = False

# Max requests per worker before restart (prevents memory leaks)
max_requests = int(os.environ.get('GUNICORN_MAX_REQUESTS', 10000))

# Random jitter for max_requests to prevent all workers restarting at once
max_requests_jitter = int(os.environ.get('GUNICORN_MAX_REQUESTS_JITTER', 1000))

# Daemon mode - run in background (set via command line usually)
daemon = False


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log level
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')

# Access log - set to '-' for stdout or file path
accesslog = os.environ.get('GUNICORN_ACCESS_LOG', '-')

# Error log - set to '-' for stderr or file path
errorlog = os.environ.get('GUNICORN_ERROR_LOG', '-')

# Access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'


# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# Limit request line size
limit_request_line = 4094

# Limit request field size
limit_request_field_size = 8190

# Limit request fields count
limit_request_fields = 100


# =============================================================================
# LIFECYCLE HOOKS
# =============================================================================

def on_starting(server):
    """Called just before the master process is initialized."""
    print(f"Starting Gunicorn with {workers} workers...")


def on_reload(server):
    """Called when receiving SIGHUP signal."""
    print("Reloading Gunicorn workers...")


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    # Each worker gets its own process, so FaceMesh pool will be per-process
    # This is intentional - prevents cross-process issues with GPU/EGL contexts
    import logging
    logger = logging.getLogger('gunicorn')
    logger.info(f"Worker {worker.pid} started")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def pre_exec(server):
    """Called just before a new master process is forked."""
    pass


def worker_exit(server, worker):
    """Called when a worker is about to exit."""
    import logging
    logger = logging.getLogger('gunicorn')
    logger.info(f"Worker {worker.pid} exiting")
    
    # Cleanup FaceMesh pool on worker exit
    try:
        from core.face_recognition_engine import cleanup_face_mesh_pool
        cleanup_face_mesh_pool()
    except Exception as e:
        logger.warning(f"Error cleaning up FaceMesh pool: {e}")


def child_exit(server, worker):
    """Called when a worker child process exits."""
    pass


# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

def when_ready(server):
    """Called when server is ready to receive requests."""
    print("=" * 60)
    print("Face Recognition - Gunicorn + Uvicorn Server Ready")
    print("=" * 60)
    print(f"Workers:     {workers}")
    print(f"Worker class: {worker_class}")
    print(f"Bind:        {bind}")
    print(f"Timeout:     {timeout}s")
    print(f"Max requests: {max_requests}")
    print("=" * 60)

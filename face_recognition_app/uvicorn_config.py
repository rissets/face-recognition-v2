"""
Uvicorn configuration for Face Recognition Application.

This configuration provides high-performance ASGI server setup
with support for HTTPS and WebSocket connections.

Usage:
  # Development (with auto-reload):
  uvicorn face_app.asgi:application --config uvicorn_config.py

  # Production:
  uvicorn face_app.asgi:application --host 0.0.0.0 --port 8000 \
    --workers 4 --loop uvloop --http httptools --ws websockets

  # Production with SSL:
  uvicorn face_app.asgi:application --host 0.0.0.0 --port 8000 \
    --workers 4 --loop uvloop --http httptools --ws websockets \
    --ssl-keyfile=/path/to/key.pem --ssl-certfile=/path/to/cert.pem
"""

import os
import multiprocessing

# Determine environment
IS_PRODUCTION = os.environ.get('ENVIRONMENT', 'development') == 'production'
DEBUG = os.environ.get('DEBUG', 'true').lower() == 'true'

# Bind settings
bind = os.environ.get('UVICORN_BIND', '0.0.0.0:8000')
host = os.environ.get('UVICORN_HOST', '0.0.0.0')
port = int(os.environ.get('UVICORN_PORT', '8000'))

# Worker settings
# For production, use multiple workers
# For development, single worker with reload
if IS_PRODUCTION:
    workers = int(os.environ.get('UVICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
    reload = False
else:
    workers = 1
    reload = True

# Event loop and HTTP parser settings for high performance
loop = 'uvloop'  # Faster than asyncio
http = 'httptools'  # Faster HTTP parsing
ws = 'websockets'  # WebSocket implementation

# SSL settings (for HTTPS)
ssl_keyfile = os.environ.get('SSL_KEYFILE', None)
ssl_certfile = os.environ.get('SSL_CERTFILE', None)
ssl_keyfile_password = os.environ.get('SSL_KEYFILE_PASSWORD', None)
ssl_version = int(os.environ.get('SSL_VERSION', '2'))  # TLS 1.2+
ssl_ciphers = os.environ.get('SSL_CIPHERS', 'TLSv1.2')

# Timeouts
timeout_keep_alive = int(os.environ.get('UVICORN_TIMEOUT_KEEP_ALIVE', '120'))
timeout_notify = int(os.environ.get('UVICORN_TIMEOUT_NOTIFY', '30'))

# Logging
log_level = os.environ.get('UVICORN_LOG_LEVEL', 'info' if IS_PRODUCTION else 'debug')
access_log = not IS_PRODUCTION or os.environ.get('UVICORN_ACCESS_LOG', 'false').lower() == 'true'

# WebSocket specific settings
ws_max_size = int(os.environ.get('WS_MAX_SIZE', str(16 * 1024 * 1024)))  # 16MB default
ws_ping_interval = float(os.environ.get('WS_PING_INTERVAL', '20.0'))
ws_ping_timeout = float(os.environ.get('WS_PING_TIMEOUT', '20.0'))

# Request limits
limit_concurrency = int(os.environ.get('UVICORN_LIMIT_CONCURRENCY', '1000')) if IS_PRODUCTION else None
limit_max_requests = int(os.environ.get('UVICORN_LIMIT_MAX_REQUESTS', '10000')) if IS_PRODUCTION else None

# Lifespan - disable for Django Channels compatibility
lifespan = 'off'

# Headers
server_header = False  # Don't expose server info
date_header = True


def get_uvicorn_config() -> dict:
    """
    Get uvicorn configuration as a dictionary.
    Can be used with uvicorn.run() programmatically.
    """
    config = {
        'host': host,
        'port': port,
        'workers': workers,
        'loop': loop,
        'http': http,
        'ws': ws,
        'reload': reload,
        'log_level': log_level,
        'access_log': access_log,
        'timeout_keep_alive': timeout_keep_alive,
        'lifespan': lifespan,
        'server_header': server_header,
        'date_header': date_header,
    }
    
    # Add SSL if configured
    if ssl_keyfile and ssl_certfile:
        config['ssl_keyfile'] = ssl_keyfile
        config['ssl_certfile'] = ssl_certfile
        if ssl_keyfile_password:
            config['ssl_keyfile_password'] = ssl_keyfile_password
    
    # Add limits for production
    if IS_PRODUCTION:
        if limit_concurrency:
            config['limit_concurrency'] = limit_concurrency
        if limit_max_requests:
            config['limit_max_requests'] = limit_max_requests
    
    return config


if __name__ == '__main__':
    import uvicorn
    
    # Run with configuration
    uvicorn.run(
        'face_app.asgi:application',
        **get_uvicorn_config()
    )

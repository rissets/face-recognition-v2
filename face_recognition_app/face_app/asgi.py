"""
ASGI config for face_app project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# CRITICAL: Limit ALL thread pools BEFORE any library imports
# This prevents thread explosion from ONNX, OpenBLAS, MKL, etc.
# =============================================================================

# Limit ONNX Runtime threads (InsightFace uses this)
os.environ.setdefault('ORT_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

# Limit OpenBLAS threads (used by numpy/scipy)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

# Limit MKL threads (Intel Math Kernel Library)
os.environ.setdefault('MKL_NUM_THREADS', '4')

# Limit other thread pools
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

# TensorFlow/CUDA thread limits (if used)
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')

# =============================================================================
# Limit Django ThreadPoolExecutor
# =============================================================================

# Get max threads from environment or use conservative default
MAX_THREAD_WORKERS = int(os.environ.get('DJANGO_MAX_THREADS', 16))

# Create bounded thread pool executor
_bounded_executor = ThreadPoolExecutor(
    max_workers=MAX_THREAD_WORKERS,
    thread_name_prefix='django_sync_'
)

# Set as default executor for asyncio event loop
# This affects all sync_to_async calls in Django
def _configure_thread_pool():
    """Configure bounded thread pool for async operations"""
    try:
        loop = asyncio.get_event_loop()
        loop.set_default_executor(_bounded_executor)
    except RuntimeError:
        # No event loop yet, will be configured when one is created
        pass

_configure_thread_pool()

# Also patch asgiref's SyncToAsync to use bounded executor
try:
    import asgiref.sync
    asgiref.sync.SyncToAsync.executor = _bounded_executor
except (ImportError, AttributeError):
    pass

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "face_app.settings")

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator

django_asgi_app = get_asgi_application()

# Import after Django is set up
import streaming.routing

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": URLRouter(
        streaming.routing.websocket_urlpatterns
    ),
})

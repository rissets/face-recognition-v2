# Multi-stage build untuk Django application
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# =============================================================================
# CRITICAL: Thread pool limits to prevent thread explosion
# =============================================================================
ENV ORT_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV TF_NUM_INTEROP_THREADS=4
ENV TF_NUM_INTRAOP_THREADS=4
ENV DJANGO_MAX_THREADS=16
ENV FACE_MESH_POOL_SIZE=4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libpq-dev \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p media static logs staticfiles

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

# Development stage
FROM base as development
USER appuser
EXPOSE 8000
# Use uvicorn for development with ASGI support (WebSocket + HTTP)
CMD ["uvicorn", "face_app.asgi:application", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Collect static files
RUN python manage.py collectstatic --noinput || true

# Change to non-root user
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Production command with Uvicorn (ASGI) for WebSocket + HTTPS support
# Single worker to prevent thread explosion from ONNX/InsightFace
# Thread limits set via ENV variables above
CMD ["uvicorn", "face_app.asgi:application", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop", "--http", "httptools", "--ws", "websockets", "--lifespan", "off", "--limit-concurrency", "100", "--limit-max-requests", "5000"]
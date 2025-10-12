# Multi-stage build untuk Django application
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99
ENV OPENCV_LOG_LEVEL=ERROR

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libpq-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libx11-dev \
    libxss1 \
    libgconf-2-4 \
    pkg-config \
    wget \
    curl \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Fix OpenCV installation to ensure headless version is used
RUN python fix_opencv.py

# Create necessary directories
RUN mkdir -p media static logs staticfiles

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

# Development stage
FROM base as development
USER appuser
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# Production stage
FROM base as production

# Install production dependencies
RUN pip install gunicorn

# Collect static files
RUN python manage.py collectstatic --noinput || true

# Change to non-root user
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Production command with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "face_app.wsgi:application"]
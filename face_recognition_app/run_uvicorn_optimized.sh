#!/bin/bash
# Optimized Uvicorn runner with thread limits
# Use this instead of gunicorn for lower CPU/thread usage

# =============================================================================
# Thread Pool Limits - CRITICAL for preventing thread explosion
# =============================================================================
export ORT_NUM_THREADS=4          # ONNX Runtime (InsightFace)
export OMP_NUM_THREADS=4          # OpenMP
export OPENBLAS_NUM_THREADS=4     # OpenBLAS (numpy)
export MKL_NUM_THREADS=4          # Intel MKL
export VECLIB_MAXIMUM_THREADS=4   # Apple Accelerate
export NUMEXPR_NUM_THREADS=4      # NumExpr
export TF_NUM_INTEROP_THREADS=4   # TensorFlow
export TF_NUM_INTRAOP_THREADS=4   # TensorFlow

# Django thread pool limit
export DJANGO_MAX_THREADS=16

# FaceMesh pool size
export FACE_MESH_POOL_SIZE=4

# =============================================================================
# Uvicorn Configuration
# =============================================================================
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
WORKERS="${WORKERS:-1}"           # Single worker recommended
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "=============================================="
echo "Starting Optimized Uvicorn Server"
echo "=============================================="
echo "Host: $HOST:$PORT"
echo "Workers: $WORKERS"
echo "ORT_NUM_THREADS: $ORT_NUM_THREADS"
echo "DJANGO_MAX_THREADS: $DJANGO_MAX_THREADS"
echo "FACE_MESH_POOL_SIZE: $FACE_MESH_POOL_SIZE"
echo "=============================================="

cd "$(dirname "$0")"

exec uvicorn face_app.asgi:application \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --timeout-keep-alive 120 \
    --limit-concurrency 100 \
    --limit-max-requests 5000

# Multi-Worker Deployment Guide

## Overview

Panduan ini menjelaskan cara menjalankan Face Recognition Service dengan multiple workers untuk performa optimal.

## Arsitektur

```
                     ┌─────────────────────────────────┐
                     │          Gunicorn Master        │
                     │     (Process Manager)           │
                     └─────────────┬───────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
    ┌─────▼─────┐            ┌─────▼─────┐            ┌─────▼─────┐
    │  Worker 1  │            │  Worker 2  │            │  Worker N  │
    │ (Uvicorn)  │            │ (Uvicorn)  │            │ (Uvicorn)  │
    └─────┬─────┘            └─────┬─────┘            └─────┬─────┘
          │                        │                        │
    ┌─────▼─────┐            ┌─────▼─────┐            ┌─────▼─────┐
    │ FaceMesh  │            │ FaceMesh  │            │ FaceMesh  │
    │ Pool (8)  │            │ Pool (8)  │            │ Pool (8)  │
    └───────────┘            └───────────┘            └───────────┘
```

## Konfigurasi Optimal

### Resource Planning

| CPU Cores | Workers | FaceMesh/Worker | Total FaceMesh | Expected Concurrent Sessions |
|-----------|---------|-----------------|----------------|------------------------------|
| 4         | 2       | 4               | 8              | ~8-16                        |
| 8         | 4       | 8               | 32             | ~32-64                       |
| 16        | 8       | 8               | 64             | ~64-128                      |
| 32+       | 12      | 8               | 96             | ~100-200                     |

### Environment Variables

```bash
# Gunicorn settings
export GUNICORN_WORKERS=4              # Number of worker processes
export GUNICORN_BIND="0.0.0.0:8003"    # Bind address
export GUNICORN_TIMEOUT=120            # Worker timeout
export GUNICORN_LOG_LEVEL="info"       # Log level

# FaceMesh Pool settings  
export FACE_MESH_POOL_SIZE=8           # FaceMesh instances per worker
```

## Menjalankan Server

### Development (Single Worker)

```bash
cd face_recognition_app
uvicorn face_app.asgi:application --reload --host 0.0.0.0 --port 8003
```

### Production (Multi-Worker dengan Gunicorn)

```bash
# Menggunakan script
./run_gunicorn.sh --workers 4 --pool-size 8

# Atau langsung dengan gunicorn
cd face_recognition_app
gunicorn face_app.asgi:application -c gunicorn_config.py -w 4
```

### Production dengan Systemd

```bash
# Copy service file
sudo cp face_recognition_app/face-service.service /etc/systemd/system/

# Edit sesuai path dan user
sudo nano /etc/systemd/system/face-service.service

# Reload dan start
sudo systemctl daemon-reload
sudo systemctl enable face-service
sudo systemctl start face-service

# Check status
sudo systemctl status face-service
journalctl -u face-service -f
```

## Monitoring

### Check Thread Count

```bash
# Per process
ps -eo pid,nlwp,cmd | grep gunicorn

# Total untuk user
ps -u aitstack -L | wc -l
```

### Check FaceMesh Pool

Tambahkan endpoint untuk monitoring:

```python
# Di views.py
from core.face_recognition_engine import _face_mesh_pool, _pool_initialized

def pool_status(request):
    if not _pool_initialized:
        return JsonResponse({'status': 'not_initialized'})
    
    return JsonResponse({
        'pool_size': _face_mesh_pool.qsize(),
        'max_size': _FACE_MESH_POOL_SIZE,
        'available': _face_mesh_pool.qsize(),
    })
```

## Troubleshooting

### Thread Count Terus Naik

1. **Penyebab**: FaceMesh tidak dikembalikan ke pool
2. **Solusi**: Pastikan menggunakan `safe_process()` yang menggunakan context manager

```python
# ✅ Benar - menggunakan safe_process
result = liveness_detector.safe_process(rgb_frame)

# ❌ Salah - akses langsung
result = liveness_detector.face_mesh.process(rgb_frame)
```

### Pool Exhausted Warning

1. **Penyebab**: Terlalu banyak concurrent sessions
2. **Solusi**: 
   - Naikkan `FACE_MESH_POOL_SIZE`
   - Tambah workers
   - Optimalkan processing time

### Worker Timeout

1. **Penyebab**: Processing terlalu lama
2. **Solusi**:
   ```bash
   export GUNICORN_TIMEOUT=180
   ```

## Benchmarking

```bash
# Test concurrent connections
ab -n 1000 -c 100 http://localhost:8003/api/health/

# WebSocket load test (requires wscat)
for i in {1..50}; do
  wscat -c ws://localhost:8003/ws/auth/process-image/test_$i/ &
done
```

## Rollback

Jika ada masalah, kembali ke single-process uvicorn:

```bash
sudo systemctl stop face-service
cd face_recognition_app
uvicorn face_app.asgi:application --host 0.0.0.0 --port 8003
```

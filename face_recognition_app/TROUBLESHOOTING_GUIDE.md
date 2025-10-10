# Troubleshooting Guide - Face Recognition Service

## Overview

Panduan lengkap untuk mengatasi masalah umum yang mungkin terjadi pada Face Recognition Third-Party Service.

## Quick Diagnostics

### System Health Check

```bash
# Check service status
sudo systemctl status face_recognition_web
sudo systemctl status face_recognition_celery
sudo systemctl status postgresql
sudo systemctl status redis
sudo systemctl status nginx

# Check disk space
df -h

# Check memory usage
free -h
htop

# Check network connectivity
netstat -tulpn | grep :8000
curl -I http://localhost:8000/api/health/
```

### Log Locations

```bash
# Application logs
/opt/face_recognition/logs/django.log
/opt/face_recognition/logs/face_recognition.log
/opt/face_recognition/logs/celery.log

# System logs
/var/log/nginx/access.log
/var/log/nginx/error.log
/var/log/postgresql/postgresql-13-main.log

# Supervisor logs
/var/log/supervisor/supervisord.log
/opt/face_recognition/logs/supervisor_web.log
/opt/face_recognition/logs/supervisor_celery.log
```

## Common Issues & Solutions

### 1. Database Connection Issues

#### Symptoms
```bash
django.db.utils.OperationalError: could not connect to server
psycopg2.OperationalError: FATAL: password authentication failed
```

#### Diagnosis
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test database connection
sudo -u postgres psql
\l  # List databases
\q  # Quit

# Test connection with app credentials
psql -h localhost -U face_app -d face_recognition_prod

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-13-main.log
```

#### Solutions

**Connection Refused:**
```bash
# Check if PostgreSQL is running
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Check listening ports
sudo netstat -tulpn | grep :5432

# Edit postgresql.conf
sudo nano /etc/postgresql/13/main/postgresql.conf
# Uncomment: listen_addresses = 'localhost'

# Restart PostgreSQL
sudo systemctl restart postgresql
```

**Authentication Failed:**
```bash
# Reset password
sudo -u postgres psql
ALTER USER face_app PASSWORD 'new_secure_password';

# Update .env file
nano /opt/face_recognition/.env
# DATABASE_URL=postgresql://face_app:new_secure_password@localhost:5432/face_recognition_prod

# Restart application
sudo supervisorctl restart face_recognition:*
```

**Too Many Connections:**
```bash
# Check current connections
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"

# Check max connections
sudo -u postgres psql -c "SHOW max_connections;"

# Edit postgresql.conf
sudo nano /etc/postgresql/13/main/postgresql.conf
# max_connections = 200

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### 2. Redis Connection Issues

#### Symptoms
```bash
redis.exceptions.ConnectionError: Error connecting to Redis
celery.exceptions.WorkerLostError: Worker exited prematurely
```

#### Diagnosis
```bash
# Check Redis status
sudo systemctl status redis

# Test Redis connection
redis-cli ping
# Should return: PONG

# Check Redis logs
sudo journalctl -u redis -f

# Check Redis configuration
redis-cli config get "*"
```

#### Solutions

**Redis Not Running:**
```bash
# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Check if Redis is listening
sudo netstat -tulpn | grep :6379
```

**Memory Issues:**
```bash
# Check Redis memory usage
redis-cli info memory

# Set maxmemory policy
redis-cli config set maxmemory-policy allkeys-lru
redis-cli config set maxmemory 1gb

# Restart Redis
sudo systemctl restart redis
```

### 3. Face Recognition Performance Issues

#### Symptoms
- Slow face recognition (> 5 seconds)
- High CPU usage
- Memory leaks
- Recognition accuracy decreased

#### Diagnosis
```bash
# Check system resources
htop
iotop
nvidia-smi  # If using GPU

# Check face recognition logs
grep "processing_time" /opt/face_recognition/logs/face_recognition.log | tail -20

# Check model loading
python manage.py shell
>>> from recognition.services import FaceRecognitionService
>>> service = FaceRecognitionService()
>>> # Time how long it takes
```

#### Solutions

**Slow Recognition:**
```python
# Check face image quality
def check_image_quality(image_path):
    import cv2
    import numpy as np
    
    img = cv2.imread(image_path)
    
    # Check resolution
    height, width = img.shape[:2]
    print(f"Resolution: {width}x{height}")
    
    # Check if image is blurry
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Sharpness score: {laplacian_var}")
    
    if laplacian_var < 100:
        print("Image is too blurry")
    
    return laplacian_var > 100
```

**Memory Optimization:**
```python
# settings.py optimization
FACE_RECOGNITION_SETTINGS = {
    'MAX_FACE_SIZE': (640, 640),  # Resize large images
    'BATCH_SIZE': 1,  # Process one image at a time
    'MODEL_CACHE_SIZE': 10,  # Limit cached models
    'CLEANUP_INTERVAL': 3600,  # Clean memory every hour
}

# Implement memory cleanup
import gc
import torch

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**GPU Issues:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Reset GPU if needed
sudo nvidia-smi --gpu-reset

# Install proper CUDA drivers
sudo apt update
sudo apt install nvidia-driver-470 nvidia-cuda-toolkit
```

### 4. Webhook Delivery Failures

#### Symptoms
```bash
Webhook delivery failed: Connection timeout
HTTP 500 errors in webhook logs
Webhook retry attempts exhausted
```

#### Diagnosis
```bash
# Check webhook logs
grep "webhook" /opt/face_recognition/logs/django.log

# Test webhook endpoint manually
curl -X POST https://client-webhook-url.com/endpoint \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Signature: test" \
  -d '{"test": "data"}'

# Check network connectivity
ping client-webhook-domain.com
nslookup client-webhook-domain.com
```

#### Solutions

**Timeout Issues:**
```python
# Increase webhook timeout in settings
WEBHOOK_SETTINGS = {
    'TIMEOUT': 30,  # Increase from default 10 seconds
    'RETRY_ATTEMPTS': 5,
    'RETRY_DELAY': 60,  # seconds
    'MAX_RETRY_DELAY': 3600,  # 1 hour max
}
```

**SSL Certificate Issues:**
```python
# Disable SSL verification for testing (NOT for production)
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_webhook_session():
    session = requests.Session()
    
    # Configure retries
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session
```

### 5. API Rate Limiting Issues

#### Symptoms
```bash
HTTP 429 Too Many Requests
Rate limit exceeded for client
API calls timing out
```

#### Diagnosis
```bash
# Check rate limit logs
grep "rate_limit" /opt/face_recognition/logs/django.log

# Monitor API usage
grep "API_USAGE" /opt/face_recognition/logs/django.log | tail -20

# Check Redis rate limiting keys
redis-cli keys "rate_limit:*"
redis-cli get "rate_limit:client_FR_DEMO123"
```

#### Solutions

**Increase Rate Limits:**
```python
# Update client rate limits in admin or via API
from clients.models import Client

client = Client.objects.get(client_id='FR_DEMO123')
client.rate_limit_per_hour = 2000
client.rate_limit_per_day = 20000
client.save()
```

**Optimize Rate Limiting:**
```python
# Use sliding window rate limiting
RATE_LIMITING = {
    'ALGORITHM': 'sliding_window',
    'WINDOW_SIZE': 3600,  # 1 hour
    'REDIS_KEY_PREFIX': 'rate_limit',
    'BURST_ALLOWANCE': 50,  # Allow burst traffic
}
```

### 6. File Storage Issues

#### Symptoms
```bash
No space left on device
Permission denied writing to media directory
Image upload failures
```

#### Diagnosis
```bash
# Check disk space
df -h

# Check file permissions
ls -la /opt/face_recognition/media/
ls -la /opt/face_recognition/media/uploads/

# Check file system errors
sudo dmesg | grep -i error

# Monitor disk usage
du -sh /opt/face_recognition/media/*
```

#### Solutions

**Disk Space:**
```bash
# Clean old log files
find /opt/face_recognition/logs/ -name "*.log.*" -mtime +30 -delete

# Clean old face images (older than 90 days)
find /opt/face_recognition/media/face_images/ -name "*.jpg" -mtime +90 -delete

# Setup log rotation
cat > /etc/logrotate.d/face_recognition << 'EOF'
/opt/face_recognition/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 facerecog facerecog
    postrotate
        supervisorctl signal HUP face_recognition_web
    endscript
}
EOF
```

**Permissions:**
```bash
# Fix ownership
sudo chown -R facerecog:facerecog /opt/face_recognition/media/

# Fix permissions
chmod 755 /opt/face_recognition/media/
chmod 755 /opt/face_recognition/media/uploads/
chmod 644 /opt/face_recognition/media/uploads/*
```

### 7. SSL/TLS Issues

#### Symptoms
```bash
SSL certificate expired
SSL handshake failures
Mixed content warnings
```

#### Diagnosis
```bash
# Check certificate expiry
openssl x509 -in /etc/letsencrypt/live/your-domain.com/cert.pem -text -noout | grep "Not After"

# Test SSL configuration
curl -I https://your-domain.com
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Check Certbot status
sudo certbot certificates
```

#### Solutions

**Renew Certificate:**
```bash
# Manual renewal
sudo certbot renew

# Force renewal
sudo certbot renew --force-renewal

# Check auto-renewal
sudo crontab -l | grep certbot
```

**Fix SSL Configuration:**
```nginx
# Update Nginx SSL config
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/your-domain.com/chain.pem;
}
```

## Performance Optimization

### 1. Database Optimization

```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Analyze table statistics
ANALYZE;

-- Reindex if needed
REINDEX DATABASE face_recognition_prod;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation 
FROM pg_stats 
WHERE tablename = 'recognition_faceembedding';
```

### 2. Application Optimization

```python
# Django optimization settings
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com']

# Database connection pooling
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'OPTIONS': {
            'MAX_CONNS': 20,
            'CONN_MAX_AGE': 3600,
        }
    }
}

# Cache configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {'max_connections': 50}
        }
    }
}

# Session optimization
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'
```

### 3. Monitoring Scripts

**monitor_performance.py:**
```python
#!/usr/bin/env python3
import psutil
import time
import json
from datetime import datetime

def collect_metrics():
    return {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'network_io': dict(psutil.net_io_counters()._asdict()),
        'process_count': len(psutil.pids()),
    }

def main():
    while True:
        metrics = collect_metrics()
        
        # Log to file
        with open('/opt/face_recognition/logs/performance.log', 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # Alert if CPU > 90% for 5 minutes
        if metrics['cpu_percent'] > 90:
            print(f"HIGH CPU ALERT: {metrics['cpu_percent']}%")
        
        # Alert if memory > 95%
        if metrics['memory_percent'] > 95:
            print(f"HIGH MEMORY ALERT: {metrics['memory_percent']}%")
        
        time.sleep(60)  # Collect every minute

if __name__ == '__main__':
    main()
```

## Emergency Procedures

### 1. Service Recovery

```bash
#!/bin/bash
# emergency_recovery.sh

echo "Starting emergency recovery..."

# Stop all services
sudo supervisorctl stop face_recognition:*
sudo systemctl stop nginx

# Check and fix file permissions
sudo chown -R facerecog:facerecog /opt/face_recognition/
sudo chmod -R 755 /opt/face_recognition/

# Clear problematic cache
redis-cli FLUSHDB

# Restart database if needed
sudo systemctl restart postgresql

# Start services in order
sudo systemctl start nginx
sudo supervisorctl start face_recognition:*

# Wait and check status
sleep 10
sudo supervisorctl status face_recognition:*
curl -I http://localhost:8000/api/health/

echo "Recovery complete"
```

### 2. Data Recovery

```bash
#!/bin/bash
# restore_from_backup.sh

BACKUP_FILE="$1"
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    exit 1
fi

# Stop application
sudo supervisorctl stop face_recognition:*

# Restore database
gunzip -c "$BACKUP_FILE" | sudo -u postgres pg_restore -d face_recognition_prod

# Restart application
sudo supervisorctl start face_recognition:*

echo "Database restored from $BACKUP_FILE"
```

### 3. Rollback Procedure

```bash
#!/bin/bash
# rollback.sh

# Get current version
CURRENT_VERSION=$(git rev-parse HEAD)
echo "Current version: $CURRENT_VERSION"

# Rollback to previous version
git checkout HEAD~1

# Update dependencies if needed
source venv/bin/activate
pip install -r requirements.txt

# Run migrations down (if needed)
python manage.py migrate --plan

# Restart services
sudo supervisorctl restart face_recognition:*

echo "Rolled back to previous version"
```

## Alerting & Notifications

### 1. Email Alerts

```python
# alert_system.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert(subject, message, severity='info'):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "alerts@your-company.com"
    sender_password = "app_password"
    
    recipients = {
        'critical': ['admin@company.com', 'devops@company.com'],
        'warning': ['devops@company.com'],
        'info': ['logs@company.com']
    }
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['Subject'] = f"[{severity.upper()}] Face Recognition: {subject}"
    
    body = f"""
    Alert Details:
    Severity: {severity.upper()}
    Service: Face Recognition API
    Message: {message}
    Time: {datetime.now()}
    Server: {socket.gethostname()}
    
    Please investigate immediately if this is a CRITICAL alert.
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        
        for email in recipients.get(severity, recipients['info']):
            msg['To'] = email
            server.sendmail(sender_email, email, msg.as_string())
        
        server.quit()
    except Exception as e:
        print(f"Failed to send alert: {e}")

# Usage
send_alert("Database connection failed", "PostgreSQL is not responding", "critical")
```

### 2. Slack Integration

```python
# slack_alerts.py
import requests
import json

def send_slack_alert(message, severity='info'):
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    
    colors = {
        'critical': '#ff0000',
        'warning': '#ffaa00',
        'info': '#00aa00'
    }
    
    payload = {
        'attachments': [{
            'color': colors.get(severity, '#00aa00'),
            'title': f'Face Recognition Alert ({severity.upper()})',
            'text': message,
            'fields': [
                {
                    'title': 'Severity',
                    'value': severity.upper(),
                    'short': True
                },
                {
                    'title': 'Timestamp',
                    'value': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'short': True
                }
            ]
        }]
    }
    
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code != 200:
            print(f"Slack alert failed: {response.status_code}")
    except Exception as e:
        print(f"Slack alert error: {e}")
```

## FAQ

### Q: Face recognition is returning false negatives for enrolled users
**A:** Check the following:
1. Image quality - ensure good lighting and clear images
2. Face angle - faces should be frontal (±15 degrees)
3. Similarity threshold - may need adjustment in settings
4. Model consistency - ensure same model used for enrollment and recognition

### Q: API is responding slowly (>2 seconds per request)
**A:** Optimization steps:
1. Check system resources (CPU, RAM, GPU)
2. Optimize database queries with proper indexing
3. Enable response caching for static data
4. Consider horizontal scaling with load balancer

### Q: Webhook deliveries are failing intermittently
**A:** Common solutions:
1. Increase timeout values
2. Implement exponential backoff for retries
3. Check client endpoint SSL certificates
4. Monitor network connectivity

### Q: Database is growing too large
**A:** Maintenance tasks:
1. Implement data retention policies
2. Archive old face images to cold storage
3. Regular database cleanup of expired sessions
4. Optimize face embedding storage

---

*Troubleshooting Guide Last Updated: October 9, 2025*
*Version: 2.0.0*
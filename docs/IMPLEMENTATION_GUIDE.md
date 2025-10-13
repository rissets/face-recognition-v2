# Face Recognition System - Dokumentasi Implementasi

## Daftar Isi

1. [Persiapan Environment](#persiapan-environment)
2. [Instalasi dan Setup](#instalasi-dan-setup)
3. [Konfigurasi Database](#konfigurasi-database)
4. [Deployment Options](#deployment-options)
5. [Environment Variables](#environment-variables)
6. [SSL/TLS Configuration](#ssltls-configuration)
7. [Load Balancing](#load-balancing)
8. [Monitoring Setup](#monitoring-setup)
9. [Backup Strategy](#backup-strategy)
10. [Troubleshooting](#troubleshooting)

---

## Persiapan Environment

### System Requirements

#### Minimum Requirements
- **OS**: Ubuntu 20.04 LTS atau CentOS 8+
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4 cores minimum, 8 cores recommended
- **Storage**: 100GB SSD minimum
- **Network**: 1Gbps connection

#### Production Requirements  
- **OS**: Ubuntu 22.04 LTS atau RHEL 9+
- **RAM**: 32GB minimum, 64GB recommended
- **CPU**: 16 cores minimum, 32 cores recommended  
- **Storage**: 500GB NVMe SSD minimum
- **GPU**: Optional, NVIDIA RTX 3080+ untuk ML acceleration
- **Network**: 10Gbps connection

### Dependencies

#### System Packages

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    python3.11 python3.11-dev python3.11-venv \
    postgresql-14 postgresql-contrib postgresql-14-pgvector \
    redis-server \
    nginx \
    docker.io docker-compose \
    git curl wget \
    build-essential cmake \
    libopencv-dev \
    ffmpeg \
    supervisor

# CentOS/RHEL
sudo dnf install -y \
    python3.11 python3.11-devel \
    postgresql14-server postgresql14-contrib \
    redis \
    nginx \
    docker docker-compose \
    git curl wget \
    gcc gcc-c++ make cmake \
    opencv-devel \
    ffmpeg \
    supervisor
```

#### Python Dependencies

```bash
# Create virtual environment
python3.11 -m venv /opt/face_recognition/env
source /opt/face_recognition/env/bin/activate

# Install core dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

---

## Instalasi dan Setup

### 1. Clone Repository

```bash
# Production setup
sudo mkdir -p /opt/face_recognition
sudo chown $USER:$USER /opt/face_recognition
cd /opt/face_recognition

git clone https://github.com/your-org/face-recognition-v2.git .
git checkout main  # atau release branch

# Set permissions
sudo chown -R www-data:www-data /opt/face_recognition
sudo chmod -R 755 /opt/face_recognition
```

### 2. Environment Setup

```bash
# Copy environment template
cp face_recognition_app/.env.example face_recognition_app/.env

# Edit environment variables
nano face_recognition_app/.env
```

### 3. Python Environment

```bash
# Create virtual environment
python3.11 -m venv env
source env/bin/activate

# Install dependencies
cd face_recognition_app
pip install -r requirements.txt

# Install additional packages untuk production
pip install gunicorn gevent psycopg2-binary
```

### 4. Database Setup

```bash
# PostgreSQL setup
sudo -u postgres createdb face_recognition_prod
sudo -u postgres createuser face_recognition_user

# Set password
sudo -u postgres psql
postgres=# ALTER USER face_recognition_user PASSWORD 'secure_password_here';
postgres=# GRANT ALL PRIVILEGES ON DATABASE face_recognition_prod TO face_recognition_user;
postgres=# \q

# Install pgvector extension
sudo -u postgres psql face_recognition_prod
face_recognition_prod=# CREATE EXTENSION IF NOT EXISTS vector;
face_recognition_prod=# \q
```

### 5. Django Setup

```bash
cd /opt/face_recognition/face_recognition_app

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic --noinput

# Load initial data (optional)
python manage.py loaddata fixtures/initial_data.json
```

### 6. Redis Setup

```bash
# Configure Redis
sudo nano /etc/redis/redis.conf

# Key configurations:
# bind 127.0.0.1
# port 6379
# maxmemory 2gb
# maxmemory-policy allkeys-lru
# save 900 1
# save 300 10

# Restart Redis
sudo systemctl restart redis
sudo systemctl enable redis
```

---

## Konfigurasi Database

### PostgreSQL Configuration

#### 1. Performance Tuning

```sql
-- /etc/postgresql/14/main/postgresql.conf

# Memory settings
shared_buffers = 4GB                    # 25% of total RAM
effective_cache_size = 12GB             # 75% of total RAM  
work_mem = 256MB                        # Per connection
maintenance_work_mem = 1GB

# Connection settings
max_connections = 200
superuser_reserved_connections = 3

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 64MB
checkpoint_segments = 64

# Query planner
random_page_cost = 1.1                  # For SSD storage
effective_io_concurrency = 200          # For SSD storage

# Logging
log_statement = 'all'                   # For debugging (disable in production)
log_min_duration_statement = 1000       # Log slow queries
```

#### 2. pgvector Optimization

```sql
-- Connect to database
\c face_recognition_prod

-- Create index untuk vector similarity search
CREATE INDEX CONCURRENTLY face_embeddings_vector_idx 
ON face_embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 1000);

-- Analyze table untuk optimal query planning
ANALYZE face_embeddings;

-- Set vector search parameters
SET ivfflat.probes = 10;  -- Adjust based on accuracy vs speed needs
```

#### 3. Database Maintenance

```bash
#!/bin/bash
# /opt/face_recognition/scripts/db_maintenance.sh

# Daily maintenance script
psql -U face_recognition_user -d face_recognition_prod -c "
-- Vacuum and analyze tables
VACUUM ANALYZE;

-- Reindex vector indexes if needed
REINDEX INDEX CONCURRENTLY face_embeddings_vector_idx;

-- Update statistics
ANALYZE face_embeddings;
ANALYZE auth_service_faceenrollment;
ANALYZE clients_clientuser;
"

# Cleanup old data
python /opt/face_recognition/face_recognition_app/manage.py cleanup_old_sessions
python /opt/face_recognition/face_recognition_app/manage.py cleanup_expired_tokens
```

### ChromaDB Configuration

```python
# face_recognition_app/face_app/settings.py

CHROMA_SETTINGS = {
    'persist_directory': '/opt/face_recognition/chroma_data',
    'collection_name': 'face_embeddings',
    'distance_metric': 'cosine',
    'embedding_dimension': 512,
    'batch_size': 1000,
    'index_params': {
        'M': 16,
        'efConstruction': 200,
        'efSearch': 100
    }
}

# Backup configuration
CHROMA_BACKUP = {
    'backup_directory': '/opt/backups/chroma',
    'retention_days': 30,
    'compress': True
}
```

---

## Deployment Options

### 1. Docker Deployment (Recommended)

#### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  web:
    build:
      context: ./face_recognition_app
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    volumes:
      - static_volume:/app/staticfiles
      - media_volume:/app/media
      - logs_volume:/app/logs
    environment:
      - DJANGO_SETTINGS_MODULE=face_app.settings.production
    depends_on:
      - postgres
      - redis
    networks:
      - face_recognition_network

  nginx:
    build:
      context: ./nginx
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - static_volume:/app/staticfiles:ro
      - media_volume:/app/media:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - logs_volume:/var/log/nginx
    depends_on:
      - web
    networks:
      - face_recognition_network

  postgres:
    image: pgvector/pgvector:pg14
    restart: unless-stopped
    environment:
      - POSTGRES_DB=face_recognition_prod
      - POSTGRES_USER=face_recognition_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - face_recognition_network

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - face_recognition_network

  celery:
    build:
      context: ./face_recognition_app
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    command: celery -A face_app worker -l info -Q default,recognition,analytics
    volumes:
      - logs_volume:/app/logs
    environment:
      - DJANGO_SETTINGS_MODULE=face_app.settings.production
    depends_on:
      - postgres
      - redis
    networks:
      - face_recognition_network

  celery_beat:
    build:
      context: ./face_recognition_app  
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    command: celery -A face_app beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
    volumes:
      - logs_volume:/app/logs
    environment:
      - DJANGO_SETTINGS_MODULE=face_app.settings.production
    depends_on:
      - postgres
      - redis
    networks:
      - face_recognition_network

volumes:
  postgres_data:
  redis_data:
  static_volume:
  media_volume:
  logs_volume:

networks:
  face_recognition_network:
    driver: bridge
```

#### Production Dockerfile

```dockerfile
# face_recognition_app/Dockerfile.prod
FROM python:3.11-slim as builder

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Production image
FROM python:3.11-slim

# Create app user
RUN groupadd -r app && useradd -r -g app app

# System dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libopencv-imgproc4.5 \
    libopencv-highgui4.5 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY --from=builder /app/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache /wheels/*

# Setup application
WORKDIR /app
COPY . /app/
RUN python manage.py collectstatic --noinput

# Security settings
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "gevent", "face_app.wsgi:application"]
```

### 2. Kubernetes Deployment

#### Namespace dan ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: face-recognition

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: face-recognition-config
  namespace: face-recognition
data:
  DJANGO_SETTINGS_MODULE: "face_app.settings.production"
  REDIS_URL: "redis://redis-service:6379"
  DATABASE_URL: "postgresql://user:password@postgres-service:5432/face_recognition_prod"
```

#### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-recognition-web
  namespace: face-recognition
spec:
  replicas: 3
  selector:
    matchLabels:
      app: face-recognition-web
  template:
    metadata:
      labels:
        app: face-recognition-web
    spec:
      containers:
      - name: web
        image: your-registry/face-recognition:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: face-recognition-config
        - secretRef:
            name: face-recognition-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi" 
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health/
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready/
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: face-recognition-service
  namespace: face-recognition
spec:
  selector:
    app: face-recognition-web
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### 3. Bare Metal Deployment

#### Systemd Services

```ini
# /etc/systemd/system/face-recognition.service
[Unit]
Description=Face Recognition Django App
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/opt/face_recognition/face_recognition_app
Environment=PATH=/opt/face_recognition/env/bin
ExecStart=/opt/face_recognition/env/bin/gunicorn \
    --bind 127.0.0.1:8000 \
    --workers 4 \
    --worker-class gevent \
    --max-requests 1000 \
    --timeout 30 \
    --keep-alive 2 \
    --log-level info \
    --access-logfile /var/log/face_recognition/access.log \
    --error-logfile /var/log/face_recognition/error.log \
    face_app.wsgi:application
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/face-recognition-celery.service
[Unit]
Description=Face Recognition Celery Worker
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/opt/face_recognition/face_recognition_app
Environment=PATH=/opt/face_recognition/env/bin
ExecStart=/opt/face_recognition/env/bin/celery \
    -A face_app worker \
    --loglevel=info \
    --logfile=/var/log/face_recognition/celery.log \
    --queues=default,recognition,analytics \
    --concurrency=4
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Environment Variables

### Production Environment

```bash
# face_recognition_app/.env.production

# Basic Django settings
DEBUG=False
SECRET_KEY=your-super-secret-key-here-minimum-50-chars
ALLOWED_HOSTS=your-domain.com,api.your-domain.com
DJANGO_SETTINGS_MODULE=face_app.settings.production

# Database configuration
DATABASE_URL=postgresql://user:password@postgres:5432/face_recognition_prod
DB_NAME=face_recognition_prod
DB_USER=face_recognition_user
DB_PASSWORD=secure_password_here
DB_HOST=postgres
DB_PORT=5432

# Redis configuration
REDIS_URL=redis://redis:6379/0
CACHE_URL=redis://redis:6379/1
CELERY_BROKER_URL=redis://redis:6379/2

# ChromaDB configuration
CHROMA_HOST=chroma
CHROMA_PORT=8000
CHROMA_PERSIST_DIRECTORY=/app/chroma_data

# Security settings
SECURE_SSL_REDIRECT=True
SECURE_PROXY_SSL_HEADER=HTTP_X_FORWARDED_PROTO,https
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True
SECURE_BROWSER_XSS_FILTER=True
SECURE_CONTENT_TYPE_NOSNIFF=True

# CORS settings (untuk frontend)
CORS_ALLOWED_ORIGINS=https://your-frontend.com,https://app.your-domain.com
CORS_ALLOW_CREDENTIALS=True

# Email configuration
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.your-provider.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@company.com
EMAIL_HOST_PASSWORD=your-email-password

# Media dan static files
MEDIA_ROOT=/app/media
STATIC_ROOT=/app/staticfiles
MEDIA_URL=/media/
STATIC_URL=/static/

# Face recognition settings
FACE_RECOGNITION_MODEL=ArcFace
FACE_DETECTION_CONFIDENCE=0.7
LIVENESS_DETECTION_ENABLED=True
ANTI_SPOOFING_ENABLED=True

# Performance settings
GUNICORN_WORKERS=4
CELERY_WORKER_CONCURRENCY=4
MAX_UPLOAD_SIZE=10485760  # 10MB

# Monitoring
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project
PROMETHEUS_METRICS_ENABLED=True
LOG_LEVEL=INFO

# External services
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_STORAGE_BUCKET_NAME=your-s3-bucket
AWS_S3_REGION_NAME=us-east-1

# Feature flags
ENABLE_WEBHOOKS=True
ENABLE_ANALYTICS=True
ENABLE_API_RATE_LIMITING=True
ENABLE_AUDIT_LOGGING=True
```

### Development Environment

```bash
# face_recognition_app/.env.development

DEBUG=True
SECRET_KEY=dev-secret-key-not-for-production
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0
DJANGO_SETTINGS_MODULE=face_app.settings.development

# Use SQLite untuk development
DATABASE_URL=sqlite:///db.sqlite3

# Local Redis (optional)
REDIS_URL=redis://localhost:6379/0

# Disable security features untuk development
SECURE_SSL_REDIRECT=False
SESSION_COOKIE_SECURE=False
CSRF_COOKIE_SECURE=False

# Local file storage
MEDIA_ROOT=./media
STATIC_ROOT=./staticfiles

# Disable external services
SENTRY_DSN=
ENABLE_WEBHOOKS=False
ENABLE_API_RATE_LIMITING=False

LOG_LEVEL=DEBUG
```

---

## SSL/TLS Configuration

### 1. Nginx SSL Configuration

```nginx
# /etc/nginx/sites-available/face-recognition
server {
    listen 80;
    server_name your-domain.com api.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com api.your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    
    # SSL Security
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";

    # Client settings
    client_max_body_size 10M;
    client_body_timeout 30s;
    client_header_timeout 30s;

    # Proxy settings
    proxy_connect_timeout 30s;
    proxy_send_timeout 30s;
    proxy_read_timeout 30s;

    # Static files
    location /static/ {
        alias /app/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location /media/ {
        alias /app/media/;
        expires 7d;
        add_header Cache-Control "public";
    }

    # API endpoints
    location /api/ {
        proxy_pass http://web:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
        
        # CORS headers
        if ($request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS, PUT, DELETE';
            add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization';
            add_header 'Access-Control-Max-Age' 1728000;
            add_header 'Content-Type' 'text/plain; charset=utf-8';
            add_header 'Content-Length' 0;
            return 204;
        }
    }

    # WebSocket endpoints
    location /ws/ {
        proxy_pass http://web:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check
    location /health/ {
        proxy_pass http://web:8000;
        access_log off;
    }
}

# Rate limiting configuration
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
}
```

### 2. Let's Encrypt SSL Setup

```bash
#!/bin/bash
# scripts/setup_ssl.sh

# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com -d api.your-domain.com

# Auto-renewal
sudo crontab -e
# Add line: 0 12 * * * /usr/bin/certbot renew --quiet

# Test renewal
sudo certbot renew --dry-run
```

### 3. Custom SSL Certificate

```bash
# Generate private key
openssl genrsa -out /etc/nginx/ssl/privkey.pem 2048

# Generate certificate signing request
openssl req -new -key /etc/nginx/ssl/privkey.pem -out /etc/nginx/ssl/cert.csr

# Generate self-signed certificate (untuk development)
openssl x509 -req -days 365 -in /etc/nginx/ssl/cert.csr -signkey /etc/nginx/ssl/privkey.pem -out /etc/nginx/ssl/fullchain.pem

# Set proper permissions
chmod 600 /etc/nginx/ssl/privkey.pem
chmod 644 /etc/nginx/ssl/fullchain.pem
```

---

## Load Balancing

### 1. Nginx Load Balancer

```nginx
# /etc/nginx/nginx.conf
upstream face_recognition_backend {
    least_conn;
    server web1:8000 max_fails=3 fail_timeout=30s;
    server web2:8000 max_fails=3 fail_timeout=30s;
    server web3:8000 max_fails=3 fail_timeout=30s;
    
    # Health checks (nginx plus feature)
    # health_check interval=30s fails=3 passes=2;
}

server {
    listen 443 ssl http2;
    server_name api.your-domain.com;
    
    location /api/ {
        proxy_pass http://face_recognition_backend;
        
        # Connection settings
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Session persistence (if needed)
        # ip_hash;
    }
}
```

### 2. HAProxy Configuration

```haproxy
# /etc/haproxy/haproxy.cfg
global
    daemon
    maxconn 4096
    log stdout local0
    
defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    
frontend face_recognition_frontend
    bind *:443 ssl crt /etc/ssl/certs/your-domain.pem
    redirect scheme https if !{ ssl_fc }
    
    # ACL untuk routing
    acl is_api path_beg /api/
    acl is_websocket hdr(upgrade) -i websocket
    
    # Route ke backend
    use_backend face_recognition_api if is_api
    use_backend face_recognition_ws if is_websocket
    default_backend face_recognition_web

backend face_recognition_api
    balance roundrobin
    option httpchk GET /health/
    
    server web1 10.0.1.10:8000 check
    server web2 10.0.1.11:8000 check
    server web3 10.0.1.12:8000 check

backend face_recognition_ws
    balance source
    option httpchk GET /health/
    
    server web1 10.0.1.10:8000 check
    server web2 10.0.1.11:8000 check
    server web3 10.0.1.12:8000 check

backend face_recognition_web
    balance roundrobin
    option httpchk GET /health/
    
    server web1 10.0.1.10:8000 check
    server web2 10.0.1.11:8000 check
    server web3 10.0.1.12:8000 check
```

### 3. Database Load Balancing

```python
# face_app/settings/production.py

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'face_recognition_prod',
        'USER': 'face_recognition_user',
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': 'postgres-master',
        'PORT': '5432',
        'OPTIONS': {
            'sslmode': 'require',
        }
    },
    'replica': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'face_recognition_prod',
        'USER': 'face_recognition_readonly',
        'PASSWORD': os.getenv('DB_REPLICA_PASSWORD'),
        'HOST': 'postgres-replica',
        'PORT': '5432',
        'OPTIONS': {
            'sslmode': 'require',
        }
    }
}

# Database router untuk read/write splitting
DATABASE_ROUTERS = ['core.routers.DatabaseRouter']

# Redis clustering
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': [
            'redis://redis-node1:6379/0',
            'redis://redis-node2:6379/0',  
            'redis://redis-node3:6379/0',
        ],
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.ShardClient',
        }
    }
}
```

---

## Monitoring Setup

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'face-recognition-django'
    static_configs:
      - targets: ['web:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'face-recognition-nginx'
    static_configs:
      - targets: ['nginx:9113']
    scrape_interval: 30s
    
  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 30s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 2. Django Metrics Integration

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
from django.conf import settings

# Custom metrics
FACE_RECOGNITION_REQUESTS = Counter(
    'face_recognition_requests_total',
    'Total face recognition requests',
    ['client_id', 'operation', 'status']
)

FACE_RECOGNITION_DURATION = Histogram(
    'face_recognition_duration_seconds',
    'Face recognition processing time',
    ['operation']
)

ACTIVE_SESSIONS = Gauge(
    'face_recognition_active_sessions',
    'Number of active recognition sessions'
)

EMBEDDING_QUALITY = Histogram(
    'face_embedding_quality_score',
    'Face embedding quality scores'
)

# Middleware untuk metrics
class MetricsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        
        response = self.get_response(request)
        
        # Record request duration
        duration = time.time() - start_time
        FACE_RECOGNITION_DURATION.observe(duration)
        
        return response
```

### 3. Alert Rules

```yaml
# alert_rules.yml
groups:
- name: face_recognition_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(face_recognition_requests_total{status="error"}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} requests/second"
      
  - alert: DatabaseConnectionFailed
    expr: up{job="postgresql"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failed"
      description: "PostgreSQL is down"
      
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, face_recognition_duration_seconds) > 2.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response times"
      description: "95th percentile response time is {{ $value }}s"
      
  - alert: LowEmbeddingQuality
    expr: histogram_quantile(0.5, face_embedding_quality_score) < 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Low embedding quality detected"
      description: "Median embedding quality is {{ $value }}"
```

### 4. Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Face Recognition System",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(face_recognition_requests_total[5m])",
            "legendFormat": "{{operation}} - {{status}}"
          }
        ]
      },
      {
        "title": "Response Times",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, face_recognition_duration_seconds)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.5, face_recognition_duration_seconds)",
            "legendFormat": "Median"
          }
        ]
      },
      {
        "title": "Active Sessions",
        "type": "singlestat",
        "targets": [
          {
            "expr": "face_recognition_active_sessions"
          }
        ]
      },
      {
        "title": "Database Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(postgresql_queries_total[5m])",
            "legendFormat": "Queries/sec"
          }
        ]
      }
    ]
  }
}
```

---

## Backup Strategy

### 1. Database Backup Script

```bash
#!/bin/bash
# scripts/backup_database.sh

set -e

# Configuration
BACKUP_DIR="/opt/backups/postgres"
RETENTION_DAYS=30
DB_NAME="face_recognition_prod"
DB_USER="face_recognition_user"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create backup directory
mkdir -p $BACKUP_DIR

# Full database backup
pg_dump -h localhost -U $DB_USER -d $DB_NAME -F c -b -v -f "$BACKUP_DIR/full_backup_$TIMESTAMP.dump"

# Schema-only backup  
pg_dump -h localhost -U $DB_USER -d $DB_NAME -s -f "$BACKUP_DIR/schema_backup_$TIMESTAMP.sql"

# Compress backups
gzip "$BACKUP_DIR/full_backup_$TIMESTAMP.dump"
gzip "$BACKUP_DIR/schema_backup_$TIMESTAMP.sql"

# Upload to S3 (optional)
if [ -n "$AWS_S3_BACKUP_BUCKET" ]; then
    aws s3 cp "$BACKUP_DIR/full_backup_$TIMESTAMP.dump.gz" "s3://$AWS_S3_BACKUP_BUCKET/postgres/"
    aws s3 cp "$BACKUP_DIR/schema_backup_$TIMESTAMP.sql.gz" "s3://$AWS_S3_BACKUP_BUCKET/postgres/"
fi

# Cleanup old backups
find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $TIMESTAMP"
```

### 2. ChromaDB Backup

```python
# scripts/backup_chroma.py
import os
import shutil
import tarfile
from datetime import datetime
from face_app.settings import CHROMA_SETTINGS

def backup_chroma():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = "/opt/backups/chroma"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Source directory
    source_dir = CHROMA_SETTINGS['persist_directory']
    
    # Create tar archive
    backup_file = f"{backup_dir}/chroma_backup_{timestamp}.tar.gz"
    
    with tarfile.open(backup_file, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    
    print(f"ChromaDB backup created: {backup_file}")
    
    # Upload to S3 if configured
    if os.getenv('AWS_S3_BACKUP_BUCKET'):
        import boto3
        s3 = boto3.client('s3')
        s3.upload_file(
            backup_file,
            os.getenv('AWS_S3_BACKUP_BUCKET'),
            f"chroma/chroma_backup_{timestamp}.tar.gz"
        )
    
    return backup_file

if __name__ == "__main__":
    backup_chroma()
```

### 3. Automated Backup with Cron

```bash
# /etc/crontab

# Database backup - daily at 2 AM
0 2 * * * root /opt/face_recognition/scripts/backup_database.sh >> /var/log/backup.log 2>&1

# ChromaDB backup - daily at 3 AM  
0 3 * * * root /opt/face_recognition/env/bin/python /opt/face_recognition/scripts/backup_chroma.py >> /var/log/backup.log 2>&1

# Configuration backup - weekly
0 4 * * 0 root /opt/face_recognition/scripts/backup_config.sh >> /var/log/backup.log 2>&1

# Log rotation
0 5 * * * root /opt/face_recognition/scripts/rotate_logs.sh >> /var/log/backup.log 2>&1
```

### 4. Disaster Recovery Procedures

```bash
#!/bin/bash
# scripts/restore_database.sh

set -e

BACKUP_FILE=$1
DB_NAME="face_recognition_prod"
DB_USER="face_recognition_user"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
systemctl stop face-recognition
systemctl stop face-recognition-celery

# Drop and recreate database
sudo -u postgres dropdb $DB_NAME
sudo -u postgres createdb $DB_NAME
sudo -u postgres psql $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Restore from backup
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | pg_restore -h localhost -U $DB_USER -d $DB_NAME -v
else
    pg_restore -h localhost -U $DB_USER -d $DB_NAME -v $BACKUP_FILE
fi

# Restart application
systemctl start face-recognition
systemctl start face-recognition-celery

echo "Database restored successfully"
```

---

## Troubleshooting

### 1. Common Issues

#### Database Connection Issues

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connections
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"

# Check logs
sudo tail -f /var/log/postgresql/postgresql-14-main.log

# Connection troubleshooting
# Test connection manually
psql -h localhost -U face_recognition_user -d face_recognition_prod -c "SELECT 1;"
```

#### Redis Connection Issues

```bash
# Check Redis status
sudo systemctl status redis

# Test connection
redis-cli ping

# Check memory usage
redis-cli info memory

# Monitor commands
redis-cli monitor
```

#### High CPU/Memory Usage

```bash
# Monitor system resources
htop
iotop
nethogs

# Django specific monitoring
# Add to settings.py for debugging
LOGGING = {
    'version': 1,
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': '/var/log/face_recognition/debug.log',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}
```

### 2. Performance Debugging

```python
# core/middleware.py - Performance profiling middleware
import time
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class PerformanceMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        
        response = self.get_response(request)
        
        duration = time.time() - start_time
        
        if duration > 1.0:  # Log slow requests
            logger.warning(
                f"Slow request: {request.method} {request.path} "
                f"took {duration:.2f}s"
            )
        
        return response

# Database query debugging
if settings.DEBUG:
    LOGGING['loggers']['django.db.backends'] = {
        'handlers': ['console'],
        'level': 'DEBUG',
    }
```

### 3. Health Check Scripts

```bash
#!/bin/bash
# scripts/health_check.sh

set -e

# Colors untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Face Recognition System Health Check"
echo "=================================="

# Check web service
echo -n "Web Service: "
if curl -f -s http://localhost:8000/health/ > /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
fi

# Check database
echo -n "Database: "
if sudo -u postgres psql face_recognition_prod -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
fi

# Check Redis
echo -n "Redis: "
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
fi

# Check disk space
echo -n "Disk Space: "
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 80 ]; then
    echo -e "${GREEN}OK (${DISK_USAGE}% used)${NC}"
elif [ $DISK_USAGE -lt 90 ]; then
    echo -e "${YELLOW}WARNING (${DISK_USAGE}% used)${NC}"
else
    echo -e "${RED}CRITICAL (${DISK_USAGE}% used)${NC}"
fi

# Check memory usage
echo -n "Memory Usage: "
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100}')
if [ $MEMORY_USAGE -lt 80 ]; then
    echo -e "${GREEN}OK (${MEMORY_USAGE}% used)${NC}"
elif [ $MEMORY_USAGE -lt 90 ]; then
    echo -e "${YELLOW}WARNING (${MEMORY_USAGE}% used)${NC}"
else
    echo -e "${RED}CRITICAL (${MEMORY_USAGE}% used)${NC}"
fi

# Check log files untuk errors
echo -n "Recent Errors: "
ERROR_COUNT=$(grep -c "ERROR" /var/log/face_recognition/*.log 2>/dev/null | awk -F: '{sum += $2} END {print sum}' || echo 0)
if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "${GREEN}None${NC}"
elif [ $ERROR_COUNT -lt 10 ]; then
    echo -e "${YELLOW}${ERROR_COUNT} errors found${NC}"
else
    echo -e "${RED}${ERROR_COUNT} errors found${NC}"
fi

echo "Health check completed."
```

### 4. Log Analysis Tools

```bash
#!/bin/bash
# scripts/analyze_logs.sh

# Analyze error patterns
echo "Most common errors:"
grep "ERROR" /var/log/face_recognition/*.log | \
    awk -F'ERROR' '{print $2}' | \
    sort | uniq -c | sort -nr | head -10

# Analyze performance
echo -e "\nSlowest endpoints:"
grep "took" /var/log/face_recognition/*.log | \
    awk '{print $(NF-1), $0}' | \
    sort -nr | head -10

# Analyze client usage
echo -e "\nTop clients by request count:"
grep "api_key" /var/log/face_recognition/*.log | \
    awk '{print $8}' | sort | uniq -c | sort -nr | head -10
```

Dokumentasi implementasi ini memberikan panduan lengkap untuk melakukan deployment dan maintenance sistem Face Recognition di berbagai environment production.
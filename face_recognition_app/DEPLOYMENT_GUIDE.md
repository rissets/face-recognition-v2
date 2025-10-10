# Deployment Guide - Face Recognition Service

## Overview

Panduan lengkap untuk deployment Face Recognition Third-Party Service di berbagai environment (development, staging, dan production).

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.4GHz
- RAM: 8GB 
- Storage: 50GB SSD
- GPU: Optional (NVIDIA dengan CUDA support untuk performance optimal)
- OS: Ubuntu 20.04+ / CentOS 8+ / macOS 10.15+

**Recommended for Production:**
- CPU: 8+ cores, 3.0GHz
- RAM: 16GB+
- Storage: 100GB+ SSD
- GPU: NVIDIA RTX series dengan 8GB+ VRAM
- Load Balancer: Nginx/HAProxy
- CDN: CloudFlare/AWS CloudFront

### Software Dependencies

```bash
# Base dependencies
Python 3.11+
PostgreSQL 13+
Redis 6+
Nginx 1.20+

# Python packages (dari requirements.txt)
Django==5.2.7
djangorestframework==3.15.1
psycopg2-binary==2.9.7
redis==4.6.0
celery==5.3.1
gunicorn==21.2.0
```

## Environment Setup

### 1. Development Environment

#### Menggunakan Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: face_recognition_dev
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: dev_password_123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/code
      - media_files:/code/media
    ports:
      - "8000:8000"
    environment:
      - DEBUG=True
      - DATABASE_URL=postgresql://postgres:dev_password_123@db:5432/face_recognition_dev
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  celery:
    build: .
    command: celery -A face_recognition_app worker -l info
    volumes:
      - .:/code
    environment:
      - DATABASE_URL=postgresql://postgres:dev_password_123@db:5432/face_recognition_dev
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
  redis_data:
  media_files:
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Install Python dependencies
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /code/

# Create media directories
RUN mkdir -p /code/media/uploads /code/media/face_images

# Set environment variables
ENV PYTHONPATH=/code
ENV DJANGO_SETTINGS_MODULE=face_recognition_app.settings

# Expose port
EXPOSE 8000

# Run migrations and start server
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]
```

#### Setup Commands

```bash
# Clone repository
git clone <repository-url>
cd face_recognition_app

# Setup Docker environment
docker-compose up --build -d

# Run migrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Load demo data
docker-compose exec web python manage.py shell < demo_setup.py
```

### 2. Production Environment

#### Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3.11 python3.11-venv python3-pip \
    postgresql postgresql-contrib redis-server nginx \
    build-essential libpq-dev supervisor git

# Create application user
sudo useradd -m -s /bin/bash facerecog
sudo usermod -aG sudo facerecog

# Setup application directory
sudo mkdir -p /opt/face_recognition
sudo chown facerecog:facerecog /opt/face_recognition
```

#### Database Setup

```bash
# PostgreSQL configuration
sudo -u postgres psql

CREATE DATABASE face_recognition_prod;
CREATE USER face_app WITH PASSWORD 'secure_production_password';
GRANT ALL PRIVILEGES ON DATABASE face_recognition_prod TO face_app;
ALTER USER face_app CREATEDB;
\q

# Configure PostgreSQL
sudo nano /etc/postgresql/13/main/postgresql.conf
# Uncomment dan edit:
# listen_addresses = 'localhost'
# shared_buffers = 256MB
# effective_cache_size = 1GB

sudo nano /etc/postgresql/13/main/pg_hba.conf
# Tambahkan:
# local   face_recognition_prod   face_app                md5

sudo systemctl restart postgresql
```

#### Application Deployment

```bash
# Switch to app user
sudo su - facerecog

# Clone and setup application
cd /opt/face_recognition
git clone <repository-url> .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cat > .env << EOF
DEBUG=False
SECRET_KEY=your_super_secret_production_key_here
DATABASE_URL=postgresql://face_app:secure_production_password@localhost:5432/face_recognition_prod
REDIS_URL=redis://localhost:6379/0
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
CORS_ALLOWED_ORIGINS=https://your-domain.com,https://www.your-domain.com

# Storage settings
MEDIA_ROOT=/opt/face_recognition/media
STATIC_ROOT=/opt/face_recognition/staticfiles

# Face recognition settings
FACE_RECOGNITION_MODEL_PATH=/opt/face_recognition/models/
EMBEDDING_DIMENSION=512
SIMILARITY_THRESHOLD=0.8

# Security settings
SECURE_SSL_REDIRECT=True
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=True
SECURE_HSTS_PRELOAD=True
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True

# Email settings (optional)
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.your-provider.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@domain.com
EMAIL_HOST_PASSWORD=your-email-password

# Monitoring (optional)
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
EOF

# Create required directories
mkdir -p media/uploads media/face_images staticfiles logs models

# Run migrations and collect static files
python manage.py migrate
python manage.py collectstatic --noinput

# Create superuser
python manage.py createsuperuser
```

#### Gunicorn Configuration

```bash
# Create Gunicorn config
cat > gunicorn.conf.py << 'EOF'
import multiprocessing

bind = "127.0.0.1:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 300
keepalive = 5

# Logging
accesslog = "/opt/face_recognition/logs/gunicorn_access.log"
errorlog = "/opt/face_recognition/logs/gunicorn_error.log"
loglevel = "info"

# Process naming
proc_name = "face_recognition_app"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
preload_app = True
worker_tmp_dir = "/dev/shm"
EOF
```

#### Supervisor Configuration

```bash
# Create supervisor config
sudo cat > /etc/supervisor/conf.d/face_recognition.conf << 'EOF'
[group:face_recognition]
programs=face_recognition_web,face_recognition_celery

[program:face_recognition_web]
command=/opt/face_recognition/venv/bin/gunicorn face_recognition_app.wsgi:application -c /opt/face_recognition/gunicorn.conf.py
directory=/opt/face_recognition
user=facerecog
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/face_recognition/logs/supervisor_web.log
environment=PATH="/opt/face_recognition/venv/bin"

[program:face_recognition_celery]
command=/opt/face_recognition/venv/bin/celery -A face_recognition_app worker --loglevel=info --concurrency=4
directory=/opt/face_recognition
user=facerecog
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/face_recognition/logs/supervisor_celery.log
environment=PATH="/opt/face_recognition/venv/bin"
EOF

# Start services
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start face_recognition:*
```

#### Nginx Configuration

```bash
# Create Nginx config
sudo cat > /etc/nginx/sites-available/face_recognition << 'EOF'
upstream face_recognition_app {
    server 127.0.0.1:8000 fail_timeout=0;
}

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_dhparam /etc/nginx/dhparam.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # File upload limits
    client_max_body_size 10M;

    # Timeouts
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;

    # Static files
    location /static/ {
        alias /opt/face_recognition/staticfiles/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        gzip_static on;
    }

    location /media/ {
        alias /opt/face_recognition/media/;
        expires 1h;
        add_header Cache-Control "private";
    }

    # API endpoints
    location /api/ {
        proxy_pass http://face_recognition_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }

    # Admin interface
    location /admin/ {
        proxy_pass http://face_recognition_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        
        # Admin rate limiting
        limit_req zone=admin burst=10 nodelay;
    }

    # Main application
    location / {
        proxy_pass http://face_recognition_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
    }
}

# Rate limiting zones
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=admin:10m rate=30r/m;
}
EOF

# Enable site and restart Nginx
sudo ln -s /etc/nginx/sites-available/face_recognition /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### SSL Certificate Setup

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## AWS Deployment

### Infrastructure Setup

**terraform/main.tf:**
```hcl
provider "aws" {
  region = var.aws_region
}

# VPC
resource "aws_vpc" "face_recognition_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "face-recognition-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "face_recognition_igw" {
  vpc_id = aws_vpc.face_recognition_vpc.id

  tags = {
    Name = "face-recognition-igw"
  }
}

# Subnets
resource "aws_subnet" "face_recognition_public_1" {
  vpc_id                  = aws_vpc.face_recognition_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "face-recognition-public-1"
  }
}

resource "aws_subnet" "face_recognition_public_2" {
  vpc_id                  = aws_vpc.face_recognition_vpc.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true

  tags = {
    Name = "face-recognition-public-2"
  }
}

resource "aws_subnet" "face_recognition_private_1" {
  vpc_id            = aws_vpc.face_recognition_vpc.id
  cidr_block        = "10.0.3.0/24"
  availability_zone = "${var.aws_region}a"

  tags = {
    Name = "face-recognition-private-1"
  }
}

resource "aws_subnet" "face_recognition_private_2" {
  vpc_id            = aws_vpc.face_recognition_vpc.id
  cidr_block        = "10.0.4.0/24"
  availability_zone = "${var.aws_region}b"

  tags = {
    Name = "face-recognition-private-2"
  }
}

# RDS Subnet Group
resource "aws_db_subnet_group" "face_recognition_db_subnet_group" {
  name       = "face-recognition-db-subnet-group"
  subnet_ids = [aws_subnet.face_recognition_private_1.id, aws_subnet.face_recognition_private_2.id]

  tags = {
    Name = "face-recognition-db-subnet-group"
  }
}

# Security Groups
resource "aws_security_group" "face_recognition_web_sg" {
  name_prefix = "face-recognition-web-"
  vpc_id      = aws_vpc.face_recognition_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "face-recognition-web-sg"
  }
}

resource "aws_security_group" "face_recognition_db_sg" {
  name_prefix = "face-recognition-db-"
  vpc_id      = aws_vpc.face_recognition_vpc.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.face_recognition_web_sg.id]
  }

  tags = {
    Name = "face-recognition-db-sg"
  }
}

# RDS Instance
resource "aws_db_instance" "face_recognition_db" {
  identifier             = "face-recognition-db"
  engine                 = "postgres"
  engine_version         = "15.3"
  instance_class         = "db.t3.medium"
  allocated_storage      = 100
  storage_type           = "gp2"
  storage_encrypted      = true

  db_name  = "facerecognition"
  username = "faceapp"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.face_recognition_db_subnet_group.name
  vpc_security_group_ids = [aws_security_group.face_recognition_db_sg.id]

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "face-recognition-final-snapshot"

  tags = {
    Name = "face-recognition-db"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "face_recognition_redis_subnet_group" {
  name       = "face-recognition-redis-subnet-group"
  subnet_ids = [aws_subnet.face_recognition_private_1.id, aws_subnet.face_recognition_private_2.id]
}

resource "aws_security_group" "face_recognition_redis_sg" {
  name_prefix = "face-recognition-redis-"
  vpc_id      = aws_vpc.face_recognition_vpc.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.face_recognition_web_sg.id]
  }

  tags = {
    Name = "face-recognition-redis-sg"
  }
}

resource "aws_elasticache_cluster" "face_recognition_redis" {
  cluster_id           = "face-recognition-redis"
  engine               = "redis"
  node_type           = "cache.t3.micro"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis7"
  port                = 6379
  subnet_group_name   = aws_elasticache_subnet_group.face_recognition_redis_subnet_group.name
  security_group_ids  = [aws_security_group.face_recognition_redis_sg.id]

  tags = {
    Name = "face-recognition-redis"
  }
}

# Application Load Balancer
resource "aws_lb" "face_recognition_alb" {
  name               = "face-recognition-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.face_recognition_web_sg.id]
  subnets           = [aws_subnet.face_recognition_public_1.id, aws_subnet.face_recognition_public_2.id]

  enable_deletion_protection = false

  tags = {
    Name = "face-recognition-alb"
  }
}

# Target Group
resource "aws_lb_target_group" "face_recognition_tg" {
  name     = "face-recognition-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.face_recognition_vpc.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/api/health/"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "face-recognition-tg"
  }
}

# Launch Template
resource "aws_launch_template" "face_recognition_template" {
  name_prefix   = "face-recognition-"
  image_id      = var.ami_id
  instance_type = "t3.large"
  key_name      = var.key_pair_name

  vpc_security_group_ids = [aws_security_group.face_recognition_web_sg.id]

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    db_host     = aws_db_instance.face_recognition_db.endpoint
    db_name     = aws_db_instance.face_recognition_db.db_name
    db_user     = aws_db_instance.face_recognition_db.username
    db_password = var.db_password
    redis_host  = aws_elasticache_cluster.face_recognition_redis.cache_nodes[0].address
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "face-recognition-instance"
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "face_recognition_asg" {
  name                = "face-recognition-asg"
  vpc_zone_identifier = [aws_subnet.face_recognition_public_1.id, aws_subnet.face_recognition_public_2.id]
  target_group_arns   = [aws_lb_target_group.face_recognition_tg.arn]
  health_check_type   = "ELB"

  min_size         = 2
  max_size         = 10
  desired_capacity = 2

  launch_template {
    id      = aws_launch_template.face_recognition_template.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "face-recognition-asg"
    propagate_at_launch = true
  }
}

# ALB Listener
resource "aws_lb_listener" "face_recognition_listener" {
  load_balancer_arn = aws_lb.face_recognition_alb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.face_recognition_tg.arn
  }
}
```

**terraform/user_data.sh:**
```bash
#!/bin/bash
set -e

# Update system
yum update -y

# Install dependencies
yum install -y python3 python3-pip git nginx supervisor
amazon-linux-extras install -y postgresql13

# Create app user
useradd -m -s /bin/bash facerecog

# Setup application
cd /opt
git clone https://github.com/your-org/face-recognition-service.git face_recognition
chown -R facerecog:facerecog face_recognition

# Switch to app user and setup
sudo -u facerecog bash << 'EOF'
cd /opt/face_recognition

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create environment file
cat > .env << 'ENVEOF'
DEBUG=False
SECRET_KEY=${secret_key}
DATABASE_URL=postgresql://${db_user}:${db_password}@${db_host}:5432/${db_name}
REDIS_URL=redis://${redis_host}:6379/0
ALLOWED_HOSTS=${allowed_hosts}
ENVEOF

# Create directories
mkdir -p media/uploads media/face_images staticfiles logs

# Run migrations
python manage.py migrate
python manage.py collectstatic --noinput
EOF

# Setup services (supervisor, nginx configs)
# ... (similar to previous configurations)

# Start services
systemctl enable nginx supervisor
systemctl start nginx supervisor
```

## Monitoring & Logging

### Application Monitoring

**monitoring/docker-compose.yml:**
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

  node_exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"

volumes:
  prometheus_data:
  grafana_data:
```

**monitoring/prometheus.yml:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'face-recognition-app'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node_exporter:9100']

  - job_name: 'postgresql'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
```

### Logging Configuration

**logging_config.py:**
```python
import os
from pathlib import Path

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/opt/face_recognition/logs/django.log',
            'maxBytes': 1024*1024*50,  # 50 MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
        'face_recognition_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/opt/face_recognition/logs/face_recognition.log',
            'maxBytes': 1024*1024*50,
            'backupCount': 5,
            'formatter': 'json',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'sentry': {
            'level': 'ERROR',
            'class': 'sentry_sdk.integrations.logging.SentryHandler',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        'face_recognition': {
            'handlers': ['face_recognition_file', 'console', 'sentry'],
            'level': 'INFO',
            'propagate': False,
        },
        'celery': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
```

## Backup & Recovery

### Database Backup

**scripts/backup_db.sh:**
```bash
#!/bin/bash

# Configuration
DB_HOST="localhost"
DB_NAME="face_recognition_prod"
DB_USER="face_app"
BACKUP_DIR="/opt/face_recognition/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/db_backup_${TIMESTAMP}.sql"
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Perform backup
echo "Starting database backup..."
pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    --no-password --verbose --format=custom \
    --file="$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "Database backup completed: $BACKUP_FILE"
    
    # Compress backup
    gzip "$BACKUP_FILE"
    echo "Backup compressed: ${BACKUP_FILE}.gz"
    
    # Upload to S3 (optional)
    if [ ! -z "$AWS_S3_BUCKET" ]; then
        aws s3 cp "${BACKUP_FILE}.gz" "s3://${AWS_S3_BUCKET}/backups/"
        echo "Backup uploaded to S3"
    fi
    
    # Cleanup old backups
    find "$BACKUP_DIR" -name "db_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete
    echo "Old backups cleaned up"
    
else
    echo "Database backup failed"
    exit 1
fi
```

### Media Files Backup

**scripts/backup_media.sh:**
```bash
#!/bin/bash

MEDIA_DIR="/opt/face_recognition/media"
BACKUP_DIR="/opt/face_recognition/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/media_backup_${TIMESTAMP}.tar.gz"

echo "Starting media files backup..."
tar -czf "$BACKUP_FILE" -C "$(dirname $MEDIA_DIR)" "$(basename $MEDIA_DIR)"

if [ $? -eq 0 ]; then
    echo "Media backup completed: $BACKUP_FILE"
    
    # Upload to S3
    if [ ! -z "$AWS_S3_BUCKET" ]; then
        aws s3 cp "$BACKUP_FILE" "s3://${AWS_S3_BUCKET}/media_backups/"
    fi
else
    echo "Media backup failed"
    exit 1
fi
```

### Cron Jobs

```bash
# Add to crontab
crontab -e

# Database backup (daily at 2 AM)
0 2 * * * /opt/face_recognition/scripts/backup_db.sh >> /opt/face_recognition/logs/backup.log 2>&1

# Media backup (weekly on Sunday at 3 AM)
0 3 * * 0 /opt/face_recognition/scripts/backup_media.sh >> /opt/face_recognition/logs/backup.log 2>&1

# Log rotation (daily)
0 1 * * * /usr/sbin/logrotate /opt/face_recognition/config/logrotate.conf

# SSL renewal check (twice daily)
0 0,12 * * * /usr/bin/certbot renew --quiet
```

## Security Hardening

### System Security

```bash
# Firewall configuration
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'

# Fail2ban configuration
sudo apt install fail2ban

cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true

[nginx-http-auth]
enabled = true

[django-auth]
enabled = true
filter = django-auth
logpath = /opt/face_recognition/logs/django.log
maxretry = 3
bantime = 1800
EOF

# System updates automation
echo 'Unattended-Upgrade::Automatic-Reboot "false";' >> /etc/apt/apt.conf.d/50unattended-upgrades
systemctl enable unattended-upgrades
```

### Application Security

**security_middleware.py:**
```python
import logging
from django.http import HttpResponseForbidden
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger('face_recognition.security')

class SecurityMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Rate limiting
        if self.is_rate_limited(request):
            logger.warning(f"Rate limit exceeded for IP: {request.META.get('REMOTE_ADDR')}")
            return HttpResponseForbidden("Rate limit exceeded")

        # Security headers
        response = self.get_response(request)
        
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        return response

    def is_rate_limited(self, request):
        ip = request.META.get('REMOTE_ADDR')
        cache_key = f"rate_limit_{ip}"
        
        requests = cache.get(cache_key, 0)
        if requests >= settings.RATE_LIMIT_PER_MINUTE:
            return True
        
        cache.set(cache_key, requests + 1, 60)  # 1 minute window
        return False
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"

# Check connection from app
python manage.py dbshell
```

2. **High Memory Usage**
```bash
# Monitor memory usage
htop
free -h
docker stats (for Docker deployment)

# Optimize Gunicorn workers
# Adjust workers in gunicorn.conf.py based on available RAM
```

3. **Slow Face Recognition**
```bash
# Check GPU usage (if available)
nvidia-smi

# Monitor CPU usage
top -p $(pgrep -f "face_recognition")

# Check model loading time
python manage.py shell -c "from recognition.models import *; print('Models loaded')"
```

### Log Analysis

```bash
# Real-time log monitoring
tail -f /opt/face_recognition/logs/django.log
tail -f /opt/face_recognition/logs/face_recognition.log

# Error analysis
grep -i error /opt/face_recognition/logs/django.log | tail -20
grep -i "recognition failed" /opt/face_recognition/logs/face_recognition.log

# Performance monitoring
grep "processing_time" /opt/face_recognition/logs/face_recognition.log | \
awk '{sum+=$NF; count++} END {print "Average processing time:", sum/count, "ms"}'
```

### Health Checks

**health_check.py:**
```python
#!/usr/bin/env python
import requests
import sys
import json
from datetime import datetime

def check_api_health():
    try:
        response = requests.get('http://localhost:8000/api/health/', timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Health: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"✗ API Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API Health Check Error: {e}")
        return False

def check_database():
    try:
        response = requests.get('http://localhost:8000/api/health/database/', timeout=10)
        if response.status_code == 200:
            print("✓ Database Connection: OK")
            return True
        else:
            print("✗ Database Connection: Failed")
            return False
    except Exception as e:
        print(f"✗ Database Check Error: {e}")
        return False

def check_redis():
    try:
        response = requests.get('http://localhost:8000/api/health/redis/', timeout=10)
        if response.status_code == 200:
            print("✓ Redis Connection: OK")
            return True
        else:
            print("✗ Redis Connection: Failed")
            return False
    except Exception as e:
        print(f"✗ Redis Check Error: {e}")
        return False

if __name__ == "__main__":
    print(f"Health Check - {datetime.now()}")
    print("-" * 40)
    
    checks = [
        check_api_health(),
        check_database(),
        check_redis()
    ]
    
    if all(checks):
        print("\n✓ All health checks passed")
        sys.exit(0)
    else:
        print("\n✗ Some health checks failed")
        sys.exit(1)
```

---

*Deployment Guide Last Updated: October 9, 2025*
*Version: 2.0.0*
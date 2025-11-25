# Face Recognition v2 - Docker Deployment

Sistem Face Recognition dengan Docker containerization lengkap yang mendukung PostgreSQL, MinIO, ChromaDB, Celery, Redis, Django API, dan Frontend Test.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Nginx Proxy   │    │   Django API    │
│   (Vue.js)      │◄───┤   (Port 80)     │───►│   (Port 8000)   │
│   Port 8080     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                       ┌────────▼────────┐              │
                       │   MinIO S3      │              │
                       │   Port 9000     │              │
                       │   Console 9001  │              │
                       └─────────────────┘              │
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌────────▼────────┐
│   ChromaDB      │    │   PostgreSQL    │    │   Redis         │
│   Port 8001     │◄───┤   Port 5433     │◄───┤   Port 6380     │
│   (Vectors)     │    │   (Main DB)     │    │   (Cache/Queue) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                       ┌────────▼────────┐    ┌─────────▼────────┐
                       │   Celery Beat   │    │   Celery Worker  │
                       │   (Scheduler)   │    │   (Background)   │
                       └─────────────────┘    └──────────────────┘
```

## 🛠️ Prerequisites

- Docker (≥ 20.10)
- Docker Compose (≥ 2.0)
- 4GB+ RAM
- 10GB+ disk space

## 🚀 Quick Start

### Development Environment

```bash
# Make setup script executable
chmod +x setup.sh

# Start development environment
./setup.sh dev
```

### Production Environment

```bash
# Copy and edit environment file
cp .env.example .env
nano .env  # Edit your production settings

# Start production environment
./setup.sh prod
```

## 📋 Services

| Service | Port | Description | URL |
|---------|------|-------------|-----|
| Nginx | 80 | Reverse Proxy | http://localhost |
| Django API | 8000 | Main Application | http://localhost:8000 |
| Frontend | 8080 | Vue.js App | http://localhost:8080 |
| PostgreSQL | 5433 | Main Database | localhost:5433 |
| Redis | 6380 | Cache & Queue | localhost:6380 |
| MinIO API | 9000 | Object Storage | http://localhost:9000 |
| MinIO Console | 9001 | Storage UI | http://localhost:9001 |
| ChromaDB | 8001 | Vector Database | http://localhost:8001 |

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Core Django Settings
DEBUG=false
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,localhost

# Database
DB_PASSWORD=your-secure-password

# MinIO Credentials
MINIO_ACCESS_KEY=your-minio-access-key
MINIO_SECRET_KEY=your-minio-secret-key

# Redis Password  
REDIS_PASSWORD=your-redis-password
```

### MinIO Storage Fix

The MinIO storage configuration has been updated with:

- Proper endpoint configuration for Docker networking
- Custom storage classes for better compatibility
- SSL/TLS configuration options
- Bucket auto-creation via minio-setup service
- Public read access for media files

## 🐳 Docker Commands

### Development

```bash
# Start development environment
./setup.sh dev

# View development logs
./setup.sh logs dev

# Stop development services
./setup.sh stop
```

### Production

```bash
# Build and start production
./setup.sh prod

# View production logs
./setup.sh logs

# Check services status
./setup.sh status
```

### Maintenance

```bash
# Backup database
./setup.sh backup

# Cleanup Docker resources
./setup.sh cleanup

# Manual container access
docker-compose exec django-app bash
docker-compose exec postgres psql -U postgres -d face_recognition
```

## 🔐 Security

### Production Checklist

- [ ] Change all default passwords
- [ ] Set strong SECRET_KEY
- [ ] Configure proper ALLOWED_HOSTS
- [ ] Enable SSL/TLS (update nginx config)
- [ ] Set up firewall rules
- [ ] Configure backup strategy
- [ ] Monitor logs and metrics

### SSL/TLS Setup

1. Obtain SSL certificates (Let's Encrypt recommended)
2. Place certificates in `nginx/ssl/` directory
3. Update nginx configuration for HTTPS
4. Update environment variables for SSL endpoints

## 📊 Monitoring & Logs

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f django-app
docker-compose logs -f celery-worker

# Development logs
docker-compose -f docker-compose.dev.yml logs -f
```

### Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# Manual health check
curl http://localhost/health/
curl http://localhost:9000/minio/health/live
```

## 🔧 Troubleshooting

### Common Issues

1. **MinIO Connection Failed**
   ```bash
   # Check MinIO is running
   docker-compose logs minio
   
   # Verify bucket creation
   docker-compose logs minio-setup
   ```

2. **Database Connection Failed**
   ```bash
   # Check PostgreSQL
   docker-compose logs postgres
   
   # Test connection
   docker-compose exec postgres psql -U postgres -d face_recognition -c "SELECT 1;"
   ```

3. **File Upload Issues**
   ```bash
   # Check MinIO permissions
   docker-compose exec django-app python manage.py shell
   >>> from django.core.files.storage import default_storage
   >>> default_storage.exists('test')
   ```

### Reset Environment

```bash
# Complete reset (removes all data)
./setup.sh cleanup

# Restart specific service
docker-compose restart django-app
```

## 🚀 Deployment to Cloud

### AWS/GCP/Azure

1. Use managed PostgreSQL service
2. Use managed Redis service  
3. Use cloud object storage (S3, GCS, Blob)
4. Update environment variables accordingly
5. Use container orchestration (ECS, GKE, AKS)

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml face-recognition
```

### Kubernetes

Use the provided Kubernetes manifests (if available) or convert Docker Compose using Kompose:

```bash
kompose convert -f docker-compose.yml
```

## 🤝 Support

For issues and questions:

1. Check logs: `./setup.sh logs`
2. Verify service status: `./setup.sh status`  
3. Review configuration: Check `.env` file
4. Restart services: `./setup.sh stop && ./setup.sh prod`

## 📝 License

This project is licensed under the MIT License.
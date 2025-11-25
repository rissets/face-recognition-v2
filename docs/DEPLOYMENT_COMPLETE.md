# 🚀 Face Recognition v2 - Complete Docker Deployment Guide

## ✅ Problem Fixed: MinIO Media Storage

Masalah dengan MinIO media storage telah diperbaiki dengan:

1. **Custom Storage Classes**: Membuat `MinIOMediaStorage` dan `MinIOStaticStorage` untuk kompatibilitas yang lebih baik
2. **Network Configuration**: Menggunakan internal Docker networking (`minio:9000`) 
3. **Auto-Bucket Creation**: Service `minio-setup` otomatis membuat bucket dan mengatur permissions
4. **SSL Configuration**: Mendukung SSL/TLS untuk production
5. **Environment Variables**: Konfigurasi yang fleksibel melalui environment variables

## 🏗️ Docker Setup Lengkap

Sistem sekarang mencakup:
- ✅ **PostgreSQL** (Database utama) 
- ✅ **Redis** (Cache & Celery Queue)
- ✅ **MinIO** (Object Storage untuk media files) - **FIXED**
- ✅ **ChromaDB** (Vector Database untuk face embeddings)
- ✅ **Celery Worker** (Background tasks)
- ✅ **Celery Beat** (Scheduled tasks)  
- ✅ **Django API** (Main application)
- ✅ **Vue.js Frontend** (Testing interface)
- ✅ **Nginx** (Reverse proxy & load balancer)

## 🚀 Quick Start

### 1. Development Environment

```bash
# Clone dan masuk ke directory
cd /Users/user/Dev/researchs/face_regocnition_v2

# Start development environment
./setup.sh dev

# Atau menggunakan Make
make dev
```

**URLs Development:**
- 🌐 **Frontend**: http://localhost:8080
- 🔗 **Django API**: http://localhost:8000
- 👨‍💼 **Admin Panel**: http://localhost:8000/admin
- 📦 **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin123)
- 🗃️ **PostgreSQL**: localhost:5433
- 🔴 **Redis**: localhost:6380
- 🎯 **ChromaDB**: http://localhost:8001

### 2. Production Environment

```bash
# Setup environment file
cp .env.example .env
nano .env  # Edit with your production settings

# Start production environment  
./setup.sh prod

# Atau menggunakan Make
make prod
```

**URLs Production (via Nginx):**
- 🌐 **Main App**: http://localhost (port 80)
- 🔗 **API**: http://localhost/api/
- 👨‍💼 **Admin**: http://localhost/admin/
- 📦 **MinIO Console**: http://localhost:9001

## 🛠️ Management Commands

```bash
# Start services
./setup.sh dev          # Development
./setup.sh prod         # Production

# Monitor
./setup.sh logs         # View logs
./setup.sh logs dev     # View dev logs
./setup.sh status       # Service status

# Maintenance  
./setup.sh backup       # Backup database
./setup.sh stop         # Stop all services
./setup.sh cleanup      # Complete cleanup

# Using Make (alternative)
make help               # Show all commands
make logs              # View logs
make migrate           # Run migrations
make shell             # Django shell
make superuser         # Create admin user
```

## 🔧 Configuration Files

### Environment Variables (.env)
```env
# Django Core
DEBUG=false
SECRET_KEY=your-production-secret-key
ALLOWED_HOSTS=your-domain.com,localhost

# Database
DB_PASSWORD=secure-postgres-password

# MinIO (Fixed Configuration)
USE_MINIO=true
MINIO_ENDPOINT=http://minio:9000  # Internal Docker network
MINIO_ACCESS_KEY=your-minio-key
MINIO_SECRET_KEY=your-minio-secret
MINIO_BUCKET_NAME=face-recognition
MINIO_USE_SSL=false  # Set true for production with SSL

# Redis  
REDIS_PASSWORD=your-redis-password

# ChromaDB
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```

## 📡 Network Architecture

```
Internet
    ↓
┌─────────────────────────────────────┐
│            Nginx (Port 80/443)     │
│         Reverse Proxy & SSL         │
└─────────────────┬───────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ↓             ↓             ↓
┌──────────┐ ┌──────────┐ ┌──────────┐
│Frontend  │ │Django API│ │MinIO API │
│Vue.js    │ │Port 8000 │ │Port 9000 │
│Port 3000 │ │          │ │          │
└──────────┘ └─────┬────┘ └──────────┘
                   │
    ┌──────────────┼──────────────┐
    ↓              ↓              ↓
┌──────────┐ ┌──────────┐ ┌──────────┐
│PostgreSQL│ │Redis     │ │ChromaDB  │
│Port 5433 │ │Port 6380 │ │Port 8001 │
└──────────┘ └──────────┘ └──────────┘
                   │
    ┌──────────────┼──────────────┐
    ↓              ↓              
┌──────────┐ ┌──────────┐
│Celery    │ │Celery    │
│Worker    │ │Beat      │
└──────────┘ └──────────┘
```

## 🔒 Security Features

### Production Security Checklist
- [ ] Change all default passwords
- [ ] Set strong SECRET_KEY  
- [ ] Configure proper ALLOWED_HOSTS
- [ ] Enable SSL/TLS certificates
- [ ] Set up firewall rules
- [ ] Configure rate limiting
- [ ] Enable security headers
- [ ] Set up monitoring & logging

### SSL/HTTPS Setup
```bash
# Generate self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem

# For production, use Let's Encrypt
certbot --nginx -d your-domain.com
```

## 🧪 Testing & Verification

### 1. Health Checks
```bash
# Check all services
curl http://localhost/health/      # Django health
curl http://localhost:9000/minio/health/live  # MinIO health
curl http://localhost:8001/api/v1/heartbeat   # ChromaDB health

# Database connection
docker-compose exec postgres psql -U postgres -d face_recognition -c "SELECT 1;"
```

### 2. MinIO Storage Test
```bash
# Django shell test
docker-compose exec django-app python manage.py shell
>>> from django.core.files.storage import default_storage  
>>> from django.core.files.base import ContentFile
>>> name = default_storage.save('test.txt', ContentFile(b'Hello MinIO!'))
>>> print(f"File saved: {name}")
>>> url = default_storage.url(name)
>>> print(f"File URL: {url}")
```

### 3. Face Recognition API Test
```bash
# Test API endpoints
curl -X GET http://localhost/api/core/health/
curl -X POST http://localhost/api/auth/enroll/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "image": "base64_image_data"}'
```

## 🐳 Docker Images

### Custom Images Built:
- **face-recognition-django**: Main Django application
- **face-recognition-frontend**: Vue.js frontend with Nginx

### External Images Used:
- **postgres:15-alpine**: PostgreSQL database
- **redis:7-alpine**: Redis cache & queue
- **minio/minio:latest**: MinIO object storage  
- **chromadb/chroma:latest**: ChromaDB vector database
- **nginx:alpine**: Nginx reverse proxy

## 📊 Monitoring & Logs

### Log Locations
```bash
# Application logs
docker-compose logs django-app
docker-compose logs celery-worker
docker-compose logs celery-beat

# Infrastructure logs  
docker-compose logs postgres
docker-compose logs redis
docker-compose logs minio
docker-compose logs chromadb
docker-compose logs nginx
```

### Persistent Volumes
- `postgres_data`: Database files
- `redis_data`: Redis persistence
- `minio_data`: MinIO object storage  
- `chroma_data`: ChromaDB vectors
- `django_media`: Uploaded media files
- `django_static`: Static assets
- `django_logs`: Application logs

## 🚀 Cloud Deployment

### AWS/GCP/Azure Recommendations
1. **Database**: Use managed PostgreSQL (RDS/Cloud SQL)
2. **Cache**: Use managed Redis (ElastiCache/MemoryStore)
3. **Storage**: Use native object storage (S3/GCS/Blob Storage)
4. **Container**: Use container orchestration (ECS/GKE/AKS)
5. **CDN**: Use CloudFront/CloudCDN for static assets

### Kubernetes Deployment
```bash
# Convert Docker Compose to Kubernetes
kompose convert -f docker-compose.yml

# Or use provided Kubernetes manifests (if available)
kubectl apply -f k8s/
```

## 🎯 Access Information

### Default Credentials
- **Django Admin**: admin/admin123 (created during setup)
- **MinIO Console**: minioadmin/minioadmin123
- **PostgreSQL**: postgres/password_rahasia  
- **Redis**: redis_password

### Ports Summary
| Service | Port | Access |
|---------|------|--------|
| Nginx | 80, 443 | Main entry point |
| Django API | 8000 | Direct API access |
| Frontend | 3000 | Direct frontend access |
| PostgreSQL | 5433 | Database access |
| Redis | 6380 | Cache access |
| MinIO API | 9000 | Object storage API |
| MinIO Console | 9001 | Storage management |
| ChromaDB | 8001 | Vector database |

## 🆘 Troubleshooting

### Common Issues & Solutions

1. **MinIO Connection Failed** ✅ FIXED
   ```bash
   # Check MinIO status
   docker-compose logs minio
   docker-compose logs minio-setup
   
   # Restart MinIO services
   docker-compose restart minio minio-setup
   ```

2. **Database Connection Error**
   ```bash
   # Check PostgreSQL
   docker-compose logs postgres
   
   # Test connection
   docker-compose exec postgres pg_isready -U postgres
   ```

3. **File Upload Issues** ✅ FIXED
   ```bash
   # Test storage in Django shell
   docker-compose exec django-app python manage.py shell
   >>> from django.core.files.storage import default_storage
   >>> default_storage.exists('test')
   ```

4. **Frontend API Connection**
   ```bash
   # Check network connectivity
   docker-compose exec frontend-app ping django-app
   
   # Check Nginx configuration
   docker-compose exec nginx nginx -t
   ```

### Reset & Clean Installation
```bash
# Complete reset (removes all data!)
./setup.sh cleanup
docker system prune -a

# Fresh installation  
./setup.sh dev  # or ./setup.sh prod
```

## ✅ Success Verification

After deployment, verify:

1. ✅ **Frontend accessible**: http://localhost (or your domain)
2. ✅ **API responding**: http://localhost/api/core/health/
3. ✅ **Admin panel working**: http://localhost/admin/
4. ✅ **MinIO accessible**: http://localhost:9001
5. ✅ **File upload working**: Test via admin or API
6. ✅ **Background tasks**: Check Celery worker logs
7. ✅ **WebSocket support**: Real-time features working
8. ✅ **Database migrations**: All tables created
9. ✅ **SSL/HTTPS**: If configured for production

## 🎉 Deployment Complete!

Sistem Face Recognition v2 dengan Docker sekarang sudah siap digunakan dengan:
- ✅ MinIO media storage yang telah diperbaiki
- ✅ Semua services terintegrasi dengan Docker Compose
- ✅ Network internal yang aman antar containers  
- ✅ Auto-setup scripts untuk development & production
- ✅ Health checks & monitoring untuk semua services
- ✅ SSL/HTTPS support untuk production
- ✅ Backup & recovery procedures

**Happy coding! 🚀**
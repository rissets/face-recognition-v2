#!/bin/bash

# Face Recognition v2 - Docker Setup Script
set -e

echo "🚀 Face Recognition v2 - Docker Setup"
echo "===================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Function to setup development environment
setup_dev() {
    echo "🔧 Setting up development environment..."
    
    # Copy environment file
    if [ ! -f .env.dev ]; then
        echo "📝 Creating .env.dev file..."
        cp .env.dev .env.dev.local 2>/dev/null || true
    fi
    
    echo "🏗️  Building development containers..."
    docker-compose -f docker-compose.dev.yml build
    
    echo "🚀 Starting development services..."
    docker-compose -f docker-compose.dev.yml up -d
    
    echo "⏳ Waiting for services to be ready..."
    sleep 30
    
    echo "📊 Running database migrations..."
    docker-compose -f docker-compose.dev.yml exec -T django-dev python manage.py migrate
    
    echo "👤 Creating superuser (skip if exists)..."
    docker-compose -f docker-compose.dev.yml exec -T django-dev python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('Superuser created: admin/admin123')
else:
    print('Superuser already exists')
" || true
    
    echo "✅ Development environment is ready!"
    echo ""
    echo "🌐 Services:"
    echo "  - Django API: http://localhost:8000"
    echo "  - Frontend: http://localhost:8080"
    echo "  - PostgreSQL: localhost:5433"
    echo "  - Redis: localhost:6380"
    echo "  - MinIO: http://localhost:9000 (admin: minioadmin/minioadmin123)"
    echo "  - MinIO Console: http://localhost:9001"
    echo "  - ChromaDB: http://localhost:8001"
    echo ""
    echo "📚 Admin Panel: http://localhost:8000/admin (admin/admin123)"
}

# Function to setup production environment
setup_prod() {
    echo "🔧 Setting up production environment..."
    
    # Check if .env exists
    if [ ! -f .env ]; then
        echo "📝 Creating .env file from template..."
        cp .env.example .env
        echo "⚠️  Please edit .env file with your production settings before continuing!"
        echo "🔑 Don't forget to change SECRET_KEY, passwords, and domain settings!"
        exit 1
    fi
    
    echo "🏗️  Building production containers..."
    docker-compose build
    
    echo "🚀 Starting production services..."
    docker-compose up -d
    
    echo "⏳ Waiting for services to be ready..."
    sleep 60
    
    echo "📊 Running database migrations..."
    docker-compose exec -T django-app python manage.py migrate
    
    echo "📦 Collecting static files..."
    docker-compose exec -T django-app python manage.py collectstatic --noinput
    
    echo "👤 Creating superuser..."
    echo "Please create admin user:"
    docker-compose exec django-app python manage.py createsuperuser
    
    echo "✅ Production environment is ready!"
    echo ""
    echo "🌐 Services:"
    echo "  - Main Application: http://localhost (via Nginx)"
    echo "  - Django API: http://localhost/api/"
    echo "  - Admin Panel: http://localhost/admin/"
    echo "  - MinIO Console: http://localhost:9001"
}

# Function to stop services
stop_services() {
    echo "🛑 Stopping services..."
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
    docker-compose down 2>/dev/null || true
    echo "✅ Services stopped!"
}

# Function to view logs
view_logs() {
    echo "📋 Viewing logs..."
    if [ "$2" = "dev" ]; then
        docker-compose -f docker-compose.dev.yml logs -f
    else
        docker-compose logs -f
    fi
}

# Function to cleanup
cleanup() {
    echo "🧹 Cleaning up Docker resources..."
    docker-compose -f docker-compose.dev.yml down -v 2>/dev/null || true
    docker-compose down -v 2>/dev/null || true
    docker system prune -f
    echo "✅ Cleanup completed!"
}

# Function to show status
show_status() {
    echo "📊 Docker Services Status:"
    echo "=========================="
    docker-compose ps 2>/dev/null || echo "Production services not running"
    echo ""
    docker-compose -f docker-compose.dev.yml ps 2>/dev/null || echo "Development services not running"
}

# Function to backup database
backup_db() {
    echo "💾 Creating database backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    docker-compose exec -T postgres pg_dump -U postgres face_recognition > "backup_${timestamp}.sql"
    echo "✅ Database backed up to backup_${timestamp}.sql"
}

# Main script logic
case "$1" in
    "dev")
        setup_dev
        ;;
    "prod")
        setup_prod
        ;;
    "stop")
        stop_services
        ;;
    "logs")
        view_logs "$@"
        ;;
    "cleanup")
        cleanup
        ;;
    "status")
        show_status
        ;;
    "backup")
        backup_db
        ;;
    *)
        echo "Usage: $0 {dev|prod|stop|logs [dev]|cleanup|status|backup}"
        echo ""
        echo "Commands:"
        echo "  dev     - Setup and start development environment"
        echo "  prod    - Setup and start production environment"
        echo "  stop    - Stop all services"
        echo "  logs    - View logs (add 'dev' for development logs)"
        echo "  cleanup - Stop services and clean up volumes"
        echo "  status  - Show services status"
        echo "  backup  - Backup database"
        echo ""
        echo "Examples:"
        echo "  $0 dev          # Start development environment"
        echo "  $0 prod         # Start production environment"
        echo "  $0 logs dev     # View development logs"
        echo "  $0 stop         # Stop all services"
        exit 1
        ;;
esac
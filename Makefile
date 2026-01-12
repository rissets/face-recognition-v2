.PHONY: help dev prod stop logs cleanup status backup build

# Default target
help:
	@echo "Face Recognition v2 - Docker Management"
	@echo "======================================"
	@echo "Available commands:"
	@echo "  make dev      - Start development environment"
	@echo "  make prod     - Start production environment"
	@echo "  make stop     - Stop all services"
	@echo "  make logs     - View production logs"
	@echo "  make logs-dev - View development logs"
	@echo "  make cleanup  - Stop and cleanup volumes"
	@echo "  make status   - Show services status"
	@echo "  make backup   - Backup database"
	@echo "  make build    - Build all images"
	@echo "  make shell    - Access Django shell"
	@echo "  make migrate  - Run database migrations"

# Development environment
dev:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Waiting for services..."
	@sleep 30
	docker-compose -f docker-compose.dev.yml exec django-dev python manage.py migrate
	@echo "Development environment ready!"

# Production environment  
prod:
	@echo "Starting production environment..."
	docker-compose up -d
	@echo "Waiting for services..."
	@sleep 60
	docker-compose exec django-app python manage.py migrate
	docker-compose exec django-app python manage.py collectstatic --noinput
	@echo "Production environment ready!"

# Stop all services
stop:
	docker-compose down
	docker-compose -f docker-compose.dev.yml down

# View logs
logs:
	docker-compose logs -f

logs-dev:
	docker-compose -f docker-compose.dev.yml logs -f

# Cleanup
cleanup:
	docker-compose down -v
	docker-compose -f docker-compose.dev.yml down -v
	docker system prune -f

# Status
status:
	@echo "Production Services:"
	@docker-compose ps
	@echo "\nDevelopment Services:"
	@docker-compose -f docker-compose.dev.yml ps

# Backup database
backup:
	@timestamp=$$(date +%Y%m%d_%H%M%S) && \
	docker-compose exec -T postgres pg_dump -U postgres face_recognition > "backup_$${timestamp}.sql" && \
	echo "Database backed up to backup_$${timestamp}.sql"

# Build images
build:
	docker-compose build
	docker-compose -f docker-compose.dev.yml build

# Django management
shell:
	docker-compose exec django-app python manage.py shell

shell-dev:
	docker-compose -f docker-compose.dev.yml exec django-dev python manage.py shell

migrate:
	docker-compose exec django-app python manage.py migrate

migrate-dev:
	docker-compose -f docker-compose.dev.yml exec django-dev python manage.py migrate

# Create superuser
superuser:
	docker-compose exec django-app python manage.py createsuperuser

superuser-dev:
	docker-compose -f docker-compose.dev.yml exec django-dev python manage.py createsuperuser
# Redis Authentication Fix Guide

## Problem
The application is experiencing Redis authentication errors, causing API endpoints (including `/api/docs/`) to return 500 errors with the message:
```
django_redis.exceptions.ConnectionInterrupted: Redis AuthenticationError: Authentication required.
```

## Root Cause
The Redis server is configured with authentication (`requirepass`), but the Django application is not configured to use the Redis password.

## Solution

### 1. Update Environment Configuration

**Production Environment (`.env.prod`):**
```bash
# Redis Configuration  
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_password
REDIS_URL=redis://:redis_password@redis:6379/1
REDIS_CACHE_URL=redis://:redis_password@redis:6379/2
REDIS_CHANNELS_URL=redis://:redis_password@redis:6379/3

# Celery Configuration
CELERY_BROKER_URL=redis://:redis_password@redis:6379/0
CELERY_RESULT_BACKEND=redis://:redis_password@redis:6379/0
```

**Development Environment (`.env.dev`):**
```bash
# For local development without Redis auth
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=""
```

### 2. Verify Docker Compose Configuration

Ensure the Redis service in `docker-compose.yml` has the correct password:
```yaml
redis:
  image: redis:7-alpine
  command: redis-server --appendonly yes --requirepass redis_password
```

### 3. Test Redis Connectivity

Use the built-in management command to test Redis connections:
```bash
# Test Redis connectivity
python manage.py test_redis

# Test with fix suggestions
python manage.py test_redis --fix-auth
```

### 4. Temporary Workarounds

If Redis issues persist, you can temporarily disable throttling:

**In `.env` file:**
```bash
DISABLE_THROTTLING=true
```

### 5. Application Features

The application now includes:

- **Safe Throttling Classes**: Gracefully handle Redis connection errors
- **Robust Redis Configuration**: Automatic fallback for development/production
- **Connection Pooling**: Better Redis connection management
- **Error Handling**: Prevents API failures due to Redis issues

### 6. Restart Services

After making configuration changes:
```bash
# Restart all services
docker-compose down
docker-compose up -d

# Or restart just the web application
docker-compose restart web
```

### 7. Verification

1. Check Redis connectivity: `python manage.py test_redis`
2. Test API endpoint: `curl http://localhost:8003/api/docs/`
3. Monitor logs: `docker-compose logs -f web`

## Environment-Specific Notes

### Development
- Redis password can be empty for local development
- Use `127.0.0.1` as Redis host for local instances

### Production
- Always use Redis password for security
- Use service names (`redis`) for Docker Compose networking
- Ensure all Redis URLs include the password

## Additional Troubleshooting

### Common Issues:
1. **Wrong password**: Verify `REDIS_PASSWORD` matches Docker Compose configuration
2. **Network issues**: Check if Redis container is running and accessible
3. **Port conflicts**: Ensure Redis port (6379/6380) is not blocked

### Debug Commands:
```bash
# Check Redis container status
docker-compose ps redis

# Test Redis directly
redis-cli -h localhost -p 6380 -a redis_password ping

# View application logs
docker-compose logs -f web

# Check environment variables
docker-compose exec web env | grep REDIS
```

## Files Modified
- `face_app/settings.py`: Updated Redis configuration with authentication
- `.env.prod`: Added Redis password and URLs
- `core/throttling.py`: Created safe throttle classes
- `core/exceptions.py`: Added Redis error handling
- `core/management/commands/test_redis.py`: Redis connectivity testing

The application should now handle Redis authentication properly and gracefully degrade if Redis is unavailable.
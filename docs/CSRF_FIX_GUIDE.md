# CSRF Fix for Production Deployment

## Problem
The application was returning a 403 CSRF verification failed error because the domain `https://human-face.hellodigi.id` was not configured as a trusted origin.

## Solution Applied

### 1. Updated Django Settings (`face_app/settings.py`)
Added `CSRF_TRUSTED_ORIGINS` configuration:

```python
# CSRF Configuration
CSRF_TRUSTED_ORIGINS = config(
    "CSRF_TRUSTED_ORIGINS",
    default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173",
    cast=lambda v: [s.strip() for s in v.split(",")]
)
```

### 2. Created Production Environment File (`.env.prod`)
```env
# Production settings with proper CSRF and CORS configuration
ALLOWED_HOSTS=localhost,127.0.0.1,human-face.hellodigi.id
CSRF_TRUSTED_ORIGINS=https://human-face.hellodigi.id
CORS_ALLOWED_ORIGINS=https://human-face.hellodigi.id
```

### 3. Updated Development Environment (`.env.dev`)
Added the production domain for testing:
```env
ALLOWED_HOSTS=localhost,127.0.0.1,django-dev,frontend-dev,human-face.hellodigi.id
CSRF_TRUSTED_ORIGINS=http://localhost:3000,...,https://human-face.hellodigi.id
CORS_ALLOWED_ORIGINS=http://localhost:3000,...,https://human-face.hellodigi.id
```

### 4. Added Production Security Settings
Enhanced security configuration for HTTPS deployment:
- SSL redirect configuration
- Secure cookies for HTTPS
- Proxy SSL header configuration

## Deployment Instructions

### For Production Deployment:

1. **Copy the production environment file:**
   ```bash
   cp .env.prod .env
   ```

2. **Update the production secrets:**
   ```bash
   # Edit .env and change these values:
   SECRET_KEY=your-production-secret-key
   DB_PASSWORD=your-secure-database-password
   MINIO_ACCESS_KEY=your-minio-access-key
   MINIO_SECRET_KEY=your-minio-secret-key
   FIELD_ENCRYPTION_KEY=your-encryption-key
   ```

3. **Deploy with Docker Compose:**
   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

4. **Run Django collectstatic and migrate:**
   ```bash
   docker-compose exec django python manage.py collectstatic --noinput
   docker-compose exec django python manage.py migrate
   ```

### For Additional Domains:

If you need to add more domains, update the environment variables:

```env
# Multiple domains separated by commas
ALLOWED_HOSTS=localhost,127.0.0.1,domain1.com,domain2.com
CSRF_TRUSTED_ORIGINS=https://domain1.com,https://domain2.com
CORS_ALLOWED_ORIGINS=https://domain1.com,https://domain2.com
```

## Testing the Fix

After deployment, test the CSRF fix by:
1. Making a POST request to any Django endpoint
2. Ensuring the request includes the CSRF token
3. Verifying no 403 errors occur

## Notes

- The `CSRF_TRUSTED_ORIGINS` must include the protocol (https://)
- `ALLOWED_HOSTS` should include the domain without protocol
- For development, both HTTP and HTTPS origins are included
- Production environment enforces HTTPS redirects and secure cookies
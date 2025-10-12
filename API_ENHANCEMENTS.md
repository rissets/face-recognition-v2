# API Enhancements - Profile Image & Session Management

## Overview
This document describes the recent enhancements to the Face Recognition API regarding profile image management, API usage tracking, and authentication session improvements.

## 🖼️ Profile Image Feature

### Automatic Profile Image Capture
During enrollment, the system automatically captures and saves the best quality frame as the user's profile image. This image is stored in MinIO/S3 and updated with each new enrollment.

#### Key Features:
- **Automatic Update**: Profile images are updated with the latest/best frame from enrollment
- **Quality-Based Selection**: Only high-quality frames are saved as profile images
- **Secure Storage**: Images are stored in MinIO/S3 with proper access controls
- **URL Generation**: Profile image URLs are automatically generated in API responses

#### API Fields Added:
- `ClientUser.profile_image_url`: URL to the user's profile image
- `AuthenticationSession.client_user_info`: Detailed client user information including profile image

### API Response Examples

#### Client User with Profile Image:
```json
{
  "id": "uuid",
  "external_user_id": "user123",
  "display_name": "John Doe",
  "is_enrolled": true,
  "profile_image_url": "https://storage.example.com/client_users/profiles/2025/10/12/profile_user123_1.jpg",
  "enrollment_completed_at": "2025-10-12T10:30:00Z"
}
```

#### Authentication Session with Client User:
```json
{
  "id": "session-uuid",
  "session_token": "sess_abc123",
  "session_type": "identification",
  "status": "completed",
  "client_user_info": {
    "id": "user-uuid",
    "external_user_id": "user123",
    "display_name": "John Doe",
    "is_enrolled": true,
    "profile_image_url": "https://storage.example.com/client_users/profiles/2025/10/12/profile_user123_1.jpg"
  }
}
```

## 📊 API Usage Tracking

### Middleware Implementation
The `ClientUsageLoggingMiddleware` automatically tracks all API calls for billing and analytics purposes.

#### Tracked Metrics:
- **Endpoint Category**: enrollment, recognition, liveness, webhook, analytics
- **HTTP Method**: GET, POST, PUT, DELETE, etc.
- **Response Status**: 200, 400, 401, 500, etc.
- **Response Time**: Processing time in milliseconds
- **Client Information**: IP address, user agent
- **Request Metadata**: Query parameters, view name

#### Endpoint Categories:
- `enrollment`: Face enrollment endpoints
- `recognition`: Face recognition and authentication endpoints  
- `liveness`: Liveness detection endpoints
- `webhook`: Webhook management endpoints
- `analytics`: Analytics and reporting endpoints

### Usage Analytics API

#### Get Client API Usage:
```http
GET /api/clients/usage/
Authorization: X-API-Key: your-api-key
```

#### Response:
```json
{
  "count": 150,
  "results": [
    {
      "id": "usage-uuid",
      "endpoint": "recognition", 
      "method": "POST",
      "status_code": 200,
      "response_time_ms": 156.3,
      "ip_address": "192.168.1.100",
      "created_at": "2025-10-12T10:30:00Z",
      "metadata": {
        "path": "/api/auth/face/process-frame/",
        "query_params": {},
        "view_name": "process_authentication_frame"
      }
    }
  ]
}
```

## 🔐 Authentication Session Improvements

### Client User Association
Authentication sessions now properly track the associated client user, especially in identification mode.

#### Key Improvements:
- **Verification Mode**: `client_user` is set when session is created
- **Identification Mode**: `client_user` is updated when user is successfully identified
- **Session Tracking**: All sessions maintain proper user association for audit trails

#### Session States:
1. **Created**: Session created with or without target user
2. **Processing**: Frames being processed for authentication
3. **Completed**: Authentication successful, user identified (if applicable)
4. **Failed**: Authentication failed or expired

### Updated Session Response:
```json
{
  "session_token": "sess_abc123",
  "session_type": "identification",
  "status": "completed",
  "client_user": "user-uuid",
  "client_user_info": {
    "id": "user-uuid",
    "external_user_id": "user123", 
    "display_name": "John Doe",
    "is_enrolled": true,
    "profile_image_url": "https://storage.example.com/.../profile.jpg"
  },
  "confidence_score": 0.95,
  "completed_at": "2025-10-12T10:30:00Z"
}
```

## 🔄 Migration Requirements

### Database Changes:
The `profile_image` field already exists in the `ClientUser` model, no migration needed.

### Storage Configuration:
Ensure MinIO/S3 is properly configured for profile image storage:

```python
# settings.py
DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
AWS_STORAGE_BUCKET_NAME = 'face-recognition-storage'
AWS_S3_CUSTOM_DOMAIN = 'storage.yourdomain.com'
```

## 📝 Breaking Changes

### API Response Changes:
- `ClientUser` responses now include `profile_image_url` field
- `AuthenticationSession` responses now include `client_user` and `client_user_info` fields
- Session summary responses include additional client user fields

### Behavior Changes:
- Profile images are now updated on each enrollment (previously only set if empty)
- Authentication sessions in identification mode now properly set `client_user` field
- API usage is automatically tracked for all authenticated API calls

## 🧪 Testing

### Profile Image Testing:
```bash
# Create enrollment session
curl -X POST /api/enrollment/create/ \
  -H "X-API-Key: your-key" \
  -d '{"user_id": "test_user", "target_samples": 3}'

# Process frames (profile image saved automatically)
# Check client user response includes profile_image_url
curl -X GET /api/clients/users/uuid/ \
  -H "X-API-Key: your-key"
```

### Session Testing:
```bash
# Create identification session
curl -X POST /api/auth/face/create/ \
  -H "X-API-Key: your-key" \
  -d '{"session_type": "identification"}'

# After successful identification, check session includes client_user_info
curl -X GET /api/auth/sessions/uuid/ \
  -H "X-API-Key: your-key"
```

### API Usage Testing:
```bash
# Make some API calls, then check usage tracking
curl -X GET /api/clients/usage/ \
  -H "X-API-Key: your-key"
```

## 🚀 Deployment Notes

1. **Storage Access**: Ensure proper MinIO/S3 permissions for profile image uploads
2. **URL Generation**: Configure proper domain settings for profile image URLs
3. **Middleware**: Verify `ClientUsageLoggingMiddleware` is enabled in settings
4. **Performance**: Monitor API usage tracking performance impact

## 📞 Support

For questions or issues related to these enhancements:
- Profile images not uploading: Check MinIO/S3 configuration
- API usage not tracking: Verify middleware is enabled
- Session client_user not set: Check authentication flow logs
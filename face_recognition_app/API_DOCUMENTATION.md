# API Documentation - Face Recognition Third-Party Service

## Overview

Dokumentasi lengkap untuk Face Recognition Third-Party Service API yang menyediakan layanan face enrollment, authentication, dan management client dengan arsitektur multi-tenant.

## Base URLs

```
Development: http://localhost:8000
Production: https://api.face-recognition.com
```

## Authentication

Sistem menggunakan JWT authentication dengan client credentials. Setiap request API harus menyertakan:

```http
Authorization: Bearer <jwt_token>
X-Client-ID: <client_id>
Content-Type: application/json
```

### Mendapatkan JWT Token

```http
POST /api/auth/token/
Content-Type: application/json

{
    "client_id": "FR_DEMO123",
    "api_key": "frapi_demo_key_12345",
    "secret_key": "demo_secret_key"
}
```

**Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "Bearer",
    "expires_in": 1800,
    "client_id": "FR_DEMO123"
}
```

## Core API Endpoints

### 1. Client Management

#### Get Client Info
```http
GET /api/clients/me/
Authorization: Bearer <token>
X-Client-ID: <client_id>
```

**Response:**
```json
{
    "id": "uuid",
    "client_id": "FR_DEMO123",
    "name": "Demo Client",
    "tier": "basic",
    "status": "active",
    "rate_limit_per_hour": 1000,
    "rate_limit_per_day": 10000,
    "features": {
        "enrollment": true,
        "recognition": true,
        "liveness_detection": false,
        "max_users_per_client": 1000
    }
}
```

#### List Client Users
```http
GET /api/clients/me/users/
Authorization: Bearer <token>
X-Client-ID: <client_id>
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 20)
- `search`: Search by external_user_id
- `is_enrolled`: Filter by enrollment status

**Response:**
```json
{
    "count": 100,
    "next": "http://api.example.com/clients/me/users/?page=2",
    "previous": null,
    "results": [
        {
            "id": "uuid",
            "external_user_id": "user_123",
            "external_user_uuid": "user-uuid",
            "profile": {
                "name": "John Doe",
                "email": "john@example.com"
            },
            "is_enrolled": true,
            "face_auth_enabled": true,
            "enrollment_completed_at": "2025-10-09T10:30:00Z",
            "last_recognition_at": "2025-10-09T15:45:00Z",
            "created_at": "2025-10-01T09:00:00Z"
        }
    ]
}
```

#### Create Client User
```http
POST /api/clients/me/users/
Authorization: Bearer <token>
X-Client-ID: <client_id>
Content-Type: application/json

{
    "external_user_id": "user_456",
    "external_user_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "profile": {
        "name": "Jane Smith",
        "email": "jane@example.com",
        "department": "Engineering"
    },
    "face_auth_enabled": true
}
```

**Response:**
```json
{
    "id": "uuid",
    "external_user_id": "user_456",
    "external_user_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "profile": {
        "name": "Jane Smith",
        "email": "jane@example.com",
        "department": "Engineering"
    },
    "is_enrolled": false,
    "face_auth_enabled": true,
    "enrollment_completed_at": null,
    "last_recognition_at": null,
    "created_at": "2025-10-09T16:20:00Z"
}
```

### 2. Authentication Sessions

#### Create Authentication Session
```http
POST /api/auth-service/sessions/
Authorization: Bearer <token>
X-Client-ID: <client_id>
Content-Type: application/json

{
    "client_user_id": "user_123",
    "session_type": "enrollment",
    "device_info": {
        "device_id": "device_abc123",
        "platform": "web",
        "user_agent": "Mozilla/5.0...",
        "screen_resolution": "1920x1080"
    },
    "expires_in_minutes": 15
}
```

**Session Types:**
- `enrollment`: Face enrollment session
- `recognition`: Face authentication session

**Response:**
```json
{
    "session_id": "sess_1234567890abcdef",
    "session_token": "tok_abcdef1234567890",
    "session_type": "enrollment",
    "status": "active",
    "client_user": {
        "id": "uuid",
        "external_user_id": "user_123",
        "profile": {
            "name": "John Doe"
        }
    },
    "expires_at": "2025-10-09T17:00:00Z",
    "created_at": "2025-10-09T16:45:00Z"
}
```

#### Get Session Status
```http
GET /api/auth-service/sessions/{session_id}/
Authorization: Bearer <token>
X-Client-ID: <client_id>
```

**Response:**
```json
{
    "session_id": "sess_1234567890abcdef",
    "session_type": "enrollment",
    "status": "active",
    "progress": {
        "samples_required": 5,
        "samples_completed": 3,
        "progress_percentage": 60.0
    },
    "quality_metrics": {
        "average_quality": 0.85,
        "min_quality_threshold": 0.7
    },
    "expires_at": "2025-10-09T17:00:00Z"
}
```

### 3. Face Enrollment

#### Submit Face Sample
```http
POST /api/auth-service/enrollments/
Authorization: Bearer <token>
X-Client-ID: <client_id>
Content-Type: application/json

{
    "session_id": "sess_1234567890abcdef",
    "face_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA...",
    "sample_number": 1,
    "capture_metadata": {
        "timestamp": "2025-10-09T16:45:30Z",
        "camera_resolution": "640x480"
    }
}
```

**Response (Success):**
```json
{
    "enrollment_id": "enr_abcdef123456",
    "sample_number": 1,
    "status": "accepted",
    "quality_metrics": {
        "face_quality_score": 0.92,
        "liveness_score": 0.88,
        "anti_spoofing_score": 0.95
    },
    "face_detection": {
        "face_count": 1,
        "face_bbox": [120, 80, 520, 400],
        "landmarks_detected": true
    },
    "session_progress": {
        "samples_completed": 1,
        "samples_required": 5,
        "progress_percentage": 20.0
    }
}
```

**Response (Quality Issues):**
```json
{
    "enrollment_id": null,
    "sample_number": 1,
    "status": "rejected",
    "rejection_reason": "quality_too_low",
    "quality_metrics": {
        "face_quality_score": 0.45,
        "liveness_score": 0.92,
        "anti_spoofing_score": 0.88
    },
    "feedback": {
        "message": "Face quality is too low. Please ensure good lighting and clear image.",
        "suggestions": [
            "Improve lighting conditions",
            "Look directly at the camera",
            "Remove any obstructions"
        ]
    }
}
```

#### Complete Enrollment
```http
POST /api/auth-service/enrollments/complete/
Authorization: Bearer <token>
X-Client-ID: <client_id>
Content-Type: application/json

{
    "session_id": "sess_1234567890abcdef"
}
```

**Response:**
```json
{
    "enrollment_completed": true,
    "user_id": "user_123",
    "total_samples": 5,
    "average_quality": 0.89,
    "enrollment_id": "enr_final_xyz789",
    "completed_at": "2025-10-09T17:02:45Z"
}
```

### 4. Face Recognition

#### Perform Face Recognition
```http
POST /api/auth-service/recognition/
Authorization: Bearer <token>
X-Client-ID: <client_id>
Content-Type: application/json

{
    "session_id": "sess_recognition_abc123",
    "face_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA...",
    "liveness_check": true,
    "anti_spoofing_check": true
}
```

**Response (Successful Recognition):**
```json
{
    "recognition_id": "rec_xyz789abc",
    "result": "success",
    "matched_user": {
        "external_user_id": "user_123",
        "profile": {
            "name": "John Doe",
            "department": "Engineering"
        }
    },
    "confidence_metrics": {
        "similarity_score": 0.94,
        "confidence_score": 0.91,
        "face_quality_score": 0.88
    },
    "security_checks": {
        "liveness_passed": true,
        "liveness_score": 0.89,
        "anti_spoofing_passed": true,
        "anti_spoofing_score": 0.92
    },
    "processing_time_ms": 245,
    "timestamp": "2025-10-09T17:15:30Z"
}
```

**Response (No Match):**
```json
{
    "recognition_id": "rec_xyz789abc",
    "result": "no_match",
    "matched_user": null,
    "confidence_metrics": {
        "highest_similarity_score": 0.45,
        "face_quality_score": 0.88
    },
    "security_checks": {
        "liveness_passed": true,
        "liveness_score": 0.89,
        "anti_spoofing_passed": true,
        "anti_spoofing_score": 0.92
    },
    "processing_time_ms": 198,
    "timestamp": "2025-10-09T17:15:30Z"
}
```

**Response (Security Check Failed):**
```json
{
    "recognition_id": "rec_xyz789abc",
    "result": "liveness_failed",
    "matched_user": null,
    "confidence_metrics": {
        "face_quality_score": 0.88
    },
    "security_checks": {
        "liveness_passed": false,
        "liveness_score": 0.34,
        "failure_reason": "spoofing_detected"
    },
    "processing_time_ms": 156,
    "timestamp": "2025-10-09T17:15:30Z"
}
```

### 5. Webhook Management

#### List Webhook Endpoints
```http
GET /api/webhooks/endpoints/
Authorization: Bearer <token>
X-Client-ID: <client_id>
```

**Response:**
```json
{
    "count": 2,
    "results": [
        {
            "id": "uuid",
            "name": "Production Webhook",
            "url": "https://myapp.com/webhooks/face-auth",
            "status": "active",
            "subscribed_events": [
                "enrollment.completed",
                "recognition.success",
                "recognition.failed"
            ],
            "statistics": {
                "total_deliveries": 1250,
                "successful_deliveries": 1198,
                "failed_deliveries": 52,
                "success_rate": 95.84
            },
            "last_delivery_at": "2025-10-09T17:10:00Z",
            "created_at": "2025-10-01T09:00:00Z"
        }
    ]
}
```

#### Create Webhook Endpoint
```http
POST /api/webhooks/endpoints/
Authorization: Bearer <token>
X-Client-ID: <client_id>
Content-Type: application/json

{
    "name": "Development Webhook",
    "url": "https://dev.myapp.com/webhooks/face-auth",
    "subscribed_events": [
        "enrollment.completed",
        "recognition.success",
        "recognition.failed",
        "session.expired"
    ],
    "max_retries": 3,
    "retry_delay_seconds": 60
}
```

**Response:**
```json
{
    "id": "uuid",
    "name": "Development Webhook",
    "url": "https://dev.myapp.com/webhooks/face-auth",
    "status": "active",
    "subscribed_events": [
        "enrollment.completed",
        "recognition.success",
        "recognition.failed",
        "session.expired"
    ],
    "secret_token": "whsec_1234567890abcdef",
    "max_retries": 3,
    "retry_delay_seconds": 60,
    "created_at": "2025-10-09T17:20:00Z"
}
```

#### Test Webhook Endpoint
```http
POST /api/webhooks/endpoints/{endpoint_id}/test/
Authorization: Bearer <token>
X-Client-ID: <client_id>
Content-Type: application/json

{
    "event_type": "recognition.success",
    "test_data": {
        "user_id": "test_user_123",
        "similarity_score": 0.95
    }
}
```

**Response:**
```json
{
    "test_successful": true,
    "delivery_id": "del_test_xyz789",
    "response_status": 200,
    "response_time_ms": 145,
    "webhook_response": {
        "status": "received",
        "processed": true
    }
}
```

### 6. Analytics & Reporting

#### Get Usage Statistics
```http
GET /api/analytics/usage/
Authorization: Bearer <token>
X-Client-ID: <client_id>
```

**Query Parameters:**
- `period`: `day`, `week`, `month` (default: `day`)
- `start_date`: Start date (ISO format)
- `end_date`: End date (ISO format)

**Response:**
```json
{
    "period": "day",
    "start_date": "2025-10-09",
    "end_date": "2025-10-09",
    "metrics": {
        "total_sessions": 150,
        "enrollment_sessions": 45,
        "recognition_sessions": 105,
        "successful_enrollments": 42,
        "successful_recognitions": 98,
        "failed_attempts": 7,
        "average_processing_time_ms": 187.5,
        "success_rate_percentage": 93.3
    },
    "rate_limiting": {
        "hourly_usage": 125,
        "hourly_limit": 1000,
        "daily_usage": 1450,
        "daily_limit": 10000,
        "usage_percentage": 14.5
    },
    "quality_metrics": {
        "average_face_quality": 0.87,
        "average_liveness_score": 0.91,
        "quality_rejections": 8
    }
}
```

#### Get System Health
```http
GET /api/analytics/health/
Authorization: Bearer <token>
X-Client-ID: <client_id>
```

**Response:**
```json
{
    "system_status": "healthy",
    "services": {
        "face_recognition_engine": "operational",
        "database": "operational",
        "webhook_delivery": "operational",
        "file_storage": "operational"
    },
    "performance": {
        "average_response_time_ms": 156.7,
        "p95_response_time_ms": 298.2,
        "requests_per_second": 12.4,
        "error_rate_percentage": 0.8
    },
    "capacity": {
        "current_load_percentage": 23.5,
        "active_sessions": 89,
        "queue_length": 2
    },
    "last_updated": "2025-10-09T17:25:00Z"
}
```

## Webhook Events

### Event Payload Structure

Semua webhook events menggunakan struktur payload yang konsisten:

```json
{
    "event_id": "evt_1234567890abcdef",
    "event_name": "recognition.success",
    "timestamp": "2025-10-09T17:30:00Z",
    "client_id": "FR_DEMO123",
    "data": {
        // Event-specific data
    },
    "signature": "sha256=a8b7c6d5e4f3..."
}
```

### Supported Events

#### 1. Session Events

**session.created**
```json
{
    "event_name": "session.created",
    "data": {
        "session_id": "sess_abc123",
        "session_type": "enrollment",
        "user_id": "user_123",
        "device_info": {...},
        "expires_at": "2025-10-09T18:00:00Z"
    }
}
```

**session.expired**
```json
{
    "event_name": "session.expired",
    "data": {
        "session_id": "sess_abc123",
        "session_type": "enrollment",
        "user_id": "user_123",
        "duration_minutes": 15,
        "completed": false
    }
}
```

#### 2. Enrollment Events

**enrollment.completed**
```json
{
    "event_name": "enrollment.completed",
    "data": {
        "session_id": "sess_abc123",
        "user_id": "user_123",
        "enrollment_id": "enr_xyz789",
        "total_samples": 5,
        "average_quality": 0.89,
        "processing_time_ms": 1250,
        "completed_at": "2025-10-09T17:35:00Z"
    }
}
```

**enrollment.failed**
```json
{
    "event_name": "enrollment.failed",
    "data": {
        "session_id": "sess_abc123",
        "user_id": "user_123",
        "failure_reason": "insufficient_samples",
        "samples_collected": 2,
        "samples_required": 5,
        "failed_at": "2025-10-09T17:45:00Z"
    }
}
```

#### 3. Recognition Events

**recognition.success**
```json
{
    "event_name": "recognition.success",
    "data": {
        "session_id": "sess_recognition_abc",
        "recognized_user_id": "user_123",
        "similarity_score": 0.94,
        "confidence_score": 0.91,
        "liveness_score": 0.89,
        "processing_time_ms": 245,
        "timestamp": "2025-10-09T17:40:00Z"
    }
}
```

**recognition.failed**
```json
{
    "event_name": "recognition.failed",
    "data": {
        "session_id": "sess_recognition_abc",
        "failure_reason": "no_match",
        "highest_similarity_score": 0.45,
        "liveness_score": 0.89,
        "processing_time_ms": 198,
        "timestamp": "2025-10-09T17:41:00Z"
    }
}
```

### Webhook Security

#### Signature Verification

Setiap webhook payload disertai dengan HMAC signature untuk verifikasi:

```python
import hmac
import hashlib

def verify_webhook_signature(payload_body, signature_header, webhook_secret):
    """
    Verify webhook signature
    
    Args:
        payload_body (str): Raw request body
        signature_header (str): X-Webhook-Signature header value
        webhook_secret (str): Webhook secret from endpoint configuration
    
    Returns:
        bool: True if signature is valid
    """
    expected_signature = hmac.new(
        webhook_secret.encode('utf-8'),
        payload_body.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Extract signature from header (format: "sha256=<signature>")
    received_signature = signature_header.split('sha256=')[1] if 'sha256=' in signature_header else signature_header
    
    return hmac.compare_digest(expected_signature, received_signature)
```

#### Headers

Webhook requests menyertakan headers berikut:

```http
Content-Type: application/json
X-Webhook-Event: recognition.success
X-Webhook-Signature: sha256=a8b7c6d5e4f3...
X-Client-ID: FR_DEMO123
User-Agent: FaceRecognition-Webhook/1.0
```

## Error Handling

### Error Response Format

Semua API errors menggunakan format response yang konsisten:

```json
{
    "error": {
        "type": "validation_error",
        "message": "Request validation failed",
        "details": {
            "face_image": ["This field is required."],
            "session_id": ["Invalid session ID format."]
        },
        "error_code": "VALIDATION_ERROR",
        "timestamp": "2025-10-09T17:50:00Z",
        "request_id": "req_abc123def456"
    }
}
```

### Common Error Codes

| HTTP Code | Error Type | Description |
|-----------|------------|-------------|
| 400 | `validation_error` | Request validation failed |
| 401 | `authentication_error` | Invalid or missing authentication |
| 403 | `permission_denied` | Insufficient permissions |
| 404 | `not_found` | Resource not found |
| 409 | `conflict_error` | Resource conflict (e.g., duplicate user) |
| 422 | `processing_error` | Business logic error |
| 429 | `rate_limit_exceeded` | Rate limit exceeded |
| 500 | `internal_error` | Internal server error |
| 503 | `service_unavailable` | Service temporarily unavailable |

### Specific Error Examples

#### Authentication Errors
```json
{
    "error": {
        "type": "authentication_error",
        "message": "Invalid JWT token",
        "error_code": "INVALID_TOKEN",
        "timestamp": "2025-10-09T17:50:00Z"
    }
}
```

#### Rate Limiting
```json
{
    "error": {
        "type": "rate_limit_exceeded",
        "message": "Hourly rate limit exceeded",
        "details": {
            "limit": 1000,
            "current_usage": 1000,
            "reset_time": "2025-10-09T18:00:00Z"
        },
        "error_code": "RATE_LIMIT_EXCEEDED"
    }
}
```

#### Face Recognition Errors
```json
{
    "error": {
        "type": "processing_error",
        "message": "No face detected in image",
        "details": {
            "image_analysis": {
                "faces_detected": 0,
                "image_quality": "good",
                "image_size": "640x480"
            }
        },
        "error_code": "NO_FACE_DETECTED"
    }
}
```

## Rate Limiting

### Rate Limit Headers

Setiap API response menyertakan rate limit information:

```http
X-RateLimit-Limit-Hour: 1000
X-RateLimit-Remaining-Hour: 847
X-RateLimit-Reset-Hour: 1696860000
X-RateLimit-Limit-Day: 10000
X-RateLimit-Remaining-Day: 8653
X-RateLimit-Reset-Day: 1696896000
```

### Rate Limit Tiers

| Tier | Hourly Limit | Daily Limit | Features |
|------|--------------|-------------|----------|
| Basic | 1,000 | 10,000 | Standard recognition |
| Premium | 5,000 | 50,000 | + Liveness detection |
| Enterprise | 20,000 | 200,000 | + Anti-spoofing + Priority support |

## SDKs & Libraries

### JavaScript/Node.js
```javascript
npm install @face-recognition/sdk

const FaceRecognition = require('@face-recognition/sdk');

const client = new FaceRecognition({
    clientId: 'FR_DEMO123',
    apiKey: 'your-api-key',
    secretKey: 'your-secret-key',
    baseUrl: 'https://api.face-recognition.com'
});

// Create enrollment session
const session = await client.createSession({
    userId: 'user_123',
    sessionType: 'enrollment'
});

// Submit face sample
const result = await client.enrollFace({
    sessionId: session.session_id,
    faceImage: base64Image
});
```

### Python
```python
pip install face-recognition-sdk

from face_recognition_sdk import FaceRecognitionClient

client = FaceRecognitionClient(
    client_id='FR_DEMO123',
    api_key='your-api-key',
    secret_key='your-secret-key',
    base_url='https://api.face-recognition.com'
)

# Create enrollment session
session = client.create_session(
    user_id='user_123',
    session_type='enrollment'
)

# Submit face sample
result = client.enroll_face(
    session_id=session['session_id'],
    face_image=base64_image
)
```

### cURL Examples

#### Get JWT Token
```bash
curl -X POST "https://api.face-recognition.com/api/auth/token/" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "FR_DEMO123",
    "api_key": "your-api-key",
    "secret_key": "your-secret-key"
  }'
```

#### Create Session
```bash
curl -X POST "https://api.face-recognition.com/api/auth-service/sessions/" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "X-Client-ID: FR_DEMO123" \
  -H "Content-Type: application/json" \
  -d '{
    "client_user_id": "user_123",
    "session_type": "enrollment",
    "device_info": {
      "device_id": "web_browser_123",
      "platform": "web"
    }
  }'
```

#### Face Enrollment
```bash
curl -X POST "https://api.face-recognition.com/api/auth-service/enrollments/" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "X-Client-ID: FR_DEMO123" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sess_1234567890",
    "face_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA...",
    "sample_number": 1
  }'
```

## Changelog

### Version 2.0.0 (2025-10-09)
- ✨ Multi-client architecture implementation
- 🔒 Enhanced security with encrypted face embeddings
- 🔄 Comprehensive webhook system
- 📊 Advanced analytics and monitoring
- 🚀 Performance optimizations
- 📚 Complete API documentation

### Version 1.5.0 (2025-09-15)
- 🔐 JWT authentication implementation
- 👤 Custom user management
- 📱 Session-based face operations
- 🎯 Rate limiting per client

### Version 1.0.0 (2025-08-01)
- 🎉 Initial release
- 👤 Basic face enrollment and recognition
- 🔧 Admin interface
- 📖 API documentation

---

*Last Updated: October 9, 2025*
*API Version: 2.0.0*
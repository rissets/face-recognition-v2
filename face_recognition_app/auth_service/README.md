# Auth Service Module

## Overview

The `auth_service` module is the core component responsible for handling face recognition authentication and enrollment operations. It provides both REST API and WebSocket interfaces for real-time face processing, liveness detection, and user authentication.

## Architecture

```
auth_service/
├── models.py              # Database models for sessions, enrollments, and logs
├── views.py               # REST API endpoints
├── consumers.py           # WebSocket consumers for real-time processing
├── serializers.py         # API request/response serializers
├── authentication.py      # Custom authentication backends
├── urls.py               # URL routing configuration
└── WEBSOCKET_GUIDE.md    # Detailed WebSocket documentation
```

## Core Components

### 1. Models (`models.py`)

#### AuthenticationSession

Manages face authentication sessions with support for both REST and WebSocket protocols.

```python
class AuthenticationSession(models.Model):
    session_token = models.UUIDField(unique=True)
    client = models.ForeignKey(Client)
    user_id = models.CharField(max_length=255)
    status = models.CharField(max_length=20)
    require_liveness = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    completed_at = models.DateTimeField(null=True)
```

**Key Methods:**

- `is_expired()`: Check if session has expired
- `mark_completed()`: Mark session as completed
- `add_attempt()`: Record authentication attempt

#### FaceEnrollment

Stores face embeddings and enrollment metadata for users.

```python
class FaceEnrollment(models.Model):
    enrollment_id = models.UUIDField(unique=True)
    client = models.ForeignKey(Client)
    user_id = models.CharField(max_length=255)
    embedding = EncryptedBinaryField()  # AES-256 encrypted
    quality_score = models.FloatField()
    liveness_verified = models.BooleanField(default=False)
    status = models.CharField(max_length=20)
```

**Features:**

- Automatic embedding encryption
- Quality score tracking
- Liveness verification status
- Multiple enrollments per user support

#### FaceRecognitionAttempt

Logs every face recognition attempt for audit and analytics.

```python
class FaceRecognitionAttempt(models.Model):
    session = models.ForeignKey(AuthenticationSession)
    success = models.BooleanField()
    confidence = models.FloatField()
    matched_enrollment = models.ForeignKey(FaceEnrollment, null=True)
    liveness_passed = models.BooleanField()
    processing_time_ms = models.IntegerField()
    failure_reason = models.CharField(max_length=100)
```

### 2. REST API Views (`views.py`)

#### Client Authentication

```python
@api_view(['POST'])
def client_authentication(request):
    """
    Authenticate client using API key and return JWT token.
    
    POST /api/auth/client/
    {
        "api_key": "your_api_key"
    }
    
    Returns:
    {
        "jwt_token": "...",
        "expires_at": "2025-11-26T10:00:00Z",
        "client_id": "uuid"
    }
    """
```

#### Enrollment Session Creation

```python
@api_view(['POST'])
@authentication_classes([JWTClientAuthentication])
def create_enrollment_session(request):
    """
    Create new enrollment session for a user.
    
    POST /api/auth/enrollment/
    Headers: Authorization: Bearer {jwt_token}
    Body:
    {
        "user_id": "external_user_123",
        "metadata": {
            "target_samples": 5,
            "enable_quality_check": true
        }
    }
    
    Returns:
    {
        "session_token": "uuid",
        "websocket_url": "wss://domain.com/ws/auth/process-image/uuid/",
        "expires_at": "2025-11-25T10:30:00Z",
        "status": "active"
    }
    """
```

#### Authentication Session Creation

```python
@api_view(['POST'])
@authentication_classes([JWTClientAuthentication])
def create_authentication_session(request):
    """
    Create new authentication session.
    
    POST /api/auth/authentication/
    Headers: Authorization: Bearer {jwt_token}
    Body:
    {
        "user_id": "external_user_123",  // Optional
        "require_liveness": true,
        "metadata": {
            "min_confidence": 0.85
        }
    }
    
    Returns:
    {
        "session_token": "uuid",
        "websocket_url": "wss://domain.com/ws/auth/process-image/uuid/",
        "expires_at": "2025-11-25T10:35:00Z",
        "status": "active"
    }
    """
```

#### Frame Processing (REST)

```python
@api_view(['POST'])
@authentication_classes([JWTClientAuthentication])
def process_frame(request):
    """
    Process a single frame for enrollment or authentication.
    
    POST /api/auth/enrollment/process-frame/
    POST /api/auth/authentication/process-frame/
    
    Body:
    {
        "session_token": "uuid",
        "frame_data": "data:image/jpeg;base64,..."
    }
    
    Returns (enrollment):
    {
        "success": true,
        "session_status": "in_progress",
        "completed_samples": 2,
        "target_samples": 5,
        "quality_score": 0.85,
        "liveness_data": {...}
    }
    
    Returns (authentication):
    {
        "success": true,
        "authenticated": true,
        "confidence": 0.94,
        "user_id": "external_user_123",
        "liveness_passed": true
    }
    """
```

### 3. WebSocket Consumer (`consumers.py`)

#### FaceProcessingConsumer

Real-time WebSocket handler for face processing.

```python
class FaceProcessingConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time face recognition processing.
    
    URL: /ws/auth/process-image/{session_token}/
    """
    
    async def connect(self):
        """Validate session token and establish connection"""
        
    async def receive(self, text_data):
        """Process incoming frame data"""
        
    async def disconnect(self, close_code):
        """Clean up resources on disconnect"""
```

**Message Types:**

**Client → Server:**

```json
{
  "type": "process_frame",
  "frame_data": "data:image/jpeg;base64,...",
  "timestamp": "2025-11-25T10:00:00Z"
}
```

**Server → Client:**

```json
// Enrollment Progress
{
  "type": "enrollment_progress",
  "completed_samples": 3,
  "target_samples": 5,
  "quality_score": 0.87,
  "liveness_data": {
    "blinks_detected": 2,
    "motion_score": 0.78
  }
}

// Enrollment Complete
{
  "type": "enrollment_complete",
  "enrollment_id": "uuid",
  "completed_samples": 5,
  "liveness_verified": true
}

// Authentication Result
{
  "type": "authentication_result",
  "authenticated": true,
  "user_id": "external_user_123",
  "confidence": 0.96
}

// Error
{
  "type": "error",
  "error_code": "poor_image_quality",
  "message": "Image quality too low"
}
```

**Key Features:**

- Automatic session validation
- Real-time frame processing
- Progressive feedback
- Error handling and recovery
- Automatic session cleanup

### 4. Authentication Backends (`authentication.py`)

#### APIKeyAuthentication

```python
class APIKeyAuthentication(BaseAuthentication):
    """
    Authenticate requests using API key in X-API-Key header.
    
    Usage:
    Headers: X-API-Key: your_api_key
    """
```

#### JWTClientAuthentication

```python
class JWTClientAuthentication(BaseAuthentication):
    """
    Authenticate requests using JWT token issued to clients.
    
    Usage:
    Headers: Authorization: Bearer {jwt_token}
    """
```

### 5. Serializers (`serializers.py`)

#### Request Serializers

- `EnrollmentRequestSerializer`: Validates enrollment session creation
- `AuthenticationRequestSerializer`: Validates authentication session creation
- `FaceFrameSerializer`: Validates frame data
- `SessionTokenSerializer`: Validates session tokens

#### Response Serializers

- `EnrollmentResponseSerializer`: Formats enrollment responses
- `AuthenticationResponseSerializer`: Formats authentication responses
- `SessionStatusSerializer`: Formats session status
- `ErrorResponseSerializer`: Standardizes error responses

## Integration Guide

### 1. Client Authentication Flow

```python
import requests

# Step 1: Authenticate with API key
response = requests.post(
    'https://api.example.com/api/auth/client/',
    json={'api_key': 'your_api_key'}
)
jwt_token = response.json()['jwt_token']

# Use JWT token for subsequent requests
headers = {'Authorization': f'Bearer {jwt_token}'}
```

### 2. Enrollment Flow (REST API)

```python
# Step 1: Create enrollment session
response = requests.post(
    'https://api.example.com/api/auth/enrollment/',
    headers=headers,
    json={
        'user_id': 'user123',
        'metadata': {'target_samples': 5}
    }
)
session_token = response.json()['session_token']

# Step 2: Send frames
for frame in capture_frames():
    response = requests.post(
        'https://api.example.com/api/auth/enrollment/process-frame/',
        headers=headers,
        json={
            'session_token': session_token,
            'frame_data': frame_to_base64(frame)
        }
    )
    
    result = response.json()
    if result['session_status'] == 'completed':
        print(f"Enrollment completed: {result['enrollment_id']}")
        break
```

### 3. Enrollment Flow (WebSocket)

```python
import asyncio
import websockets
import json

async def enroll_user():
    # Step 1: Create session (REST API)
    session_token = create_enrollment_session()
    
    # Step 2: Connect to WebSocket
    ws_url = f'wss://api.example.com/ws/auth/process-image/{session_token}/'
    
    async with websockets.connect(ws_url) as websocket:
        # Step 3: Send frames
        async for frame in capture_frames_async():
            await websocket.send(json.dumps({
                'type': 'process_frame',
                'frame_data': frame_to_base64(frame)
            }))
            
            # Step 4: Receive responses
            response = await websocket.recv()
            data = json.loads(response)
            
            if data['type'] == 'enrollment_complete':
                print(f"Enrollment completed: {data['enrollment_id']}")
                break
            elif data['type'] == 'enrollment_progress':
                print(f"Progress: {data['completed_samples']}/{data['target_samples']}")

asyncio.run(enroll_user())
```

### 4. Authentication Flow (WebSocket)

```python
async def authenticate_user():
    # Step 1: Create authentication session
    session_token = create_authentication_session()
    
    # Step 2: Connect to WebSocket
    ws_url = f'wss://api.example.com/ws/auth/process-image/{session_token}/'
    
    async with websockets.connect(ws_url) as websocket:
        # Step 3: Send frames until authenticated
        async for frame in capture_frames_async():
            await websocket.send(json.dumps({
                'type': 'process_frame',
                'frame_data': frame_to_base64(frame)
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            
            if data['type'] == 'authentication_result':
                if data['authenticated']:
                    print(f"Authentication successful: {data['user_id']}")
                    print(f"Confidence: {data['confidence']}")
                else:
                    print(f"Authentication failed: {data.get('reason')}")
                break

asyncio.run(authenticate_user())
```

## Configuration

### Settings

Key settings in `settings.py`:

```python
# Face Recognition Settings
FACE_RECOGNITION_THRESHOLD = 0.65  # Minimum confidence for match
FACE_MIN_QUALITY_SCORE = 0.6       # Minimum image quality
FACE_MIN_LIVENESS_FRAMES = 3       # Minimum frames for liveness
FACE_MIN_LIVENESS_BLINKS = 1       # Minimum blinks required

# Session Settings
SESSION_EXPIRY_MINUTES = 30         # Session timeout
MAX_ENROLLMENT_ATTEMPTS = 50        # Max frames per enrollment
MAX_AUTH_ATTEMPTS = 20              # Max frames per authentication

# ChromaDB Settings
CHROMADB_HOST = 'localhost'
CHROMADB_PORT = 8000
CHROMADB_COLLECTION = 'face_embeddings'

# Encryption
FIELD_ENCRYPTION_KEY = 'your-encryption-key'
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379/0

# ChromaDB
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Security
SECRET_KEY=your-django-secret-key
FIELD_ENCRYPTION_KEY=your-encryption-key

# API Keys (for testing)
TEST_CLIENT_API_KEY=test_key_here
```

## Testing

### Unit Tests

```bash
# Run all auth_service tests
python manage.py test auth_service

# Run specific test class
python manage.py test auth_service.tests.EnrollmentTestCase

# Run with coverage
coverage run --source='auth_service' manage.py test auth_service
coverage report
```

### Example Tests

```python
from django.test import TestCase
from auth_service.models import AuthenticationSession, FaceEnrollment
from clients.models import Client

class EnrollmentTestCase(TestCase):
    def setUp(self):
        self.client = Client.objects.create(name='Test Client')
        
    def test_create_enrollment_session(self):
        """Test enrollment session creation"""
        response = self.client.post('/api/auth/enrollment/', {
            'user_id': 'test_user_123',
            'metadata': {'target_samples': 3}
        })
        
        self.assertEqual(response.status_code, 201)
        self.assertIn('session_token', response.json())
```

### Manual Testing

```bash
# Test client authentication
curl -X POST http://localhost:8000/api/auth/client/ \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your_test_api_key"}' | jq

# Test enrollment session creation
curl -X POST http://localhost:8000/api/auth/enrollment/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "metadata": {"target_samples": 3}
  }' | jq
```

## Performance Considerations

### Optimization Tips

1. **Use WebSocket for real-time processing**: More efficient than REST API for multiple frames
2. **Batch frame processing**: Process frames in batches when possible
3. **Cache embeddings**: Use Redis to cache frequently accessed embeddings
4. **Index database queries**: Ensure proper indexes on session_token, user_id
5. **Connection pooling**: Configure Django database connection pooling

### Monitoring

```python
# Check active sessions
from auth_service.models import AuthenticationSession
active_sessions = AuthenticationSession.objects.filter(
    status='active',
    expires_at__gt=timezone.now()
).count()

# Monitor processing times
from django.db.models import Avg
avg_time = FaceRecognitionAttempt.objects.aggregate(
    Avg('processing_time_ms')
)['processing_time_ms__avg']
```

## Security Considerations

1. **Always use HTTPS in production**: Protect API keys and JWT tokens
2. **Rotate API keys regularly**: Every 90 days recommended
3. **Set appropriate session expiry**: Balance security and UX
4. **Monitor failed attempts**: Detect potential abuse
5. **Rate limiting**: Prevent brute force attacks
6. **Audit logging**: Track all authentication attempts

## Troubleshooting

### Common Issues

**Session expired errors:**
- Check `expires_at` field
- Increase `SESSION_EXPIRY_MINUTES` if needed
- Ensure proper timezone handling

**WebSocket connection refused:**
- Verify Daphne/Channels is running
- Check WebSocket URL format
- Verify session token is valid

**Low recognition accuracy:**
- Adjust `FACE_RECOGNITION_THRESHOLD`
- Improve image quality
- Re-enroll users with better samples

**Liveness detection failing:**
- Check MediaPipe installation
- Adjust `FACE_MIN_LIVENESS_BLINKS`
- Review liveness threshold settings

## Additional Resources

- [WebSocket Guide](./WEBSOCKET_GUIDE.md) - Detailed WebSocket documentation
- [Main README](../README.md) - Complete application documentation
- [Authentication Guide](../docs/authentication.md) - Authentication flow details
- [Enrollment Guide](../docs/enrollment.md) - Enrollment process details

## Support

For issues specific to the auth_service module:

1. Check the logs: `logs/auth_service.log`
2. Review model definitions in `models.py`
3. Check consumer implementation in `consumers.py`
4. Verify authentication backend configuration
5. Test with provided cURL examples

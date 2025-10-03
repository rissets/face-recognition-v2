# Face Recognition Django Application

A comprehensive Django application for real-time face recognition authentication with liveness detection, obstacle detection, and anti-spoofing capabilities.

## Features

- 🔐 **Real-time Face Authentication**: WebSocket-based real-time face recognition
- 👁️ **Liveness Detection**: Blink detection using MediaPipe
- 🚫 **Obstacle Detection**: Advanced obstacle detection for enhanced security
- 🎯 **InsightFace Integration**: High-accuracy face recognition using InsightFace models
- 📊 **ChromaDB Vector Storage**: Efficient embedding storage and similarity search
- 🔒 **Security Features**: Data encryption, JWT authentication, rate limiting
- 📈 **Analytics & Monitoring**: Comprehensive logging and metrics
- 🎛️ **Modern Admin Interface**: Django-unfold admin with custom dashboards
- 🌊 **Background Processing**: Celery for async tasks
- 🔄 **WebRTC Support**: Real-time video streaming

## Technology Stack

- **Backend**: Django 5.2.7, Django REST Framework
- **Real-time**: Django Channels, WebSockets, WebRTC
- **Face Recognition**: InsightFace, MediaPipe, OpenCV
- **Vector Database**: ChromaDB, FAISS (fallback)
- **Database**: PostgreSQL with pgvector extension
- **Cache/Queue**: Redis, Celery
- **Authentication**: JWT, Custom User Model
- **Admin**: Django-unfold
- **Security**: Cryptography, Rate limiting

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd face_recognition_app

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configurations
nano .env
```

### 3. Database Setup

```bash
# Install PostgreSQL with pgvector extension
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres psql
CREATE EXTENSION vector;

# Create database
createdb face_recognition_db

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

### 4. InsightFace Models Setup

```bash
# Download InsightFace models
mkdir -p models/insightface
cd models/insightface

# Download models (example - adjust URLs as needed)
wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip buffalo_l.zip
```

### 5. Redis Setup

```bash
# Install and start Redis
# Ubuntu/Debian:
sudo apt-get install redis-server
sudo systemctl start redis-server

# macOS:
brew install redis
brew services start redis
```

### 6. Start Services

```bash
# Terminal 1: Django development server
python manage.py runserver

# Terminal 2: Celery worker
celery -A face_app worker --loglevel=info

# Terminal 3: Celery beat (for scheduled tasks)
celery -A face_app beat --loglevel=info

# Terminal 4: ChromaDB (if using separate instance)
chroma run --host localhost --port 8000
```

## API Endpoints

### Authentication
- `POST /api/auth/register/` - User registration
- `POST /api/auth/login/` - User login
- `POST /api/auth/refresh/` - Token refresh
- `POST /api/auth/logout/` - User logout

### Face Recognition
- `POST /api/core/enroll/start/` - Start enrollment session
- `POST /api/core/enroll/frame/` - Process enrollment frame
- `POST /api/core/enroll/complete/` - Complete enrollment
- `POST /api/core/auth/frame/` - Process authentication frame
- `GET /api/core/auth/status/` - Check authentication status

### WebRTC Signaling
- `POST /api/streaming/webrtc/offer/` - WebRTC offer
- `POST /api/streaming/webrtc/answer/` - WebRTC answer
- `POST /api/streaming/webrtc/ice/` - ICE candidate

### Analytics
- `GET /api/analytics/user-activity/` - User activity logs
- `GET /api/analytics/system-metrics/` - System metrics
- `GET /api/analytics/performance/` - Performance statistics

## WebSocket Connections

### Face Recognition WebSocket
```javascript
// Connect to face recognition WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/face-recognition/');

// Send frame for processing
ws.send(JSON.stringify({
    'type': 'process_frame',
    'frame_data': base64_image_data,
    'session_type': 'enrollment'  // or 'authentication'
}));
```

### WebRTC Signaling WebSocket
```javascript
// Connect to WebRTC signaling
const signalingWs = new WebSocket('ws://localhost:8000/ws/webrtc-signaling/');

// Send WebRTC offer
signalingWs.send(JSON.stringify({
    'type': 'offer',
    'offer': rtcOffer
}));
```

## Frontend Integration Guide

### 1. Camera Access and Frame Capture

```javascript
// Get camera access
async function initCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
            width: 640, 
            height: 480,
            facingMode: 'user'
        }
    });
    
    const video = document.getElementById('video');
    video.srcObject = stream;
    return stream;
}

// Capture frame from video
function captureFrame(video) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.8);
}
```

### 2. WebSocket Integration

```javascript
class FaceRecognitionClient {
    constructor(wsUrl) {
        this.ws = new WebSocket(wsUrl);
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
    }
    
    startEnrollment(userId) {
        this.ws.send(JSON.stringify({
            'type': 'start_enrollment',
            'user_id': userId
        }));
    }
    
    processFrame(frameData, sessionType) {
        this.ws.send(JSON.stringify({
            'type': 'process_frame',
            'frame_data': frameData,
            'session_type': sessionType
        }));
    }
    
    handleMessage(data) {
        switch(data.type) {
            case 'enrollment_progress':
                this.updateEnrollmentProgress(data.progress);
                break;
            case 'authentication_result':
                this.handleAuthResult(data.result);
                break;
            case 'liveness_check':
                this.handleLivenessCheck(data.status);
                break;
        }
    }
}
```

### 3. Complete Enrollment Flow

```javascript
async function performEnrollment(userId) {
    const client = new FaceRecognitionClient('ws://localhost:8000/ws/face-recognition/');
    const stream = await initCamera();
    const video = document.getElementById('video');
    
    client.startEnrollment(userId);
    
    // Capture frames at intervals
    const frameCapture = setInterval(() => {
        const frameData = captureFrame(video);
        client.processFrame(frameData, 'enrollment');
    }, 100); // 10 FPS
    
    // Stop after enrollment completion
    client.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'enrollment_complete') {
            clearInterval(frameCapture);
            stream.getTracks().forEach(track => track.stop());
        }
    };
}
```

### 4. Authentication Flow

```javascript
async function performAuthentication() {
    const client = new FaceRecognitionClient('ws://localhost:8000/ws/face-recognition/');
    const stream = await initCamera();
    const video = document.getElementById('video');
    
    const frameCapture = setInterval(() => {
        const frameData = captureFrame(video);
        client.processFrame(frameData, 'authentication');
    }, 200); // 5 FPS for authentication
    
    client.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'authentication_result') {
            if (data.result.success) {
                window.location.href = '/dashboard/';
            } else {
                showError('Authentication failed');
            }
            clearInterval(frameCapture);
            stream.getTracks().forEach(track => track.stop());
        }
    };
}
```

## Testing

### Run Tests
```bash
# Run all tests
python manage.py test

# Run specific test modules
python manage.py test core.tests
python manage.py test users.tests

# Run with coverage
coverage run --source='.' manage.py test
coverage report
```

### Test Registration Process
```python
from django.test import TestCase
from django.contrib.auth import get_user_model
from core.models import EnrollmentSession

User = get_user_model()

class RegistrationTestCase(TestCase):
    def test_user_registration(self):
        """Test user registration process"""
        response = self.client.post('/api/auth/register/', {
            'email': 'test@example.com',
            'password': 'testpassword123',
            'first_name': 'Test',
            'last_name': 'User'
        })
        self.assertEqual(response.status_code, 201)
        self.assertTrue(User.objects.filter(email='test@example.com').exists())
```

### Test Enrollment Process
```python
class EnrollmentTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpassword123'
        )
    
    def test_enrollment_start(self):
        """Test enrollment session start"""
        self.client.force_authenticate(user=self.user)
        response = self.client.post('/api/core/enroll/start/')
        self.assertEqual(response.status_code, 201)
        
        session = EnrollmentSession.objects.get(user=self.user)
        self.assertEqual(session.status, 'in_progress')
```

## Security Considerations

1. **Data Encryption**: All face embeddings and personal data are encrypted
2. **Rate Limiting**: API endpoints have rate limiting to prevent abuse
3. **CORS Configuration**: Properly configured for production
4. **JWT Security**: Secure token handling with refresh mechanism
5. **Input Validation**: All inputs are validated and sanitized
6. **Audit Logging**: Comprehensive logging of all security events

## Production Deployment

### Docker Deployment
```dockerfile
# Dockerfile example
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "face_app.wsgi:application", "--bind", "0.0.0.0:8000"]
```

### Environment Variables (Production)
```bash
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
SECURE_SSL_REDIRECT=True
SECURE_HSTS_SECONDS=31536000
```

## Monitoring and Maintenance

- **Health Checks**: `/health/` endpoint for monitoring
- **Metrics**: Prometheus-compatible metrics at `/metrics/`
- **Admin Interface**: Comprehensive admin at `/admin/`
- **Logs**: Structured logging with rotation
- **Celery Monitoring**: Use Flower for Celery task monitoring

## Support

For issues and questions:
1. Check the documentation
2. Review the logs in `/var/log/face_recognition/`
3. Monitor the admin interface for system status
4. Check Celery task queue status

## License

[Your License Here]
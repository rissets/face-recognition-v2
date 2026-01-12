# WebSocket Face Authentication - Quick Start

## 🚀 Quick Start Guide

### Prerequisites

```bash
# Install dependencies
pip install channels channels-redis websockets cryptography
```

### 1. Start the Server

```bash
# Make sure Redis is running
redis-server

# Start Django server
python manage.py runserver
```

### 2. Test with Python Script

```bash
# Enrollment
python test_websocket_auth.py \
  YOUR_API_KEY \
  YOUR_SECRET_KEY \
  http://localhost:8000 \
  enrollment \
  user123

# Authentication (Verification)
python test_websocket_auth.py \
  YOUR_API_KEY \
  YOUR_SECRET_KEY \
  http://localhost:8000 \
  authentication \
  user123

# Authentication (Identification - without user_id)
python test_websocket_auth.py \
  YOUR_API_KEY \
  YOUR_SECRET_KEY \
  http://localhost:8000 \
  authentication
```

### 3. Test with HTML Demo

```bash
# Open in browser
open websocket_demo.html
# or
python -m http.server 8080
# then navigate to http://localhost:8080/websocket_demo.html
```

## 📚 Documentation

- **[WEBSOCKET_GUIDE.md](face_recognition_app/auth_service/WEBSOCKET_GUIDE.md)** - Complete client documentation
- **[WEBSOCKET_IMPLEMENTATION.md](WEBSOCKET_IMPLEMENTATION.md)** - Implementation guide (Indonesian)
- **[WEBSOCKET_DIAGRAMS.md](WEBSOCKET_DIAGRAMS.md)** - Architecture diagrams
- **[WEBSOCKET_SUMMARY.md](WEBSOCKET_SUMMARY.md)** - Implementation summary

## 🔑 API Usage

### Step 1: Authenticate Client

```bash
curl -X POST http://localhost:8000/api/auth/client/ \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "YOUR_API_KEY",
    "secret_key": "YOUR_SECRET_KEY"
  }'
```

**Response:**
```json
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "client_id": "client123",
  "expires_at": "2025-10-27T13:00:00Z"
}
```

### Step 2: Create Session

```bash
curl -X POST http://localhost:8000/api/auth/enrollment/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "metadata": {"target_samples": 3}
  }'
```

**Response:**
```json
{
  "session_token": "abc123...",
  "enrollment_id": "uuid-here",
  "websocket_url": "ws://localhost:8000/ws/auth/process-image/abc123/",
  "status": "pending",
  "target_samples": 3,
  "expires_at": "2025-10-27T12:00:00Z"
}
```

### Step 3: Connect WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/auth/process-image/abc123/');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Step 4: Send Frames

```javascript
// Capture from video
const canvas = document.createElement('canvas');
canvas.getContext('2d').drawImage(video, 0, 0);
const imageData = canvas.toDataURL('image/jpeg', 0.8);

// Send frame
ws.send(JSON.stringify({
  type: 'frame',
  image: imageData
}));
```

### Step 5: Handle Completion

```javascript
if (data.type === 'enrollment_complete') {
  // Decrypt response
  const decrypted = decryptResponse(
    data.encrypted_data.encrypted_payload,
    YOUR_SECRET_KEY
  );
  console.log('Enrollment ID:', decrypted.id);
  console.log('Timestamp:', decrypted.timestamp);
}
```

## 🔐 Decryption Example (Python)

```python
import base64
import hashlib
import json
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

def decrypt_response(encrypted_payload: str, secret_key: str) -> dict:
    # Decode base64
    encrypted_data = base64.b64decode(encrypted_payload)
    
    # Extract IV and ciphertext
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    
    # Derive key
    key = hashlib.sha256(secret_key.encode()).digest()
    
    # Decrypt
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    
    # Unpad
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    
    # Parse JSON
    return json.loads(data.decode('utf-8'))

# Usage
encrypted_payload = response['encrypted_data']['encrypted_payload']
result = decrypt_response(encrypted_payload, 'your_secret_key')
print(result)
# {'id': 'abc123', 'timestamp': '2025-10-27T12:34:56.789Z', ...}
```

## 🔐 Decryption Example (JavaScript)

```javascript
// Using CryptoJS library
function decryptResponse(encryptedPayload, secretKey) {
  const encrypted = CryptoJS.enc.Base64.parse(encryptedPayload);
  
  // Extract IV and ciphertext
  const iv = CryptoJS.lib.WordArray.create(encrypted.words.slice(0, 4));
  const ciphertext = CryptoJS.lib.WordArray.create(encrypted.words.slice(4));
  
  // Derive key
  const key = CryptoJS.SHA256(secretKey);
  
  // Decrypt
  const decrypted = CryptoJS.AES.decrypt(
    { ciphertext: ciphertext },
    key,
    { iv: iv, mode: CryptoJS.mode.CBC, padding: CryptoJS.pad.Pkcs7 }
  );
  
  return JSON.parse(decrypted.toString(CryptoJS.enc.Utf8));
}

// Usage
const result = decryptResponse(
  response.encrypted_data.encrypted_payload,
  'your_secret_key'
);
console.log(result);
```

## 📊 Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `frame` | Client→Server | Face image frame |
| `ping` | Client→Server | Keep-alive |
| `connection_established` | Server→Client | Connection success |
| `frame_processed` | Server→Client | Frame result |
| `enrollment_complete` | Server→Client | Enrollment done |
| `authentication_complete` | Server→Client | Auth done |
| `error` | Server→Client | Error occurred |
| `pong` | Server→Client | Ping response |

## ⚙️ Configuration

### Frame Processing
- **Max FPS**: 10 frames per second
- **Max Frames**: 120 per session
- **Min Frames**: 3 for completion
- **Throttle**: 100ms minimum interval

### Session
- **Timeout**: 5 minutes
- **Max Concurrent**: Unlimited (resource-dependent)

### Image
- **Format**: JPEG, PNG
- **Quality**: 80% recommended
- **Resolution**: 640x480 or 1280x720 recommended
- **Max Size**: 100KB recommended

## 🐛 Troubleshooting

### WebSocket won't connect
```bash
# Check session is active
curl http://localhost:8000/api/auth/sessions/{id}/status/ \
  -H "Authorization: Bearer YOUR_API_KEY"

# Check Redis is running
redis-cli ping

# Check Django Channels
python manage.py check
```

### Decryption fails
- Verify secret key is correct
- Check encrypted payload is complete
- Ensure proper base64 decoding

### Liveness detection fails
- Ensure good lighting
- Ask user to blink naturally
- Allow natural head movement

## 📁 File Structure

```
.
├── face_recognition_app/
│   └── auth_service/
│       ├── consumers.py              # WebSocket consumer
│       ├── views.py                  # REST API (modified)
│       ├── serializers.py            # Serializers (modified)
│       └── WEBSOCKET_GUIDE.md        # Documentation
├── streaming/
│   └── routing.py                    # WebSocket routing (modified)
├── test_websocket_auth.py            # Python test script
├── websocket_demo.html               # HTML demo
├── WEBSOCKET_IMPLEMENTATION.md       # Implementation guide
├── WEBSOCKET_DIAGRAMS.md             # Architecture diagrams
└── WEBSOCKET_SUMMARY.md              # Summary
```

## 🆘 Support

For issues or questions:
1. Check documentation files
2. Run test scripts for debugging
3. Check server logs
4. Verify Redis and Django are running

## 📝 License

Same as the main project.

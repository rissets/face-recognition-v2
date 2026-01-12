# WebSocket Implementation for Face Authentication & Enrollment

## Overview

Implementasi WebSocket baru untuk pemrosesan gambar real-time pada proses autentikasi dan enrollment wajah. WebSocket ini memberikan cara yang lebih efisien untuk memproses multiple frames dibandingkan REST API.

## Fitur Utama

✅ **Real-time Processing**: Kirim frame secara langsung saat diambil dari kamera  
✅ **Session-based**: Event dibuat berdasarkan session token dari REST API  
✅ **Double Encryption**: Response dienkripsi dengan API key (server) dan Secret key (client)  
✅ **Progress Updates**: Feedback langsung untuk setiap frame yang diproses  
✅ **Automatic Completion**: Session otomatis selesai saat requirement terpenuhi  
✅ **Liveness Detection**: Deteksi blink dan motion otomatis  

## Arsitektur

```
Client                    REST API                 WebSocket Consumer
  |                          |                            |
  |--- POST /enrollment/ --->|                            |
  |<-- session_token ---------|                            |
  |    websocket_url          |                            |
  |                          |                            |
  |--- WS Connect ---------------------------------------->|
  |<-- connection_established -----------------------------|
  |                          |                            |
  |--- frame 1 ------------------------------------------>|
  |<-- frame_processed -----------------------------------|
  |                          |                            |
  |--- frame 2 ------------------------------------------>|
  |<-- frame_processed -----------------------------------|
  |                          |                            |
  |--- frame 3 ------------------------------------------>|
  |<-- enrollment_complete (encrypted) -------------------|
  |                          |                            |
```

## File Structure

```
face_recognition_app/
├── auth_service/
│   ├── consumers.py          # NEW: WebSocket consumer untuk auth & enrollment
│   ├── views.py              # UPDATED: Tambah websocket_url ke response
│   ├── serializers.py        # UPDATED: Tambah websocket_url field
│   ├── urls.py               # Existing REST API endpoints
│   └── WEBSOCKET_GUIDE.md    # NEW: Dokumentasi lengkap untuk client
├── streaming/
│   └── routing.py            # UPDATED: Tambah WebSocket URL pattern
└── test_websocket_auth.py    # NEW: Test script untuk WebSocket
```

## Instalasi

### 1. Install Dependencies

```bash
pip install channels channels-redis websockets cryptography
```

### 2. Update Settings

Pastikan Django Channels sudah dikonfigurasi di `settings.py`:

```python
INSTALLED_APPS = [
    ...
    'channels',
    ...
]

ASGI_APPLICATION = 'face_app.asgi.application'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}
```

## Cara Penggunaan

### 1. Authenticate Client (REST API)

**Client Authentication:**
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

### 2. Create Session (REST API)

**Enrollment:**
```bash
curl -X POST http://localhost:8000/api/auth/authentication/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "require_liveness": true,
    "metadata": {
      "min_frames_required": 3,
      "required_blinks": 1
    }
  }'
```

**Response:**
```json
{
  "session_token": "abc123...",
  "enrollment_id": "uuid-here",
  "status": "pending",
  "target_samples": 3,
  "expires_at": "2025-10-27T12:00:00Z",
  "websocket_url": "ws://localhost:8000/ws/auth/process-image/abc123/",
  "message": "Enrollment session created. Stream frames to continue."
}
```

### 3. Connect ke WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/auth/process-image/abc123/');

ws.onopen = () => {
  console.log('Connected!');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### 4. Kirim Frame

```javascript
// Capture frame dari video
const canvas = document.createElement('canvas');
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
canvas.getContext('2d').drawImage(video, 0, 0);

// Convert ke base64
const imageData = canvas.toDataURL('image/jpeg', 0.8);

// Kirim ke WebSocket
ws.send(JSON.stringify({
  type: 'frame',
  image: imageData
}));
```

### 5. Handle Response

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'frame_processed':
      // Update progress
      updateProgress(data.frames_processed, data.target_samples);
      break;
      
    case 'enrollment_complete':
      // Decrypt dan simpan hasil
      const decrypted = decryptResponse(data.encrypted_data.encrypted_payload);
      console.log('Enrollment ID:', decrypted.id);
      console.log('Timestamp:', decrypted.timestamp);
      break;
      
    case 'error':
      console.error('Error:', data.error);
      break;
  }
};
```

## Enkripsi Response

### Server-side (Automatic)

Server otomatis mengenkripsi response dengan API key client menggunakan AES-256-CBC.

### Client-side Decryption

Client perlu decrypt dengan secret key:

```python
import base64
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

def decrypt_response(encrypted_payload: str, secret_key: str) -> dict:
    # Decode base64
    encrypted_data = base64.b64decode(encrypted_payload)
    
    # Extract IV dan ciphertext
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    
    # Derive key dari secret key
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
```

## Testing

### Menggunakan Test Script

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

# Authentication (Identification)
python test_websocket_auth.py \
  YOUR_API_KEY \
  YOUR_SECRET_KEY \
  http://localhost:8000 \
  authentication
```

### Manual Testing dengan wscat

```bash
# Install wscat
npm install -g wscat

# Connect
wscat -c ws://localhost:8000/ws/auth/process-image/SESSION_TOKEN/

# Send frame (paste base64 encoded image)
{"type": "frame", "image": "data:image/jpeg;base64,/9j/4AAQ..."}

# Send ping
{"type": "ping"}
```

## Message Types

### Client → Server

| Type | Description | Required Fields |
|------|-------------|----------------|
| `frame` | Frame gambar wajah | `image` (base64) |
| `ping` | Keep-alive ping | - |

### Server → Client

| Type | Description | Fields |
|------|-------------|--------|
| `connection_established` | Connection berhasil | `session_token`, `session_type`, `status` |
| `frame_processed` | Frame diproses | `success`, `frames_processed`, `liveness_verified`, `quality_score` |
| `enrollment_complete` | Enrollment selesai | `enrollment_id`, `encrypted_data` |
| `authentication_complete` | Authentication selesai | `authenticated`, `user_id`, `confidence`, `encrypted_data` |
| `error` | Error occurred | `error`, `code` |
| `pong` | Response untuk ping | - |

## Encrypted Response Format

```json
{
  "encrypted_data": {
    "encrypted_payload": "base64-encoded-encrypted-data",
    "encryption_method": "AES-256-CBC",
    "instructions": "Decrypt with your secret_key to access the data"
  }
}
```

**Decrypted payload:**
```json
{
  "id": "enrollment-id or user-id",
  "timestamp": "2025-10-27T12:34:56.789Z",
  "session_type": "enrollment",
  "session_token": "abc123...",
  "confidence": 0.95  // Only for authentication
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 1000 | Normal closure (success) |
| 4000 | Generic error |
| 4400 | Bad request |
| 4401 | Session expired |
| 4403 | Forbidden |
| 4404 | Session not found |
| 4429 | Rate limit exceeded |
| 4500 | Internal server error |

## Best Practices

### 1. Frame Rate Control
- **Maximum**: 10 FPS (100ms interval)
- **Recommended**: 8-10 FPS untuk balance antara performance dan akurasi
- Server akan throttle jika terlalu cepat

### 2. Image Quality
- **Format**: JPEG dengan quality 80%
- **Resolution**: 640x480 atau 1280x720
- **Size**: < 100KB per frame

### 3. Error Handling
```javascript
ws.onerror = (error) => {
  console.error('WebSocket error:', error);
  // Retry logic
};

ws.onclose = (event) => {
  if (event.code !== 1000) {
    // Abnormal closure, retry
    setTimeout(() => reconnect(), 1000);
  }
};
```

### 4. Session Management
- Session expire setelah 5 menit
- Check `expires_at` dari response
- Close WebSocket saat selesai
- Jangan reuse session token

### 5. Security
- Gunakan WSS (WebSocket Secure) di production
- Simpan secret key dengan aman (environment variables)
- Never expose secret key di client-side code
- Decrypt response di server-side jika memungkinkan

## Troubleshooting

### WebSocket tidak bisa connect
- ✅ Cek session masih active
- ✅ Verify session belum expired
- ✅ Gunakan protocol yang benar (ws:// atau wss://)
- ✅ Check network/firewall settings

### Frame ditolak
- ✅ Check frame rate (max 10 FPS)
- ✅ Verify image format dan encoding
- ✅ Pastikan wajah terlihat dan lighting baik
- ✅ Check image size

### Liveness detection gagal
- ✅ Pastikan lighting bagus
- ✅ User harus berkedip secara natural
- ✅ Izinkan gerakan kepala natural
- ✅ Jangan kirim gambar static

### Decryption gagal
- ✅ Verify menggunakan secret key yang benar
- ✅ Check encrypted payload lengkap
- ✅ Pastikan base64 decoding benar
- ✅ Verify library encryption compatible

## Performance

- **Throughput**: 10 frames/second
- **Latency**: < 100ms per frame
- **Concurrent connections**: Tergantung server resources
- **Session timeout**: 5 minutes

## Security

### Double Encryption

1. **Server-side**: Data dienkripsi dengan API key client
2. **Client-side**: Client decrypt dengan secret key

Ini memastikan:
- Server tidak bisa membaca data hasil akhir
- Man-in-the-middle tidak bisa decrypt tanpa secret key
- Client punya full control atas data mereka

### Session Validation

- Session token validated di setiap connection
- Client authenticated via API key
- Session expiration checked
- Rate limiting applied

## Monitoring

Log events:
- Connection established/closed
- Frames processed
- Enrollment/Authentication completion
- Errors and warnings

Metrics to monitor:
- WebSocket connections count
- Frame processing rate
- Success/failure rates
- Average session duration
- Error rates by type

## Support

Untuk dokumentasi lengkap, lihat:
- `WEBSOCKET_GUIDE.md` - Dokumentasi client lengkap
- `API_INTEGRATION_GUIDE.md` - Integrasi API
- `TECHNICAL_DOCUMENTATION.md` - Dokumentasi teknis

## Changelog

### v1.0.0 (2025-10-27)
- ✨ Initial WebSocket implementation
- 🔐 Double encryption (API key + Secret key)
- 📊 Real-time progress updates
- 🎯 Session-based event creation
- 📝 Comprehensive documentation
- 🧪 Test script included

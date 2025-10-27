# WebSocket API Guide for Face Authentication & Enrollment

## Overview

This guide explains how to use the WebSocket API for real-time face authentication and enrollment processing. The WebSocket connection provides a more efficient way to process multiple frames compared to the REST API.

## Features

- **Real-time frame processing**: Send frames as they are captured
- **Progress updates**: Receive immediate feedback on each frame
- **Liveness detection**: Automatic blink and motion detection
- **Encrypted responses**: Security through double encryption (API key + Secret key)
- **Automatic session management**: Session automatically completes when requirements are met

## Getting Started

### 1. Authenticate Client

First, authenticate your client to get a JWT token:

```bash
POST /api/auth/client/
Content-Type: application/json

{
  "api_key": "YOUR_API_KEY"
}
```

**Response:**
```json
{
  "jwt_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_at": "2025-10-27T12:00:00Z"
}
```

### 2. Create a Session

Create an enrollment or authentication session using the JWT token:

**Enrollment:**
```bash
POST /api/auth/enrollment/
Authorization: Bearer YOUR_JWT_TOKEN
Content-Type: application/json

{
  "user_id": "user123",
  "metadata": {
    "target_samples": 3,
    "device_info": {}
  }
}
```

**Authentication:**
```bash
POST /api/auth/authentication/
Authorization: Bearer YOUR_JWT_TOKEN
Content-Type: application/json

{
  "user_id": "user123",  // Optional for identification
  "require_liveness": true,
  "metadata": {
    "min_frames_required": 3,
    "required_blinks": 1
  }
}
```

**Response includes:**
```json
{
  "session_token": "abc123...",
  "websocket_url": "wss://your-domain.com/ws/auth/process-image/abc123/",
  "expires_at": "2025-10-27T12:00:00Z",
  "status": "active"
}
```

### 3. Connect to WebSocket

Connect to the WebSocket URL returned from the session creation:

```javascript
const ws = new WebSocket('wss://your-domain.com/ws/auth/process-image/abc123/');

ws.onopen = () => {
  console.log('Connected to WebSocket');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  handleMessage(data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
  console.log('WebSocket closed:', event.code);
};
```

### 4. Send Frame Data

Capture frames from the camera and send them as base64-encoded images:

```javascript
async function sendFrame(videoElement) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0);
  
  const imageData = canvas.toDataURL('image/jpeg', 0.8);
  
  ws.send(JSON.stringify({
    type: 'frame',
    image: imageData
  }));
}

// Send frames at regular intervals (e.g., 10 fps)
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    sendFrame(videoElement);
  }
}, 100); // 10 fps
```

## Message Types

### Client → Server

#### 1. Frame Message
```json
{
  "type": "frame",
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

#### 2. Ping Message
```json
{
  "type": "ping"
}
```

### Server → Client

#### 1. Connection Established
```json
{
  "type": "connection_established",
  "session_token": "abc123...",
  "session_type": "enrollment",
  "status": "active",
  "message": "Enrollment session ready"
}
```

#### 2. Frame Processed (Progress Update)
```json
{
  "type": "frame_processed",
  "success": true,
  "frames_processed": 2,
  "target_samples": 3,
  "liveness_verified": false,
  "liveness_score": 0.5,
  "quality_score": 0.85,
  "message": "Continue capturing frames"
}
```

#### 3. Enrollment Complete
```json
{
  "type": "enrollment_complete",
  "success": true,
  "enrollment_id": "uuid-here",
  "frames_processed": 3,
  "liveness_verified": true,
  "encrypted_data": {
    "encrypted_payload": "base64-encoded-encrypted-data",
    "encryption_method": "AES-256-CBC",
    "instructions": "Decrypt with your secret_key to access the data"
  },
  "message": "Enrollment completed successfully"
}
```

#### 4. Authentication Complete
```json
{
  "type": "authentication_complete",
  "success": true,
  "authenticated": true,
  "user_id": "user123",
  "confidence": 0.95,
  "frames_processed": 3,
  "encrypted_data": {
    "encrypted_payload": "base64-encoded-encrypted-data",
    "encryption_method": "AES-256-CBC",
    "instructions": "Decrypt with your secret_key to access the data"
  },
  "message": "Authentication successful"
}
```

#### 5. Error
```json
{
  "type": "error",
  "error": "No face detected",
  "code": 4000
}
```

#### 6. Pong
```json
{
  "type": "pong"
}
```

## Decrypting Responses

When enrollment or authentication is successful, the server returns encrypted data. This data is encrypted twice for maximum security:

1. **Server-side encryption**: Data is first encrypted with your API key
2. **Client-side decryption**: You decrypt it with your secret key

### Decryption Example (Python)

```python
import json
import base64
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

def decrypt_response(encrypted_payload: str, secret_key: str) -> dict:
    """
    Decrypt the encrypted response from the server.
    
    Args:
        encrypted_payload: Base64-encoded encrypted data
        secret_key: Your client secret key
    
    Returns:
        Decrypted payload as dictionary
    """
    # Decode base64
    encrypted_data = base64.b64decode(encrypted_payload)
    
    # Extract IV (first 16 bytes)
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    
    # Derive key from secret key
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
secret_key = "your_secret_key_here"

decrypted = decrypt_response(encrypted_payload, secret_key)
print(decrypted)
# Output: {
#   "id": "enrollment-id or user-id",
#   "timestamp": "2025-10-27T12:34:56.789Z",
#   "session_type": "enrollment",
#   "session_token": "abc123...",
#   "confidence": 0.95  // Only for authentication
# }
```

### Decryption Example (JavaScript/Node.js)

```javascript
const crypto = require('crypto');

function decryptResponse(encryptedPayload, secretKey) {
  // Decode base64
  const encryptedData = Buffer.from(encryptedPayload, 'base64');
  
  // Extract IV (first 16 bytes)
  const iv = encryptedData.slice(0, 16);
  const ciphertext = encryptedData.slice(16);
  
  // Derive key from secret key
  const key = crypto.createHash('sha256').update(secretKey).digest();
  
  // Decrypt
  const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
  let decrypted = decipher.update(ciphertext);
  decrypted = Buffer.concat([decrypted, decipher.final()]);
  
  // Parse JSON
  return JSON.parse(decrypted.toString('utf-8'));
}

// Usage
const encryptedPayload = response.encrypted_data.encrypted_payload;
const secretKey = 'your_secret_key_here';

const decrypted = decryptResponse(encryptedPayload, secretKey);
console.log(decrypted);
```

## Error Codes

| Code | Description |
|------|-------------|
| 1000 | Normal closure (success) |
| 4000 | Generic error |
| 4400 | Bad request (invalid session token or session status) |
| 4401 | Session expired |
| 4403 | Forbidden (authentication failed) |
| 4404 | Session or client not found |
| 4429 | Too many frames (rate limit exceeded) |
| 4500 | Internal server error |

## Best Practices

### 1. Frame Rate
- Send frames at 10 FPS maximum (100ms interval)
- Server will throttle if frames are sent too quickly
- Adjust based on network conditions

### 2. Image Quality
- Use JPEG format with 80% quality for optimal balance
- Recommended resolution: 640x480 or 1280x720
- Keep file size under 100KB per frame

### 3. Error Handling
```javascript
let reconnectAttempts = 0;
const maxReconnectAttempts = 3;

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
  if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
    // Attempt to reconnect for non-normal closures
    reconnectAttempts++;
    setTimeout(() => {
      connectWebSocket();
    }, 1000 * reconnectAttempts);
  }
};
```

### 4. Session Management
- Sessions expire after 5 minutes of inactivity
- Check `expires_at` timestamp from session creation
- Close WebSocket when session is complete
- Don't reuse session tokens

### 5. Security
- Always use WSS (WebSocket Secure) in production
- Store secret keys securely (environment variables, key vaults)
- Never expose secret keys in client-side code
- Decrypt responses server-side when possible

## Complete Example

```javascript
class FaceAuthClient {
  constructor(apiKey, secretKey, baseUrl) {
    this.apiKey = apiKey;
    this.secretKey = secretKey;
    this.baseUrl = baseUrl;
    this.ws = null;
  }

  async createEnrollmentSession(userId) {
    const response = await fetch(`${this.baseUrl}/api/auth/enrollment/`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        user_id: userId,
        metadata: { target_samples: 3 }
      })
    });
    
    return response.json();
  }

  connectWebSocket(websocketUrl) {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(websocketUrl);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        resolve();
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };
      
      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      };
    });
  }

  handleMessage(data) {
    switch(data.type) {
      case 'connection_established':
        console.log('Connection established:', data);
        break;
      
      case 'frame_processed':
        console.log('Frame processed:', data);
        this.onProgress?.(data);
        break;
      
      case 'enrollment_complete':
        console.log('Enrollment complete!');
        const decrypted = this.decryptResponse(
          data.encrypted_data.encrypted_payload
        );
        this.onComplete?.(decrypted);
        break;
      
      case 'error':
        console.error('Error:', data.error);
        this.onError?.(data.error);
        break;
    }
  }

  sendFrame(imageData) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'frame',
        image: imageData
      }));
    }
  }

  decryptResponse(encryptedPayload) {
    // Use crypto library to decrypt
    // See decryption examples above
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Usage
const client = new FaceAuthClient(apiKey, secretKey, baseUrl);

client.onProgress = (data) => {
  console.log(`Progress: ${data.frames_processed}/${data.target_samples}`);
};

client.onComplete = (decrypted) => {
  console.log('Enrollment ID:', decrypted.id);
  console.log('Timestamp:', decrypted.timestamp);
};

client.onError = (error) => {
  console.error('Error:', error);
};

// Create session and connect
const session = await client.createEnrollmentSession('user123');
await client.connectWebSocket(session.websocket_url);

// Start sending frames
startCamera((frame) => {
  client.sendFrame(frame);
});
```

## Troubleshooting

### WebSocket won't connect
- Check that the session is still active
- Verify the session hasn't expired
- Ensure you're using the correct protocol (ws:// or wss://)
- Check network/firewall settings

### Frames are rejected
- Check frame rate (max 10 FPS)
- Verify image format and encoding
- Ensure face is visible and well-lit
- Check image size (should be reasonable, not too large)

### Liveness detection fails
- Ensure good lighting conditions
- Ask user to blink naturally
- Allow for natural head movement
- Don't send static images

### Decryption fails
- Verify you're using the correct secret key
- Check that the encrypted payload is complete
- Ensure proper base64 decoding
- Verify encryption library compatibility

## Support

For additional support or questions, please contact our support team or refer to the main API documentation.

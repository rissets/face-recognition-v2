# WebSocket Integration Guide

Complete guide for integrating face recognition enrollment and authentication using WebSocket connections.

## Table of Contents

- [Overview](#overview)
- [Authentication Flow](#authentication-flow)
- [Enrollment Flow](#enrollment-flow)
- [Authentication Flow](#authentication-flow-1)
- [Message Types](#message-types)
- [Code Examples](#code-examples)
- [Error Handling](#error-handling)
- [Security Considerations](#security-considerations)

---

## Overview

The Face Recognition API provides real-time face processing through WebSocket connections. This enables:

- **Real-time feedback** during enrollment and authentication
- **Liveness detection** with blink and motion tracking
- **Obstacle detection** (glasses, masks, hands, hats)
- **Quality assessment** for each frame
- **Session-based processing** with persistent state

### Architecture

```
Client Application
    ↓ (1) REST API: Create Session
API Server
    ↓ (2) Return session_token + websocket_url
Client
    ↓ (3) WebSocket Connect
WebSocket Handler
    ↓ (4) Stream Frames
Face Recognition Engine
    ↓ (5) Process & Send Feedback
Client (Real-time Updates)
```

---

## Authentication Flow

### Step 1: Authenticate Client

Before creating any session, authenticate with your API credentials.

**Endpoint:** `POST /api/core/auth/client/`

**Request:**
```json
{
  "api_key": "frapi_xxxxx",
  "api_secret": "your_secret_key"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "JWT",
  "expires_in": 3600
}
```

**Usage:**
```http
Authorization: JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Enrollment Flow

### Step 1: Create Enrollment Session

**Endpoint:** `POST /api/auth/enrollment/`

**Headers:**
```http
Authorization: JWT <access_token>
Content-Type: application/json
```

**Request:**
```json
{
  "user_id": "user123",
  "session_type": "webcam",
  "metadata": {
    "target_samples": 3,
    "required_blinks": 1,
    "device_info": {
      "platform": "web",
      "browser": "Chrome"
    }
  }
}
```

**Response:**
```json
{
  "session_token": "sess_xxxxx",
  "enrollment_id": "uuid-xxxx-xxxx",
  "status": "pending",
  "target_samples": 3,
  "expires_at": "2025-10-27T15:00:00Z",
  "websocket_url": "wss://api.example.com/ws/auth/process-image/sess_xxxxx/",
  "message": "Enrollment session created. Stream frames to continue."
}
```

### Step 2: Connect to WebSocket

**URL:** Use `websocket_url` from response

**Protocol:** WSS (WebSocket Secure) for HTTPS, WS for HTTP

**Connection:**
```javascript
const ws = new WebSocket(websocket_url);

ws.onopen = () => {
  console.log('WebSocket connected');
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

### Step 3: Send Frame Messages

Send frames at ~10 FPS for optimal processing.

**Message Format:**
```json
{
  "type": "frame",
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Python Example:**
```python
import cv2
import base64
import json

# Capture frame
ret, frame = cap.read()

# Encode to JPEG
_, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

# Encode to base64
image_base64 = base64.b64encode(buffer).decode('utf-8')
image_data = f"data:image/jpeg;base64,{image_base64}"

# Send via WebSocket
message = {
    "type": "frame",
    "image": image_data
}
await ws.send(json.dumps(message))
```

### Step 4: Process Server Messages

#### Connection Established
```json
{
  "type": "connection_established",
  "session_type": "enrollment",
  "session_token": "sess_xxxxx",
  "message": "WebSocket connection established"
}
```

#### Frame Rejected (Obstacles Detected)
```json
{
  "type": "frame_rejected",
  "reason": "Obstacles detected blocking face capture",
  "obstacles": ["glasses", "hand_covering"],
  "obstacle_confidence": {
    "glasses": 0.85,
    "hand_covering": 0.72
  },
  "message": "Remove obstacles and try again"
}
```

#### Frame Processed Successfully
```json
{
  "type": "frame_processed",
  "success": true,
  "frames_processed": 2,
  "target_samples": 3,
  "quality_score": 0.87,
  "quality_threshold": 0.65,
  "liveness_score": 0.92,
  "liveness_verified": false,
  "blinks_detected": 1,
  "blinks_required": 1,
  "blinks_ok": true,
  "motion_events": 0,
  "motion_required": 1,
  "motion_ok": false,
  "no_obstacles": true,
  "obstacles": [],
  "message": "Motion required (0/1) | Blink required (1/1) ✓",
  "visual_data": {
    "face_bbox": [120, 80, 520, 480],
    "liveness_indicators": {
      "ear": 0.285,
      "blinks": 1,
      "motion_events": 0,
      "frame_counter": 15
    }
  }
}
```

#### Enrollment Complete
```json
{
  "type": "enrollment_complete",
  "success": true,
  "enrollment_id": "uuid-xxxx-xxxx",
  "frames_processed": 3,
  "liveness_verified": true,
  "blinks_detected": 2,
  "motion_verified": true,
  "quality_score": 0.89,
  "encrypted_data": {
    "encrypted_payload": "base64_encrypted_data",
    "algorithm": "AES-256-CBC"
  },
  "message": "Enrollment completed successfully"
}
```

### Step 5: Decrypt Response (Optional)

The encrypted data contains sensitive enrollment information.

**Decryption (Python):**
```python
import hashlib
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

def decrypt_response(encrypted_payload: str, api_key: str) -> dict:
    # Decode base64
    encrypted_data = base64.b64decode(encrypted_payload)
    
    # Extract IV (first 16 bytes)
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    
    # Derive key from API key
    key = hashlib.sha256(api_key.encode()).digest()
    
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
decrypted = decrypt_response(encrypted_payload, api_key)
print(f"Enrollment ID: {decrypted['id']}")
print(f"Timestamp: {decrypted['timestamp']}")
```

**Decryption (JavaScript):**
```javascript
async function decryptResponse(encryptedPayload, apiKey) {
  // Decode base64
  const encryptedBytes = CryptoJS.enc.Base64.parse(encryptedPayload);
  
  // Extract IV (first 16 bytes)
  const iv = CryptoJS.lib.WordArray.create(encryptedBytes.words.slice(0, 4));
  
  // Extract ciphertext
  const ciphertext = CryptoJS.lib.WordArray.create(encryptedBytes.words.slice(4));
  
  // Derive key from API key
  const key = CryptoJS.SHA256(apiKey);
  
  // Decrypt
  const decrypted = CryptoJS.AES.decrypt(
    { ciphertext: ciphertext },
    key,
    { 
      iv: iv,
      mode: CryptoJS.mode.CBC,
      padding: CryptoJS.pad.Pkcs7
    }
  );
  
  // Convert to string and parse JSON
  const decryptedText = decrypted.toString(CryptoJS.enc.Utf8);
  return JSON.parse(decryptedText);
}
```

---

## Authentication Flow

### Step 1: Create Authentication Session

**Endpoint:** `POST /api/auth/authentication/`

**Headers:**
```http
Authorization: JWT <access_token>
Content-Type: application/json
```

**Request (Verification - with user_id):**
```json
{
  "user_id": "user123",
  "session_type": "webcam",
  "require_liveness": true,
  "metadata": {
    "min_frames_required": 3,
    "required_blinks": 1,
    "device_info": {
      "platform": "web"
    }
  }
}
```

**Request (Identification - without user_id):**
```json
{
  "session_type": "webcam",
  "require_liveness": true,
  "metadata": {
    "min_frames_required": 3,
    "required_blinks": 1
  }
}
```

**Response:**
```json
{
  "session_token": "sess_yyyyy",
  "status": "active",
  "expires_at": "2025-10-27T15:00:00Z",
  "session_type": "verification",
  "websocket_url": "wss://api.example.com/ws/auth/process-image/sess_yyyyy/",
  "message": "Authentication session created. Stream frames to continue."
}
```

### Step 2-3: Connect and Send Frames

Same as enrollment flow.

### Step 4: Process Authentication Messages

#### Frame Processed
```json
{
  "type": "frame_processed",
  "success": true,
  "frames_processed": 2,
  "min_frames_required": 3,
  "quality_score": 0.88,
  "liveness_score": 0.91,
  "liveness_verified": false,
  "blinks_detected": 0,
  "blinks_required": 1,
  "blinks_ok": false,
  "motion_events": 1,
  "motion_required": 1,
  "motion_ok": true,
  "no_obstacles": true,
  "message": "Need: Blink (0/1) OR Motion (1/1) ✓",
  "visual_data": { ... }
}
```

**Note:** Authentication uses **OR logic** - only need blink **OR** motion (not both).

#### Authentication Complete (Success)
```json
{
  "type": "authentication_complete",
  "success": true,
  "authenticated": true,
  "user_id": "user123",
  "confidence": 0.94,
  "frames_processed": 3,
  "blinks_detected": 1,
  "motion_verified": true,
  "encrypted_data": {
    "encrypted_payload": "base64_encrypted_data",
    "algorithm": "AES-256-CBC"
  },
  "message": "Authentication successful"
}
```

#### Authentication Complete (Failed)
```json
{
  "type": "authentication_complete",
  "success": false,
  "authenticated": false,
  "frames_processed": 5,
  "message": "Face not recognized"
}
```

---

## Message Types

### Client → Server

| Type | Description | Required Fields |
|------|-------------|----------------|
| `frame` | Send face image frame | `type`, `image` |
| `ping` | Keep connection alive | `type` |

### Server → Client

| Type | Description | Status |
|------|-------------|--------|
| `connection_established` | WebSocket connected | Initial |
| `frame_rejected` | Frame rejected due to obstacles | Error |
| `frame_processed` | Frame processed successfully | Progress |
| `enrollment_complete` | Enrollment finished | Success/Fail |
| `authentication_complete` | Authentication finished | Success/Fail |
| `error` | Error occurred | Error |
| `pong` | Response to ping | Keep-alive |

---

## Liveness Requirements

### Enrollment (Strict - AND Logic)
```
Success = (Blinks ≥ 1) AND (Motion ≥ 1) AND (Obstacles = 0) AND (Quality OK) AND (Frames ≥ 3)
```

**Requirements:**
- ✅ At least 1 blink detected
- ✅ At least 1 motion event (head movement)
- ✅ No obstacles (glasses, mask, hand, hat)
- ✅ Quality score ≥ threshold
- ✅ Minimum frames processed

### Authentication (Flexible - OR Logic)
```
Success = ((Blinks ≥ 1) OR (Motion ≥ 1)) AND (Obstacles = 0) AND (Quality OK) AND (Frames ≥ 3)
```

**Requirements:**
- ✅ At least 1 blink **OR** at least 1 motion event
- ✅ No obstacles (glasses, mask, hand, hat)
- ✅ Quality score ≥ threshold
- ✅ Minimum frames processed

---

## Obstacle Detection

The system detects and rejects frames with the following obstacles:

| Obstacle | Threshold | Description |
|----------|-----------|-------------|
| **Glasses** | > 0.50 | Eyeglasses or sunglasses |
| **Mask** | > 0.30 | Face mask covering mouth/nose |
| **Hat** | > 0.30 | Hat or cap covering forehead |
| **Hand** | > 0.30 | Hand covering face |

**Frame Rejection:**
- ANY obstacle detected → Frame is rejected
- Must remove obstacle to proceed
- Clear visual feedback provided

---

## Code Examples

### Complete Python Client

```python
import asyncio
import cv2
import base64
import json
import websockets
import requests

class FaceAuthClient:
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip('/')
        self.jwt_token = None
        self.ws = None
    
    def authenticate_client(self):
        """Get JWT token"""
        url = f"{self.base_url}/api/core/auth/client/"
        response = requests.post(url, json={
            "api_key": self.api_key,
            "api_secret": self.secret_key
        })
        response.raise_for_status()
        self.jwt_token = response.json()['access_token']
        return self.jwt_token
    
    def create_enrollment_session(self, user_id: str):
        """Create enrollment session"""
        url = f"{self.base_url}/api/auth/enrollment/"
        headers = {
            "Authorization": f"JWT {self.jwt_token}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json={
            "user_id": user_id,
            "session_type": "webcam",
            "metadata": {"target_samples": 3}
        })
        response.raise_for_status()
        return response.json()
    
    async def connect_websocket(self, websocket_url: str):
        """Connect to WebSocket"""
        # Convert https to wss if needed
        if websocket_url.startswith('https://'):
            websocket_url = websocket_url.replace('https://', 'wss://', 1)
        elif websocket_url.startswith('http://'):
            websocket_url = websocket_url.replace('http://', 'ws://', 1)
        
        self.ws = await websockets.connect(websocket_url)
    
    async def send_frame(self, frame):
        """Send frame to WebSocket"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        await self.ws.send(json.dumps({
            "type": "frame",
            "image": f"data:image/jpeg;base64,{image_base64}"
        }))
    
    async def receive_messages(self):
        """Receive messages from WebSocket"""
        async for message in self.ws:
            data = json.loads(message)
            print(f"Received: {data['type']}")
            
            if data['type'] in ['enrollment_complete', 'authentication_complete']:
                break
    
    async def run_enrollment(self, user_id: str):
        """Run complete enrollment"""
        # Step 1: Authenticate
        self.authenticate_client()
        
        # Step 2: Create session
        session = self.create_enrollment_session(user_id)
        print(f"Session: {session['session_token']}")
        
        # Step 3: Connect WebSocket
        await self.connect_websocket(session['websocket_url'])
        
        # Step 4: Start camera and send frames
        cap = cv2.VideoCapture(0)
        
        receiver = asyncio.create_task(self.receive_messages())
        
        try:
            while not receiver.done():
                ret, frame = cap.read()
                if ret:
                    await self.send_frame(frame)
                    await asyncio.sleep(0.1)  # 10 FPS
        finally:
            cap.release()
            await self.ws.close()

# Usage
async def main():
    client = FaceAuthClient(
        api_key="frapi_xxxx",
        secret_key="your_secret",
        base_url="https://api.example.com"
    )
    await client.run_enrollment("user123")

asyncio.run(main())
```

### Complete JavaScript Client

```javascript
class FaceAuthClient {
  constructor(apiKey, secretKey, baseUrl) {
    this.apiKey = apiKey;
    this.secretKey = secretKey;
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.jwtToken = null;
    this.ws = null;
  }
  
  async authenticateClient() {
    const response = await fetch(`${this.baseUrl}/api/core/auth/client/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: this.apiKey,
        api_secret: this.secretKey
      })
    });
    
    if (!response.ok) throw new Error('Authentication failed');
    
    const data = await response.json();
    this.jwtToken = data.access_token;
    return this.jwtToken;
  }
  
  async createEnrollmentSession(userId) {
    const response = await fetch(`${this.baseUrl}/api/auth/enrollment/`, {
      method: 'POST',
      headers: {
        'Authorization': `JWT ${this.jwtToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        user_id: userId,
        session_type: 'webcam',
        metadata: { target_samples: 3 }
      })
    });
    
    if (!response.ok) throw new Error('Session creation failed');
    return response.json();
  }
  
  connectWebSocket(websocketUrl) {
    // Convert http/https to ws/wss
    let wsUrl = websocketUrl;
    if (wsUrl.startsWith('https://')) {
      wsUrl = wsUrl.replace('https://', 'wss://');
    } else if (wsUrl.startsWith('http://')) {
      wsUrl = wsUrl.replace('http://', 'ws://');
    }
    
    this.ws = new WebSocket(wsUrl);
    
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (error) => reject(error);
    });
  }
  
  sendFrame(imageData) {
    this.ws.send(JSON.stringify({
      type: 'frame',
      image: imageData
    }));
  }
  
  async runEnrollment(userId) {
    // Authenticate
    await this.authenticateClient();
    
    // Create session
    const session = await this.createEnrollmentSession(userId);
    console.log('Session:', session.session_token);
    
    // Connect WebSocket
    await this.connectWebSocket(session.websocket_url);
    
    // Handle messages
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Received:', data.type);
      
      if (data.type === 'enrollment_complete') {
        console.log('Enrollment complete!');
      }
    };
    
    // Start camera and send frames
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    
    setInterval(() => {
      if (this.ws.readyState === WebSocket.OPEN) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        this.sendFrame(imageData);
      }
    }, 100); // 10 FPS
  }
}

// Usage
const client = new FaceAuthClient(
  'frapi_xxxx',
  'your_secret',
  'https://api.example.com'
);

client.runEnrollment('user123');
```

---

## Error Handling

### Common Error Codes

| Code | Message | Solution |
|------|---------|----------|
| 400 | Invalid image data | Check image encoding |
| 401 | Unauthorized | Refresh JWT token |
| 404 | Session not found | Create new session |
| 429 | Frame rate too high | Reduce FPS to ~10 |
| 4429 | Maximum frames exceeded | Session limit reached |

### WebSocket Close Codes

| Code | Reason | Action |
|------|--------|--------|
| 1000 | Normal closure | Success - session complete |
| 1001 | Going away | Reconnect if needed |
| 4429 | Too many frames | Create new session |

### Error Message Format

```json
{
  "type": "error",
  "error": "Error description",
  "code": 400,
  "details": {
    "field": "Additional error details"
  }
}
```

---

## Security Considerations

### 1. API Key Protection

```python
# ❌ DON'T: Hardcode keys
api_key = "frapi_xxxx"

# ✅ DO: Use environment variables
import os
api_key = os.getenv('FACE_API_KEY')
```

### 2. JWT Token Management

```javascript
// Store token securely
sessionStorage.setItem('jwt_token', token);

// Refresh before expiry
if (Date.now() > tokenExpiry - 60000) {
  await refreshToken();
}
```

### 3. WebSocket Security

- Always use **WSS** (WebSocket Secure) in production
- Validate server certificate
- Handle disconnections gracefully
- Implement reconnection logic with exponential backoff

### 4. Frame Rate Limiting

```python
# Client-side throttling
await asyncio.sleep(0.1)  # 10 FPS max

# Implement exponential backoff on 429 errors
if response_code == 429:
    await asyncio.sleep(backoff_time)
    backoff_time *= 2
```

### 5. Encrypted Data

```python
# Never log decrypted sensitive data
decrypted = decrypt_response(payload, api_key)
# ❌ DON'T: logger.info(f"Data: {decrypted}")
# ✅ DO: logger.info("Data decrypted successfully")
```

---

## Best Practices

### 1. Frame Quality

- **Resolution**: 640x480 minimum
- **Format**: JPEG with 80% quality
- **Frame Rate**: ~10 FPS optimal
- **Lighting**: Well-lit environment
- **Distance**: Face fills 40-60% of frame

### 2. User Experience

```javascript
// Show clear instructions
function showInstruction(message) {
  switch(message) {
    case 'blink_required':
      display('Please blink naturally');
      break;
    case 'motion_required':
      display('Move your head slightly');
      break;
    case 'obstacle_detected':
      display('Remove glasses/mask');
      break;
  }
}
```

### 3. Progress Feedback

```javascript
// Display progress bar
const progress = (framesProcessed / targetSamples) * 100;
updateProgressBar(progress);

// Show liveness status
displayStatus({
  blinks: `${blinksDetected}/${blinksRequired} ${blinksOk ? '✅' : '❌'}`,
  motion: `${motionEvents}/${motionRequired} ${motionOk ? '✅' : '❌'}`,
  obstacles: noObstacles ? '✅ Clear' : '⛔ Remove obstacles'
});
```

### 4. Error Recovery

```python
# Implement retry logic
max_retries = 3
for attempt in range(max_retries):
    try:
        await send_frame(frame)
        break
    except Exception as e:
        if attempt == max_retries - 1:
            raise
        await asyncio.sleep(2 ** attempt)
```

### 5. Session Management

```javascript
// Track session state
class SessionManager {
  constructor() {
    this.sessionToken = null;
    this.expiresAt = null;
  }
  
  isExpired() {
    return Date.now() > this.expiresAt;
  }
  
  async renewIfNeeded() {
    if (this.isExpired()) {
      await this.createNewSession();
    }
  }
}
```

---

## Testing

### Local Testing

```bash
# Start local server
python manage.py runserver

# Test with local URL
python test_websocket_auth.py \
  frapi_xxxx \
  your_secret \
  http://127.0.0.1:8000 \
  enrollment \
  user123
```

### Production Testing

```bash
# Test with production URL
python test_websocket_auth.py \
  frapi_xxxx \
  your_secret \
  https://api.example.com \
  enrollment \
  user123
```

### Testing Checklist

- [ ] Client authentication works
- [ ] Session creation successful
- [ ] WebSocket connection established
- [ ] Frame sending/receiving works
- [ ] Liveness detection triggered
- [ ] Obstacle detection works
- [ ] Enrollment completes
- [ ] Authentication works
- [ ] Error handling functions
- [ ] Reconnection logic works

---

## Support

For issues or questions:

- **Documentation**: `/docs/` endpoint
- **API Reference**: `/api/schema/` endpoint
- **Health Check**: `/health/` endpoint

---

## Version History

- **v2.0.0** (2025-10-27)
  - WebSocket-based real-time processing
  - Enhanced liveness detection
  - Obstacle detection
  - OR logic for authentication
  - Encrypted response payload

- **v1.0.0** (2024-01-01)
  - Initial REST API release

---

## License

Copyright © 2025. All rights reserved.

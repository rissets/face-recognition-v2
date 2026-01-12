# Quick Start Guide: Authentication with 3-Second Timeout

## Ringkasan Perubahan

✅ **Authentication sekarang menggunakan Optimal Passive Liveness Detection**
- Timeout: **3 detik** (auto-close WebSocket)
- Required blinks: **1+** (lebih fleksibel dari enrollment)
- Anti-spoofing: YOLO device detection + screen detection + blink analysis

## Cara Kerja

### 1. WebSocket Connection
```javascript
// Client membuat WebSocket connection
const ws = new WebSocket(`ws://server/ws/auth/${sessionToken}/`);

ws.onopen = () => {
  console.log('Connected - 3 second timeout active');
};
```

### 2. Connection Confirmation
Server akan mengirim konfirmasi dengan timeout info:
```json
{
  "type": "connection_established",
  "session_type": "verification",
  "timeout": 3.0,
  "message": "Verification session ready"
}
```

### 3. Frame Processing
Kirim frame video ke server:
```javascript
// Capture from video element
const canvas = document.createElement('canvas');
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
const ctx = canvas.getContext('2d');
ctx.drawImage(video, 0, 0);

// Send as base64
const imageData = canvas.toDataURL('image/jpeg', 0.8);
ws.send(JSON.stringify({
  type: 'frame',
  image: imageData
}));
```

### 4. Progress Updates
Server mengirim update real-time dengan countdown:
```json
{
  "type": "frame_processed",
  "success": true,
  "elapsed_time": 1.5,
  "timeout": 3.0,
  "liveness_score": 0.75,
  "blinks_detected": 1,
  "message": "⏱️ 1.5s remaining | Blink (1/1) ✅"
}
```

### 5. Auto-Close Setelah Timeout
Setelah 3 detik, WebSocket otomatis close dengan final result:
```json
{
  "type": "authentication_timeout",
  "success": true,
  "timeout": true,
  "elapsed_time": 3.0,
  "is_live": true,
  "liveness_score": 0.85,
  "blink_count": 1,
  "reason": "TIMEOUT (3.0s): REAL - 1 blink detected"
}
```

## Response Types

### Success - Authentication Passed
```json
{
  "type": "authentication_complete",
  "success": true,
  "authenticated": true,
  "user_id": "user123",
  "confidence": 0.92,
  "encrypted_data": { ... }
}
```

### Timeout - Liveness Verified
```json
{
  "type": "authentication_timeout",
  "success": true,
  "is_live": true,
  "liveness_score": 0.85,
  "blink_count": 1
}
```

### Rejection - Device Detected
```json
{
  "type": "frame_rejected",
  "success": false,
  "device_detected": "cell phone",
  "reason": "📱 Spoofing attempt detected"
}
```

### Rejection - No Blinks
```json
{
  "type": "authentication_timeout",
  "success": false,
  "is_live": false,
  "liveness_score": 0.0,
  "blink_count": 0,
  "reason": "TIMEOUT (3.0s): SPOOF - No blinks detected"
}
```

## Frontend Implementation Example

```javascript
class AuthenticationSession {
  constructor(sessionToken) {
    this.sessionToken = sessionToken;
    this.ws = null;
    this.timeout = null;
    this.startTime = null;
  }
  
  connect() {
    this.ws = new WebSocket(`ws://server/ws/auth/${this.sessionToken}/`);
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch(data.type) {
        case 'connection_established':
          this.startTime = Date.now();
          this.timeout = data.timeout * 1000; // Convert to ms
          this.startCapture();
          break;
          
        case 'frame_processed':
          this.updateUI(data);
          break;
          
        case 'authentication_timeout':
          this.handleTimeout(data);
          break;
          
        case 'authentication_complete':
          this.handleSuccess(data);
          break;
          
        case 'frame_rejected':
          this.handleRejection(data);
          break;
      }
    };
    
    this.ws.onclose = () => {
      console.log('Session closed');
      this.stopCapture();
    };
  }
  
  updateUI(data) {
    const elapsed = data.elapsed_time || 0;
    const remaining = data.timeout - elapsed;
    
    // Update countdown
    document.getElementById('countdown').textContent = 
      `${remaining.toFixed(1)}s remaining`;
    
    // Update blink count
    document.getElementById('blinks').textContent = 
      `Blinks: ${data.blinks_detected}`;
    
    // Update liveness score
    document.getElementById('liveness').textContent = 
      `Liveness: ${(data.liveness_score * 100).toFixed(0)}%`;
  }
  
  handleTimeout(data) {
    if (data.success) {
      // Liveness verified - show success
      this.showSuccess('Authentication completed!');
    } else {
      // Failed liveness check
      this.showError(data.reason || 'Authentication failed');
    }
  }
  
  startCapture() {
    // Start sending frames every 100ms
    this.captureInterval = setInterval(() => {
      this.captureAndSend();
    }, 100);
  }
  
  stopCapture() {
    if (this.captureInterval) {
      clearInterval(this.captureInterval);
    }
  }
  
  captureAndSend() {
    // Capture frame from video element
    const canvas = document.createElement('canvas');
    canvas.width = this.video.videoWidth;
    canvas.height = this.video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(this.video, 0, 0);
    
    // Send to server
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    this.ws.send(JSON.stringify({
      type: 'frame',
      image: imageData
    }));
  }
}

// Usage
const session = new AuthenticationSession('session_token_here');
session.connect();
```

## UI Recommendations

1. **Countdown Timer**: Show remaining time prominently
2. **Blink Counter**: Display real-time blink detection
3. **Instructions**: "Blink at least once within 3 seconds"
4. **Visual Feedback**: Green border when blink detected
5. **Progress Bar**: Visual countdown representation

## Testing

```bash
# Test authentication mode locally
python test_optimal_liveness_auth.py auth

# Or use the main script
cd face_recognition_app/core
python passive_liveness_optimal.py --auth
```

## Configuration

Untuk mengubah timeout, edit di `consumers.py`:
```python
self._auth_timeout = 3.0  # Change to desired seconds
```

## Performance Tips

1. Send frames at 10 FPS (100ms interval) - optimal balance
2. Use JPEG quality 0.8 untuk reduce bandwidth
3. Handle WebSocket close gracefully
4. Implement reconnection logic untuk network issues
5. Show loading state during initial connection

## Security Notes

- Server melakukan anti-spoofing checks (YOLO, screen detection)
- Blink pattern dianalisa untuk detect fake videos
- Head movement tracking untuk verify 3D liveness
- Multiple detection methods untuk reduce false positives
- Auto-timeout prevents indefinite sessions

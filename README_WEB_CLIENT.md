# Face Recognition WebSocket Client

## 🎯 Quick Summary

Anda punya **2 cara** untuk menggunakan Face Recognition API:

| Method | WebSocket Status | Recommended |
|--------|------------------|-------------|
| **Python CLI** | ✅ Working | ⭐ **YES** |
| **Web Browser** | ❌ CORS blocked | ❌ No |

## ⚡ Quick Start (30 Seconds)

```bash
# Single command - sudah configured dengan production credentials
python test_websocket_auth.py --profile production enrollment 653384
```

✅ **Works perfectly!** WebSocket connects, face detection running, liveness works.

## 🤔 Why Use CLI Instead of Web?

### Current Status:

**Python CLI:**
```
✅ HTTPS API: Working
✅ WebSocket: Working
✅ Face Detection: Working  
✅ Liveness Detection: Working
✅ Real-time Feedback: Working
```

**Web Browser:**
```
✅ HTTPS API: Working
❌ WebSocket: BLOCKED by CORS policy
❌ Cannot send frames
❌ Cannot receive results
```

### The Issue:

Server `https://face.ahu.go.id` tidak mengizinkan WebSocket connections dari browser origins (CORS policy). Tapi Python client bisa bypass ini karena tidak dibatasi oleh browser security.

## 📦 Files Created

1. **test_websocket_auth.py** ⭐ - CLI client with config support
2. **web_config.json** - Configuration file dengan credentials
3. **web_face_auth.html** - Web interface (WebSocket may fail)
4. **web_server.py** - HTTP server untuk web interface
5. **QUICK_START.md** - Detailed quick start guide
6. **WEBSOCKET_TROUBLESHOOTING.md** - Why browser fails, CLI works

## 🚀 Usage Examples

### Enrollment

```bash
# With config profile
python test_websocket_auth.py --profile production enrollment 653384

# With old photo for similarity comparison  
python test_websocket_auth.py --profile production enrollment 653384 /path/to/old_photo.jpg
```

### Authentication (Verification)

```bash
python test_websocket_auth.py --profile production authentication 653384
```

### Authentication (Identification)

```bash
# Tanpa user_id - akan identify dari database
python test_websocket_auth.py --profile production authentication
```

## ⚙️ Configuration

File `web_config.json` sudah configured:

```json
{
  "default_profile": "production",
  "profiles": {
    "production": {
      "base_url": "https://face.ahu.go.id",
      "api_key": "frapi_YY7OEJn1FCyDoGiGLwiueTw79hkQWduGNy2L-XbsCB4",
      "secret_key": "_lwZfcqmdsi5PtRLjmOeTDgTxP5JaRAN3r4i6IpCSOC6ndL536sO9ZuFVjbgLshbmuNKmBButy_wZgdyXEw-DA"
    }
  }
}
```

No need untuk type credentials setiap kali!

## 🎥 What You'll See (CLI)

When running CLI client:

1. **Authentication** - Client authenticates dengan API
2. **Session Creation** - Enrollment/authentication session dibuat
3. **WebSocket Connection** - ✅ Connects successfully
4. **Camera Opens** - OpenCV window shows your webcam
5. **Real-time Overlays:**
   - Face bounding box dengan corner markers
   - Face mesh landmarks (oval, eyebrows, nose, lips)
   - Eye regions untuk blink detection
   - Status panel: frames, liveness score, blinks, motion, quality
6. **Live Feedback:**
   ```
   📊 Frame 5/10
      Liveness Score: 0.87
      Blinks: 2/1 ✅
      Motion: 3/1 ✅
      No Obstacles: ✅
      Quality: 0.92 (threshold: 0.65)
      💬 Continue - Looking good!
   ```
7. **Result:**
   - Enrollment: Displays enrollment ID, similarity score
   - Authentication: Shows authenticated status, confidence, user ID

## 🔍 Troubleshooting

### WebSocket Connection Failed (Browser)

**Expected!** Browser WebSocket di-block oleh CORS.

**Solution:** Use Python CLI (recommended)

```bash
python test_websocket_auth.py --profile production enrollment 653384
```

### Camera Not Found (CLI)

```bash
# Try different camera index
python test_websocket_auth.py --profile production enrollment 653384
# Then when it asks, try camera 0, 1, or 2
```

### ModuleNotFoundError

Install dependencies:

```bash
pip install opencv-python websockets requests cryptography numpy
```

## 📊 Features

### CLI Client Features:
- ✅ Real-time face detection
- ✅ Face mesh landmarks visualization
- ✅ Blink detection with EAR tracking
- ✅ Motion detection
- ✅ Obstacle detection (mask, glasses, etc)
- ✅ Quality score monitoring
- ✅ Liveness score calculation
- ✅ Old photo similarity comparison
- ✅ Encrypted response handling
- ✅ Frame-by-frame feedback
- ✅ Visual overlays and indicators

### Enrollment Features:
- Multi-frame capture (3 samples)
- Face quality validation
- Liveness verification (blinks + motion)
- Obstacle detection and rejection
- Similarity with old profile photo
- Automatic retry on failure

### Authentication Features:
- Verification (1:1 matching)
- Identification (1:N search)
- Confidence scoring
- Liveness requirement
- Real-time feedback

## 🌐 Web Interface (Optional)

Jika ingin test web interface (note: WebSocket will fail):

```bash
# Start server
python web_server.py

# Open browser
http://localhost:8080
```

Web interface will:
- ✅ Load configuration automatically
- ✅ Call HTTPS API successfully
- ❌ Fail to connect WebSocket (CORS)

You'll see error modal suggesting to use CLI.

## 💡 Best Practices

### For Production:
```bash
# Always use CLI
python test_websocket_auth.py --profile production enrollment "${USER_ID}"
```

### For Development:
```bash
# Use config profiles
python test_websocket_auth.py --profile local enrollment test_user
```

### For Integration:
Import and use the `FaceAuthWebSocketClient` class directly in your code:

```python
from test_websocket_auth import FaceAuthWebSocketClient

client = FaceAuthWebSocketClient(api_key, secret_key, base_url)
await client.run_enrollment(user_id="123", old_photo_path="photo.jpg")
```

## 📚 Documentation

- **QUICK_START.md** - Complete quick start guide
- **WEBSOCKET_TROUBLESHOOTING.md** - Why browser fails, solutions
- **WEB_CLIENT_README.md** - Web interface documentation

## ✅ TL;DR

**Single working command:**

```bash
python test_websocket_auth.py --profile production enrollment 653384
```

✅ Configured  
✅ Working  
✅ Ready to use  

That's it! 🎉

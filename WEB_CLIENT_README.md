# Face Recognition WebSocket Client - Web Interface

Web-based client untuk Face Recognition API dengan dukungan real-time WebSocket untuk enrollment dan authentication.

## 📋 File-File yang Dibuat

- `web_face_auth.html` - Interface web HTML dengan WebSocket client
- `web_server.py` - Python HTTP server untuk melayani HTML

## 🚀 Quick Start

### 1. Jalankan Web Server

```bash
# Cara 1: Run dengan host default (localhost:8080)
python web_server.py

# Cara 2: Run pada port tertentu
python web_server.py --port 3000

# Cara 3: Run agar accessible dari network lain
python web_server.py --public --port 8080
```

### 2. Buka Browser

Buka browser dan akses: `http://localhost:8080`

### 3. Konfigurasi Server

Pada interface web, isi konfigurasi:
- **Base URL**: URL server Face Recognition API (contoh: `https://face.ahu.go.id`)
- **API Key**: API key Anda
- **Secret Key**: Secret key Anda

### 4. Pilih Mode

- **Enrollment** 📝 - Mendaftarkan wajah user baru
- **Authentication** ✓ - Verifikasi/Identifikasi wajah

### 5. Konfigurasi Kamera

- Pilih device kamera
- Set durasi (detik)
- Untuk enrollment, opsional upload foto lama untuk perbandingan similarity

### 6. Click "Connect & Start"

Client akan:
1. Authenticate dengan server API
2. Membuat session (enrollment/authentication)
3. Connect ke WebSocket
4. Capture frame dari webcam dan mengirim ke server
5. Menampilkan real-time feedback dan liveness detection

## 🎯 Fitur

### Real-Time Visual Feedback
- ✅ Face bounding box detection
- ✅ Face mesh landmarks visualization
- ✅ Eye regions untuk blink detection
- ✅ Status panel dengan metrics
- ✅ Live log output

### Liveness Detection
- ✅ Blink detection
- ✅ Motion detection
- ✅ Eye aspect ratio (EAR) tracking
- ✅ Quality score monitoring

### Enrollment Features
- ✅ Multi-frame enrollment
- ✅ Obstacle detection (glasses, mask, dll)
- ✅ Similarity comparison dengan foto lama
- ✅ Encrypted response handling

### Authentication Features
- ✅ Verification (dengan user ID)
- ✅ Identification (tanpa user ID)
- ✅ Confidence score
- ✅ Liveness requirement

## 🔧 API Configuration

Pastikan server Face Recognition API sudah running dengan endpoints:
- `POST /api/core/auth/client/` - Client authentication
- `POST /api/auth/enrollment/` - Create enrollment session
- `POST /api/auth/authentication/` - Create authentication session
- WebSocket endpoint untuk frame processing

## 📊 Status Monitoring

Interface menampilkan:
- Connection status
- Session token
- Frames processed
- Liveness score
- Blink count
- Motion events
- Quality score
- Real-time log output

## ⚙️ Server Options

```bash
python web_server.py --help

Optional arguments:
  --host HOST       Host to bind to (default: 127.0.0.1)
  --port PORT       Port to bind to (default: 8080)
  --public          Bind to 0.0.0.0 for external access
```

## 🌐 Network Access

### Local Only (Default)
```bash
python web_server.py
# Access: http://localhost:8080
```

### Network Access
```bash
python web_server.py --public --port 8080
# Access: http://<your-ip>:8080
```

## 🔒 Security Considerations

⚠️ **Development Use Only!**

- Jangan expose server ke internet tanpa authentication
- API keys dan secrets visible di console - gunakan dengan hati-hati
- Untuk production, implementasikan:
  - HTTPS/WSS encryption
  - API key validation
  - CORS restrictions
  - Rate limiting
  - Authentication layer

## 🐛 Troubleshooting

### Camera tidak terdeteksi
- Pastikan browser memiliki permission akses camera
- Check browser console untuk error details
- Coba restart browser

### WebSocket connection failed
- Verify base URL sudah benar
- Check API credentials (API Key, Secret Key)
- Ensure server API sudah running
- Check browser console untuk error details

### Frame processing issues
- Ensure adequate lighting
- Face harus visible dalam frame
- Check quality score di status panel
- Blink dan gerakkan kepala untuk liveness detection

### API Authentication failed
- Verify API Key dan Secret Key
- Check base URL (format: https://domain atau http://localhost:port)
- Ensure server API endpoints accessible

## 📖 Comparison dengan Python CLI

File `test_websocket_auth.py` adalah CLI version dengan fitur yang sama tapi:

| Feature | Web | CLI |
|---------|-----|-----|
| Visual Interface | ✅ | ❌ |
| Real-time Feedback | ✅ | ✅ |
| Face Mesh Visualization | ✅ | ❌ |
| Camera Input | ✅ | ✅ |
| Automation | ❌ | ✅ |

### Gunakan Web jika:
- Ingin visual interface
- Testing interactively
- Demo kepada user

### Gunakan CLI jika:
- Automation/scripting
- Batch processing
- Integration dengan backend

## 📝 Enrollment Example

1. **Konfigurasi**
   - Base URL: `https://face.ahu.go.id`
   - API Key: `your_api_key`
   - Secret Key: `your_secret_key`

2. **Setup**
   - Mode: Enrollment
   - User ID: `user123`
   - Old Photo: (optional) `/path/to/old_photo.jpg`

3. **Start**
   - Click "Connect & Start"
   - Pastikan wajah terlihat jelas dalam frame
   - Blink alami
   - Gerakkan kepala slightly left/right

4. **Complete**
   - Server akan process frame-frame
   - Enrollment complete saat cukup frames dengan quality baik

## 🔐 Authentication Example

1. **Konfigurasi** - sama seperti enrollment
2. **Setup**
   - Mode: Authentication
   - User ID: `user123` (untuk verification) atau kosongkan untuk identification
3. **Start** - sama seperti enrollment
4. **Result** - akan menampilkan authenticated status dan confidence score

## 📞 Support

Untuk issue atau pertanyaan:
1. Check browser console (F12) untuk error details
2. Check server console output
3. Verify API configuration
4. Check API server logs

## 📄 License

Bagian dari Face Recognition System v2

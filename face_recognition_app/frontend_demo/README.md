# Face Recognition System Demo

Demo aplikasi frontend untuk menguji semua endpoint API dari sistem face recognition Django backend.

## 🚀 Fitur Demo

### 1. Authentication
- **User Registration**: Registrasi pengguna baru dengan email dan password
- **User Login**: Login menggunakan email dan password (JWT token)
- **User Profile**: Melihat dan mengelola profil pengguna

### 2. Face Enrollment
- **Camera Integration**: Akses kamera untuk mengambil sampel wajah
- **Real-time Processing**: Pemrosesan frame wajah secara real-time
- **Progress Tracking**: Monitoring progress enrollment dengan progress bar
- **Blink Detection**: Deteksi kedipan mata untuk liveness detection
- **Quality Feedback**: Feedback kualitas gambar dan pencahayaan

### 3. Face Recognition
- **Face Authentication**: Autentikasi menggunakan wajah
- **Confidence Scoring**: Skor confidence dari proses pengenalan
- **Attempt History**: Riwayat percobaan autentikasi
- **Real-time Results**: Hasil autentikasi secara real-time

### 4. Streaming & WebRTC
- **Video Streaming**: Streaming video menggunakan WebRTC
- **Signaling**: WebRTC signaling (offer, answer, ICE candidates)
- **Session Management**: Manajemen sesi streaming
- **Connection Status**: Status koneksi WebRTC

### 5. Analytics
- **Dashboard**: Dashboard analitik dengan statistik
- **Authentication Logs**: Log autentikasi pengguna
- **Security Alerts**: Alert keamanan sistem
- **Statistics**: Statistik penggunaan sistem

### 6. User Management
- **User Profile**: Profil pengguna lengkap
- **Device Management**: Manajemen device yang terdaftar
- **Authentication History**: Riwayat autentikasi pengguna
- **Security Settings**: Pengaturan keamanan

### 7. System Status
- **Health Monitoring**: Monitoring kesehatan sistem
- **API Testing**: Testing endpoint API custom
- **System Configuration**: Konfigurasi sistem
- **Performance Metrics**: Metrik performa sistem

## 📋 Endpoint API yang Diimplementasikan

### Core API (`/api/`)
- `POST /api/auth/register/` - User registration
- `POST /api/auth/token/` - User login (JWT)
- `GET /api/auth/profile/` - User profile
- `POST /api/enrollment/create/` - Create enrollment session
- `POST /api/enrollment/process-frame/` - Process enrollment frame
- `POST /api/auth/face/create/` - Create authentication session
- `POST /api/auth/face/process-frame/` - Process authentication frame
- `POST /api/webrtc/signal/` - WebRTC signaling

### Recognition API (`/api/recognition/`)
- `GET /api/recognition/embeddings/` - Face embeddings
- `GET /api/recognition/sessions/` - Enrollment sessions
- `GET /api/recognition/attempts/` - Authentication attempts

### Streaming API (`/api/streaming/`)
- `GET /api/streaming/sessions/` - Streaming sessions
- `POST /api/streaming/sessions/create/` - Create streaming session
- `POST /api/streaming/signaling/` - WebRTC signaling

### Analytics API (`/api/analytics/`)
- `GET /api/analytics/auth-logs/` - Authentication logs
- `GET /api/analytics/security-alerts/` - Security alerts
- `GET /api/analytics/dashboard/` - Analytics dashboard
- `GET /api/analytics/statistics/` - System statistics

### Users API (`/api/users/`)
- `GET /api/users/devices/` - User devices
- `GET /api/user/auth-history/` - Authentication history
- `GET /api/user/security-alerts/` - User security alerts

### System API (`/api/system/`)
- `GET /api/system/status/` - System status

## 🛠️ Cara Menjalankan Demo

### 1. Persiapan Backend Django
Pastikan backend Django sudah berjalan di `127.0.0.1:8000`:

```bash
cd face_recognition_app
python manage.py runserver 127.0.0.1:8000
```

### 2. Menjalankan Demo Frontend

#### Opsi 1: Menggunakan Live Server (Recommended)
```bash
# Jika menggunakan Python
cd frontend_demo
python -m http.server 8080

# Jika menggunakan Node.js
npx live-server --port=8080

# Jika menggunakan PHP
php -S localhost:8080
```

#### Opsi 2: Menggunakan VS Code Live Server Extension
1. Install extension "Live Server" di VS Code
2. Buka file `index.html`
3. Klik "Go Live" di status bar

#### Opsi 3: Langsung membuka di browser
Buka file `index.html` langsung di browser (mungkin ada CORS issues)

### 3. Akses Demo
Buka browser dan navigasi ke:
```
http://localhost:8080
```

## 📖 Panduan Penggunaan

### 1. User Registration & Login
1. Buka tab **Authentication**
2. Isi form registrasi dengan:
   - Email
   - First Name, Last Name
   - Password & Confirm Password
3. Klik "Register"
4. Setelah registrasi berhasil, login menggunakan email dan password
5. Status autentikasi akan berubah dan semua tab akan terbuka

### 2. Face Enrollment
1. Setelah login, buka tab **Enrollment**
2. Klik "Start Enrollment"
3. Berikan akses kamera saat diminta
4. Ikuti instruksi di layar:
   - Lihat ke kamera
   - Kedipkan mata secara natural
   - Pastikan pencahayaan cukup
5. Progress enrollment akan ditampilkan
6. Proses selesai setelah 5 sampel berhasil dikumpulkan

### 3. Face Authentication
1. Buka tab **Face Recognition**
2. Klik "Start Authentication"
3. Lihat ke kamera
4. Sistem akan mencoba mengenali wajah
5. Hasil autentikasi akan ditampilkan
6. Lihat riwayat attempts di panel samping

### 4. WebRTC Streaming
1. Buka tab **Streaming**
2. Klik "Start Streaming"
3. Test WebRTC signaling:
   - Create Offer
   - Create Answer
   - Add ICE Candidate
4. Monitor log signaling di bawah

### 5. Analytics
1. Buka tab **Analytics**
2. Klik "Load Analytics" untuk dashboard
3. Klik "Load Statistics" untuk statistik
4. Lihat authentication logs dan security alerts
5. Data akan auto-refresh setiap 30 detik

### 6. User Management
1. Buka tab **Users**
2. Load user profile, devices, dan history
3. Monitor security alerts khusus user

### 7. System Status
1. Buka tab **System**
2. Klik "Check System Status" untuk health check
3. Test custom endpoint di panel API Testing

## 🔧 Konfigurasi

### Server Configuration
Default server: `127.0.0.1:8000`

Untuk mengubah server, edit file `js/api.js`:
```javascript
constructor() {
    this.baseURL = 'http://your-server:port';
    // ...
}
```

### CORS Configuration
Pastikan Django backend mengizinkan CORS dari frontend:

```python
# settings.py
CORS_ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
```

## 📱 Browser Compatibility

Demo membutuhkan browser modern dengan dukungan:
- WebRTC (getUserMedia)
- Canvas API
- ES6+ JavaScript
- Fetch API

Tested browsers:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 🐛 Troubleshooting

### Camera Issues
- Pastikan browser memiliki permission kamera
- Coba refresh halaman jika kamera tidak terdeteksi
- Periksa apakah aplikasi lain menggunakan kamera

### API Connection Issues
- Pastikan Django backend berjalan di `127.0.0.1:8000`
- Cek console browser untuk error CORS
- Verify endpoint URLs di network tab

### Authentication Issues
- Clear localStorage jika ada masalah token
- Pastikan email/password benar saat login
- Check token expiry di console

## 📝 Log Monitoring

Semua aktivitas API tercatat di:
1. **Global Log** di bawah halaman - menampilkan semua request/response
2. **Browser Console** - untuk debugging
3. **Network Tab** - untuk monitoring HTTP requests

## 🔐 Security Notes

Demo ini untuk testing purposes. Dalam production:
- Gunakan HTTPS
- Implement proper CORS policies
- Validate semua input
- Use secure token storage
- Implement rate limiting

## 📞 Support

Jika ada masalah:
1. Cek browser console untuk error
2. Verify backend Django sedang berjalan
3. Test endpoint menggunakan tool API Testing di tab System
4. Periksa network connectivity

## 🚀 Next Steps

Untuk development lebih lanjut:
1. Implementasi real face detection (OpenCV.js)
2. Improve WebRTC signaling server
3. Add real-time notifications
4. Implement proper error handling
5. Add unit tests
6. Optimize performance

---

**Demo Ready! 🎉**

Aplikasi demo lengkap untuk menguji semua fitur Face Recognition System. Semua endpoint backend telah diimplementasikan dengan UI yang user-friendly dan monitoring lengkap.
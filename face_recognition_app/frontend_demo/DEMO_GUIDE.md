# Face Recognition Demo Application

Demo aplikasi lengkap untuk sistem Face Recognition dengan semua fitur endpoint yang tersedia.

## 🚀 Fitur Utama

### ✅ Sudah Berfungsi:
- **User Registration & Login** - Registrasi dan login user dengan JWT authentication
- **Face Enrollment** - Proses enrollment wajah dengan kamera real-time
- **Face Authentication** - Autentikasi menggunakan wajah
- **WebRTC Streaming** - Streaming video untuk face recognition
- **Analytics Dashboard** - Dashboard analitik untuk monitoring
- **User Management** - Manajemen profil user dan device
- **System Status** - Monitor status sistem dan health check

### 🔧 Perbaikan Terbaru:
1. **JWT Authentication**: Fixed custom serializer untuk accept email instead of username
2. **Session Management**: Fixed enrollment/authentication session handling
3. **Error Handling**: Improved error handling dengan session restart capability
4. **Database Issues**: Fixed SerializerMethodField naming dan null constraints
5. **API Field Mapping**: Fixed field names antara frontend dan backend

## 📡 API Endpoints

### Authentication
- `POST /api/auth/register/` - User registration
- `POST /api/auth/token/` - Login dengan email/password
- `GET /api/auth/profile/` - Get user profile

### Face Recognition
- `POST /api/enrollment/create/` - Create enrollment session
- `POST /api/enrollment/process-frame/` - Process enrollment frame
- `POST /api/auth/face/create/` - Create authentication session  
- `POST /api/auth/face/process-frame/` - Process authentication frame

### WebRTC Streaming
- `GET /api/streaming/sessions/` - List streaming sessions
- `POST /api/streaming/sessions/create/` - Create streaming session
- `POST /api/streaming/signaling/` - WebRTC signaling

### Analytics
- `GET /api/analytics/auth-logs/` - Authentication logs
- `GET /api/analytics/security-alerts/` - Security alerts
- `GET /api/analytics/dashboard/` - Analytics dashboard
- `GET /api/analytics/statistics/` - Statistics

### Recognition Data
- `GET /api/recognition/sessions/` - Enrollment sessions
- `GET /api/recognition/embeddings/` - Face embeddings
- `GET /api/recognition/attempts/` - Authentication attempts

### Users
- `GET /api/users/profile/` - User profile
- `GET /api/users/devices/` - User devices

### System
- `GET /api/system/status/` - System status

## 🛠️ Cara Menjalankan

### 1. Start Django Backend
```bash
cd /Users/user/Dev/researchs/face_regocnition_v2/face_recognition_app
python manage.py runserver 127.0.0.1:8000
```

### 2. Start Frontend Demo
```bash
# Opsi 1: Menggunakan script
./start_demo.sh

# Opsi 2: Manual
cd frontend_demo
python3 -m http.server 8080
```

### 3. Akses Demo
- Frontend: http://127.0.0.1:8080
- Backend API: http://127.0.0.1:8000
- API Documentation: http://127.0.0.1:8000/api/docs/

## 🎯 Cara Menggunakan Demo

1. **Registration**: Buat akun baru di tab Authentication
2. **Login**: Login dengan email dan password
3. **Enrollment**: Enroll wajah di tab Face Enrollment
4. **Authentication**: Test face authentication di tab Face Recognition
5. **Analytics**: Lihat logs dan statistics di tab Analytics
6. **System Monitor**: Monitor system health di tab System

## 🔍 Troubleshooting

### Session Expired Errors
- Jika terjadi "Invalid or expired session", demo akan otomatis offer restart session
- Klik OK untuk create session baru

### Camera Permission
- Pastikan browser memiliki permission untuk mengakses camera
- Gunakan HTTPS untuk production (saat ini menggunakan HTTP untuk development)

### CORS Issues  
- Backend sudah dikonfigurasi untuk accept requests dari localhost:8080
- Jika ada masalah CORS, pastikan frontend running di port 8080

## 📋 Status Implementasi

| Feature | Status | Notes |
|---------|--------|-------|
| User Auth | ✅ | JWT dengan email-based login |
| Face Enrollment | ✅ | Real-time camera dengan progress tracking |
| Face Authentication | ✅ | Dengan session restart capability |
| WebRTC Streaming | ✅ | Basic streaming session management |
| Analytics | ✅ | Logs, alerts, statistics |
| User Management | ✅ | Profile, devices, history |
| System Monitor | ✅ | Health check dan status |
| Error Handling | ✅ | Comprehensive error handling |

## 🐛 Known Issues

- Face recognition engine mungkin return mock data (tergantung implementasi backend)
- WebRTC signaling masih basic implementation  
- Camera quality detection belum optimal
- Perlu HTTPS untuk production deployment

## 📝 Development Notes

- Semua API calls di-log untuk debugging
- Session management menggunakan JWT + custom session tokens
- Real-time camera processing dengan interval 1 second
- Error handling dengan user-friendly messages dan restart options
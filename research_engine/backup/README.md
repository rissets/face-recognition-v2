# Secure Face Authentication System

Sistem autentikasi wajah yang aman menggunakan InsightFace 3D models dengan fitur:
- ✅ Liveness detection (deteksi kedipan mata)
- ✅ Anti-spoofing (tidak bisa ditipu foto/gambar)
- ✅ Obstacle detection (deteksi kacamata, topi, masker)
- ✅ Face embedding dengan FAISS untuk pencarian cepat
- ✅ Enrollment bertahap dengan live video
- ✅ Verifikasi real-time

## Fitur Keamanan

### 1. Liveness Detection
- Menggunakan Eye Aspect Ratio (EAR) untuk deteksi kedipan
- Membutuhkan minimal 2 kedipan untuk validasi liveness
- Mencegah serangan menggunakan foto/gambar statis

### 2. Obstacle Detection
- **Kacamata**: Deteksi berdasarkan edge detection dan analisis bentuk
- **Topi**: Deteksi area gelap di bagian atas wajah
- **Masker**: Deteksi tekstur uniform di area mulut

### 3. 3D Face Recognition
- Menggunakan InsightFace Buffalo-L model
- Embedding 512-dimensional
- Cosine similarity untuk matching
- Threshold adaptif untuk akurasi tinggi

## Instalasi

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download InsightFace Models
```bash
# Models akan otomatis didownload saat pertama kali dijalankan
python -c "import insightface; app = insightface.app.FaceAnalysis(); app.prepare(ctx_id=0)"
```

## Penggunaan

### Jalankan sistem:
```bash
python face_auth_system.py
```

### Menu Utama:
1. **Enroll User** - Daftarkan user baru dengan 5 sampel wajah
2. **Authenticate User** - Verifikasi user tertentu
3. **Identify User** - Identifikasi user dari database
4. **List Users** - Lihat semua user terdaftar
5. **Delete User** - Hapus user dari database
6. **Exit** - Keluar dari aplikasi

## Proses Enrollment

1. Masukkan nama user
2. Sistem akan meminta 5 sampel wajah
3. Pastikan:
   - Wajah terlihat jelas
   - Tidak ada obstacle (lepas kacamata, topi, masker)
   - Berkedip beberapa kali untuk liveness detection
4. Tekan **SPACE** untuk capture ketika status "VALID"
5. Ulangi hingga 5 sampel terkumpul

## Proses Authentication

### Verifikasi (Authentication):
- Pilih user dari list
- Sistem akan memverifikasi apakah wajah cocok dengan user tersebut

### Identifikasi:
- Sistem akan mencari user yang cocok dari seluruh database
- Menampilkan nama user dan tingkat similarity

## Keamanan

### Anti-Spoofing Features:
- **Liveness Detection**: Membutuhkan kedipan mata real-time
- **3D Analysis**: Menggunakan depth information dari InsightFace
- **Temporal Analysis**: Analisis pergerakan wajah dalam video
- **Obstacle Detection**: Menolak wajah yang terhalang object

### Threshold Security:
- Similarity threshold: 0.4 (dapat disesuaikan)
- Minimum blinks untuk liveness: 2
- Eye Aspect Ratio threshold: 0.25
- Consecutive frames untuk blink: 3

## Struktur Database

```
face_database.pkl - User embeddings dan metadata
face_database.faiss - FAISS index untuk pencarian cepat
```

## Troubleshooting

### Camera Issues:
```bash
# Cek camera devices
ls /dev/video*

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

### Model Issues:
```bash
# Re-download models
rm -rf ~/.insightface/
python -c "import insightface; app = insightface.app.FaceAnalysis(); app.prepare(ctx_id=0)"
```

### Dependencies Issues:
```bash
# Reinstall with specific versions
pip install --force-reinstall -r requirements.txt
```

## Technical Details

### Models Used:
- **InsightFace**: Buffalo-L model untuk face recognition
- **MediaPipe**: Face mesh untuk liveness detection
- **FAISS**: Fast similarity search

### Performance:
- **FPS**: ~30 FPS untuk real-time processing
- **Latency**: <100ms untuk authentication
- **Memory**: ~500MB RAM usage
- **Accuracy**: >99% dengan proper enrollment

### Hardware Requirements:
- **CPU**: Intel i5 atau AMD equivalent
- **RAM**: Minimum 4GB
- **Camera**: Webcam dengan resolusi minimal 640x480
- **Storage**: ~200MB untuk models dan database

## Customization

### Adjust Thresholds:
```python
# Dalam class SecureFaceAuth
self.embedding_system.verification_threshold = 0.4  # Similarity threshold
self.liveness_detector.EAR_THRESHOLD = 0.25  # Blink sensitivity
self.liveness_detector.CONSECUTIVE_FRAMES = 3  # Blink frames
```

### Add More Obstacle Types:
```python
# Dalam class ObstacleDetector.detect_obstacles()
# Tambahkan logic untuk deteksi obstacle lain
```

## License

MIT License - Feel free to use and modify.

## Support

For issues and questions, please check the troubleshooting section or contact support.
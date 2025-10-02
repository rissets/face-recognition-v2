# Panduan Cepat - Face Authentication System

## Quick Start

### 1. Setup (Hanya sekali)
```bash
./setup.sh
```

### 2. Jalankan Sistem
```bash
./run.sh          # Sistem utama
./run_demo.sh     # Demo & testing
```

### 3. Aktivasi Manual Environment (jika diperlukan)
```bash
source env/bin/activate    # atau
./activate_env.sh
```

## Workflow Penggunaan

### 📋 Enrollment User Baru
1. Pilih menu **"1. Enroll User"**
2. Masukkan nama user
3. Ikuti instruksi di layar:
   - Pastikan wajah terlihat jelas
   - Lepas kacamata/topi/masker
   - Berkedip beberapa kali
   - Tekan **SPACE** saat status "VALID"
4. Ulangi hingga 5 sampel terkumpul

### 🔐 Verifikasi User
1. Pilih menu **"2. Authenticate User"**
2. Pilih user dari daftar
3. Lihat ke kamera dan berkedip
4. Sistem akan menampilkan hasil verifikasi

### 🔍 Identifikasi User
1. Pilih menu **"3. Identify User"**
2. Lihat ke kamera dan berkedip
3. Sistem akan mencari user yang cocok

## Tips Penggunaan

### ✅ Untuk Hasil Terbaik:
- Gunakan pencahayaan yang baik
- Posisi wajah tegak lurus ke kamera
- Jarak 50-80cm dari kamera
- Berkedip natural (2-3 kali)
- Hindari gerakan terlalu cepat

### ❌ Hindari:
- Memakai kacamata hitam
- Lighting terlalu gelap/terang
- Wajah miring atau tertutup
- Background yang terlalu ramai

## Troubleshooting

### Kamera Tidak Terdeteksi
```bash
# Check camera
ls /dev/video*
# Test manual
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'ERROR')"
```

### Model Tidak Terdownload
```bash
# Re-download models
rm -rf ~/.insightface/
./setup.sh
```

### Environment Issues
```bash
# Reset environment
rm -rf env/
./setup.sh
```

## Struktur File

```
face_regocnition_v2/
├── face_auth_system.py     # Sistem utama
├── demo.py                 # Demo & testing
├── setup.sh                # Install dependencies
├── run.sh                  # Jalankan sistem utama
├── run_demo.sh             # Jalankan demo
├── activate_env.sh         # Aktivasi environment
├── config.json             # Konfigurasi sistem
├── requirements.txt        # Python dependencies
├── README.md               # Dokumentasi lengkap
└── QUICK_START.md          # Panduan cepat (file ini)
```

## Keamanan

### Fitur Anti-Spoofing:
- ✅ Liveness detection (kedipan mata)
- ✅ 3D face analysis
- ✅ Temporal movement analysis
- ✅ Obstacle detection

### Database:
- `face_database.pkl` - User embeddings
- `face_database.faiss` - Index untuk pencarian cepat

## Performance

- **FPS**: ~30 untuk real-time
- **Latency**: <100ms untuk authentication
- **Accuracy**: >99% dengan enrollment yang proper
- **Memory**: ~500MB RAM

## Support

Jika ada masalah:
1. Cek camera dengan demo
2. Pastikan lighting cukup
3. Reset environment jika perlu
4. Cek requirements.txt untuk dependencies
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# 1. Tentukan provider yang tersedia untuk Anda (macOS)
providers_list = ['CoreMLExecutionProvider', 'CPUExecutionProvider']

print(f"Menggunakan providers: {providers_list}")

# 2. Masukkan 'providers' saat inisialisasi FaceAnalysis
#    Ini memperbaiki TypeError 'unexpected keyword argument' yang pertama
app = FaceAnalysis(name='buffalo_l', 
                   allowed_modules=['detection', 'liveness'],
                   providers=providers_list)

# 3. Prepare model
print("Mempersiapkan model (det_size=640x640)...")
app.prepare(ctx_id=0, det_size=(640, 640))

print("Model siap. Membuka webcam...")

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam.")
    exit()

# Threshold untuk liveness
LIVENESS_THRESHOLD = 0.5 

print("Tekan 'q' untuk keluar...")

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame.")
        break

    # Deteksi wajah dan analisis liveness
    faces = app.get(frame)

    # Iterasi melalui setiap wajah yang terdeteksi
    for face in faces:
        bbox = face.bbox.astype(int)
        
        # Ambil skor liveness
        liveness_score = face.liveness_score

        # --- PERBAIKAN KEDUA (TypeError) ---
        # Cek dulu apakah skor liveness ada (tidak None)
        # Ini terjadi jika model liveness ('liveness.onnx') tidak ditemukan
        
        print(f"Liveness Score: {liveness_score}")  # Debug: Cetak skor liveness
        if liveness_score is None:
            label = "LIVENESS N/A"
            color = (128, 128, 128) # Abu-abu
            
        elif liveness_score > LIVENESS_THRESHOLD:
            # Liveness terdeteksi REAL
            label = f"REAL: {liveness_score:.2f}"
            color = (0, 255, 0)  # Hijau
            
        else:
            # Liveness terdeteksi SPOOF
            label = f"SPOOF: {liveness_score:.2f}"
            color = (0, 0, 255)  # Merah

        # --- AKHIR PERBAIKAN ---

        # Gambar bounding box dan label pada frame
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Tampilkan frame
    cv2.imshow('Passive Liveness Detection (InsightFace)', frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
import cv2
import insightface
import numpy as np
import onnxruntime as ort

# =============================================================================
# Konfigurasi
# =============================================================================

# 1. Path ke model anti-spoofing ONNX Anda
# PASTIKAN INI SUDAH BENAR
ANTI_SPOOF_MODEL_PATH = "models/model_fas.onnx"

# 2. Threshold untuk liveness
# Coba turunkan jika masih sulit, misal 0.7
LIVENESS_THRESHOLD = 0.9

# =============================================================================
# Fungsi Helper
# =============================================================================


def softmax(x):
    """Menghitung softmax untuk array input."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def preprocess_liveness(image_crop_bgr):
    """
    Pre-processing gambar untuk model anti-spoofing.
    Model ini mengharapkan input 256x256, RGB, ternormalisasi.
    """
    # 1. Resize ke ukuran input model (sesuai error Anda)
    img_resized = cv2.resize(image_crop_bgr, (128, 128))

    # 2. PENTING: Konversi BGR (OpenCV default) ke RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 3. Konversi ke float32 dan Normalisasi
    # Mengubah piksel dari [0, 255] ke [-1, 1]
    img_data = img_rgb.astype(np.float32)
    img_data = (img_data - 127.5) / 128.0

    # 4. Transpose dari HWC (Height, Width, Channel) ke CHW (Channel, Height, Width)
    img_data = np.transpose(img_data, (2, 0, 1))

    # 5. Tambahkan dimensi Batch (Batch, Channel, Height, Width)
    img_data = np.expand_dims(img_data, axis=0)

    return img_data


# =============================================================================
# Inisialisasi Model
# =============================================================================


print("Mempersiapkan model...")
app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

ort_session = ort.InferenceSession(
    ANTI_SPOOF_MODEL_PATH, providers=["CPUExecutionProvider"]
)
input_name = ort_session.get_inputs()[0].name
print("Model siap.")

# =============================================================================
# Loop Webcam Utama
# =============================================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam.")
    exit()

print("Menjalankan deteksi. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    result_frame = frame.copy()

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        pad = 10
        face_crop = frame[
            max(0, y1 - pad) : min(y2 + pad, frame.shape[0]),
            max(0, x1 - pad) : min(x2 + pad, frame.shape[1]),
        ]

        if face_crop.size == 0:
            continue

        # 2. CEK ANTI-SPOOFING
        # Fungsi di bawah ini SEKARANG sudah mengkonversi ke RGB
        input_blob = preprocess_liveness(face_crop)

        raw_output = ort_session.run(None, {input_name: input_blob})[0]
        probabilities = softmax(raw_output)

        # Ambil skor
        # Indeks [0][0] = FAKE
        # Indeks [0][1] = REAL (ASLI)
        fake_score = probabilities[0][0]
        real_score = probabilities[0][1]

        # --- DIAGNOSTIK: Lihat skor di terminal ---
        print(f"FAKE: {fake_score:.4f} | REAL: {real_score:.4f}")
        # -------------------------------------------

        label = ""  # Akan diisi di bawah

        if real_score > LIVENESS_THRESHOLD:
            color = (0, 255, 0)  # Hijau
            label = f"ASLI: {real_score:.2f}"

            # Di sinilah Anda melakukan logika pengenalan
            # print(f"Wajah ASLI terdeteksi, embedding: {face.rec_embedding[:5]}...") # Contoh

        else:
            color = (0, 0, 255)  # Merah
            label = f"PALSU: {real_score:.2f}"

        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    cv2.imshow("InsightFace Anti-Spoofing", result_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Selesai.")

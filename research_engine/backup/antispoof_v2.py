import cv2
from insightface.app import FaceAnalysis

print("Loading InsightFace FaceAnalysis model...")
# Inisialisasi FaceAnalysis dengan modul 'detection' dan 'antispoof'
# 'detection' untuk menemukan wajah
# 'antispoof' untuk mengecek liveness
app = FaceAnalysis(allowed_modules=["detection", "antispoof"])
# 'ctx_id=0' berarti menggunakan GPU 0. Ganti menjadi -1 jika Anda hanya punya CPU.
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded successfully.")

# Mulai capture webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Balik frame secara horizontal (seperti cermin)
    frame = cv2.flip(frame, 1)

    # Salin frame untuk menghindari modifikasi pada frame asli
    img_draw = frame.copy()

    try:
        # Dapatkan hasil analisis wajah dari InsightFace
        # Ini akan mendeteksi wajah DAN menjalankan cek antispoofing
        faces = app.get(frame)

        if faces:
            for face in faces:
                # Dapatkan bounding box
                bbox = face.bbox.astype(int)

                # --- INI BAGIAN KUNCINYA ---
                # 'face.is_live' akan berisi:
                # 1: jika terdeteksi LIVE (orang asli)
                # -1: jika terdeteksi SPOOF (foto, video, topeng)
                # 0: jika tidak yakin

                label = "UNSURE"
                color = (255, 150, 0)  # Oranye untuk "UNSURE"

                if face.is_live == 1:
                    label = "LIVE"
                    color = (0, 255, 0)  # Hijau untuk "LIVE"
                elif face.is_live == -1:
                    label = "SPOOF"
                    color = (0, 0, 255)  # Merah untuk "SPOOF"

                # Gambar kotak di sekitar wajah
                cv2.rectangle(
                    img_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2
                )

                # Tulis label (LIVE/SPOOF) di atas kotak
                cv2.putText(
                    img_draw,
                    label,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

    except Exception as e:
        print(f"An error occurred during face analysis: {e}")

    # Tampilkan hasilnya
    cv2.imshow("InsightFace Antispoofing", img_draw)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Hentikan webcam dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
print("Webcam released and windows closed.")

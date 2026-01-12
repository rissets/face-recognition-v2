# Passive Liveness Detection v4

Versi ini membangun sistem passive liveness end-to-end dengan kombinasi *texture*, natural blink, mikro gerakan, pola pantulan cahaya, serta deteksi artefak spoof memakai YOLO. Semua sinyal digabung menggunakan *attention fusion network* sehingga fleksibel untuk fine-tuning lebih lanjut.

## Fitur Utama
- **Mediapipe Face Mesh** untuk landmark wajah akurat dan stabil.
- **InsightFace** (opsional) untuk embedding serta estimasi kualitas wajah.
- **Texture Analyzer** berbasis LBP + entropy.
- **Eye Blink Detector** memakai pergerakan kelopak mata (EAR).
- **Micro-movement Analyzer** berbasis pergeseran landmark (mediapipe) untuk menangkap getaran halus.
- **Light Reflection Analyzer** mendeteksi highlight intens, area terang rendah saturasi, dan pola persegi panjang khas layar.
- **YOLO artefact detector** (ultralytics atau torch hub) pada seluruh frame untuk mendeteksi layar/alat spoof. Jika perangkat seperti *cell phone* terdeteksi dekat wajah, modul langsung mengunci verdict spoof dan memberi penalty tinggi.
- **Depth variation** dari landmark 3D Mediapipe + InsightFace (68 points) untuk membedakan permukaan flat vs wajah asli.
- **Head movement** berbasis pose InsightFace (fallback ke estimasi dari mediapipe face mesh) untuk memastikan ada dinamika kepala.
- **Attention Fusion Network** (PyTorch) + skor prior heuristik untuk kestabilan awal.
- API terstruktur melalui `PassiveLivenessDetector`.

## Instalasi Dependensi

```bash
pip install opencv-python mediapipe insightface torch torchvision torchaudio ultralytics
```

> Catatan:
> - InsightFace memerlukan `onnxruntime-gpu` atau `onnxruntime` sesuai perangkat.
> - Jika tidak ingin mengunduh YOLO dari internet, sediakan sendiri bobotnya dan set `config.yolo_model_path`.
> - Error `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` biasanya muncul dari ketidakcocokan protobuf dengan Mediapipe; gunakan `pip install --upgrade --force-reinstall "protobuf<4.21"` (contoh `3.20.3`).

## Struktur Folder
- `config.py` – konfigurasi utama (`PassiveLivenessConfig`).
- `feature_extractors.py` – ekstraktor fitur (texture, blink, movement, reflection, artefact).
- `attention_fusion.py` – jaringan attention dan utilitas tensor.
- `pipeline.py` – kelas `PassiveLivenessDetector`.
- `run_passive_liveness.py` – demo/CLI capture kamera.

Jika checkpoint attention tidak tersedia, pipeline otomatis menggunakan kombinasi heuristik (texture/blink/movement/head_movement/reflection/quality/artifact/depth) plus streak-based hysteresis. Kriteria spoof menegaskan perangkat layar (YOLO), permukaan flat (depth rendah InsightFace/Mediapipe), refleksi tinggi, serta kepala yang terlalu statis; verdict *live* baru keluar setelah beberapa frame konsisten.

## Cara Pakai (CLI)

```bash
python -m research_engine.livenessv4.run_passive_liveness \
  --camera 0 \
  --yolo-weights /path/ke/yolov8n.pt \
  --fusion-checkpoint /path/ke/checkpoint.pt
```

Opsi:
- `--headless` menonaktifkan tampilan jendela dan mencetak hasil JSON per frame.
- `--yolo-weights` opsional; bila tidak diset, script coba memakai bobot default `yolov8n.pt`. Jika gagal memuat, artefact score akan 0.
- `--fusion-checkpoint` opsional; bila kosong, jaringan attention memakai bobot inisialisasi dan digabung dengan skor prior heuristik.

## Integrasi Programatik

```python
from research_engine.livenessv4 import PassiveLivenessDetector, PassiveLivenessConfig
import cv2

config = PassiveLivenessConfig(use_gpu=True, yolo_model_path="models/yolo/yolov8n.pt")
detector = PassiveLivenessDetector(config=config)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = detector.process_frame(frame)
    print(result["verdict"], result["smoothed_probability"])
```

Gunakan `detector.export_history()` untuk debugging pipeline.

## Training / Fine-tuning
- `PassiveLivenessDetector` menerima `fusion_checkpoint` sehingga Anda bisa melatih ulang `AttentionFusionNetwork` secara terpisah (misal dengan dataset internal).
- Sediakan label *live/spoof* per frame/sequence, ekstrak fitur per modul (tersedia via `result["weighted_features"]`), lalu latih model attention/MLP sesuai kebutuhan. Simpan `state_dict` PyTorch dan muat lewat argumen `--fusion-checkpoint`.

## Tips Akurasi
1. Gunakan kamera resolusi tinggi dan pencahayaan memadai.
2. Pastikan `insightface` berhasil memuat model (`buffalo_l` default). Jika GPU tersedia, set `use_gpu=True` pada config.
3. Kalibrasi threshold sesuai data produksi dengan mengevaluasi `export_history()` pada dataset real dan spoof.
4. Untuk YOLO, latih ulang model khusus artefak spoof (masker, layar, kacamata AR) dan simpan sebagai checkpoint custom.
5. Kombinasikan dengan detektor kualitas gambar (blur, brightness) untuk menolak input buruk sebelum inference.

## Lisensi / Dataset
- Contoh ini memakai model publik (Mediapipe, InsightFace, YOLO). Periksa lisensi masing-masing bila hendak dipakai secara komersial.
- Dataset untuk training tidak disertakan; gunakan dataset internal atau open-source (contoh: SiW, CelebA-Spoof, CASIA-FASD).

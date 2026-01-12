# Real-time Liveness Detection System

## Overview
Sistem deteksi liveness wajah real-time yang telah dilatih dengan model CNN untuk membedakan wajah asli dari gambar/foto.

## Quick Start

### 1. Aktivasi Environment
```bash
cd /Users/user/Dev/researchs/face_regocnition_v2
source env/bin/activate
cd research_engine/livenessv2
```

### 2. Jalankan Real-time Detection

#### CNN-based Detector (Recommended - Menggunakan Model Terlatih)
```bash
python run_liveness_realtime.py --detector cnn
```

#### Fast Detector (Multi-feature Analysis)
```bash
python run_liveness_realtime.py --detector fast
```

#### Advanced Detector (Multi-branch CNN)
```bash
python run_liveness_realtime.py --detector advanced
```

### 3. Test Model Loading (Tanpa Camera)
```bash
python run_liveness_realtime.py --test-load
```

## Controls
- **'q'**: Quit/keluar
- **'s'**: Save current frame
- **'d'**: Toggle debug info (fast detector)

## Model Information
- **Trained Model**: `models/best_model.h5`
- **Model Parameters**: 17,201,442 parameters
- **Input Size**: 128x128 pixels
- **Classes**: Real vs Fake

## Detection Methods

### 1. CNN Detector (Default)
- Menggunakan model CNN yang sudah dilatih
- Akurasi tinggi berdasarkan pembelajaran dari dataset
- Input preprocessing: resize, normalization
- Confidence threshold: 0.5

### 2. Fast Detector  
- Multi-feature analysis tanpa deep learning
- Texture pattern analysis
- Color distribution analysis
- Edge characteristic analysis
- Motion artifact detection
- Confidence threshold: 0.6

### 3. Advanced Detector
- Multi-branch CNN architecture
- Advanced texture & depth analysis
- Temporal smoothing
- Quality-based face detection
- Stability assessment

## Performance
- **FPS**: ~15-30 FPS (tergantung hardware)
- **Resolution**: 1280x720 (dapat disesuaikan)
- **Face Detection**: Haar Cascades (fallback jika DNN tidak tersedia)

## Troubleshooting

### Model Loading Issues
```bash
# Test if model can be loaded
python run_liveness_realtime.py --test-load

# Check model file exists
ls -la models/best_model.h5
```

### Camera Issues
```bash
# Try different camera index
python run_liveness_realtime.py --detector cnn --camera 1
```

### Performance Issues
- Gunakan fast detector untuk hardware dengan performa rendah
- Kurangi resolusi camera jika diperlukan
- Pastikan environment sudah aktif dan dependencies terinstall

## Files Structure
```
livenessv2/
├── models/
│   ├── best_model.h5          # Model terbaik dari training
│   └── final_model.h5         # Model final
├── liveness_detector.py       # CNN-based detector
├── fast_liveness_detector.py  # Fast multi-feature detector  
├── advanced_liveness_detector.py # Advanced multi-branch detector
├── run_liveness_realtime.py   # Main runner script
└── train_liveness_model.py    # Training script
```

## Next Steps
- Eksperimen dengan threshold confidence yang berbeda
- Test dengan berbagai kondisi lighting
- Evaluasi performa dengan berbagai jenis attack (print, screen, video)
- Fine-tune model jika diperlukan dengan data tambahan
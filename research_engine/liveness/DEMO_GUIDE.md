# 🎯 Advanced Liveness Detection Demo Guide

Panduan lengkap untuk menggunakan sistem deteksi liveness canggih yang telah dibuat.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd research_engine/liveness/
pip install -r requirements.txt
```

### 2. Run Comprehensive Demo (Recommended)
```bash
python comprehensive_liveness_demo.py --camera 0
```

## 📱 Available Demos

### 1. 🔬 Advanced Liveness Detector
**File**: `advanced_liveness_detector.py`
**Description**: Multi-method computer vision analysis

```bash
# Basic usage
python advanced_liveness_detector.py

# With specific camera
python advanced_liveness_detector.py --camera 0

# Debug mode
python advanced_liveness_detector.py --debug
```

**Features**:
- 8 detection methods (texture, frequency, depth, color, reflection, print detection, temporal, shadow)
- Real-time analysis tanpa perlu gerakan user
- Comprehensive quality assessment
- Detailed method-by-method scoring

### 2. 🧠 Deep Learning Detector  
**File**: `deep_liveness_detector.py`
**Description**: Feature-based neural network models

```bash
# Basic usage
python deep_liveness_detector.py

# Save results
python deep_liveness_detector.py --save-results

# Debug mode  
python deep_liveness_detector.py --debug
```

**Features**:
- 5 feature extractors (texture, color, depth, frequency, motion)
- Ensemble classification
- Risk factor analysis
- Automatic model initialization

### 3. 🏆 State-of-the-Art Anti-Spoofing
**File**: `sota_antispoof_detector.py`  
**Description**: Latest neural network architectures

```bash
# Basic usage
python sota_antispoof_detector.py

# Use GPU (if available)
python sota_antispoof_detector.py --use-gpu

# Save results
python sota_antispoof_detector.py --save-results --model-dir ./models
```

**Features**:
- CDCN, FAS, SiW models (dengan fallback jika tidak tersedia)
- Automatic model downloading
- Attack type classification
- Advanced ensemble prediction

### 4. 🎯 Comprehensive Demo (BEST)
**File**: `comprehensive_liveness_demo.py`
**Description**: Combines all methods for maximum accuracy

```bash
# Use all available models
python comprehensive_liveness_demo.py

# Use specific models only
python comprehensive_liveness_demo.py --models advanced deep

# Full configuration
python comprehensive_liveness_demo.py --camera 0 --save-results --debug --fps-limit 25
```

**Features**:
- Ensemble dari semua model yang tersedia
- Weighted voting system
- Agreement scoring antar model
- Comprehensive visualization

## 🎮 Controls & Usage

### Keyboard Controls (All Demos):
- **'q'** - Quit application
- **'s'** - Save current frame dan results (jika --save-results enabled)
- **'r'** - Reset statistics  
- **'h'** - Show help information

### Command Line Options:
```bash
# Camera selection
--camera 0          # Use camera index 0 (default)
--camera 1          # Use camera index 1

# Performance tuning
--fps-limit 25      # Limit FPS to 25
--use-gpu          # Use GPU if available (SOTA model)

# Debugging & saving
--debug            # Enable debug logging
--save-results     # Enable saving frames/results
--model-dir ./models  # Specify model directory

# Model selection (comprehensive demo only)
--models advanced deep    # Use specific models only
--models sota            # Use only SOTA model
```

## 🧪 Testing Scenarios

### ✅ Live Face Testing:
1. **Setup**: Posisi wajah 30-60cm dari camera
2. **Lighting**: Pastikan pencahayaan merata dari depan
3. **Expected**: Confidence > 0.6, Risk = LOW/MEDIUM, Green bounding box

### ❌ Spoof Attack Testing:

#### 1. Photo Attack:
- Tunjukkan foto wajah (printed atau digital)
- **Expected**: Confidence < 0.5, "Print Attack" detected

#### 2. Screen Attack:
- Tampilkan foto/video di ponsel/tablet
- **Expected**: "Screen Attack" detected, moire patterns identified

#### 3. Video Replay:
- Putar video wajah di laptop/TV
- **Expected**: "Replay Attack" detected, temporal inconsistency

#### 4. Quality Tests:
- Test dengan lighting buruk, blur, atau jarak terlalu jauh
- **Expected**: Low quality score, confidence penalty

## 📊 Understanding Results

### 🎯 Confidence Levels:
- **0.8 - 1.0**: 🟢 Very High (sangat yakin live)
- **0.6 - 0.8**: 🟢 High (yakin live)
- **0.4 - 0.6**: 🟡 Medium (tidak yakin)
- **0.2 - 0.4**: 🔴 Low (kemungkinan fake)
- **0.0 - 0.2**: 🔴 Very Low (sangat yakin fake)

### 🚨 Risk Assessment:
- **LOW (Green)**: Aman, confidence tinggi
- **MEDIUM (Yellow)**: Moderate risk, perlu verifikasi tambahan
- **HIGH (Red)**: Bahaya tinggi, kemungkinan besar spoof

### 📈 Performance Metrics:

#### Advanced Detector:
- **Processing Time**: ~50-100ms per frame
- **Methods**: 8 detection algorithms
- **Best For**: Comprehensive analysis, detailed insights

#### Deep Learning Detector:  
- **Processing Time**: ~30-80ms per frame
- **Features**: 5 feature extractors + ensemble
- **Best For**: Feature-based classification, risk analysis

#### SOTA Anti-Spoofing:
- **Processing Time**: ~100-200ms per frame (depends on models available)
- **Models**: Up to 5 neural networks
- **Best For**: State-of-the-art accuracy, attack classification

#### Comprehensive Demo:
- **Processing Time**: ~150-300ms per frame (combines all)
- **Accuracy**: Highest (ensemble dari semua model)
- **Best For**: Production use, maximum security

## 🔧 Performance Tuning

### 🚀 For Better Speed:
```bash
# Use single model
python advanced_liveness_detector.py --camera 0

# Lower FPS
python comprehensive_liveness_demo.py --fps-limit 15

# Specific models only
python comprehensive_liveness_demo.py --models advanced
```

### 🎯 For Better Accuracy:
```bash
# Use all models
python comprehensive_liveness_demo.py

# Enable GPU
python sota_antispoof_detector.py --use-gpu

# Save results for analysis
python comprehensive_liveness_demo.py --save-results --debug
```

### 🔍 For Debugging:
```bash
# Debug mode
python comprehensive_liveness_demo.py --debug

# Check specific model
python advanced_liveness_detector.py --debug

# Save all results
python comprehensive_liveness_demo.py --save-results --debug
```

## 📁 Output Files

### Saved Files (when using --save-results):
- **Frame Files**: `*_frame_*.jpg` - Screenshot dengan annotations
- **Result Files**: `*_results_*.json` - Detailed detection results

### Example Result JSON:
```json
{
  "timestamp": 1701234567,
  "ensemble_result": {
    "is_live": true,
    "confidence": 0.87,
    "risk_level": "LOW",
    "model_votes": {
      "advanced": {"confidence": 0.85, "live_ratio": 1.0},
      "deep": {"confidence": 0.89, "live_ratio": 1.0}
    }
  }
}
```

## 🚨 Troubleshooting

### Problem: Camera tidak terdeteksi
```bash
# Test berbagai camera index
python comprehensive_liveness_demo.py --camera 1
python comprehensive_liveness_demo.py --camera 2

# Check available cameras
ls /dev/video*  # Linux
# Atau gunakan aplikasi camera lain untuk test
```

### Problem: FPS rendah / lag
```bash  
# Gunakan single model
python advanced_liveness_detector.py

# Turunkan FPS limit
python comprehensive_liveness_demo.py --fps-limit 10

# Check CPU usage
top | grep python
```

### Problem: Import errors
```bash
# Install core dependencies
pip install opencv-python numpy scipy

# Install optional dependencies
pip install torch torchvision  # For SOTA models
pip install onnxruntime       # For ONNX models

# Check Python version
python --version  # Harus 3.7+
```

### Problem: Poor accuracy
1. **Lighting**: Pastikan pencahayaan cukup dan merata
2. **Distance**: Jaga jarak 30-60cm dari camera  
3. **Stability**: Hindari gerakan camera berlebihan
4. **Resolution**: Pastikan camera resolusi cukup (min 640x480)
5. **Models**: Gunakan comprehensive demo untuk akurasi terbaik

## 💡 Pro Tips

### 🎯 Best Practices:
1. **Testing Environment**: 
   - Lighting merata dari depan
   - Background kontras dengan wajah
   - Jarak optimal 30-60cm

2. **Attack Testing**:
   - Test dengan berbagai jenis foto (printed, digital)
   - Coba berbagai ukuran layar (ponsel, tablet, laptop)
   - Test dengan video berkualitas tinggi
   - Gunakan foto dengan resolusi berbeda

3. **Performance Monitoring**:
   - Monitor FPS dan processing time
   - Check confidence trends
   - Analisis agreement score antar model
   - Save results untuk evaluasi offline

### 🔍 Analysis Tips:
1. **Model Comparison**: Jalankan berbagai model dan bandingkan hasil
2. **Parameter Tuning**: Experiment dengan threshold values
3. **Temporal Analysis**: Perhatikan consistency across frames
4. **Quality Metrics**: Monitor frame quality scores

### 🚀 Production Ready:
1. **Model Selection**: Pilih model sesuai trade-off speed vs accuracy
2. **Threshold Tuning**: Adjust berdasarkan security requirements
3. **Error Handling**: Implement robust error handling
4. **Monitoring**: Add logging dan performance monitoring
5. **Updates**: Regular update model dan algorithm

---

## 🎉 Ready to Test!

Mulai dengan comprehensive demo untuk experience terbaik:

```bash
python comprehensive_liveness_demo.py --camera 0 --save-results
```

Kemudian coba test dengan berbagai attack scenarios untuk melihat performa sistem!

**Good luck testing! 🚀**
# Tuning Guide untuk Balanced Liveness Detection

## Masalah yang Ditemukan

1. **Terlalu Lenient** (`passive_liveness_advanced.py`):
   - Foto/video dari HP bisa lolos
   - Threshold terlalu rendah

2. **Terlalu Strict** (`passive_liveness_strict.py`):
   - Wajah asli di-reject
   - Pixel grid detection terlalu sensitif
   - Screen detection over-aggressive

## Solusi: Balanced Version

Kita perlu:
1. **Screen detection yang smart**: Deteksi layar HP tapi tidak salah deteksi webcam
2. **Multi-stage verification**: Kombinasi beberapa metode, bukan hanya satu
3. **Adaptive thresholds**: Threshold yang menyesuaikan kondisi
4. **Temporal consistency**: Butuh beberapa frame konsisten

## Parameter yang Perlu Di-tune

### 1. Pixel Grid Detection
```python
# STRICT (over-sensitive):
if peak_count > 50: return 0.1  # Terlalu mudah detect
elif peak_count > 20: return 0.4

# BALANCED (better):
if peak_count > 100: return 0.2  # Lebih tinggi threshold
elif peak_count > 60: return 0.5
else: return 0.9  # Give benefit of doubt
```

### 2. Rectangular Boundary
```python
# STRICT:
if len(horizontal) >= 2 and len(vertical) >= 2: return 0.1

# BALANCED:
# Harus punya edges yang SANGAT jelas dan PANJANG
if len(horizontal) >= 3 and len(vertical) >= 3:
    # Check if they form rectangle
    if edges_form_rectangle(): return 0.1
else: return 1.0
```

### 3. Final Decision
```python
# STRICT:
min_score = min(scores.values())
if min_score < 0.35: reject()

# BALANCED:
# Butuh MULTIPLE low scores untuk reject
low_scores = [s for s in scores.values() if s < 0.4]
if len(low_scores) >= 3: reject()
```

## Rekomendasi Implementation

Gunakan **hybrid approach**:
- Jika YOLO deteksi HP dengan confidence >0.7: **REJECT IMMEDIATELY**
- Jika screen detection sangat jelas (score <0.2): **REJECT**  
- Jika tidak ada kedipan setelah 5 detik: **REJECT**
- Sisanya: gunakan **weighted average** dengan threshold 0.55

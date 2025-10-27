# WebSocket Detection Improvements - Perbaikan Blink, Motion & Obstacle Detection

## Masalah yang Diperbaiki

### 1. ❌ Blink Detection Tidak Terdeteksi
**Penyebab:**
- Session state tidak menyimpan semua variabel penting (`blink_counter`, `valid_blinks`, `last_blink_time`)
- Tidak ada logging untuk debugging
- Threshold mungkin terlalu ketat

**Solusi:**
- ✅ Tambah semua variabel state ke session manager
- ✅ Tambah debug logging setiap 30 frames
- ✅ Tambah logging ketika blink terdeteksi
- ✅ Tambah logging ketika blink ditolak (duration out of range)

### 2. ❌ Motion Detection Tidak Terdeteksi
**Penyebab:**
- `last_bbox_center` dan `last_bbox_size` tidak di-restore dari cache
- Motion tracking tidak menyimpan state dengan benar

**Solusi:**
- ✅ Simpan `last_bbox_center`, `last_bbox_size`, `last_motion_time` ke cache
- ✅ Restore `last_bbox_center` sebagai numpy array dengan benar
- ✅ Tambah debug logging untuk motion detection
- ✅ Tambah logging ketika motion terdeteksi

### 3. ❌ Obstacle Detection Tidak Terdeteksi
**Penyebab:**
- Tidak ada logging untuk debugging
- Threshold mungkin terlalu tinggi

**Solusi:**
- ✅ Tambah logging untuk semua obstacle detection
- ✅ Tambah warning untuk blocking obstacles (mask, hand_covering)
- ✅ Tambah fallback logging untuk traditional detection

### 4. ⚠️ Polygon Data Kurang Lengkap
**Penyebab:**
- Hanya mengirim eye regions
- Tidak ada face mesh landmarks lainnya

**Solusi:**
- ✅ Tambah face oval (36 points)
- ✅ Tambah lips outer (21 points)
- ✅ Tambah left eyebrow (10 points)
- ✅ Tambah right eyebrow (10 points)
- ✅ Tambah nose bridge (10 points)
- ✅ Total: ~100+ MediaPipe landmarks

## Perubahan File

### 1. `core/session_manager.py`

#### Tambahan State Variables:
```python
# Di store_liveness_detector():
'valid_blinks': int(getattr(liveness_detector, 'valid_blinks', 0)),
'blink_counter': int(getattr(liveness_detector, 'blink_counter', 0)),
'motion_score': float(getattr(liveness_detector, 'motion_score', 0.0)),
'last_bbox_center': json_serializable(getattr(liveness_detector, 'last_bbox_center', None)),
'last_bbox_size': float(getattr(liveness_detector, 'last_bbox_size', 0.0)),
'last_motion_time': float(getattr(liveness_detector, 'last_motion_time', 0.0)),
'blink_start_frame': int(liveness_detector.blink_start_frame) if liveness_detector.blink_start_frame else None,
'last_blink_time': float(getattr(liveness_detector, 'last_blink_time', 0.0)),
'eye_visibility_score': float(getattr(liveness_detector, 'eye_visibility_score', 0.0)),
'blink_quality_scores': json_serializable(getattr(liveness_detector, 'blink_quality_scores', [])),
```

#### Restore dengan Numpy Array:
```python
# Di get_liveness_detector():
last_bbox_center = state.get('last_bbox_center')
if last_bbox_center:
    detector.last_bbox_center = np.array(last_bbox_center, dtype=np.float32)
else:
    detector.last_bbox_center = None
```

### 2. `core/face_recognition_engine.py`

#### Enhanced Blink Detection Logging:
```python
# Log setiap 30 frames
if self.frame_counter % 30 == 0:
    logger.info(f"Frame {self.frame_counter}: EAR={avg_ear:.3f}, Threshold={adaptive_threshold:.3f}, Blinks={self.total_blinks}, Motion={self.motion_events}")

# Log ketika blink dimulai
if avg_ear < adaptive_threshold:
    if self.blink_start_frame is None:
        logger.debug(f"Blink started at frame {self.frame_counter}, EAR={avg_ear:.3f}")

# Log ketika blink terdeteksi
logger.info(f"✓ BLINK DETECTED! Total: {self.total_blinks}, Duration: {blink_duration}, EAR: {avg_ear:.3f}")

# Log ketika blink ditolak
logger.debug(f"Blink rejected: duration={blink_duration} out of range [{self.MIN_BLINK_DURATION}, {self.MAX_BLINK_DURATION}]")
```

#### Enhanced Motion Detection Logging:
```python
# Log initialization
logger.debug(f"Motion tracking initialized: center={current_center}, size={current_size:.1f}")

# Log ketika motion terdeteksi
logger.info(f"✓ MOTION DETECTED! Events: {self.motion_events}, Shift: {normalized_shift:.3f}, Score: {self.motion_score:.3f}")

# Log setiap 30 frames
if self.frame_counter % 30 == 0:
    logger.debug(f"Motion: shift={normalized_shift:.4f}, sensitivity={self.MOTION_SENSITIVITY}, events={self.motion_events}")
```

#### Enhanced Obstacle Detection Logging:
```python
# Log ketika obstacles terdeteksi
logger.debug(f"Glasses detected: confidence={glasses_conf:.2f}")
logger.info(f"⚠️ MASK DETECTED: confidence={mask_conf:.2f}")
logger.info(f"⚠️ HAND COVERING DETECTED: confidence={hand_conf:.2f}")

# Log summary
if obstacles:
    logger.info(f"Obstacles detected: {obstacles}")
```

### 3. `auth_service/consumers.py`

#### Extended Visual Data:
```python
visual_data["face_mesh_landmarks"] = {
    "face_oval": face_oval_points,        # 36 points
    "lips": lips_points,                  # 21 points
    "left_eyebrow": left_eyebrow_points,  # 10 points
    "right_eyebrow": right_eyebrow_points,# 10 points
    "nose": nose_points                   # 10 points
}
```

#### Enhanced Liveness Indicators:
```python
"liveness_indicators": {
    "blinks": liveness_result.get("blinks_detected", 0),
    "ear": liveness_result.get("ear", 0.0),
    "blink_verified": liveness_result.get("blink_detected", False),
    "motion_verified": liveness_result.get("motion_verified", False),
    "motion_score": liveness_result.get("motion_score", 0.0),
    "motion_events": liveness_result.get("motion_events", 0),
    "frame_counter": liveness_detector.frame_counter
}
```

### 4. `test_websocket_auth.py`

#### Enhanced Visual Overlays:
- **Face Bounding Box**: Green rectangle dengan corner markers
- **Face Oval**: Cyan thin line (36 MediaPipe landmarks)
- **Eyebrows**: Green lines (left & right, 10 points each)
- **Nose**: White line (10 points)
- **Lips**: Red outline (21 points)
- **Eye Regions**: Yellow dengan semi-transparent fill (16 points each)

#### Enhanced Status Panel:
```python
# Semi-transparent background panel
[OK]/[--]/[XX] Frame counter
[OK]/[--] Blinks: X
EAR: 0.XXX with progress bar
[OK]/[??] Motion: OK (events) / Move head
[OK]/[!!] Quality: 0.XX with progress bar
[OK]/[XX] Liveness: VERIFIED / NOT OK
OBSTACLES DETECTED: (if any)
  - mask
  - glasses
```

## Testing Guide

### 1. Restart Server
```bash
cd face_recognition_app
daphne -b 0.0.0.0 -p 8000 face_app.asgi:application
```

### 2. Run Test dengan Logging
```bash
cd ..
python test_websocket_auth.py \
  YOUR_API_KEY \
  YOUR_SECRET_KEY \
  http://127.0.0.1:8000 \
  enrollment \
  testuser123
```

### 3. Check Server Logs

Anda seharusnya melihat log seperti ini:

**Blink Detection:**
```
INFO - Frame 30: EAR=0.245, Threshold=0.123, Blinks=0, Motion=0
DEBUG - Blink started at frame 45, EAR=0.089
INFO - ✓ BLINK DETECTED! Total: 1, Duration: 3, EAR: 0.089
```

**Motion Detection:**
```
DEBUG - Motion tracking initialized: center=[320.5 240.0], size=285.3
INFO - ✓ MOTION DETECTED! Events: 1, Shift: 0.145, Score: 0.145
DEBUG - Motion: shift=0.0234, sensitivity=0.12, events=1
```

**Obstacle Detection:**
```
DEBUG - Glasses detected: confidence=0.65
INFO - ⚠️ MASK DETECTED: confidence=0.78
INFO - Obstacles detected: ['glasses', 'mask']
```

### 4. Expected Camera Display

Camera window akan menampilkan:

1. **Face Mesh Visualization:**
   - Cyan oval mengelilingi wajah
   - Green eyebrows (left & right)
   - White nose bridge
   - Red lips outline
   - Yellow eye regions dengan semi-transparent fill

2. **Status Panel (top-left):**
   ```
   Frame: 45
   [OK] Blinks: 1
   EAR: 0.245 ████████░░
   [OK] Motion: OK (2)
   [OK] Quality: 0.82 ████████░░
   [OK] Liveness: VERIFIED
   ```

3. **Progress Messages (bottom):**
   ```
   Capturing frames (2/3)
   Improve lighting or position
   ```

### 5. Troubleshooting

#### Blink Tidak Terdeteksi:
- ✅ Check server logs: `Frame X: EAR=...`
- ✅ Pastikan EAR value berubah ketika Anda blink (dari ~0.25 ke <0.15)
- ✅ Pastikan yellow eye polygons terlihat di camera
- ✅ Coba blink lebih lambat dan jelas

#### Motion Tidak Terdeteksi:
- ✅ Check server logs: `Motion: shift=...`
- ✅ Pastikan green face bounding box terlihat
- ✅ Gerakkan kepala lebih jelas (kiri-kanan atau atas-bawah)
- ✅ Motion sensitivity: 0.12 (normalized shift)

#### Obstacle Tidak Terdeteksi:
- ✅ Check server logs: `Obstacles detected: ...`
- ✅ Coba pakai kacamata (glasses)
- ✅ Coba tutup mulut dengan tangan (hand_covering)
- ✅ Coba pakai masker (mask) - akan block enrollment!

## Threshold Settings

Current settings (dapat diubah di `settings.py`):

```python
FACE_RECOGNITION_CONFIG = {
    # Blink Detection
    'EAR_THRESHOLD': 0.18,          # Base threshold
    'ADAPTIVE_FACTOR': 0.5,          # Adaptive threshold = baseline * 0.5
    'CONSECUTIVE_FRAMES': 2,         # Minimum frames for blink
    'MIN_BLINK_DURATION': 1,        # Minimum 1 frame
    'MAX_BLINK_DURATION': 10,       # Maximum 10 frames
    
    # Motion Detection
    'MOTION_SENSITIVITY': 0.12,      # Normalized shift required
    'MOTION_EVENT_INTERVAL': 0.35,  # Seconds between events
    
    # Quality
    'CAPTURE_QUALITY_THRESHOLD': 0.65,
    'AUTH_QUALITY_THRESHOLD': 0.4,
    
    # Liveness Requirements
    'LIVENESS_THRESHOLD': 1,         # Minimum blinks
    'LIVENESS_MOTION_EVENTS': 1,    # OR minimum motion events
}
```

## Summary

✅ **Blink Detection**: Sekarang dengan full state persistence & detailed logging
✅ **Motion Detection**: Bbox tracking dengan numpy array restoration & logging
✅ **Obstacle Detection**: Comprehensive logging untuk all detectors
✅ **Visual Feedback**: 100+ MediaPipe landmarks dengan semi-transparent overlays
✅ **Debug Capability**: Extensive logging untuk troubleshooting

**Testing**: Restart server, run test script, check logs untuk detailed diagnostics!

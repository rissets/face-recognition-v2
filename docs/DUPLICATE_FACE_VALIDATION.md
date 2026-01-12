# Validasi Duplicate Face pada Enrollment

## 📋 Ringkasan

Sistem sekarang memiliki validasi untuk mencegah wajah yang sama didaftarkan (enrolled) pada lebih dari satu user. Ketika seseorang mencoba melakukan enrollment dengan wajah yang sudah terdaftar pada user lain, sistem akan menolak enrollment tersebut dan memberikan pesan error yang jelas.

## 🔍 Cara Kerja

### 1. **Pengecekan Similarity**
Saat proses enrollment akan diselesaikan, sistem melakukan:
- Menghitung embedding wajah dari frame yang dikumpulkan
- Mencari similarity dengan semua embedding yang sudah ada di database (ChromaDB)
- Menggunakan threshold **60% similarity** sebagai batas deteksi duplicate
- Membandingkan dengan semua user yang terdaftar

### 2. **Validasi User ID**
- Sistem mengecek apakah embedding yang mirip berasal dari user yang berbeda
- Format User ID: `{client_id}:{external_user_id}`
- Jika similarity >= 60% dengan user lain → **DITOLAK**
- Jika similarity dengan user yang sama → **DIIZINKAN** (re-enrollment)

### 3. **Response Error**
Ketika duplicate terdeteksi, client menerima:

#### WebSocket Response:
```json
{
  "type": "enrollment_failed",
  "success": false,
  "error": "duplicate_face",
  "error_code": "DUPLICATE_FACE_DETECTED",
  "message": "❌ This face has already been enrolled by another user",
  "details": {
    "reason": "Face already exists in system",
    "similarity_score": 0.85,
    "conflicting_user": "client123:user456"
  }
}
```

#### HTTP REST API Response (409 Conflict):
```json
{
  "success": false,
  "error": "duplicate_face",
  "error_code": "DUPLICATE_FACE_DETECTED",
  "message": "❌ This face has already been enrolled by another user",
  "details": {
    "conflicting_user_id": "client123:user456",
    "similarity_score": 0.85,
    "frames_used": 5,
    "quality_score": 0.92
  },
  "session_status": "failed"
}
```

## 📝 File yang Dimodifikasi

### 1. `face_recognition_engine.py`
**Fungsi Baru:**
```python
def check_face_duplicate(self, embedding, current_user_id, similarity_threshold=0.6)
```
- Mencari embedding yang mirip di database
- Membandingkan dengan user_id yang berbeda
- Return: `is_duplicate`, `conflicting_user_id`, `similarity_score`

**Fungsi Diupdate:**
```python
def complete_enrollment_with_session(self, session_token, user_id)
```
- Menambahkan validasi duplicate sebelum menyimpan embedding
- Menolak enrollment jika duplicate terdeteksi

### 2. `consumers.py` (WebSocket)
**Fungsi Diupdate:**
```python
def _sync_complete_enrollment(self, enrollment, face_data, liveness_data)
```
- Menambahkan validasi duplicate sebelum menyimpan
- Update status enrollment menjadi "failed" dengan reason "duplicate_face"
- Mengirim error message ke client via WebSocket

**Penanganan Error:**
- Deteksi failure reason dari metadata enrollment
- Kirim custom error message dengan close code 4001
- Menyertakan informasi conflicting user dan similarity score

### 3. `views.py` (HTTP REST API)
**Endpoint Diupdate:**
```python
POST /api/auth-service/enrollment/process-frame/
```
- Menangkap error duplicate_face dari `complete_enrollment_with_session()`
- Update session status menjadi "failed"
- Return HTTP 409 Conflict dengan detail error

## 🎯 Threshold Similarity

| Similarity Score | Interpretasi | Action |
|-----------------|--------------|---------|
| >= 0.6 (60%)    | Same person  | ❌ DITOLAK (duplicate) |
| < 0.6 (60%)     | Different person | ✅ DIIZINKAN |

**Catatan:** Threshold 60% dipilih untuk:
- Menghindari false positive (menolak orang yang sebenarnya berbeda)
- Cukup ketat untuk mendeteksi wajah yang sama
- Berdasarkan karakteristik InsightFace embeddings

## 🔄 Alur Enrollment dengan Validasi

```
1. User mulai enrollment
   ↓
2. Sistem collect frame + liveness check
   ↓
3. Cukup frame terkumpul (min 3-5 frames)
   ↓
4. Calculate averaged embedding
   ↓
5. ✅ CHECK DUPLICATE FACE ← VALIDASI BARU
   ├─ Cari similarity dengan existing embeddings
   ├─ Filter hasil dengan threshold 60%
   └─ Bandingkan user_id
   ↓
6a. ❌ Duplicate Detected
    ├─ Status: failed
    ├─ Reason: duplicate_face
    ├─ Log: conflicting_user_id
    └─ Response: Error 409
   ↓
6b. ✅ No Duplicate
    ├─ Save embedding ke ChromaDB
    ├─ Update ClientUser.is_enrolled = True
    └─ Response: Success
```

## 🧪 Testing

### Test Case 1: Enrollment Normal
```bash
# User A melakukan enrollment pertama kali
POST /api/auth-service/enrollment/process-frame/
# Expected: ✅ SUCCESS
```

### Test Case 2: Duplicate Detection
```bash
# User A sudah enrolled
# User B mencoba enroll dengan wajah yang sama
POST /api/auth-service/enrollment/process-frame/
# Expected: ❌ REJECTED - duplicate_face
```

### Test Case 3: Re-enrollment Same User
```bash
# User A sudah enrolled
# User A mencoba enroll lagi (update)
POST /api/auth-service/enrollment/process-frame/
# Expected: ✅ SUCCESS (same user, allowed)
```

## 📊 Logging

Sistem mencatat event berikut:

### Success:
```
✅ Face validation passed - No duplicate found for user client123:user456
✅ Enrollment completed for user client123:user456 - Embedding ID: abc123
```

### Duplicate Detected:
```
⚠️ Duplicate face detected! Face already enrolled for user 'client123:user789' 
   with similarity 0.850
❌ ENROLLMENT REJECTED: Face already enrolled for user 'client123:user789' 
   (similarity: 85.0%). Current user: client123:user456
```

## 🛡️ Security Considerations

1. **Privacy**: Conflicting user_id dikirim ke client - pertimbangkan untuk hash/obfuscate di production
2. **False Positives**: Threshold 60% dapat disesuaikan jika terlalu ketat/longgar
3. **Performance**: Search similarity dilakukan setiap enrollment completion
4. **Fail-Open**: Jika terjadi error saat check duplicate, enrollment tetap diizinkan untuk menghindari blocking legitimate users

## 🔧 Konfigurasi

Untuk mengubah threshold similarity:
```python
# Di face_recognition_engine.py
duplicate_check = self.check_face_duplicate(
    embedding=embedding,
    current_user_id=user_id,
    similarity_threshold=0.7  # Ubah dari 0.6 ke 0.7 untuk lebih ketat
)
```

## 📞 Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `DUPLICATE_FACE_DETECTED` | Wajah sudah terdaftar di user lain | 409 Conflict |

## ✅ Checklist Implementation

- [x] Fungsi `check_face_duplicate()` di `face_recognition_engine.py`
- [x] Update `_sync_complete_enrollment()` di `consumers.py`
- [x] Update `complete_enrollment_with_session()` di `face_recognition_engine.py`
- [x] Error handling di WebSocket consumer
- [x] Error handling di HTTP REST API
- [x] Logging untuk tracking duplicate detection
- [x] Clear error messages untuk client
- [x] Metadata untuk audit trail

## 🚀 Deployment Notes

1. **Database Migration**: Tidak perlu migration, menggunakan existing tables
2. **Backward Compatibility**: Full compatible dengan existing enrollments
3. **Testing Required**: Test dengan real faces sebelum production
4. **Monitoring**: Monitor log untuk frequency of duplicate detections

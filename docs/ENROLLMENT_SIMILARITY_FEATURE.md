# Enrollment with Old Photo Similarity Feature

## Overview
Fitur ini menambahkan kemampuan untuk membandingkan hasil enrollment dengan foto profil lama yang sudah ada, serta menyimpan foto terbaik dari proses enrollment.

## Perubahan pada Model ClientUser

### Field Baru
1. **`old_profile_photo`** (ImageField)
   - Path: `client_users/old_profiles/%Y/%m/%d/`
   - Menyimpan foto profil lama untuk perbandingan
   - Optional (null=True, blank=True)

2. **`similarity_with_old_photo`** (FloatField)
   - Range: 0.0 - 1.0
   - Menyimpan skor similaritas antara foto enrollment dengan foto lama
   - Optional (null=True, blank=True)

3. **`profile_image`** (ImageField) - Already exists
   - Path: `client_users/profiles/%Y/%m/%d/`
   - Sekarang di-update dengan frame terbaik dari enrollment

## Alur Proses Enrollment

### 1. Create User dengan Old Profile Photo
```bash
POST /api/clients/users/
Content-Type: multipart/form-data

{
  "external_user_id": "user123",
  "old_profile_photo": <file>,
  "profile": {"name": "John Doe"},
  "face_auth_enabled": true
}
```

### 2. WebSocket Enrollment Process

#### Connection
```javascript
ws://domain/ws/auth/enrollment/<session_token>/
```

#### Proses yang Terjadi
1. **Frame Processing**: Setiap frame yang dikirim akan diproses untuk face detection dan liveness
2. **Best Frame Selection**: Frame terbaik dengan quality score tertinggi akan dipilih
3. **Similarity Calculation**: 
   - Jika user memiliki `old_profile_photo`, sistem akan:
     - Extract embedding dari old photo
     - Calculate cosine similarity dengan embedding enrollment
     - Save similarity score ke `similarity_with_old_photo`
4. **Profile Image Save**: 
   - Frame enrollment akan disimpan ke `profile_image`
   - Format: `enrollment_{external_user_id}_{timestamp}.jpg`

#### Enrollment Complete Response
```json
{
  "type": "enrollment_complete",
  "success": true,
  "enrollment_id": "uuid",
  "frames_processed": 5,
  "liveness_verified": true,
  "blinks_detected": 2,
  "motion_verified": true,
  "quality_score": 0.95,
  "similarity_with_old_photo": 0.87,  // NEW FIELD
  "encrypted_data": {...},
  "visual_data": {...},
  "message": "Enrollment completed successfully"
}
```

### 3. Response Scenarios

#### Scenario A: User dengan Old Photo
```json
{
  "similarity_with_old_photo": 0.87,  // Float value (0.0 - 1.0)
  ...
}
```

#### Scenario B: User tanpa Old Photo
```json
{
  "similarity_with_old_photo": null,  // null
  ...
}
```

#### Scenario C: Error dalam Similarity Calculation
```json
{
  "similarity_with_old_photo": null,  // null (enrollment tetap berhasil)
  ...
}
```
**Note**: Enrollment tidak akan gagal jika similarity calculation error.

## Retrieve User Data

### GET `/api/clients/users/{external_user_id}/`

Response:
```json
{
  "id": "uuid",
  "external_user_id": "user123",
  "profile": {"name": "John Doe"},
  "is_enrolled": true,
  "face_auth_enabled": true,
  "profile_image_url": "http://domain/media/client_users/profiles/2026/01/06/enrollment_user123_20260106_143025.jpg",
  "old_profile_photo_url": "http://domain/media/client_users/old_profiles/2026/01/06/old_photo.jpg",
  "similarity_with_old_photo": 0.87,
  "enrollment_completed_at": "2026-01-06T14:30:25Z",
  ...
}
```

## Technical Implementation

### Code Changes

#### 1. consumers.py - `process_enrollment_frame()`
```python
# Update method signature
async def complete_enrollment(..., frame: np.ndarray = None) -> tuple[bool, Optional[float]]:
    ...
```

#### 2. consumers.py - `_sync_complete_enrollment()`
Proses yang ditambahkan:
1. Save frame as profile_image
2. Load old_profile_photo if exists
3. Detect face in old photo
4. Extract embedding from old photo
5. Calculate cosine similarity
6. Save similarity_with_old_photo
7. Update ClientUser fields

#### 3. Similarity Calculation Formula
```python
similarity_score = np.dot(embedding_new, embedding_old) / 
                  (np.linalg.norm(embedding_new) * np.linalg.norm(embedding_old))
```

### Error Handling

1. **Old Photo tidak ada**: 
   - Similarity tidak dihitung
   - `similarity_with_old_photo = null`
   - Enrollment tetap berhasil

2. **Old Photo tidak valid**:
   - Tidak bisa di-load/corrupted
   - Log warning
   - Enrollment tetap berhasil

3. **Face tidak terdeteksi di old photo**:
   - Log warning "No face detected in old profile photo"
   - Enrollment tetap berhasil

4. **Error saat save profile_image**:
   - Log error
   - Enrollment tetap berhasil
   - profile_image tidak ter-update

## Logging

### Success Logs
```
✅ Saved profile_image for user {external_user_id}
✅ Calculated similarity with old photo: 0.870
✅ Enrollment completed for user {engine_user_id}
✅ Similarity with old photo: 0.870
```

### Warning Logs
```
⚠️ No embedding found in old profile photo
⚠️ No face detected in old profile photo
⚠️ Failed to load old profile photo from {path}
```

### Error Logs
```
❌ Failed to save profile_image: {error}
❌ Error calculating similarity with old photo: {error}
```

## Testing

### Test Files
1. **clients/test_create_client_user.py**
   - Test model fields
   - Test serializers
   - Test CRUD operations

2. **auth_service/test_enrollment_similarity.py**
   - Test enrollment with/without old photo
   - Test similarity calculation
   - Test profile_image saving
   - Test response structure

### Running Tests
```bash
# Test ClientUser model
python manage.py test clients.test_create_client_user

# Test enrollment similarity
python manage.py test auth_service.test_enrollment_similarity

# Run all tests
python manage.py test
```

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/clients/users/` | POST | Create user (dapat include old_profile_photo) |
| `/api/clients/users/{id}/` | GET | Retrieve user dengan similarity_with_old_photo |
| `/api/clients/users/{id}/` | PATCH | Update user (dapat update old_profile_photo) |
| `/ws/auth/enrollment/<token>/` | WebSocket | Enrollment process dengan similarity checking |

## Migration

Migration file: `clients/migrations/0003_clientuser_old_profile_photo_and_more.py`

Fields added:
- `old_profile_photo`
- `similarity_with_old_photo`

```bash
python manage.py migrate clients
```

## Compatibility

- ✅ Backward compatible
- ✅ Existing users tanpa old_profile_photo tetap bisa enrollment
- ✅ Enrollment tanpa similarity checking tetap berfungsi
- ✅ Error dalam similarity calculation tidak mempengaruhi enrollment success

## Performance Considerations

1. **Additional Processing**: 
   - ~50-100ms untuk load dan process old photo
   - ~10-20ms untuk similarity calculation

2. **Storage**: 
   - 2 foto per user (old_profile_photo + profile_image)
   - Average: 200-500KB per foto

3. **Optimization**:
   - Similarity calculation hanya jika old_profile_photo exists
   - Error handling untuk mencegah enrollment failure
   - Async processing untuk tidak blocking

## Security

1. **File Validation**: 
   - ImageField validation dari Django
   - Content-Type checking

2. **Access Control**:
   - Client-specific isolation
   - API Key authentication

3. **Data Encryption**:
   - Embeddings encrypted in database
   - Standard Django media file permissions

## Future Enhancements

1. **Threshold-based Warning**:
   - Alert jika similarity < threshold (e.g., 0.5)
   - Possible identity change detection

2. **Historical Tracking**:
   - Track similarity score history
   - Detect gradual facial changes

3. **Multiple Old Photos**:
   - Compare dengan multiple historical photos
   - Average similarity calculation

4. **Automatic Re-enrollment Suggestion**:
   - Suggest re-enrollment jika similarity terlalu rendah
   - Face change detection over time

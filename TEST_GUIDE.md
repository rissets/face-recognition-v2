# Enrollment with Old Photo Similarity - Test Guide

## Quick Start

### 1. Create Sample Old Photo
```bash
python3 create_sample_photo.py old_photo.jpg "TEST USER"
```

### 2. Run Enrollment Test
```bash
# Method 1: Using helper script
./test_enrollment_with_similarity.sh user123 old_photo.jpg

# Method 2: Direct Python script
python3 test_websocket_auth.py \
    YOUR_API_KEY \
    YOUR_SECRET_KEY \
    http://localhost:8000 \
    enrollment \
    user123 \
    old_photo.jpg
```

## What Happens

### Step 1: User Creation
- Creates user via POST `/api/clients/users/`
- Uploads `old_photo.jpg` as `old_profile_photo`
- Sets user metadata (name, email, phone)

### Step 2: WebSocket Enrollment
- Opens video stream from camera
- Processes frames for face detection
- Performs liveness checks
- Completes enrollment when successful

### Step 3: Similarity Calculation
- Extracts face embedding from enrollment frame
- Loads `old_profile_photo` from user record
- Extracts face embedding from old photo
- Calculates cosine similarity (0.0 - 1.0)
- Saves similarity score to database

### Step 4: Results Display
The WebSocket response includes:
```json
{
    "type": "enrollment_complete",
    "success": true,
    "user_id": "user123",
    "similarity_with_old_photo": 0.8745,
    "message": "Enrollment successful"
}
```

In the console, you'll see:
```
📸 SIMILARITY WITH OLD PHOTO:
   Similarity Score: 87.45%
   Status: ✅ Very High Similarity (Same Person)
```

In the video overlay:
```
╔════════════════════════════════════╗
║ Similarity: 87.45%                 ║
║ [■■■■■■■■■■■■■■■■■■□□]             ║
╚════════════════════════════════════╝
```

## Similarity Interpretation

| Score Range | Status | Icon | Meaning |
|------------|--------|------|---------|
| 85% - 100% | Very High | ✅ | Same person, high confidence |
| 70% - 84% | High | ✓ | Likely same person |
| 50% - 69% | Moderate | ⚠️ | Possible match, needs review |
| 0% - 49% | Low | ❌ | Different person or poor quality |

## Color Coding

- **Green Bar** (≥70%): Good match, enrollment accepted
- **Yellow Bar** (50-69%): Moderate match, may need review
- **Red Bar** (<50%): Poor match, investigate

## Testing Scenarios

### Scenario 1: Same Person (Expected High Similarity)
```bash
# Use user's actual old photo
./test_enrollment_with_similarity.sh user123 /path/to/real_old_photo.jpg
# Expected: 70-95% similarity
```

### Scenario 2: Different Person (Expected Low Similarity)
```bash
# Use someone else's photo
./test_enrollment_with_similarity.sh user123 /path/to/different_person.jpg
# Expected: 20-60% similarity
```

### Scenario 3: No Old Photo (Similarity N/A)
```bash
# Create user without old photo first
python3 test_websocket_auth.py API_KEY SECRET_KEY http://localhost:8000 enrollment user456
# Expected: No similarity calculation
```

### Scenario 4: Poor Quality Old Photo
```bash
# Use low resolution or blurry photo
./test_enrollment_with_similarity.sh user789 blurry_photo.jpg
# Expected: May fail with "No face detected in old photo"
```

## Troubleshooting

### "No face detected in old photo"
- **Cause**: Old photo doesn't contain a clear face
- **Solution**: Use a photo with a clear, frontal face
- **Check**: Run `python3 -c "import cv2; from face_recognition_app.recognition.engine import FaceRecognitionEngine; engine = FaceRecognitionEngine(); face = engine.detect_face(cv2.imread('old_photo.jpg')); print('Face detected!' if face is not None else 'No face')"`

### "Failed to calculate similarity"
- **Cause**: Error during embedding extraction or comparison
- **Solution**: Check logs for detailed error
- **Note**: Enrollment still succeeds, similarity is just not calculated

### Similarity is unexpectedly low
- **Possible reasons**:
  1. Different lighting conditions
  2. Different facial angles
  3. Significant time gap (aging, weight change)
  4. Facial hair changes
  5. Glasses/accessories
- **Solution**: This is expected behavior - similarity reflects actual facial differences

### User already exists
- **Error**: `User with this user_id already exists`
- **Solution**: Use a different user_id or delete existing user

## Environment Variables

Set these before running tests:

```bash
export API_KEY="your_api_key_here"
export SECRET_KEY="your_secret_key_here"
export BASE_URL="http://localhost:8000"
export CAMERA_INDEX="0"

# Then run
./test_enrollment_with_similarity.sh test_user
```

## Sample Photo Generator

The `create_sample_photo.py` script creates a test image with:
- Gradient background
- Text overlay
- Simple face-like drawing
- Timestamp

Usage:
```bash
# Default (old_photo_sample.jpg)
python3 create_sample_photo.py

# Custom filename
python3 create_sample_photo.py my_old_photo.jpg

# Custom text
python3 create_sample_photo.py my_old_photo.jpg "JOHN DOE 2020"
```

## API Endpoint Details

### POST /api/clients/users/
Creates a new client user with optional old profile photo.

**Request:**
```bash
curl -X POST http://localhost:8000/api/clients/users/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "user_id=user123" \
  -F "name=Test User" \
  -F "email=test@example.com" \
  -F "phone=+1234567890" \
  -F "old_profile_photo=@old_photo.jpg"
```

**Response:**
```json
{
    "id": 1,
    "user_id": "user123",
    "name": "Test User",
    "email": "test@example.com",
    "phone": "+1234567890",
    "old_profile_photo_url": "http://localhost:8000/media/client_users/old_profiles/2024/01/15/old_photo.jpg",
    "similarity_with_old_photo": null,
    "is_enrolled": false,
    "is_active": true
}
```

### GET /api/clients/users/{user_id}/
Retrieves user details including similarity score after enrollment.

**Response after enrollment:**
```json
{
    "id": 1,
    "user_id": "user123",
    "name": "Test User",
    "similarity_with_old_photo": 0.8745,
    "is_enrolled": true
}
```

## Video Overlay Features

The video window shows:

1. **Face Detection Box**: Green rectangle around detected face
2. **Liveness Status**: "Blink detected", "Head movement detected", etc.
3. **Frame Count**: Number of frames captured
4. **Similarity Score**: After enrollment completion
   - Percentage display
   - Colored progress bar
   - Status icon

## Next Steps After Enrollment

1. **Verify in Database**:
```bash
# In Django shell
python face_recognition_app/manage.py shell

from clients.models import ClientUser
user = ClientUser.objects.get(user_id='user123')
print(f"Similarity: {user.similarity_with_old_photo}")
print(f"Old photo: {user.old_profile_photo.url if user.old_profile_photo else 'N/A'}")
print(f"Profile image: {user.profile_image.url if user.profile_image else 'N/A'}")
```

2. **Test Authentication**:
```bash
python3 test_websocket_auth.py API_KEY SECRET_KEY http://localhost:8000 auth user123
```

3. **Review Similarity**:
- Check if similarity meets your threshold
- Adjust business logic if needed
- Consider adding alerts for low similarity

## Performance Notes

- Similarity calculation adds ~100-500ms to enrollment
- Does not affect enrollment success (even if similarity fails)
- Old photo is loaded only once during enrollment
- Embedding calculation is CPU-bound

## Security Considerations

1. **Old Photos**: Stored securely in media directory
2. **Access Control**: API requires authentication
3. **Data Privacy**: Embeddings not exposed via API
4. **Encryption**: WebSocket uses AES encryption for data transmission

## Integration Example

```python
import requests
import json
from websocket import create_connection

# Step 1: Create user with old photo
files = {'old_profile_photo': open('old_photo.jpg', 'rb')}
data = {
    'user_id': 'user123',
    'name': 'John Doe',
    'email': 'john@example.com'
}
response = requests.post(
    'http://localhost:8000/api/clients/users/',
    files=files,
    data=data,
    headers={'Authorization': 'Bearer YOUR_TOKEN'}
)
print(f"User created: {response.json()}")

# Step 2: Perform enrollment via WebSocket
# (Use test_websocket_auth.py or similar WebSocket client)

# Step 3: Check similarity
response = requests.get(
    f'http://localhost:8000/api/clients/users/{user_id}/',
    headers={'Authorization': 'Bearer YOUR_TOKEN'}
)
user_data = response.json()
similarity = user_data.get('similarity_with_old_photo')

if similarity and similarity >= 0.70:
    print(f"✅ High similarity: {similarity:.2%}")
else:
    print(f"⚠️ Low similarity: {similarity:.2%if similarity else 'N/A'}")
```

## Conclusion

This feature enables tracking facial changes over time and verifying identity consistency. The enrollment process is robust - similarity calculation failures don't prevent successful enrollment, but provide valuable additional verification data when available.

For questions or issues, check the main documentation in `docs/ENROLLMENT_SIMILARITY_FEATURE.md`.

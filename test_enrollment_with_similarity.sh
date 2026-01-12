#!/bin/bash

# Enhanced enrollment test script with old photo similarity
# Usage: ./test_enrollment_with_similarity.sh [user_id] [old_photo_path]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}  Enrollment Test with Old Photo Similarity${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Configuration
USER_ID="${1:-test_user_$(date +%s)}"
OLD_PHOTO_PATH="${2:-}"
API_KEY="${API_KEY:-your_api_key}"
SECRET_KEY="${SECRET_KEY:-your_secret_key}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  User ID: $USER_ID"
echo "  API Key: ${API_KEY:0:10}..."
echo "  Base URL: $BASE_URL"
echo "  Camera: $CAMERA_INDEX"
echo ""

# Check if old photo is provided
if [ -z "$OLD_PHOTO_PATH" ]; then
    echo -e "${YELLOW}No old photo provided. Creating sample photo...${NC}"
    OLD_PHOTO_PATH="old_photo_sample_${USER_ID}.jpg"
    python3 create_sample_photo.py "$OLD_PHOTO_PATH" "USER: $USER_ID"
    echo ""
fi

# Verify old photo exists
if [ ! -f "$OLD_PHOTO_PATH" ]; then
    echo -e "${RED}❌ Old photo not found: $OLD_PHOTO_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Old photo ready: $OLD_PHOTO_PATH${NC}"
echo ""

# Display photo info
FILE_SIZE=$(du -h "$OLD_PHOTO_PATH" | cut -f1)
IMAGE_INFO=$(python3 -c "import cv2; img=cv2.imread('$OLD_PHOTO_PATH'); print(f'{img.shape[1]}x{img.shape[0]}') if img is not None else print('Invalid')")
echo "  File size: $FILE_SIZE"
echo "  Dimensions: $IMAGE_INFO"
echo ""

# Run enrollment test
echo -e "${BLUE}Starting enrollment test...${NC}"
echo -e "${YELLOW}Press 'q' in video window to quit${NC}"
echo ""

python3 test_websocket_auth.py \
    "$API_KEY" \
    "$SECRET_KEY" \
    "$BASE_URL" \
    enrollment \
    "$USER_ID" \
    "$OLD_PHOTO_PATH"

RESULT=$?

echo ""
if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}==================================================${NC}"
    echo -e "${GREEN}  ✅ Enrollment test completed successfully${NC}"
    echo -e "${GREEN}==================================================${NC}"
    echo ""
    echo -e "${YELLOW}You can now test authentication:${NC}"
    echo "  ./test_enrollment_with_similarity.sh auth $USER_ID"
else
    echo -e "${RED}==================================================${NC}"
    echo -e "${RED}  ❌ Enrollment test failed (exit code: $RESULT)${NC}"
    echo -e "${RED}==================================================${NC}"
fi

echo ""
echo -e "${BLUE}Test completed at $(date)${NC}"

#!/bin/bash
# Start Celery worker with pool=solo for InsightFace compatibility
# InsightFace uses ONNX which is not fork-safe, so we use solo pool

cd /Users/user/Dev/researchs/face_regocnition_v2/face_recognition_app

# Activate virtual environment
source ../env/bin/activate

echo "Starting Celery worker with --pool=solo (fork-safe for InsightFace)..."
echo ""
echo "NOTE: Using solo pool because InsightFace/ONNX is not safe with forking."
echo "This means only 1 task runs at a time, but it's reliable."
echo ""
echo "For parallel processing, consider using --pool=threads instead:"
echo "  celery -A face_app worker --pool=threads --concurrency=4 --loglevel=info"
echo ""

# Run celery with solo pool (fork-safe)
celery -A face_app worker --pool=solo --loglevel=info

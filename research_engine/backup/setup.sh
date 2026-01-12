#!/bin/bash

echo "========================================="
echo "SECURE FACE AUTHENTICATION SETUP"
echo "========================================="

# Detect if virtual environment exists
if [ -d "venv" ]; then
    echo "🔍 Virtual environment ditemukan"
    echo "🔧 Mengaktivasi virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment diaktivasi"
elif [ -d "env" ]; then
    echo "🔍 Virtual environment ditemukan di folder 'env'"
    echo "🔧 Mengaktivasi virtual environment..."
    source env/bin/activate
    echo "✅ Virtual environment diaktivasi"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Virtual environment sudah aktif: $VIRTUAL_ENV"
else
    echo "⚠️  Virtual environment tidak ditemukan"
    echo "🔧 Membuat virtual environment baru..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment dibuat dan diaktivasi"
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 tidak ditemukan. Silakan install Python3 terlebih dahulu."
    exit 1
fi

echo "✅ Python3 ditemukan: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 tidak ditemukan. Silakan install pip3 terlebih dahulu."
    exit 1
fi

echo "✅ pip3 ditemukan"

# Upgrade pip
echo "📦 Upgrading pip..."
pip3 install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Download InsightFace models
echo "📥 Downloading InsightFace models..."
python3 -c "
import insightface
print('Downloading InsightFace models...')
app = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))
print('✅ Models downloaded successfully!')
"

# Test camera
echo "📹 Testing camera..."
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ Camera test successful!')
    cap.release()
else:
    print('❌ Camera test failed!')
"

echo ""
echo "========================================="
echo "SETUP COMPLETE!"
echo "========================================="
echo "Untuk menjalankan sistem:"
echo "1. Aktivasi environment: source venv/bin/activate"
echo "2. Jalankan sistem: python3 face_auth_system.py"
echo ""
echo "Atau jalankan demo: python3 demo.py"
echo "========================================="

# Create activation script for convenience
cat > activate_env.sh << 'EOF'
#!/bin/bash
echo "🔧 Mengaktivasi face recognition environment..."
source venv/bin/activate
echo "✅ Environment aktif!"
echo "Sekarang Anda bisa menjalankan:"
echo "  python3 face_auth_system.py  # Sistem utama"
echo "  python3 demo.py              # Demo & testing"
EOF

chmod +x activate_env.sh
echo "📝 Script aktivasi dibuat: ./activate_env.sh"
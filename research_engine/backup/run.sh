#!/bin/bash

# Script untuk menjalankan Face Authentication System
# Otomatis mengaktivasi environment

echo "🚀 Memulai Face Authentication System..."

# Check dan aktivasi environment
if [ -d "venv" ]; then
    echo "🔧 Mengaktivasi virtual environment..."
    source venv/bin/activate
elif [ -d "env" ]; then
    echo "🔧 Mengaktivasi virtual environment..."
    source env/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Virtual environment sudah aktif"
else
    echo "❌ Virtual environment tidak ditemukan!"
    echo "Jalankan ./setup.sh terlebih dahulu"
    exit 1
fi

# Check if main script exists
if [ ! -f "face_auth_system.py" ]; then
    echo "❌ face_auth_system.py tidak ditemukan!"
    exit 1
fi

# Run the system
echo "✅ Menjalankan sistem..."
python3 face_auth_system.py
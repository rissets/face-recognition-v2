#!/bin/bash

# Script untuk menjalankan Demo Face Authentication System
# Otomatis mengaktivasi environment

echo "🎮 Memulai Face Authentication Demo..."

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

# Check if demo script exists
if [ ! -f "demo.py" ]; then
    echo "❌ demo.py tidak ditemukan!"
    exit 1
fi

# Run the demo
echo "✅ Menjalankan demo..."
python3 demo.py
#!/bin/bash

# Face Recognition Demo Launcher
echo "🚀 Face Recognition System Demo"
echo "================================="

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "✅ Python3 found"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    echo "✅ Python found"  
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python."
    exit 1
fi

# Check if backend is running
echo "🔍 Checking Django backend..."
curl -s http://127.0.0.1:8000/api/system/status/ > /dev/null

if [ $? -eq 0 ]; then
    echo "✅ Django backend is running on 127.0.0.1:8000"
else
    echo "⚠️  Django backend not detected on 127.0.0.1:8000"
    echo "   Please start Django backend first:"
    echo "   cd face_recognition_app && python manage.py runserver 127.0.0.1:8000"
    echo ""
    echo "Continuing anyway..."
fi

# Start frontend demo server
echo ""
echo "🌐 Starting frontend demo server..."
echo "   Frontend URL: http://localhost:8080"
echo "   Backend URL:  http://127.0.0.1:8000"
echo ""
echo "📖 Demo Instructions:"
echo "   1. Open http://localhost:8080 in your browser"
echo "   2. Register a new user or login"
echo "   3. Test face enrollment and authentication"
echo "   4. Explore all API endpoints"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start Python HTTP server
cd "$(dirname "$0")"
$PYTHON_CMD -m http.server 8080
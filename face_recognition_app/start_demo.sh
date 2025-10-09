#!/bin/bash

# Face Recognition Demo Application
# Complete setup and start script

echo "🚀 Starting Face Recognition Demo Application..."

# Check if Django server is running
if ! curl -s http://127.0.0.1:8000/api/system/status/ > /dev/null 2>&1; then
    echo "⚠️  Django server is not running at http://127.0.0.1:8000"
    echo "Please start the Django server first with:"
    echo "python manage.py runserver 127.0.0.1:8000"
    exit 1
fi

# Start simple HTTP server for frontend
echo "🌐 Starting frontend demo server..."

cd frontend_demo

# Use Python's built-in HTTP server
if command -v python3 &> /dev/null; then
    echo "📱 Frontend demo available at: http://127.0.0.1:8080"
    echo "🔧 Backend API running at: http://127.0.0.1:8000"
    echo ""
    echo "📖 Demo Features:"
    echo "  ✅ User Registration & Login"
    echo "  ✅ Face Enrollment Process"
    echo "  ✅ Face Authentication"
    echo "  ✅ WebRTC Streaming"
    echo "  ✅ Analytics Dashboard"
    echo "  ✅ User Management"
    echo "  ✅ System Status Monitor"
    echo ""
    echo "🎯 Open http://127.0.0.1:8080 in your browser to start!"
    echo "Press Ctrl+C to stop the demo server"
    echo ""
    
    python3 -m http.server 8080
else
    echo "❌ Python 3 is required to run the demo"
    exit 1
fi
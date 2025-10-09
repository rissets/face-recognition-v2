#!/usr/bin/env python3
import os
import sys
import django

# Add the face_recognition_app directory to Python path
sys.path.insert(0, '/Users/user/Dev/researchs/face_regocnition_v2/face_recognition_app')

# Set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_app.settings')

# Setup Django
django.setup()

from streaming.models import StreamingSession
from django.contrib.auth.models import User

try:
    # Get or create a test user
    user, created = User.objects.get_or_create(
        username='testuser',
        defaults={'email': 'test@example.com'}
    )
    print(f"User: {user.username} ({'created' if created else 'exists'})")
    
    # Create a new streaming session
    session = StreamingSession.objects.create(user=user)
    
    print(f"Session ID: {session.id}")
    print(f"Session Token: {session.session_token}")
    print(f"Token Length: {len(session.session_token)}")
    print(f"Is Active: {session.is_active}")
    print("✅ Session token generated successfully!")
    
    # Delete the test session
    session.delete()
    print("🗑️  Test session deleted")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
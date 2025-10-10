#!/usr/bin/env python3
"""
Test script untuk memverifikasi sistem face recognition yang telah diperbaiki
"""
import os
import sys
import requests
import json
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Add the Django project path
sys.path.append('/Users/user/Dev/researchs/face_regocnition_v2/face_recognition_app')

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_app.settings')
import django
django.setup()

from clients.models import Client
from auth_service.models import ClientUser

# Test configuration
BASE_URL = "http://localhost:8000"
API_KEY = "test-api-key"  # Replace with actual API key
CLIENT_ID = "test-client"  # Replace with actual client ID
TEST_USER_ID = "test-user-enhanced"

def create_test_image():
    """Create a simple test image"""
    # Create a simple face-like pattern (not a real face, just for testing)
    img = Image.new('RGB', (640, 480), color='lightblue')
    # Add some basic shapes to simulate a face
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Face outline (circle)
    draw.ellipse([220, 140, 420, 340], outline='black', width=3)
    
    # Eyes
    draw.ellipse([260, 180, 290, 210], fill='black')
    draw.ellipse([350, 180, 380, 210], fill='black')
    
    # Nose
    draw.line([320, 220, 320, 260], fill='black', width=2)
    
    # Mouth
    draw.arc([290, 270, 350, 300], start=0, end=180, fill='black', width=2)
    
    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    return buffer.getvalue()

def test_enrollment_session():
    """Test creating an enrollment session"""
    print("🧪 Testing enrollment session creation...")
    
    url = f"{BASE_URL}/auth/enrollment-session/"
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    
    payload = {
        "user_id": TEST_USER_ID,
        "metadata": {
            "target_samples": 5,
            "device_info": {
                "device_type": "test",
                "browser": "test-script"
            }
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 201:
            data = response.json()
            print(f"✅ Enrollment session created successfully")
            print(f"Session Token: {data.get('session_token')}")
            print(f"Target Samples: {data.get('target_samples')}")
            return data.get('session_token')
        else:
            print(f"❌ Failed to create enrollment session: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing enrollment session: {e}")
        return None

def test_enrollment_frames(session_token, num_frames=5):
    """Test processing enrollment frames"""
    print(f"\n🧪 Testing enrollment frame processing ({num_frames} frames)...")
    
    url = f"{BASE_URL}/auth/process-face-image/"
    headers = {
        'Authorization': f'Bearer {API_KEY}',
    }
    
    for i in range(num_frames):
        print(f"Processing frame {i+1}/{num_frames}...")
        
        # Create test image
        image_data = create_test_image()
        
        files = {
            'image': ('test_frame.jpg', image_data, 'image/jpeg')
        }
        
        data = {
            'session_token': session_token,
            'frame_number': i + 1,
        }
        
        try:
            response = requests.post(url, files=files, data=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Frame {i+1}: {'✅ Accepted' if result.get('frame_accepted', result.get('success')) else '❌ Rejected'}")
                print(f"  Progress: {result.get('enrollment_progress', 0):.1f}%")
                print(f"  Quality: {result.get('quality_score', 0):.3f}")
                print(f"  Liveness: {result.get('liveness_score', 0):.3f}")
                print(f"  Anti-spoofing: {result.get('anti_spoofing_score', 0):.3f}")
                
                if result.get('enrollment_complete'):
                    print(f"🎉 Enrollment completed!")
                    return True
                elif not result.get('requires_more_frames', True):
                    print(f"🎉 Enrollment completed!")
                    return True
                    
            else:
                print(f"  Frame {i+1}: ❌ Error - {response.text}")
                
            # Small delay between frames
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Frame {i+1}: ❌ Exception - {e}")
    
    return False

def test_authentication_session():
    """Test creating an authentication session"""
    print(f"\n🧪 Testing authentication session creation...")
    
    url = f"{BASE_URL}/auth/authentication-session/"
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    
    payload = {
        "user_id": TEST_USER_ID,  # Verification mode
        "require_liveness": True,
        "metadata": {
            "device_info": {
                "device_type": "test",
                "browser": "test-script"
            }
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 201:
            data = response.json()
            print(f"✅ Authentication session created successfully")
            print(f"Session Token: {data.get('session_token')}")
            print(f"Session Type: {data.get('session_type')}")
            return data.get('session_token')
        else:
            print(f"❌ Failed to create authentication session: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing authentication session: {e}")
        return None

def test_authentication_frames(session_token, num_frames=3):
    """Test processing authentication frames"""
    print(f"\n🧪 Testing authentication frame processing ({num_frames} frames)...")
    
    url = f"{BASE_URL}/auth/process-face-image/"
    headers = {
        'Authorization': f'Bearer {API_KEY}',
    }
    
    for i in range(num_frames):
        print(f"Processing frame {i+1}/{num_frames}...")
        
        # Create test image (same pattern as enrollment for positive match)
        image_data = create_test_image()
        
        files = {
            'image': ('test_auth_frame.jpg', image_data, 'image/jpeg')
        }
        
        data = {
            'session_token': session_token,
            'frame_number': i + 1,
        }
        
        try:
            response = requests.post(url, files=files, data=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    print(f"  Frame {i+1}: ✅ Authentication successful!")
                    print(f"  Similarity: {result.get('similarity_score', 0):.3f}")
                    print(f"  Quality: {result.get('quality_score', 0):.3f}")
                    print(f"  Liveness: {result.get('liveness_score', 0):.3f}")
                    
                    matched_user = result.get('matched_user', {})
                    if matched_user:
                        print(f"  Matched User: {matched_user.get('external_user_id')}")
                        print(f"  Client User ID: {result.get('client_user_id')}")
                    
                    return True
                else:
                    print(f"  Frame {i+1}: ⚠️  {result.get('message', 'Needs more frames')}")
                    if not result.get('requires_more_frames', True):
                        print(f"  Authentication failed: {result.get('error', 'Unknown error')}")
                        return False
                        
            else:
                print(f"  Frame {i+1}: ❌ Error - {response.text}")
                
            # Small delay between frames
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Frame {i+1}: ❌ Exception - {e}")
    
    print("❌ Authentication did not complete within expected frames")
    return False

def test_analytics():
    """Test analytics endpoints"""
    print(f"\n🧪 Testing analytics endpoints...")
    
    # Test daily analytics
    url = f"{BASE_URL}/analytics/api/daily/"
    headers = {
        'Authorization': f'Bearer {API_KEY}',
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("✅ Daily analytics retrieved successfully")
            print(f"  Report date: {data.get('report', {}).get('date')}")
            print(f"  Enrollments: {data.get('report', {}).get('enrollments', {})}")
            print(f"  Authentications: {data.get('report', {}).get('authentications', {})}")
        else:
            print(f"❌ Daily analytics failed: {response.text}")
    except Exception as e:
        print(f"❌ Analytics test error: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Face Recognition System Enhanced Tests\n")
    
    # Test 1: Enrollment
    session_token = test_enrollment_session()
    if session_token:
        enrollment_success = test_enrollment_frames(session_token, 5)
        
        if enrollment_success:
            # Test 2: Authentication
            auth_session_token = test_authentication_session()
            if auth_session_token:
                auth_success = test_authentication_frames(auth_session_token, 3)
                
                # Test 3: Analytics
                test_analytics()
                
                if auth_success:
                    print("\n🎉 All tests completed successfully!")
                else:
                    print("\n⚠️  Authentication test failed")
            else:
                print("\n❌ Could not create authentication session")
        else:
            print("\n❌ Enrollment test failed")
    else:
        print("\n❌ Could not create enrollment session")

if __name__ == "__main__":
    main()
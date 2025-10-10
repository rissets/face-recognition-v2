#!/usr/bin/env python
"""
Test script untuk HybridImageField
"""

import os
import sys
import django
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_recognition_app.settings')
django.setup()

import base64
from auth_service.serializers import FaceImageUploadSerializer, HybridImageField
from django.core.files.base import ContentFile
from rest_framework import serializers

def test_hybrid_image_field():
    """Test HybridImageField dengan berbagai input"""
    
    # Test 1: Base64 string dengan data URL
    print("Test 1: Base64 with data URL")
    try:
        # Create a simple base64 test image (1x1 pixel PNG)
        test_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
        test_b64 = base64.b64encode(test_png).decode('utf-8')
        data_url = f"data:image/png;base64,{test_b64}"
        
        field = HybridImageField()
        result = field.to_internal_value(data_url)
        print(f"✓ Success: {result.name}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Plain base64 string
    print("\nTest 2: Plain base64 string")
    try:
        field = HybridImageField()
        result = field.to_internal_value(test_b64)
        print(f"✓ Success: {result.name}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Dict dengan base64 key
    print("\nTest 3: Dict with base64 key")
    try:
        data = {"base64": test_b64}
        field = HybridImageField()
        result = field.to_internal_value(data)
        print(f"✓ Success: {result.name}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Invalid base64
    print("\nTest 4: Invalid base64")
    try:
        field = HybridImageField()
        result = field.to_internal_value("invalid_base64_data")
        print(f"✓ Unexpected success: {result}")
        
    except Exception as e:
        print(f"✓ Expected error: {e}")
    
    # Test 5: Empty data
    print("\nTest 5: Empty data")
    try:
        field = HybridImageField()
        result = field.to_internal_value("")
        print(f"✗ Unexpected success: {result}")
        
    except Exception as e:
        print(f"✓ Expected error: {e}")

def test_serializer():
    """Test FaceImageUploadSerializer"""
    
    print("\n" + "="*50)
    print("Testing FaceImageUploadSerializer")
    print("="*50)
    
    # Create test data
    test_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    test_b64 = base64.b64encode(test_png).decode('utf-8')
    
    # Test 1: Valid data
    print("\nTest 1: Valid data")
    data = {
        'image': f"data:image/png;base64,{test_b64}",
        'session_token': 'test-session-123'
    }
    
    serializer = FaceImageUploadSerializer(data=data)
    if serializer.is_valid():
        print("✓ Serializer is valid")
        print(f"  Image: {serializer.validated_data['image'].name}")
        print(f"  Session: {serializer.validated_data['session_token']}")
    else:
        print(f"✗ Serializer errors: {serializer.errors}")
    
    # Test 2: Missing image
    print("\nTest 2: Missing image")
    data = {
        'session_token': 'test-session-123'
    }
    
    serializer = FaceImageUploadSerializer(data=data)
    if serializer.is_valid():
        print("✗ Unexpected success")
    else:
        print(f"✓ Expected error: {serializer.errors}")
    
    # Test 3: Invalid image
    print("\nTest 3: Invalid image")
    data = {
        'image': 'invalid_image_data',
        'session_token': 'test-session-123'
    }
    
    serializer = FaceImageUploadSerializer(data=data)
    if serializer.is_valid():
        print("✗ Unexpected success")
    else:
        print(f"✓ Expected error: {serializer.errors}")

if __name__ == '__main__':
    print("Testing HybridImageField")
    print("="*50)
    test_hybrid_image_field()
    
    test_serializer()
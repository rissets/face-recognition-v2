#!/usr/bin/env python3
"""
Debug script untuk mengidentifikasi masalah dengan DeepFace anti-spoofing
"""

import cv2
import numpy as np
from deepface import DeepFace
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_deepface_installation():
    """Test DeepFace installation dan dependencies"""
    print("=== TESTING DEEPFACE INSTALLATION ===")
    
    try:
        import deepface
        print(f"✓ DeepFace imported successfully")
        print(f"✓ DeepFace version: {deepface.__version__ if hasattr(deepface, '__version__') else 'Unknown'}")
    except ImportError as e:
        print(f"✗ DeepFace import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow not found")
    
    try:
        import keras
        print(f"✓ Keras version: {keras.__version__}")
    except ImportError:
        print("✗ Keras not found")
    
    return True

def test_webcam():
    """Test webcam functionality"""
    print("\n=== TESTING WEBCAM ===")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot open webcam")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("✗ Cannot read frame from webcam")
        cap.release()
        return False
    
    print(f"✓ Webcam working - Frame shape: {frame.shape}")
    cap.release()
    return frame

def test_face_detection():
    """Test basic face detection"""
    print("\n=== TESTING FACE DETECTION ===")
    
    # Create a test image with a face (or use webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot open webcam for face detection test")
        return False
    
    print("Please position your face in front of the camera and press SPACE to capture test image...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Position face and press SPACE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Detection Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            test_frame = frame.copy()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Test basic OpenCV face detection
    gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    print(f"✓ OpenCV detected {len(faces)} faces")
    
    # Save test image
    cv2.imwrite("test_image.jpg", test_frame)
    print("✓ Test image saved as 'test_image.jpg'")
    
    return test_frame, faces

def test_deepface_backends():
    """Test different DeepFace backends"""
    print("\n=== TESTING DEEPFACE BACKENDS ===")
    
    # Load test image
    if not os.path.exists("test_image.jpg"):
        print("✗ Test image not found. Run face detection test first.")
        return False
    
    test_image = cv2.imread("test_image.jpg")
    
    backends = ['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe']
    
    for backend in backends:
        try:
            print(f"\nTesting backend: {backend}")
            result = DeepFace.analyze(
                img_path=test_image,
                actions=['emotion'],  # Use emotion first as it's more stable
                enforce_detection=False,
                detector_backend=backend,
                silent=True
            )
            
            if result:
                print(f"✓ {backend}: {len(result)} faces detected")
            else:
                print(f"✗ {backend}: No faces detected")
                
        except Exception as e:
            print(f"✗ {backend}: Error - {str(e)}")

def test_deepface_spoof():
    """Test DeepFace anti-spoofing specifically"""
    print("\n=== TESTING DEEPFACE ANTI-SPOOFING ===")
    
    if not os.path.exists("test_image.jpg"):
        print("✗ Test image not found. Run face detection test first.")
        return False
    
    test_image = cv2.imread("test_image.jpg")
    
    # Test different configurations
    configs = [
        {'enforce_detection': True, 'silent': True},
        {'enforce_detection': False, 'silent': True},
        {'enforce_detection': False, 'silent': False},
    ]
    
    for i, config in enumerate(configs):
        try:
            print(f"\nConfiguration {i+1}: {config}")
            result = DeepFace.analyze(
                img_path=test_image,
                actions=['spoof'],
                **config
            )
            
            if result:
                print(f"✓ Success: {len(result)} faces analyzed")
                for j, face in enumerate(result):
                    spoof_data = face.get('spoof', {})
                    print(f"  Face {j+1}: {spoof_data}")
            else:
                print("✗ No results returned")
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")

def test_simple_spoof_detection():
    """Test simple anti-spoofing detection"""
    print("\n=== TESTING SIMPLE SPOOF DETECTION ===")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Cannot open webcam")
        return False
    
    print("Starting simple anti-spoofing test...")
    print("Press 'q' to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Simple texture analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Basic face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size > 0:
                face_texture = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                
                # Simple heuristic
                if face_texture > 100:
                    label = f"POSSIBLY REAL ({face_texture:.0f})"
                    color = (0, 255, 0)
                else:
                    label = f"POSSIBLY FAKE ({face_texture:.0f})"
                    color = (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Overall texture: {laplacian_var:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Simple Anti-Spoofing Test", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Simple Anti-Spoofing Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Simple test completed")

def main():
    print("=== DEEPFACE ANTI-SPOOFING DEBUG TOOL ===\n")
    
    # Test 1: Installation
    if not test_deepface_installation():
        print("Please install required packages and try again")
        return
    
    # Test 2: Webcam
    if not test_webcam():
        print("Please check your webcam and try again")
        return
    
    # Test 3: Face Detection
    print("\nRunning face detection test...")
    face_test_result = test_face_detection()
    if not face_test_result:
        print("Face detection test failed")
        return
    
    # Test 4: DeepFace backends
    test_deepface_backends()
    
    # Test 5: DeepFace anti-spoofing
    test_deepface_spoof()
    
    # Test 6: Simple anti-spoofing
    print("\nWould you like to test simple anti-spoofing? (y/n): ", end="")
    if input().lower().startswith('y'):
        test_simple_spoof_detection()
    
    print("\n=== DEBUG COMPLETE ===")
    print("Check the logs above to identify any issues.")
    print("If DeepFace anti-spoofing is not working, you can use the enhanced version with basic detection.")

if __name__ == "__main__":
    main()
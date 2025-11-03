"""
Test script untuk Passive Liveness Detection System
Memverifikasi semua komponen berfungsi dengan baik
"""

import cv2
import numpy as np
import time
import sys

def test_imports():
    """Test semua imports"""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe version: {mp.__version__}")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import insightface
        print(f"✓ InsightFace version: {insightface.__version__}")
    except ImportError as e:
        print(f"⚠ InsightFace not available: {e}")
    
    try:
        from ultralytics import YOLO
        print(f"✓ YOLO (Ultralytics) available")
    except ImportError as e:
        print(f"⚠ YOLO not available: {e}")
    
    try:
        import numpy as np
        import scipy
        import sklearn
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ SciPy version: {scipy.__version__}")
        print(f"✓ Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scientific packages import failed: {e}")
        return False
    
    print("\n✓ All critical imports successful!\n")
    return True


def test_detector_initialization():
    """Test inisialisasi detector"""
    print("=" * 60)
    print("Testing Detector Initialization...")
    print("=" * 60)
    
    try:
        from passive_liveness_advanced import PassiveLivenessDetector
        
        print("Initializing detector (this may take a moment)...")
        start_time = time.time()
        
        detector = PassiveLivenessDetector()
        
        init_time = time.time() - start_time
        print(f"✓ Detector initialized successfully in {init_time:.2f} seconds")
        
        return detector
    except Exception as e:
        print(f"✗ Detector initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_individual_analyzers():
    """Test setiap analyzer secara individual"""
    print("=" * 60)
    print("Testing Individual Analyzers...")
    print("=" * 60)
    
    try:
        from passive_liveness_advanced import (
            TextureAnalyzer,
            EyeBlinkDetector,
            MicroMovementAnalyzer,
            LightReflectionAnalyzer,
            SpoofingArtifactDetector,
            AttentionMechanism
        )
        
        # Create dummy data
        dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        dummy_landmarks = np.random.rand(468, 2) * 100
        dummy_eye = np.random.rand(6, 2) * 100
        
        # 1. Texture Analyzer
        print("\n1. Testing Texture Analyzer...")
        texture_analyzer = TextureAnalyzer()
        score, conf = texture_analyzer.analyze(dummy_image)
        print(f"   ✓ Texture score: {score:.3f}, confidence: {conf:.3f}")
        
        # 2. Blink Detector
        print("\n2. Testing Eye Blink Detector...")
        blink_detector = EyeBlinkDetector()
        score, conf, count = blink_detector.detect(dummy_eye, dummy_eye)
        print(f"   ✓ Blink score: {score:.3f}, confidence: {conf:.3f}, count: {count}")
        
        # 3. Movement Analyzer
        print("\n3. Testing Micro-movement Analyzer...")
        movement_analyzer = MicroMovementAnalyzer()
        score, conf = movement_analyzer.analyze_subtle_motion(dummy_landmarks)
        print(f"   ✓ Movement score: {score:.3f}, confidence: {conf:.3f}")
        
        # 4. Reflection Analyzer
        print("\n4. Testing Light Reflection Analyzer...")
        reflection_analyzer = LightReflectionAnalyzer()
        score, conf = reflection_analyzer.analyze(dummy_image, dummy_landmarks)
        print(f"   ✓ Reflection score: {score:.3f}, confidence: {conf:.3f}")
        
        # 5. Spoofing Detector
        print("\n5. Testing Spoofing Artifact Detector...")
        spoofing_detector = SpoofingArtifactDetector()
        score, conf = spoofing_detector.analyze(dummy_image, dummy_image)
        print(f"   ✓ Spoofing score: {score:.3f}, confidence: {conf:.3f}")
        
        # 6. Attention Mechanism
        print("\n6. Testing Attention Mechanism...")
        attention = AttentionMechanism()
        scores_dict = {
            'texture': 0.8,
            'blink': 0.7,
            'movement': 0.9,
            'reflection': 0.75,
            'spoofing': 0.85
        }
        features_dict = {
            'texture': {'confidence': 0.9},
            'blink': {'confidence': 0.8},
            'movement': {'confidence': 0.85},
            'reflection': {'confidence': 0.7},
            'spoofing': {'confidence': 0.8}
        }
        final_score, weights = attention.fuse_scores(scores_dict, features_dict)
        print(f"   ✓ Fused score: {final_score:.3f}")
        print(f"   ✓ Attention weights: {weights}")
        
        print("\n✓ All analyzers working correctly!\n")
        return True
        
    except Exception as e:
        print(f"✗ Analyzer testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webcam():
    """Test akses webcam"""
    print("=" * 60)
    print("Testing Webcam Access...")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open webcam")
        return False
    
    ret, frame = cap.read()
    
    if not ret:
        print("✗ Cannot read frame from webcam")
        cap.release()
        return False
    
    print(f"✓ Webcam accessible")
    print(f"✓ Frame shape: {frame.shape}")
    print(f"✓ Frame type: {frame.dtype}")
    
    cap.release()
    print()
    return True


def test_full_pipeline(detector):
    """Test full pipeline dengan webcam"""
    print("=" * 60)
    print("Testing Full Pipeline...")
    print("=" * 60)
    
    if detector is None:
        print("✗ Detector not initialized")
        return False
    
    print("\nOpening webcam for 5-second test...")
    print("Please position your face in front of the camera.")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open webcam")
        return False
    
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    
    print("\nProcessing frames (5 seconds)...")
    
    while time.time() - start_time < 5.0:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        try:
            # Run detection
            is_live, score, details = detector.detect(frame)
            
            if 'error' not in details:
                detection_count += 1
                
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"\nFrame {frame_count}:")
                    print(f"  Live: {is_live}, Score: {score:.3f}")
                    print(f"  Individual scores:")
                    for name, s in details['scores'].items():
                        print(f"    - {name}: {s:.3f}")
            
        except Exception as e:
            print(f"✗ Detection error on frame {frame_count}: {e}")
    
    cap.release()
    
    print(f"\n" + "=" * 60)
    print(f"Test completed!")
    print(f"Total frames: {frame_count}")
    print(f"Successful detections: {detection_count}")
    print(f"FPS: {frame_count / 5.0:.1f}")
    
    if detection_count > 0:
        print(f"✓ Full pipeline working correctly!")
        return True
    else:
        print(f"⚠ No faces detected in test period")
        return False


def test_performance_benchmark(detector):
    """Benchmark performance"""
    print("\n" + "=" * 60)
    print("Performance Benchmark...")
    print("=" * 60)
    
    if detector is None:
        print("✗ Detector not initialized")
        return
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warm-up
    print("\nWarming up...")
    for _ in range(5):
        detector.detect(test_frame)
    
    # Benchmark
    print("Running benchmark (100 iterations)...")
    times = []
    
    for i in range(100):
        start = time.time()
        detector.detect(test_frame)
        times.append(time.time() - start)
        
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i + 1}/100")
    
    times = np.array(times)
    
    print(f"\n" + "=" * 60)
    print(f"Benchmark Results:")
    print(f"  Mean time: {np.mean(times)*1000:.2f} ms")
    print(f"  Median time: {np.median(times)*1000:.2f} ms")
    print(f"  Std dev: {np.std(times)*1000:.2f} ms")
    print(f"  Min time: {np.min(times)*1000:.2f} ms")
    print(f"  Max time: {np.max(times)*1000:.2f} ms")
    print(f"  Expected FPS: {1.0/np.mean(times):.1f}")
    print("=" * 60)


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("PASSIVE LIVENESS DETECTION SYSTEM - TEST SUITE")
    print("=" * 60 + "\n")
    
    # Test 1: Imports
    if not test_imports():
        print("\n✗ Import test failed. Please install required packages:")
        print("  pip install -r requirements.txt")
        return
    
    # Test 2: Individual Analyzers
    if not test_individual_analyzers():
        print("\n✗ Analyzer test failed. Check error messages above.")
        return
    
    # Test 3: Webcam
    if not test_webcam():
        print("\n⚠ Webcam test failed. Full pipeline test will be skipped.")
        webcam_available = False
    else:
        webcam_available = True
    
    # Test 4: Detector Initialization
    detector = test_detector_initialization()
    if detector is None:
        print("\n✗ Detector initialization failed. Cannot proceed with pipeline test.")
        return
    
    # Test 5: Full Pipeline (if webcam available)
    if webcam_available:
        if not test_full_pipeline(detector):
            print("\n⚠ Full pipeline test had issues. Check error messages above.")
    
    # Test 6: Performance Benchmark
    response = input("\nRun performance benchmark? (y/n): ").lower()
    if response == 'y':
        test_performance_benchmark(detector)
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED!")
    print("=" * 60)
    print("\n✓ System is ready to use!")
    print("\nTo run the main application:")
    print("  python passive_liveness_advanced.py")
    print()


if __name__ == "__main__":
    main()

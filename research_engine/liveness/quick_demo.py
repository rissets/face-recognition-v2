#!/usr/bin/env python3
"""
Quick Demo - Real-time Liveness Detection
=========================================

Simple script to quickly test the liveness detection system.
"""

import cv2
import numpy as np
import time
import logging

# Configure minimal logging for demo
logging.basicConfig(level=logging.WARNING)

def create_synthetic_test():
    """Create a synthetic test without camera"""
    print("🔍 Real-time Liveness Detection - Synthetic Test")
    print("="*50)
    
    try:
        from realtime_liveness_detector import RealtimeLivenessDetector, create_detector_config
        
        # Create detector
        config = create_detector_config(
            strict_mode=False,
            enable_challenges=False,  # Disable for synthetic test
            min_blinks=2,
            liveness_threshold=0.6
        )
        
        detector = RealtimeLivenessDetector(config)
        detector.start_detection()
        
        print("✅ Detector initialized successfully")
        print("🎬 Running synthetic test frames...")
        
        # Simulate detection session
        for frame_num in range(60):  # 2 seconds at 30fps
            # Create synthetic frame with face-like features
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            
            # Add face region
            cv2.circle(frame, (320, 240), 80, (180, 150, 120), -1)
            
            # Add eyes (simulate blink every 20 frames)
            eye_open = (frame_num % 20) > 2
            eye_height = 8 if eye_open else 2
            
            cv2.ellipse(frame, (290, 220), (12, eye_height), 0, 0, 360, (50, 50, 50), -1)
            cv2.ellipse(frame, (350, 220), (12, eye_height), 0, 0, 360, (50, 50, 50), -1)
            
            # Add nose and mouth
            cv2.circle(frame, (320, 250), 6, (160, 130, 100), -1)
            cv2.ellipse(frame, (320, 270), (15, 6), 0, 0, 360, (120, 80, 80), -1)
            
            # Process frame
            annotated_frame, analysis = detector.process_frame(frame)
            
            if frame_num % 15 == 0:  # Print progress
                status = analysis.get('status', 'processing')
                if 'blink_analysis' in analysis and analysis['blink_analysis'].get('status') == 'active':
                    blinks = analysis['blink_analysis'].get('total_blinks', 0)
                    print(f"Frame {frame_num}: Status={status}, Blinks detected: {blinks}")
            
            time.sleep(0.033)  # ~30 FPS
            
            if not detector.detection_active:
                break
        
        # Get result
        result = detector.stop_detection()
        
        print("\n📊 DETECTION RESULT:")
        print(f"Is Live: {'✅ YES' if result.is_live else '❌ NO'}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Blinks Detected: {result.frame_analysis.get('total_blinks', 0)}")
        print(f"Session Duration: {result.frame_analysis.get('session_duration', 0):.1f}s")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install opencv-python mediapipe numpy")
        return False
    except Exception as e:
        print(f"❌ Error during synthetic test: {e}")
        return False

def test_camera():
    """Test with real camera"""
    print("🎥 Real-time Liveness Detection - Camera Test")
    print("="*50)
    
    try:
        from realtime_liveness_detector import RealtimeLivenessDetector, create_detector_config
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera. Testing with synthetic data instead.")
            return create_synthetic_test()
        
        print("✅ Camera opened successfully")
        
        # Create detector
        config = create_detector_config()
        detector = RealtimeLivenessDetector(config)
        
        print("📋 Instructions:")
        print("1. Position your face in the camera view")
        print("2. Press 's' to start detection")
        print("3. Blink naturally several times")
        print("4. Follow any challenges that appear")
        print("5. Press 'q' to quit")
        print("\nPress any key to continue...")
        
        detection_started = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and not detection_started:
                if not detector.detection_active:
                    detector.start_detection()
                    detection_started = True
                    print("🚀 Detection started!")
            
            # Process frame
            if detection_started:
                annotated_frame, analysis = detector.process_frame(frame)
                
                # Add simple instructions
                cv2.putText(annotated_frame, "Liveness Detection Active", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if 'blink_analysis' in analysis:
                    blink_info = analysis['blink_analysis']
                    if blink_info.get('status') == 'active':
                        blinks = blink_info.get('total_blinks', 0)
                        cv2.putText(annotated_frame, f"Blinks: {blinks}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                frame_to_show = annotated_frame
            else:
                # Show instructions
                cv2.putText(frame, "Press 's' to start detection", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frame_to_show = frame
            
            cv2.imshow('Liveness Detection Demo', frame_to_show)
            
            # Check if detection completed
            if detection_started and not detector.detection_active and detector.final_result:
                result = detector.final_result
                print(f"\n📊 DETECTION RESULT:")
                print(f"Is Live: {'✅ YES' if result.is_live else '❌ NO'}")
                print(f"Confidence: {result.confidence:.3f}")
                print(f"Challenges Passed: {result.challenges_passed}")
                print("Press 's' to start new detection or 'q' to quit")
                
                detection_started = False
                detector.final_result = None
        
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during camera test: {e}")
        return False

def main():
    """Main demo function"""
    print("🔍 Real-time Liveness Detection - Quick Demo")
    print("="*60)
    print("This demo will test the liveness detection system.")
    print()
    
    # Check what's available
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install: pip install opencv-python mediapipe numpy")
        return
    
    # Ask user preference
    print("Choose test mode:")
    print("1. Camera test (requires webcam)")
    print("2. Synthetic test (no camera needed)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            success = test_camera()
        elif choice == "2":
            success = create_synthetic_test()
        else:
            print("Invalid choice. Running synthetic test...")
            success = create_synthetic_test()
        
        if success:
            print("\n🎉 Demo completed successfully!")
            print("For more advanced features, try:")
            print("- python liveness_demo.py  (full desktop demo)")
            print("- python liveness_web_app.py  (web interface)")
            print("- python test_liveness.py  (comprehensive tests)")
        else:
            print("\n❌ Demo encountered issues. Check error messages above.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
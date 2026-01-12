#!/usr/bin/env python3
"""
Test script for Optimal Passive Liveness Detection with 3-second auth mode
"""

import cv2
import time
from face_recognition_app.core.passive_liveness_optimal import OptimizedPassiveLivenessDetector


def test_auth_mode():
    """Test authentication mode with 3-second timeout"""
    print("=" * 70)
    print("Testing Optimal Passive Liveness - AUTHENTICATION MODE")
    print("=" * 70)
    print("\nSettings:")
    print("  • Timeout: 3 seconds")
    print("  • Required blinks: 1+ (more relaxed than enrollment)")
    print("  • Auto-close: Yes")
    print("\nInstructions:")
    print("  1. Look at camera")
    print("  2. Blink at least once within 3 seconds")
    print("  3. Session will auto-close after 3 seconds")
    print("\nPress SPACE to start, ESC to quit")
    print("=" * 70)
    
    # Initialize detector with 3-second timeout for auth
    detector = OptimizedPassiveLivenessDetector(debug=True, max_duration=3.0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return
    
    print("\n✅ Camera ready! Press SPACE to start authentication...")
    
    started = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show waiting screen
        if not started:
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Add text overlay
            cv2.rectangle(display_frame, (10, 10), (w-10, 80), (0, 0, 0), -1)
            cv2.putText(display_frame, "Press SPACE to start authentication", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Authentication Test', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                started = True
                print("\n🚀 Starting authentication session...")
                detector.blink_detector.start_time = time.time()  # Reset timer
            continue
        
        # Run detection
        is_live, score, details = detector.detect(frame)
        
        # Visualize results
        display_frame = detector.visualize_results(frame, details)
        
        # Check for timeout
        if details.get('timeout', False):
            print("\n" + "=" * 70)
            print("⏱️  TIMEOUT REACHED - FINAL RESULT")
            print("=" * 70)
            print(f"  • Result: {'✅ REAL' if is_live else '❌ SPOOF'}")
            print(f"  • Score: {score:.3f}")
            print(f"  • Blink count: {details.get('blink_count', 0)}")
            print(f"  • Reason: {details.get('reason', 'N/A')}")
            print("=" * 70)
            
            # Show result for 2 seconds
            cv2.imshow('Authentication Test', display_frame)
            cv2.waitKey(2000)
            break
        
        cv2.imshow('Authentication Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset
            print("\n🔄 Resetting session...")
            detector.blink_detector.start_time = time.time()
            detector.blink_detector.blink_counter = 0
            detector.final_scores.clear()
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed!")


def test_enrollment_mode():
    """Test enrollment mode with 5-second timeout"""
    print("=" * 70)
    print("Testing Optimal Passive Liveness - ENROLLMENT MODE")
    print("=" * 70)
    print("\nSettings:")
    print("  • Timeout: 5 seconds")
    print("  • Required blinks: 2+ (stricter than auth)")
    print("  • Auto-close: Yes")
    print("\nInstructions:")
    print("  1. Look at camera")
    print("  2. Blink at least twice within 5 seconds")
    print("  3. Session will auto-close after 5 seconds")
    print("\nPress SPACE to start, ESC to quit")
    print("=" * 70)
    
    # Initialize detector with 5-second timeout for enrollment
    detector = OptimizedPassiveLivenessDetector(debug=True, max_duration=5.0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return
    
    print("\n✅ Camera ready! Press SPACE to start enrollment...")
    
    started = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show waiting screen
        if not started:
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Add text overlay
            cv2.rectangle(display_frame, (10, 10), (w-10, 80), (0, 0, 0), -1)
            cv2.putText(display_frame, "Press SPACE to start enrollment", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Enrollment Test', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                started = True
                print("\n🚀 Starting enrollment session...")
                detector.blink_detector.start_time = time.time()  # Reset timer
            continue
        
        # Run detection
        is_live, score, details = detector.detect(frame)
        
        # Visualize results
        display_frame = detector.visualize_results(frame, details)
        
        # Check for timeout
        if details.get('timeout', False):
            print("\n" + "=" * 70)
            print("⏱️  TIMEOUT REACHED - FINAL RESULT")
            print("=" * 70)
            print(f"  • Result: {'✅ REAL' if is_live else '❌ SPOOF'}")
            print(f"  • Score: {score:.3f}")
            print(f"  • Blink count: {details.get('blink_count', 0)}")
            print(f"  • Reason: {details.get('reason', 'N/A')}")
            print("=" * 70)
            
            # Show result for 2 seconds
            cv2.imshow('Enrollment Test', display_frame)
            cv2.waitKey(2000)
            break
        
        cv2.imshow('Enrollment Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset
            print("\n🔄 Resetting session...")
            detector.blink_detector.start_time = time.time()
            detector.blink_detector.blink_counter = 0
            detector.final_scores.clear()
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed!")


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("OPTIMAL PASSIVE LIVENESS DETECTION - TEST SUITE")
    print("=" * 70)
    print("\nAvailable modes:")
    print("  1. Authentication mode (3 seconds, 1+ blinks)")
    print("  2. Enrollment mode (5 seconds, 2+ blinks)")
    print("\nUsage:")
    print("  python test_optimal_liveness_auth.py auth     - Test auth mode")
    print("  python test_optimal_liveness_auth.py enroll   - Test enrollment mode")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'auth':
            test_auth_mode()
        elif mode == 'enroll':
            test_enrollment_mode()
        else:
            print(f"\n❌ Unknown mode: {mode}")
            print("Use 'auth' or 'enroll'")
    else:
        # Default to auth mode
        test_auth_mode()

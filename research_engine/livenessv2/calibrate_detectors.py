#!/usr/bin/env python3
"""
Liveness Detection Comparison & Calibration Tool
===============================================

This tool helps compare and calibrate the different liveness detectors
to ensure consistent behavior across models.
"""

import cv2
import numpy as np
import time
from collections import deque

def test_detector_parameters():
    """Test and compare detector parameters"""
    
    print("=" * 60)
    print("🔧 LIVENESS DETECTOR PARAMETER COMPARISON")
    print("=" * 60)
    
    # CNN Detector parameters
    print("\n📱 CNN Detector (Trained Model):")
    print(f"  • Model: best_model.h5")
    print(f"  • Confidence Threshold: 0.5")
    print(f"  • Smoothing: Simple average (window=5)")
    print(f"  • Logic: real > fake AND real > threshold")
    
    # Advanced Detector parameters  
    print("\n🚀 Advanced Detector (Demo Model):")
    print(f"  • Model: Creates new demo model")
    print(f"  • Confidence Threshold: 0.6 (stricter)")
    print(f"  • Smoothing: Weighted average (window=15)")
    print(f"  • Logic: max(real,fake) < threshold → UNCERTAIN")
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS:")
    print("=" * 60)
    
    print("\n1. For consistent behavior, use the SAME model:")
    print("   python run_liveness_realtime.py --detector cnn")
    print("   python run_liveness_realtime.py --detector advanced --model models/best_model.h5")
    
    print("\n2. Adjust confidence thresholds:")
    print("   • Lower threshold (0.3-0.5) → More sensitive to REAL")
    print("   • Higher threshold (0.6-0.8) → More strict, less false positives")
    
    print("\n3. Test with different lighting conditions")
    print("   • Good lighting → Both should work similarly") 
    print("   • Poor lighting → Advanced detector may be more conservative")

def run_calibration_test():
    """Run live calibration test"""
    
    print("\n" + "=" * 60)
    print("🧪 RUNNING LIVE CALIBRATION TEST")
    print("=" * 60)
    
    try:
        from liveness_detector import LivenessDetector
        from advanced_liveness_detector import RealTimeLivenessDetector
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    # Initialize detectors with same model
    model_path = "models/best_model.h5"
    
    print(f"\n🔄 Initializing detectors with same model: {model_path}")
    
    try:
        cnn_detector = LivenessDetector(model_path=model_path, confidence_threshold=0.5)
        advanced_detector = RealTimeLivenessDetector(model_path=model_path, use_advanced_model=True)
        
        print("✅ Both detectors initialized successfully")
        
        # Test with webcam for 30 seconds
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
            
        print("\n🎥 Starting 30-second comparison test...")
        print("Look at the camera - comparing CNN vs Advanced predictions")
        
        start_time = time.time()
        cnn_predictions = []
        advanced_predictions = []
        
        while time.time() - start_time < 30:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            # Simple face detection for testing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                
                # Get predictions from both detectors
                cnn_pred, cnn_conf = cnn_detector.predict_liveness(face_roi)
                adv_pred, adv_conf, _ = advanced_detector.predict_with_temporal_smoothing(face_roi)
                
                cnn_predictions.append((cnn_pred, cnn_conf))
                advanced_predictions.append((adv_pred, adv_conf))
                
                # Draw results
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"CNN: {cnn_pred} ({cnn_conf:.2f})", (x, y-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"ADV: {adv_pred} ({adv_conf:.2f})", (x, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow('Detector Comparison', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Analyze results
        if cnn_predictions and advanced_predictions:
            print(f"\n📊 ANALYSIS RESULTS:")
            print(f"Total predictions: {len(cnn_predictions)}")
            
            cnn_real_count = sum(1 for pred, _ in cnn_predictions if pred == "REAL")
            adv_real_count = sum(1 for pred, _ in advanced_predictions if pred == "REAL")
            
            print(f"CNN 'REAL' predictions: {cnn_real_count} ({cnn_real_count/len(cnn_predictions)*100:.1f}%)")
            print(f"Advanced 'REAL' predictions: {adv_real_count} ({adv_real_count/len(advanced_predictions)*100:.1f}%)")
            
            avg_cnn_conf = np.mean([conf for _, conf in cnn_predictions])
            avg_adv_conf = np.mean([conf for _, conf in advanced_predictions])
            
            print(f"Average CNN confidence: {avg_cnn_conf:.3f}")
            print(f"Average Advanced confidence: {avg_adv_conf:.3f}")
        
    except Exception as e:
        print(f"❌ Calibration test failed: {e}")

def main():
    """Main function"""
    test_detector_parameters()
    
    response = input("\n🧪 Run live calibration test? (y/n): ").strip().lower()
    if response == 'y':
        run_calibration_test()
    
    print("\n" + "=" * 60)
    print("✅ Calibration complete!")
    print("💡 Recommendation: Use CNN detector with trained model for best results")
    print("=" * 60)

if __name__ == "__main__":
    main()
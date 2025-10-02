#!/usr/bin/env python3
"""
Interactive blink detection test dengan threshold adjustment
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveBlink:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe eye landmarks (comprehensive)
        self.LEFT_EYE_ALL = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_ALL = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Key points for EAR calculation (6-point model)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]   # outer, top_outer, top_inner, inner, bottom_inner, bottom_outer
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380] # outer, top_outer, top_inner, inner, bottom_inner, bottom_outer
        
        self.threshold_factor = 0.85  # Adjustable
        self.blink_counter = 0
        self.total_blinks = 0
        self.ear_history = []
        self.baseline_ear = None
        
        logger.info("InteractiveBlink initialized")
    
    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        eye_points = np.array(eye_points)
        
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if C == 0:
            return 0.0
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blink(self, frame):
        """Detect blink in frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, 0.0, None, None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract eye coordinates
        left_eye_points = []
        right_eye_points = []
        
        for idx in self.LEFT_EYE:
            x = landmarks.landmark[idx].x * w
            y = landmarks.landmark[idx].y * h
            left_eye_points.append([x, y])
        
        for idx in self.RIGHT_EYE:
            x = landmarks.landmark[idx].x * w
            y = landmarks.landmark[idx].y * h
            right_eye_points.append([x, y])
        
        # Calculate EAR
        left_ear = self.calculate_ear(left_eye_points)
        right_ear = self.calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Add to history
        self.ear_history.append(avg_ear)
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)
        
        # Calculate adaptive baseline
        if len(self.ear_history) >= 10:
            recent_ears = self.ear_history[-10:]
            baseline = np.mean(recent_ears)
            
            if self.baseline_ear is None:
                self.baseline_ear = baseline
            else:
                self.baseline_ear = 0.9 * self.baseline_ear + 0.1 * baseline
            
            adaptive_threshold = self.baseline_ear * self.threshold_factor
            
            # Blink detection
            if avg_ear < adaptive_threshold:
                self.blink_counter += 1
            else:
                if self.blink_counter >= 2:
                    self.total_blinks += 1
                    logger.info(f"BLINK DETECTED! Total: {self.total_blinks}")
                self.blink_counter = 0
        
        return True, avg_ear, left_eye_points, right_eye_points
    
    def adjust_threshold(self, delta):
        """Adjust threshold factor"""
        self.threshold_factor = max(0.7, min(0.95, self.threshold_factor + delta))
        logger.info(f"Threshold factor adjusted to: {self.threshold_factor:.2f}")

def main():
    print("="*60)
    print("INTERACTIVE BLINK DETECTION TEST")
    print("="*60)
    print("Kontrol:")
    print("- 'q': Keluar")
    print("- 'r': Reset counter")
    print("- '+': Tingkatkan sensitivity (kurangi threshold)")
    print("- '-': Kurangi sensitivity (naikkan threshold)")
    print("- SPACE: Force detect blink (untuk testing)")
    print("="*60)
    
    blink_detector = InteractiveBlink()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Tidak dapat membuka kamera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Detect blink
            face_detected, ear, left_eye, right_eye = blink_detector.detect_blink(frame)
            
            # Draw visualization
            if face_detected and left_eye and right_eye:
                # Get landmarks for comprehensive visualization
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = blink_detector.face_mesh.process(rgb_frame)
                h, w = frame.shape[:2]
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    
                    # Draw all eye landmarks (small dots)
                    for idx in blink_detector.LEFT_EYE_ALL:
                        if idx < len(landmarks.landmark):
                            x = int(landmarks.landmark[idx].x * w)
                            y = int(landmarks.landmark[idx].y * h)
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    
                    for idx in blink_detector.RIGHT_EYE_ALL:
                        if idx < len(landmarks.landmark):
                            x = int(landmarks.landmark[idx].x * w)
                            y = int(landmarks.landmark[idx].y * h)
                            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
                
                # Draw key EAR calculation points (larger)
                left_eye = np.array(left_eye, dtype=np.int32)
                right_eye = np.array(right_eye, dtype=np.int32)
                
                for point in left_eye:
                    cv2.circle(frame, tuple(point), 4, (0, 255, 255), -1)  # Yellow key points
                
                for point in right_eye:
                    cv2.circle(frame, tuple(point), 4, (0, 255, 255), -1)  # Yellow key points
                
                # Draw eye contours
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                cv2.polylines(frame, [right_eye], True, (255, 0, 0), 2)
                
                # Fill eyes if blinking
                if blink_detector.baseline_ear and ear < (blink_detector.baseline_ear * blink_detector.threshold_factor):
                    cv2.fillPoly(frame, [left_eye], (0, 0, 255))
                    cv2.fillPoly(frame, [right_eye], (0, 0, 255))
            
            # Status display
            y = 30
            cv2.putText(frame, f"Frame: {frame_count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y += 35
            cv2.putText(frame, f"Total Blinks: {blink_detector.total_blinks}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            y += 35
            cv2.putText(frame, f"EAR: {ear:.4f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y += 30
            if blink_detector.baseline_ear is not None:
                adaptive_threshold = blink_detector.baseline_ear * blink_detector.threshold_factor
                cv2.putText(frame, f"Baseline: {blink_detector.baseline_ear:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 25
                cv2.putText(frame, f"Threshold: {adaptive_threshold:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 25
                cv2.putText(frame, f"Factor: {blink_detector.threshold_factor:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Building baseline...", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            y += 30
            cv2.putText(frame, f"Blink Counter: {blink_detector.blink_counter}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y += 30
            status = "FACE DETECTED" if face_detected else "NO FACE"
            color = (0, 255, 0) if face_detected else (0, 0, 255)
            cv2.putText(frame, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # EAR visualization bar (right side)
            if ear > 0:
                bar_x = frame.shape[1] - 120
                bar_y = 50
                bar_height = 300
                bar_width = 40
                
                # Background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                
                # Current EAR level
                ear_normalized = min(1.0, ear / 2.0)
                ear_level = int(ear_normalized * bar_height)
                cv2.rectangle(frame, (bar_x, bar_y + bar_height - ear_level), 
                             (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
                
                # Threshold and baseline lines
                if blink_detector.baseline_ear is not None:
                    adaptive_threshold = blink_detector.baseline_ear * blink_detector.threshold_factor
                    
                    # Threshold line (red)
                    threshold_level = int((adaptive_threshold / 2.0) * bar_height)
                    cv2.line(frame, (bar_x - 5, bar_y + bar_height - threshold_level), 
                            (bar_x + bar_width + 5, bar_y + bar_height - threshold_level), (0, 0, 255), 3)
                    
                    # Baseline line (yellow)
                    baseline_level = int((blink_detector.baseline_ear / 2.0) * bar_height)
                    cv2.line(frame, (bar_x - 5, bar_y + bar_height - baseline_level), 
                            (bar_x + bar_width + 5, bar_y + bar_height - baseline_level), (0, 255, 255), 2)
                
                cv2.putText(frame, "EAR", (bar_x - 5, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Controls info
            cv2.putText(frame, "Controls: q=quit, r=reset, +/- adjust sensitivity", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Interactive Blink Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                blink_detector.blink_counter = 0
                blink_detector.total_blinks = 0
                blink_detector.baseline_ear = None
                blink_detector.ear_history = []
                frame_count = 0
                print("All counters reset!")
            elif key == ord('+') or key == ord('='):
                blink_detector.adjust_threshold(-0.02)  # More sensitive
            elif key == ord('-'):
                blink_detector.adjust_threshold(0.02)   # Less sensitive
            elif key == ord(' '):
                # Force blink detection for testing
                blink_detector.total_blinks += 1
                print(f"Manual blink added! Total: {blink_detector.total_blinks}")
        
        print(f"\nTest completed!")
        print(f"Total frames: {frame_count}")
        print(f"Total blinks detected: {blink_detector.total_blinks}")
        print(f"Final threshold factor: {blink_detector.threshold_factor:.2f}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
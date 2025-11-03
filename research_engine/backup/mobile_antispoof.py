#!/usr/bin/env python3
"""
Mobile-Style Anti-Spoofing System
Version 1.0 - Pendekatan praktis untuk face login seperti di mobile device
Menggunakan multiple detection methods: texture analysis, motion detection, dan liveness detection
"""

import cv2
import numpy as np
import logging
import time
import os
from pathlib import Path
import threading
from collections import deque

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mobile_antispoof.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MobileAntiSpoofDetector:
    def __init__(self):
        """Initialize Mobile-Style Anti-Spoofing Detector"""
        logger.info("🚀 Inisialisasi Mobile Anti-Spoofing System...")
        
        # Load face cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Could not load face cascade")
            logger.info("✓ Face cascade loaded")
        except Exception as e:
            logger.error(f"❌ Error loading cascade: {e}")
            raise
        
        # Detection settings
        self.frame_skip = 2  # Process every 2nd frame for better performance
        
        # Multi-method detection parameters
        self.texture_threshold = 50  # Texture variance threshold
        self.motion_threshold = 3.0  # Motion detection threshold
        self.face_size_threshold = 80  # Minimum face size
        self.stability_frames = 5  # Frames needed for stable detection
        
        # History tracking
        self.face_history = deque(maxlen=10)  # Track face positions
        self.texture_history = deque(maxlen=7)  # Track texture scores
        self.motion_history = deque(maxlen=5)  # Track motion scores
        self.size_history = deque(maxlen=8)  # Track face sizes
        
        # Liveness detection
        self.blink_detector = BlinkDetector()
        self.head_movement_detector = HeadMovementDetector()
        
        # Results
        self.current_result = "UNKNOWN"
        self.confidence_score = 0.0
        self.detection_reasons = []
        
        # Stats
        self.frame_count = 0
        self.real_count = 0
        self.fake_count = 0
        self.processing_times = []
        
        logger.info("✓ Mobile Anti-Spoofing Detector ready!")

    def analyze_texture(self, face_region):
        """Analyze face texture - real faces have more texture variation than printed photos"""
        try:
            # Convert to grayscale
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region
            
            # Calculate Laplacian variance (texture measure)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = laplacian.var()
            
            # Calculate local binary pattern variance
            # Simple approximation of LBP
            kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
            lbp_response = cv2.filter2D(gray, cv2.CV_64F, kernel)
            lbp_variance = lbp_response.var()
            
            # Combined texture score
            texture_score = (texture_variance + lbp_variance) / 2
            
            # Add to history
            self.texture_history.append(texture_score)
            
            # Decision based on texture
            is_real_texture = texture_score > self.texture_threshold
            confidence = min(texture_score / 100.0, 1.0)  # Normalize to 0-1
            
            return is_real_texture, confidence, texture_score
            
        except Exception as e:
            logger.debug(f"Texture analysis error: {e}")
            return None, 0.0, 0.0

    def analyze_motion(self, current_face_pos):
        """Analyze face motion - real faces have natural micro-movements"""
        try:
            if len(self.face_history) < 2:
                self.face_history.append(current_face_pos)
                return None, 0.0, 0.0
            
            # Calculate motion between frames
            prev_pos = self.face_history[-1]
            curr_pos = current_face_pos
            
            # Motion vector
            motion_x = abs(curr_pos[0] - prev_pos[0])
            motion_y = abs(curr_pos[1] - prev_pos[1])
            motion_magnitude = np.sqrt(motion_x**2 + motion_y**2)
            
            # Add to history
            self.face_history.append(current_face_pos)
            self.motion_history.append(motion_magnitude)
            
            # Analyze motion pattern
            if len(self.motion_history) >= 3:
                recent_motion = list(self.motion_history)[-3:]
                avg_motion = np.mean(recent_motion)
                motion_std = np.std(recent_motion)
                
                # Real faces have some motion but not too erratic
                has_natural_motion = 0.5 < avg_motion < 15.0
                has_stable_motion = motion_std < 8.0
                
                is_real_motion = has_natural_motion and has_stable_motion
                confidence = 0.7 if is_real_motion else 0.3
                
                return is_real_motion, confidence, avg_motion
            
            return None, 0.0, motion_magnitude
            
        except Exception as e:
            logger.debug(f"Motion analysis error: {e}")
            return None, 0.0, 0.0

    def analyze_face_size_stability(self, face_size):
        """Analyze face size stability - real faces have consistent size when person is still"""
        try:
            self.size_history.append(face_size)
            
            if len(self.size_history) < 4:
                return None, 0.0, face_size
            
            # Calculate size variation
            recent_sizes = list(self.size_history)[-4:]
            avg_size = np.mean(recent_sizes)
            size_std = np.std(recent_sizes)
            size_cv = size_std / avg_size if avg_size > 0 else 1.0  # Coefficient of variation
            
            # Real faces have stable size (low coefficient of variation)
            is_stable_size = size_cv < 0.1  # Less than 10% variation
            confidence = max(0.1, 1.0 - size_cv * 5)  # Higher confidence for lower variation
            
            return is_stable_size, confidence, size_cv
            
        except Exception as e:
            logger.debug(f"Size stability error: {e}")
            return None, 0.0, 0.0

    def detect_faces(self, frame):
        """Detect faces with enhanced parameters"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better detection
            gray = cv2.equalizeHist(gray)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Smaller scale factor for better detection
                minNeighbors=4,    # Reduced for better sensitivity
                minSize=(self.face_size_threshold, self.face_size_threshold),
                maxSize=(300, 300),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []

    def make_decision(self):
        """Make final real/fake decision based on multiple factors"""
        try:
            decisions = []
            confidences = []
            reasons = []
            
            # Texture analysis
            if len(self.texture_history) >= 3:
                recent_textures = list(self.texture_history)[-3:]
                avg_texture = np.mean(recent_textures)
                
                if avg_texture > self.texture_threshold:
                    decisions.append(True)
                    confidences.append(min(avg_texture / 100.0, 1.0))
                    reasons.append(f"Good texture ({avg_texture:.1f})")
                else:
                    decisions.append(False)
                    confidences.append(0.3)
                    reasons.append(f"Poor texture ({avg_texture:.1f})")
            
            # Motion analysis
            if len(self.motion_history) >= 3:
                recent_motion = list(self.motion_history)[-3:]
                avg_motion = np.mean(recent_motion)
                
                if 0.5 < avg_motion < 10.0:
                    decisions.append(True)
                    confidences.append(0.8)
                    reasons.append(f"Natural motion ({avg_motion:.1f})")
                else:
                    decisions.append(False)
                    confidences.append(0.2)
                    reasons.append(f"Unnatural motion ({avg_motion:.1f})")
            
            # Size stability
            if len(self.size_history) >= 4:
                recent_sizes = list(self.size_history)[-4:]
                size_cv = np.std(recent_sizes) / np.mean(recent_sizes)
                
                if size_cv < 0.08:  # Very stable
                    decisions.append(True)
                    confidences.append(0.9)
                    reasons.append(f"Stable size (CV: {size_cv:.3f})")
                else:
                    decisions.append(False)
                    confidences.append(0.4)
                    reasons.append(f"Unstable size (CV: {size_cv:.3f})")
            
            # Blink detection (if available)
            blink_result = self.blink_detector.get_recent_blinks()
            if blink_result is not None:
                if blink_result > 0:
                    decisions.append(True)
                    confidences.append(0.95)
                    reasons.append(f"Blink detected ({blink_result})")
                else:
                    decisions.append(False)
                    confidences.append(0.1)
                    reasons.append("No blinks")
            
            # Make final decision
            if len(decisions) == 0:
                return "PROCESSING", 0.0, ["Insufficient data"]
            
            # Weighted voting
            real_votes = sum(1 for d in decisions if d)
            fake_votes = sum(1 for d in decisions if not d)
            
            # Calculate weighted confidence
            if real_votes > fake_votes:
                final_decision = "REAL"
                avg_confidence = np.mean([c for i, c in enumerate(confidences) if decisions[i]])
            elif fake_votes > real_votes:
                final_decision = "FAKE"
                avg_confidence = np.mean([c for i, c in enumerate(confidences) if not decisions[i]])
            else:
                # Tie - use confidence scores
                real_conf = np.mean([c for i, c in enumerate(confidences) if decisions[i]])
                fake_conf = np.mean([c for i, c in enumerate(confidences) if not decisions[i]])
                
                if real_conf > fake_conf:
                    final_decision = "REAL"
                    avg_confidence = real_conf
                else:
                    final_decision = "FAKE"
                    avg_confidence = fake_conf
            
            return final_decision, avg_confidence, reasons
            
        except Exception as e:
            logger.error(f"Decision making error: {e}")
            return "ERROR", 0.0, [str(e)]

    def run_detection(self):
        """Run mobile-style anti-spoofing detection"""
        logger.info("🎬 Starting Mobile Anti-Spoofing Detection...")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera")
            return
        
        # Set camera properties for mobile-like experience
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logger.info("✓ Camera initialized")
        
        print("\n" + "="*80)
        print("    📱 MOBILE ANTI-SPOOFING SYSTEM v1.0")
        print("="*80)
        print("Features:")
        print("✓ Texture Analysis (like photos vs real skin)")
        print("✓ Motion Detection (natural micro-movements)")
        print("✓ Size Stability (consistent face size)")
        print("✓ Blink Detection (liveness check)")
        print("✓ Mobile-optimized algorithms")
        print("\nControls:")
        print("q=quit | s=screenshot | d=debug | r=reset")
        print("="*80)
        
        show_debug = True
        
        # Performance tracking
        fps_counter = 0
        fps_start = time.time()
        display_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                self.frame_count += 1
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end = time.time()
                    display_fps = 30 / (fps_end - fps_start)
                    fps_start = fps_end
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process faces
                should_process = len(faces) > 0 and (self.frame_count % self.frame_skip == 0)
                
                if should_process:
                    # Process the largest face
                    if len(faces) > 0:
                        # Sort by area
                        faces_with_area = [(x, y, w, h, w*h) for x, y, w, h in faces]
                        faces_with_area.sort(key=lambda f: f[4], reverse=True)
                        x, y, w, h, _ = faces_with_area[0]
                        
                        start_time = time.time()
                        
                        # Extract face region
                        face_region = frame[y:y+h, x:x+w]
                        face_center = (x + w//2, y + h//2)
                        face_size = w * h
                        
                        # Multi-method analysis
                        texture_result = self.analyze_texture(face_region)
                        motion_result = self.analyze_motion(face_center)
                        size_result = self.analyze_face_size_stability(face_size)
                        
                        # Update blink detector
                        self.blink_detector.update(face_region)
                        
                        processing_time = time.time() - start_time
                        self.processing_times.append(processing_time)
                        
                        # Make decision
                        decision, confidence, reasons = self.make_decision()
                        
                        self.current_result = decision
                        self.confidence_score = confidence
                        self.detection_reasons = reasons
                        
                        # Update stats
                        if decision == "REAL":
                            self.real_count += 1
                        elif decision == "FAKE":
                            self.fake_count += 1
                        
                        # Log result
                        logger.info(f"Frame {self.frame_count}: {decision} "
                                  f"- Conf: {confidence:.3f}, Time: {processing_time:.3f}s")
                        logger.info(f"Reasons: {', '.join(reasons)}")
                
                # Draw results
                for (x, y, w, h) in faces:
                    # Color based on result
                    if self.current_result == "REAL":
                        color = (0, 255, 0)  # Green
                        status_text = f"REAL ({self.confidence_score:.2f})"
                    elif self.current_result == "FAKE":
                        color = (0, 0, 255)  # Red
                        status_text = f"FAKE ({self.confidence_score:.2f})"
                    else:
                        color = (0, 255, 255)  # Yellow
                        status_text = self.current_result
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw status text
                    cv2.putText(frame, status_text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Debug information
                    if show_debug and self.detection_reasons:
                        debug_y = y + h + 20
                        for reason in self.detection_reasons[-3:]:  # Show last 3 reasons
                            cv2.putText(frame, reason, (x, debug_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            debug_y += 15
                
                # Status information
                status_y = 25
                status_lines = [
                    f"FPS: {display_fps:.1f} | Frames: {self.frame_count} | Faces: {len(faces)}",
                    f"Real: {self.real_count} | Fake: {self.fake_count} | Current: {self.current_result}",
                    f"Texture samples: {len(self.texture_history)} | Motion samples: {len(self.motion_history)}"
                ]
                
                for line in status_lines:
                    cv2.putText(frame, line, (10, status_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    status_y += 20
                
                # Performance indicator
                if self.processing_times:
                    avg_time = np.mean(self.processing_times[-10:])
                    perf_color = (0, 255, 0) if avg_time < 0.02 else (0, 255, 255) if avg_time < 0.05 else (0, 0, 255)
                    cv2.putText(frame, f"Avg: {avg_time:.3f}s", (frame.shape[1]-120, 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, perf_color, 1)
                
                # Show frame
                cv2.imshow('Mobile Anti-Spoofing System v1.0', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"mobile_antispoof_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Screenshot: {filename}")
                elif key == ord('d'):
                    show_debug = not show_debug
                    logger.info(f"🔍 Debug: {'ON' if show_debug else 'OFF'}")
                elif key == ord('r'):
                    # Reset
                    self.frame_count = 0
                    self.real_count = 0
                    self.fake_count = 0
                    self.face_history.clear()
                    self.texture_history.clear()
                    self.motion_history.clear()
                    self.size_history.clear()
                    self.processing_times.clear()
                    self.current_result = "UNKNOWN"
                    self.confidence_score = 0.0
                    self.detection_reasons = []
                    logger.info("🔄 Stats reset")
        
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Final summary
            logger.info("=== MOBILE ANTISPOOFING SESSION SUMMARY ===")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Real detections: {self.real_count}")
            logger.info(f"Fake detections: {self.fake_count}")
            
            if self.frame_count > 0:
                real_percentage = (self.real_count / (self.real_count + self.fake_count)) * 100 if (self.real_count + self.fake_count) > 0 else 0
                logger.info(f"Real percentage: {real_percentage:.1f}%")
            
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                logger.info(f"Average processing time: {avg_time:.3f}s")


class BlinkDetector:
    def __init__(self):
        """Simple blink detector for liveness"""
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.blink_history = deque(maxlen=30)  # Store 30 frames of eye detection
        self.recent_blinks = 0
        
    def update(self, face_region):
        """Update blink detection"""
        try:
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region
                
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 3)
            
            # Record eye detection
            self.blink_history.append(len(eyes))
            
            # Detect blinks (when eyes disappear and reappear)
            if len(self.blink_history) >= 10:
                recent = list(self.blink_history)[-10:]
                
                # Look for pattern: eyes visible -> not visible -> visible
                blinks = 0
                for i in range(1, len(recent)-1):
                    if recent[i-1] >= 2 and recent[i] < 2 and recent[i+1] >= 2:
                        blinks += 1
                
                self.recent_blinks = blinks
                
        except Exception as e:
            logger.debug(f"Blink detection error: {e}")
    
    def get_recent_blinks(self):
        """Get recent blink count"""
        return self.recent_blinks if len(self.blink_history) >= 10 else None


class HeadMovementDetector:
    def __init__(self):
        """Simple head movement detector"""
        self.position_history = deque(maxlen=20)
        
    def update(self, face_center):
        """Update head movement tracking"""
        self.position_history.append(face_center)
        
    def get_movement_score(self):
        """Calculate movement score"""
        if len(self.position_history) < 5:
            return 0.0
            
        positions = list(self.position_history)[-5:]
        movements = []
        
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            movement = np.sqrt(dx**2 + dy**2)
            movements.append(movement)
        
        return np.mean(movements) if movements else 0.0


if __name__ == "__main__":
    try:
        detector = MobileAntiSpoofDetector()
        detector.run_detection()
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        print(f"\n❌ Error: {e}")
        print("📋 Check log: mobile_antispoof.log")
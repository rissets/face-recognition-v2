#!/usr/bin/env python3
"""
DeepFace Anti-Spoofing IMPROVED Realtime System
Version 3.0 - Fixed Detection Issues & Enhanced Accuracy
Based on working app.py implementation
"""

import cv2
import numpy as np
import logging
import time
import os
import tempfile
from pathlib import Path
from deepface_antispoofing import DeepFaceAntiSpoofing
from collections import deque

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepface_improved.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedDeepFaceDetector:
    def __init__(self):
        """Initialize Improved DeepFace Anti-Spoofing Detector"""
        logger.info("🚀 Inisialisasi DeepFace IMPROVED Anti-Spoofing...")
        
        try:
            # Initialize DeepFace Anti-Spoofing
            self.deepface_analyzer = DeepFaceAntiSpoofing()
            logger.info("✓ DeepFace Anti-Spoofing initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing DeepFace: {e}")
            raise
        
        # Create directories
        self.temp_folder = "temp_frames"
        os.makedirs(self.temp_folder, exist_ok=True)
        
        # Detection settings
        self.frame_skip = 5  # Process every 5th frame for better accuracy
        self.frame_count = 0
        self.last_analysis_time = 0
        self.min_analysis_interval = 0.5  # Minimum 0.5 seconds between analyses
        
        # Face detection
        cascade_path = os.path.join(os.path.dirname(__file__), 'antispoofing', 'data', 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("❌ Failed to load face cascade")
            raise Exception("Could not load face cascade classifier")
        
        logger.info("✓ Face cascade loaded successfully")
        
        # Results tracking
        self.current_result = {"is_real": None, "confidence": 0.0, "spoof_type": "Unknown"}
        self.result_history = deque(maxlen=5)  # Keep last 5 results for smoothing
        self.stable_result = None
        self.stable_confidence = 0.0
        
        # Stats
        self.total_analyses = 0
        self.real_count = 0
        self.fake_count = 0
        self.processing_times = []
        
        logger.info("✓ Improved DeepFace Detector ready!")

    def detect_faces(self, frame):
        """Detect faces using OpenCV Haar Cascades"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),  # Minimum face size
                maxSize=(400, 400),  # Maximum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []

    def extract_face_region(self, frame, x, y, w, h, padding=20):
        """Extract face region with padding"""
        try:
            height, width = frame.shape[:2]
            
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            
            face_region = frame[y1:y2, x1:x2]
            
            # Ensure minimum face size
            if face_region.shape[0] < 50 or face_region.shape[1] < 50:
                return None
            
            return face_region
            
        except Exception as e:
            logger.error(f"Error extracting face region: {e}")
            return None

    def analyze_face_with_deepface(self, face_region):
        """Analyze face using DeepFace - same as working app.py"""
        try:
            start_time = time.time()
            
            # Save face to temporary file (same approach as app.py)
            timestamp = int(time.time() * 1000000)
            temp_path = os.path.join(self.temp_folder, f"face_{timestamp}.jpg")
            
            # Ensure good image quality
            cv2.imwrite(temp_path, face_region, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Analyze using DeepFace (same method as app.py)
            result = self.deepface_analyzer.analyze_deepface(temp_path)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            # Parse result (same format as app.py)
            analysis_result = {
                'is_real': result.get('is_real') == 'True',
                'confidence': float(result.get('confidence', 0.0)),
                'spoof_type': result.get('spoof_type', 'Unknown'),
                'success': result.get('success', 'True') == 'True',
                'processing_time': processing_time
            }
            
            logger.info(f"DeepFace analysis: {analysis_result}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in DeepFace analysis: {e}")
            return None

    def smooth_results(self, new_result):
        """Smooth results using history"""
        if new_result is None:
            return self.stable_result, self.stable_confidence
        
        # Add to history
        self.result_history.append(new_result)
        
        # Need at least 2 results
        if len(self.result_history) < 2:
            self.stable_result = new_result['is_real']
            self.stable_confidence = new_result['confidence']
            return self.stable_result, self.stable_confidence
        
        # Calculate weighted average based on confidence
        recent_results = list(self.result_history)
        
        # Weight recent results more heavily
        weights = [i+1 for i in range(len(recent_results))]  # [1, 2, 3, 4, 5]
        total_weight = sum(weights)
        
        # Calculate weighted confidence
        weighted_confidence = sum(r['confidence'] * w for r, w in zip(recent_results, weights)) / total_weight
        
        # Determine result based on majority vote with confidence weighting
        real_score = sum(w for r, w in zip(recent_results, weights) if r['is_real'])
        fake_score = sum(w for r, w in zip(recent_results, weights) if not r['is_real'])
        
        # Decision with hysteresis
        if real_score > fake_score * 1.2:  # Bias towards real detection
            stable_is_real = True
        else:
            stable_is_real = False
        
        self.stable_result = stable_is_real
        self.stable_confidence = weighted_confidence
        
        return self.stable_result, self.stable_confidence

    def run_improved_detection(self):
        """Run improved realtime detection"""
        logger.info("🎬 Starting Improved DeepFace Detection...")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logger.info("✓ Camera initialized")
        
        print("\n" + "="*80)
        print("    🎯 DEEPFACE ANTI-SPOOFING IMPROVED v3.0")
        print("="*80)
        print("Improvements:")
        print("✓ Fixed Detection Algorithm (based on working app.py)")
        print("✓ Better Face Extraction")
        print("✓ Enhanced Result Smoothing")
        print("✓ Improved Accuracy & Stability")
        print("✓ Optimized Performance")
        print("\nControls:")
        print("q=quit | s=screenshot | d=debug | r=reset | space=force analysis")
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
                    logger.warning("Failed to read frame")
                    continue
                
                self.frame_count += 1
                current_time = time.time()
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end = time.time()
                    display_fps = 30 / (fps_end - fps_start)
                    fps_start = fps_end
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process faces
                should_analyze = (
                    len(faces) > 0 and 
                    (self.frame_count % self.frame_skip == 0 or 
                     current_time - self.last_analysis_time > self.min_analysis_interval)
                )
                
                if should_analyze:
                    # Take the largest face
                    if len(faces) > 0:
                        # Sort by area (largest first)
                        faces_with_area = [(x, y, w, h, w*h) for x, y, w, h in faces]
                        faces_with_area.sort(key=lambda f: f[4], reverse=True)
                        x, y, w, h, _ = faces_with_area[0]
                        
                        # Extract face region
                        face_region = self.extract_face_region(frame, x, y, w, h)
                        
                        if face_region is not None:
                            # Analyze with DeepFace
                            analysis_result = self.analyze_face_with_deepface(face_region)
                            
                            if analysis_result and analysis_result.get('success', False):
                                self.current_result = analysis_result
                                self.total_analyses += 1
                                self.last_analysis_time = current_time
                                
                                # Update stats
                                if analysis_result['is_real']:
                                    self.real_count += 1
                                else:
                                    self.fake_count += 1
                
                # Get smoothed result
                stable_is_real, stable_confidence = self.smooth_results(self.current_result if self.current_result['is_real'] is not None else None)
                
                # Draw results
                for (x, y, w, h) in faces:
                    # Determine color and text
                    if stable_is_real is not None:
                        if stable_is_real:
                            color = (0, 255, 0)  # Green for REAL
                            status_text = f"REAL ({stable_confidence:.2f})"
                        else:
                            color = (0, 0, 255)  # Red for FAKE
                            status_text = f"FAKE ({stable_confidence:.2f})"
                    else:
                        color = (128, 128, 128)  # Gray for analyzing
                        status_text = "ANALYZING..."
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw status text
                    cv2.putText(frame, status_text, (x, y-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Debug information
                    if show_debug and self.current_result['is_real'] is not None:
                        debug_y = y + h + 25
                        debug_info = [
                            f"Type: {self.current_result.get('spoof_type', 'Unknown')}",
                            f"History: {len(self.result_history)}",
                            f"Analyses: {self.total_analyses}"
                        ]
                        
                        for info in debug_info:
                            cv2.putText(frame, info, (x, debug_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            debug_y += 18
                
                # Status information
                status_y = 30
                status_lines = [
                    f"FPS: {display_fps:.1f} | Frames: {self.frame_count} | Faces: {len(faces)}",
                    f"Analyses: {self.total_analyses} | Real: {self.real_count} | Fake: {self.fake_count}",
                    f"Debug: {'ON' if show_debug else 'OFF'} | Stable: {stable_is_real} ({stable_confidence:.2f})"
                ]
                
                for line in status_lines:
                    cv2.putText(frame, line, (10, status_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    status_y += 20
                
                # Performance indicator
                if self.processing_times:
                    avg_time = np.mean(self.processing_times[-10:])
                    perf_color = (0, 255, 0) if avg_time < 0.1 else (0, 255, 255) if avg_time < 0.2 else (0, 0, 255)
                    cv2.putText(frame, f"Avg: {avg_time:.3f}s", (frame.shape[1]-150, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, perf_color, 1)
                
                # Show frame
                cv2.imshow('DeepFace Anti-Spoofing IMPROVED v3.0', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Screenshot
                    timestamp = int(time.time())
                    filename = f"deepface_improved_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Screenshot saved: {filename}")
                elif key == ord('d'):
                    show_debug = not show_debug
                    logger.info(f"🔍 Debug mode: {'ON' if show_debug else 'OFF'}")
                elif key == ord('r'):
                    # Reset
                    self.frame_count = 0
                    self.total_analyses = 0
                    self.real_count = 0
                    self.fake_count = 0
                    self.result_history.clear()
                    self.processing_times.clear()
                    self.current_result = {"is_real": None, "confidence": 0.0, "spoof_type": "Unknown"}
                    self.stable_result = None
                    self.stable_confidence = 0.0
                    logger.info("🔄 Stats reset")
                elif key == ord(' '):
                    # Force analysis
                    self.last_analysis_time = 0
                    logger.info("⚡ Forced analysis on next frame")
        
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Clean temp directory
            try:
                import shutil
                if os.path.exists(self.temp_folder):
                    shutil.rmtree(self.temp_folder)
                logger.info("🧹 Cleanup completed")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
            
            # Final summary
            logger.info("=== IMPROVED SESSION SUMMARY ===")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Total analyses: {self.total_analyses}")
            logger.info(f"Real detections: {self.real_count}")
            logger.info(f"Fake detections: {self.fake_count}")
            
            if self.total_analyses > 0:
                real_percentage = (self.real_count / self.total_analyses) * 100
                fake_percentage = (self.fake_count / self.total_analyses) * 100
                logger.info(f"Real percentage: {real_percentage:.1f}%")
                logger.info(f"Fake percentage: {fake_percentage:.1f}%")
            
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                logger.info(f"Average processing time: {avg_time:.3f}s")

if __name__ == "__main__":
    try:
        detector = ImprovedDeepFaceDetector()
        detector.run_improved_detection()
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        print(f"\n❌ Error: {e}")
        print("📋 Check log: deepface_improved.log")
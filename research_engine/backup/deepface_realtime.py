#!/usr/bin/env python3
"""
DeepFace Anti-Spoofing Realtime System
Menggunakan library deepface-antispoofing untuk deteksi real vs fake faces
Optimized untuk realtime performance dengan accuracy tinggi
"""

import cv2
import numpy as np
import logging
import time
import os
import tempfile
from pathlib import Path
from deepface_antispoofing import DeepFaceAntiSpoofing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepface_realtime.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepFaceRealtimeDetector:
    def __init__(self):
        """Initialize DeepFace Realtime Anti-Spoofing Detector"""
        logger.info("🚀 Inisialisasi DeepFace Realtime Anti-Spoofing...")
        
        try:
            # Initialize DeepFace Anti-Spoofing
            self.deepface = DeepFaceAntiSpoofing()
            logger.info("✓ DeepFace Anti-Spoofing berhasil diinisialisasi")
        except Exception as e:
            logger.error(f"❌ Error inisialisasi DeepFace: {e}")
            raise
        
        # Performance settings
        self.frame_skip = 3  # Process every 3rd frame untuk performance
        self.frame_count = 0
        self.process_count = 0
        
        # Detection settings
        self.confidence_threshold = 0.7  # Threshold untuk confidence
        self.temp_dir = tempfile.mkdtemp()  # Temporary directory untuk image files
        
        # Stats tracking
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        self.processing_times = []
        
        # Results caching untuk stability
        self.result_cache = []
        self.cache_size = 5
        
        logger.info("✓ DeepFace Realtime Detector siap digunakan!")

    def save_temp_image(self, face_region):
        """Save face region to temporary image file"""
        try:
            timestamp = int(time.time() * 1000)  # milliseconds
            temp_path = os.path.join(self.temp_dir, f"temp_face_{timestamp}.jpg")
            
            # Resize face untuk optimal processing
            if face_region.shape[0] < 112 or face_region.shape[1] < 112:
                face_region = cv2.resize(face_region, (112, 112))
            
            cv2.imwrite(temp_path, face_region)
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving temp image: {e}")
            return None

    def cleanup_temp_image(self, temp_path):
        """Clean up temporary image file"""
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.debug(f"Error cleaning temp file: {e}")

    def detect_faces_simple(self, frame):
        """Simple face detection using OpenCV Haar Cascades untuk speed"""
        try:
            # Load cascade classifier
            if not hasattr(self, 'face_cascade'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []

    def analyze_with_deepface(self, face_region):
        """Analyze face using DeepFace Anti-Spoofing"""
        temp_path = None
        try:
            start_time = time.time()
            
            # Save face region to temporary file
            temp_path = self.save_temp_image(face_region)
            if not temp_path:
                return None, 0.0, 0.0
            
            # Analyze dengan deepface antispoofing
            result = self.deepface.analyze_deepface(temp_path)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Parse hasil
            is_real = result.get('is_real', 'False') == 'True'
            confidence = float(result.get('confidence', 0.0))
            spoof_type = result.get('spoof_type', 'Unknown')
            
            logger.debug(f"DeepFace result: {result}")
            
            return is_real, confidence, processing_time
            
        except Exception as e:
            logger.error(f"Error in DeepFace analysis: {e}")
            return None, 0.0, 0.0
        finally:
            # Cleanup temporary file
            self.cleanup_temp_image(temp_path)

    def smooth_results(self, is_real, confidence):
        """Smooth results using cache untuk stability"""
        # Add to cache
        self.result_cache.append((is_real, confidence))
        
        # Keep cache size limited
        if len(self.result_cache) > self.cache_size:
            self.result_cache.pop(0)
        
        # Calculate smoothed result
        if len(self.result_cache) >= 3:  # Need at least 3 results
            real_votes = sum([1 for r, c in self.result_cache if r])
            total_votes = len(self.result_cache)
            avg_confidence = np.mean([c for r, c in self.result_cache])
            
            # Majority voting with confidence weighting
            smoothed_is_real = (real_votes / total_votes) > 0.5
            smoothed_confidence = avg_confidence
            
            return smoothed_is_real, smoothed_confidence
        else:
            return is_real, confidence

    def run_realtime(self):
        """Run realtime anti-spoofing detection"""
        logger.info("🎥 Memulai DeepFace Realtime Detection...")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Tidak dapat membuka camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("✓ Camera berhasil diinisialisasi")
        
        print("\n" + "="*70)
        print("    🎯 DEEPFACE ANTI-SPOOFING REALTIME SYSTEM")
        print("="*70)
        print("Features:")
        print("✓ DeepFace Anti-Spoofing Engine")
        print("✓ Advanced ML-based Detection")
        print("✓ Realtime Performance Optimization")
        print("✓ Result Smoothing & Caching")
        print("✓ Detailed Analytics")
        print("\nControls:")
        print("q=quit | s=screenshot | d=debug | r=reset | t=toggle processing")
        print("="*70)
        
        show_debug = False
        processing_enabled = True
        last_result = None
        last_confidence = 0.0
        
        try:
            fps_counter = 0
            fps_start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️ Tidak dapat membaca frame dari camera")
                    continue
                
                self.frame_count += 1
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                
                # Process frame (with frame skipping)
                should_process = (self.frame_count % self.frame_skip == 0) and processing_enabled
                
                if should_process:
                    # Detect faces
                    faces = self.detect_faces_simple(frame)
                    
                    for (x, y, w, h) in faces:
                        self.detection_count += 1
                        self.process_count += 1
                        
                        # Extract face region with padding
                        padding = 20
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(frame.shape[1], x + w + padding)
                        y2 = min(frame.shape[0], y + h + padding)
                        
                        face_region = frame[y1:y2, x1:x2]
                        
                        if face_region.size == 0:
                            continue
                        
                        # Analyze with DeepFace
                        is_real, confidence, proc_time = self.analyze_with_deepface(face_region)
                        
                        if is_real is not None:
                            # Smooth results
                            is_real, confidence = self.smooth_results(is_real, confidence)
                            
                            # Update stats
                            if is_real:
                                self.real_count += 1
                                result_text = "REAL"
                                color = (0, 255, 0)  # Green
                            else:
                                self.fake_count += 1
                                result_text = "FAKE"
                                color = (0, 0, 255)  # Red
                            
                            last_result = result_text
                            last_confidence = confidence
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                            
                            # Result label
                            confidence_text = f"{result_text} ({confidence:.2f})"
                            cv2.putText(frame, confidence_text, (x, y-15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                            # Debug info
                            if show_debug:
                                debug_y = y + h + 25
                                debug_texts = [
                                    f"Process: {proc_time:.3f}s",
                                    f"Avg: {np.mean(self.processing_times[-10:]):.3f}s" if self.processing_times else "Avg: N/A",
                                    f"Cache: {len(self.result_cache)}"
                                ]
                                
                                for text in debug_texts:
                                    cv2.putText(frame, text, (x, debug_y),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                    debug_y += 18
                            
                            # Log detection
                            logger.info(f"Frame {self.frame_count}: {result_text} - "
                                      f"Conf: {confidence:.3f}, Time: {proc_time:.3f}s")
                
                # Display current result (even if not processing this frame)
                if last_result:
                    status_color = (0, 255, 0) if last_result == "REAL" else (0, 0, 255)
                    status_text = f"Status: {last_result} ({last_confidence:.2f})"
                    cv2.putText(frame, status_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                # Display info
                info_lines = [
                    f"Frames: {self.frame_count} | Processed: {self.process_count}",
                    f"Real: {self.real_count} | Fake: {self.fake_count}",
                    f"Processing: {'ON' if processing_enabled else 'OFF'} | Debug: {'ON' if show_debug else 'OFF'}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(frame, line, (10, 70 + i*25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Performance info
                if hasattr(locals(), 'fps'):
                    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow('DeepFace Anti-Spoofing Realtime', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = int(time.time())
                    filename = f"deepface_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Screenshot saved: {filename}")
                elif key == ord('d'):
                    show_debug = not show_debug
                    logger.info(f"🔍 Debug mode: {'ON' if show_debug else 'OFF'}")
                elif key == ord('r'):
                    # Reset stats
                    self.frame_count = 0
                    self.process_count = 0
                    self.detection_count = 0
                    self.real_count = 0
                    self.fake_count = 0
                    self.result_cache.clear()
                    self.processing_times.clear()
                    last_result = None
                    last_confidence = 0.0
                    logger.info("🔄 Stats reset")
                elif key == ord('t'):
                    processing_enabled = not processing_enabled
                    logger.info(f"⚡ Processing: {'ENABLED' if processing_enabled else 'DISABLED'}")
        
        except KeyboardInterrupt:
            logger.info("🛑 Aplikasi dihentikan oleh user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Clean up temporary directory
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("🧹 Temporary files cleaned up")
            except Exception as e:
                logger.warning(f"Warning cleaning temp dir: {e}")
            
            # Final summary
            logger.info("=== DEEPFACE REALTIME SESSION SUMMARY ===")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Processed frames: {self.process_count}")
            logger.info(f"Total detections: {self.detection_count}")
            logger.info(f"Real faces: {self.real_count}")
            logger.info(f"Fake faces: {self.fake_count}")
            
            if self.detection_count > 0:
                real_percentage = (self.real_count / self.detection_count) * 100
                fake_percentage = (self.fake_count / self.detection_count) * 100
                logger.info(f"Real percentage: {real_percentage:.1f}%")
                logger.info(f"Fake percentage: {fake_percentage:.1f}%")
            
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                logger.info(f"Average processing time: {avg_time:.3f}s")
                logger.info(f"Estimated max FPS: {1/avg_time:.1f}")

if __name__ == "__main__":
    try:
        detector = DeepFaceRealtimeDetector()
        detector.run_realtime()
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        print(f"\n❌ Error: {e}")
        print("📝 Check the log file for details: deepface_realtime.log")
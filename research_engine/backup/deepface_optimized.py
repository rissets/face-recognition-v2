#!/usr/bin/env python3
"""
DeepFace Anti-Spoofing OPTIMIZED Realtime System
Version 2.0 - Enhanced Performance & Accuracy
"""

import cv2
import numpy as np
import logging
import time
import os
import tempfile
import threading
import queue
from pathlib import Path
from deepface_antispoofing import DeepFaceAntiSpoofing
from collections import deque

# Setup logging dengan level yang lebih clean
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepface_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepFaceOptimizedDetector:
    def __init__(self):
        """Initialize Optimized DeepFace Anti-Spoofing Detector"""
        logger.info("🚀 Inisialisasi DeepFace OPTIMIZED Anti-Spoofing...")
        
        try:
            # Initialize DeepFace Anti-Spoofing
            self.deepface = DeepFaceAntiSpoofing()
            logger.info("✓ DeepFace Anti-Spoofing engine loaded")
        except Exception as e:
            logger.error(f"❌ Error loading DeepFace: {e}")
            raise
        
        # Optimized performance settings
        self.frame_skip = 2  # Process every 2nd frame (faster)
        self.face_resize = (112, 112)  # Optimal size for processing
        self.confidence_threshold = 0.5  # Balanced threshold
        
        # Threading untuk async processing
        self.processing_queue = queue.Queue(maxsize=3)  # Limit queue size
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.stop_processing = False
        
        # Smart caching system
        self.result_history = deque(maxlen=7)  # Last 7 results for smoothing
        self.stable_result = None
        self.stable_confidence = 0.0
        self.stability_counter = 0
        
        # Stats tracking
        self.frame_count = 0
        self.processed_count = 0
        self.real_detections = 0
        self.fake_detections = 0
        self.processing_times = deque(maxlen=50)  # Keep last 50 times
        
        # Face detection optimization
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        logger.info("✓ Optimized DeepFace Detector ready!")

    def start_processing_thread(self):
        """Start background processing thread"""
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        logger.info("🔄 Background processing thread started")

    def stop_processing_thread(self):
        """Stop background processing thread"""
        self.stop_processing = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        logger.info("⏹️ Background processing thread stopped")

    def _processing_worker(self):
        """Background worker for processing faces"""
        while not self.stop_processing:
            try:
                # Get face from queue with timeout
                face_data = self.processing_queue.get(timeout=0.1)
                if face_data is None:
                    continue
                
                frame_id, face_region = face_data
                
                # Process face
                result = self._analyze_face(face_region)
                
                # Put result back
                if not self.result_queue.full():
                    self.result_queue.put((frame_id, result))
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")

    def _analyze_face(self, face_region):
        """Analyze single face region"""
        temp_path = None
        try:
            start_time = time.time()
            
            # Optimize face region
            if face_region.shape[0] != 112 or face_region.shape[1] != 112:
                face_region = cv2.resize(face_region, self.face_resize)
            
            # Save to temp file
            timestamp = int(time.time() * 1000000)  # microseconds for uniqueness
            temp_path = os.path.join(self.temp_dir, f"face_{timestamp}.jpg")
            cv2.imwrite(temp_path, face_region, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Analyze with DeepFace
            result = self.deepface.analyze_deepface(temp_path)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Parse result
            is_real = result.get('is_real', 'False') == 'True'
            confidence = float(result.get('confidence', 0.0))
            
            return {
                'is_real': is_real,
                'confidence': confidence,
                'processing_time': processing_time,
                'spoof_type': result.get('spoof_type', 'Unknown')
            }
            
        except Exception as e:
            logger.debug(f"Error in face analysis: {e}")
            return None
        finally:
            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def get_stable_result(self, new_result):
        """Get stable result using smart smoothing"""
        if new_result is None:
            return self.stable_result, self.stable_confidence
        
        # Add to history
        self.result_history.append(new_result)
        
        # Need at least 3 results for stability
        if len(self.result_history) < 3:
            return new_result['is_real'], new_result['confidence']
        
        # Calculate stability metrics
        recent_results = list(self.result_history)[-5:]  # Last 5 results
        
        # Count real vs fake
        real_count = sum(1 for r in recent_results if r['is_real'])
        fake_count = len(recent_results) - real_count
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in recent_results])
        
        # Determine stable result with hysteresis
        if real_count > fake_count:
            stable_is_real = True
        elif fake_count > real_count:
            stable_is_real = False
        else:
            # Tie - use confidence to decide
            stable_is_real = avg_confidence > self.confidence_threshold
        
        # Update stability
        if (stable_is_real == self.stable_result):
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            self.stable_result = stable_is_real
            self.stable_confidence = avg_confidence
        
        return self.stable_result, self.stable_confidence

    def detect_faces_fast(self, frame):
        """Fast face detection with optimization"""
        try:
            # Resize frame for faster detection
            scale = 0.5
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,  # Reduced for speed
                minSize=(40, 40),  # Smaller minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Scale back to original size
            faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) 
                    for (x, y, w, h) in faces]
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []

    def run_optimized(self):
        """Run optimized realtime detection"""
        logger.info("🎬 Starting DeepFace OPTIMIZED Detection...")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera")
            return
        
        # Optimal camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)  # Balanced resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Start processing thread
        self.start_processing_thread()
        
        logger.info("✓ Camera and processing ready")
        
        print("\n" + "="*75)
        print("    🎯 DEEPFACE ANTI-SPOOFING OPTIMIZED v2.0")
        print("="*75)
        print("Optimizations:")
        print("✓ Threaded Processing Pipeline")
        print("✓ Smart Result Smoothing")
        print("✓ Optimized Face Detection")
        print("✓ Reduced Latency Processing")
        print("✓ Enhanced Performance Monitoring")
        print("\nControls:")
        print("q=quit | s=screenshot | d=debug | r=reset | p=pause processing")
        print("="*75)
        
        show_debug = True  # Start with debug ON
        processing_active = True
        
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
                
                # Process frame
                should_process = (self.frame_count % self.frame_skip == 0) and processing_active
                
                if should_process:
                    # Fast face detection
                    faces = self.detect_faces_fast(frame)
                    
                    for (x, y, w, h) in faces:
                        # Extract face with minimal padding
                        padding = 10
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(frame.shape[1], x + w + padding)
                        y2 = min(frame.shape[0], y + h + padding)
                        
                        face_region = frame[y1:y2, x1:x2]
                        
                        if face_region.size == 0:
                            continue
                        
                        # Add to processing queue (non-blocking)
                        try:
                            self.processing_queue.put((self.frame_count, face_region), block=False)
                        except queue.Full:
                            pass  # Skip if queue is full
                
                # Get latest result
                latest_result = None
                try:
                    while not self.result_queue.empty():
                        _, latest_result = self.result_queue.get_nowait()
                        self.processed_count += 1
                except queue.Empty:
                    pass
                
                # Update stable result
                stable_is_real, stable_confidence = self.get_stable_result(latest_result)
                
                # Update stats
                if latest_result:
                    if latest_result['is_real']:
                        self.real_detections += 1
                    else:
                        self.fake_detections += 1
                
                # Draw results on all detected faces
                faces = self.detect_faces_fast(frame)
                for (x, y, w, h) in faces:
                    # Color based on stable result
                    if stable_is_real is not None:
                        if stable_is_real:
                            color = (0, 255, 0)  # Green for REAL
                            text = f"REAL ({stable_confidence:.2f})"
                        else:
                            color = (0, 0, 255)  # Red for FAKE  
                            text = f"FAKE ({stable_confidence:.2f})"
                    else:
                        color = (128, 128, 128)  # Gray for analyzing
                        text = "ANALYZING..."
                    
                    # Draw face box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw result text
                    cv2.putText(frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Debug info
                    if show_debug and latest_result:
                        debug_y = y + h + 20
                        debug_info = [
                            f"Proc: {latest_result.get('processing_time', 0):.3f}s",
                            f"Stability: {self.stability_counter}",
                            f"History: {len(self.result_history)}"
                        ]
                        
                        for info in debug_info:
                            cv2.putText(frame, info, (x, debug_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            debug_y += 15
                
                # Display status info
                status_y = 30
                status_info = [
                    f"FPS: {display_fps:.1f} | Frames: {self.frame_count} | Processed: {self.processed_count}",
                    f"Real: {self.real_detections} | Fake: {self.fake_detections} | Queue: {self.processing_queue.qsize()}",
                    f"Processing: {'ON' if processing_active else 'OFF'} | Debug: {'ON' if show_debug else 'OFF'}"
                ]
                
                for info in status_info:
                    cv2.putText(frame, info, (10, status_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    status_y += 20
                
                # Performance indicator
                if self.processing_times:
                    avg_time = np.mean(list(self.processing_times)[-10:])
                    perf_color = (0, 255, 0) if avg_time < 0.05 else (0, 255, 255) if avg_time < 0.1 else (0, 0, 255)
                    cv2.putText(frame, f"Avg: {avg_time:.3f}s", (frame.shape[1]-150, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, perf_color, 1)
                
                # Show frame
                cv2.imshow('DeepFace Anti-Spoofing OPTIMIZED v2.0', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"deepface_optimized_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Screenshot: {filename}")
                elif key == ord('d'):
                    show_debug = not show_debug
                    logger.info(f"🔍 Debug: {'ON' if show_debug else 'OFF'}")
                elif key == ord('r'):
                    # Reset stats
                    self.frame_count = 0
                    self.processed_count = 0
                    self.real_detections = 0
                    self.fake_detections = 0
                    self.result_history.clear()
                    self.processing_times.clear()
                    self.stable_result = None
                    self.stability_counter = 0
                    logger.info("🔄 Stats reset")
                elif key == ord('p'):
                    processing_active = not processing_active
                    logger.info(f"⚡ Processing: {'ACTIVE' if processing_active else 'PAUSED'}")
        
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
        
        finally:
            # Cleanup
            self.stop_processing_thread()
            cap.release()
            cv2.destroyAllWindows()
            
            # Clean temp directory
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except:
                pass
            
            # Final summary
            logger.info("=== OPTIMIZED SESSION SUMMARY ===")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Processed: {self.processed_count}")
            logger.info(f"Real detections: {self.real_detections}")
            logger.info(f"Fake detections: {self.fake_detections}")
            
            if self.processed_count > 0:
                accuracy = ((self.real_detections + self.fake_detections) / self.processed_count) * 100
                logger.info(f"Detection accuracy: {accuracy:.1f}%")
            
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                logger.info(f"Average processing time: {avg_time:.3f}s")
                logger.info(f"Theoretical max FPS: {1/avg_time:.1f}")

if __name__ == "__main__":
    try:
        detector = DeepFaceOptimizedDetector()
        detector.run_optimized()
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        print(f"\n❌ Error: {e}")
        print("📋 Check log: deepface_optimized.log")
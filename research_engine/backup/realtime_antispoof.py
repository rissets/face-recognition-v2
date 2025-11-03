#!/usr/bin/env python3
"""
Realtime Anti-Spoofing System - Optimized untuk performa realtime
"""

import cv2
import numpy as np
import time
import logging
import os
import insightface
from insightface.app import FaceAnalysis
import threading
from collections import deque

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_antispoof_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealtimeAntiSpoofingDetector:
    def __init__(self):
        logger.info("Inisialisasi Realtime Anti-Spoofing Detector...")
        
        # Statistik
        self.frame_count = 0
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        self.confidence_threshold = 0.7  # More strict for better spoof detection
        
        # Realtime optimization
        self.process_every_n_frames = 2  # Process every 2nd frame
        self.current_frame_skip = 0
        self.last_result = None
        self.result_cache_time = 0.5  # Cache result for 0.5 seconds
        
        # Threading untuk processing
        self.processing_queue = deque(maxlen=3)
        self.result_queue = deque(maxlen=3)
        self.processing_thread = None
        self.stop_processing = False
        
        # Initialize webcam dengan setting optimal
        self.init_camera()
        
        # Initialize InsightFace
        self.init_insightface()
        
        logger.info("Realtime Detector berhasil diinisialisasi!")
        
    def init_camera(self):
        """Initialize camera dengan pengaturan realtime optimal"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # Try different camera indices
            for i in range(1, 4):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    logger.info(f"Menggunakan camera index {i}")
                    break
            else:
                raise Exception("Tidak dapat membuka webcam")
        
        # Optimal settings untuk realtime
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Auto-adjustment
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        
        logger.info("Camera initialized untuk realtime processing")
        
    def init_insightface(self):
        """Initialize InsightFace dengan setting realtime"""
        try:
            logger.info("Memuat InsightFace untuk realtime...")
            
            # CPU provider untuk konsistensi
            providers = ['CPUExecutionProvider']
            
            self.face_app = FaceAnalysis(
                providers=providers,
                allowed_modules=['detection']  # Hanya detection untuk speed
            )
            # Smaller detection size untuk speed
            self.face_app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.6)
            logger.info("✓ InsightFace berhasil dimuat untuk realtime")
            
        except Exception as e:
            logger.error(f"Error loading InsightFace: {e}")
            # Fallback ke OpenCV
            self.face_app = None
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("✓ Menggunakan OpenCV sebagai fallback")
    
    def detect_faces_fast(self, frame):
        """Fast face detection"""
        if self.face_app is not None:
            try:
                faces = self.face_app.get(frame)
                face_data = []
                
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x, y, x2, y2 = bbox
                    w, h = x2 - x, y2 - y
                    
                    # Filter minimum size
                    if w >= 60 and h >= 60:
                        face_data.append({
                            'bbox': (x, y, w, h),
                            'confidence': face.det_score
                        })
                
                return face_data
                
            except Exception as e:
                logger.debug(f"InsightFace error: {e}")
                return self.detect_faces_opencv(frame)
        else:
            return self.detect_faces_opencv(frame)
    
    def detect_faces_opencv(self, frame):
        """Fallback OpenCV detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        face_data = []
        for (x, y, w, h) in faces:
            face_data.append({
                'bbox': (x, y, w, h),
                'confidence': 0.8
            })
        
        return face_data
    
    def fast_spoof_detection(self, face_roi):
        """Enhanced fast anti-spoofing analysis"""
        if face_roi.size == 0:
            return {'is_real': False, 'confidence': 0.0, 'method': 'Empty ROI'}
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Enhanced texture analysis
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Multi-scale edge detection
        edges_50_150 = cv2.Canny(gray, 50, 150)
        edges_30_100 = cv2.Canny(gray, 30, 100)
        edge_density_high = np.sum(edges_50_150 > 0) / (edges_50_150.shape[0] * edges_50_150.shape[1])
        edge_density_low = np.sum(edges_30_100 > 0) / (edges_30_100.shape[0] * edges_30_100.shape[1])
        
        # 3. Enhanced color analysis
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        
        # HSV analysis
        hue_std = np.std(hsv[:,:,0])
        saturation_mean = np.mean(hsv[:,:,1])
        saturation_std = np.std(hsv[:,:,1])
        value_std = np.std(hsv[:,:,2])
        
        # LAB analysis (better for skin detection)
        a_channel_std = np.std(lab[:,:,1])  # Green-Red axis
        b_channel_std = np.std(lab[:,:,2])  # Blue-Yellow axis
        
        # 4. Screen/Photo detection features
        # Check for pixelation patterns (common in screens)
        kernel = np.ones((3,3), np.uint8)
        morphed = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        pixelation_score = np.std(morphed)
        
        # Check for compression artifacts
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_complexity = len(contours)
        
        # 5. Brightness and contrast analysis
        brightness_mean = np.mean(gray)
        contrast_std = np.std(gray)
        
        # 6. Frequency domain analysis (simplified)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        high_freq_energy = np.mean(magnitude_spectrum[gray.shape[0]//4:3*gray.shape[0]//4, 
                                                    gray.shape[1]//4:3*gray.shape[1]//4])
        
        # Enhanced scoring with better thresholds
        # Real faces typically have:
        # - Higher texture variation (laplacian_var > 200)
        # - More natural color variation
        # - Less pixelation
        # - Higher frequency content
        
        # Texture score (more strict)
        texture_score = 1.0 if laplacian_var > 300 else (
            0.8 if laplacian_var > 200 else (
                0.5 if laplacian_var > 100 else 0.2
            )
        )
        
        # Edge score (consider both high and low frequency edges)
        edge_ratio = edge_density_high / (edge_density_low + 1e-7)
        edge_score = 1.0 if edge_ratio > 0.3 and edge_density_high > 0.05 else (
            0.6 if edge_density_high > 0.03 else 0.3
        )
        
        # Color diversity score (real faces have more color variation)
        color_diversity = hue_std + a_channel_std + b_channel_std
        color_score = 1.0 if color_diversity > 30 and saturation_mean > 20 else (
            0.7 if color_diversity > 20 else 0.3
        )
        
        # Anti-screen/photo score
        # Screens/photos typically have:
        # - More uniform pixelation
        # - Less high-frequency content
        # - More regular patterns
        screen_penalty = 0.0
        if pixelation_score < 10:  # Too uniform (likely screen)
            screen_penalty += 0.3
        if high_freq_energy < 50:  # Low high-freq content
            screen_penalty += 0.2
        if contrast_std < 20:  # Low contrast variation
            screen_penalty += 0.2
        
        # Combined score with penalties
        base_score = (texture_score * 0.4 + edge_score * 0.3 + color_score * 0.3)
        final_score = max(0.0, base_score - screen_penalty)
        
        # Decision with stricter threshold
        is_real = final_score > self.confidence_threshold
        
        return {
            'is_real': is_real,
            'confidence': final_score,
            'texture_score': laplacian_var,
            'edge_density': edge_density_high,
            'color_variance': saturation_std,
            'color_diversity': color_diversity,
            'screen_penalty': screen_penalty,
            'pixelation_score': pixelation_score,
            'high_freq_energy': high_freq_energy,
            'method': 'Enhanced Anti-Screen Detection'
        }
    
    def process_frame_async(self, frame, faces):
        """Async frame processing"""
        if not faces:
            return None
        
        results = []
        for face_data in faces:
            x, y, w, h = face_data['bbox']
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                result = self.fast_spoof_detection(face_roi)
                result['bbox'] = (x, y, w, h)
                results.append(result)
        
        return results
    
    def run(self):
        """Main realtime detection loop"""
        logger.info("Memulai Realtime Anti-Spoofing Detection...")
        print("\n" + "="*50)
        print("    REALTIME ANTI-SPOOFING DETECTOR")
        print("="*50)
        print("Optimized untuk:")
        print("✓ Realtime Performance (30+ FPS)")
        print("✓ Low Latency Detection")  
        print("✓ Frame Skipping Optimization")
        print("✓ Cached Results")
        print("\nControls:")
        print("q=quit | s=screenshot | d=debug | r=reset | t=threshold")
        print("="*50 + "\n")
        
        prev_frame_time = 0
        debug_mode = False
        cached_results = []
        last_process_time = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Gagal membaca frame")
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            # Calculate FPS
            current_time = time.time()
            if prev_frame_time != 0:
                fps = 1 / (current_time - prev_frame_time)
            else:
                fps = 0
            prev_frame_time = current_time
            
            # Frame skipping untuk realtime
            self.current_frame_skip += 1
            should_process = (self.current_frame_skip >= self.process_every_n_frames)
            
            if should_process or (current_time - last_process_time) > self.result_cache_time:
                self.current_frame_skip = 0
                last_process_time = current_time
                
                # Detect faces
                faces = self.detect_faces_fast(frame)
                
                if faces:
                    # Process detections
                    results = self.process_frame_async(frame, faces)
                    if results:
                        cached_results = results
                        self.detection_count += 1
            
            # Use cached results atau empty
            results = cached_results
            
            # Status
            status_text = "🔍 Realtime Processing..."
            status_color = (0, 255, 255)  # Cyan
            
            # Draw results
            if results:
                for result in results:
                    x, y, w, h = result['bbox']
                    is_real = result['is_real']
                    confidence = result['confidence']
                    
                    # Update statistics
                    if should_process:  # Only count when actually processing
                        if is_real:
                            self.real_count += 1
                        else:
                            self.fake_count += 1
                    
                    # Visual feedback
                    if is_real:
                        label = f"✅ ASLI ({confidence:.2f})"
                        color = (0, 255, 0)  # Green
                        status_text = f"✅ WAJAH ASLI - Confidence: {confidence:.2f}"
                        status_color = (0, 255, 0)
                    else:
                        label = f"❌ PALSU ({confidence:.2f})"
                        color = (0, 0, 255)  # Red
                        status_text = f"❌ WAJAH PALSU - Confidence: {confidence:.2f}"
                        status_color = (0, 0, 255)
                    
                    # Draw bounding box
                    thickness = 3 if is_real else 2
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Debug info
                    if debug_mode:
                        cv2.putText(frame, f"Texture: {result['texture_score']:.0f}", 
                                  (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        cv2.putText(frame, f"Edge: {result['edge_density']:.3f}", 
                                  (x, y+h+32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        cv2.putText(frame, f"Color: {result['color_diversity']:.1f}", 
                                  (x, y+h+44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        cv2.putText(frame, f"Screen: -{result['screen_penalty']:.2f}", 
                                  (x, y+h+56), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        cv2.putText(frame, f"HiFreq: {result['high_freq_energy']:.0f}", 
                                  (x, y+h+68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw realtime UI
            self.draw_realtime_ui(frame, fps, status_text, status_color, debug_mode)
            
            # Show frame
            cv2.imshow("🚀 Realtime Anti-Spoofing", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Aplikasi dihentikan oleh user")
                break
            elif key == ord('s'):
                screenshot_name = f"realtime_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('r'):
                self.reset_statistics()
            elif key == ord('t'):
                self.adjust_threshold()
    
    def draw_realtime_ui(self, frame, fps, status_text, status_color, debug_mode):
        """Draw realtime optimized UI"""
        h, w = frame.shape[:2]
        
        # Compact info panel
        panel_height = 140
        cv2.rectangle(frame, (10, 10), (400, panel_height), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (400, panel_height), (0, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "REALTIME ANTI-SPOOF", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Performance
        cv2.putText(frame, f"FPS: {int(fps)}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Statistics
        cv2.putText(frame, f"Detections: {self.detection_count}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Real: {self.real_count} | Fake: {self.fake_count}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Threshold
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f}", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Debug indicator
        if debug_mode:
            cv2.putText(frame, "DEBUG", (w-80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Status bar
        cv2.rectangle(frame, (10, h-60), (w-10, h-10), (30, 30, 30), -1)
        cv2.putText(frame, status_text, (15, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Controls
        cv2.putText(frame, "q=quit s=screenshot d=debug r=reset t=threshold", 
                   (15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def reset_statistics(self):
        """Reset statistics"""
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        print("📊 Statistics reset!")
        logger.info("Statistics reset")
    
    def adjust_threshold(self):
        """Interactive threshold adjustment"""
        print(f"\n⚙️  Current threshold: {self.confidence_threshold}")
        print("Quick settings:")
        print("1. High sensitivity (0.4) - More fake detections")
        print("2. Balanced (0.6) - Default")  
        print("3. Conservative (0.8) - More real detections")
        print("4. Custom value")
        
        try:
            choice = input("Choose (1-4): ").strip()
            
            if choice == '1':
                self.confidence_threshold = 0.4
                print("✅ Set to high sensitivity")
            elif choice == '2':
                self.confidence_threshold = 0.6
                print("✅ Set to balanced")
            elif choice == '3':
                self.confidence_threshold = 0.8
                print("✅ Set to conservative")
            elif choice == '4':
                value = float(input("Enter threshold (0.0-1.0): "))
                if 0.0 <= value <= 1.0:
                    self.confidence_threshold = value
                    print(f"✅ Threshold set to {value}")
                else:
                    print("❌ Invalid range")
            
            logger.info(f"Threshold changed to {self.confidence_threshold}")
            
        except (ValueError, KeyboardInterrupt):
            print("❌ Cancelled")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_processing = True
        
        logger.info("=== REALTIME SESSION SUMMARY ===")
        logger.info(f"Total frames: {self.frame_count}")
        logger.info(f"Total detections: {self.detection_count}")
        logger.info(f"Real faces: {self.real_count}")
        logger.info(f"Fake faces: {self.fake_count}")
        
        if self.detection_count > 0:
            real_pct = (self.real_count / self.detection_count) * 100
            fake_pct = (self.fake_count / self.detection_count) * 100
            logger.info(f"Real percentage: {real_pct:.1f}%")
            logger.info(f"Fake percentage: {fake_pct:.1f}%")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n📊 Session Summary:")
        print(f"Total Frames: {self.frame_count}")
        print(f"Detections: {self.detection_count}")
        if self.detection_count > 0:
            print(f"Real: {self.real_count} ({(self.real_count/self.detection_count)*100:.1f}%)")
            print(f"Fake: {self.fake_count} ({(self.fake_count/self.detection_count)*100:.1f}%)")

def main():
    print("🚀 REALTIME ANTI-SPOOFING SYSTEM")
    print("Optimized for real-time performance...")
    
    try:
        detector = RealtimeAntiSpoofingDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"❌ Error: {str(e)}")
    finally:
        try:
            detector.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()
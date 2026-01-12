#!/usr/bin/env python3
"""
Enhanced Anti-Spoofing dengan fokus deteksi foto/video di layar
"""

import cv2
import numpy as np
import time
import logging
import os
import insightface
from insightface.app import FaceAnalysis

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_screen_detection_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedScreenDetector:
    def __init__(self):
        logger.info("Inisialisasi Enhanced Screen Detection...")
        
        # Statistik
        self.frame_count = 0
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        self.confidence_threshold = 0.75  # Lebih ketat untuk deteksi layar
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Tidak dapat membuka webcam")
        
        # Optimal settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Initialize InsightFace
        self.init_insightface()
        
        logger.info("Enhanced Screen Detector siap!")
        
    def init_insightface(self):
        """Initialize InsightFace"""
        try:
            logger.info("Loading InsightFace...")
            self.face_app = FaceAnalysis(
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection']
            )
            self.face_app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.6)
            logger.info("✓ InsightFace loaded")
            
        except Exception as e:
            logger.error(f"Error loading InsightFace: {e}")
            self.face_app = None
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("✓ Using OpenCV fallback")
    
    def detect_faces(self, frame):
        """Face detection"""
        if self.face_app is not None:
            try:
                faces = self.face_app.get(frame)
                face_data = []
                
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x, y, x2, y2 = bbox
                    w, h = x2 - x, y2 - y
                    
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
        """OpenCV fallback"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        face_data = []
        for (x, y, w, h) in faces:
            face_data.append({
                'bbox': (x, y, w, h),
                'confidence': 0.8
            })
        
        return face_data
    
    def enhanced_screen_detection(self, face_roi):
        """Advanced screen/photo detection"""
        if face_roi.size == 0:
            return {'is_real': False, 'confidence': 0.0, 'method': 'Empty ROI'}
        
        h, w = face_roi.shape[:2]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. TEXTURE ANALYSIS YANG LEBIH KETAT
        # Multiple Laplacian kernels
        lap_3x3 = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap_5x5 = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        
        texture_var_3 = np.var(lap_3x3)
        texture_var_5 = np.var(lap_5x5)
        texture_combined = (texture_var_3 + texture_var_5) / 2
        
        # 2. EDGE ANALYSIS MULTI-SCALE
        # Multiple Canny thresholds
        edges_low = cv2.Canny(gray, 20, 60)
        edges_med = cv2.Canny(gray, 50, 150)
        edges_high = cv2.Canny(gray, 100, 200)
        
        edge_density_low = np.sum(edges_low > 0) / (h * w)
        edge_density_med = np.sum(edges_med > 0) / (h * w)
        edge_density_high = np.sum(edges_high > 0) / (h * w)
        
        # Real faces have more mid-frequency edges
        edge_quality = edge_density_med / (edge_density_low + 1e-7)
        
        # 3. COLOR ANALYSIS ENHANCED
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        
        # HSV analysis
        hue_mean = np.mean(hsv[:,:,0])
        hue_std = np.std(hsv[:,:,0])
        sat_mean = np.mean(hsv[:,:,1])
        sat_std = np.std(hsv[:,:,1])
        val_mean = np.mean(hsv[:,:,2])
        val_std = np.std(hsv[:,:,2])
        
        # LAB analysis (better for skin tones)
        l_std = np.std(lab[:,:,0])  # Lightness
        a_std = np.std(lab[:,:,1])  # Green-Red
        b_std = np.std(lab[:,:,2])  # Blue-Yellow
        
        # Skin tone check (real faces have specific hue ranges)
        skin_hue_range = (hue_mean >= 0 and hue_mean <= 25) or (hue_mean >= 160 and hue_mean <= 180)
        skin_saturation_ok = sat_mean >= 30 and sat_mean <= 150
        
        # 4. SCREEN/DISPLAY ARTIFACT DETECTION
        # Check for pixel grid patterns (LCD/OLED screens)
        # Sobel untuk deteksi pola grid
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Check for regular patterns (screens have regular pixel grids)
        grad_x_std = np.std(sobel_x)
        grad_y_std = np.std(sobel_y)
        gradient_uniformity = min(grad_x_std, grad_y_std) / (max(grad_x_std, grad_y_std) + 1e-7)
        
        # 5. FREQUENCY DOMAIN ANALYSIS
        # FFT untuk melihat pola frekuensi
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Check for periodic patterns (common in screens)
        center_y, center_x = h // 2, w // 2
        
        # High frequency content (real faces have more)
        high_freq_region = magnitude_spectrum[center_y-h//6:center_y+h//6, 
                                           center_x-w//6:center_x+w//6]
        high_freq_energy = np.mean(high_freq_region)
        
        # Low frequency content (screens often have strong low freq)
        low_freq_region = magnitude_spectrum[center_y-h//3:center_y+h//3, 
                                          center_x-w//3:center_x+w//3]
        low_freq_energy = np.mean(low_freq_region)
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-7)
        
        # 6. UNIFORMITY ANALYSIS
        # Screens often have more uniform regions
        # Check for large uniform areas
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        uniformity = np.std(blurred)
        
        # 7. SCORING DENGAN THRESHOLD YANG LEBIH KETAT
        
        # Texture score (real faces: texture_combined > 400)
        texture_score = 1.0 if texture_combined > 500 else (
            0.7 if texture_combined > 300 else (
                0.4 if texture_combined > 150 else 0.1
            )
        )
        
        # Edge score (real faces have good edge quality)
        edge_score = 1.0 if edge_quality > 0.4 and edge_density_med > 0.04 else (
            0.6 if edge_quality > 0.2 and edge_density_med > 0.02 else 0.2
        )
        
        # Color score (check skin tone characteristics)
        color_score = 1.0 if (skin_hue_range and skin_saturation_ok and 
                             a_std > 5 and b_std > 5 and sat_std > 15) else (
            0.6 if (sat_std > 10 and a_std > 3) else 0.3
        )
        
        # Frequency score (real faces have better freq distribution)
        freq_score = 1.0 if freq_ratio > 0.1 and high_freq_energy > 100 else (
            0.5 if freq_ratio > 0.05 else 0.2
        )
        
        # Anti-screen penalties
        screen_penalty = 0.0
        
        # Penalty for too uniform gradients (screen characteristic)
        if gradient_uniformity > 0.8:
            screen_penalty += 0.3
            
        # Penalty for too uniform overall (printed photo characteristic)
        if uniformity < 15:
            screen_penalty += 0.4
            
        # Penalty for poor texture (screen/photo characteristic)
        if texture_combined < 200:
            screen_penalty += 0.5
            
        # Penalty for unnatural color distribution
        if not skin_hue_range or not skin_saturation_ok:
            screen_penalty += 0.2
        
        # FINAL SCORING
        base_score = (texture_score * 0.3 + edge_score * 0.25 + 
                     color_score * 0.25 + freq_score * 0.2)
        final_score = max(0.0, base_score - screen_penalty)
        
        # Decision dengan threshold ketat
        is_real = final_score > self.confidence_threshold
        
        return {
            'is_real': is_real,
            'confidence': final_score,
            'texture_combined': texture_combined,
            'edge_quality': edge_quality,
            'color_diversity': a_std + b_std + sat_std,
            'freq_ratio': freq_ratio,
            'screen_penalty': screen_penalty,
            'uniformity': uniformity,
            'gradient_uniformity': gradient_uniformity,
            'skin_tone_match': skin_hue_range and skin_saturation_ok,
            'high_freq_energy': high_freq_energy,
            'method': 'Enhanced Screen Detection'
        }
    
    def run(self):
        """Main loop"""
        logger.info("Starting Enhanced Screen Detection...")
        print("\n" + "="*60)
        print("        ENHANCED SCREEN/PHOTO DETECTION")
        print("="*60)
        print("Specially tuned untuk mendeteksi:")
        print("📱 Foto di layar HP/tablet")
        print("💻 Video di layar komputer")
        print("📄 Foto yang dicetak")
        print("📺 Video call/streaming")
        print("\nThreshold: 0.75 (Ketat untuk anti-spoofing)")
        print("Controls: q=quit s=screenshot d=debug r=reset t=threshold")
        print("="*60 + "\n")
        
        prev_frame_time = 0
        debug_mode = True  # Default debug ON untuk analisis
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            
            # FPS calculation
            current_time = time.time()
            if prev_frame_time != 0:
                fps = 1 / (current_time - prev_frame_time)
            else:
                fps = 0
            prev_frame_time = current_time
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Status
            status_text = "🔍 Scanning for spoofs..."
            status_color = (0, 255, 255)
            
            if faces:
                self.detection_count += 1
                
                for face_data in faces:
                    x, y, w, h = face_data['bbox']
                    
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if face_roi.size > 0:
                        # Enhanced detection
                        result = self.enhanced_screen_detection(face_roi)
                        
                        is_real = result['is_real']
                        confidence = result['confidence']
                        
                        # Update stats
                        if is_real:
                            self.real_count += 1
                            label = f"✅ REAL ({confidence:.3f})"
                            color = (0, 255, 0)
                            status_text = f"✅ REAL FACE - Confidence: {confidence:.3f}"
                            status_color = (0, 255, 0)
                        else:
                            self.fake_count += 1
                            label = f"❌ FAKE ({confidence:.3f})"
                            color = (0, 0, 255)
                            status_text = f"❌ FAKE/SCREEN DETECTED - Confidence: {confidence:.3f}"
                            status_color = (0, 0, 255)
                        
                        # Draw results
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Detailed debug info
                        if debug_mode:
                            info_y = y + h + 15
                            cv2.putText(frame, f"Texture: {result['texture_combined']:.0f}", 
                                      (x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            cv2.putText(frame, f"Edge: {result['edge_quality']:.2f}", 
                                      (x, info_y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            cv2.putText(frame, f"Color: {result['color_diversity']:.0f}", 
                                      (x, info_y+24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            cv2.putText(frame, f"Freq: {result['freq_ratio']:.3f}", 
                                      (x, info_y+36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            cv2.putText(frame, f"Penalty: -{result['screen_penalty']:.2f}", 
                                      (x, info_y+48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            cv2.putText(frame, f"Skin: {'✓' if result['skin_tone_match'] else '✗'}", 
                                      (x, info_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Log details
                        logger.info(f"Frame {self.frame_count}: {'REAL' if is_real else 'FAKE'} - "
                                  f"Conf: {confidence:.3f}, Texture: {result['texture_combined']:.0f}, "
                                  f"EdgeQ: {result['edge_quality']:.2f}, Penalty: {result['screen_penalty']:.2f}")
            
            # Draw UI
            self.draw_ui(frame, fps, status_text, status_color, debug_mode)
            
            # Show frame
            cv2.imshow("🚫 Enhanced Screen/Photo Detection", frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Application stopped by user")
                break
            elif key == ord('s'):
                screenshot_name = f"screen_detection_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('r'):
                self.reset_stats()
            elif key == ord('t'):
                self.adjust_threshold()
    
    def draw_ui(self, frame, fps, status_text, status_color, debug_mode):
        """Draw UI"""
        h, w = frame.shape[:2]
        
        # Info panel
        cv2.rectangle(frame, (10, 10), (450, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 160), (255, 100, 0), 2)
        
        # Title
        cv2.putText(frame, "ENHANCED SCREEN DETECTION", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        
        # Stats
        cv2.putText(frame, f"FPS: {int(fps)} | Frame: {self.frame_count}", 
                   (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {self.detection_count}", 
                   (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Real: {self.real_count} | Fake: {self.fake_count}", 
                   (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f} (Strict)", 
                   (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Debug indicator
        if debug_mode:
            cv2.putText(frame, "DEBUG ON", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Status
        cv2.putText(frame, status_text, (15, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    def reset_stats(self):
        """Reset statistics"""
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        print("📊 Statistics reset!")
    
    def adjust_threshold(self):
        """Adjust threshold"""
        print(f"\n⚙️  Current threshold: {self.confidence_threshold}")
        print("Preset options:")
        print("1. Very Strict (0.8) - Almost no false positives")
        print("2. Strict (0.75) - Default for screen detection")  
        print("3. Moderate (0.65) - Balanced")
        print("4. Lenient (0.55) - More permissive")
        print("5. Custom value")
        
        try:
            choice = input("Choose (1-5): ").strip()
            
            if choice == '1':
                self.confidence_threshold = 0.8
                print("✅ Set to very strict")
            elif choice == '2':
                self.confidence_threshold = 0.75
                print("✅ Set to strict (default)")
            elif choice == '3':
                self.confidence_threshold = 0.65
                print("✅ Set to moderate")
            elif choice == '4':
                self.confidence_threshold = 0.55
                print("✅ Set to lenient")
            elif choice == '5':
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
        """Cleanup"""
        logger.info("=== ENHANCED DETECTION SESSION SUMMARY ===")
        logger.info(f"Total frames: {self.frame_count}")
        logger.info(f"Total detections: {self.detection_count}")
        logger.info(f"Real faces: {self.real_count}")
        logger.info(f"Fake/Screen faces: {self.fake_count}")
        
        if self.detection_count > 0:
            real_pct = (self.real_count / self.detection_count) * 100
            fake_pct = (self.fake_count / self.detection_count) * 100
            logger.info(f"Real percentage: {real_pct:.1f}%")
            logger.info(f"Fake percentage: {fake_pct:.1f}%")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    print("🚫 ENHANCED SCREEN/PHOTO DETECTION SYSTEM")
    print("Specially designed to detect faces on screens and photos...")
    
    try:
        detector = EnhancedScreenDetector()
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
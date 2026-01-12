#!/usr/bin/env python3
"""
Advanced Anti-Spoofing System menggunakan InsightFace dan Computer Vision techniques
"""

import cv2
import numpy as np
import time
import logging
import os
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
import pickle

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_antispoof_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedAntiSpoofingDetector:
    def __init__(self):
        logger.info("Inisialisasi Advanced Anti-Spoofing Detector...")
        
        # Statistik
        self.frame_count = 0
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        self.confidence_threshold = 0.6
        
        # History untuk temporal analysis
        self.face_history = []
        self.max_history = 10
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Tidak dapat membuka webcam")
        
        # Set optimal webcam settings untuk realtime
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Realtime optimization settings
        self.process_every_n_frames = 3  # Process every 3rd frame
        self.current_frame_skip = 0
        
        # Initialize InsightFace
        self.init_insightface()
        
        # Load atau create feature scaler
        self.init_feature_scaler()
        
        logger.info("Detector berhasil diinisialisasi!")
        
    def init_insightface(self):
        """Initialize InsightFace untuk deteksi wajah yang akurat"""
        try:
            logger.info("Memuat InsightFace models...")
            self.face_app = FaceAnalysis(
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']
            )
            self.face_app.prepare(ctx_id=0, det_size=(320, 320))  # Smaller for speed
            logger.info("✓ InsightFace berhasil dimuat")
            
        except Exception as e:
            logger.error(f"Error loading InsightFace: {e}")
            # Fallback ke OpenCV
            self.face_app = None
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("✓ Menggunakan OpenCV Haar Cascade sebagai fallback")
    
    def init_feature_scaler(self):
        """Initialize feature scaler untuk normalisasi"""
        scaler_path = "feature_scaler.pkl"
        try:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✓ Feature scaler loaded")
            else:
                self.scaler = StandardScaler()
                logger.info("✓ New feature scaler created")
        except Exception as e:
            logger.error(f"Error with feature scaler: {e}")
            self.scaler = StandardScaler()
    
    def detect_faces_insightface(self, frame):
        """Deteksi wajah menggunakan InsightFace"""
        if self.face_app is None:
            return self.detect_faces_opencv(frame)
        
        try:
            faces = self.face_app.get(frame)
            face_boxes = []
            
            for face in faces:
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                w, h = x2 - x, y2 - y
                
                # Filter face yang terlalu kecil
                if w > 80 and h > 80:
                    face_boxes.append({
                        'bbox': (x, y, w, h),
                        'confidence': face.det_score,
                        'embedding': face.embedding,
                        'landmarks': face.kps,
                        'age': getattr(face, 'age', None),
                        'gender': getattr(face, 'gender', None)
                    })
            
            return face_boxes
            
        except Exception as e:
            logger.debug(f"InsightFace detection error: {e}")
            return self.detect_faces_opencv(frame)
    
    def detect_faces_opencv(self, frame):
        """Fallback detection menggunakan OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
        
        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append({
                'bbox': (x, y, w, h),
                'confidence': 0.8,
                'embedding': None,
                'landmarks': None,
                'age': None,
                'gender': None
            })
        
        return face_boxes
    
    def extract_advanced_features(self, face_roi):
        """Extract comprehensive features untuk anti-spoofing"""
        if face_roi.size == 0:
            return np.zeros(20)  # Return zeros if empty
        
        features = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        
        # 1. Texture Features
        # Laplacian variance (sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        features.append(laplacian_var)
        
        # Sobel gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features.extend([np.mean(sobel_magnitude), np.std(sobel_magnitude)])
        
        # 2. LBP (Local Binary Pattern) features
        lbp = self.calculate_lbp(gray)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
        features.extend([np.mean(lbp_hist), np.std(lbp_hist), np.var(lbp)])
        
        # 3. Color Features
        # HSV statistics
        features.extend([np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]),  # Hue
                        np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]),   # Saturation
                        np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])])  # Value
        
        # LAB statistics
        features.extend([np.mean(lab[:,:,1]), np.std(lab[:,:,1]),  # A channel
                        np.mean(lab[:,:,2]), np.std(lab[:,:,2])])  # B channel
        
        # 4. Frequency Domain Features
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        features.extend([np.mean(magnitude_spectrum), np.std(magnitude_spectrum)])
        
        # 5. Structural Features
        # GLCM-like features
        features.append(self.calculate_contrast(gray))
        features.append(self.calculate_homogeneity(gray))
        features.append(self.calculate_energy(gray))
        
        return np.array(features)
    
    def calculate_lbp(self, gray, radius=1, n_points=8):
        """Calculate Local Binary Pattern"""
        h, w = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                binary_code = 0
                
                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = int(round(i + radius * np.cos(angle)))
                    y = int(round(j + radius * np.sin(angle)))
                    
                    if 0 <= x < h and 0 <= y < w:
                        if gray[x, y] >= center:
                            binary_code |= (1 << p)
                
                lbp[i, j] = binary_code
        
        return lbp
    
    def calculate_contrast(self, gray):
        """Calculate contrast measure"""
        return np.std(gray) / (np.mean(gray) + 1e-7)
    
    def calculate_homogeneity(self, gray):
        """Calculate homogeneity measure"""
        gray_norm = gray / 255.0
        return 1.0 / (1.0 + np.var(gray_norm))
    
    def calculate_energy(self, gray):
        """Calculate energy measure"""
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        return np.sum(hist ** 2)
    
    def temporal_analysis(self, current_features, face_id=0):
        """Analyze temporal consistency"""
        # Add current features to history
        self.face_history.append({
            'features': current_features,
            'timestamp': time.time(),
            'face_id': face_id
        })
        
        # Keep only recent history
        if len(self.face_history) > self.max_history:
            self.face_history.pop(0)
        
        if len(self.face_history) < 3:
            return 0.5  # Neutral score for insufficient history
        
        # Calculate temporal consistency
        recent_features = [h['features'] for h in self.face_history[-5:]]
        feature_matrix = np.array(recent_features)
        
        # Calculate variance across time for each feature
        temporal_variance = np.var(feature_matrix, axis=0)
        consistency_score = 1.0 / (1.0 + np.mean(temporal_variance))
        
        return consistency_score
    
    def advanced_spoof_detection(self, face_roi, landmarks=None, face_id=0):
        """Advanced anti-spoofing analysis"""
        # Extract comprehensive features
        features = self.extract_advanced_features(face_roi)
        
        # Temporal analysis
        temporal_score = self.temporal_analysis(features, face_id)
        
        # Rule-based classification with improved thresholds
        texture_score = features[0]  # Laplacian variance
        edge_strength = features[1]  # Sobel magnitude mean
        color_variance = features[9]  # Saturation std
        contrast = features[17]  # Contrast measure
        
        # Multi-criteria decision
        criteria_scores = []
        
        # Criterion 1: Texture quality
        if texture_score > 150:
            criteria_scores.append(0.8)
        elif texture_score > 80:
            criteria_scores.append(0.6)
        else:
            criteria_scores.append(0.2)
        
        # Criterion 2: Edge sharpness
        if edge_strength > 20:
            criteria_scores.append(0.8)
        elif edge_strength > 10:
            criteria_scores.append(0.6)
        else:
            criteria_scores.append(0.3)
        
        # Criterion 3: Color diversity
        if color_variance > 30:
            criteria_scores.append(0.7)
        elif color_variance > 15:
            criteria_scores.append(0.5)
        else:
            criteria_scores.append(0.3)
        
        # Criterion 4: Contrast
        if contrast > 0.3:
            criteria_scores.append(0.7)
        else:
            criteria_scores.append(0.4)
        
        # Criterion 5: Temporal consistency
        criteria_scores.append(temporal_score)
        
        # Landmark analysis bonus
        landmark_bonus = 1.0
        if landmarks is not None:
            landmark_bonus = self.analyze_landmark_quality(landmarks, face_roi.shape)
        
        # Combined score
        base_score = np.mean(criteria_scores)
        final_score = base_score * landmark_bonus
        
        # Decision
        is_real = final_score > self.confidence_threshold
        
        return {
            'is_real': is_real,
            'confidence': final_score,
            'texture_score': texture_score,
            'edge_strength': edge_strength,
            'color_variance': color_variance,
            'contrast': contrast,
            'temporal_score': temporal_score,
            'landmark_bonus': landmark_bonus,
            'criteria_scores': criteria_scores,
            'method': 'Advanced Multi-Criteria'
        }
    
    def analyze_landmark_quality(self, landmarks, face_shape):
        """Analyze landmark quality and proportions"""
        if landmarks is None or len(landmarks) < 5:
            return 1.0
        
        try:
            # Calculate face proportions
            face_height, face_width = face_shape[:2]
            landmarks = landmarks.astype(int)
            
            # Eye distance proportion
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            eye_distance = np.linalg.norm(left_eye - right_eye)
            eye_ratio = eye_distance / face_width
            
            # Nose position
            nose = landmarks[2]
            nose_x_ratio = nose[0] / face_width
            
            # Mouth position  
            mouth_left = landmarks[3]
            mouth_right = landmarks[4]
            mouth_center_x = (mouth_left[0] + mouth_right[0]) / 2
            mouth_x_ratio = mouth_center_x / face_width
            
            # Quality score based on proportions
            quality_score = 1.0
            
            # Good eye distance ratio (typically 0.25-0.4)
            if 0.25 <= eye_ratio <= 0.4:
                quality_score *= 1.1
            else:
                quality_score *= 0.9
            
            # Centered features
            if 0.4 <= nose_x_ratio <= 0.6 and 0.4 <= mouth_x_ratio <= 0.6:
                quality_score *= 1.1
            else:
                quality_score *= 0.95
            
            return min(quality_score, 1.3)  # Cap bonus
            
        except Exception as e:
            logger.debug(f"Landmark analysis error: {e}")
            return 1.0
    
    def run(self):
        """Main detection loop"""
        logger.info("Memulai Advanced Anti-Spoofing Detection...")
        print("\n=== ADVANCED ANTI-SPOOFING DETECTOR ===")
        print("Fitur:")
        print("- InsightFace detection engine")
        print("- Multi-criteria spoof analysis")
        print("- Temporal consistency checking")
        print("- Advanced texture & color analysis")
        print("\nKontrol:")
        print("- 'q': Keluar")
        print("- 's': Screenshot")
        print("- 'd': Toggle debug mode")
        print("- 'r': Reset statistik")
        print("- 't': Adjust threshold")
        print("- 'h': Toggle history display")
        print("==========================================\n")
        
        prev_frame_time = 0
        debug_mode = False
        show_history = False
        
        # Cache untuk hasil deteksi
        cached_result = None
        last_analysis_time = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Gagal membaca frame")
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            # Calculate FPS
            new_frame_time = time.time()
            if prev_frame_time != 0:
                fps = 1 / (new_frame_time - prev_frame_time)
            else:
                fps = 0
            prev_frame_time = new_frame_time
            
            # Status
            status_text = "Mencari wajah..."
            status_color = (255, 255, 0)  # Yellow
            
            # Skip frame processing untuk realtime performance
            self.current_frame_skip += 1
            should_process = (self.current_frame_skip >= self.process_every_n_frames)
            
            if should_process:
                self.current_frame_skip = 0
                # Detect faces
                faces = self.detect_faces_insightface(frame)
                current_time = time.time()
                
                # Update cache dengan hasil terbaru
                if faces:
                    cached_result = faces
                    last_analysis_time = current_time
            else:
                # Gunakan cached result untuk frame yang di-skip
                faces = cached_result if cached_result else []
            
            if faces:
                self.detection_count += 1
                
                for i, face_data in enumerate(faces):
                    x, y, w, h = face_data['bbox']
                    detection_confidence = face_data['confidence']
                    landmarks = face_data['landmarks']
                    
                    # Extract face ROI with padding
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        # Advanced spoof detection (hanya untuk frame yang diproses)
                        if should_process:
                            result = self.advanced_spoof_detection(face_roi, landmarks, i)
                            # Cache result
                            face_data['cached_result'] = result
                        else:
                            # Gunakan cached result
                            result = face_data.get('cached_result')
                            if result is None:
                                # Fallback jika tidak ada cache
                                result = self.advanced_spoof_detection(face_roi, landmarks, i)
                                face_data['cached_result'] = result
                        
                        is_real = result['is_real']
                        confidence = result['confidence']
                        method = result['method']
                        
                        # Update statistics
                        if is_real:
                            self.real_count += 1
                            label = f"ASLI ({confidence:.2f})"
                            color = (0, 255, 0)  # Green
                            status_text = f"WAJAH ASLI TERDETEKSI (Score: {confidence:.2f})"
                            status_color = (0, 255, 0)
                        else:
                            self.fake_count += 1
                            label = f"PALSU ({confidence:.2f})"
                            color = (0, 0, 255)  # Red
                            status_text = f"WAJAH PALSU TERDETEKSI (Score: {confidence:.2f})"
                            status_color = (0, 0, 255)
                        
                        # Draw results
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                        cv2.putText(frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        # Additional info
                        if debug_mode:
                            cv2.putText(frame, f"Texture: {result['texture_score']:.0f}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            cv2.putText(frame, f"Edge: {result['edge_strength']:.0f}", (x, y+h+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            cv2.putText(frame, f"Temporal: {result['temporal_score']:.2f}", (x, y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Draw landmarks
                        if landmarks is not None and debug_mode:
                            for landmark in landmarks:
                                cv2.circle(frame, tuple(landmark.astype(int)), 2, (255, 255, 0), -1)
                        
                        # Log result
                        logger.info(f"Frame {self.frame_count}: {'ASLI' if is_real else 'PALSU'} - "
                                  f"Score: {confidence:.3f}, Texture: {result['texture_score']:.0f}")
            
            # Draw UI
            self.draw_advanced_ui(frame, fps, status_text, status_color, debug_mode, show_history)
            
            # Show frame
            cv2.imshow("Advanced Anti-Spoofing Detection", frame)
            
            # Handle key presses dengan timeout minimal untuk realtime
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Aplikasi dihentikan oleh user")
                break
            elif key == ord('s'):
                screenshot_name = f"advanced_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_name, frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
                print(f"Screenshot disimpan: {screenshot_name}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                logger.info(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('h'):
                show_history = not show_history
                logger.info(f"History display: {'ON' if show_history else 'OFF'}")
            elif key == ord('r'):
                self.reset_statistics()
            elif key == ord('t'):
                self.adjust_threshold()
    
    def draw_advanced_ui(self, frame, fps, status_text, status_color, debug_mode, show_history):
        """Draw advanced UI elements"""
        h, w = frame.shape[:2]
        
        # Main info panel
        cv2.rectangle(frame, (5, 5), (450, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (450, 200), (255, 255, 255), 2)
        
        # Performance info
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection statistics
        cv2.putText(frame, f"Deteksi Total: {self.detection_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Real: {self.real_count}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Fake: {self.fake_count}", (120, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Accuracy
        if self.detection_count > 0:
            real_pct = (self.real_count / self.detection_count) * 100
            fake_pct = (self.fake_count / self.detection_count) * 100
            cv2.putText(frame, f"Real: {real_pct:.1f}% | Fake: {fake_pct:.1f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # System info
        cv2.putText(frame, f"Engine: InsightFace", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"History: {len(self.face_history)}/{self.max_history}", (200, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mode indicators
        if debug_mode:
            cv2.putText(frame, "DEBUG", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if show_history:
            cv2.putText(frame, "HISTORY", (w-100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Status
        cv2.putText(frame, status_text, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Controls
        cv2.putText(frame, "q=quit s=screenshot d=debug r=reset t=threshold h=history", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    def reset_statistics(self):
        """Reset statistik"""
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        self.face_history.clear()
        logger.info("Statistik dan history direset")
        print("Statistik direset!")
    
    def adjust_threshold(self):
        """Adjust confidence threshold"""
        print(f"\nCurrent threshold: {self.confidence_threshold}")
        print("Rekomendasi:")
        print("- 0.4: Lebih sensitif (lebih banyak deteksi fake)")
        print("- 0.6: Balanced (default)")
        print("- 0.8: Konservatif (lebih percaya real)")
        
        try:
            new_threshold = float(input("Masukkan threshold baru (0.0-1.0): "))
            if 0.0 <= new_threshold <= 1.0:
                self.confidence_threshold = new_threshold
                logger.info(f"Threshold updated to: {self.confidence_threshold}")
                print(f"Threshold diupdate ke: {self.confidence_threshold}")
            else:
                print("Threshold harus antara 0.0 dan 1.0")
        except ValueError:
            print("Input tidak valid")
        except KeyboardInterrupt:
            print("\nBatal mengubah threshold")
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("=== STATISTIK FINAL ===")
        logger.info(f"Total frames: {self.frame_count}")
        logger.info(f"Total deteksi: {self.detection_count}")
        logger.info(f"Wajah asli: {self.real_count}")
        logger.info(f"Wajah palsu: {self.fake_count}")
        
        if self.detection_count > 0:
            real_percentage = (self.real_count / self.detection_count) * 100
            fake_percentage = (self.fake_count / self.detection_count) * 100
            logger.info(f"Persentase asli: {real_percentage:.1f}%")
            logger.info(f"Persentase palsu: {fake_percentage:.1f}%")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    print("=== ADVANCED ANTI-SPOOFING SYSTEM ===")
    print("Powered by InsightFace & Advanced Computer Vision")
    print("Initializing...")
    
    try:
        detector = AdvancedAntiSpoofingDetector()
        detector.run()
    except KeyboardInterrupt:
        logger.info("Aplikasi dihentikan dengan Ctrl+C")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            detector.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Advanced Anti-Spoofing System menggunakan InsightFace dan Deep Learning
"""

import cv2
import numpy as np
import time
import logging
import os
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

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

class SimpleAntiSpoofNet(nn.Module):
    """Simple CNN untuk anti-spoofing detection"""
    def __init__(self):
        super(SimpleAntiSpoofNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)  # real vs fake
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class AdvancedAntiSpoofingDetector:
    def __init__(self):
        logger.info("Inisialisasi Advanced Anti-Spoofing Detector...")
        
        # Statistik
        self.frame_count = 0
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        self.confidence_threshold = 0.7
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Tidak dapat membuka webcam")
        
        # Set optimal webcam settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Initialize InsightFace
        self.init_insightface()
        
        # Initialize anti-spoofing model
        self.init_antispoofing_model()
        
        # Transform untuk preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Detector berhasil diinisialisasi!")
        
    def init_insightface(self):
        """Initialize InsightFace untuk deteksi wajah yang akurat"""
        try:
            logger.info("Memuat InsightFace models...")
            self.face_app = FaceAnalysis(
                providers=['CPUExecutionProvider'],  # Gunakan GPU jika tersedia: ['CUDAExecutionProvider', 'CPUExecutionProvider']
                allowed_modules=['detection', 'recognition']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✓ InsightFace berhasil dimuat")
            
        except Exception as e:
            logger.error(f"Error loading InsightFace: {e}")
            # Fallback ke OpenCV
            self.face_app = None
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("✓ Menggunakan OpenCV Haar Cascade sebagai fallback")
    
    def init_antispoofing_model(self):
        """Initialize model anti-spoofing"""
        try:
            # Coba load pre-trained model jika ada
            model_path = "antispoofing_model.pth"
            if os.path.exists(model_path):
                logger.info("Memuat pre-trained anti-spoofing model...")
                self.antispoof_model = SimpleAntiSpoofNet()
                self.antispoof_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.antispoof_model.eval()
                self.use_pretrained_model = True
                logger.info("✓ Pre-trained model berhasil dimuat")
            else:
                logger.info("Pre-trained model tidak ditemukan, menggunakan heuristic analysis")
                self.antispoof_model = None
                self.use_pretrained_model = False
                
        except Exception as e:
            logger.error(f"Error loading anti-spoofing model: {e}")
            self.antispoof_model = None
            self.use_pretrained_model = False
    
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
                if w > 50 and h > 50:
                    face_boxes.append({
                        'bbox': (x, y, w, h),
                        'confidence': face.det_score,
                        'embedding': face.embedding,
                        'landmarks': face.kps
                    })
            
            return face_boxes
            
        except Exception as e:
            logger.debug(f"InsightFace detection error: {e}")
            return self.detect_faces_opencv(frame)
    
    def detect_faces_opencv(self, frame):
        """Fallback detection menggunakan OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append({
                'bbox': (x, y, w, h),
                'confidence': 0.8,  # Default confidence
                'embedding': None,
                'landmarks': None
            })
        
        return face_boxes
    
    def advanced_texture_analysis(self, face_roi):
        """Advanced texture analysis untuk spoof detection"""
        if face_roi.size == 0:
            return 0.0, {}
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian Variance (sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Sobel Edge Density
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_density = np.mean(sobel_magnitude)
        
        # 3. Local Binary Pattern variance
        def get_lbp_variance(image):
            h, w = image.shape
            lbp = np.zeros_like(image)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = image[i, j]
                    code = 0
                    code |= (image[i-1, j-1] >= center) << 7
                    code |= (image[i-1, j] >= center) << 6
                    code |= (image[i-1, j+1] >= center) << 5
                    code |= (image[i, j+1] >= center) << 4
                    code |= (image[i+1, j+1] >= center) << 3
                    code |= (image[i+1, j] >= center) << 2
                    code |= (image[i+1, j-1] >= center) << 1
                    code |= (image[i, j-1] >= center) << 0
                    lbp[i, j] = code
            return np.var(lbp)
        
        lbp_var = get_lbp_variance(gray)
        
        # 4. Color analysis
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        color_variance = np.var(hsv[:,:,1])  # Saturation variance
        
        # 5. Frequency domain analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        freq_energy = np.mean(magnitude_spectrum)
        
        # Combine features
        features = {
            'laplacian_var': laplacian_var,
            'edge_density': edge_density,
            'lbp_variance': lbp_var,
            'color_variance': color_variance,
            'freq_energy': freq_energy
        }
        
        # Weighted score (tuned for better performance)
        score = (
            laplacian_var * 0.3 +
            edge_density * 0.2 +
            lbp_var * 0.2 +
            color_variance * 0.15 +
            freq_energy * 0.15
        )
        
        return score, features
    
    def predict_with_model(self, face_roi):
        """Prediksi menggunakan deep learning model"""
        if not self.use_pretrained_model or self.antispoof_model is None:
            return None
        
        try:
            # Preprocessing
            pil_image = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                outputs = self.antispoof_model(input_tensor)
                probabilities = outputs[0]
                fake_prob = probabilities[0].item()
                real_prob = probabilities[1].item()
                
            return {
                'is_real': real_prob > fake_prob,
                'confidence': max(real_prob, fake_prob),
                'real_prob': real_prob,
                'fake_prob': fake_prob
            }
            
        except Exception as e:
            logger.debug(f"Model prediction error: {e}")
            return None
    
    def analyze_face_liveness(self, face_roi, landmarks=None):
        """Advanced liveness detection"""
        # 1. Model-based prediction
        model_result = self.predict_with_model(face_roi)
        
        # 2. Texture analysis
        texture_score, texture_features = self.advanced_texture_analysis(face_roi)
        
        # 3. Landmark-based analysis (jika tersedia)
        landmark_score = 1.0
        if landmarks is not None:
            # Analisis simetri dan proporsi landmark
            landmarks = landmarks.astype(int)
            if len(landmarks) >= 5:  # 5 point landmarks
                # Hitung simetri wajah
                left_eye = landmarks[0]
                right_eye = landmarks[1]
                nose = landmarks[2]
                
                eye_distance = np.linalg.norm(left_eye - right_eye)
                face_width = face_roi.shape[1]
                
                # Real faces typically have proportional features
                proportion_score = eye_distance / face_width
                if 0.2 < proportion_score < 0.4:
                    landmark_score = 1.2
                else:
                    landmark_score = 0.8
        
        # Combine results
        if model_result is not None:
            # Prioritize model result if available
            final_confidence = model_result['confidence'] * 0.7 + (texture_score / 500) * 0.3
            is_real = model_result['is_real'] and texture_score > 80
            method = "Deep Learning + Texture"
        else:
            # Use heuristic approach
            # Improved thresholds based on testing
            is_real = texture_score > 120 and texture_features['edge_density'] > 15
            final_confidence = min(texture_score / 200, 1.0) * landmark_score
            method = "Advanced Heuristic"
        
        return {
            'is_real': is_real,
            'confidence': final_confidence,
            'texture_score': texture_score,
            'texture_features': texture_features,
            'method': method,
            'model_result': model_result
        }
    
    def run(self):
        """Main detection loop"""
        logger.info("Memulai Advanced Anti-Spoofing Detection...")
        print("\n=== ADVANCED ANTI-SPOOFING DETECTOR ===")
        print("Menggunakan InsightFace + Advanced Analysis")
        print("Kontrol:")
        print("- 'q': Keluar")
        print("- 's': Screenshot")
        print("- 'd': Toggle debug mode")
        print("- 'r': Reset statistik")
        print("- 't': Adjust threshold")
        print("==========================================\n")
        
        prev_frame_time = 0
        debug_mode = False
        show_features = False
        
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
            
            # Detect faces
            faces = self.detect_faces_insightface(frame)
            
            if faces:
                self.detection_count += 1
                
                for i, face_data in enumerate(faces):
                    x, y, w, h = face_data['bbox']
                    confidence = face_data['confidence']
                    landmarks = face_data['landmarks']
                    
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if face_roi.size > 0:
                        # Analyze liveness
                        result = self.analyze_face_liveness(face_roi, landmarks)
                        
                        is_real = result['is_real']
                        final_confidence = result['confidence']
                        method = result['method']
                        texture_score = result['texture_score']
                        
                        # Update statistics
                        if is_real:
                            self.real_count += 1
                            label = f"ASLI ({final_confidence:.1%})"
                            color = (0, 255, 0)  # Green
                            status_text = f"WAJAH ASLI TERDETEKSI ({method})"
                            status_color = (0, 255, 0)
                        else:
                            self.fake_count += 1
                            label = f"PALSU ({final_confidence:.1%})"
                            color = (0, 0, 255)  # Red
                            status_text = f"WAJAH PALSU TERDETEKSI ({method})"
                            status_color = (0, 0, 255)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                        
                        # Labels
                        cv2.putText(frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.putText(frame, f"Score: {texture_score:.0f}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, method, (x, y+h+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Draw landmarks if available
                        if landmarks is not None and debug_mode:
                            for landmark in landmarks:
                                cv2.circle(frame, tuple(landmark.astype(int)), 2, (255, 255, 0), -1)
                        
                        # Log result
                        logger.info(f"Frame {self.frame_count}: {'ASLI' if is_real else 'PALSU'} - "
                                  f"Confidence: {final_confidence:.1%}, Score: {texture_score:.0f}, Method: {method}")
            
            # Draw UI
            self.draw_advanced_ui(frame, fps, status_text, status_color, debug_mode, show_features)
            
            # Show frame
            cv2.imshow("Advanced Anti-Spoofing Detection", frame)
            
            # Handle key presses
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
            elif key == ord('f'):
                show_features = not show_features
                logger.info(f"Feature display: {'ON' if show_features else 'OFF'}")
            elif key == ord('r'):
                self.reset_statistics()
                logger.info("Statistik direset")
            elif key == ord('t'):
                self.adjust_threshold()
    
    def draw_advanced_ui(self, frame, fps, status_text, status_color, debug_mode, show_features):
        """Draw advanced UI elements"""
        h, w = frame.shape[:2]
        
        # Background untuk info
        cv2.rectangle(frame, (5, 5), (400, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (400, 180), (255, 255, 255), 2)
        
        # Top info
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Deteksi: {self.detection_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Real: {self.real_count} | Fake: {self.fake_count}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Model info
        model_info = "Deep Learning" if self.use_pretrained_model else "Heuristic"
        cv2.putText(frame, f"Mode: {model_info}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Engine: InsightFace", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if debug_mode:
            cv2.putText(frame, "DEBUG MODE", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Status
        cv2.putText(frame, status_text, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Controls
        cv2.putText(frame, "q=quit s=screenshot d=debug r=reset t=threshold", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def reset_statistics(self):
        """Reset statistik"""
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        print("Statistik direset!")
    
    def adjust_threshold(self):
        """Adjust confidence threshold"""
        print(f"Current threshold: {self.confidence_threshold}")
        try:
            new_threshold = float(input("Enter new threshold (0.0-1.0): "))
            if 0.0 <= new_threshold <= 1.0:
                self.confidence_threshold = new_threshold
                print(f"Threshold updated to: {self.confidence_threshold}")
            else:
                print("Invalid threshold. Must be between 0.0 and 1.0")
        except ValueError:
            print("Invalid input")
    
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
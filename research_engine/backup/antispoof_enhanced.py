import cv2
import time
import logging
import numpy as np
from deepface import DeepFace
import os

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('antispoof_enhanced_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AntiSpoofingDetector:
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.spoof_count = 0
        self.real_count = 0
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Tidak dapat membuka webcam")
            raise Exception("Webcam tidak tersedia")
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("AntiSpoofing Detector initialized")
        
    def detect_faces_basic(self, frame):
        """Basic face detection using OpenCV Haar Cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def analyze_frame_texture(self, face_roi):
        """Simple texture analysis for basic spoof detection"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (sharpness measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate LBP-like texture measure
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Simple heuristic: real faces typically have more texture variation
        texture_score = laplacian_var * (std_intensity / (mean_intensity + 1))
        
        return texture_score
    
    def detect_with_deepface(self, frame):
        """Enhanced DeepFace detection with better error handling"""
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # DeepFace expects BGR, which is what OpenCV uses
                pass
                
            # Ensure the image is in the right format
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Check if image is not empty
            if frame.size == 0:
                logger.warning("Empty frame received")
                return None
                
            logger.debug("Starting DeepFace analysis...")
            
            # Try different backends if default fails
            backends = ['opencv', 'retinaface', 'mtcnn', 'ssd']
            
            for backend in backends:
                try:
                    results = DeepFace.analyze(
                        img_path=frame, 
                        actions=['spoof'], 
                        enforce_detection=False,
                        silent=True,
                        detector_backend=backend
                    )
                    
                    if results:
                        logger.info(f"DeepFace analysis successful with {backend} backend")
                        return results
                        
                except Exception as be:
                    logger.debug(f"Backend {backend} failed: {str(be)}")
                    continue
            
            logger.warning("All DeepFace backends failed")
            return None
            
        except Exception as e:
            logger.error(f"DeepFace analysis error: {str(e)}")
            return None
    
    def run(self):
        """Main detection loop"""
        logger.info("Starting anti-spoofing detection...")
        print("\n=== ENHANCED ANTI-SPOOFING DETECTOR ===")
        print("Kontrol:")
        print("- 'q': Keluar")
        print("- 's': Screenshot")
        print("- 'd': Toggle debug mode")
        print("==========================================\n")
        
        prev_frame_time = 0
        debug_mode = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Gagal membaca frame")
                break
                
            self.frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            # Initialize status
            status_text = "Mencari wajah..."
            status_color = (255, 255, 0)  # Yellow
            
            # Calculate FPS
            new_frame_time = time.time()
            if prev_frame_time != 0:
                fps = 1 / (new_frame_time - prev_frame_time)
            else:
                fps = 0
            prev_frame_time = new_frame_time
            
            # Try DeepFace first
            deepface_results = None
            if self.frame_count % 5 == 0:  # Analyze every 5th frame for performance
                deepface_results = self.detect_with_deepface(frame)
            
            # Fallback to basic detection
            basic_faces = self.detect_faces_basic(frame)
            
            # Process results
            faces_detected = False
            
            if deepface_results and len(deepface_results) > 0:
                # Process DeepFace results
                faces_detected = True
                self.detection_count += 1
                
                for i, face_result in enumerate(deepface_results):
                    region = face_result['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    
                    spoof_data = face_result['spoof']
                    is_real = spoof_data['is_real']
                    confidence = spoof_data['confidence']
                    
                    # Update counters
                    if is_real:
                        self.real_count += 1
                        label = f"ASLI ({confidence:.1%})"
                        color = (0, 255, 0)  # Green
                        status_text = "WAJAH ASLI TERDETEKSI"
                        status_color = (0, 255, 0)
                    else:
                        self.spoof_count += 1
                        label = f"PALSU ({confidence:.1%})"
                        color = (0, 0, 255)  # Red
                        status_text = "WAJAH PALSU TERDETEKSI!"
                        status_color = (0, 0, 255)
                    
                    # Draw results
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    logger.info(f"Frame {self.frame_count}: {'ASLI' if is_real else 'PALSU'} - {confidence:.1%}")
                    
            elif len(basic_faces) > 0:
                # Process basic face detection with simple anti-spoofing
                faces_detected = True
                
                for (x, y, w, h) in basic_faces:
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if face_roi.size > 0:
                        # Simple texture analysis
                        texture_score = self.analyze_frame_texture(face_roi)
                        
                        # Simple threshold (you may need to adjust this)
                        is_real = texture_score > 100  # Adjust threshold as needed
                        
                        if is_real:
                            label = f"MUNGKIN ASLI (Score: {texture_score:.0f})"
                            color = (0, 255, 255)  # Cyan (uncertain)
                            status_text = "Wajah terdeteksi (analisis sederhana)"
                            status_color = (0, 255, 255)
                        else:
                            label = f"MUNGKIN PALSU (Score: {texture_score:.0f})"
                            color = (0, 165, 255)  # Orange (uncertain)
                            status_text = "Kemungkinan spoof (analisis sederhana)"
                            status_color = (0, 165, 255)
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, "BASIC MODE", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw UI
            self.draw_ui(frame, fps, status_text, status_color, debug_mode)
            
            # Show frame
            cv2.imshow("Enhanced Anti-Spoofing Detection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Aplikasi dihentikan oleh user")
                break
            elif key == ord('s'):
                screenshot_name = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_name, frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                logger.info(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    def draw_ui(self, frame, fps, status_text, status_color, debug_mode):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Top info
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Deteksi: {self.detection_count}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Statistics
        if debug_mode:
            cv2.putText(frame, f"Real: {self.real_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Spoof: {self.spoof_count}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, "DEBUG MODE", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Status
        cv2.putText(frame, status_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Controls
        cv2.putText(frame, "q=quit, s=screenshot, d=debug", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info(f"Statistik Final:")
        logger.info(f"- Total frames: {self.frame_count}")
        logger.info(f"- Total deteksi: {self.detection_count}")
        logger.info(f"- Wajah asli: {self.real_count}")
        logger.info(f"- Wajah palsu: {self.spoof_count}")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        detector = AntiSpoofingDetector()
        detector.run()
    except KeyboardInterrupt:
        logger.info("Aplikasi dihentikan dengan Ctrl+C")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        try:
            detector.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()
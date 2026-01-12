#!/usr/bin/env python3
"""
Simple Mobile Anti-Spoofing System
Version 1.0 - Pendekatan sederhana untuk testing photo vs real face
Menggunakan metode yang mirip dengan sistem mobile device authentication
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMobileAntiSpoof:
    def __init__(self):
        """Initialize Simple Mobile Anti-Spoofing"""
        logger.info("🚀 Simple Mobile Anti-Spoofing System...")
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Thresholds (tuned for real vs photo detection)
        self.texture_threshold = 30    # Lower threshold for texture
        self.motion_threshold = 1.5    # Lower threshold for motion
        self.brightness_threshold = 0.15  # Brightness variation threshold
        
        # History
        self.motion_history = []
        self.texture_history = []
        self.face_positions = []
        
        logger.info("✓ Ready!")

    def calculate_texture_score(self, face_region):
        """Calculate texture richness - real faces have more texture than photos"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            
            # Method 1: Laplacian variance (edge detection)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_var = laplacian.var()
            
            # Method 2: Standard deviation of pixel values
            pixel_std = np.std(gray)
            
            # Method 3: Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            gradient_mean = np.mean(gradient_mag)
            
            # Combined score
            texture_score = (texture_var * 0.4 + pixel_std * 0.3 + gradient_mean * 0.3)
            
            return texture_score
            
        except Exception as e:
            logger.debug(f"Texture calculation error: {e}")
            return 0

    def calculate_brightness_variation(self, face_region):
        """Calculate brightness variation across face regions"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            
            # Divide face into regions and calculate brightness variation
            h, w = gray.shape
            regions = [
                gray[0:h//2, 0:w//2],          # Top-left
                gray[0:h//2, w//2:w],          # Top-right
                gray[h//2:h, 0:w//2],          # Bottom-left
                gray[h//2:h, w//2:w]           # Bottom-right
            ]
            
            region_means = [np.mean(region) for region in regions]
            brightness_std = np.std(region_means)
            
            return brightness_std / 255.0  # Normalize
            
        except Exception as e:
            logger.debug(f"Brightness calculation error: {e}")
            return 0

    def analyze_face(self, face_region, face_center):
        """Analyze face for real vs fake determination"""
        results = {}
        
        # 1. Texture Analysis
        texture_score = self.calculate_texture_score(face_region)
        self.texture_history.append(texture_score)
        if len(self.texture_history) > 5:
            self.texture_history.pop(0)
        
        avg_texture = np.mean(self.texture_history)
        texture_real = avg_texture > self.texture_threshold
        results['texture'] = {'score': avg_texture, 'is_real': texture_real}
        
        # 2. Motion Analysis
        if len(self.face_positions) > 0:
            prev_pos = self.face_positions[-1]
            motion = np.sqrt((face_center[0] - prev_pos[0])**2 + (face_center[1] - prev_pos[1])**2)
            self.motion_history.append(motion)
            if len(self.motion_history) > 10:
                self.motion_history.pop(0)
        
        self.face_positions.append(face_center)
        if len(self.face_positions) > 10:
            self.face_positions.pop(0)
        
        if len(self.motion_history) >= 3:
            avg_motion = np.mean(self.motion_history)
            motion_real = avg_motion > self.motion_threshold
            results['motion'] = {'score': avg_motion, 'is_real': motion_real}
        else:
            results['motion'] = {'score': 0, 'is_real': None}
        
        # 3. Brightness Variation
        brightness_var = self.calculate_brightness_variation(face_region)
        brightness_real = brightness_var > self.brightness_threshold
        results['brightness'] = {'score': brightness_var, 'is_real': brightness_real}
        
        return results

    def make_decision(self, analysis_results):
        """Make final decision based on analysis"""
        votes = []
        scores = []
        reasons = []
        
        # Texture vote (most important for photo detection)
        if 'texture' in analysis_results:
            texture_data = analysis_results['texture']
            votes.append(texture_data['is_real'])
            scores.append(texture_data['score'] / 100.0)  # Normalize
            reasons.append(f"Texture: {texture_data['score']:.1f} ({'✓' if texture_data['is_real'] else '✗'})")
        
        # Motion vote
        if 'motion' in analysis_results and analysis_results['motion']['is_real'] is not None:
            motion_data = analysis_results['motion']
            votes.append(motion_data['is_real'])
            scores.append(min(motion_data['score'] / 10.0, 1.0))  # Normalize
            reasons.append(f"Motion: {motion_data['score']:.1f} ({'✓' if motion_data['is_real'] else '✗'})")
        
        # Brightness vote
        if 'brightness' in analysis_results:
            brightness_data = analysis_results['brightness']
            votes.append(brightness_data['is_real'])
            scores.append(brightness_data['score'] * 5)  # Scale up
            reasons.append(f"Brightness: {brightness_data['score']:.3f} ({'✓' if brightness_data['is_real'] else '✗'})")
        
        if not votes:
            return "PROCESSING", 0.0, ["No data"]
        
        # Decision based on majority vote with texture having more weight
        real_votes = sum(1 for v in votes if v)
        fake_votes = sum(1 for v in votes if not v)
        
        # Give texture analysis double weight
        if 'texture' in analysis_results and analysis_results['texture']['is_real']:
            real_votes += 1
        elif 'texture' in analysis_results and not analysis_results['texture']['is_real']:
            fake_votes += 1
        
        if real_votes > fake_votes:
            decision = "REAL"
            confidence = np.mean([s for i, s in enumerate(scores) if votes[i]]) if any(votes) else 0.5
        else:
            decision = "FAKE"  
            confidence = np.mean([s for i, s in enumerate(scores) if not votes[i]]) if any(not v for v in votes) else 0.5
        
        # Boost confidence for clear cases
        if abs(real_votes - fake_votes) >= 2:
            confidence = min(confidence * 1.2, 1.0)
        
        return decision, confidence, reasons

    def run_detection(self):
        """Run detection with camera"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera")
            return
        
        print("\n" + "="*70)
        print("    📱 SIMPLE MOBILE ANTI-SPOOFING v1.0")
        print("="*70)
        print("Test Instructions:")
        print("1. Show your real face - should detect as REAL")
        print("2. Show a photo of yourself - should detect as FAKE")
        print("3. Try different lighting conditions")
        print("4. Move slightly for better motion detection")
        print("\nControls: q=quit | r=reset")
        print("="*70)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                
                decision = "NO FACE"
                confidence = 0.0
                reasons = []
                
                if len(faces) > 0:
                    # Use largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    face_region = frame[y:y+h, x:x+w]
                    face_center = (x + w//2, y + h//2)
                    
                    # Analyze every few frames
                    if frame_count % 3 == 0:
                        analysis = self.analyze_face(face_region, face_center)
                        decision, confidence, reasons = self.make_decision(analysis)
                        
                        logger.info(f"Frame {frame_count}: {decision} (Conf: {confidence:.3f})")
                        logger.info(f"Reasons: {' | '.join(reasons)}")
                    
                    # Draw results
                    color = (0, 255, 0) if decision == "REAL" else (0, 0, 255) if decision == "FAKE" else (128, 128, 128)
                    
                    # Face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Status text
                    status_text = f"{decision}"
                    if confidence > 0:
                        status_text += f" ({confidence:.2f})"
                    
                    cv2.putText(frame, status_text, (x, y-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Reasons (debug info)
                    if reasons:
                        debug_y = y + h + 25
                        for reason in reasons[:2]:  # Show top 2 reasons
                            cv2.putText(frame, reason, (x, debug_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            debug_y += 18
                
                # Instructions
                cv2.putText(frame, "Show REAL face vs PHOTO to test", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Simple Mobile Anti-Spoofing', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset histories
                    self.motion_history.clear()
                    self.texture_history.clear()
                    self.face_positions.clear()
                    logger.info("🔄 Reset")
                    
        except KeyboardInterrupt:
            logger.info("🛑 Stopped")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = SimpleMobileAntiSpoof()
        detector.run_detection()
    except Exception as e:
        logger.error(f"Error: {e}")
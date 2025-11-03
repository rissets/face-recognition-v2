#!/usr/bin/env python3
"""
Test Anti-Spoofing dengan Wajah Asli
Test untuk memastikan sistem tidak menolak wajah asli
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_real_face.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealFaceTestDetector:
    def __init__(self):
        """Initialize Real Face Test Detector"""
        logger.info("Inisialisasi Real Face Test Detector...")
        
        # Initialize InsightFace
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info("✓ InsightFace berhasil dimuat")
        
        # Detection parameters - LEBIH LENIENT untuk real faces
        self.REAL_FACE_THRESHOLD = 0.60  # Lebih rendah untuk real faces
        self.texture_threshold = 20000    # Lebih tinggi untuk real faces
        self.edge_threshold = 0.40       # Lebih rendah untuk real faces
        self.color_threshold = 35        # Lebih rendah untuk real faces
        
        # Stats tracking
        self.frame_count = 0
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        
        logger.info("Real Face Test Detector berhasil diinisialisasi!")

    def analyze_texture_quality(self, face_region):
        """Analyze texture quality - optimized for real faces"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture variance (lower = smoother/fake, higher = natural texture)
            texture_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
            
            # Multi-scale texture analysis
            scales = [1.0, 0.5, 0.25]
            texture_scores = []
            
            for scale in scales:
                if scale != 1.0:
                    h, w = gray.shape
                    resized = cv2.resize(gray, (int(w*scale), int(h*scale)))
                    resized = cv2.resize(resized, (w, h))  # Back to original size
                else:
                    resized = gray
                
                # Laplacian variance for this scale
                lap_var = np.var(cv2.Laplacian(resized, cv2.CV_64F))
                texture_scores.append(lap_var)
            
            # Average texture score
            avg_texture = np.mean(texture_scores)
            
            return avg_texture
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            return 0

    def analyze_edge_quality(self, face_region):
        """Analyze edge quality - optimized for real faces"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density and sharpness
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Sobel gradients for edge strength
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
            
            # Combined edge quality (normalized)
            edge_quality = (edge_density * 1000 + edge_strength / 100) / 2
            
            return edge_quality
            
        except Exception as e:
            logger.error(f"Error in edge analysis: {e}")
            return 0

    def analyze_color_diversity(self, face_region):
        """Analyze color diversity - real faces have more natural variation"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Calculate color variance in each channel
            color_vars = []
            
            # HSV variance
            for i in range(3):
                color_vars.append(np.var(hsv[:, :, i]))
            
            # LAB variance  
            for i in range(3):
                color_vars.append(np.var(lab[:, :, i]))
            
            # Average color diversity
            color_diversity = np.mean(color_vars)
            
            return color_diversity
            
        except Exception as e:
            logger.error(f"Error in color analysis: {e}")
            return 0

    def detect_realface(self, face_region):
        """
        Detect if face is real - OPTIMIZED FOR REAL FACES
        Returns (confidence, is_real, debug_info)
        """
        try:
            # 1. Texture Analysis (real faces have natural texture)
            texture_score = self.analyze_texture_quality(face_region)
            texture_real = texture_score < self.texture_threshold  # Lower = more natural
            
            # 2. Edge Quality (real faces have sharper, natural edges)
            edge_quality = self.analyze_edge_quality(face_region)
            edge_real = edge_quality > self.edge_threshold  # Higher = sharper edges
            
            # 3. Color Diversity (real faces have more color variation)
            color_diversity = self.analyze_color_diversity(face_region)
            color_real = color_diversity > self.color_threshold  # Higher = more diverse
            
            # 4. Calculate overall confidence (LENIENT for real faces)
            confidence_factors = []
            
            # Texture confidence (inverted - lower texture = higher confidence for real)
            texture_conf = min(1.0, (self.texture_threshold - texture_score) / self.texture_threshold)
            confidence_factors.append(max(0.3, texture_conf))  # Minimum 0.3
            
            # Edge confidence
            edge_conf = min(1.0, edge_quality / (self.edge_threshold * 2))
            confidence_factors.append(max(0.3, edge_conf))
            
            # Color confidence
            color_conf = min(1.0, color_diversity / (self.color_threshold * 2))
            confidence_factors.append(max(0.3, color_conf))
            
            # Overall confidence (weighted average)
            overall_confidence = np.mean(confidence_factors)
            
            # Decision (LENIENT threshold for real faces)
            is_real = overall_confidence >= self.REAL_FACE_THRESHOLD
            
            debug_info = {
                'texture_score': texture_score,
                'edge_quality': edge_quality,
                'color_diversity': color_diversity,
                'texture_real': texture_real,
                'edge_real': edge_real,
                'color_real': color_real,
                'confidence': overall_confidence
            }
            
            return overall_confidence, is_real, debug_info
            
        except Exception as e:
            logger.error(f"Error in real face detection: {e}")
            return 0.0, False, {}

    def run_test(self):
        """Run real face test"""
        logger.info("Memulai Real Face Test...")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Tidak dapat membuka camera")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Camera berhasil diinisialisasi")
        
        print("\n" + "="*60)
        print("    REAL FACE ANTI-SPOOFING TEST")
        print("="*60)
        print("Optimized untuk:")
        print("✓ Test Real Face Detection")
        print("✓ Lenient Thresholds")
        print("✓ Detailed Analysis") 
        print("✓ Debug Information")
        print("\nControls:")
        print("q=quit | s=screenshot | d=debug | r=reset")
        print("="*60)
        
        show_debug = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Tidak dapat membaca frame dari camera")
                    continue
                
                self.frame_count += 1
                
                # Detect faces
                faces = self.app.get(frame)
                
                # Process each face
                for face in faces:
                    self.detection_count += 1
                    
                    # Extract face region with padding
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Add padding
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)
                    
                    face_region = frame[y1:y2, x1:x2]
                    
                    if face_region.size == 0:
                        continue
                    
                    # Detect if real face
                    confidence, is_real, debug_info = self.detect_realface(face_region)
                    
                    # Update stats
                    if is_real:
                        self.real_count += 1
                        result_text = "REAL"
                        color = (0, 255, 0)  # Green
                    else:
                        self.fake_count += 1
                        result_text = "FAKE"
                        color = (0, 0, 255)  # Red
                    
                    # Draw bounding box and result
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Result text
                    result_label = f"{result_text} ({confidence:.2f})"
                    cv2.putText(frame, result_label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Debug info
                    if show_debug and debug_info:
                        debug_y = y1 + 30
                        debug_texts = [
                            f"Texture: {debug_info.get('texture_score', 0):.0f}",
                            f"Edge: {debug_info.get('edge_quality', 0):.2f}",
                            f"Color: {debug_info.get('color_diversity', 0):.0f}"
                        ]
                        
                        for text in debug_texts:
                            cv2.putText(frame, text, (x1, debug_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            debug_y += 20
                    
                    # Log detection
                    logger.info(f"Frame {self.frame_count}: {result_text} - "
                              f"Conf: {confidence:.3f}, "
                              f"Texture: {debug_info.get('texture_score', 0):.0f}, "
                              f"Edge: {debug_info.get('edge_quality', 0):.2f}, "
                              f"Color: {debug_info.get('color_diversity', 0):.0f}")
                
                # Display info
                info_text = f"Frames: {self.frame_count} | Detections: {self.detection_count} | Real: {self.real_count} | Fake: {self.fake_count}"
                cv2.putText(frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Real Face Anti-Spoofing Test', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = int(time.time())
                    filename = f"real_face_test_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('d'):
                    show_debug = not show_debug
                    logger.info(f"Debug mode: {'ON' if show_debug else 'OFF'}")
                elif key == ord('r'):
                    # Reset stats
                    self.frame_count = 0
                    self.detection_count = 0
                    self.real_count = 0
                    self.fake_count = 0
                    logger.info("Stats reset")
        
        except KeyboardInterrupt:
            logger.info("Test dihentikan oleh user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Final summary
            logger.info("=== REAL FACE TEST SUMMARY ===")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Total detections: {self.detection_count}")
            logger.info(f"Real faces: {self.real_count}")
            logger.info(f"Fake faces: {self.fake_count}")
            
            if self.detection_count > 0:
                real_percentage = (self.real_count / self.detection_count) * 100
                fake_percentage = (self.fake_count / self.detection_count) * 100
                logger.info(f"Real percentage: {real_percentage:.1f}%")
                logger.info(f"Fake percentage: {fake_percentage:.1f}%")

if __name__ == "__main__":
    detector = RealFaceTestDetector()
    detector.run_test()
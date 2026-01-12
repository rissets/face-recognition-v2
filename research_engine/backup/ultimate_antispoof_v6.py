#!/usr/bin/env python3
"""
ULTIMATE Anti-Spoofing System v6.0 - Enhanced Edition
Fix untuk motion detection error dan meningkatkan akurasi deteksi fake faces
"""

import cv2
import numpy as np
import time
import logging
import insightface
from collections import deque
import argparse
import threading
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import math

@dataclass
class DetectionResult:
    """Result container untuk analisis anti-spoofing"""
    is_real: bool
    confidence: float
    metrics: Dict[str, float]
    frame_id: int

class UltimateAntiSpoofing:
    def __init__(self):
        self.setup_logging()
        
        # InsightFace setup
        self.logger.info("🚀 Inisialisasi ULTIMATE Anti-Spoofing System...")
        self.face_app = insightface.app.FaceAnalysis()
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))
        self.logger.info("✓ InsightFace initialized")
        
        # Adaptive thresholds - SEIMBANG untuk membedakan fake dan real
        self.thresholds = {
            'low_sensitivity': {
                'texture_min': 80,       # Lebih rendah untuk real faces
                'edge_min': 8.0,         # Lebih rendah untuk real faces
                'color_min': 1000,       # Lebih rendah untuk real faces
                'motion_min': 5.0        # Lebih rendah untuk real faces
            },
            'medium_sensitivity': {
                'texture_min': 100,      # Seimbang
                'edge_min': 12.0,        # Seimbang
                'color_min': 1300,       # Seimbang
                'motion_min': 8.0        # Seimbang
            },
            'high_sensitivity': {
                'texture_min': 130,      # Tinggi untuk deteksi fake
                'edge_min': 16.0,        # Tinggi untuk deteksi fake
                'color_min': 1600,       # Tinggi untuk deteksi fake
                'motion_min': 12.0       # Tinggi untuk deteksi fake
            }
        }
        
        # Current sensitivity
        self.current_sensitivity = 'medium_sensitivity'
        
        # Motion tracking - DIPERBAIKI
        self.motion_window = deque(maxlen=5)
        self.prev_gray = None
        self.motion_roi = None
        
        # Result smoothing
        self.result_window = deque(maxlen=7)  # Temporal smoothing
        self.frame_counter = 0
        
        # Performance metrics
        self.processing_times = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'face_detections': 0,
            'real_classifications': 0,
            'fake_classifications': 0
        }
        
        self.logger.info("✓ Ultimate Anti-Spoofing Detector ready!")

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_face_roi(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract face ROI dengan enhanced preprocessing"""
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Expand bbox dengan safety margin
            h, w = frame.shape[:2]
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
                
            # Resize untuk konsistensi analisis
            if face_roi.shape[0] < 64 or face_roi.shape[1] < 64:
                face_roi = cv2.resize(face_roi, (128, 128), interpolation=cv2.INTER_CUBIC)
            
            return face_roi
            
        except Exception as e:
            self.logger.error(f"Error extracting face ROI: {e}")
            return None

    def analyze_texture_quality(self, face_roi: np.ndarray) -> float:
        """Enhanced multi-scale texture analysis"""
        try:
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi.copy()
            
            # Multi-scale texture analysis
            texture_scores = []
            
            # Scale 1: Original size
            laplacian1 = cv2.Laplacian(gray, cv2.CV_64F)
            texture_scores.append(np.var(laplacian1))
            
            # Scale 2: Downsampled
            gray_small = cv2.resize(gray, (64, 64))
            laplacian2 = cv2.Laplacian(gray_small, cv2.CV_64F)
            texture_scores.append(np.var(laplacian2) * 0.7)  # Weight factor
            
            # Gabor filter untuk texture richness
            kernel = cv2.getGaborKernel((21, 21), 8, 0, 10, 0.5, 0, ktype=cv2.CV_32F)
            gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            texture_scores.append(np.std(gabor_response) * 2.0)
            
            # Median untuk robustness
            final_score = np.median(texture_scores)
            
            return float(final_score)
            
        except Exception as e:
            self.logger.error(f"Error in texture analysis: {e}")
            return 50.0

    def analyze_edge_quality(self, face_roi: np.ndarray) -> float:
        """Enhanced edge quality assessment"""
        try:
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi.copy()
            
            # Canny edges dengan multiple thresholds
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            edges3 = cv2.Canny(gray, 80, 200)
            
            # Edge density analysis
            edge_densities = [
                np.sum(edges1) / (gray.shape[0] * gray.shape[1] * 255) * 100,
                np.sum(edges2) / (gray.shape[0] * gray.shape[1] * 255) * 100,
                np.sum(edges3) / (gray.shape[0] * gray.shape[1] * 255) * 100
            ]
            
            # Gradient magnitude analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_score = np.mean(gradient_magnitude)
            
            # Combine scores
            edge_score = np.mean(edge_densities) + (gradient_score / 10.0)
            
            return float(edge_score)
            
        except Exception as e:
            self.logger.error(f"Error in edge analysis: {e}")
            return 8.0

    def analyze_color_diversity(self, face_roi: np.ndarray) -> float:
        """Enhanced color diversity dengan skin tone analysis"""
        try:
            if len(face_roi.shape) != 3:
                return 1000.0
            
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            
            # Color histogram analysis
            hist_b = cv2.calcHist([face_roi], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([face_roi], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([face_roi], [2], None, [32], [0, 256])
            
            # Histogram entropy
            def calculate_entropy(hist):
                hist = hist.flatten()
                hist = hist[hist > 0]
                hist = hist / np.sum(hist)
                return -np.sum(hist * np.log2(hist + 1e-7))
            
            entropy_score = (calculate_entropy(hist_b) + 
                           calculate_entropy(hist_g) + 
                           calculate_entropy(hist_r)) * 200
            
            # Color variance in different spaces
            bgr_variance = np.var(face_roi.reshape(-1, 3), axis=0).sum()
            hsv_variance = np.var(hsv.reshape(-1, 3), axis=0).sum() * 0.5
            lab_variance = np.var(lab.reshape(-1, 3), axis=0).sum() * 0.3
            
            # Skin tone consistency check
            skin_mask = self.create_skin_mask(face_roi)
            skin_consistency = np.sum(skin_mask) / (face_roi.shape[0] * face_roi.shape[1])
            skin_bonus = skin_consistency * 300  # Bonus untuk natural skin
            
            total_score = entropy_score + bgr_variance + hsv_variance + lab_variance + skin_bonus
            
            return float(total_score)
            
        except Exception as e:
            self.logger.error(f"Error in color analysis: {e}")
            return 1200.0

    def create_skin_mask(self, face_roi: np.ndarray) -> np.ndarray:
        """Create skin tone mask untuk natural skin detection"""
        try:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            
            # HSV skin range
            lower_hsv = np.array([0, 20, 70])
            upper_hsv = np.array([20, 255, 255])
            mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            # YCrCb skin range
            lower_ycrcb = np.array([0, 133, 77])
            upper_ycrcb = np.array([255, 173, 127])
            mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
            
            # Combine masks
            skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            return skin_mask
            
        except Exception as e:
            self.logger.error(f"Error creating skin mask: {e}")
            return np.zeros((face_roi.shape[0], face_roi.shape[1]), dtype=np.uint8)

    def analyze_motion_liveness(self, face_roi: np.ndarray, bbox: np.ndarray) -> float:
        """Enhanced motion analysis dengan proper error handling"""
        try:
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi.copy()
            
            # Resize untuk konsistensi
            target_size = (64, 64)
            gray_resized = cv2.resize(gray, target_size)
            
            if self.prev_gray is None or self.prev_gray.shape != gray_resized.shape:
                self.prev_gray = gray_resized.copy()
                return 10.0  # Default motion score
            
            # Motion calculation dengan error handling
            try:
                # Optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray_resized, 
                    np.array([[32, 32]], dtype=np.float32), 
                    None,
                    winSize=(15, 15),
                    maxLevel=2
                )
                
                if flow[0] is not None and len(flow[0]) > 0:
                    motion_magnitude = np.linalg.norm(flow[0][0] - np.array([32, 32]))
                else:
                    motion_magnitude = 0.0
                    
            except Exception:
                # Fallback: frame difference
                diff = cv2.absdiff(self.prev_gray, gray_resized)
                motion_magnitude = np.mean(diff) / 10.0
            
            # Update previous frame
            self.prev_gray = gray_resized.copy()
            
            # Motion smoothing
            self.motion_window.append(motion_magnitude)
            
            if len(self.motion_window) >= 3:
                # Temporal motion analysis
                motion_changes = [abs(self.motion_window[i] - self.motion_window[i-1]) 
                                for i in range(1, len(self.motion_window))]
                motion_variability = np.std(motion_changes) if motion_changes else 0.0
                
                # Natural motion bonus
                natural_motion = motion_variability * 5.0
                avg_motion = np.mean(list(self.motion_window))
                
                total_motion = avg_motion + natural_motion
            else:
                total_motion = motion_magnitude
            
            return float(total_motion)
            
        except Exception as e:
            self.logger.error(f"Error in motion analysis: {e}")
            return 8.0

    def make_decision(self, texture: float, edge: float, color: float, motion: float) -> Tuple[bool, float]:
        """Enhanced decision making dengan balanced thresholds"""
        current_thresh = self.thresholds[self.current_sensitivity]
        
        # Individual scores (0-1 scale)
        texture_score = min(1.0, texture / current_thresh['texture_min'])
        edge_score = min(1.0, edge / current_thresh['edge_min'])
        color_score = min(1.0, color / current_thresh['color_min'])
        motion_score = min(1.0, motion / current_thresh['motion_min'])
        
        # Weighted scoring - BALANCED
        weights = {
            'texture': 0.30,    # Texture penting
            'edge': 0.25,       # Edge quality penting
            'color': 0.30,      # Color diversity sangat penting untuk screen detection
            'motion': 0.15      # Motion bonus
        }
        
        final_score = (
            texture_score * weights['texture'] +
            edge_score * weights['edge'] +
            color_score * weights['color'] +
            motion_score * weights['motion']
        )
        
        # BALANCED threshold - 0.65 untuk medium sensitivity
        confidence_thresholds = {
            'low_sensitivity': 0.55,    # Lebih mudah untuk REAL
            'medium_sensitivity': 0.65, # Balanced
            'high_sensitivity': 0.75    # Lebih ketat
        }
        
        confidence_threshold = confidence_thresholds[self.current_sensitivity]
        is_real = final_score >= confidence_threshold
        
        # Screen detection: Color diversity yang terlalu tinggi = screen
        # Wajah di layar biasanya punya color diversity sangat tinggi (>50000)
        if color > 50000:  # Screen detection threshold
            is_real = False
            final_score = min(final_score, 0.3)  # Cap untuk screen
        
        # Natural skin bonus - wajah asli punya skin tone natural
        if color < 5000 and texture > 50:  # Terlalu rendah = mungkin flat/printed
            is_real = False
            final_score = min(final_score, 0.4)
        
        return is_real, final_score

    def smooth_results(self, current_result: DetectionResult) -> DetectionResult:
        """Temporal smoothing untuk stability"""
        self.result_window.append(current_result)
        
        if len(self.result_window) < 3:
            return current_result
        
        # Majority voting untuk classification
        real_votes = sum(1 for r in self.result_window if r.is_real)
        fake_votes = len(self.result_window) - real_votes
        
        # Confidence averaging
        avg_confidence = np.mean([r.confidence for r in self.result_window])
        
        # Balanced decision - tidak terlalu bias
        sensitivity_thresholds = {
            'low_sensitivity': 0.5,     # Mudah untuk REAL
            'medium_sensitivity': 0.6,  # Balanced
            'high_sensitivity': 0.7     # Ketat untuk REAL
        }
        
        confidence_req = sensitivity_thresholds[self.current_sensitivity]
        is_real = real_votes >= fake_votes and avg_confidence >= confidence_req
        
        return DetectionResult(
            is_real=is_real,
            confidence=avg_confidence,
            metrics=current_result.metrics,
            frame_id=current_result.frame_id
        )

    def process_frame(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Process single frame untuk anti-spoofing analysis"""
        start_time = time.time()
        
        try:
            self.frame_counter += 1
            self.stats['total_frames'] += 1
            
            # Face detection
            faces = self.face_app.get(frame)
            
            if not faces:
                return None
                
            self.stats['face_detections'] += 1
            
            # Process first face
            face = faces[0]
            bbox = face.bbox
            
            # Extract face ROI
            face_roi = self.extract_face_roi(frame, bbox)
            if face_roi is None:
                return None
            
            # Multi-modal analysis
            texture_score = self.analyze_texture_quality(face_roi)
            edge_score = self.analyze_edge_quality(face_roi)
            color_score = self.analyze_color_diversity(face_roi)
            motion_score = self.analyze_motion_liveness(face_roi, bbox)
            
            # Decision making
            is_real, confidence = self.make_decision(texture_score, edge_score, color_score, motion_score)
            
            # Update statistics
            if is_real:
                self.stats['real_classifications'] += 1
            else:
                self.stats['fake_classifications'] += 1
            
            # Create result
            result = DetectionResult(
                is_real=is_real,
                confidence=confidence,
                metrics={
                    'texture': texture_score,
                    'edge': edge_score,
                    'color': color_score,
                    'motion': motion_score
                },
                frame_id=self.frame_counter
            )
            
            # Apply temporal smoothing
            smoothed_result = self.smooth_results(result)
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return smoothed_result
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return None

    def draw_results(self, frame: np.ndarray, result: DetectionResult, faces: list) -> np.ndarray:
        """Draw results pada frame"""
        if not faces:
            return frame
            
        face = faces[0]
        bbox = face.bbox.astype(int)
        
        # Colors
        color = (0, 255, 0) if result.is_real else (0, 0, 255)
        text_color = (255, 255, 255)
        
        # Draw bbox
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Status text
        status = "ASLI" if result.is_real else "PALSU"
        confidence_text = f"{status} ({result.confidence:.2f})"
        
        # Draw main text
        cv2.putText(frame, confidence_text, (bbox[0], bbox[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
        
        # Draw metrics
        metrics = result.metrics
        y_offset = bbox[3] + 30
        metrics_text = [
            f"Texture: {metrics['texture']:.0f}",
            f"Edge: {metrics['edge']:.1f}",
            f"Color: {metrics['color']:.0f}",
            f"Motion: {metrics['motion']:.1f}"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(frame, text, (bbox[0], y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
        return frame

    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements"""
        h, w = frame.shape[:2]
        
        # Header
        header_text = f"ULTIMATE Anti-Spoof v6.0 - {self.current_sensitivity.replace('_', ' ').title()}"
        cv2.putText(frame, header_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Statistics
        real_rate = (self.stats['real_classifications'] / max(self.stats['face_detections'], 1)) * 100
        fake_rate = (self.stats['fake_classifications'] / max(self.stats['face_detections'], 1)) * 100
        
        stats_text = [
            f"Frames: {self.stats['total_frames']}",
            f"Faces: {self.stats['face_detections']}",
            f"Real: {self.stats['real_classifications']} ({real_rate:.1f}%)",
            f"Fake: {self.stats['fake_classifications']} ({fake_rate:.1f}%)"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (w - 300, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Performance
        if self.processing_times:
            avg_time = np.mean(list(self.processing_times))
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return frame

    def run_realtime(self):
        """Main realtime detection loop"""
        self.logger.info("🎬 Starting Ultimate Anti-Spoofing Detection...")
        
        # Camera setup
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("❌ Cannot open camera")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.logger.info("✓ Camera initialized")
        
        print("\n" + "="*90)
        print("    🎯 ULTIMATE ANTI-SPOOFING REALTIME SYSTEM v6.0")
        print("="*90)
        print("Technologies:")
        print("✓ InsightFace Advanced Face Detection")
        print("✓ Enhanced Multi-Scale Texture Analysis")
        print("✓ Advanced Edge Quality Assessment")
        print("✓ Color Diversity & Skin Tone Analysis")
        print("✓ Improved Motion Detection & Liveness")
        print("✓ Stricter Adaptive Threshold System")
        print("✓ Intelligent Result Smoothing")
        print("\nControls:")
        print("q=quit | s=screenshot | d=debug | r=reset | 1-3=sensitivity")
        print("="*90)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process frame
                result = self.process_frame(frame)
                
                # Draw results
                if result:
                    faces = self.face_app.get(frame)
                    frame = self.draw_results(frame, result, faces)
                    
                    # Log every 10th frame
                    if self.frame_counter % 10 == 0:
                        status = "ASLI" if result.is_real else "PALSU"
                        metrics = result.metrics
                        self.logger.info(
                            f"Frame {self.frame_counter}: {status} - "
                            f"Conf: {result.confidence:.3f}, "
                            f"Texture: {metrics['texture']:.0f}, "
                            f"Edge: {metrics['edge']:.2f}, "
                            f"Color: {metrics['color']:.0f}"
                        )
                
                # Draw UI
                frame = self.draw_ui(frame)
                
                # Display
                cv2.imshow("Ultimate Anti-Spoofing v6.0", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"screenshot_{int(time.time())}.jpg", frame)
                    print("📸 Screenshot saved!")
                elif key == ord('r'):
                    self.stats = {'total_frames': 0, 'face_detections': 0, 
                                 'real_classifications': 0, 'fake_classifications': 0}
                    self.frame_counter = 0
                    print("🔄 Statistics reset!")
                elif key == ord('1'):
                    self.current_sensitivity = 'low_sensitivity'
                    print("📊 Sensitivity: LOW")
                elif key == ord('2'):
                    self.current_sensitivity = 'medium_sensitivity'
                    print("📊 Sensitivity: MEDIUM")
                elif key == ord('3'):
                    self.current_sensitivity = 'high_sensitivity'
                    print("📊 Sensitivity: HIGH")
                    
        except KeyboardInterrupt:
            self.logger.info("⏹️ Stopping detection...")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print session summary
            print("\n" + "="*50)
            print("=== ULTIMATE ANTISPOOFING SESSION SUMMARY ===")
            print(f"Total frames processed: {self.stats['total_frames']}")
            print(f"Total face detections: {self.stats['face_detections']}")
            print(f"Real face classifications: {self.stats['real_classifications']}")
            print(f"Fake face classifications: {self.stats['fake_classifications']}")
            
            if self.stats['face_detections'] > 0:
                real_rate = (self.stats['real_classifications'] / self.stats['face_detections']) * 100
                fake_rate = (self.stats['fake_classifications'] / self.stats['face_detections']) * 100
                print(f"Real classification rate: {real_rate:.1f}%")
                print(f"Fake classification rate: {fake_rate:.1f}%")
            
            if self.processing_times:
                avg_time = np.mean(list(self.processing_times))
                min_time = np.min(list(self.processing_times))
                max_time = np.max(list(self.processing_times))
                print(f"Processing time - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
                print(f"Theoretical max FPS: {1.0/avg_time:.1f}")
            
            print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Ultimate Anti-Spoofing System v6.0')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], 
                       default='medium', help='Detection sensitivity')
    
    args = parser.parse_args()
    
    # Create detector
    detector = UltimateAntiSpoofing()
    
    # Set sensitivity
    detector.current_sensitivity = f"{args.sensitivity}_sensitivity"
    
    # Run detection
    detector.run_realtime()

if __name__ == "__main__":
    main()
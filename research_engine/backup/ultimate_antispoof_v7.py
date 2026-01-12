#!/usr/bin/env python3
"""
ULTIMATE Anti-Spoofing System v7.0 - Adaptive Balanced Edition
Sistem yang dapat membedakan wajah asli dan fake dengan akurat
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
        self.logger.info("🚀 Inisialisasi ULTIMATE Anti-Spoofing System v7.0...")
        self.face_app = insightface.app.FaceAnalysis()
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))
        self.logger.info("✓ InsightFace initialized")
        
        # Adaptive baseline learning
        self.baseline_metrics = {
            'texture_samples': deque(maxlen=50),
            'edge_samples': deque(maxlen=50),
            'color_samples': deque(maxlen=50),
            'motion_samples': deque(maxlen=50),
            'learned': False
        }
        
        # Current sensitivity
        self.current_sensitivity = 'medium_sensitivity'
        
        # Motion tracking
        self.motion_window = deque(maxlen=5)
        self.prev_gray = None
        
        # Result smoothing
        self.result_window = deque(maxlen=5)  # Shorter window untuk responsiveness
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

    def learn_baseline(self, texture: float, edge: float, color: float, motion: float):
        """Learn baseline metrics dari environment normal"""
        self.baseline_metrics['texture_samples'].append(texture)
        self.baseline_metrics['edge_samples'].append(edge)
        self.baseline_metrics['color_samples'].append(color)
        self.baseline_metrics['motion_samples'].append(motion)
        
        # Mark as learned setelah 30 samples
        if len(self.baseline_metrics['texture_samples']) >= 30:
            self.baseline_metrics['learned'] = True
            
            # Calculate baseline stats
            self.baseline_stats = {
                'texture_mean': np.mean(list(self.baseline_metrics['texture_samples'])),
                'texture_std': np.std(list(self.baseline_metrics['texture_samples'])),
                'edge_mean': np.mean(list(self.baseline_metrics['edge_samples'])),
                'edge_std': np.std(list(self.baseline_metrics['edge_samples'])),
                'color_mean': np.mean(list(self.baseline_metrics['color_samples'])),
                'color_std': np.std(list(self.baseline_metrics['color_samples'])),
                'motion_mean': np.mean(list(self.baseline_metrics['motion_samples'])),
                'motion_std': np.std(list(self.baseline_metrics['motion_samples']))
            }
            
            self.logger.info("📊 Baseline metrics learned!")
            self.logger.info(f"Texture: {self.baseline_stats['texture_mean']:.0f}±{self.baseline_stats['texture_std']:.0f}")
            self.logger.info(f"Edge: {self.baseline_stats['edge_mean']:.1f}±{self.baseline_stats['edge_std']:.1f}")
            self.logger.info(f"Color: {self.baseline_stats['color_mean']:.0f}±{self.baseline_stats['color_std']:.0f}")

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
            
            # Scale 1: Original Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_scores.append(np.var(laplacian))
            
            # Scale 2: Local Binary Pattern simulation
            kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            texture_scores.append(np.var(response) * 0.5)
            
            # Scale 3: Gradient-based texture
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            texture_scores.append(np.std(gradient_magnitude))
            
            # Robust combination
            final_score = np.median(texture_scores)
            
            return float(final_score)
            
        except Exception as e:
            self.logger.error(f"Error in texture analysis: {e}")
            return 100.0

    def analyze_edge_quality(self, face_roi: np.ndarray) -> float:
        """Enhanced edge quality assessment"""
        try:
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi.copy()
            
            # Multi-threshold edge detection
            edges1 = cv2.Canny(gray, 50, 150)
            edges2 = cv2.Canny(gray, 80, 200)
            
            # Edge density
            edge_density1 = np.sum(edges1) / (gray.shape[0] * gray.shape[1] * 255) * 100
            edge_density2 = np.sum(edges2) / (gray.shape[0] * gray.shape[1] * 255) * 100
            
            # Average edge quality
            edge_score = (edge_density1 + edge_density2) / 2.0
            
            return float(edge_score)
            
        except Exception as e:
            self.logger.error(f"Error in edge analysis: {e}")
            return 10.0

    def analyze_color_diversity(self, face_roi: np.ndarray) -> float:
        """Simplified but effective color diversity analysis"""
        try:
            if len(face_roi.shape) != 3:
                return 2000.0
            
            # Simple color variance analysis
            bgr_var = np.var(face_roi.reshape(-1, 3), axis=0)
            total_variance = np.sum(bgr_var)
            
            # HSV variance untuk additional info
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            hsv_var = np.var(hsv.reshape(-1, 3), axis=0)
            hsv_score = np.sum(hsv_var) * 0.3
            
            # Combined score
            color_score = total_variance + hsv_score
            
            return float(color_score)
            
        except Exception as e:
            self.logger.error(f"Error in color analysis: {e}")
            return 2000.0

    def analyze_motion_liveness(self, face_roi: np.ndarray, bbox: np.ndarray) -> float:
        """Enhanced motion analysis"""
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
                return 10.0
            
            # Frame difference
            diff = cv2.absdiff(self.prev_gray, gray_resized)
            motion_magnitude = np.mean(diff)
            
            # Update previous frame
            self.prev_gray = gray_resized.copy()
            
            # Motion smoothing
            self.motion_window.append(motion_magnitude)
            
            if len(self.motion_window) >= 3:
                avg_motion = np.mean(list(self.motion_window))
                motion_std = np.std(list(self.motion_window))
                total_motion = avg_motion + motion_std * 2.0  # Variability bonus
            else:
                total_motion = motion_magnitude
            
            return float(total_motion)
            
        except Exception as e:
            self.logger.error(f"Error in motion analysis: {e}")
            return 10.0

    def make_adaptive_decision(self, texture: float, edge: float, color: float, motion: float) -> Tuple[bool, float]:
        """Adaptive decision making berdasarkan learned baseline"""
        
        if not self.baseline_metrics['learned']:
            # Learning phase - assume REAL untuk learning
            self.learn_baseline(texture, edge, color, motion)
            return True, 0.8  # Neutral confidence during learning
        
        # Get baseline stats
        baseline = self.baseline_stats
        
        # Calculate deviation scores (0-1, lower = more similar to baseline)
        texture_dev = abs(texture - baseline['texture_mean']) / max(baseline['texture_std'], 10)
        edge_dev = abs(edge - baseline['edge_mean']) / max(baseline['edge_std'], 1.0)
        color_dev = abs(color - baseline['color_mean']) / max(baseline['color_std'], 100)
        motion_dev = abs(motion - baseline['motion_mean']) / max(baseline['motion_std'], 5.0)
        
        # Normalize deviations (cap at reasonable values)
        texture_dev = min(texture_dev, 3.0)
        edge_dev = min(edge_dev, 3.0)
        color_dev = min(color_dev, 3.0)
        motion_dev = min(motion_dev, 3.0)
        
        # Calculate similarity scores (higher = more similar to baseline)
        texture_sim = max(0, 1.0 - texture_dev / 3.0)
        edge_sim = max(0, 1.0 - edge_dev / 3.0)
        color_sim = max(0, 1.0 - color_dev / 3.0)
        motion_sim = max(0, 1.0 - motion_dev / 3.0)
        
        # Weighted similarity score
        weights = {
            'texture': 0.25,
            'edge': 0.25,
            'color': 0.30,  # Most important for screen detection
            'motion': 0.20
        }
        
        similarity_score = (
            texture_sim * weights['texture'] +
            edge_sim * weights['edge'] +
            color_sim * weights['color'] +
            motion_sim * weights['motion']
        )
        
        # Screen detection heuristics
        screen_indicators = 0
        
        # Very high color diversity = likely screen
        if color > baseline['color_mean'] + 3 * baseline['color_std']:
            screen_indicators += 1
        
        # Very low texture = likely screen/printed
        if texture < baseline['texture_mean'] - 2 * baseline['texture_std']:
            screen_indicators += 1
        
        # Very low edge quality = likely screen
        if edge < baseline['edge_mean'] - 2 * baseline['edge_std']:
            screen_indicators += 1
        
        # Decision logic
        sensitivity_thresholds = {
            'low_sensitivity': 0.4,     # Easy
            'medium_sensitivity': 0.5,  # Balanced  
            'high_sensitivity': 0.6     # Strict
        }
        
        threshold = sensitivity_thresholds[self.current_sensitivity]
        
        # If too many screen indicators, definitely fake
        if screen_indicators >= 2:
            is_real = False
            confidence = max(0.1, similarity_score * 0.5)
        else:
            is_real = similarity_score >= threshold
            confidence = similarity_score
        
        return is_real, confidence

    def smooth_results(self, current_result: DetectionResult) -> DetectionResult:
        """Simplified temporal smoothing"""
        self.result_window.append(current_result)
        
        if len(self.result_window) < 3:
            return current_result
        
        # Simple majority voting
        real_votes = sum(1 for r in self.result_window if r.is_real)
        fake_votes = len(self.result_window) - real_votes
        
        # Average confidence
        avg_confidence = np.mean([r.confidence for r in self.result_window])
        
        # Final decision
        is_real = real_votes >= fake_votes
        
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
            
            # Adaptive decision making
            is_real, confidence = self.make_adaptive_decision(texture_score, edge_score, color_score, motion_score)
            
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
        learning_status = "LEARNING" if not self.baseline_metrics['learned'] else self.current_sensitivity.replace('_', ' ').title()
        header_text = f"ULTIMATE Anti-Spoof v7.0 - {learning_status}"
        cv2.putText(frame, header_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Learning progress
        if not self.baseline_metrics['learned']:
            progress = len(self.baseline_metrics['texture_samples'])
            progress_text = f"Learning Progress: {progress}/30"
            cv2.putText(frame, progress_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Statistics
        if self.stats['face_detections'] > 0:
            real_rate = (self.stats['real_classifications'] / self.stats['face_detections']) * 100
            fake_rate = (self.stats['fake_classifications'] / self.stats['face_detections']) * 100
            
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
        self.logger.info("🎬 Starting Ultimate Anti-Spoofing Detection v7.0...")
        
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
        print("    🎯 ULTIMATE ANTI-SPOOFING REALTIME SYSTEM v7.0")
        print("="*90)
        print("Technologies:")
        print("✓ Adaptive Baseline Learning")
        print("✓ InsightFace Advanced Face Detection")
        print("✓ Multi-Scale Texture Analysis")
        print("✓ Edge Quality Assessment")
        print("✓ Color Diversity Analysis")
        print("✓ Motion Detection & Liveness")
        print("✓ Smart Screen Detection")
        print("\nPhases:")
        print("1. LEARNING (30 frames) - System learns your environment")
        print("2. DETECTION - Real vs Fake classification")
        print("\nControls:")
        print("q=quit | s=screenshot | r=reset | 1-3=sensitivity")
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
                    
                    # Log every 10th frame (only after learning)
                    if self.baseline_metrics['learned'] and self.frame_counter % 10 == 0:
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
                cv2.imshow("Ultimate Anti-Spoofing v7.0", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"screenshot_{int(time.time())}.jpg", frame)
                    print("📸 Screenshot saved!")
                elif key == ord('r'):
                    # Reset everything
                    self.stats = {'total_frames': 0, 'face_detections': 0, 
                                 'real_classifications': 0, 'fake_classifications': 0}
                    self.frame_counter = 0
                    self.baseline_metrics = {
                        'texture_samples': deque(maxlen=50),
                        'edge_samples': deque(maxlen=50),
                        'color_samples': deque(maxlen=50),
                        'motion_samples': deque(maxlen=50),
                        'learned': False
                    }
                    print("🔄 System reset! Learning phase restarted.")
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
    parser = argparse.ArgumentParser(description='Ultimate Anti-Spoofing System v7.0')
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
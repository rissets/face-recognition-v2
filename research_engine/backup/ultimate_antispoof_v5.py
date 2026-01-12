#!/usr/bin/env python3
"""
ULTIMATE Anti-Spoofing Realtime System
Version 5.0 - Hybrid Approach dengan Multiple Detection Methods
Kombinasi terbaik dari semua teknik yang sudah diuji
"""

import cv2
import numpy as np
import logging
import time
import os
from pathlib import Path
from collections import deque
import insightface
from insightface.app import FaceAnalysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_antispoof.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltimateAntiSpoofDetector:
    def __init__(self):
        """Initialize Ultimate Anti-Spoofing Detector"""
        logger.info("🚀 Inisialisasi ULTIMATE Anti-Spoofing System...")
        
        # Initialize InsightFace untuk face detection yang akurat
        try:
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(320, 320))
            logger.info("✓ InsightFace initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing InsightFace: {e}")
            raise
        
        # Detection parameters - TUNED dari hasil testing
        self.texture_threshold_low = 15000   # Untuk real faces
        self.texture_threshold_high = 25000  # Untuk screen detection
        self.edge_threshold = 0.35           # Balance antara real dan fake
        self.color_threshold = 35            # Color diversity
        self.motion_threshold = 5.0          # Motion detection
        
        # Adaptive thresholds
        self.adaptive_real_threshold = 0.65  # Lebih lenient untuk real faces
        self.adaptive_fake_threshold = 0.45  # Lebih strict untuk fake detection
        
        # Frame processing
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        
        # Result smoothing dengan weighted history
        self.result_history = deque(maxlen=9)  # Lebih banyak history
        self.motion_history = deque(maxlen=5)
        self.stable_result = None
        self.stable_confidence = 0.0
        
        # Statistics
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        self.processing_times = []
        
        # Previous frame untuk motion detection
        self.prev_frame = None
        
        logger.info("✓ Ultimate Anti-Spoofing Detector ready!")

    def calculate_texture_variance(self, face_region):
        """Advanced texture analysis dengan multiple scales"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale texture analysis
            texture_scores = []
            scales = [1.0, 0.8, 0.6]
            
            for scale in scales:
                if scale != 1.0:
                    h, w = gray.shape
                    scaled = cv2.resize(gray, (int(w*scale), int(h*scale)))
                    scaled = cv2.resize(scaled, (w, h))
                else:
                    scaled = gray.copy()
                
                # Laplacian variance untuk texture
                laplacian = cv2.Laplacian(scaled, cv2.CV_64F)
                variance = np.var(laplacian)
                texture_scores.append(variance)
            
            # Weighted average (recent scales more important)
            weights = [3, 2, 1]
            weighted_texture = sum(score * weight for score, weight in zip(texture_scores, weights)) / sum(weights)
            
            return weighted_texture
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            return 0

    def calculate_edge_quality(self, face_region):
        """Enhanced edge quality analysis"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Multiple edge detection methods
            # 1. Canny edges
            edges_canny = cv2.Canny(gray, 50, 150)
            edge_density_canny = np.sum(edges_canny > 0) / (edges_canny.shape[0] * edges_canny.shape[1])
            
            # 2. Sobel gradients
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_strength = np.mean(edge_magnitude)
            
            # Combined edge quality
            edge_quality = (edge_density_canny * 1000 + edge_strength / 100) / 2
            
            return edge_quality
            
        except Exception as e:
            logger.error(f"Error in edge analysis: {e}")
            return 0

    def calculate_color_diversity(self, face_region):
        """Advanced color diversity analysis"""
        try:
            # Multiple color spaces
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Color variance in each channel
            color_vars = []
            
            # HSV analysis
            for i in range(3):
                color_vars.append(np.var(hsv[:, :, i]))
            
            # LAB analysis
            for i in range(3):
                color_vars.append(np.var(lab[:, :, i]))
            
            # Skin tone analysis (additional check for natural skin)
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            
            # Skin tone range in HSV
            skin_mask = ((h_channel >= 0) & (h_channel <= 20)) | ((h_channel >= 160) & (h_channel <= 180))
            skin_mask = skin_mask & (s_channel >= 30) & (s_channel <= 170)
            
            skin_percentage = np.sum(skin_mask) / (face_region.shape[0] * face_region.shape[1])
            
            # Combined color diversity with skin tone bonus
            base_diversity = np.mean(color_vars)
            skin_bonus = skin_percentage * 10  # Bonus for natural skin tones
            
            return base_diversity + skin_bonus
            
        except Exception as e:
            logger.error(f"Error in color analysis: {e}")
            return 0

    def detect_motion(self, current_face):
        """Motion detection untuk liveness"""
        try:
            if self.prev_frame is None:
                self.prev_frame = current_face.copy()
                return 0.0
            
            # Calculate optical flow
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_face, cv2.COLOR_BGR2GRAY)
            
            # Frame difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_score = np.mean(diff)
            
            # Update previous frame
            self.prev_frame = current_face.copy()
            
            # Add to motion history
            self.motion_history.append(motion_score)
            
            return motion_score
            
        except Exception as e:
            logger.error(f"Error in motion detection: {e}")
            return 0.0

    def advanced_spoofing_detection(self, face_region):
        """Ultimate spoofing detection dengan multiple indicators"""
        try:
            start_time = time.time()
            
            # 1. Texture Analysis
            texture_score = self.calculate_texture_variance(face_region)
            
            # 2. Edge Quality
            edge_quality = self.calculate_edge_quality(face_region)
            
            # 3. Color Diversity
            color_diversity = self.calculate_color_diversity(face_region)
            
            # 4. Motion Detection
            motion_score = self.detect_motion(face_region)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # ADAPTIVE SCORING SYSTEM
            scores = []
            
            # Texture scoring (inverse - lower texture = more suspicious for screens)
            if texture_score < self.texture_threshold_low:
                texture_score_norm = 0.2  # Very suspicious (too smooth)
            elif texture_score > self.texture_threshold_high:
                texture_score_norm = 0.4  # Moderately suspicious (too noisy)
            else:
                texture_score_norm = 0.8  # Good natural texture
            scores.append(texture_score_norm)
            
            # Edge quality scoring
            edge_score_norm = min(1.0, edge_quality / (self.edge_threshold * 2))
            edge_score_norm = max(0.2, edge_score_norm)  # Minimum score
            scores.append(edge_score_norm)
            
            # Color diversity scoring
            color_score_norm = min(1.0, color_diversity / (self.color_threshold * 2))
            color_score_norm = max(0.2, color_score_norm)
            scores.append(color_score_norm)
            
            # Motion scoring (small motion is good, too much or too little is suspicious)
            if len(self.motion_history) >= 3:
                avg_motion = np.mean(list(self.motion_history))
                if 2.0 <= avg_motion <= 8.0:
                    motion_score_norm = 0.8
                elif avg_motion < 1.0:
                    motion_score_norm = 0.3  # Too static (photo/screen)
                else:
                    motion_score_norm = 0.6  # Too much motion
                scores.append(motion_score_norm)
            
            # Weighted final score
            if len(scores) == 4:
                weights = [0.3, 0.25, 0.25, 0.2]  # Texture, Edge, Color, Motion
            else:
                weights = [0.35, 0.32, 0.33]  # Without motion initially
            
            final_confidence = sum(score * weight for score, weight in zip(scores, weights))
            
            # Adaptive decision making
            if final_confidence >= self.adaptive_real_threshold:
                is_real = True
            elif final_confidence <= self.adaptive_fake_threshold:
                is_real = False
            else:
                # Uncertain zone - use additional heuristics
                if texture_score > 20000 and edge_quality < 0.3:
                    is_real = False  # Likely screen/photo
                else:
                    is_real = final_confidence > 0.55  # Slight bias toward real
            
            analysis_result = {
                'is_real': is_real,
                'confidence': final_confidence,
                'texture_score': texture_score,
                'edge_quality': edge_quality,
                'color_diversity': color_diversity,
                'motion_score': motion_score,
                'processing_time': processing_time
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in spoofing detection: {e}")
            return None

    def smart_result_smoothing(self, new_result):
        """Intelligent result smoothing dengan temporal consistency"""
        if new_result is None:
            return self.stable_result, self.stable_confidence
        
        # Add to history
        self.result_history.append(new_result)
        
        # Need sufficient history
        if len(self.result_history) < 5:
            self.stable_result = new_result['is_real']
            self.stable_confidence = new_result['confidence']
            return self.stable_result, self.stable_confidence
        
        # Weighted scoring dengan temporal consistency
        recent_results = list(self.result_history)
        
        # Time-based weights (more recent = higher weight)
        time_weights = [i+1 for i in range(len(recent_results))]
        total_time_weight = sum(time_weights)
        
        # Confidence-based weights
        conf_weights = [r['confidence'] for r in recent_results]
        
        # Combined weights
        combined_weights = [t*c for t, c in zip(time_weights, conf_weights)]
        total_combined_weight = sum(combined_weights)
        
        # Weighted decision
        real_score = sum(w for r, w in zip(recent_results, combined_weights) if r['is_real'])
        fake_score = total_combined_weight - real_score
        
        # Confidence calculation
        weighted_confidence = sum(r['confidence'] * w for r, w in zip(recent_results, combined_weights)) / total_combined_weight
        
        # Decision with hysteresis
        confidence_threshold = 0.6
        if real_score > fake_score * 1.1:  # Slight bias toward real
            stable_is_real = True
        elif fake_score > real_score * 1.2:  # Stronger bias against fake
            stable_is_real = False
        else:
            # Use confidence threshold
            stable_is_real = weighted_confidence > confidence_threshold
        
        self.stable_result = stable_is_real
        self.stable_confidence = weighted_confidence
        
        return self.stable_result, self.stable_confidence

    def run_ultimate_detection(self):
        """Run ultimate anti-spoofing detection"""
        logger.info("🎬 Starting Ultimate Anti-Spoofing Detection...")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera")
            return
        
        # Optimal camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logger.info("✓ Camera initialized")
        
        print("\n" + "="*90)
        print("    🎯 ULTIMATE ANTI-SPOOFING REALTIME SYSTEM v5.0")
        print("="*90)
        print("Technologies:")
        print("✓ InsightFace Advanced Face Detection")
        print("✓ Multi-Scale Texture Analysis")
        print("✓ Advanced Edge Quality Assessment")
        print("✓ Color Diversity & Skin Tone Analysis")
        print("✓ Motion Detection & Liveness")
        print("✓ Adaptive Threshold System")
        print("✓ Intelligent Result Smoothing")
        print("\nControls:")
        print("q=quit | s=screenshot | d=debug | r=reset | 1-3=sensitivity")
        print("="*90)
        
        show_debug = True
        
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
                
                # Detect faces with InsightFace
                faces = self.app.get(frame)
                
                # Process faces
                should_process = len(faces) > 0 and (self.frame_count % self.frame_skip == 0)
                current_result = None
                
                if should_process:
                    # Process largest face
                    if len(faces) > 0:
                        # Sort by area
                        faces_with_area = [(face, face.bbox[2] * face.bbox[3]) for face in faces]
                        faces_with_area.sort(key=lambda x: x[1], reverse=True)
                        largest_face = faces_with_area[0][0]
                        
                        # Extract face region
                        bbox = largest_face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        
                        # Add padding
                        padding = 20
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding) 
                        x2 = min(frame.shape[1], x2 + padding)
                        y2 = min(frame.shape[0], y2 + padding)
                        
                        face_region = frame[y1:y2, x1:x2]
                        
                        if face_region.size > 0:
                            # Ultimate spoofing detection
                            current_result = self.advanced_spoofing_detection(face_region)
                            
                            if current_result:
                                self.detection_count += 1
                                
                                # Update stats
                                if current_result['is_real']:
                                    self.real_count += 1
                                else:
                                    self.fake_count += 1
                                
                                logger.info(f"Frame {self.frame_count}: {'REAL' if current_result['is_real'] else 'FAKE'} "
                                          f"- Conf: {current_result['confidence']:.3f}, "
                                          f"Texture: {current_result['texture_score']:.0f}, "
                                          f"Edge: {current_result['edge_quality']:.2f}, "
                                          f"Color: {current_result['color_diversity']:.0f}")
                
                # Get smoothed result
                stable_is_real, stable_confidence = self.smart_result_smoothing(current_result)
                
                # Draw results
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    if stable_is_real is not None:
                        if stable_is_real:
                            color = (0, 255, 0)  # Green for REAL
                            status_text = f"REAL ({stable_confidence:.2f})"
                        else:
                            color = (0, 0, 255)  # Red for FAKE
                            status_text = f"FAKE ({stable_confidence:.2f})"
                    else:
                        color = (128, 128, 128)  # Gray for processing
                        status_text = "ANALYZING..."
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw status text
                    cv2.putText(frame, status_text, (x1, y1-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Debug information
                    if show_debug and current_result:
                        debug_y = y2 + 25
                        debug_info = [
                            f"T:{current_result['texture_score']:.0f} E:{current_result['edge_quality']:.2f}",
                            f"C:{current_result['color_diversity']:.0f} M:{current_result['motion_score']:.1f}",
                            f"History:{len(self.result_history)} Det:{self.detection_count}"
                        ]
                        
                        for info in debug_info:
                            cv2.putText(frame, info, (x1, debug_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            debug_y += 18
                
                # Status information
                status_y = 30
                status_lines = [
                    f"FPS: {display_fps:.1f} | Frames: {self.frame_count} | Faces: {len(faces)}",
                    f"Detections: {self.detection_count} | Real: {self.real_count} | Fake: {self.fake_count}",
                    f"Thresholds: R>{self.adaptive_real_threshold:.2f} F<{self.adaptive_fake_threshold:.2f}"
                ]
                
                for line in status_lines:
                    cv2.putText(frame, line, (10, status_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    status_y += 20
                
                # Performance indicator
                if self.processing_times:
                    avg_time = np.mean(self.processing_times[-10:])
                    perf_color = (0, 255, 0) if avg_time < 0.02 else (0, 255, 255) if avg_time < 0.05 else (0, 0, 255)
                    cv2.putText(frame, f"Avg: {avg_time:.3f}s", (frame.shape[1]-150, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, perf_color, 1)
                
                # Show frame
                cv2.imshow('Ultimate Anti-Spoofing System v5.0', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"ultimate_antispoof_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Screenshot: {filename}")
                elif key == ord('d'):
                    show_debug = not show_debug
                    logger.info(f"🔍 Debug: {'ON' if show_debug else 'OFF'}")
                elif key == ord('r'):
                    # Reset all stats and history
                    self.frame_count = 0
                    self.detection_count = 0
                    self.real_count = 0
                    self.fake_count = 0
                    self.result_history.clear()
                    self.motion_history.clear()
                    self.processing_times.clear()
                    self.stable_result = None
                    self.stable_confidence = 0.0
                    self.prev_frame = None
                    logger.info("🔄 Full reset completed")
                elif key == ord('1'):
                    # Conservative (favor real)
                    self.adaptive_real_threshold = 0.5
                    self.adaptive_fake_threshold = 0.3
                    logger.info("🎯 Sensitivity: CONSERVATIVE (favor real)")
                elif key == ord('2'):
                    # Balanced
                    self.adaptive_real_threshold = 0.65
                    self.adaptive_fake_threshold = 0.45
                    logger.info("🎯 Sensitivity: BALANCED")
                elif key == ord('3'):
                    # Aggressive (favor fake detection)
                    self.adaptive_real_threshold = 0.75
                    self.adaptive_fake_threshold = 0.55
                    logger.info("🎯 Sensitivity: AGGRESSIVE (favor fake detection)")
        
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Final comprehensive summary
            logger.info("=== ULTIMATE ANTISPOOFING SESSION SUMMARY ===")
            logger.info(f"Total frames processed: {self.frame_count}")
            logger.info(f"Total face detections: {self.detection_count}")
            logger.info(f"Real face classifications: {self.real_count}")
            logger.info(f"Fake face classifications: {self.fake_count}")
            
            if self.detection_count > 0:
                real_percentage = (self.real_count / self.detection_count) * 100
                fake_percentage = (self.fake_count / self.detection_count) * 100
                logger.info(f"Real classification rate: {real_percentage:.1f}%")
                logger.info(f"Fake classification rate: {fake_percentage:.1f}%")
            
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                max_time = np.max(self.processing_times)
                min_time = np.min(self.processing_times)
                logger.info(f"Processing time - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
                logger.info(f"Theoretical max FPS: {1/avg_time:.1f}")

if __name__ == "__main__":
    try:
        detector = UltimateAntiSpoofDetector()
        detector.run_ultimate_detection()
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        print(f"\n❌ Error: {e}")
        print("📋 Check log: ultimate_antispoof.log")
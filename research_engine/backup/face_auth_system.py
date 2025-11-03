import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import faiss
import pickle
import os
import time
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import json
import logging
import math

def json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON serializable format"""
    # Handle basic Python types first
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [json_serializable(item) for item in obj]
    # Handle numpy types
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Handle complex objects with __dict__ safely
    elif hasattr(obj, '__dict__') and not isinstance(obj, (np.generic)):
        try:
            return {key: json_serializable(value) for key, value in obj.__dict__.items()}
        except (AttributeError, TypeError):
            return str(obj)
    else:
        return str(obj)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LivenessDetector:
    """Deteksi liveness untuk mencegah spoofing dengan foto/gambar"""
    
    def __init__(self):
        logger.info("Initializing LivenessDetector...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Extended eye landmarks for comprehensive detection
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Enhanced eye landmarks for optimal EAR calculation
        # Left eye: [outer_corner, top_outer, top_inner, inner_corner, bottom_inner, bottom_outer]
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]  
        # Right eye: [outer_corner, top_outer, top_inner, inner_corner, bottom_inner, bottom_outer]  
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
        
        # Additional eye points for better validation
        self.LEFT_EYE_CENTER = 468  # Eye center for distance calculation
        self.RIGHT_EYE_CENTER = 473
        
        # Enhanced blink detection parameters
        self.EAR_THRESHOLD = 0.8  # Base threshold
        self.ADAPTIVE_FACTOR = 0.75  # More sensitive adaptive factor
        self.CONSECUTIVE_FRAMES = 2  # Frames needed for blink confirmation
        self.MIN_BLINK_DURATION = 1  # Minimum frames for valid blink
        self.MAX_BLINK_DURATION = 10  # Maximum frames for valid blink
        
        # Tracking variables
        self.blink_counter = 0
        self.frame_counter = 0
        self.total_blinks = 0
        self.valid_blinks = 0  # Only natural blinks
        self.ear_history = []  # EAR history for baseline
        self.baseline_ear = None  # Adaptive baseline
        self.blink_start_frame = None
        self.last_blink_time = 0
        
        # Quality metrics
        self.eye_visibility_score = 0.0
        self.blink_quality_scores = []
        
        logger.info(f"LivenessDetector initialized with EAR_THRESHOLD={self.EAR_THRESHOLD}")
        
    def calculate_ear(self, landmarks, eye_indices):
        """Enhanced Eye Aspect Ratio calculation with quality validation"""
        try:
            eye_points = []
            confidence_scores = []
            
            for idx in eye_indices:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x, y = landmark.x, landmark.y
                    eye_points.append([x, y])
                    
                    # Use visibility score if available (MediaPipe confidence)
                    # MediaPipe face mesh doesn't have visibility, so use presence score
                    visibility = getattr(landmark, 'visibility', getattr(landmark, 'presence', 1.0))
                    confidence_scores.append(visibility)
            
            if len(eye_points) < 6:
                logger.debug(f"Insufficient eye points: {len(eye_points)}")
                return 0.0, 0.0
                
            eye_points = np.array(eye_points)
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 1.0
            
            # Validate eye shape (basic sanity check)
            if not self._validate_eye_shape(eye_points):
                logger.debug("Invalid eye shape detected")
                return 0.0, 0.0
            
            # Enhanced 6-point EAR calculation
            # Points: [outer_corner, top_outer, top_inner, inner_corner, bottom_inner, bottom_outer]
            
            # Multiple vertical measurements for better accuracy
            A = np.linalg.norm(eye_points[1] - eye_points[5])  # top_outer to bottom_outer
            B = np.linalg.norm(eye_points[2] - eye_points[4])  # top_inner to bottom_inner
            
            # Horizontal distance (eye width)
            C = np.linalg.norm(eye_points[0] - eye_points[3])  # outer_corner to inner_corner
            
            if C < 0.001:  # Avoid division by very small numbers
                logger.debug("Eye width too small")
                return 0.0, 0.0
            
            # Standard EAR formula with validation
            ear = (A + B) / (2.0 * C)
            
            # Quality score based on confidence and measurements
            # Since MediaPipe face mesh doesn't have visibility scores, use geometric validation
            quality = min(1.0, C * 15)  # Quality based on eye width (larger eyes = better quality)
            
            logger.debug(f"EAR: A={A:.4f}, B={B:.4f}, C={C:.4f}, EAR={ear:.4f}, Quality={quality:.3f}")
            return ear, quality
            
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.0, 0.0
    
    def _validate_eye_shape(self, eye_points):
        """Validate if detected points form a reasonable eye shape"""
        try:
            # Check if points form a reasonable eye contour
            outer_corner, top_outer, top_inner, inner_corner, bottom_inner, bottom_outer = eye_points
            
            # Eye should be roughly horizontal
            horizontal_dist = abs(outer_corner[0] - inner_corner[0])
            vertical_dist = abs(outer_corner[1] - inner_corner[1])
            
            if horizontal_dist < vertical_dist * 2:  # Eye should be wider than tall
                return False
            
            # Top points should be above bottom points
            if (top_outer[1] > bottom_outer[1]) or (top_inner[1] > bottom_inner[1]):
                return False
                
            return True
            
        except Exception:
            return False
        
    def detect_blink(self, frame):
        """Enhanced blink detection with improved validation and quality metrics"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            self.frame_counter += 1
            current_time = time.time()
            
            if results.multi_face_landmarks:
                logger.debug("Face landmarks detected")
                
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate EAR for both eyes with quality scores
                    left_ear, left_quality = self.calculate_ear(face_landmarks, self.LEFT_EYE_POINTS)
                    right_ear, right_quality = self.calculate_ear(face_landmarks, self.RIGHT_EYE_POINTS)
                    
                    if left_ear == 0.0 or right_ear == 0.0:
                        logger.debug("Invalid EAR values")
                        return False, 0.0
                    
                    avg_ear = (left_ear + right_ear) / 2.0
                    avg_quality = (left_quality + right_quality) / 2.0
                    self.eye_visibility_score = avg_quality
                    
                    # Add measurements to history (lowered quality threshold since MediaPipe doesn't have visibility)
                    if avg_quality > 0.3:  # Lower threshold to allow building baseline
                        self.ear_history.append(avg_ear)
                        if len(self.ear_history) > 50:  # Keep more history for stability
                            self.ear_history.pop(0)
                    
                    # Calculate adaptive baseline with quality weighting
                    if len(self.ear_history) >= 10:  # Reduced samples needed for stability
                        recent_ears = self.ear_history[-10:]
                        baseline = np.mean(recent_ears)
                        baseline_std = np.std(recent_ears)
                        
                        if self.baseline_ear is None:
                            self.baseline_ear = baseline
                        else:
                            # Smooth update with quality weighting
                            alpha = 0.05 * avg_quality  # Quality-weighted learning rate
                            self.baseline_ear = (1 - alpha) * self.baseline_ear + alpha * baseline
                        
                        # Dynamic adaptive threshold based on variability
                        stability_factor = min(1.0, 1.0 / (baseline_std + 0.01))
                        adaptive_threshold = self.baseline_ear * (self.ADAPTIVE_FACTOR + 0.1 * (1 - stability_factor))
                        
                        logger.debug(f"Left EAR: {left_ear:.4f} (Q:{left_quality:.2f}), Right EAR: {right_ear:.4f} (Q:{right_quality:.2f})")
                        logger.debug(f"Avg EAR: {avg_ear:.4f}, Baseline: {self.baseline_ear:.4f}, Threshold: {adaptive_threshold:.4f}")
                        logger.debug(f"Stability: {stability_factor:.3f}, Blink counter: {self.blink_counter}")
                        
                        # Enhanced blink detection logic
                        if avg_ear < adaptive_threshold and avg_quality > 0.3:  # Lower quality threshold
                            if self.blink_counter == 0:
                                self.blink_start_frame = self.frame_counter
                            self.blink_counter += 1
                            logger.debug(f"Potential blink frame {self.blink_counter}")
                        else:
                            # End of potential blink
                            if self.blink_counter >= self.MIN_BLINK_DURATION:
                                blink_duration = self.blink_counter
                                
                                # Validate blink duration and timing
                                if (self.MIN_BLINK_DURATION <= blink_duration <= self.MAX_BLINK_DURATION and 
                                    current_time - self.last_blink_time > 0.2):  # Minimum 200ms between blinks
                                    
                                    self.total_blinks += 1
                                    self.valid_blinks += 1
                                    self.last_blink_time = current_time
                                    
                                    # Record blink quality
                                    blink_quality = avg_quality * stability_factor
                                    self.blink_quality_scores.append(blink_quality)
                                    if len(self.blink_quality_scores) > 10:
                                        self.blink_quality_scores.pop(0)
                                    
                                    logger.info(f"VALID BLINK DETECTED! Duration: {blink_duration} frames, Quality: {blink_quality:.3f}")
                                    logger.info(f"Total blinks: {self.total_blinks}, Valid blinks: {self.valid_blinks}")
                                else:
                                    logger.debug(f"Invalid blink: duration={blink_duration}, time_gap={current_time - self.last_blink_time:.2f}")
                            
                            self.blink_counter = 0
                            self.blink_start_frame = None
                        
                        return True, avg_ear
                    else:
                        # Still building baseline
                        logger.debug(f"Building baseline... {len(self.ear_history)}/10 frames")
                        return True, avg_ear
            else:
                logger.debug("No face landmarks detected")
            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Error in detect_blink: {e}")
            return False, 0.0
    
    def is_live(self, blink_count_threshold=2):
        """Tentukan apakah subjek adalah hidup berdasarkan kedipan"""
        is_alive = self.total_blinks >= blink_count_threshold
        logger.debug(f"Liveness check: {self.total_blinks} >= {blink_count_threshold} = {is_alive}")
        return is_alive
    
    def reset(self):
        """Reset counter untuk sesi baru"""
        logger.info("Resetting liveness detector counters")
        self.blink_counter = 0
        self.frame_counter = 0
        self.total_blinks = 0
        self.ear_history = []
    
    def draw_eye_landmarks(self, frame, landmarks):
        """Draw eye landmarks for debugging dengan area mata yang lebih luas"""
        try:
            h, w = frame.shape[:2]
            
            # Draw all left eye landmarks (comprehensive)
            left_eye_coords = []
            for idx in self.LEFT_EYE_LANDMARKS:
                if idx < len(landmarks.landmark):
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    left_eye_coords.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Draw all right eye landmarks (comprehensive)
            right_eye_coords = []
            for idx in self.RIGHT_EYE_LANDMARKS:
                if idx < len(landmarks.landmark):
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    right_eye_coords.append((x, y))
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
            
            # Draw key points for EAR calculation (larger circles)
            for idx in self.LEFT_EYE_POINTS:
                if idx < len(landmarks.landmark):
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow for key points
            
            for idx in self.RIGHT_EYE_POINTS:
                if idx < len(landmarks.landmark):
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow for key points
            
            # Draw eye contours
            if len(left_eye_coords) > 3:
                left_eye_coords = np.array(left_eye_coords, dtype=np.int32)
                cv2.polylines(frame, [left_eye_coords], True, (0, 255, 0), 1)
            
            if len(right_eye_coords) > 3:
                right_eye_coords = np.array(right_eye_coords, dtype=np.int32)
                cv2.polylines(frame, [right_eye_coords], True, (255, 0, 0), 1)
                    
        except Exception as e:
            logger.error(f"Error drawing eye landmarks: {e}")
    
    def get_debug_info(self):
        """Get debug information"""
        return {
            'total_blinks': self.total_blinks,
            'blink_counter': self.blink_counter,
            'ear_threshold': self.EAR_THRESHOLD,
            'consecutive_frames': self.CONSECUTIVE_FRAMES,
            'ear_history_length': len(self.ear_history),
            'last_ear': self.ear_history[-1] if self.ear_history else 0.0
        }

class ObstacleDetector:
    """Enhanced obstacle detection dengan algoritma yang lebih canggih"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # MediaPipe Face Mesh untuk deteksi landmark wajah
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices untuk area kritis
        self.EYE_REGION_LEFT = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.EYE_REGION_RIGHT = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.MOUTH_REGION = [0, 17, 18, 200, 199, 175, 13, 269, 270, 267, 269, 270, 267, 271, 272]
        self.NOSE_REGION = [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 307, 375, 321, 308, 324, 318]
        
        # Thresholds untuk deteksi
        self.OCCLUSION_THRESHOLD = 0.3  # 30% landmark harus terdeteksi
        self.BRIGHTNESS_THRESHOLD = 0.15  # Threshold untuk refleksi kacamata
        self.TEXTURE_THRESHOLD = 25  # Threshold untuk deteksi masker
        
    def detect_obstacles(self, frame, face_bbox):
        """Enhanced obstacle detection dengan multiple algorithms"""
        obstacles = []
        confidence_scores = {}
        
        # Extract face region dengan padding
        x1, y1, x2, y2 = face_bbox
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return obstacles, confidence_scores
            
        # Deteksi menggunakan face mesh landmarks
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_roi)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Deteksi kacamata dengan multiple methods - threshold tinggi untuk mengurangi false positive
                glasses_conf = self._detect_glasses_advanced(face_roi, face_landmarks)
                if glasses_conf > 0.3:  # Much higher threshold to avoid false positives
                    obstacles.append("glasses")
                    confidence_scores["glasses"] = glasses_conf
                
                # Deteksi masker dengan landmark analysis
                mask_conf = self._detect_mask_advanced(face_roi, face_landmarks)
                if mask_conf > 0.5:  # Slightly lower for balance
                    obstacles.append("mask")
                    confidence_scores["mask"] = mask_conf
                
                # Deteksi topi dengan shadow analysis
                hat_conf = self._detect_hat_advanced(face_roi, face_landmarks)
                if hat_conf > 0.45:  # Slightly higher to reduce false positives
                    obstacles.append("hat")
                    confidence_scores["hat"] = hat_conf
                
                # Deteksi hand covering
                hand_conf = self._detect_hand_covering(face_roi, face_landmarks)
                if hand_conf > 0.4:  # Lower for better detection
                    obstacles.append("hand")
                    confidence_scores["hand"] = hand_conf
        else:
            # Fallback ke deteksi tradisional jika tidak ada landmark
            traditional_obstacles = self._detect_obstacles_traditional(face_roi)
            obstacles.extend(traditional_obstacles)
        
        return obstacles, confidence_scores
    
    def _detect_glasses_advanced(self, face_roi, landmarks):
        """Advanced glasses detection using reflections and landmarks"""
        try:
            h, w = face_roi.shape[:2]
            confidence = 0.0
            
            # 1. Enhanced reflection detection - more conservative
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)  # More blur to reduce noise
            
            # Detect bright spots (reflections) - much higher threshold
            bright_spots = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)[1]
            bright_area = np.sum(bright_spots > 0) / (h * w)
            
            # Only very bright and substantial reflections count as glasses
            if bright_area > 0.15:  # Much higher threshold - need substantial reflection
                confidence += 0.3
            
            # 2. Edge detection around eyes
            eye_regions = []
            for idx in self.EYE_REGION_LEFT + self.EYE_REGION_RIGHT:
                if idx < len(landmarks.landmark):
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    eye_regions.append((x, y))
            
            if eye_regions:
                # Create mask around eyes
                eye_mask = np.zeros(gray.shape, dtype=np.uint8)
                eye_points = np.array(eye_regions, dtype=np.int32)
                hull = cv2.convexHull(eye_points)
                cv2.fillPoly(eye_mask, [hull], 255)
                
                # Edge detection in eye area - more selective
                edges = cv2.Canny(gray, 80, 160)  # Higher thresholds for cleaner edges
                eye_edges = cv2.bitwise_and(edges, eye_mask)
                edge_density = np.sum(eye_edges > 0) / max(1, np.sum(eye_mask > 0))
                
                # Need significant edge density to suggest glasses frames
                if edge_density > 0.2:  # Much higher threshold for edge density
                    confidence += 0.2
            
            # 3. Enhanced spectral and geometric analysis
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            l_std = np.std(l_channel)
            l_mean = np.mean(l_channel)
            
            # Need both high variation AND high brightness to suggest glasses
            if l_std > 45 and l_mean > 130:  # Much higher thresholds
                confidence += 0.2
            
            # 4. Frame detection using Hough lines - more selective
            edges_full = cv2.Canny(gray, 60, 120)
            lines = cv2.HoughLinesP(edges_full, 1, np.pi/180, threshold=60, minLineLength=40, maxLineGap=5)
            
            frame_lines = 0
            if lines is not None:
                # Filter for lines that could be glasses frames (horizontal-ish)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    # Only count horizontal or near-horizontal lines
                    if angle < 15 or angle > 165:
                        frame_lines += 1
            
            # Need many frame-like lines to suggest glasses
            if frame_lines > 15:  # Much higher threshold
                confidence += 0.3
            
            # 5. Template matching for common glasses shapes
            eye_region_upper = face_roi[:int(h*0.6), :]
            if eye_region_upper.size > 0:
                # Look for substantial horizontal structures (glasses frames)
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 4))
                horizontal_lines = cv2.morphologyEx(edges_full[:int(h*0.6), :], cv2.MORPH_CLOSE, horizontal_kernel)
                horizontal_ratio = np.sum(horizontal_lines > 0) / horizontal_lines.size
                
                # Need substantial horizontal structures to suggest glasses
                if horizontal_ratio > 0.04:  # Much higher threshold
                    confidence += 0.2
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error in glasses detection: {e}")
            return 0.0
    
    def _detect_mask_advanced(self, face_roi, landmarks):
        """Advanced mask detection using mouth visibility and texture"""
        try:
            h, w = face_roi.shape[:2]
            confidence = 0.0
            
            # 1. Mouth region visibility check
            mouth_visible_points = 0
            mouth_total_points = len(self.MOUTH_REGION)
            
            for idx in self.MOUTH_REGION:
                if idx < len(landmarks.landmark):
                    # Check if landmark is visible (MediaPipe confidence)
                    visibility = getattr(landmarks.landmark[idx], 'visibility', 1.0)
                    if visibility > 0.5:
                        mouth_visible_points += 1
            
            mouth_visibility_ratio = mouth_visible_points / mouth_total_points
            if mouth_visibility_ratio < 0.5:  # Less than 50% mouth visible
                confidence += 0.5
            
            # 2. Texture analysis in lower face
            lower_face = face_roi[int(h * 0.6):, :]
            if lower_face.size > 0:
                gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
                
                # Local Binary Pattern for texture
                texture_std = np.std(gray_lower)
                if texture_std < self.TEXTURE_THRESHOLD:  # Uniform texture = mask
                    confidence += 0.3
                
                # Color uniformity check
                hsv_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
                hue_std = np.std(hsv_lower[:, :, 0])
                if hue_std < 15:  # Very uniform color
                    confidence += 0.2
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error in mask detection: {e}")
            return 0.0
    
    def _detect_hat_advanced(self, face_roi, landmarks):
        """Advanced hat detection using shadow analysis"""
        try:
            h, w = face_roi.shape[:2]
            confidence = 0.0
            
            # 1. Shadow detection in upper region
            upper_region = face_roi[:int(h * 0.3), :]
            if upper_region.size > 0:
                gray_upper = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
                
                # Detect very dark areas (shadows)
                dark_mask = gray_upper < 40
                dark_ratio = np.sum(dark_mask) / dark_mask.size
                
                if dark_ratio > 0.4:  # 40% dark area indicates hat shadow
                    confidence += 0.6
            
            # 2. Forehead visibility check
            forehead_landmarks = [9, 10, 151, 337, 299, 333, 298, 301]  # Forehead area
            visible_forehead = 0
            
            for idx in forehead_landmarks:
                if idx < len(landmarks.landmark):
                    visibility = getattr(landmarks.landmark[idx], 'visibility', 1.0)
                    if visibility > 0.5:
                        visible_forehead += 1
            
            if visible_forehead < len(forehead_landmarks) * 0.5:
                confidence += 0.4
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error in hat detection: {e}")
            return 0.0
    
    def _detect_hand_covering(self, face_roi, landmarks):
        """Detect hand covering parts of face"""
        try:
            h, w = face_roi.shape[:2]
            confidence = 0.0
            
            # Skin color detection
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Check for unusual skin patches (hands)
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            large_skin_areas = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > (h * w * 0.1):  # Large skin area (potentially hand)
                    # Check if it's not the main face
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect_ratio = cw / ch
                    
                    if aspect_ratio > 0.5 and aspect_ratio < 2.0:  # Hand-like shape
                        large_skin_areas += 1
            
            if large_skin_areas > 1:  # Multiple large skin areas = hand covering
                confidence = 0.7
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error in hand detection: {e}")
            return 0.0
    
    def _detect_obstacles_traditional(self, face_roi):
        """Fallback traditional detection methods"""
        obstacles = []
        
        h, w = face_roi.shape[:2]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Very conservative brightness check for glasses - avoid false positives
        bright_pixels = np.sum(gray > 240)  # Even higher threshold
        bright_ratio = bright_pixels / (h * w)
        
        # Conservative edge-based glasses detection
        edges = cv2.Canny(gray, 80, 160)  # Higher thresholds for cleaner edges
        edge_pixels = np.sum(edges > 0)
        edge_ratio = edge_pixels / (h * w)
        
        # Only detect glasses if there are BOTH substantial bright reflections AND many frame edges
        # This reduces false positives from normal eye reflections
        if bright_ratio > 0.08 and edge_ratio > 0.15:  # Both conditions with higher thresholds
            obstacles.append("glasses")
        
        # Simple darkness check for hat
        upper_region = gray[:int(h * 0.3), :]
        if upper_region.size > 0:
            dark_pixels = np.sum(upper_region < 50)
            if dark_pixels / upper_region.size > 0.5:
                obstacles.append("hat")
        
        return obstacles

class CameraGuideSystem:
    """Sistem panduan kamera dengan overlay visual dan logging"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Guide parameters
        self.OPTIMAL_FACE_SIZE = (200, 250)  # Optimal face size in pixels
        self.FACE_SIZE_TOLERANCE = 0.3  # 30% tolerance
        self.CENTER_TOLERANCE = 50  # Pixels from center
        
        # Colors untuk guide
        self.COLOR_GOOD = (0, 255, 0)      # Green
        self.COLOR_WARNING = (0, 255, 255)  # Yellow  
        self.COLOR_BAD = (0, 0, 255)       # Red
        self.COLOR_GUIDE = (255, 255, 255) # White
        
        # Eye area tracking
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
    def draw_face_guide(self, frame, face_bbox=None):
        """Draw face position guide overlay"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw center guide oval
        oval_w, oval_h = self.OPTIMAL_FACE_SIZE
        cv2.ellipse(frame, (center_x, center_y), (oval_w//2, oval_h//2), 0, 0, 360, self.COLOR_GUIDE, 2)
        
        # Draw center crosshair
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), self.COLOR_GUIDE, 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), self.COLOR_GUIDE, 1)
        
        # Draw corner guides
        corner_size = 30
        corner_thickness = 3
        corners = [
            (50, 50), (w - 50, 50), 
            (50, h - 50), (w - 50, h - 50)
        ]
        
        for (cx, cy) in corners:
            # Top-left corner style brackets
            if cx < w // 2 and cy < h // 2:  # Top-left
                cv2.line(frame, (cx, cy), (cx + corner_size, cy), self.COLOR_GUIDE, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy + corner_size), self.COLOR_GUIDE, corner_thickness)
            elif cx > w // 2 and cy < h // 2:  # Top-right
                cv2.line(frame, (cx, cy), (cx - corner_size, cy), self.COLOR_GUIDE, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy + corner_size), self.COLOR_GUIDE, corner_thickness)
            elif cx < w // 2 and cy > h // 2:  # Bottom-left
                cv2.line(frame, (cx, cy), (cx + corner_size, cy), self.COLOR_GUIDE, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy - corner_size), self.COLOR_GUIDE, corner_thickness)
            else:  # Bottom-right
                cv2.line(frame, (cx, cy), (cx - corner_size, cy), self.COLOR_GUIDE, corner_thickness)
                cv2.line(frame, (cx, cy), (cx, cy - corner_size), self.COLOR_GUIDE, corner_thickness)
        
        # Face position feedback
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Check position
            position_status = self._get_position_status(face_center_x, face_center_y, center_x, center_y)
            size_status = self._get_size_status(face_width, face_height)
            
            # Draw face bounding box with status color
            color = self._get_status_color(position_status, size_status)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw feedback text
            self._draw_feedback_text(frame, position_status, size_status, face_center_x, face_center_y)
        
        return frame
    
    def draw_eye_area_guides(self, frame, landmarks=None):
        """Draw eye area guides and tracking"""
        if landmarks is None:
            # Draw eye area guides when no face detected
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Eye area rectangles
            eye_width, eye_height = 80, 40
            left_eye_center = (center_x - 60, center_y - 20)
            right_eye_center = (center_x + 60, center_y - 20)
            
            # Draw eye guide rectangles
            cv2.rectangle(frame, 
                         (left_eye_center[0] - eye_width//2, left_eye_center[1] - eye_height//2),
                         (left_eye_center[0] + eye_width//2, left_eye_center[1] + eye_height//2),
                         self.COLOR_GUIDE, 1)
            
            cv2.rectangle(frame, 
                         (right_eye_center[0] - eye_width//2, right_eye_center[1] - eye_height//2),
                         (right_eye_center[0] + eye_width//2, right_eye_center[1] + eye_height//2),
                         self.COLOR_GUIDE, 1)
            
            # Labels
            cv2.putText(frame, "LEFT EYE", (left_eye_center[0] - 30, left_eye_center[1] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_GUIDE, 1)
            cv2.putText(frame, "RIGHT EYE", (right_eye_center[0] - 35, right_eye_center[1] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_GUIDE, 1)
        else:
            # Draw detected eye landmarks
            h, w = frame.shape[:2]
            
            # Left eye
            left_eye_points = []
            for idx in self.LEFT_EYE_LANDMARKS:
                if idx < len(landmarks.landmark):
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    left_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 1, self.COLOR_GOOD, -1)
            
            # Right eye
            right_eye_points = []
            for idx in self.RIGHT_EYE_LANDMARKS:
                if idx < len(landmarks.landmark):
                    x = int(landmarks.landmark[idx].x * w)
                    y = int(landmarks.landmark[idx].y * h)
                    right_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 1, self.COLOR_GOOD, -1)
            
            # Draw eye contours
            if len(left_eye_points) > 3:
                left_eye_points = np.array(left_eye_points, dtype=np.int32)
                cv2.polylines(frame, [left_eye_points], True, self.COLOR_GOOD, 1)
            
            if len(right_eye_points) > 3:
                right_eye_points = np.array(right_eye_points, dtype=np.int32)
                cv2.polylines(frame, [right_eye_points], True, self.COLOR_GOOD, 1)
        
        return frame
    
    def analyze_frame_quality(self, frame, face_bbox=None, landmarks=None):
        """Analyze frame quality untuk logging"""
        h, w = frame.shape[:2]
        quality_metrics = {
            'brightness': 0.0,
            'contrast': 0.0,
            'sharpness': 0.0,
            'face_size_score': 0.0,
            'face_position_score': 0.0,
            'eye_visibility_score': 0.0,
            'overall_score': 0.0
        }
        
        # Brightness analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        quality_metrics['brightness'] = min(1.0, brightness / 128.0)  # Normalize to 0-1
        
        # Contrast analysis
        contrast = np.std(gray)
        quality_metrics['contrast'] = min(1.0, contrast / 64.0)
        
        # Sharpness analysis (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        quality_metrics['sharpness'] = min(1.0, sharpness / 1000.0)
        
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_width = x2 - x1
            face_height = y2 - y1
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2
            
            # Face size score
            optimal_width, optimal_height = self.OPTIMAL_FACE_SIZE
            size_diff_w = abs(face_width - optimal_width) / optimal_width
            size_diff_h = abs(face_height - optimal_height) / optimal_height
            quality_metrics['face_size_score'] = max(0.0, 1.0 - (size_diff_w + size_diff_h) / 2)
            
            # Face position score
            center_x, center_y = w // 2, h // 2
            position_diff = math.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
            quality_metrics['face_position_score'] = max(0.0, 1.0 - position_diff / 100.0)
        
        if landmarks is not None:
            # Eye visibility score
            visible_landmarks = 0
            total_landmarks = len(self.LEFT_EYE_LANDMARKS) + len(self.RIGHT_EYE_LANDMARKS)
            
            for idx in self.LEFT_EYE_LANDMARKS + self.RIGHT_EYE_LANDMARKS:
                if idx < len(landmarks.landmark):
                    # Since MediaPipe face mesh doesn't have visibility, just count present landmarks
                    visible_landmarks += 1
            
            quality_metrics['eye_visibility_score'] = visible_landmarks / total_landmarks
        
        # Overall score (weighted average)
        weights = {
            'brightness': 0.1,
            'contrast': 0.1,
            'sharpness': 0.2,
            'face_size_score': 0.3,
            'face_position_score': 0.2,
            'eye_visibility_score': 0.2
        }
        
        overall_score = sum(quality_metrics[key] * weights[key] for key in weights.keys())
        quality_metrics['overall_score'] = overall_score
        
        return quality_metrics
    
    def _get_position_status(self, face_x, face_y, center_x, center_y):
        """Get face position status"""
        distance = math.sqrt((face_x - center_x)**2 + (face_y - center_y)**2)
        
        if distance <= self.CENTER_TOLERANCE:
            return "CENTERED"
        elif distance <= self.CENTER_TOLERANCE * 2:
            return "CLOSE"
        else:
            # Determine direction
            dx = face_x - center_x
            dy = face_y - center_y
            
            if abs(dx) > abs(dy):
                return "MOVE LEFT" if dx > 0 else "MOVE RIGHT"
            else:
                return "MOVE UP" if dy < 0 else "MOVE DOWN"
    
    def _get_size_status(self, face_width, face_height):
        """Get face size status"""
        optimal_width, optimal_height = self.OPTIMAL_FACE_SIZE
        
        width_ratio = face_width / optimal_width
        height_ratio = face_height / optimal_height
        avg_ratio = (width_ratio + height_ratio) / 2
        
        if 0.7 <= avg_ratio <= 1.3:
            return "GOOD SIZE"
        elif avg_ratio < 0.7:
            return "MOVE CLOSER"
        else:
            return "MOVE BACK"
    
    def _get_status_color(self, position_status, size_status):
        """Get color based on status"""
        if position_status == "CENTERED" and size_status == "GOOD SIZE":
            return self.COLOR_GOOD
        elif position_status in ["CLOSE", "CENTERED"] or size_status == "GOOD SIZE":
            return self.COLOR_WARNING
        else:
            return self.COLOR_BAD
    
    def _draw_feedback_text(self, frame, position_status, size_status, face_x, face_y):
        """Draw feedback text"""
        h, w = frame.shape[:2]
        
        # Position feedback
        if position_status != "CENTERED":
            cv2.putText(frame, position_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_WARNING, 2)
        
        # Size feedback
        if size_status != "GOOD SIZE":
            cv2.putText(frame, size_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_WARNING, 2)
        
        # Good status
        if position_status == "CENTERED" and size_status == "GOOD SIZE":
            cv2.putText(frame, "POSITION OPTIMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_GOOD, 2)

class FaceEmbeddingSystem:
    """Sistem embedding wajah menggunakan InsightFace"""
    
    def __init__(self, model_name='buffalo_l'):
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # FAISS index untuk pencarian similarity
        self.dimension = 512  # Dimensi embedding InsightFace
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product untuk cosine similarity
        
        # Storage untuk metadata
        self.user_embeddings = {}
        self.user_names = []
        
        # Threshold untuk verifikasi
        self.verification_threshold = 0.4
        
    def extract_embedding(self, frame):
        """Extract embedding dari frame"""
        faces = self.app.get(frame)
        
        if len(faces) == 0:
            return None, None
            
        # Ambil wajah dengan confidence tertinggi
        face = max(faces, key=lambda x: x.det_score)
        
        # Normalize embedding
        embedding = face.normed_embedding
        
        # Bounding box
        bbox = face.bbox.astype(int)
        
        return embedding, bbox
    
    def add_user_embedding(self, user_name, embedding):
        """Tambah embedding user ke database"""
        if user_name not in self.user_embeddings:
            self.user_embeddings[user_name] = []
            
        self.user_embeddings[user_name].append(embedding)
        
        # Tambah ke FAISS index
        embedding_normalized = embedding / np.linalg.norm(embedding)
        self.index.add(embedding_normalized.reshape(1, -1))
        self.user_names.append(user_name)
        
    def verify_user(self, embedding, user_name):
        """Verifikasi user berdasarkan embedding"""
        if user_name not in self.user_embeddings:
            return False, 0.0
            
        user_embeds = self.user_embeddings[user_name]
        similarities = []
        
        for stored_embed in user_embeds:
            # Cosine similarity
            similarity = np.dot(embedding, stored_embed) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_embed)
            )
            similarities.append(similarity)
            
        max_similarity = max(similarities)
        is_verified = max_similarity > self.verification_threshold
        
        return is_verified, max_similarity
    
    def identify_user(self, embedding):
        """Identifikasi user dari embedding"""
        if self.index.ntotal == 0:
            return None, 0.0
            
        embedding_normalized = embedding / np.linalg.norm(embedding)
        
        # Search di FAISS
        D, I = self.index.search(embedding_normalized.reshape(1, -1), k=1)
        
        if len(I[0]) > 0 and D[0][0] > self.verification_threshold:
            user_name = self.user_names[I[0][0]]
            similarity = D[0][0]
            return user_name, similarity
            
        return None, 0.0
    
    def save_database(self, filepath):
        """Simpan database ke file"""
        data = {
            'user_embeddings': self.user_embeddings,
            'user_names': self.user_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        # Save FAISS index
        faiss.write_index(self.index, filepath.replace('.pkl', '.faiss'))
        
    def load_database(self, filepath):
        """Load database dari file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.user_embeddings = data['user_embeddings']
            self.user_names = data['user_names']
            
            # Rebuild FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)
            for user_name in self.user_embeddings:
                for embedding in self.user_embeddings[user_name]:
                    embedding_normalized = embedding / np.linalg.norm(embedding)
                    self.index.add(embedding_normalized.reshape(1, -1))

class SecureFaceAuth:
    """Enhanced secure face authentication system"""
    
    def __init__(self):
        self.liveness_detector = LivenessDetector()
        self.obstacle_detector = ObstacleDetector()
        self.embedding_system = FaceEmbeddingSystem()
        self.camera_guide = CameraGuideSystem()
        
        # Load existing database
        self.db_path = "face_database.pkl"
        self.embedding_system.load_database(self.db_path)
        
        # Camera dan logging
        self.cap = None
        self.frame_logger = []
        self.quality_threshold = 0.4  # Lowered threshold since MediaPipe doesn't have visibility scores
    
    def json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON serializable format"""
        return json_serializable(obj)
        
    def initialize_camera(self):
        """Inisialisasi kamera"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise Exception("Tidak dapat membuka kamera")
            
    def enroll_user(self, user_name, required_samples=5):
        """Enhanced enrollment dengan visual guides dan quality metrics"""
        print(f"🎯 Memulai enrollment untuk user: {user_name}")
        print(f"📊 Dibutuhkan {required_samples} sampel berkualitas tinggi")
        print("\n📋 INSTRUKSI:")
        print("1. 👁️  Posisikan wajah dalam guide oval putih")
        print("2. 🚫 Pastikan tidak ada obstacle (kacamata, topi, masker)")
        print("3. 👀 Berkedip natural beberapa kali untuk liveness")
        print("4. 🤖 Capture otomatis saat kondisi optimal")
        print("5. ⌨️  Tekan 'q' untuk quit")
        print("\n🎨 GUIDE WARNA:")
        print("   🟢 Hijau = Optimal (Auto capture)")
        print("   🟡 Kuning = Perlu adjustment")
        print("   🔴 Merah = Perbaiki posisi/hapus obstacle")
        
        collected_samples = 0
        embeddings = []
        quality_logs = []
        last_capture_time = 0
        capture_cooldown = 2.0  # 2 seconds between captures
        
        # Reset detectors
        self.liveness_detector.reset()
        
        while collected_samples < required_samples:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Flip frame untuk mirror effect
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Extract embedding dan face landmarks
            embedding, bbox = self.embedding_system.extract_embedding(frame)
            landmarks = None
            
            if embedding is not None:
                x1, y1, x2, y2 = bbox
                
                # Get face landmarks untuk detailed analysis
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_mesh_results = self.camera_guide.face_mesh.process(rgb_frame)
                
                if face_mesh_results.multi_face_landmarks:
                    landmarks = face_mesh_results.multi_face_landmarks[0]
                
                # Enhanced obstacle detection
                obstacles, obstacle_confidence = self.obstacle_detector.detect_obstacles(frame, bbox)
                
                # Deteksi liveness dengan enhanced metrics
                is_face_detected, ear = self.liveness_detector.detect_blink(frame)
                
                # Analyze frame quality
                quality_metrics = self.camera_guide.analyze_frame_quality(frame, bbox, landmarks)
                
                # Draw camera guides
                display_frame = self.camera_guide.draw_face_guide(display_frame, bbox)
                display_frame = self.camera_guide.draw_eye_area_guides(display_frame, landmarks)
                
                # Draw eye landmarks untuk debugging
                if landmarks:
                    self.liveness_detector.draw_eye_landmarks(display_frame, landmarks)
                
                # Status panel
                status_y = 30
                panel_bg = (0, 0, 0)  # Black background for text
                
                # Sample progress
                progress_text = f"📊 Samples: {collected_samples}/{required_samples}"
                cv2.putText(display_frame, progress_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                status_y += 25
                blink_text = f"👁️  Blinks: {self.liveness_detector.valid_blinks} (Total: {self.liveness_detector.total_blinks})"
                cv2.putText(display_frame, blink_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                status_y += 25
                quality_text = f"🎯 Quality: {quality_metrics['overall_score']:.2f}"
                quality_color = (0, 255, 0) if quality_metrics['overall_score'] > self.quality_threshold else (0, 255, 255)
                cv2.putText(display_frame, quality_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
                
                # Obstacles info
                status_y += 25
                if obstacles:
                    obstacle_text = f"🚫 Obstacles: {', '.join(obstacles)}"
                    for i, (obstacle, conf) in enumerate(obstacle_confidence.items()):
                        if i == 0:
                            cv2.putText(display_frame, obstacle_text, (10, status_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            status_y += 20
                            cv2.putText(display_frame, f"   {obstacle}: {conf:.2f}", (10, status_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.putText(display_frame, "✅ No obstacles", (10, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Eye visibility
                status_y += 25
                eye_score = self.liveness_detector.eye_visibility_score
                eye_text = f"👁️  Eye Visibility: {eye_score:.2f}"
                eye_color = (0, 255, 0) if eye_score > 0.8 else (0, 255, 255) if eye_score > 0.6 else (0, 0, 255)
                cv2.putText(display_frame, eye_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, eye_color, 1)
                
                # Validation status (adjusted thresholds)
                is_valid_sample = (len(obstacles) == 0 and 
                                 self.liveness_detector.valid_blinks >= 1 and  # Reduced blink requirement
                                 quality_metrics['overall_score'] > self.quality_threshold and
                                 eye_score > 0.5)  # Lowered eye score requirement
                
                current_time = time.time()
                status_y += 30
                
                # Auto capture logic
                if is_valid_sample and (current_time - last_capture_time) > capture_cooldown:
                    # Auto capture!
                    embeddings.append(embedding)
                    collected_samples += 1
                    last_capture_time = current_time
                    
                    print(f"✅ Auto capture {collected_samples}! Quality: {quality_metrics['overall_score']:.3f}")
                    
                    # Reset liveness detector for next sample
                    self.liveness_detector.reset()
                    
                    cv2.putText(display_frame, f"✅ AUTO CAPTURED {collected_samples}/{required_samples}!", 
                               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                elif is_valid_sample:
                    # Ready but in cooldown
                    cooldown_remaining = capture_cooldown - (current_time - last_capture_time)
                    cv2.putText(display_frame, f"🕐 Next capture in {cooldown_remaining:.1f}s", 
                               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    issues = []
                    if obstacles:
                        issues.append("Remove obstacles")
                    if self.liveness_detector.valid_blinks < 1:
                        issues.append("Blink naturally")
                    if quality_metrics['overall_score'] <= self.quality_threshold:
                        issues.append("Improve position/lighting")
                    if eye_score <= 0.5:
                        issues.append("Eyes not clear")
                    
                    issue_text = "⚠️  " + ", ".join(issues[:2])  # Show first 2 issues
                    cv2.putText(display_frame, issue_text, (10, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Log quality metrics
                if len(quality_logs) == 0 or len(quality_logs) % 30 == 0:  # Log every second
                    log_entry = {
                        'timestamp': float(time.time()),
                        'quality': self.json_serializable(quality_metrics),
                        'obstacles': list(obstacles),
                        'blinks': int(self.liveness_detector.valid_blinks),
                        'is_valid': bool(is_valid_sample)
                    }
                    quality_logs.append(log_entry)
                    logger.info(f"Frame quality: {quality_metrics['overall_score']:.3f}, "
                              f"Obstacles: {obstacles}, Valid blinks: {self.liveness_detector.valid_blinks}")
            else:
                # No face detected - show guides only
                display_frame = self.camera_guide.draw_face_guide(display_frame)
                display_frame = self.camera_guide.draw_eye_area_guides(display_frame)
                
                cv2.putText(display_frame, "❌ No face detected - Position face in guide", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Enhanced Face Enrollment - Auto Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
        
        # Save embeddings and logs
        if embeddings:
            for embedding in embeddings:
                self.embedding_system.add_user_embedding(user_name, embedding)
            
            self.embedding_system.save_database(self.db_path)
            
            # Save quality logs
            log_filename = f"enrollment_log_{user_name}_{int(time.time())}.json"
            with open(log_filename, 'w') as f:
                json.dump({
                    'user_name': user_name,
                    'samples_collected': len(embeddings),
                    'quality_logs': quality_logs,
                    'average_quality': float(np.mean([log['quality']['overall_score'] for log in quality_logs]))
                }, f, indent=2, default=json_serializable)
            
            print(f"✅ Enrollment berhasil!")
            print(f"   📊 {len(embeddings)} sampel berkualitas disimpan")
            print(f"   📝 Log disimpan ke: {log_filename}")
            print(f"   🎯 Rata-rata kualitas: {np.mean([log['quality']['overall_score'] for log in quality_logs]):.3f}")
        else:
            print("❌ Enrollment gagal! Tidak ada sampel yang memenuhi standar kualitas.")
            
    def authenticate_user(self, user_name=None):
        """Enhanced authentication dengan durasi minimum 3 detik dan validasi blink"""
        mode = "VERIFICATION" if user_name else "IDENTIFICATION"
        target = f" for {user_name}" if user_name else ""
        
        print(f"🔐 Memulai {mode}{target}")
        print("📋 INSTRUKSI:")
        print("1. 👁️  Posisikan wajah dalam guide oval")
        print("2. 🚫 Pastikan tidak ada obstacle")
        print("3. 👀 Berkedip natural untuk liveness (minimal 2x)")
        print("4. ⏱️  Proses minimal 3 detik dengan kondisi stabil")
        print("5. ⌨️  Tekan 'q' untuk quit")
        
        start_time = time.time()
        auth_timeout = 30
        min_duration = 3.0  # Minimum 3 seconds
        min_blinks = 2  # Minimum blinks required
        min_frames = 5  # Minimum frames with good conditions
        
        auth_logs = []
        valid_frames = []
        stable_start_time = None
        
        # Reset liveness detector
        self.liveness_detector.reset()
        
        while time.time() - start_time < auth_timeout:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            current_time = time.time()
            elapsed_time = current_time - start_time
            remaining_time = max(0, auth_timeout - elapsed_time)
            
            # Extract embedding and landmarks
            embedding, bbox = self.embedding_system.extract_embedding(frame)
            landmarks = None
            
            if embedding is not None:
                x1, y1, x2, y2 = bbox
                
                # Get face landmarks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_mesh_results = self.camera_guide.face_mesh.process(rgb_frame)
                
                if face_mesh_results.multi_face_landmarks:
                    landmarks = face_mesh_results.multi_face_landmarks[0]
                
                # Enhanced obstacle detection
                obstacles, obstacle_confidence = self.obstacle_detector.detect_obstacles(frame, bbox)
                
                # Deteksi liveness
                is_face_detected, ear = self.liveness_detector.detect_blink(frame)
                
                # Analyze frame quality
                quality_metrics = self.camera_guide.analyze_frame_quality(frame, bbox, landmarks)
                
                # Draw camera guides
                display_frame = self.camera_guide.draw_face_guide(display_frame, bbox)
                display_frame = self.camera_guide.draw_eye_area_guides(display_frame, landmarks)
                
                # Draw eye landmarks
                if landmarks:
                    self.liveness_detector.draw_eye_landmarks(display_frame, landmarks)
                
                # Status panel
                status_y = 30
                
                # Mode and timer
                timer_color = (0, 255, 0) if remaining_time > 10 else (0, 255, 255) if remaining_time > 5 else (0, 0, 255)
                cv2.putText(display_frame, f"🔐 {mode} - Time: {remaining_time:.1f}s", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 2)
                
                status_y += 25
                blink_text = f"👁️  Valid Blinks: {self.liveness_detector.valid_blinks}"
                cv2.putText(display_frame, blink_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                status_y += 25
                quality_text = f"🎯 Quality: {quality_metrics['overall_score']:.2f}"
                quality_color = (0, 255, 0) if quality_metrics['overall_score'] > self.quality_threshold else (0, 255, 255)
                cv2.putText(display_frame, quality_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
                
                # Obstacles
                status_y += 25
                if obstacles:
                    obstacle_text = f"🚫 Issues: {', '.join(obstacles)}"
                    cv2.putText(display_frame, obstacle_text, (10, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(display_frame, "✅ Clear", (10, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Check if current conditions are good for authentication
                current_conditions_good = (
                    len(obstacles) == 0 and 
                    quality_metrics['overall_score'] > self.quality_threshold
                )
                
                if current_conditions_good:
                    # Add to valid frames
                    valid_frames.append({
                        'embedding': embedding,
                        'quality': quality_metrics['overall_score'],
                        'timestamp': current_time
                    })
                    
                    # Start stable period timer
                    if stable_start_time is None:
                        stable_start_time = current_time
                        print("🕐 Kondisi stabil terdeteksi, memulai validasi...")
                    
                    # Check validation progress
                    stable_duration = current_time - stable_start_time
                    enough_blinks = self.liveness_detector.total_blinks >= min_blinks
                    enough_time = stable_duration >= min_duration
                    enough_frames = len(valid_frames) >= min_frames
                    
                    # Progress display
                    status_y += 25
                    progress_text = f"⏱️ Duration: {stable_duration:.1f}/{min_duration}s | 👁️ Blinks: {self.liveness_detector.total_blinks}/{min_blinks} | 📊 Frames: {len(valid_frames)}/{min_frames}"
                    cv2.putText(display_frame, progress_text, (10, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    if enough_time and enough_blinks and enough_frames:
                        # Authentication ready - use best quality frame
                        best_frame = max(valid_frames, key=lambda x: x['quality'])
                        best_embedding = best_frame['embedding']
                        
                        print(f"✅ Validasi selesai! Durasi: {stable_duration:.1f}s, Blinks: {self.liveness_detector.total_blinks}")
                        
                        # Perform authentication
                        if user_name:
                            # Verification mode
                            is_verified, similarity = self.embedding_system.verify_user(best_embedding, user_name)
                            
                            auth_log = {
                                'timestamp': float(current_time),
                                'mode': 'verification',
                                'target_user': user_name,
                                'similarity': float(similarity),
                                'is_verified': bool(is_verified),
                                'quality': self.json_serializable(best_frame['quality']),
                                'obstacles': list(obstacles),
                                'blinks': int(self.liveness_detector.total_blinks),
                                'validation_duration': float(stable_duration),
                                'frames_analyzed': len(valid_frames)
                            }
                            auth_logs.append(auth_log)
                            
                            if is_verified:
                                success_text = f"✅ VERIFIED: {user_name}"
                                cv2.putText(display_frame, success_text, 
                                           (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.8, (0, 255, 0), 2)
                                
                                similarity_text = f"🎯 Similarity: {similarity:.3f} | Duration: {stable_duration:.1f}s"
                                cv2.putText(display_frame, similarity_text, 
                                           (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.5, (0, 255, 0), 2)
                                
                                cv2.imshow('Enhanced Authentication - Auto Validation', display_frame)
                                cv2.waitKey(2000)
                                
                                logger.info(f"✅ User {user_name} verified successfully. Similarity: {similarity:.3f}, Duration: {stable_duration:.1f}s")
                                return True, user_name, similarity
                            else:
                                fail_text = f"❌ VERIFICATION FAILED"
                                cv2.putText(display_frame, fail_text, 
                                           (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.7, (0, 0, 255), 2)
                                
                                similarity_text = f"📊 Similarity: {similarity:.3f} (Need > {self.embedding_system.verification_threshold})"
                                cv2.putText(display_frame, similarity_text, 
                                           (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.4, (0, 0, 255), 1)
                        else:
                            # Identification mode
                            identified_user, similarity = self.embedding_system.identify_user(best_embedding)
                            
                            auth_log = {
                                'timestamp': float(current_time),
                                'mode': 'identification',
                                'identified_user': identified_user,
                                'similarity': float(similarity),
                                'quality': self.json_serializable(best_frame['quality']),
                                'obstacles': list(obstacles),
                                'blinks': int(self.liveness_detector.total_blinks),
                                'validation_duration': float(stable_duration),
                                'frames_analyzed': len(valid_frames)
                            }
                            auth_logs.append(auth_log)
                            
                            if identified_user:
                                success_text = f"✅ IDENTIFIED: {identified_user}"
                                cv2.putText(display_frame, success_text, 
                                           (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.8, (0, 255, 0), 2)
                                
                                similarity_text = f"🎯 Confidence: {similarity:.3f} | Duration: {stable_duration:.1f}s"
                                cv2.putText(display_frame, similarity_text, 
                                           (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.5, (0, 255, 0), 2)
                                
                                cv2.imshow('Enhanced Authentication - Auto Validation', display_frame)
                                cv2.waitKey(2000)
                                
                                logger.info(f"✅ User identified as {identified_user}. Confidence: {similarity:.3f}, Duration: {stable_duration:.1f}s")
                                return True, identified_user, similarity
                            else:
                                # User not found
                                fail_text = f"❌ USER NOT FOUND"
                                cv2.putText(display_frame, fail_text, 
                                           (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.7, (0, 0, 255), 2)
                                
                                similarity_text = f"📊 Max Similarity: {similarity:.3f}"
                                cv2.putText(display_frame, similarity_text, 
                                           (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.5, (0, 0, 255), 1)
                else:
                    # Reset stable period if conditions not met
                    if stable_start_time is not None:
                        print("⚠️ Kondisi tidak stabil, reset validasi...")
                    stable_start_time = None
                    valid_frames = []
                    
                    # Show what's needed
                    status_y += 25
                    issues = []
                    if obstacles:
                        issues.append("Remove obstacles")
                    if quality_metrics['overall_score'] <= self.quality_threshold:
                        issues.append("Improve position/lighting")
                    if self.liveness_detector.total_blinks < min_blinks:
                        issues.append("Blink naturally")
                    
                    issue_text = "⚠️ Need: " + ", ".join(issues[:3])
                    cv2.putText(display_frame, issue_text, (10, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # No face detected
                display_frame = self.camera_guide.draw_face_guide(display_frame)
                display_frame = self.camera_guide.draw_eye_area_guides(display_frame)
                
                timer_color = (0, 255, 0) if remaining_time > 10 else (0, 255, 255) if remaining_time > 5 else (0, 0, 255)
                cv2.putText(display_frame, f"🔐 {mode} - Time: {remaining_time:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 2)
                
                cv2.putText(display_frame, "❌ No face detected - Position face in guide", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Enhanced Authentication - Auto Validation', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Timeout or manual quit
        logger.info(f"❌ Authentication timeout or cancelled. Mode: {mode}, Attempts: {len(auth_logs)}")
        
        # Save authentication log
        if auth_logs:
            log_filename = f"auth_log_{mode.lower()}_{int(time.time())}.json"
            with open(log_filename, 'w') as f:
                json.dump({
                    'mode': mode,
                    'target_user': user_name,
                    'timestamp': float(start_time),
                    'duration': float(time.time() - start_time),
                    'attempts': len(auth_logs),
                    'logs': auth_logs
                }, f, indent=2, default=json_serializable)
            
            print(f"📝 Authentication log saved: {log_filename}")
        
        return False, None, 0.0
    
    def list_users(self):
        """List semua user yang terdaftar"""
        users = list(self.embedding_system.user_embeddings.keys())
        return users
    
    def delete_user(self, user_name):
        """Hapus user dari database"""
        if user_name in self.embedding_system.user_embeddings:
            del self.embedding_system.user_embeddings[user_name]
            
            # Rebuild FAISS index
            self.embedding_system.index = faiss.IndexFlatIP(self.embedding_system.dimension)
            self.embedding_system.user_names = []
            
            for uname in self.embedding_system.user_embeddings:
                for embedding in self.embedding_system.user_embeddings[uname]:
                    embedding_normalized = embedding / np.linalg.norm(embedding)
                    self.embedding_system.index.add(embedding_normalized.reshape(1, -1))
                    self.embedding_system.user_names.append(uname)
            
            self.embedding_system.save_database(self.db_path)
            return True
        return False
    
    def close(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function untuk terminal interface"""
    auth_system = SecureFaceAuth()
    
    try:
        auth_system.initialize_camera()
        
        while True:
            print("\n" + "="*50)
            print("SECURE FACE AUTHENTICATION SYSTEM")
            print("="*50)
            print("1. Enroll User")
            print("2. Authenticate User (Verify)")
            print("3. Identify User")
            print("4. List Users")
            print("5. Delete User")
            print("6. Exit")
            print("="*50)
            
            choice = input("Pilih opsi (1-6): ").strip()
            
            if choice == '1':
                user_name = input("Masukkan nama user: ").strip()
                if user_name:
                    samples = input("Jumlah sampel (default: 5): ").strip()
                    samples = int(samples) if samples.isdigit() else 5
                    auth_system.enroll_user(user_name, samples)
                else:
                    print("Nama user tidak valid!")
                    
            elif choice == '2':
                users = auth_system.list_users()
                if not users:
                    print("Belum ada user yang terdaftar!")
                    continue
                    
                print("User terdaftar:")
                for i, user in enumerate(users, 1):
                    print(f"{i}. {user}")
                    
                user_choice = input("Pilih user untuk verifikasi: ").strip()
                if user_choice.isdigit() and 1 <= int(user_choice) <= len(users):
                    selected_user = users[int(user_choice) - 1]
                    success, user, similarity = auth_system.authenticate_user(selected_user)
                    
                    if success:
                        print(f"✅ Autentikasi BERHASIL!")
                        print(f"User: {user}")
                        print(f"Similarity: {similarity:.3f}")
                    else:
                        print("❌ Autentikasi GAGAL!")
                else:
                    print("Pilihan tidak valid!")
                    
            elif choice == '3':
                success, user, similarity = auth_system.authenticate_user()
                
                if success:
                    print(f"✅ User teridentifikasi: {user}")
                    print(f"Similarity: {similarity:.3f}")
                else:
                    print("❌ User tidak dikenali!")
                    
            elif choice == '4':
                users = auth_system.list_users()
                if users:
                    print("User terdaftar:")
                    for i, user in enumerate(users, 1):
                        count = len(auth_system.embedding_system.user_embeddings[user])
                        print(f"{i}. {user} ({count} samples)")
                else:
                    print("Belum ada user yang terdaftar!")
                    
            elif choice == '5':
                users = auth_system.list_users()
                if not users:
                    print("Belum ada user yang terdaftar!")
                    continue
                    
                print("User terdaftar:")
                for i, user in enumerate(users, 1):
                    print(f"{i}. {user}")
                    
                user_choice = input("Pilih user untuk dihapus: ").strip()
                if user_choice.isdigit() and 1 <= int(user_choice) <= len(users):
                    selected_user = users[int(user_choice) - 1]
                    confirm = input(f"Yakin ingin menghapus user '{selected_user}'? (y/N): ").strip().lower()
                    
                    if confirm == 'y':
                        if auth_system.delete_user(selected_user):
                            print(f"User '{selected_user}' berhasil dihapus!")
                        else:
                            print("Gagal menghapus user!")
                    else:
                        print("Penghapusan dibatalkan.")
                else:
                    print("Pilihan tidak valid!")
                    
            elif choice == '6':
                print("Terima kasih! Sampai jumpa.")
                break
                
            else:
                print("Pilihan tidak valid! Silakan pilih 1-6.")
                
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        auth_system.close()

if __name__ == "__main__":
    main()
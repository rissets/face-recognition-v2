#!/usr/bin/env python3
"""
Real-time Liveness Detection System
==================================

Advanced liveness detection to distinguish between real faces and fake/spoofed faces.
Combines multiple detection techniques for maximum accuracy.

Features:
- Real-time blink detection with EAR analysis
- 3D head movement tracking
- Texture analysis for printed photos
- Light reflection analysis
- Multi-modal scoring system
- Anti-spoofing with obstacle detection

Author: Face Recognition Team
Version: 2.0
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LivenessResult:
    """Result of liveness detection"""
    is_live: bool
    confidence: float
    score_breakdown: Dict[str, float]
    challenges_passed: List[str]
    challenges_failed: List[str] 
    frame_analysis: Dict[str, Any]
    
class RealtimeLivenessDetector:
    """
    Real-time liveness detection system with multiple anti-spoofing techniques
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the real-time liveness detector"""
        logger.info("Initializing RealtimeLivenessDetector...")
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Initialize MediaPipe components
        self._init_mediapipe()
        
        # Initialize detection parameters
        self._init_parameters()
        
        # Initialize tracking variables
        self._init_tracking_vars()
        
        logger.info("RealtimeLivenessDetector initialized successfully")
    
    def _load_config(self, config: Optional[Dict]) -> Dict:
        """Load configuration with defaults"""
        default_config = {
            # Blink detection
            'ear_threshold': 0.25,
            'blink_frames_threshold': 3,
            'min_blinks_required': 2,
            'blink_timeout': 10.0,  # seconds
            
            # Head movement
            'head_movement_threshold': 15.0,  # degrees
            'movement_timeout': 8.0,  # seconds
            
            # Texture analysis
            'texture_variance_threshold': 100,
            'edge_density_threshold': 0.1,
            
            # 3D depth analysis
            'depth_variation_threshold': 0.05,
            'nose_depth_threshold': 0.02,
            
            # Light reflection
            'reflection_threshold': 200,
            'reflection_area_min': 0.001,
            
            # Challenge system
            'enable_challenges': True,
            'challenge_timeout': 15.0,  # seconds
            
            # Scoring
            'liveness_threshold': 0.7,
            'spoof_penalty': -0.3,
            
            # Quality requirements
            'min_face_size': 100,
            'max_face_size': 400,
            'brightness_range': (50, 200),
        }
        
        if config:
            default_config.update(config)
            
        return default_config
    
    def _init_mediapipe(self):
        """Initialize MediaPipe components"""
        # Face mesh for detailed landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Face detection for bbox
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def _init_parameters(self):
        """Initialize detection parameters"""
        # Eye landmarks for blink detection
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
        
        # Face landmarks for head pose estimation
        self.FACE_POSE_LANDMARKS = [
            1,    # nose tip
            33,   # left eye outer corner
            362,  # right eye outer corner
            61,   # left mouth corner
            291,  # right mouth corner
            199   # chin
        ]
        
        # Landmark indices for different regions
        self.FOREHEAD_LANDMARKS = [9, 10, 151, 337, 299, 333, 298, 301]
        self.NOSE_LANDMARKS = [1, 2, 5, 4, 6, 19, 20]
        self.MOUTH_LANDMARKS = [0, 17, 18, 200, 199, 175, 13, 269, 270, 267]
    
    def _init_tracking_vars(self):
        """Initialize tracking variables"""
        # Blink tracking
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_time = 0
        self.ear_history = deque(maxlen=30)
        self.baseline_ear = None
        
        # Head movement tracking
        self.head_poses = deque(maxlen=20)
        self.movement_detected = False
        self.movement_start_time = None
        
        # Challenge system
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenges_completed = []
        self.challenge_queue = ['blink', 'turn_left', 'turn_right', 'nod']
        
        # Frame analysis history
        self.frame_scores = deque(maxlen=10)
        self.spoof_indicators = deque(maxlen=5)
        
        # Session tracking
        self.session_start_time = time.time()
        self.frames_processed = 0
        
        # Detection state
        self.detection_active = False
        self.final_result = None
    
    def start_detection(self) -> None:
        """Start the liveness detection session"""
        logger.info("Starting liveness detection session")
        self.detection_active = True
        self.session_start_time = time.time()
        self._reset_session()
    
    def stop_detection(self) -> LivenessResult:
        """Stop detection and return final result"""
        logger.info("Stopping liveness detection session")
        self.detection_active = False
        
        if self.final_result is None:
            self.final_result = self._calculate_final_result()
            
        return self.final_result
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame for liveness detection
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, analysis_result)
        """
        if not self.detection_active:
            return frame, {'status': 'detection_not_active'}
        
        self.frames_processed += 1
        current_time = time.time()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        face_detection_results = self.face_detection.process(rgb_frame)
        face_mesh_results = self.face_mesh.process(rgb_frame)
        
        if not face_detection_results.detections or not face_mesh_results.multi_face_landmarks:
            return self._draw_no_face_detected(frame), {'status': 'no_face_detected'}
        
        # Get primary face
        face_bbox = self._get_face_bbox(face_detection_results.detections[0], frame.shape)
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        
        # Frame quality analysis
        frame_quality = self._analyze_frame_quality(frame, face_bbox)
        
        if frame_quality['overall_quality'] < 0.4:
            return self._draw_poor_quality(frame), {'status': 'poor_quality', 'quality': frame_quality}
        
        # Core liveness analysis
        analysis_result = self._analyze_liveness(frame, face_bbox, face_landmarks, current_time)
        
        # Update tracking
        self._update_tracking(analysis_result, current_time)
        
        # Check for completion
        if self._should_complete_detection(current_time):
            self.final_result = self._calculate_final_result()
            self.detection_active = False
        
        # Draw annotations
        annotated_frame = self._draw_annotations(frame, face_bbox, face_landmarks, analysis_result)
        
        return annotated_frame, analysis_result
    
    def _analyze_liveness(self, frame: np.ndarray, face_bbox: Tuple, 
                         landmarks: Any, current_time: float) -> Dict:
        """Comprehensive liveness analysis"""
        
        analysis = {
            'timestamp': current_time,
            'blink_analysis': self._analyze_blinks(landmarks),
            'head_movement': self._analyze_head_movement(landmarks),
            'texture_analysis': self._analyze_texture(frame, face_bbox),
            'depth_analysis': self._analyze_3d_depth(landmarks),
            'reflection_analysis': self._analyze_reflections(frame, face_bbox, landmarks),
            'spoof_detection': self._detect_spoofing_indicators(frame, face_bbox, landmarks),
            'challenge_status': self._check_challenge_completion(landmarks, current_time)
        }
        
        # Calculate frame score
        frame_score = self._calculate_frame_score(analysis)
        analysis['frame_score'] = frame_score
        
        return analysis
    
    def _analyze_blinks(self, landmarks: Any) -> Dict:
        """Analyze eye blinks using EAR (Eye Aspect Ratio)"""
        try:
            # Calculate EAR for both eyes
            left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_POINTS)
            right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_POINTS)
            
            if left_ear is None or right_ear is None:
                return {'status': 'error', 'ear': 0.0}
            
            avg_ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(avg_ear)
            
            # Calculate adaptive baseline
            if len(self.ear_history) >= 10:
                if self.baseline_ear is None:
                    self.baseline_ear = np.mean(list(self.ear_history)[-10:])
                else:
                    # Smooth update
                    self.baseline_ear = 0.9 * self.baseline_ear + 0.1 * avg_ear
                
                # Dynamic threshold
                threshold = self.baseline_ear * 0.75
                
                # Blink detection
                if avg_ear < threshold:
                    self.blink_counter += 1
                else:
                    if self.blink_counter >= self.config['blink_frames_threshold']:
                        self.total_blinks += 1
                        self.last_blink_time = time.time()
                        logger.info(f"Blink detected! Total: {self.total_blinks}")
                    self.blink_counter = 0
                
                return {
                    'status': 'active',
                    'ear': avg_ear,
                    'threshold': threshold,
                    'baseline': self.baseline_ear,
                    'blink_counter': self.blink_counter,
                    'total_blinks': self.total_blinks,
                    'is_blinking': self.blink_counter > 0
                }
            else:
                return {
                    'status': 'calibrating',
                    'ear': avg_ear,
                    'frames_needed': 10 - len(self.ear_history)
                }
                
        except Exception as e:
            logger.error(f"Error in blink analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_ear(self, landmarks: Any, eye_points: List[int]) -> Optional[float]:
        """Calculate Eye Aspect Ratio"""
        try:
            points = []
            for idx in eye_points:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    points.append([landmark.x, landmark.y])
            
            if len(points) != 6:
                return None
            
            points = np.array(points)
            
            # Calculate distances
            A = np.linalg.norm(points[1] - points[5])  # vertical
            B = np.linalg.norm(points[2] - points[4])  # vertical
            C = np.linalg.norm(points[0] - points[3])  # horizontal
            
            if C < 0.001:
                return None
            
            ear = (A + B) / (2.0 * C)
            return ear
            
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return None
    
    def _analyze_head_movement(self, landmarks: Any) -> Dict:
        """Analyze head pose and movement"""
        try:
            # Get 3D model points (approximate)
            model_points = np.array([
                (0.0, 0.0, 0.0),       # Nose tip
                (-30.0, -125.0, -30.0), # Left eye left corner
                (30.0, -125.0, -30.0),  # Right eye right corner
                (-20.0, 25.0, -30.0),   # Left mouth corner
                (20.0, 25.0, -30.0),    # Right mouth corner
                (0.0, 110.0, -10.0)     # Chin
            ], dtype=np.float64)
            
            # Get 2D image points
            image_points = []
            for idx in self.FACE_POSE_LANDMARKS:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    image_points.append([landmark.x * 640, landmark.y * 480])  # Assume 640x480
            
            if len(image_points) != 6:
                return {'status': 'insufficient_landmarks'}
            
            image_points = np.array(image_points, dtype=np.float64)
            
            # Camera parameters (approximate)
            focal_length = 640
            center = (320, 240)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if not success:
                return {'status': 'pose_estimation_failed'}
            
            # Convert rotation vector to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
            
            pitch, yaw, roll = angles
            
            # Store pose history
            pose = {'pitch': pitch, 'yaw': yaw, 'roll': roll, 'timestamp': time.time()}
            self.head_poses.append(pose)
            
            # Calculate movement
            movement_score = 0.0
            if len(self.head_poses) >= 5:
                poses = list(self.head_poses)[-5:]
                yaw_range = max(p['yaw'] for p in poses) - min(p['yaw'] for p in poses)
                pitch_range = max(p['pitch'] for p in poses) - min(p['pitch'] for p in poses)
                
                movement_score = max(yaw_range, pitch_range)
            
            return {
                'status': 'active',
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'movement_score': movement_score,
                'sufficient_movement': movement_score > self.config['head_movement_threshold']
            }
            
        except Exception as e:
            logger.error(f"Error in head movement analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles"""
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        return (math.degrees(x), math.degrees(y), math.degrees(z))
    
    def _analyze_texture(self, frame: np.ndarray, face_bbox: Tuple) -> Dict:
        """Analyze texture to detect printed photos or screens"""
        try:
            x1, y1, x2, y2 = face_bbox
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return {'status': 'invalid_roi'}
            
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Texture variance
            texture_variance = np.var(gray_face)
            
            # Edge density
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Local Binary Pattern analysis
            lbp_variance = self._calculate_lbp_variance(gray_face)
            
            # Frequency domain analysis
            freq_analysis = self._analyze_frequency_domain(gray_face)
            
            # Combine scores
            texture_score = min(1.0, (
                min(texture_variance / 1000, 1.0) * 0.3 +
                min(edge_density / 0.2, 1.0) * 0.3 +
                lbp_variance * 0.2 +
                freq_analysis * 0.2
            ))
            
            return {
                'status': 'active',
                'texture_variance': texture_variance,
                'edge_density': edge_density,
                'lbp_variance': lbp_variance,
                'frequency_score': freq_analysis,
                'texture_score': texture_score,
                'is_natural': texture_score > 0.5
            }
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_lbp_variance(self, gray_image: np.ndarray) -> float:
        """Calculate Local Binary Pattern variance"""
        try:
            h, w = gray_image.shape
            lbp = np.zeros((h-2, w-2), dtype=np.uint8)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray_image[i, j]
                    code = 0
                    
                    # 8-connected neighbors
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i-1, j-1] = code
            
            # Calculate variance of LBP histogram
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            return np.var(hist)
            
        except Exception as e:
            logger.error(f"Error calculating LBP variance: {e}")
            return 0.0
    
    def _analyze_frequency_domain(self, gray_image: np.ndarray) -> float:
        """Analyze frequency domain characteristics"""
        try:
            # FFT analysis
            f_transform = np.fft.fft2(gray_image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # High frequency content
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            
            # Create mask for high frequencies
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x)**2 + (y - center_y)**2) > (min(h, w) * 0.1)**2
            
            high_freq_energy = np.mean(magnitude_spectrum[mask])
            total_energy = np.mean(magnitude_spectrum)
            
            freq_ratio = high_freq_energy / (total_energy + 1e-7)
            
            return min(1.0, freq_ratio)
            
        except Exception as e:
            logger.error(f"Error in frequency analysis: {e}")
            return 0.0
    
    def _analyze_3d_depth(self, landmarks: Any) -> Dict:
        """Analyze 3D depth characteristics"""
        try:
            # Get key facial points
            nose_tip = landmarks.landmark[1]  # Nose tip
            left_eye = landmarks.landmark[33]  # Left eye outer corner
            right_eye = landmarks.landmark[362]  # Right eye outer corner
            
            # Calculate relative depth using landmark positions
            # In a real face, nose should be closer than eyes
            nose_z = nose_tip.z if hasattr(nose_tip, 'z') else 0
            left_eye_z = left_eye.z if hasattr(left_eye, 'z') else 0
            right_eye_z = right_eye.z if hasattr(right_eye, 'z') else 0
            
            # Depth variation
            depth_values = [nose_z, left_eye_z, right_eye_z]
            depth_variation = np.std(depth_values) if len(depth_values) > 1 else 0
            
            # Nose prominence
            avg_eye_z = (left_eye_z + right_eye_z) / 2
            nose_prominence = abs(nose_z - avg_eye_z)
            
            return {
                'status': 'active',
                'nose_z': nose_z,
                'avg_eye_z': avg_eye_z,
                'depth_variation': depth_variation,
                'nose_prominence': nose_prominence,
                'has_depth': depth_variation > self.config['depth_variation_threshold']
            }
            
        except Exception as e:
            logger.error(f"Error in depth analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_reflections(self, frame: np.ndarray, face_bbox: Tuple, landmarks: Any) -> Dict:
        """Analyze light reflections in eyes"""
        try:
            # Extract eye regions
            h, w = frame.shape[:2]
            
            # Get eye landmarks
            left_eye_landmarks = [landmarks.landmark[i] for i in self.LEFT_EYE_POINTS]
            right_eye_landmarks = [landmarks.landmark[i] for i in self.RIGHT_EYE_POINTS]
            
            # Convert to pixel coordinates
            left_eye_points = [(int(lm.x * w), int(lm.y * h)) for lm in left_eye_landmarks]
            right_eye_points = [(int(lm.x * w), int(lm.y * h)) for lm in right_eye_landmarks]
            
            # Extract eye regions
            left_eye_roi = self._extract_eye_roi(frame, left_eye_points)
            right_eye_roi = self._extract_eye_roi(frame, right_eye_points)
            
            # Analyze reflections in each eye
            left_reflection = self._detect_eye_reflection(left_eye_roi)
            right_reflection = self._detect_eye_reflection(right_eye_roi)
            
            return {
                'status': 'active',
                'left_eye_reflection': left_reflection,
                'right_eye_reflection': right_reflection,
                'has_natural_reflections': left_reflection['has_reflection'] or right_reflection['has_reflection']
            }
            
        except Exception as e:
            logger.error(f"Error in reflection analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _extract_eye_roi(self, frame: np.ndarray, eye_points: List[Tuple]) -> np.ndarray:
        """Extract eye region of interest"""
        try:
            if len(eye_points) < 4:
                return np.array([])
            
            # Get bounding box
            xs = [p[0] for p in eye_points]
            ys = [p[1] for p in eye_points]
            
            x1, x2 = max(0, min(xs) - 5), min(frame.shape[1], max(xs) + 5)
            y1, y2 = max(0, min(ys) - 5), min(frame.shape[0], max(ys) + 5)
            
            return frame[y1:y2, x1:x2]
            
        except Exception as e:
            logger.error(f"Error extracting eye ROI: {e}")
            return np.array([])
    
    def _detect_eye_reflection(self, eye_roi: np.ndarray) -> Dict:
        """Detect reflections in eye region"""
        try:
            if eye_roi.size == 0:
                return {'has_reflection': False, 'reflection_score': 0.0}
            
            # Convert to grayscale
            gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            
            # Find bright spots (potential reflections)
            _, bright_mask = cv2.threshold(gray_eye, self.config['reflection_threshold'], 255, cv2.THRESH_BINARY)
            
            # Calculate reflection area
            reflection_area = np.sum(bright_mask > 0) / bright_mask.size
            
            # Check for circular/oval reflections (more natural)
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            natural_reflections = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5:  # Minimum reflection size
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.5:  # Reasonably circular
                            natural_reflections += 1
            
            return {
                'has_reflection': reflection_area > self.config['reflection_area_min'],
                'reflection_score': min(1.0, reflection_area / 0.1),
                'natural_reflections': natural_reflections,
                'reflection_area': reflection_area
            }
            
        except Exception as e:
            logger.error(f"Error detecting eye reflection: {e}")
            return {'has_reflection': False, 'reflection_score': 0.0}
    
    def _detect_spoofing_indicators(self, frame: np.ndarray, face_bbox: Tuple, landmarks: Any) -> Dict:
        """Detect various spoofing indicators"""
        try:
            x1, y1, x2, y2 = face_bbox
            face_roi = frame[y1:y2, x1:x2]
            
            indicators = {
                'screen_glare': self._detect_screen_glare(face_roi),
                'print_artifacts': self._detect_print_artifacts(face_roi),
                'uniform_illumination': self._detect_uniform_illumination(face_roi),
                'missing_micro_textures': self._detect_missing_micro_textures(face_roi),
                'color_histogram_anomaly': self._detect_color_anomaly(face_roi)
            }
            
            # Calculate spoof probability
            spoof_score = sum(indicators.values()) / len(indicators)
            
            return {
                'status': 'active',
                'indicators': indicators,
                'spoof_score': spoof_score,
                'likely_spoof': spoof_score > 0.5
            }
            
        except Exception as e:
            logger.error(f"Error in spoof detection: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _detect_screen_glare(self, face_roi: np.ndarray) -> float:
        """Detect screen glare patterns"""
        try:
            # Convert to HSV for better glare detection
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            # Detect very bright, saturated areas (screen glare)
            bright_mask = (hsv[:, :, 2] > 240) & (hsv[:, :, 1] < 50)
            glare_area = np.sum(bright_mask) / bright_mask.size
            
            return min(1.0, glare_area * 10)  # Amplify small glare areas
            
        except Exception:
            return 0.0
    
    def _detect_print_artifacts(self, face_roi: np.ndarray) -> float:
        """Detect printing artifacts like dots or lines"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # High-pass filter to detect fine details
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(gray, -1, kernel)
            
            # Look for regular patterns (printing dots)
            pattern_strength = np.std(filtered)
            
            return min(1.0, pattern_strength / 50)
            
        except Exception:
            return 0.0
    
    def _detect_uniform_illumination(self, face_roi: np.ndarray) -> float:
        """Detect unnaturally uniform illumination"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate local variance
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
            
            # Uniform illumination has low variance
            avg_variance = np.mean(local_variance)
            uniformity_score = max(0, 1.0 - avg_variance / 100)
            
            return uniformity_score
            
        except Exception:
            return 0.0
    
    def _detect_missing_micro_textures(self, face_roi: np.ndarray) -> float:
        """Detect missing skin micro-textures"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur and calculate difference
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            texture_detail = cv2.absdiff(gray, blurred)
            
            # Real skin should have micro-textures
            texture_score = np.mean(texture_detail)
            missing_texture_score = max(0, 1.0 - texture_score / 10)
            
            return missing_texture_score
            
        except Exception:
            return 0.0
    
    def _detect_color_anomaly(self, face_roi: np.ndarray) -> float:
        """Detect color distribution anomalies"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            
            # Analyze color distribution
            hsv_std = np.std(hsv, axis=(0, 1))
            lab_std = np.std(lab, axis=(0, 1))
            
            # Natural faces have certain color variation patterns
            color_variation = np.mean(hsv_std) + np.mean(lab_std)
            
            # Too little variation suggests artificial image
            anomaly_score = max(0, 1.0 - color_variation / 100)
            
            return anomaly_score
            
        except Exception:
            return 0.0
    
    def _check_challenge_completion(self, landmarks: Any, current_time: float) -> Dict:
        """Check if current challenge is completed"""
        if not self.config['enable_challenges']:
            return {'status': 'disabled'}
        
        # Start new challenge if none active
        if self.current_challenge is None and self.challenge_queue:
            self.current_challenge = self.challenge_queue.pop(0)
            self.challenge_start_time = current_time
            logger.info(f"Started challenge: {self.current_challenge}")
        
        if self.current_challenge is None:
            return {'status': 'all_completed', 'completed': self.challenges_completed}
        
        # Check timeout
        if current_time - self.challenge_start_time > self.config['challenge_timeout']:
            logger.warning(f"Challenge {self.current_challenge} timed out")
            self.current_challenge = None
            return {'status': 'timeout'}
        
        # Check challenge completion
        completed = False
        
        if self.current_challenge == 'blink':
            completed = self.total_blinks >= self.config['min_blinks_required']
        elif self.current_challenge in ['turn_left', 'turn_right', 'nod']:
            # Check head movement
            if len(self.head_poses) >= 5:
                poses = list(self.head_poses)[-5:]
                if self.current_challenge == 'turn_left':
                    yaw_range = max(p['yaw'] for p in poses) - min(p['yaw'] for p in poses)
                    completed = yaw_range > 20 and any(p['yaw'] < -10 for p in poses)
                elif self.current_challenge == 'turn_right':
                    yaw_range = max(p['yaw'] for p in poses) - min(p['yaw'] for p in poses)
                    completed = yaw_range > 20 and any(p['yaw'] > 10 for p in poses)
                elif self.current_challenge == 'nod':
                    pitch_range = max(p['pitch'] for p in poses) - min(p['pitch'] for p in poses)
                    completed = pitch_range > 15
        
        if completed:
            self.challenges_completed.append(self.current_challenge)
            logger.info(f"Challenge completed: {self.current_challenge}")
            self.current_challenge = None
        
        return {
            'status': 'active',
            'current_challenge': self.current_challenge,
            'completed': self.challenges_completed,
            'time_remaining': max(0, self.config['challenge_timeout'] - (current_time - (self.challenge_start_time or current_time)))
        }
    
    def _calculate_frame_score(self, analysis: Dict) -> float:
        """Calculate overall frame liveness score"""
        scores = []
        weights = []
        
        # Blink analysis
        if analysis['blink_analysis']['status'] == 'active':
            blink_score = min(1.0, analysis['blink_analysis']['total_blinks'] / self.config['min_blinks_required'])
            scores.append(blink_score)
            weights.append(0.25)
        
        # Head movement
        if analysis['head_movement']['status'] == 'active':
            movement_score = 1.0 if analysis['head_movement']['sufficient_movement'] else 0.3
            scores.append(movement_score)
            weights.append(0.2)
        
        # Texture analysis
        if analysis['texture_analysis']['status'] == 'active':
            texture_score = analysis['texture_analysis']['texture_score']
            scores.append(texture_score)
            weights.append(0.2)
        
        # 3D depth
        if analysis['depth_analysis']['status'] == 'active':
            depth_score = 1.0 if analysis['depth_analysis']['has_depth'] else 0.5
            scores.append(depth_score)
            weights.append(0.15)
        
        # Reflections
        if analysis['reflection_analysis']['status'] == 'active':
            reflection_score = 1.0 if analysis['reflection_analysis']['has_natural_reflections'] else 0.3
            scores.append(reflection_score)
            weights.append(0.1)
        
        # Spoof detection (penalty)
        if analysis['spoof_detection']['status'] == 'active':
            spoof_penalty = analysis['spoof_detection']['spoof_score'] * self.config['spoof_penalty']
            scores.append(1.0 + spoof_penalty)  # Add penalty (negative)
            weights.append(0.1)
        
        if not scores:
            return 0.0
        
        # Weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return max(0.0, min(1.0, weighted_score))
    
    def _update_tracking(self, analysis: Dict, current_time: float):
        """Update tracking variables"""
        frame_score = analysis['frame_score']
        self.frame_scores.append(frame_score)
        
        # Update spoof indicators
        if analysis['spoof_detection']['status'] == 'active':
            self.spoof_indicators.append(analysis['spoof_detection']['likely_spoof'])
    
    def _should_complete_detection(self, current_time: float) -> bool:
        """Check if detection should be completed"""
        session_duration = current_time - self.session_start_time
        
        # Minimum requirements
        min_blinks_met = self.total_blinks >= self.config['min_blinks_required']
        min_duration = session_duration > 5.0  # At least 5 seconds
        
        # Quality check
        sufficient_frames = len(self.frame_scores) >= 10
        good_quality = sufficient_frames and np.mean(list(self.frame_scores)[-10:]) > 0.5
        
        # Challenge completion (if enabled)
        challenges_done = not self.config['enable_challenges'] or len(self.challenges_completed) >= 2
        
        # Timeout
        max_duration_reached = session_duration > 30.0  # Maximum 30 seconds
        
        return (min_blinks_met and min_duration and good_quality and challenges_done) or max_duration_reached
    
    def _calculate_final_result(self) -> LivenessResult:
        """Calculate final liveness result"""
        try:
            # Calculate overall confidence
            if not self.frame_scores:
                confidence = 0.0
            else:
                confidence = np.mean(list(self.frame_scores))
            
            # Check requirements
            blink_requirement = self.total_blinks >= self.config['min_blinks_required']
            quality_requirement = confidence > 0.4
            spoof_check = not self.spoof_indicators or np.mean(list(self.spoof_indicators)) < 0.7
            
            # Final decision
            is_live = (confidence > self.config['liveness_threshold'] and 
                      blink_requirement and quality_requirement and spoof_check)
            
            # Score breakdown
            score_breakdown = {
                'blink_score': min(1.0, self.total_blinks / self.config['min_blinks_required']),
                'quality_score': confidence,
                'spoof_score': 1.0 - (np.mean(list(self.spoof_indicators)) if self.spoof_indicators else 0.0),
                'challenge_score': len(self.challenges_completed) / max(1, len(self.challenge_queue) + len(self.challenges_completed))
            }
            
            # Challenges
            challenges_passed = self.challenges_completed.copy()
            challenges_failed = [c for c in ['blink', 'turn_left', 'turn_right', 'nod'] 
                               if c not in challenges_passed and c not in (self.challenge_queue or [])]
            
            # Frame analysis
            frame_analysis = {
                'total_frames': self.frames_processed,
                'session_duration': time.time() - self.session_start_time,
                'total_blinks': self.total_blinks,
                'avg_frame_score': confidence
            }
            
            result = LivenessResult(
                is_live=is_live,
                confidence=confidence,
                score_breakdown=score_breakdown,
                challenges_passed=challenges_passed,
                challenges_failed=challenges_failed,
                frame_analysis=frame_analysis
            )
            
            logger.info(f"Final liveness result: {is_live} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating final result: {e}")
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                score_breakdown={},
                challenges_passed=[],
                challenges_failed=[],
                frame_analysis={'error': str(e)}
            )
    
    def _get_face_bbox(self, detection: Any, frame_shape: Tuple) -> Tuple[int, int, int, int]:
        """Get face bounding box from detection"""
        bbox = detection.location_data.relative_bounding_box
        h, w = frame_shape[:2]
        
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        
        return (x1, y1, x2, y2)
    
    def _analyze_frame_quality(self, frame: np.ndarray, face_bbox: Tuple) -> Dict:
        """Analyze frame quality"""
        try:
            x1, y1, x2, y2 = face_bbox
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Size check
            size_ok = (self.config['min_face_size'] <= min(face_width, face_height) <= 
                      self.config['max_face_size'])
            
            # Brightness check
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray[y1:y2, x1:x2])
            brightness_ok = self.config['brightness_range'][0] <= brightness <= self.config['brightness_range'][1]
            
            # Sharpness check
            laplacian = cv2.Laplacian(gray[y1:y2, x1:x2], cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_ok = sharpness > 100
            
            overall_quality = (
                (1.0 if size_ok else 0.3) * 0.4 +
                (1.0 if brightness_ok else 0.3) * 0.3 +
                (min(1.0, sharpness / 500) if sharpness_ok else 0.3) * 0.3
            )
            
            return {
                'size_ok': size_ok,
                'brightness_ok': brightness_ok,
                'sharpness_ok': sharpness_ok,
                'brightness': brightness,
                'sharpness': sharpness,
                'face_size': (face_width, face_height),
                'overall_quality': overall_quality
            }
            
        except Exception as e:
            logger.error(f"Error analyzing frame quality: {e}")
            return {'overall_quality': 0.0}
    
    def _draw_annotations(self, frame: np.ndarray, face_bbox: Tuple, 
                         landmarks: Any, analysis: Dict) -> np.ndarray:
        """Draw annotations on frame"""
        annotated_frame = frame.copy()
        
        try:
            x1, y1, x2, y2 = face_bbox
            
            # Draw face bounding box
            color = (0, 255, 0) if analysis['frame_score'] > 0.7 else (0, 255, 255) if analysis['frame_score'] > 0.5 else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw eye landmarks
            h, w = frame.shape[:2]
            for idx in self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(annotated_frame, (x, y), 2, (255, 255, 0), -1)
            
            # Draw status text
            self._draw_status_text(annotated_frame, analysis)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing annotations: {e}")
            return frame
    
    def _draw_status_text(self, frame: np.ndarray, analysis: Dict):
        """Draw status text on frame"""
        try:
            y_offset = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Frame score
            score_text = f"Liveness Score: {analysis['frame_score']:.2f}"
            color = (0, 255, 0) if analysis['frame_score'] > 0.7 else (0, 255, 255) if analysis['frame_score'] > 0.5 else (0, 0, 255)
            cv2.putText(frame, score_text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
            
            # Blink count
            if analysis['blink_analysis']['status'] == 'active':
                blink_text = f"Blinks: {analysis['blink_analysis']['total_blinks']}"
                cv2.putText(frame, blink_text, (10, y_offset), font, font_scale, (255, 255, 255), thickness)
                y_offset += 25
            
            # Current challenge
            if analysis['challenge_status']['status'] == 'active':
                challenge_text = f"Challenge: {analysis['challenge_status']['current_challenge']}"
                cv2.putText(frame, challenge_text, (10, y_offset), font, font_scale, (255, 255, 0), thickness)
                y_offset += 25
            
            # Spoof warning
            if analysis['spoof_detection']['status'] == 'active' and analysis['spoof_detection']['likely_spoof']:
                cv2.putText(frame, "SPOOFING DETECTED!", (10, y_offset), font, font_scale, (0, 0, 255), thickness)
            
        except Exception as e:
            logger.error(f"Error drawing status text: {e}")
    
    def _draw_no_face_detected(self, frame: np.ndarray) -> np.ndarray:
        """Draw no face detected message"""
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame, "No face detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return annotated_frame
    
    def _draw_poor_quality(self, frame: np.ndarray) -> np.ndarray:
        """Draw poor quality message"""
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame, "Poor image quality", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return annotated_frame
    
    def _reset_session(self):
        """Reset session variables"""
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_time = 0
        self.ear_history.clear()
        self.baseline_ear = None
        self.head_poses.clear()
        self.movement_detected = False
        self.movement_start_time = None
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenges_completed.clear()
        self.challenge_queue = ['blink', 'turn_left', 'turn_right', 'nod']
        self.frame_scores.clear()
        self.spoof_indicators.clear()
        self.frames_processed = 0
        self.final_result = None
        
    def get_current_status(self) -> Dict:
        """Get current detection status"""
        return {
            'detection_active': self.detection_active,
            'session_duration': time.time() - self.session_start_time if self.detection_active else 0,
            'frames_processed': self.frames_processed,
            'total_blinks': self.total_blinks,
            'current_challenge': self.current_challenge,
            'challenges_completed': self.challenges_completed.copy(),
            'average_score': np.mean(list(self.frame_scores)) if self.frame_scores else 0.0
        }

def create_detector_config(
    strict_mode: bool = False,
    enable_challenges: bool = True,
    min_blinks: int = 2,
    liveness_threshold: float = 0.7
) -> Dict:
    """
    Create detector configuration
    
    Args:
        strict_mode: Enable strict detection parameters
        enable_challenges: Enable interactive challenges
        min_blinks: Minimum blinks required
        liveness_threshold: Minimum confidence for liveness
    
    Returns:
        Configuration dictionary
    """
    config = {
        'min_blinks_required': min_blinks,
        'enable_challenges': enable_challenges,
        'liveness_threshold': liveness_threshold
    }
    
    if strict_mode:
        config.update({
            'ear_threshold': 0.2,
            'texture_variance_threshold': 150,
            'reflection_threshold': 180,
            'spoof_penalty': -0.5
        })
    
    return config

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Liveness Detection')
    parser.add_argument('--strict', action='store_true', help='Enable strict mode')
    parser.add_argument('--no-challenges', action='store_true', help='Disable challenges')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_detector_config(
        strict_mode=args.strict,
        enable_challenges=not args.no_challenges
    )
    
    # Initialize detector
    detector = RealtimeLivenessDetector(config)
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        logger.error("Cannot open camera")
        exit()
    
    logger.info("Starting real-time liveness detection. Press 's' to start, 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and not detector.detection_active:
                detector.start_detection()
                logger.info("Detection started!")
            elif key == ord('r'):
                detector.stop_detection()
                logger.info("Detection reset!")
            
            # Process frame
            annotated_frame, analysis = detector.process_frame(frame)
            
            # Show result
            cv2.imshow('Real-time Liveness Detection', annotated_frame)
            
            # Check if detection completed
            if not detector.detection_active and detector.final_result is not None:
                result = detector.final_result
                print(f"\n=== LIVENESS DETECTION RESULT ===")
                print(f"Is Live: {result.is_live}")
                print(f"Confidence: {result.confidence:.3f}")
                print(f"Challenges Passed: {result.challenges_passed}")
                print(f"Score Breakdown: {result.score_breakdown}")
                
                # Reset for next detection
                detector.final_result = None
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")
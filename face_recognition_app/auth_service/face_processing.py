"""
Enhanced Face Authentication System with Liveness Detection and Anti-Spoofing
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import math
import time
from dataclasses import dataclass
from django.conf import settings


@dataclass
class LivenessResult:
    """Result of liveness detection"""
    is_live: bool
    confidence: float
    methods_used: List[str]
    blink_detected: bool = False
    motion_detected: bool = False
    texture_score: float = 0.0
    face_movements: List[Dict] = None
    
    def __post_init__(self):
        if self.face_movements is None:
            self.face_movements = []


@dataclass
class ObstacleResult:
    """Result of obstacle detection"""
    has_obstacle: bool
    obstacle_type: str = ""
    confidence: float = 0.0
    bbox: Optional[List[int]] = None


@dataclass
class FaceQuality:
    """Face quality metrics"""
    overall_score: float
    brightness: float
    sharpness: float  
    contrast: float
    pose_score: float
    occlusion_score: float
    
    
class LivenessDetector:
    """
    Advanced liveness detection using multiple methods
    """
    
    def __init__(self):
        self.config = getattr(settings, 'FACE_RECOGNITION_CONFIG', {})
        self.liveness_threshold = self.config.get('LIVENESS_THRESHOLD', 0.8)
        self.methods = self.config.get('LIVENESS_METHODS', ['blink_detection', 'motion_detection'])
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks for blink detection
        self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        
        # Previous frame data for motion detection
        self.previous_landmarks = None
        self.blink_counter = 0
        self.eye_aspect_ratios = []
        self.face_positions = []
        self.start_time = time.time()
    
    def detect_liveness(self, frame: np.ndarray) -> LivenessResult:
        """
        Detect liveness using multiple methods
        """
        results = {}
        methods_used = []
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(rgb_frame)
        
        if not mesh_results.multi_face_landmarks:
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                methods_used=[],
                blink_detected=False,
                motion_detected=False
            )
        
        face_landmarks = mesh_results.multi_face_landmarks[0]
        
        # Blink detection
        if 'blink_detection' in self.methods:
            blink_result = self._detect_blink(face_landmarks, frame.shape)
            results['blink'] = blink_result
            methods_used.append('blink_detection')
        
        # Motion detection
        if 'motion_detection' in self.methods:
            motion_result = self._detect_motion(face_landmarks)
            results['motion'] = motion_result  
            methods_used.append('motion_detection')
        
        # Texture analysis
        if 'texture_analysis' in self.methods:
            texture_result = self._analyze_texture(frame, face_landmarks)
            results['texture'] = texture_result
            methods_used.append('texture_analysis')
        
        # Combine results
        overall_confidence, is_live = self._combine_results(results)
        
        return LivenessResult(
            is_live=is_live,
            confidence=overall_confidence,
            methods_used=methods_used,
            blink_detected=results.get('blink', {}).get('detected', False),
            motion_detected=results.get('motion', {}).get('detected', False),
            texture_score=results.get('texture', {}).get('score', 0.0),
            face_movements=self.face_positions[-10:]  # Last 10 positions
        )
    
    def _detect_blink(self, landmarks, frame_shape) -> Dict:
        """
        Detect eye blinks using Eye Aspect Ratio (EAR)
        """
        def eye_aspect_ratio(eye_landmarks):
            # Compute distances between eye landmarks
            h, w = frame_shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for idx in eye_landmarks:
                landmark = landmarks.landmark[idx]
                points.append([int(landmark.x * w), int(landmark.y * h)])
            
            # Calculate EAR
            # Vertical distances
            A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
            B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
            
            # Horizontal distance
            C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
            
            if C == 0:
                return 0
            
            ear = (A + B) / (2.0 * C)
            return ear
        
        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(self.LEFT_EYE_LANDMARKS)
        right_ear = eye_aspect_ratio(self.RIGHT_EYE_LANDMARKS)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        self.eye_aspect_ratios.append(avg_ear)
        
        # Keep only recent EAR values
        if len(self.eye_aspect_ratios) > 30:
            self.eye_aspect_ratios.pop(0)
        
        # Detect blink (EAR drops below threshold)
        EAR_THRESHOLD = 0.25
        CONSECUTIVE_FRAMES = 3
        
        blink_detected = False
        if len(self.eye_aspect_ratios) >= CONSECUTIVE_FRAMES:
            recent_ears = self.eye_aspect_ratios[-CONSECUTIVE_FRAMES:]
            if all(ear < EAR_THRESHOLD for ear in recent_ears):
                # Check if there was a previous higher EAR (indicating eye was open)
                if len(self.eye_aspect_ratios) > CONSECUTIVE_FRAMES:
                    prev_ears = self.eye_aspect_ratios[-(CONSECUTIVE_FRAMES*2):-CONSECUTIVE_FRAMES]
                    if any(ear > EAR_THRESHOLD for ear in prev_ears):
                        self.blink_counter += 1
                        blink_detected = True
        
        return {
            'detected': blink_detected,
            'ear': avg_ear,
            'blink_count': self.blink_counter,
            'confidence': min(self.blink_counter * 0.3, 1.0)  # More blinks = higher confidence
        }
    
    def _detect_motion(self, landmarks) -> Dict:
        """
        Detect face motion and head movement
        """
        # Get key facial points for motion tracking
        h, w = 480, 640  # Assume default resolution
        
        # Nose tip (landmark 1)
        nose_tip = landmarks.landmark[1]
        current_position = [nose_tip.x * w, nose_tip.y * h, nose_tip.z]
        
        self.face_positions.append(current_position)
        
        # Keep only recent positions
        if len(self.face_positions) > 30:
            self.face_positions.pop(0)
        
        motion_detected = False
        motion_magnitude = 0.0
        
        if len(self.face_positions) >= 10:
            # Calculate motion between recent frames
            recent_positions = np.array(self.face_positions[-10:])
            
            # Calculate movement variance
            position_variance = np.var(recent_positions, axis=0)
            motion_magnitude = np.sum(position_variance)
            
            # Detect significant motion
            MOTION_THRESHOLD = 50.0  # Adjust based on testing
            if motion_magnitude > MOTION_THRESHOLD:
                motion_detected = True
        
        return {
            'detected': motion_detected,
            'magnitude': motion_magnitude,
            'confidence': min(motion_magnitude / 200.0, 1.0)  # Normalize to 0-1
        }
    
    def _analyze_texture(self, frame, landmarks) -> Dict:
        """
        Analyze face texture to detect printed photos or screens
        """
        h, w = frame.shape[:2]
        
        # Extract face region
        face_points = []
        face_contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        for idx in face_contour_indices:
            landmark = landmarks.landmark[idx]
            face_points.append([int(landmark.x * w), int(landmark.y * h)])
        
        face_points = np.array(face_points)
        
        # Create mask for face region
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_points], 255)
        
        # Extract face ROI
        face_roi = cv2.bitwise_and(frame, frame, mask=mask)
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture metrics
        # 1. Local Binary Pattern variance (texture richness)
        lbp_var = self._calculate_lbp_variance(face_gray, mask)
        
        # 2. High frequency content (edge density)
        edges = cv2.Canny(face_gray, 50, 150)
        edge_density = np.sum(edges > 0) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
        
        # 3. Color distribution analysis
        color_variance = self._calculate_color_variance(face_roi, mask)
        
        # Combine metrics
        texture_score = (lbp_var * 0.4 + edge_density * 0.3 + color_variance * 0.3)
        
        return {
            'score': texture_score,
            'lbp_variance': lbp_var,
            'edge_density': edge_density,
            'color_variance': color_variance,
            'confidence': texture_score
        }
    
    def _calculate_lbp_variance(self, gray_image, mask):
        """Calculate Local Binary Pattern variance"""
        # Simple LBP implementation
        h, w = gray_image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if mask[i, j] == 0:
                    continue
                    
                center = gray_image[i, j]
                code = 0
                
                # 8-neighbor LBP
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i-1, j-1] = code
        
        # Calculate variance of LBP values
        valid_lbp = lbp[mask[1:-1, 1:-1] > 0]
        if len(valid_lbp) > 0:
            return np.var(valid_lbp) / 255.0  # Normalize
        return 0.0
    
    def _calculate_color_variance(self, color_image, mask):
        """Calculate color variance in face region"""
        face_pixels = color_image[mask > 0]
        if len(face_pixels) == 0:
            return 0.0
        
        # Calculate variance across color channels
        variances = []
        for channel in range(3):
            channel_pixels = face_pixels[:, channel]
            if len(channel_pixels) > 1:
                variances.append(np.var(channel_pixels))
        
        if variances:
            return np.mean(variances) / (255.0 ** 2)  # Normalize
        return 0.0
    
    def _combine_results(self, results) -> Tuple[float, bool]:
        """
        Combine results from different liveness detection methods
        """
        scores = []
        weights = {
            'blink': 0.4,
            'motion': 0.3,
            'texture': 0.3
        }
        
        total_weight = 0
        weighted_score = 0
        
        for method, result in results.items():
            if method in weights and 'confidence' in result:
                weight = weights[method]
                score = result['confidence']
                
                weighted_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_confidence = weighted_score / total_weight
        else:
            overall_confidence = 0.0
        
        is_live = overall_confidence >= self.liveness_threshold
        
        return overall_confidence, is_live


class ObstacleDetector:
    """
    Detect obstacles like hands, masks, or objects covering the face
    """
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
    
    def detect_obstacles(self, frame: np.ndarray) -> ObstacleResult:
        """
        Detect obstacles in front of the face
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        if not face_results.multi_face_landmarks:
            return ObstacleResult(has_obstacle=False)
        
        # Check for hands near face
        if hand_results.multi_hand_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            hand_obstacle = self._check_hand_obstacle(
                hand_results.multi_hand_landmarks, 
                face_landmarks, 
                frame.shape
            )
            if hand_obstacle['detected']:
                return ObstacleResult(
                    has_obstacle=True,
                    obstacle_type='hand',
                    confidence=hand_obstacle['confidence'],
                    bbox=hand_obstacle.get('bbox')
                )
        
        # Check for mask or occlusion
        mask_obstacle = self._check_face_occlusion(face_results.multi_face_landmarks[0], frame)
        if mask_obstacle['detected']:
            return ObstacleResult(
                has_obstacle=True,
                obstacle_type='occlusion',
                confidence=mask_obstacle['confidence']
            )
        
        return ObstacleResult(has_obstacle=False)
    
    def _check_hand_obstacle(self, hand_landmarks_list, face_landmarks, frame_shape) -> Dict:
        """
        Check if hands are obstructing the face
        """
        h, w = frame_shape[:2]
        
        # Get face bounding box
        face_points = []
        for landmark in face_landmarks.landmark:
            face_points.append([landmark.x * w, landmark.y * h])
        
        face_points = np.array(face_points)
        face_bbox = [
            np.min(face_points[:, 0]), np.min(face_points[:, 1]),
            np.max(face_points[:, 0]), np.max(face_points[:, 1])
        ]
        
        for hand_landmarks in hand_landmarks_list:
            # Get hand bounding box
            hand_points = []
            for landmark in hand_landmarks.landmark:
                hand_points.append([landmark.x * w, landmark.y * h])
            
            hand_points = np.array(hand_points)
            hand_bbox = [
                np.min(hand_points[:, 0]), np.min(hand_points[:, 1]),
                np.max(hand_points[:, 0]), np.max(hand_points[:, 1])
            ]
            
            # Calculate overlap
            overlap = self._calculate_bbox_overlap(face_bbox, hand_bbox)
            
            if overlap > 0.2:  # 20% overlap threshold
                return {
                    'detected': True,
                    'confidence': min(overlap * 2, 1.0),
                    'bbox': hand_bbox
                }
        
        return {'detected': False}
    
    def _check_face_occlusion(self, face_landmarks, frame) -> Dict:
        """
        Check for face occlusion (masks, objects, etc.)
        """
        h, w = frame.shape[:2]
        
        # Key facial regions to check
        mouth_landmarks = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        nose_landmarks = [19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220]
        
        # Check mouth region visibility
        mouth_visible = self._check_region_visibility(mouth_landmarks, face_landmarks, frame)
        nose_visible = self._check_region_visibility(nose_landmarks, face_landmarks, frame)
        
        # If both mouth and nose are significantly occluded, likely wearing mask
        if not mouth_visible and not nose_visible:
            return {
                'detected': True,
                'confidence': 0.9
            }
        
        return {'detected': False}
    
    def _check_region_visibility(self, landmark_indices, face_landmarks, frame) -> bool:
        """
        Check if a facial region is visible (not occluded)
        """
        h, w = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get region points
        region_points = []
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            region_points.append([int(landmark.x * w), int(landmark.y * h)])
        
        region_points = np.array(region_points)
        
        # Create mask for region
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [region_points], 255)
        
        # Extract region
        region = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        
        # Calculate variance (visible skin should have reasonable variance)
        region_pixels = region[mask > 0]
        if len(region_pixels) == 0:
            return False
        
        variance = np.var(region_pixels)
        
        # Low variance suggests uniform color (possible occlusion)
        VARIANCE_THRESHOLD = 100  # Adjust based on testing
        return variance > VARIANCE_THRESHOLD
    
    def _calculate_bbox_overlap(self, bbox1, bbox2) -> float:
        """
        Calculate intersection over union of two bounding boxes
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union


class FaceQualityAssessment:
    """
    Assess face image quality for better recognition
    """
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
    
    def assess_quality(self, frame: np.ndarray) -> FaceQuality:
        """
        Assess overall face image quality
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return FaceQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Assess different quality metrics
        brightness = self._assess_brightness(frame)
        sharpness = self._assess_sharpness(frame, face_landmarks)
        contrast = self._assess_contrast(frame, face_landmarks)
        pose_score = self._assess_pose(face_landmarks)
        occlusion_score = self._assess_occlusion(face_landmarks, frame)
        
        # Calculate overall score
        overall_score = (
            brightness * 0.2 +
            sharpness * 0.3 +
            contrast * 0.2 +
            pose_score * 0.2 +
            occlusion_score * 0.1
        )
        
        return FaceQuality(
            overall_score=overall_score,
            brightness=brightness,
            sharpness=sharpness,
            contrast=contrast,
            pose_score=pose_score,
            occlusion_score=occlusion_score
        )
    
    def _assess_brightness(self, frame: np.ndarray) -> float:
        """Assess image brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Optimal brightness range: 80-180
        if 80 <= mean_brightness <= 180:
            return 1.0
        elif mean_brightness < 80:
            return max(0.0, mean_brightness / 80.0)
        else:
            return max(0.0, 1.0 - (mean_brightness - 180) / 75.0)
    
    def _assess_sharpness(self, frame: np.ndarray, face_landmarks) -> float:
        """Assess image sharpness using Laplacian variance"""
        h, w = frame.shape[:2]
        
        # Extract face region
        face_points = []
        for landmark in face_landmarks.landmark:
            face_points.append([int(landmark.x * w), int(landmark.y * h)])
        
        face_points = np.array(face_points)
        x, y, fw, fh = cv2.boundingRect(face_points)
        
        # Ensure valid ROI
        x = max(0, x)
        y = max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)
        
        if fw <= 0 or fh <= 0:
            return 0.0
        
        face_roi = frame[y:y+fh, x:x+fw]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize (typical good values are above 100)
        return min(1.0, variance / 500.0)
    
    def _assess_contrast(self, frame: np.ndarray, face_landmarks) -> float:
        """Assess image contrast"""
        h, w = frame.shape[:2]
        
        # Extract face region
        face_points = []
        for landmark in face_landmarks.landmark:
            face_points.append([int(landmark.x * w), int(landmark.y * h)])
        
        face_points = np.array(face_points)
        x, y, fw, fh = cv2.boundingRect(face_points)
        
        x = max(0, x)
        y = max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)
        
        if fw <= 0 or fh <= 0:
            return 0.0
        
        face_roi = frame[y:y+fh, x:x+fw]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast using standard deviation
        std_dev = np.std(gray_roi)
        
        # Normalize (good contrast typically has std > 30)
        return min(1.0, std_dev / 60.0)
    
    def _assess_pose(self, face_landmarks) -> float:
        """Assess face pose (frontal is better)"""
        # Get key points for pose estimation
        # Nose tip
        nose_tip = face_landmarks.landmark[1]
        # Left eye corner
        left_eye = face_landmarks.landmark[33]
        # Right eye corner  
        right_eye = face_landmarks.landmark[263]
        # Left mouth corner
        left_mouth = face_landmarks.landmark[61]
        # Right mouth corner
        right_mouth = face_landmarks.landmark[291]
        
        # Calculate symmetry
        eye_distance = abs(left_eye.x - right_eye.x)
        mouth_distance = abs(left_mouth.x - right_mouth.x)
        
        # Check if face is roughly frontal
        face_center_x = (left_eye.x + right_eye.x) / 2
        nose_deviation = abs(nose_tip.x - face_center_x)
        
        # Calculate pose score based on symmetry and alignment
        symmetry_score = 1.0 - min(1.0, abs(eye_distance - mouth_distance) * 10)
        alignment_score = 1.0 - min(1.0, nose_deviation * 20)
        
        return (symmetry_score + alignment_score) / 2
    
    def _assess_occlusion(self, face_landmarks, frame: np.ndarray) -> float:
        """Assess face occlusion"""
        # Simple occlusion assessment based on landmark visibility
        # In a real implementation, this would be more sophisticated
        
        # Check if key landmarks are within frame bounds
        h, w = frame.shape[:2]
        visible_landmarks = 0
        total_landmarks = len(face_landmarks.landmark)
        
        for landmark in face_landmarks.landmark:
            if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                visible_landmarks += 1
        
        visibility_ratio = visible_landmarks / total_landmarks
        return visibility_ratio


class FaceProcessor:
    """Main face processing class for enrollment and authentication"""
    
    def __init__(self):
        """Initialize face processor with all components"""
        self.liveness_detector = LivenessDetector()
        self.obstacle_detector = ObstacleDetector()
        self.face_quality = FaceQualityAssessment()
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def process_image(self, image_file) -> Dict:
        """
        Process uploaded image for face detection, quality assessment, and embedding extraction
        
        Args:
            image_file: Uploaded image file
            
        Returns:
            Dict containing processing results
        """
        try:
            # Read image
            image_bytes = image_file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'face_detected': False, 'error': 'Invalid image format'}
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face landmarks
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {'face_detected': False, 'error': 'No face detected'}
            
            if len(results.multi_face_landmarks) > 1:
                return {'face_detected': False, 'error': 'Multiple faces detected'}
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Assess face quality
            quality_result = self.face_quality.assess_quality(face_landmarks, rgb_image)
            
            # Detect obstacles
            obstacle_result = self.obstacle_detector.detect_obstacles(rgb_image)
            
            if obstacle_result.has_obstacle:
                return {
                    'face_detected': True,
                    'error': f'Obstacle detected: {obstacle_result.obstacle_type}',
                    'quality_score': quality_result.overall_score
                }
            
            # Extract face embedding (simplified - in real implementation use FaceNet or similar)
            embedding = self._extract_face_embedding(face_landmarks, rgb_image)
            
            return {
                'face_detected': True,
                'embedding': embedding,
                'quality_score': quality_result.overall_score,
                'landmarks': self._serialize_landmarks(face_landmarks),
                'liveness_score': 0.8,  # Placeholder - would use liveness detector for video
                'brightness': quality_result.brightness,
                'sharpness': quality_result.sharpness,
                'contrast': quality_result.contrast,
                'pose_score': quality_result.pose_score
            }
            
        except Exception as e:
            return {'face_detected': False, 'error': f'Processing error: {str(e)}'}
    
    def find_matching_user(self, client, query_embedding) -> Dict:
        """
        Find matching user from client's enrolled faces
        
        Args:
            client: Client instance
            query_embedding: Face embedding to match
            
        Returns:
            Dict containing match results
        """
        try:
            from .models import FaceEnrollment
            from clients.models import ClientUser
            
            # Get all completed enrollments for this client
            enrollments = FaceEnrollment.objects.filter(
                client=client,
                status='completed',
                client_user__is_active=True
            )
            
            if not enrollments.exists():
                return {'matched_user': None, 'confidence_score': 0.0}
            
            best_match = None
            best_score = 0.0
            comparison_scores = {}
            
            # Compare with each enrollment
            for enrollment in enrollments:
                if enrollment.embedding_vector:
                    # Calculate similarity (cosine similarity)
                    similarity = self._calculate_similarity(
                        query_embedding, 
                        enrollment.embedding_vector
                    )
                    
                    comparison_scores[enrollment.client_user.username] = similarity
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = enrollment.client_user
            
            # Apply threshold for acceptance
            confidence_threshold = 0.75  # Configurable
            
            if best_score >= confidence_threshold:
                return {
                    'matched_user': best_match,
                    'confidence_score': best_score,
                    'comparison_scores': comparison_scores
                }
            else:
                return {
                    'matched_user': None,
                    'confidence_score': best_score,
                    'comparison_scores': comparison_scores
                }
                
        except Exception as e:
            return {
                'matched_user': None, 
                'confidence_score': 0.0,
                'error': f'Matching error: {str(e)}'
            }
    
    def _extract_face_embedding(self, face_landmarks, image: np.ndarray) -> List[float]:
        """
        Extract face embedding from landmarks and image
        This is a simplified version - in production use FaceNet, ArcFace, etc.
        """
        # Get key facial points
        h, w = image.shape[:2]
        key_points = []
        
        # Extract coordinates for key landmarks (simplified)
        landmark_indices = [10, 152, 234, 454, 70, 63, 105, 66, 55, 285]  # Key face points
        
        for idx in landmark_indices:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                key_points.extend([landmark.x * w, landmark.y * h])
        
        # Normalize the coordinates (simplified embedding)
        embedding = []
        for i in range(0, len(key_points), 2):
            if i + 1 < len(key_points):
                # Normalize to 0-1 range
                x_norm = key_points[i] / w
                y_norm = key_points[i + 1] / h
                embedding.extend([x_norm, y_norm])
        
        # Pad or truncate to fixed size (512 dimensions)
        target_size = 512
        while len(embedding) < target_size:
            embedding.append(0.0)
        
        return embedding[:target_size]
    
    def _serialize_landmarks(self, face_landmarks) -> List[Dict]:
        """Serialize face landmarks to JSON-compatible format"""
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append({
                'x': float(landmark.x),
                'y': float(landmark.y),
                'z': float(landmark.z)
            })
        return landmarks
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert to 0-1 range
            return (similarity + 1) / 2
            
        except Exception as e:
            return 0.0
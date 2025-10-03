"""
Core Face Recognition Engine
Adapted from the existing face_auth_system.py
"""
import logging
import time
import json
import numpy as np
import cv2
import mediapipe as mp
from insightface.app import FaceAnalysis
import faiss
import chromadb
from typing import Dict, List, Tuple, Optional, Any
from django.conf import settings
from django.core.cache import cache
import hashlib
import uuid


logger = logging.getLogger('face_recognition')


def json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON serializable format"""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__') and not isinstance(obj, (np.generic)):
        try:
            return {key: json_serializable(value) for key, value in obj.__dict__.items()}
        except (AttributeError, TypeError):
            return str(obj)
    else:
        return str(obj)


class LivenessDetector:
    """Enhanced liveness detection for spoofing prevention"""
    
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
        
        # Eye landmarks for blink detection
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Enhanced eye landmarks for optimal EAR calculation
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]  
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
        
        # Blink detection parameters
        self.EAR_THRESHOLD = 0.8
        self.ADAPTIVE_FACTOR = 0.75
        self.CONSECUTIVE_FRAMES = 2
        self.MIN_BLINK_DURATION = 1
        self.MAX_BLINK_DURATION = 10
        
        # Tracking variables
        self.reset()
        
    def reset(self):
        """Reset detector for new session"""
        self.blink_counter = 0
        self.frame_counter = 0
        self.total_blinks = 0
        self.valid_blinks = 0
        self.ear_history = []
        self.baseline_ear = None
        self.blink_start_frame = None
        self.last_blink_time = 0
        self.eye_visibility_score = 0.0
        self.blink_quality_scores = []
        
    def calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio"""
        try:
            eye_points = []
            confidence_scores = []
            
            for idx in eye_indices:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    eye_points.append([point.x, point.y])
                    confidence_scores.append(1.0)  # MediaPipe doesn't provide visibility scores
            
            if len(eye_points) < 6:
                logger.debug(f"Insufficient eye points: {len(eye_points)}")
                return None, 0.0
                
            eye_points = np.array(eye_points)
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 1.0
            
            # Validate eye shape
            if not self._validate_eye_shape(eye_points):
                logger.debug("Eye shape validation failed")
                return None, 0.0
            
            # Enhanced 6-point EAR calculation
            A = np.linalg.norm(eye_points[1] - eye_points[5])  # top_outer to bottom_outer
            B = np.linalg.norm(eye_points[2] - eye_points[4])  # top_inner to bottom_inner
            C = np.linalg.norm(eye_points[0] - eye_points[3])  # outer_corner to inner_corner
            
            if C < 0.001:
                logger.debug("Eye width too small")
                return None, 0.0
            
            ear = (A + B) / (2.0 * C)
            quality = min(1.0, C * 15)  # Quality based on eye width
            
            logger.debug(f"EAR: A={A:.4f}, B={B:.4f}, C={C:.4f}, EAR={ear:.4f}, Quality={quality:.3f}")
            return ear, quality
            
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return None, 0.0
    
    def _validate_eye_shape(self, eye_points):
        """Validate if detected points form a reasonable eye shape"""
        try:
            outer_corner, top_outer, top_inner, inner_corner, bottom_inner, bottom_outer = eye_points
            
            # Eye should be roughly horizontal
            horizontal_dist = abs(outer_corner[0] - inner_corner[0])
            vertical_dist = abs(outer_corner[1] - inner_corner[1])
            
            if horizontal_dist < vertical_dist * 2:
                logger.debug("Eye orientation validation failed")
                return False
            
            # Top points should be above bottom points
            if (top_outer[1] > bottom_outer[1]) or (top_inner[1] > bottom_inner[1]):
                return False
                
            return True
            
        except Exception:
            return False
        
    def detect_blink(self, frame):
        """Enhanced blink detection"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            self.frame_counter += 1
            current_time = time.time()
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate EAR for both eyes
                    left_ear, left_quality = self.calculate_ear(face_landmarks, self.LEFT_EYE_POINTS)
                    right_ear, right_quality = self.calculate_ear(face_landmarks, self.RIGHT_EYE_POINTS)
                    
                    if left_ear is not None and right_ear is not None:
                        # Average EAR
                        avg_ear = (left_ear + right_ear) / 2.0
                        avg_quality = (left_quality + right_quality) / 2.0
                        
                        self.ear_history.append(avg_ear)
                        if len(self.ear_history) > 30:  # Keep last 30 frames
                            self.ear_history.pop(0)
                        
                        # Adaptive threshold
                        if len(self.ear_history) >= 10:
                            self.baseline_ear = np.mean(self.ear_history[-10:])
                            adaptive_threshold = self.baseline_ear * self.ADAPTIVE_FACTOR
                        else:
                            adaptive_threshold = self.EAR_THRESHOLD
                        
                        # Blink detection logic
                        if avg_ear < adaptive_threshold:
                            if self.blink_start_frame is None:
                                self.blink_start_frame = self.frame_counter
                            self.blink_counter += 1
                        else:
                            if self.blink_counter >= self.CONSECUTIVE_FRAMES:
                                # Valid blink detected
                                blink_duration = self.blink_counter
                                if self.MIN_BLINK_DURATION <= blink_duration <= self.MAX_BLINK_DURATION:
                                    self.total_blinks += 1
                                    self.valid_blinks += 1
                                    self.blink_quality_scores.append(avg_quality)
                                    self.last_blink_time = current_time
                                    
                                    logger.info(f"Blink detected! Total: {self.total_blinks}, Duration: {blink_duration}")
                            
                            self.blink_counter = 0
                            self.blink_start_frame = None
                        
                        return {
                            'blinks_detected': self.total_blinks,
                            'ear': avg_ear,
                            'quality': avg_quality,
                            'threshold': adaptive_threshold,
                            'frame_count': self.frame_counter
                        }
            else:
                logger.debug("No face landmarks detected")
            
            return {
                'blinks_detected': self.total_blinks,
                'ear': 0.0,
                'quality': 0.0,
                'threshold': self.EAR_THRESHOLD,
                'frame_count': self.frame_counter
            }
            
        except Exception as e:
            logger.error(f"Error in detect_blink: {e}")
            return {
                'blinks_detected': self.total_blinks,
                'ear': 0.0,
                'quality': 0.0,
                'threshold': self.EAR_THRESHOLD,
                'frame_count': self.frame_counter,
                'error': str(e)
            }
    
    def is_live(self, blink_count_threshold=2):
        """Determine if subject is live based on blinks"""
        is_alive = self.total_blinks >= blink_count_threshold
        logger.debug(f"Liveness check: {self.total_blinks} >= {blink_count_threshold} = {is_alive}")
        return is_alive
    
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
    """Enhanced obstacle detection system"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark regions
        self.EYE_REGION_LEFT = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.EYE_REGION_RIGHT = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.MOUTH_REGION = [0, 17, 18, 200, 199, 175, 13, 269, 270, 267, 269, 270, 267, 271, 272]
        self.NOSE_REGION = [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 307, 375, 321, 308, 324, 318]
        
        # Detection thresholds
        self.OCCLUSION_THRESHOLD = 0.3
        self.BRIGHTNESS_THRESHOLD = 0.15
        self.TEXTURE_THRESHOLD = 25
        
    def detect_obstacles(self, frame, face_bbox):
        """Enhanced obstacle detection"""
        obstacles = []
        confidence_scores = {}
        
        try:
            # Extract face region with padding
            x1, y1, x2, y2 = face_bbox
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return obstacles, confidence_scores
                
            # Detect using face mesh landmarks
            rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_roi)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Glasses detection
                    glasses_conf = self._detect_glasses_advanced(face_roi, face_landmarks)
                    if glasses_conf > 0.3:
                        obstacles.append('glasses')
                        confidence_scores['glasses'] = glasses_conf
                    
                    # Mask detection
                    mask_conf = self._detect_mask_advanced(face_roi, face_landmarks)
                    if mask_conf > 0.4:
                        obstacles.append('mask')
                        confidence_scores['mask'] = mask_conf
                    
                    # Hat detection
                    hat_conf = self._detect_hat_advanced(face_roi, face_landmarks)
                    if hat_conf > 0.3:
                        obstacles.append('hat')
                        confidence_scores['hat'] = hat_conf
                    
                    # Hand covering detection
                    hand_conf = self._detect_hand_covering(face_roi, face_landmarks)
                    if hand_conf > 0.3:
                        obstacles.append('hand_covering')
                        confidence_scores['hand_covering'] = hand_conf
            else:
                # Fallback to traditional detection
                traditional_obstacles = self._detect_obstacles_traditional(face_roi)
                obstacles.extend(traditional_obstacles)
            
            return obstacles, confidence_scores
            
        except Exception as e:
            logger.error(f"Error in obstacle detection: {e}")
            return [], {}
    
    def _detect_glasses_advanced(self, face_roi, landmarks):
        """Advanced glasses detection"""
        try:
            h, w = face_roi.shape[:2]
            confidence = 0.0
            
            # Reflection detection
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            
            bright_spots = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)[1]
            bright_area = np.sum(bright_spots > 0) / (h * w)
            
            if bright_area > 0.15:
                confidence += 0.3
            
            # Edge detection around eyes
            eye_regions = []
            for idx in self.EYE_REGION_LEFT + self.EYE_REGION_RIGHT:
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    eye_regions.append([int(point.x * w), int(point.y * h)])
            
            if eye_regions:
                # Create eye mask and detect edges
                eye_mask = np.zeros((h, w), dtype=np.uint8)
                if len(eye_regions) > 2:
                    hull = cv2.convexHull(np.array(eye_regions))
                    cv2.fillPoly(eye_mask, [hull], 255)
                    
                    edges = cv2.Canny(gray, 50, 150)
                    eye_edges = cv2.bitwise_and(edges, eye_mask)
                    edge_ratio = np.sum(eye_edges > 0) / np.sum(eye_mask > 0) if np.sum(eye_mask > 0) > 0 else 0
                    
                    if edge_ratio > 0.1:
                        confidence += 0.4
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error in glasses detection: {e}")
            return 0.0
    
    def _detect_mask_advanced(self, face_roi, landmarks):
        """Advanced mask detection"""
        try:
            h, w = face_roi.shape[:2]
            confidence = 0.0
            
            # Mouth region visibility check
            mouth_visible_points = 0
            mouth_total_points = len(self.MOUTH_REGION)
            
            for idx in self.MOUTH_REGION:
                if idx < len(landmarks.landmark):
                    mouth_visible_points += 1
            
            mouth_visibility_ratio = mouth_visible_points / mouth_total_points
            if mouth_visibility_ratio < 0.5:
                confidence += 0.5
            
            # Texture analysis in lower face
            lower_face = face_roi[int(h * 0.6):, :]
            if lower_face.size > 0:
                gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
                texture_var = cv2.Laplacian(gray_lower, cv2.CV_64F).var()
                
                if texture_var < self.TEXTURE_THRESHOLD:
                    confidence += 0.3
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error in mask detection: {e}")
            return 0.0
    
    def _detect_hat_advanced(self, face_roi, landmarks):
        """Advanced hat detection"""
        try:
            h, w = face_roi.shape[:2]
            confidence = 0.0
            
            # Shadow detection in upper region
            upper_region = face_roi[:int(h * 0.3), :]
            if upper_region.size > 0:
                gray_upper = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
                dark_pixels = np.sum(gray_upper < 50)
                dark_ratio = dark_pixels / (upper_region.shape[0] * upper_region.shape[1])
                
                if dark_ratio > 0.3:
                    confidence += 0.4
            
            # Forehead visibility check
            forehead_landmarks = [9, 10, 151, 337, 299, 333, 298, 301]
            visible_forehead = 0
            
            for idx in forehead_landmarks:
                if idx < len(landmarks.landmark):
                    visible_forehead += 1
            
            if visible_forehead < len(forehead_landmarks) * 0.5:
                confidence += 0.3
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error in hat detection: {e}")
            return 0.0
    
    def _detect_hand_covering(self, face_roi, landmarks):
        """Detect hand covering parts of face"""
        try:
            # Simplified hand detection based on occlusion patterns
            h, w = face_roi.shape[:2]
            
            # Check for unusual occlusion patterns
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Look for skin-colored regions that don't match face structure
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            # Skin color range (approximate)
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / (h * w)
            
            # If there's too much skin area, might indicate hand covering
            if skin_ratio > 0.7:
                return 0.4
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error in hand covering detection: {e}")
            return 0.0
    
    def _detect_obstacles_traditional(self, face_roi):
        """Fallback traditional detection methods"""
        obstacles = []
        
        try:
            h, w = face_roi.shape[:2]
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Conservative glasses detection
            bright_pixels = np.sum(gray > 240)
            bright_ratio = bright_pixels / (h * w)
            
            edges = cv2.Canny(gray, 80, 160)
            edge_pixels = np.sum(edges > 0)
            edge_ratio = edge_pixels / (h * w)
            
            if bright_ratio > 0.08 and edge_ratio > 0.15:
                obstacles.append('glasses')
            
            # Simple darkness check for hat
            upper_region = gray[:int(h * 0.3), :]
            if upper_region.size > 0:
                if np.mean(upper_region) < 60:
                    obstacles.append('hat')
            
            return obstacles
            
        except Exception as e:
            logger.error(f"Error in traditional obstacle detection: {e}")
            return []


class ChromaEmbeddingStore:
    """ChromaDB integration for face embeddings"""
    
    def __init__(self):
        try:
            chroma_config = settings.CHROMA_DB_CONFIG
            self.client = chromadb.HttpClient(
                host=chroma_config['host'],
                port=chroma_config['port']
            )
            self.collection_name = chroma_config['collection_name']
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Face recognition embeddings"}
                )
                
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def add_embedding(self, user_id: str, embedding: np.ndarray, metadata: dict = None):
        """Add face embedding to ChromaDB"""
        try:
            if self.collection is None:
                return False
            
            embedding_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
            
            self.collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[metadata or {}],
                ids=[embedding_id]
            )
            
            logger.debug(f"Added embedding for user {user_id}")
            return embedding_id
            
        except Exception as e:
            logger.error(f"Failed to add embedding: {e}")
            return None
    
    def search_similar(self, embedding: np.ndarray, top_k: int = 5, threshold: float = 0.4):
        """Search for similar embeddings"""
        try:
            if self.collection is None:
                return []
            
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=top_k
            )
            
            matches = []
            if results['ids'] and results['distances']:
                for i, (embedding_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                    # Convert distance to similarity (ChromaDB uses L2 distance)
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity >= threshold:
                        matches.append({
                            'embedding_id': embedding_id,
                            'similarity': similarity,
                            'distance': distance,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                        })
            
            return matches
            
        except Exception as e:
            logger.error(f"Failed to search embeddings: {e}")
            return []
    
    def delete_user_embeddings(self, user_id: str):
        """Delete all embeddings for a user"""
        try:
            if self.collection is None:
                return False
            
            # Query all embeddings for the user
            results = self.collection.get()
            user_embedding_ids = []
            
            for embedding_id in results['ids']:
                if embedding_id.startswith(f"{user_id}_"):
                    user_embedding_ids.append(embedding_id)
            
            if user_embedding_ids:
                self.collection.delete(ids=user_embedding_ids)
                logger.info(f"Deleted {len(user_embedding_ids)} embeddings for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings for user {user_id}: {e}")
            return False


class FaceRecognitionEngine:
    """Main face recognition engine"""
    
    def __init__(self):
        self.config = settings.FACE_RECOGNITION_CONFIG
        
        # Initialize InsightFace
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=self.config['DET_SIZE'])
        
        # Initialize components
        self.liveness_detector = LivenessDetector()
        self.obstacle_detector = ObstacleDetector()
        self.embedding_store = ChromaEmbeddingStore()
        
        # FAISS index for fast similarity search (fallback)
        self.dimension = self.config['EMBEDDING_DIMENSION']
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.user_embeddings = {}
        self.user_names = []
        
        logger.info("Face recognition engine initialized")
    
    def extract_embedding(self, frame):
        """Extract face embedding from frame"""
        try:
            faces = self.app.get(frame)
            
            if len(faces) == 0:
                return None, "No face detected"
            
            if len(faces) > 1:
                return None, "Multiple faces detected"
            
            # Get the face with highest confidence
            face = max(faces, key=lambda x: x.det_score)
            
            # Check minimum face size
            bbox = face.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            
            min_size = self.config['MIN_FACE_SIZE']
            max_size = self.config['MAX_FACE_SIZE']
            
            if face_width < min_size or face_height < min_size:
                return None, "Face too small"
            
            if face_width > max_size or face_height > max_size:
                return None, "Face too large"
            
            # Normalize embedding
            embedding = face.normed_embedding
            
            return {
                'embedding': embedding,
                'bbox': bbox,
                'confidence': float(face.det_score),
                'landmarks': face.kps if hasattr(face, 'kps') else None
            }, None
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None, f"System error: {str(e)}"
    
    def process_frame_for_enrollment(self, frame, user_id):
        """Process frame for user enrollment"""
        try:
            # Extract embedding
            result, error = self.extract_embedding(frame)
            if error:
                return None, error
            
            embedding = result['embedding']
            bbox = result['bbox']
            confidence = result['confidence']
            
            # Check obstacles
            obstacles, obstacle_confidence = self.obstacle_detector.detect_obstacles(frame, bbox)
            if obstacles:
                return None, f"Obstacles detected: {', '.join(obstacles)}"
            
            # Check liveness
            liveness_result = self.liveness_detector.detect_blink(frame)
            
            # Quality assessment
            quality_score = self._assess_image_quality(frame, bbox)
            
            if quality_score < self.config['CAPTURE_QUALITY_THRESHOLD']:
                return None, f"Image quality too low: {quality_score:.2f}"
            
            return {
                'embedding': embedding,
                'bbox': bbox,
                'confidence': confidence,
                'quality_score': quality_score,
                'liveness_data': liveness_result,
                'obstacles': obstacles,
                'obstacle_confidence': obstacle_confidence
            }, None
            
        except Exception as e:
            logger.error(f"Error processing enrollment frame: {e}")
            return None, f"Processing error: {str(e)}"
    
    def authenticate_user(self, frame, user_id=None):
        """Authenticate user with face recognition"""
        try:
            # Extract embedding
            result, error = self.extract_embedding(frame)
            if error:
                return {
                    'success': False,
                    'error': error,
                    'similarity_score': 0.0
                }
            
            embedding = result['embedding']
            bbox = result['bbox']
            confidence = result['confidence']
            
            # Check obstacles
            obstacles, obstacle_confidence = self.obstacle_detector.detect_obstacles(frame, bbox)
            if obstacles:
                return {
                    'success': False,
                    'error': f"Obstacles detected: {', '.join(obstacles)}",
                    'obstacles': obstacles,
                    'similarity_score': 0.0
                }
            
            # Check liveness
            liveness_result = self.liveness_detector.detect_blink(frame)
            if not self.liveness_detector.is_live():
                return {
                    'success': False,
                    'error': "Liveness check failed",
                    'liveness_data': liveness_result,
                    'similarity_score': 0.0
                }
            
            # Quality assessment
            quality_score = self._assess_image_quality(frame, bbox)
            if quality_score < 0.5:  # Lower threshold for authentication
                return {
                    'success': False,
                    'error': f"Image quality too low: {quality_score:.2f}",
                    'similarity_score': 0.0
                }
            
            # Find matching user
            matches = self.embedding_store.search_similar(
                embedding, 
                top_k=5, 
                threshold=self.config['VERIFICATION_THRESHOLD']
            )
            
            if not matches:
                return {
                    'success': False,
                    'error': "No matching face found",
                    'similarity_score': 0.0
                }
            
            best_match = matches[0]
            
            # If specific user_id provided, verify it matches
            if user_id:
                user_match = None
                for match in matches:
                    if match['embedding_id'].startswith(f"{user_id}_"):
                        user_match = match
                        break
                
                if not user_match:
                    return {
                        'success': False,
                        'error': "Face does not match user",
                        'similarity_score': 0.0
                    }
                
                best_match = user_match
            
            return {
                'success': True,
                'user_id': best_match['embedding_id'].split('_')[0],
                'similarity_score': best_match['similarity'],
                'confidence': confidence,
                'quality_score': quality_score,
                'liveness_data': liveness_result,
                'match_data': best_match
            }
            
        except Exception as e:
            logger.error(f"Error in authentication: {e}")
            return {
                'success': False,
                'error': f"Authentication error: {str(e)}",
                'similarity_score': 0.0
            }
    
    def _assess_image_quality(self, frame, bbox):
        """Assess image quality"""
        try:
            x1, y1, x2, y2 = bbox
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Brightness (should be well-lit, not too dark or bright)
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # Contrast
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 64.0)
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_score = min(1.0, sharpness / 1000.0)
            
            # Face size score
            face_area = (x2 - x1) * (y2 - y1)
            optimal_area = 200 * 250  # From camera guide
            size_ratio = face_area / optimal_area
            size_score = 1.0 - abs(1.0 - size_ratio) if size_ratio <= 2.0 else 0.5
            
            # Weighted average
            weights = [0.3, 0.25, 0.3, 0.15]  # brightness, contrast, sharpness, size
            scores = [brightness_score, contrast_score, sharpness_score, size_score]
            
            quality_score = sum(w * s for w, s in zip(weights, scores))
            
            logger.debug(f"Quality assessment - Brightness: {brightness_score:.2f}, "
                        f"Contrast: {contrast_score:.2f}, Sharpness: {sharpness_score:.2f}, "
                        f"Size: {size_score:.2f}, Overall: {quality_score:.2f}")
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return 0.0
    
    def save_embedding(self, user_id: str, embedding: np.ndarray, metadata: dict = None):
        """Save embedding to storage"""
        try:
            # Save to ChromaDB
            embedding_id = self.embedding_store.add_embedding(user_id, embedding, metadata)
            
            if embedding_id:
                # Also save to FAISS for fallback
                self._add_to_faiss(user_id, embedding)
                logger.info(f"Saved embedding for user {user_id}")
                return embedding_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error saving embedding: {e}")
            return None
    
    def _add_to_faiss(self, user_id: str, embedding: np.ndarray):
        """Add embedding to FAISS index"""
        try:
            embedding_normalized = embedding / np.linalg.norm(embedding)
            self.faiss_index.add(embedding_normalized.reshape(1, -1))
            
            if user_id not in self.user_embeddings:
                self.user_embeddings[user_id] = []
            
            self.user_embeddings[user_id].append(embedding)
            self.user_names.append(user_id)
            
        except Exception as e:
            logger.error(f"Error adding to FAISS: {e}")
    
    def reset_liveness_detector(self):
        """Reset liveness detector for new session"""
        self.liveness_detector.reset()
    
    def get_system_status(self):
        """Get system status information"""
        return {
            'insightface_ready': self.app is not None,
            'chromadb_ready': self.embedding_store.client is not None,
            'faiss_embeddings': self.faiss_index.ntotal,
            'liveness_detector_ready': self.liveness_detector is not None,
            'obstacle_detector_ready': self.obstacle_detector is not None,
        }
"""
Session Manager for Face Recognition Engine
Handles stateful components like LivenessDetector across HTTP requests
"""
import json
import logging
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Type

from django.core.cache import caches
from django.conf import settings
from django.utils import timezone

from core.face_recognition_engine import LivenessDetector, ObstacleDetector


logger = logging.getLogger('face_recognition')


# Global in-memory cache for LivenessDetectors (much faster than Redis for same-process access)
_liveness_detector_cache: Dict[str, LivenessDetector] = {}
_liveness_detector_cache_lock = threading.Lock()
_cache_cleanup_counter = 0
_MAX_CACHED_DETECTORS = 100  # Limit to prevent memory bloat


class SessionManager:
    """
    Manages session state for face recognition components across HTTP requests.
    Uses Redis cache to store and retrieve stateful objects like LivenessDetector.
    
    OPTIMIZATION: Uses in-memory cache first, falls back to Redis for persistence.
    This avoids expensive serialization/deserialization on every frame.
    """
    
    def __init__(self, cache_name='sessions'):
        self.cache = caches[cache_name]
        self.session_timeout = settings.FACE_RECOGNITION_CONFIG.get('SESSION_TIMEOUT_MINUTES', 10) * 60
        
    def create_session_token(self) -> str:
        """Create a unique session token"""
        return str(uuid.uuid4())
    
    def _get_cache_key(self, session_token: str, component: str) -> str:
        """Generate cache key for session component"""
        return f"face_session:{session_token}:{component}"
    
    def _cleanup_old_detectors(self):
        """Cleanup old detectors from in-memory cache if too many"""
        global _cache_cleanup_counter, _liveness_detector_cache
        
        _cache_cleanup_counter += 1
        
        # Only cleanup every 50 operations
        if _cache_cleanup_counter < 50:
            return
            
        _cache_cleanup_counter = 0
        
        with _liveness_detector_cache_lock:
            if len(_liveness_detector_cache) > _MAX_CACHED_DETECTORS:
                # Remove oldest half of entries
                keys_to_remove = list(_liveness_detector_cache.keys())[:len(_liveness_detector_cache) // 2]
                for key in keys_to_remove:
                    del _liveness_detector_cache[key]
                logger.debug(f"Cleaned up {len(keys_to_remove)} old liveness detectors from memory cache")
    
    def store_liveness_detector(self, session_token: str, liveness_detector: LivenessDetector) -> None:
        """Store LivenessDetector state in cache"""
        from .face_recognition_engine import json_serializable
        
        # OPTIMIZATION: Store in in-memory cache first (fast path)
        with _liveness_detector_cache_lock:
            _liveness_detector_cache[session_token] = liveness_detector
        
        # Cleanup old detectors periodically
        self._cleanup_old_detectors()
        
        # Also store state in Redis for persistence (but less frequently)
        # Only update Redis every 5 frames to reduce overhead
        if liveness_detector.frame_count % 5 != 0:
            return  # Skip Redis update for most frames
            
        cache_key = self._get_cache_key(session_token, 'liveness_detector')
        
        # Serialize detector state with proper JSON serialization
        state = {
            'blink_count': int(liveness_detector.blink_count),
            'total_blinks': int(getattr(liveness_detector, 'total_blinks', 0)),
            'valid_blinks': int(getattr(liveness_detector, 'valid_blinks', 0)),
            'blink_counter': int(getattr(liveness_detector, 'blink_counter', 0)),
            'frame_counter': int(getattr(liveness_detector, 'frame_counter', 0)),
            'motion_events': int(getattr(liveness_detector, 'motion_events', 0)),
            'motion_score': float(getattr(liveness_detector, 'motion_score', 0.0)),
            'motion_history': json_serializable(liveness_detector.motion_history[-10:]),  # Keep last 10 for memory
            'last_bbox_center': json_serializable(getattr(liveness_detector, 'last_bbox_center', None)),
            'last_bbox_size': float(getattr(liveness_detector, 'last_bbox_size', 0.0)) if getattr(liveness_detector, 'last_bbox_size', None) else None,
            'last_motion_time': float(getattr(liveness_detector, 'last_motion_time', 0.0)),
            'previous_landmarks': json_serializable(liveness_detector.previous_landmarks),
            'blink_frames': int(liveness_detector.blink_frames),
            'blink_start_frame': int(liveness_detector.blink_start_frame) if liveness_detector.blink_start_frame else None,
            'last_ear': float(liveness_detector.last_ear),
            'last_blink_time': float(getattr(liveness_detector, 'last_blink_time', 0.0)),
            'frame_count': int(liveness_detector.frame_count),
            'baseline_ear': float(liveness_detector.baseline_ear) if liveness_detector.baseline_ear else None,
            'ear_history': json_serializable(getattr(liveness_detector, 'ear_history', [])),
            'eye_visibility_score': float(getattr(liveness_detector, 'eye_visibility_score', 0.0)),
            'blink_quality_scores': json_serializable(getattr(liveness_detector, 'blink_quality_scores', [])),
            'created_at': timezone.now().isoformat(),
        }
        
        serialized_state = json.dumps(json_serializable(state))
        self.cache.set(cache_key, serialized_state, timeout=self.session_timeout)
    
    def get_liveness_detector(self, session_token: str) -> Optional[LivenessDetector]:
        """Retrieve LivenessDetector from cache and restore state
        
        OPTIMIZATION: Check in-memory cache first (fast path), then Redis (slow path)
        """
        from .face_recognition_engine import LivenessDetector
        import numpy as np
        
        # FAST PATH: Check in-memory cache first
        with _liveness_detector_cache_lock:
            if session_token in _liveness_detector_cache:
                return _liveness_detector_cache[session_token]
        
        # SLOW PATH: Check Redis cache
        cache_key = self._get_cache_key(session_token, 'liveness_detector')
        cached_state = self.cache.get(cache_key)
        
        if not cached_state:
            return None
        
        try:
            state = json.loads(cached_state)
            
            # Create new detector WITHOUT initializing MediaPipe (skip_init=True)
            detector = LivenessDetector(skip_init=True)
            detector.blink_count = state.get('blink_count', 0)
            detector.total_blinks = state.get('total_blinks', 0)
            detector.valid_blinks = state.get('valid_blinks', 0)
            detector.blink_counter = state.get('blink_counter', 0)
            detector.frame_counter = state.get('frame_counter', 0)
            detector.motion_events = state.get('motion_events', 0)
            detector.motion_score = state.get('motion_score', 0.0)
            detector.motion_history = state.get('motion_history', [])
            
            # Restore bbox tracking for motion
            last_bbox_center = state.get('last_bbox_center')
            if last_bbox_center:
                detector.last_bbox_center = np.array(last_bbox_center, dtype=np.float32)
            else:
                detector.last_bbox_center = None
            
            detector.last_bbox_size = state.get('last_bbox_size')
            detector.last_motion_time = state.get('last_motion_time', 0.0)
            
            detector.previous_landmarks = state.get('previous_landmarks')
            detector.blink_frames = state.get('blink_frames', 0)
            detector.blink_start_frame = state.get('blink_start_frame')
            detector.last_ear = state.get('last_ear', 0.0)
            detector.last_blink_time = state.get('last_blink_time', 0.0)
            detector.frame_count = state.get('frame_count', 0)
            detector.baseline_ear = state.get('baseline_ear')
            detector.ear_history = state.get('ear_history', [])
            detector.eye_visibility_score = state.get('eye_visibility_score', 0.0)
            detector.blink_quality_scores = state.get('blink_quality_scores', [])
            
            # OPTIMIZATION: Only reinitialize MediaPipe if configured to do so
            # Skipping this saves significant CPU/GPU resources
            from django.conf import settings
            skip_reinit = settings.FACE_RECOGNITION_CONFIG.get('SKIP_MEDIAPIPE_REINIT', False)
            if not skip_reinit:
                detector.reinitialize_face_mesh()
            
            # Store in in-memory cache for fast subsequent access
            with _liveness_detector_cache_lock:
                _liveness_detector_cache[session_token] = detector
            
            return detector
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to restore liveness detector state: {e}")
            return None
    
    def clear_liveness_detector_cache(self, session_token: str) -> None:
        """Clear liveness detector from both in-memory and Redis cache"""
        with _liveness_detector_cache_lock:
            if session_token in _liveness_detector_cache:
                del _liveness_detector_cache[session_token]
        
        cache_key = self._get_cache_key(session_token, 'liveness_detector')
        self.cache.delete(cache_key)
    
    def store_enrollment_data(self, session_token: str, embeddings: list, metadata: dict) -> None:
        """Store enrollment embeddings temporarily for averaging"""
        cache_key = self._get_cache_key(session_token, 'enrollment_embeddings')
        
        data = {
            'embeddings': embeddings,
            'metadata': metadata,
            'updated_at': timezone.now().isoformat(),
        }
        
        self.cache.set(cache_key, json.dumps(data), timeout=self.session_timeout)
        logger.debug(f"Stored {len(embeddings)} embeddings for session {session_token}")
    
    def get_enrollment_data(self, session_token: str) -> tuple[list, dict]:
        """Retrieve enrollment embeddings for averaging"""
        cache_key = self._get_cache_key(session_token, 'enrollment_embeddings')
        cached_data = self.cache.get(cache_key)
        
        if not cached_data:
            return [], {}
        
        try:
            data = json.loads(cached_data)
            return data.get('embeddings', []), data.get('metadata', {})
        except json.JSONDecodeError:
            return [], {}
    
    def add_enrollment_embedding(self, session_token: str, embedding: list, frame_metadata: dict) -> int:
        """Add a single embedding to enrollment collection"""
        from .face_recognition_engine import json_serializable
        
        embeddings, metadata = self.get_enrollment_data(session_token)
        
        embeddings.append(embedding)
        # Ensure frame_metadata is JSON serializable
        serializable_metadata = json_serializable(frame_metadata)
        metadata[f'frame_{len(embeddings)}'] = serializable_metadata
        
        self.store_enrollment_data(session_token, embeddings, metadata)
        return len(embeddings)
    
    def clear_session(self, session_token: str) -> None:
        """Clear all session data including in-memory cache"""
        # Clear in-memory cache first
        self.clear_liveness_detector_cache(session_token)
        
        # Clear Redis cache
        components = ['liveness_detector', 'enrollment_embeddings']
        for component in components:
            cache_key = self._get_cache_key(session_token, component)
            self.cache.delete(cache_key)
    
    def cleanup_session(self, session_token: str) -> None:
        """Clean up failed session - alias for clear_session"""
        self.clear_session(session_token)
    
    def initialize_session(self, session_token: str, session_type: str = 'enrollment') -> None:
        """Initialize a new session with basic data"""
        cache_key = self._get_cache_key(session_token, 'session_info')
        
        session_info = {
            'session_type': session_type,
            'created_at': timezone.now().isoformat(),
            'frame_count': 0,
        }
        
        self.cache.set(cache_key, json.dumps(session_info), timeout=self.session_timeout)
        logger.debug(f"Initialized {session_type} session {session_token}")
    
    def get_session_data(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get general session data"""
        cache_key = self._get_cache_key(session_token, 'session_info')
        cached_data = self.cache.get(cache_key)
        
        if not cached_data:
            return None
            
        try:
            return json.loads(cached_data)
        except json.JSONDecodeError:
            return None
    
    def update_session_data(self, session_token: str, data: Dict[str, Any]) -> None:
        """Update general session data"""
        from .face_recognition_engine import json_serializable
        
        cache_key = self._get_cache_key(session_token, 'session_info')
        
        # Ensure all data is JSON serializable
        serializable_data = json_serializable(data)
        serialized_data = json.dumps(serializable_data)
        
        self.cache.set(cache_key, serialized_data, timeout=self.session_timeout)
        logger.debug(f"Updated session data for {session_token}: {list(data.keys())}")
    
    def complete_enrollment(self, session_token: str) -> Dict[str, Any]:
        """Complete enrollment and return results"""
        embeddings, metadata = self.get_enrollment_data(session_token)
        
        if not embeddings:
            return {
                'success': False,
                'error': 'No embeddings found for completion'
            }
        
        # Clear session after completion
        self.clear_session(session_token)
        
        return {
            'success': True,
            'user_id': metadata.get('user_id'),
            'embeddings_count': len(embeddings)
        }
    
    def get_or_create_liveness_detector(self, session_token: str) -> LivenessDetector:
        """Get existing detector or create new one"""
        detector = self.get_liveness_detector(session_token)
        
        if detector is None:
            detector = LivenessDetector()
            # Immediately store the new detector in cache
            self.store_liveness_detector(session_token, detector)
            logger.debug(f"Created new liveness detector for session {session_token}")
        
        return detector
    
    def update_liveness_detector(self, session_token: str, detector: LivenessDetector) -> None:
        """Update detector state in cache after processing frame"""
        self.store_liveness_detector(session_token, detector)
    
    def get_session_stats(self, session_token: str) -> dict:
        """Get statistics for current session"""
        detector = self.get_liveness_detector(session_token)
        embeddings, metadata = self.get_enrollment_data(session_token)
        
        stats = {
            'session_token': session_token,
            'liveness_blink_count': detector.blink_count if detector else 0,
            'liveness_frame_count': detector.frame_count if detector else 0,
            'enrollment_embeddings_count': len(embeddings),
            'has_liveness_state': detector is not None,
        }
        
        return stats


# Global session manager instance
session_manager = SessionManager()
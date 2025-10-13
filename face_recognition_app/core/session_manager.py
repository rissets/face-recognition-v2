"""
Session Manager for Face Recognition Engine
Handles stateful components like LivenessDetector across HTTP requests
"""
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Type

from django.core.cache import caches
from django.conf import settings
from django.utils import timezone

from core.face_recognition_engine import LivenessDetector, ObstacleDetector


logger = logging.getLogger('face_recognition')

class SessionManager:
    """
    Manages session state for face recognition components across HTTP requests.
    Uses Redis cache to store and retrieve stateful objects like LivenessDetector.
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
    
    def store_liveness_detector(self, session_token: str, liveness_detector: LivenessDetector) -> None:
        """Store LivenessDetector state in cache"""
        from .face_recognition_engine import json_serializable
        
        cache_key = self._get_cache_key(session_token, 'liveness_detector')
        
        # Serialize detector state with proper JSON serialization
        state = {
            'blink_count': int(liveness_detector.blink_count),
            'total_blinks': int(getattr(liveness_detector, 'total_blinks', 0)),
            'frame_counter': int(getattr(liveness_detector, 'frame_counter', 0)),
            'motion_events': int(getattr(liveness_detector, 'motion_events', 0)),
            'motion_history': json_serializable(liveness_detector.motion_history[-10:]),  # Keep last 10 for memory
            'previous_landmarks': json_serializable(liveness_detector.previous_landmarks),
            'blink_frames': int(liveness_detector.blink_frames),
            'last_ear': float(liveness_detector.last_ear),
            'frame_count': int(liveness_detector.frame_count),
            'baseline_ear': float(liveness_detector.baseline_ear) if liveness_detector.baseline_ear else None,
            'ear_history': json_serializable(getattr(liveness_detector, 'ear_history', [])),
            'created_at': timezone.now().isoformat(),
        }
        
        serialized_state = json.dumps(json_serializable(state))
        self.cache.set(cache_key, serialized_state, timeout=self.session_timeout)
        logger.debug(f"Stored liveness detector state for session {session_token}")
    
    def get_liveness_detector(self, session_token: str) -> Optional[LivenessDetector]:
        """Retrieve LivenessDetector from cache and restore state"""
        from .face_recognition_engine import LivenessDetector
        
        cache_key = self._get_cache_key(session_token, 'liveness_detector')
        cached_state = self.cache.get(cache_key)
        
        if not cached_state:
            logger.debug(f"No cached liveness detector found for session {session_token}")
            return None
        
        try:
            state = json.loads(cached_state)
            
            # Create new detector and restore state
            detector = LivenessDetector()
            detector.blink_count = state.get('blink_count', 0)
            detector.total_blinks = state.get('total_blinks', 0)
            detector.frame_counter = state.get('frame_counter', 0)
            detector.motion_events = state.get('motion_events', 0)
            detector.motion_history = state.get('motion_history', [])
            detector.previous_landmarks = state.get('previous_landmarks')
            detector.blink_frames = state.get('blink_frames', 0)
            detector.last_ear = state.get('last_ear', 0.0)
            detector.frame_count = state.get('frame_count', 0)
            detector.baseline_ear = state.get('baseline_ear')
            detector.ear_history = state.get('ear_history', [])
            
            logger.debug(f"Restored liveness detector for session {session_token} - total_blinks: {detector.total_blinks}, frame_counter: {detector.frame_counter}")
            return detector
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to restore liveness detector state: {e}")
            return None
    
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
        """Clear all session data"""
        components = ['liveness_detector', 'enrollment_embeddings']
        
        for component in components:
            cache_key = self._get_cache_key(session_token, component)
            self.cache.delete(cache_key)
        
        logger.debug(f"Cleared session data for {session_token}")
    
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
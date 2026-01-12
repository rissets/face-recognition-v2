"""
Optimized Old Photo Embedding Extractor

This module provides a highly optimized face embedding extraction from old profile photos.
Key optimizations:
1. Singleton FaceAnalysis instance - avoid recreating detector each time
2. Progressive detection strategies with fallback
3. Redis caching of extracted embeddings
4. Async task integration for background processing
"""

import logging
import threading
import hashlib
from typing import Optional, Tuple
from functools import lru_cache

import cv2
import numpy as np
from django.conf import settings
from django.core.cache import caches

logger = logging.getLogger("core.old_photo_extractor")


class OldPhotoEmbeddingExtractor:
    """
    Singleton class for extracting face embeddings from old profile photos.
    Uses a shared FaceAnalysis instance to avoid repeated initialization overhead.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Detection strategies in order of preference (fastest to most thorough)
    STRATEGIES = [
        {"name": "standard", "det_size": (640, 640), "det_thresh": 0.5, "preprocess": None},
        {"name": "lenient", "det_size": (640, 640), "det_thresh": 0.3, "preprocess": None},
        {"name": "clahe_lenient", "det_size": (640, 640), "det_thresh": 0.3, "preprocess": "clahe"},
        {"name": "very_lenient", "det_size": (640, 640), "det_thresh": 0.2, "preprocess": "clahe"},
        {"name": "gamma", "det_size": (640, 640), "det_thresh": 0.3, "preprocess": "gamma"},
        {"name": "large", "det_size": (1024, 1024), "det_thresh": 0.2, "preprocess": "upscale"},
    ]
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._face_apps = {}  # Cached FaceAnalysis instances by (det_size, det_thresh)
        self._init_lock = threading.Lock()
        
        # Cache configuration
        self._cache_ttl = getattr(settings, 'OLD_PHOTO_EMBEDDING_CACHE_TTL', 86400)  # 24 hours
        
        self._initialized = True
        logger.info("OldPhotoEmbeddingExtractor singleton initialized")
    
    def _get_face_app(self, det_size: Tuple[int, int], det_thresh: float):
        """Get or create a FaceAnalysis instance with specific settings"""
        key = (det_size, det_thresh)
        
        if key not in self._face_apps:
            with self._init_lock:
                if key not in self._face_apps:
                    try:
                        from insightface.app import FaceAnalysis
                        
                        app = FaceAnalysis(
                            name='buffalo_l',
                            allowed_modules=['detection', 'recognition'],
                            providers=['CPUExecutionProvider']
                        )
                        app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
                        
                        self._face_apps[key] = app
                        logger.info(f"Created FaceAnalysis instance: det_size={det_size}, thresh={det_thresh}")
                    except Exception as e:
                        logger.error(f"Failed to create FaceAnalysis: {e}")
                        return None
        
        return self._face_apps.get(key)
    
    def _apply_preprocessing(self, image: np.ndarray, method: Optional[str]) -> np.ndarray:
        """Apply image preprocessing based on strategy"""
        if method is None:
            return image
        
        if method == "clahe":
            # CLAHE enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        if method == "gamma":
            # Gamma correction
            gamma = 1.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        
        if method == "upscale":
            # Upscale small images
            height, width = image.shape[:2]
            if max(height, width) < 1024:
                scale = 1280.0 / max(height, width)
                new_size = (int(width * scale), int(height * scale))
                return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
            return image
        
        return image
    
    def _get_cache_key(self, image_bytes: bytes) -> str:
        """Generate cache key from image content"""
        return f"old_photo_emb:{hashlib.md5(image_bytes).hexdigest()}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Try to get embedding from Redis cache"""
        try:
            cache = caches['embeddings']
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                return np.frombuffer(cached_data, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to Redis cache"""
        try:
            cache = caches['embeddings']
            cache.set(cache_key, embedding.astype(np.float32).tobytes(), self._cache_ttl)
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract face embedding from image bytes.
        Uses progressive detection strategies for robustness.
        
        Args:
            image_bytes: JPEG/PNG image as bytes
            
        Returns:
            Normalized embedding vector or None if extraction fails
        """
        # Check Redis cache first
        cache_key = self._get_cache_key(image_bytes)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            logger.debug("Old photo embedding retrieved from cache")
            return cached
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.warning("Failed to decode image bytes")
            return None
        
        # Try each strategy in order
        for strategy in self.STRATEGIES:
            try:
                face_app = self._get_face_app(
                    strategy["det_size"],
                    strategy["det_thresh"]
                )
                
                if face_app is None:
                    continue
                
                # Apply preprocessing
                processed = self._apply_preprocessing(image, strategy["preprocess"])
                
                # Detect faces
                faces = face_app.get(processed)
                
                if faces and len(faces) > 0:
                    logger.info(f"✅ Strategy '{strategy['name']}' found {len(faces)} face(s)")
                    
                    # Get the largest face
                    largest_face = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    )
                    
                    embedding = largest_face.embedding
                    if embedding is not None:
                        if not isinstance(embedding, np.ndarray):
                            embedding = np.array(embedding)
                        
                        # Normalize for cosine similarity
                        embedding_norm = embedding / np.linalg.norm(embedding)
                        embedding_norm = embedding_norm.astype(np.float32)
                        
                        # Cache the result
                        self._save_to_cache(cache_key, embedding_norm)
                        
                        return embedding_norm
                        
            except Exception as e:
                logger.warning(f"Strategy '{strategy['name']}' failed: {e}")
                continue
        
        logger.warning("⚠️ No face detected after trying all strategies")
        return None
    
    def extract_from_client_user(self, client_user) -> Optional[np.ndarray]:
        """
        Extract embedding from a ClientUser's old_profile_photo.
        First checks database cache, then extracts if needed.
        
        Args:
            client_user: ClientUser instance
            
        Returns:
            Normalized embedding vector or None
        """
        # Check database cache first (fastest)
        cached = client_user.get_cached_old_photo_embedding()
        if cached is not None:
            logger.debug(f"Using DB-cached old photo embedding for user {client_user.external_user_id}")
            return cached
        
        # Check if old_profile_photo exists
        if not client_user.old_profile_photo:
            logger.debug(f"No old_profile_photo for user {client_user.external_user_id}")
            return None
        
        try:
            # Read from storage
            old_photo_file = client_user.old_profile_photo.open('rb')
            image_bytes = old_photo_file.read()
            old_photo_file.close()
            
            # Extract embedding
            embedding = self.extract_embedding(image_bytes)
            
            if embedding is not None:
                # Cache in database for future use
                try:
                    client_user.cache_old_photo_embedding(embedding)
                    logger.info(f"💾 Cached old photo embedding to DB for user {client_user.external_user_id}")
                except Exception as e:
                    logger.warning(f"Failed to cache old photo embedding to DB: {e}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting old photo embedding for user {client_user.external_user_id}: {e}")
            return None


# Singleton instance getter
def get_old_photo_extractor() -> OldPhotoEmbeddingExtractor:
    """Get the singleton OldPhotoEmbeddingExtractor instance"""
    return OldPhotoEmbeddingExtractor()


# Convenience function for direct use
def extract_old_photo_embedding(client_user) -> Optional[np.ndarray]:
    """
    Convenience function to extract old photo embedding from a ClientUser.
    
    Args:
        client_user: ClientUser instance
        
    Returns:
        Normalized embedding vector or None
    """
    return get_old_photo_extractor().extract_from_client_user(client_user)

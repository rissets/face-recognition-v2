"""
Async Face Processing Utilities
Provides thread pool executor and async wrappers for CPU-intensive operations
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Callable, Any, Optional
import numpy as np
import cv2

logger = logging.getLogger('face_recognition')

# Shared thread pool for CPU-intensive operations
# Using max 4 workers to prevent CPU overload while allowing parallel processing
_face_processing_executor = ThreadPoolExecutor(
    max_workers=4, 
    thread_name_prefix="face_proc_"
)


def get_executor() -> ThreadPoolExecutor:
    """Get the shared thread pool executor"""
    return _face_processing_executor


def run_in_executor(func: Callable) -> Callable:
    """
    Decorator to run synchronous function in thread pool executor
    
    Usage:
        @run_in_executor
        def heavy_computation(data):
            # CPU intensive work
            return result
        
        # Then in async context:
        result = await heavy_computation(data)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_face_processing_executor, lambda: func(*args, **kwargs))
    return wrapper


async def run_sync_in_executor(func: Callable, *args, **kwargs) -> Any:
    """
    Run a synchronous function in the thread pool executor
    
    Args:
        func: Synchronous function to run
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
    
    Returns:
        Result from func
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        return await loop.run_in_executor(
            _face_processing_executor,
            lambda: func(*args, **kwargs)
        )
    else:
        return await loop.run_in_executor(_face_processing_executor, func, *args)


class AsyncFaceProcessor:
    """
    Async wrapper for face processing operations
    Offloads CPU-intensive work to thread pool
    """
    
    def __init__(self, face_engine):
        """
        Initialize with face engine instance
        
        Args:
            face_engine: FaceRecognitionEngine instance
        """
        self.face_engine = face_engine
        self._executor = _face_processing_executor
    
    async def detect_faces(self, frame: np.ndarray) -> list:
        """Async face detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.face_engine.detect_faces,
            frame
        )
    
    async def extract_embedding(self, frame: np.ndarray) -> tuple:
        """Async embedding extraction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.face_engine.extract_embedding,
            frame
        )
    
    async def authenticate_user(self, frame: np.ndarray, user_id: str = None) -> dict:
        """Async user authentication"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.face_engine.authenticate_user,
            frame,
            user_id
        )
    
    async def process_enrollment_frame(self, frame: np.ndarray, user_id: str) -> tuple:
        """Async enrollment frame processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.face_engine.process_frame_for_enrollment,
            frame,
            user_id
        )
    
    async def detect_liveness(self, frame: np.ndarray, bbox: Optional[tuple] = None) -> dict:
        """Async liveness detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.face_engine.liveness_detector.detect_blink,
            frame,
            bbox
        )
    
    async def detect_obstacles(self, frame: np.ndarray, bbox: tuple) -> tuple:
        """Async obstacle detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.face_engine.obstacle_detector.detect_obstacles,
            frame,
            bbox
        )
    
    async def assess_quality(self, frame: np.ndarray, bbox: tuple) -> float:
        """Async image quality assessment"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.face_engine._assess_image_quality,
            frame,
            bbox
        )
    
    async def search_similar(
        self, 
        embedding: np.ndarray, 
        top_k: int = 5, 
        threshold: float = 0.4,
        client_id: str = None
    ) -> list:
        """Async similarity search"""
        loop = asyncio.get_event_loop()
        
        # Use optimized store if available
        if hasattr(self.face_engine, 'optimized_store') and self.face_engine.optimized_store:
            return await loop.run_in_executor(
                self._executor,
                lambda: self.face_engine.optimized_store.search_similar_with_legacy_fallback(
                    embedding, top_k, threshold, client_id
                )
            )
        else:
            return await loop.run_in_executor(
                self._executor,
                lambda: self.face_engine.embedding_store.search_similar(
                    embedding, top_k, threshold
                )
            )


class BatchFaceProcessor:
    """
    Batch processing for multiple frames/embeddings
    Useful for enrollment with multiple samples
    """
    
    def __init__(self, face_engine):
        self.face_engine = face_engine
        self._executor = _face_processing_executor
    
    async def batch_extract_embeddings(self, frames: list) -> list:
        """
        Extract embeddings from multiple frames in parallel
        
        Args:
            frames: List of image frames
        
        Returns:
            List of (result, error) tuples
        """
        loop = asyncio.get_event_loop()
        
        async def extract_single(frame):
            return await loop.run_in_executor(
                self._executor,
                self.face_engine.extract_embedding,
                frame
            )
        
        # Process all frames concurrently
        tasks = [extract_single(frame) for frame in frames]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def batch_search_similar(
        self, 
        embeddings: list, 
        top_k: int = 5, 
        threshold: float = 0.4,
        client_id: str = None
    ) -> list:
        """
        Search similar embeddings in batch
        
        Args:
            embeddings: List of embedding vectors
            top_k: Number of top results per query
            threshold: Similarity threshold
            client_id: Client ID for isolation
        
        Returns:
            List of search results
        """
        loop = asyncio.get_event_loop()
        
        async def search_single(embedding):
            if hasattr(self.face_engine, 'optimized_store') and self.face_engine.optimized_store:
                return await loop.run_in_executor(
                    self._executor,
                    lambda: self.face_engine.optimized_store.search_similar_with_legacy_fallback(
                        embedding, top_k, threshold, client_id
                    )
                )
            else:
                return await loop.run_in_executor(
                    self._executor,
                    lambda: self.face_engine.embedding_store.search_similar(
                        embedding, top_k, threshold
                    )
                )
        
        tasks = [search_single(emb) for emb in embeddings]
        return await asyncio.gather(*tasks, return_exceptions=True)


def cleanup_executor():
    """Cleanup thread pool executor on shutdown"""
    global _face_processing_executor
    _face_processing_executor.shutdown(wait=False)
    logger.info("Face processing executor shutdown")


# Register cleanup on module unload
import atexit
atexit.register(cleanup_executor)

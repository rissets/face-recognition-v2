"""
Embedding utilities for face recognition
Handles embedding averaging, normalization, and quality assessment
"""
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from django.conf import settings


logger = logging.getLogger('face_recognition')

class EmbeddingAverager:
    """
    Handles multi-frame embedding averaging for robust face templates
    """
    
    def __init__(self):
        self.config = settings.FACE_RECOGNITION_CONFIG
        self.min_embeddings = self.config.get('MIN_ENROLLMENT_FRAMES', 3)
        self.max_embeddings = self.config.get('MAX_ENROLLMENT_SAMPLES', 5)
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding quality and format"""
        if embedding is None:
            return False
        
        if not isinstance(embedding, np.ndarray):
            return False
            
        if embedding.shape[0] != self.config.get('EMBEDDING_DIMENSION', 512):
            logger.warning(f"Invalid embedding dimension: {embedding.shape[0]}")
            return False
            
        # Check for zero or invalid embeddings
        if np.all(embedding == 0) or np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logger.warning("Invalid embedding values detected")
            return False
            
        # Check embedding magnitude (should be normalized or close to it)
        magnitude = np.linalg.norm(embedding)
        if magnitude < 0.1 or magnitude > 2.0:
            logger.warning(f"Unusual embedding magnitude: {magnitude}")
            return False
            
        return True
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit vector"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def calculate_embedding_quality(self, embeddings: List[np.ndarray]) -> float:
        """
        Calculate quality score based on embedding consistency
        Higher consistency = higher quality
        """
        if len(embeddings) < 2:
            return 1.0
        
        # Convert to normalized embeddings
        normalized_embeddings = [self.normalize_embedding(emb) for emb in embeddings]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(normalized_embeddings)):
            for j in range(i + 1, len(normalized_embeddings)):
                sim = np.dot(normalized_embeddings[i], normalized_embeddings[j])
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Quality is based on consistency (higher mean similarity = better quality)
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        # Quality score: high mean similarity with low variance is best
        quality = mean_similarity - (std_similarity * 0.5)
        return max(0.0, min(1.0, quality))  # Clamp between 0 and 1
    
    def average_embeddings(self, embeddings: List[np.ndarray], weights: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Average multiple embeddings into a single robust template
        
        Args:
            embeddings: List of face embeddings
            weights: Optional weights for each embedding (quality-based)
            
        Returns:
            Tuple of (averaged_embedding, metadata)
        """
        if not embeddings:
            raise ValueError("No embeddings provided for averaging")
        
        # Validate all embeddings
        valid_embeddings = []
        valid_weights = []
        
        for i, embedding in enumerate(embeddings):
            if self.validate_embedding(embedding):
                valid_embeddings.append(embedding)
                if weights and i < len(weights):
                    valid_weights.append(weights[i])
                else:
                    valid_weights.append(1.0)
        
        if not valid_embeddings:
            raise ValueError("No valid embeddings found")
        
        if len(valid_embeddings) < self.min_embeddings:
            logger.warning(f"Only {len(valid_embeddings)} valid embeddings, minimum is {self.min_embeddings}")
        
        # Normalize embeddings first
        normalized_embeddings = [self.normalize_embedding(emb) for emb in valid_embeddings]
        
        # Weighted average
        if len(valid_weights) != len(normalized_embeddings):
            valid_weights = [1.0] * len(normalized_embeddings)
        
        # Convert to numpy arrays
        embedding_matrix = np.array(normalized_embeddings)
        weight_array = np.array(valid_weights)
        
        # Normalize weights
        weight_array = weight_array / np.sum(weight_array)
        
        # Calculate weighted average
        averaged_embedding = np.average(embedding_matrix, axis=0, weights=weight_array)
        
        # Re-normalize the result
        final_embedding = self.normalize_embedding(averaged_embedding)
        
        # Calculate quality metrics
        quality_score = self.calculate_embedding_quality(valid_embeddings)
        
        metadata = {
            'total_embeddings_provided': len(embeddings),
            'valid_embeddings_used': len(valid_embeddings),
            'quality_score': quality_score,
            'embedding_dimension': final_embedding.shape[0],
            'weights_used': valid_weights,
            'final_magnitude': np.linalg.norm(final_embedding),
        }
        
        logger.info(f"Averaged {len(valid_embeddings)} embeddings (quality: {quality_score:.3f})")
        
        return final_embedding, metadata
    
    def should_add_more_samples(self, current_count: int, quality_score: float) -> bool:
        """
        Determine if more samples should be collected based on count and quality
        """
        if current_count < self.min_embeddings:
            return True
        
        if current_count >= self.max_embeddings:
            return False
        
        # If quality is low, collect more samples (up to max)
        quality_threshold = 0.7
        if quality_score < quality_threshold and current_count < self.max_embeddings:
            return True
        
        return False
    
    def get_collection_progress(self, current_count: int, quality_score: float) -> Dict:
        """
        Get enrollment progress information
        """
        needs_more = self.should_add_more_samples(current_count, quality_score)
        
        progress = {
            'current_samples': current_count,
            'minimum_samples': self.min_embeddings,
            'maximum_samples': self.max_embeddings,
            'quality_score': quality_score,
            'needs_more_samples': needs_more,
            'progress_percentage': min(100, (current_count / self.max_embeddings) * 100),
            'is_sufficient': current_count >= self.min_embeddings,
            'can_complete': current_count >= self.min_embeddings,
        }
        
        return progress


# Global instance
embedding_averager = EmbeddingAverager()
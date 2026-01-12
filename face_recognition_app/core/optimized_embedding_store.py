"""
Optimized Embedding Store with Per-Client Collections and Redis Caching
This module provides high-performance face embedding storage and retrieval.
"""
import logging
import time
import json
import hashlib
import numpy as np
import chromadb
from typing import Dict, List, Optional, Any, Tuple
from django.conf import settings
from django.core.cache import cache
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger('face_recognition')

# Thread pool for async operations
_thread_pool = ThreadPoolExecutor(max_workers=4)
_lock = threading.Lock()


class OptimizedChromaEmbeddingStore:
    """
    Optimized ChromaDB integration with:
    1. Per-client collection isolation
    2. Redis caching for embeddings
    3. Connection pooling
    4. Metadata pre-filtering
    """
    
    # Cache settings
    EMBEDDING_CACHE_TTL = 3600  # 1 hour
    SEARCH_CACHE_TTL = 300  # 5 minutes
    USER_EMBEDDING_CACHE_PREFIX = "face_emb:"
    SEARCH_RESULT_CACHE_PREFIX = "face_search:"
    
    def __init__(self, client_id: str = None):
        """
        Initialize with optional client isolation
        
        Args:
            client_id: If provided, use per-client collection for data isolation
        """
        self.client_id = client_id
        self._collections_cache = {}
        self._client = None
        
        try:
            chroma_config = settings.CHROMA_DB_CONFIG
            self._client = chromadb.HttpClient(
                host=chroma_config['host'],
                port=chroma_config['port']
            )
            
            # Get or create base collection (for backward compatibility)
            self.base_collection_name = chroma_config['collection_name']
            
            logger.info(f"OptimizedChromaEmbeddingStore initialized (client_id: {client_id})")
            
        except Exception as e:
            logger.error(f"Failed to initialize OptimizedChromaEmbeddingStore: {e}")
            self._client = None
    
    def _get_collection_name(self, client_id: str = None) -> str:
        """Get collection name for client (isolated or shared)"""
        cid = client_id or self.client_id
        if cid:
            # Per-client collection for data isolation
            return f"{self.base_collection_name}_{cid}"
        return self.base_collection_name
    
    def _get_collection(self, client_id: str = None):
        """Get or create collection with caching"""
        collection_name = self._get_collection_name(client_id)
        
        # Check cache first
        if collection_name in self._collections_cache:
            return self._collections_cache[collection_name]
        
        with _lock:
            # Double-check after acquiring lock
            if collection_name in self._collections_cache:
                return self._collections_cache[collection_name]
            
            try:
                try:
                    collection = self._client.get_collection(collection_name)
                except Exception:
                    collection = self._client.create_collection(
                        name=collection_name,
                        metadata={"description": f"Face embeddings for {client_id or 'all'}", "hnsw:space": "cosine"}
                    )
                
                self._collections_cache[collection_name] = collection
                logger.info(f"Collection '{collection_name}' cached")
                return collection
                
            except Exception as e:
                logger.error(f"Failed to get/create collection {collection_name}: {e}")
                return None
    
    def _get_legacy_collection(self):
        """Get the legacy shared collection for backward compatibility"""
        return self._get_collection(client_id=None)
    
    def _clean_metadata_for_chroma(self, metadata: dict) -> dict:
        """Clean metadata to ensure ChromaDB compatibility"""
        if not metadata:
            return {}
        
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = None
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, (list, tuple, np.ndarray)):
                cleaned[key] = json.dumps(value.tolist() if hasattr(value, 'tolist') else list(value))
            elif isinstance(value, dict):
                cleaned[key] = json.dumps(value)
            else:
                cleaned[key] = str(value)
        
        return cleaned
    
    def _get_embedding_cache_key(self, user_id: str) -> str:
        """Generate cache key for user embedding"""
        return f"{self.USER_EMBEDDING_CACHE_PREFIX}{user_id}"
    
    def _get_search_cache_key(self, embedding: np.ndarray, client_id: str, top_k: int, threshold: float) -> str:
        """Generate cache key for search results"""
        # Create hash of embedding for cache key
        emb_hash = hashlib.md5(embedding.tobytes()).hexdigest()[:16]
        return f"{self.SEARCH_RESULT_CACHE_PREFIX}{client_id or 'all'}:{emb_hash}:{top_k}:{threshold}"
    
    def add_embedding(self, user_id: str, embedding: np.ndarray, metadata: dict = None, client_id: str = None) -> Optional[str]:
        """
        Add face embedding with client isolation and caching
        
        Args:
            user_id: User identifier
            embedding: Face embedding vector
            metadata: Additional metadata
            client_id: Client ID for isolation (uses instance client_id if not provided)
        """
        try:
            cid = client_id or self.client_id
            collection = self._get_collection(cid)
            
            if collection is None:
                return None
            
            import uuid
            embedding_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
            
            # Ensure user_id and client_id in metadata
            if metadata is None:
                metadata = {}
            metadata['user_id'] = user_id
            if cid:
                metadata['client_id'] = cid
            
            clean_metadata = self._clean_metadata_for_chroma(metadata)
            
            collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[clean_metadata],
                ids=[embedding_id]
            )
            
            # Cache the embedding
            cache_key = self._get_embedding_cache_key(user_id)
            cache.set(cache_key, {
                'embedding': embedding.tolist(),
                'metadata': clean_metadata,
                'embedding_id': embedding_id
            }, timeout=self.EMBEDDING_CACHE_TTL)
            
            # Invalidate search caches for this client
            self._invalidate_search_cache(cid)
            
            logger.debug(f"Added embedding for user {user_id} (client: {cid})")
            return embedding_id
            
        except Exception as e:
            logger.error(f"Failed to add embedding: {e}")
            return None
    
    def _invalidate_search_cache(self, client_id: str = None):
        """Invalidate search cache for a client (called after adding new embeddings)"""
        try:
            # We can't easily invalidate all search caches, but they have short TTL
            # This is a placeholder for more sophisticated cache invalidation
            pass
        except Exception:
            pass
    
    def get_user_embedding(self, user_id: str, client_id: str = None) -> Optional[Dict]:
        """
        Get cached embedding for user
        
        Args:
            user_id: User identifier
            client_id: Client ID for isolation
        """
        # Check Redis cache first
        cache_key = self._get_embedding_cache_key(user_id)
        cached = cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for user embedding: {user_id}")
            return cached
        
        # Query ChromaDB
        try:
            cid = client_id or self.client_id
            collection = self._get_collection(cid)
            
            if collection is None:
                return None
            
            # Query by user_id prefix
            results = collection.get(
                where={"user_id": user_id} if user_id else None,
                include=["embeddings", "metadatas"]
            )
            
            if results['ids'] and len(results['ids']) > 0:
                # Get the most recent embedding (last one)
                idx = -1
                result = {
                    'embedding_id': results['ids'][idx],
                    'embedding': np.array(results['embeddings'][idx]),
                    'metadata': results['metadatas'][idx] if results['metadatas'] else {}
                }
                
                # Cache it
                cache.set(cache_key, {
                    'embedding': results['embeddings'][idx],
                    'metadata': result['metadata'],
                    'embedding_id': result['embedding_id']
                }, timeout=self.EMBEDDING_CACHE_TTL)
                
                logger.debug(f"Cache miss, loaded from DB for user: {user_id}")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user embedding: {e}")
            return None
    
    def search_similar(
        self, 
        embedding: np.ndarray, 
        top_k: int = 5, 
        threshold: float = 0.4,
        client_id: str = None,
        user_id_filter: str = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Search for similar embeddings with client isolation and caching
        
        Args:
            embedding: Query embedding
            top_k: Number of top results
            threshold: Minimum similarity threshold
            client_id: Client ID for isolation
            user_id_filter: Optional filter for specific user
            use_cache: Whether to use cache (default True)
        """
        cid = client_id or self.client_id
        
        # Check cache first
        if use_cache:
            cache_key = self._get_search_cache_key(embedding, cid, top_k, threshold)
            cached = cache.get(cache_key)
            if cached:
                logger.debug("Search cache hit")
                return cached
        
        try:
            collection = self._get_collection(cid)
            
            if collection is None:
                # Fallback to legacy collection
                collection = self._get_legacy_collection()
                if collection is None:
                    return []
            
            # Build where clause for filtering
            where_clause = None
            if cid:
                where_clause = {"client_id": cid}
            if user_id_filter:
                if where_clause:
                    where_clause = {"$and": [{"client_id": cid}, {"user_id": user_id_filter}]}
                else:
                    where_clause = {"user_id": user_id_filter}
            
            # Query ChromaDB with filtering
            results = collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances"]
            )
            
            matches = []
            if results['ids'] and results['distances']:
                for i, (embedding_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                    # Convert distance to similarity (ChromaDB uses L2 distance by default)
                    # For cosine space: similarity = 1 - distance/2
                    similarity = 1.0 - (distance / 2.0)
                    
                    if similarity >= threshold:
                        matches.append({
                            'embedding_id': embedding_id,
                            'similarity': similarity,
                            'distance': distance,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'source': 'chroma_optimized'
                        })
            
            # Cache results
            if use_cache and matches:
                cache_key = self._get_search_cache_key(embedding, cid, top_k, threshold)
                cache.set(cache_key, matches, timeout=self.SEARCH_CACHE_TTL)
            
            return matches
            
        except Exception as e:
            logger.error(f"Failed to search embeddings: {e}")
            return []
    
    def search_similar_with_legacy_fallback(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.4,
        client_id: str = None
    ) -> List[Dict]:
        """
        Search with fallback to legacy collection if per-client collection is empty
        """
        # First try per-client collection
        matches = self.search_similar(embedding, top_k, threshold, client_id)
        
        if matches:
            return matches
        
        # Fallback to legacy collection (for backward compatibility)
        if client_id:
            logger.debug(f"Falling back to legacy collection for client {client_id}")
            legacy_matches = self.search_similar(embedding, top_k, threshold, client_id=None)
            
            # Filter by client_id in metadata
            filtered = []
            for match in legacy_matches:
                meta = match.get('metadata', {})
                meta_client = meta.get('client_id', '')
                # Extract client_id from user_id if not in metadata
                user_id = meta.get('user_id', '')
                if ':' in user_id:
                    user_client_id = user_id.split(':')[0]
                else:
                    user_client_id = ''
                
                if meta_client == client_id or user_client_id == client_id:
                    filtered.append(match)
            
            return filtered
        
        return []
    
    def delete_user_embeddings(self, user_id: str, client_id: str = None) -> bool:
        """Delete all embeddings for a user with cache invalidation"""
        try:
            cid = client_id or self.client_id
            collection = self._get_collection(cid)
            
            if collection is None:
                return False
            
            # Query all embeddings for the user
            results = collection.get()
            user_embedding_ids = []
            
            for embedding_id in results['ids']:
                if embedding_id.startswith(f"{user_id}_"):
                    user_embedding_ids.append(embedding_id)
            
            if user_embedding_ids:
                collection.delete(ids=user_embedding_ids)
                logger.info(f"Deleted {len(user_embedding_ids)} embeddings for user {user_id}")
            
            # Invalidate cache
            cache_key = self._get_embedding_cache_key(user_id)
            cache.delete(cache_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user embeddings: {e}")
            return False
    
    def count_embeddings(self, client_id: str = None) -> int:
        """Get count of embeddings in collection"""
        try:
            collection = self._get_collection(client_id or self.client_id)
            if collection:
                return collection.count()
            return 0
        except Exception:
            return 0
    
    def get_collection_stats(self, client_id: str = None) -> Dict[str, Any]:
        """Get statistics for collection"""
        try:
            collection = self._get_collection(client_id or self.client_id)
            if collection:
                return {
                    'name': collection.name,
                    'count': collection.count(),
                    'client_id': client_id or self.client_id
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


class EmbeddingCacheManager:
    """
    Centralized cache manager for face embeddings
    Handles caching of user embeddings in Redis for fast retrieval
    """
    
    CACHE_PREFIX = "face_user_emb:"
    DEFAULT_TTL = 3600  # 1 hour
    
    @classmethod
    def cache_user_embedding(cls, client_id: str, user_id: str, embedding: np.ndarray, metadata: dict = None) -> bool:
        """Cache user embedding in Redis"""
        try:
            cache_key = f"{cls.CACHE_PREFIX}{client_id}:{user_id}"
            data = {
                'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                'metadata': metadata or {},
                'cached_at': time.time()
            }
            cache.set(cache_key, data, timeout=cls.DEFAULT_TTL)
            logger.debug(f"Cached embedding for {client_id}:{user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
            return False
    
    @classmethod
    def get_cached_embedding(cls, client_id: str, user_id: str) -> Optional[Tuple[np.ndarray, dict]]:
        """Get cached user embedding from Redis"""
        try:
            cache_key = f"{cls.CACHE_PREFIX}{client_id}:{user_id}"
            data = cache.get(cache_key)
            if data:
                embedding = np.array(data['embedding'])
                metadata = data.get('metadata', {})
                logger.debug(f"Cache hit for {client_id}:{user_id}")
                return embedding, metadata
            return None
        except Exception as e:
            logger.error(f"Failed to get cached embedding: {e}")
            return None
    
    @classmethod
    def invalidate_user_cache(cls, client_id: str, user_id: str) -> bool:
        """Invalidate cached embedding for user"""
        try:
            cache_key = f"{cls.CACHE_PREFIX}{client_id}:{user_id}"
            cache.delete(cache_key)
            return True
        except Exception:
            return False
    
    @classmethod
    def preload_client_embeddings(cls, client_id: str, embedding_store: 'OptimizedChromaEmbeddingStore') -> int:
        """Preload all embeddings for a client into cache"""
        try:
            collection = embedding_store._get_collection(client_id)
            if not collection:
                return 0
            
            results = collection.get(include=["embeddings", "metadatas"])
            count = 0
            
            for i, (emb_id, embedding, metadata) in enumerate(zip(
                results['ids'], 
                results['embeddings'], 
                results['metadatas'] or [{}] * len(results['ids'])
            )):
                user_id = metadata.get('user_id', '')
                if user_id:
                    # Extract external user id from composite user_id (client_id:external_user_id)
                    if ':' in user_id:
                        _, ext_user_id = user_id.split(':', 1)
                    else:
                        ext_user_id = user_id
                    
                    cls.cache_user_embedding(client_id, ext_user_id, np.array(embedding), metadata)
                    count += 1
            
            logger.info(f"Preloaded {count} embeddings for client {client_id}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to preload embeddings: {e}")
            return 0


# Global instance for backward compatibility
_default_store = None


def get_optimized_store(client_id: str = None) -> OptimizedChromaEmbeddingStore:
    """Get optimized embedding store (singleton pattern for non-client-specific use)"""
    global _default_store
    
    if client_id:
        return OptimizedChromaEmbeddingStore(client_id=client_id)
    
    if _default_store is None:
        _default_store = OptimizedChromaEmbeddingStore()
    
    return _default_store

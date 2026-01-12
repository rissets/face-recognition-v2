#!/usr/bin/env python
"""
Performance Benchmark Script for Face Recognition System

This script measures the performance improvements from the optimizations:
1. Per-client ChromaDB collections
2. Redis embedding caching
3. Cached embeddings in PostgreSQL

Usage:
    python benchmark_performance.py [--iterations N] [--client CLIENT_ID]
"""
import os
import sys
import time
import argparse
import logging
from statistics import mean, stdev

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_app.settings')

import django
django.setup()

import numpy as np
from django.conf import settings
from clients.models import Client, ClientUser
from core.face_recognition_engine import FaceRecognitionEngine, ChromaEmbeddingStore
from core.optimized_embedding_store import OptimizedChromaEmbeddingStore, EmbeddingCacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('benchmark')


class PerformanceBenchmark:
    """Benchmark face recognition system performance"""
    
    def __init__(self, client_id: str = None, iterations: int = 10):
        self.client_id = client_id
        self.iterations = iterations
        
        # Initialize stores
        self.legacy_store = ChromaEmbeddingStore()
        self.optimized_store = OptimizedChromaEmbeddingStore(client_id=client_id)
        
        # Generate test embedding
        self.test_embedding = np.random.randn(512).astype(np.float32)
        self.test_embedding = self.test_embedding / np.linalg.norm(self.test_embedding)
    
    def benchmark_search(self, store, name: str, use_client_filter: bool = False):
        """Benchmark search operation"""
        times = []
        
        for i in range(self.iterations):
            start = time.perf_counter()
            
            if use_client_filter and hasattr(store, 'search_similar_with_legacy_fallback'):
                results = store.search_similar_with_legacy_fallback(
                    self.test_embedding, 
                    top_k=5, 
                    threshold=0.3,
                    client_id=self.client_id
                )
            else:
                results = store.search_similar(
                    self.test_embedding,
                    top_k=5,
                    threshold=0.3
                )
            
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        return {
            'name': name,
            'iterations': self.iterations,
            'min_ms': min(times),
            'max_ms': max(times),
            'avg_ms': mean(times),
            'std_ms': stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_cached_verification(self, user: ClientUser):
        """Benchmark verification using cached embedding"""
        cached_embedding = user.get_cached_embedding()
        
        if cached_embedding is None:
            return {'error': 'No cached embedding'}
        
        times = []
        
        for i in range(self.iterations):
            start = time.perf_counter()
            
            # Simulate verification
            similarity = float(np.dot(
                self.test_embedding / np.linalg.norm(self.test_embedding),
                cached_embedding / np.linalg.norm(cached_embedding)
            ))
            
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
        
        return {
            'name': 'Cached Verification',
            'iterations': self.iterations,
            'min_ms': min(times),
            'max_ms': max(times),
            'avg_ms': mean(times),
            'std_ms': stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_redis_cache(self, client_id: str, user_id: str):
        """Benchmark Redis cache lookup"""
        times = []
        
        for i in range(self.iterations):
            start = time.perf_counter()
            
            result = EmbeddingCacheManager.get_cached_embedding(client_id, user_id)
            
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
        
        cache_hit = result is not None
        
        return {
            'name': f'Redis Cache {"(HIT)" if cache_hit else "(MISS)"}',
            'iterations': self.iterations,
            'min_ms': min(times),
            'max_ms': max(times),
            'avg_ms': mean(times),
            'std_ms': stdev(times) if len(times) > 1 else 0
        }
    
    def run_all_benchmarks(self):
        """Run all benchmarks and return results"""
        results = []
        
        print("\n" + "="*60)
        print("🚀 Face Recognition Performance Benchmark")
        print("="*60)
        print(f"   Client ID: {self.client_id or 'All'}")
        print(f"   Iterations: {self.iterations}")
        print("="*60 + "\n")
        
        # Get embedding count
        try:
            legacy_count = self.legacy_store.collection.count() if self.legacy_store.collection else 0
            optimized_count = self.optimized_store.count_embeddings()
            print(f"📊 Legacy collection: {legacy_count} embeddings")
            print(f"📊 Optimized collection: {optimized_count} embeddings\n")
        except Exception as e:
            print(f"⚠️ Could not get embedding counts: {e}\n")
        
        # 1. Legacy search (no filtering)
        print("🔍 Running Legacy Search benchmark...")
        result = self.benchmark_search(self.legacy_store, "Legacy ChromaDB Search")
        results.append(result)
        print(f"   Avg: {result['avg_ms']:.3f}ms (±{result['std_ms']:.3f}ms)")
        
        # 2. Optimized search with client filtering
        print("🔍 Running Optimized Search benchmark...")
        result = self.benchmark_search(
            self.optimized_store, 
            "Optimized ChromaDB Search (Client Filtered)",
            use_client_filter=True
        )
        results.append(result)
        print(f"   Avg: {result['avg_ms']:.3f}ms (±{result['std_ms']:.3f}ms)")
        
        # 3. Redis cache lookup
        if self.client_id:
            try:
                user = ClientUser.objects.filter(
                    client__client_id=self.client_id, 
                    is_enrolled=True
                ).first()
                
                if user:
                    print(f"🔍 Running Redis Cache benchmark (user: {user.external_user_id})...")
                    result = self.benchmark_redis_cache(self.client_id, user.external_user_id)
                    results.append(result)
                    print(f"   Avg: {result['avg_ms']:.3f}ms (±{result['std_ms']:.3f}ms)")
                    
                    # 4. Cached verification
                    if user.cached_embedding:
                        print("🔍 Running Cached Verification benchmark...")
                        result = self.benchmark_cached_verification(user)
                        results.append(result)
                        print(f"   Avg: {result['avg_ms']:.3f}ms (±{result['std_ms']:.3f}ms)")
                    else:
                        print("⚠️ Skipping cached verification - no cached embedding")
            except Exception as e:
                print(f"⚠️ Could not run user-specific benchmarks: {e}")
        
        # Summary
        print("\n" + "="*60)
        print("📈 BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        for r in results:
            print(f"\n{r['name']}:")
            print(f"   Min: {r['min_ms']:.3f}ms | Max: {r['max_ms']:.3f}ms")
            print(f"   Avg: {r['avg_ms']:.3f}ms (±{r['std_ms']:.3f}ms)")
        
        # Calculate improvements
        if len(results) >= 2:
            legacy_avg = results[0]['avg_ms']
            optimized_avg = results[1]['avg_ms']
            if legacy_avg > 0:
                improvement = ((legacy_avg - optimized_avg) / legacy_avg) * 100
                print(f"\n🎯 Search Improvement: {improvement:.1f}% faster")
        
        if len(results) >= 4:
            chroma_avg = results[1]['avg_ms']
            cached_avg = results[3]['avg_ms']
            if chroma_avg > 0:
                improvement = ((chroma_avg - cached_avg) / chroma_avg) * 100
                print(f"🎯 Cached vs ChromaDB: {improvement:.1f}% faster")
        
        print("\n" + "="*60 + "\n")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark face recognition performance')
    parser.add_argument('--iterations', '-n', type=int, default=10,
                        help='Number of iterations per benchmark')
    parser.add_argument('--client', '-c', type=str, default=None,
                        help='Client ID to benchmark')
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(
        client_id=args.client,
        iterations=args.iterations
    )
    
    benchmark.run_all_benchmarks()


if __name__ == '__main__':
    main()

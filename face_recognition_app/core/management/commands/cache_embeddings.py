"""
Management command to cache embeddings for existing enrolled users.
This one-time migration improves authentication performance by caching
embeddings in PostgreSQL and Redis for faster lookup.
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from clients.models import ClientUser
from core.face_recognition_engine import FaceRecognitionEngine
from core.optimized_embedding_store import EmbeddingCacheManager
import numpy as np
import logging

logger = logging.getLogger('face_recognition')


class Command(BaseCommand):
    help = 'Cache face embeddings for enrolled users to improve authentication performance'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--client',
            type=str,
            help='Only process users for a specific client ID',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be cached without making changes',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Re-cache even if already cached',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of users to process at a time',
        )
    
    def handle(self, *args, **options):
        client_id = options.get('client')
        dry_run = options.get('dry_run')
        force = options.get('force')
        batch_size = options.get('batch_size')
        
        self.stdout.write(self.style.NOTICE(
            f"🚀 Starting embedding cache migration..."
        ))
        
        # Get enrolled users
        queryset = ClientUser.objects.filter(is_enrolled=True).select_related('client')
        
        if client_id:
            queryset = queryset.filter(client__client_id=client_id)
            self.stdout.write(f"   Filtering by client: {client_id}")
        
        if not force:
            queryset = queryset.filter(cached_embedding__isnull=True)
            self.stdout.write("   Skipping users with existing cache")
        
        total_users = queryset.count()
        self.stdout.write(f"   Found {total_users} users to process")
        
        if total_users == 0:
            self.stdout.write(self.style.SUCCESS("✅ No users to process!"))
            return
        
        if dry_run:
            self.stdout.write(self.style.WARNING("   DRY RUN - no changes will be made"))
        
        # Initialize face engine
        face_engine = FaceRecognitionEngine()
        
        cached_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Process in batches
        for offset in range(0, total_users, batch_size):
            batch = queryset[offset:offset + batch_size]
            
            for user in batch:
                try:
                    engine_user_id = f"{user.client.client_id}:{user.external_user_id}"
                    
                    # Get embedding from ChromaDB
                    if face_engine.optimized_store:
                        user_data = face_engine.optimized_store.get_user_embedding(
                            engine_user_id, 
                            user.client.client_id
                        )
                    else:
                        user_data = face_engine.embedding_store.get_user_embedding(engine_user_id)
                    
                    if not user_data:
                        self.stdout.write(f"   ⚠️  No embedding found for {user.external_user_id}")
                        skipped_count += 1
                        continue
                    
                    embedding = user_data.get('embedding')
                    if embedding is None:
                        self.stdout.write(f"   ⚠️  Empty embedding for {user.external_user_id}")
                        skipped_count += 1
                        continue
                    
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding, dtype=np.float32)
                    
                    if dry_run:
                        self.stdout.write(f"   [DRY] Would cache embedding for {user.external_user_id}")
                    else:
                        # Cache in database
                        user.cache_embedding(embedding)
                        
                        # Also cache in Redis
                        EmbeddingCacheManager.cache_user_embedding(
                            user.client.client_id,
                            user.external_user_id,
                            embedding,
                            {'source': 'migration', 'migrated_at': timezone.now().isoformat()}
                        )
                        
                        self.stdout.write(f"   ✅ Cached embedding for {user.external_user_id}")
                    
                    cached_count += 1
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(
                        f"   ❌ Error caching {user.external_user_id}: {e}"
                    ))
                    failed_count += 1
            
            # Progress update
            processed = min(offset + batch_size, total_users)
            self.stdout.write(f"   Progress: {processed}/{total_users}")
        
        # Summary
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(
            f"✅ Migration complete!"
        ))
        self.stdout.write(f"   Cached: {cached_count}")
        self.stdout.write(f"   Skipped: {skipped_count}")
        self.stdout.write(f"   Failed: {failed_count}")
        
        if dry_run:
            self.stdout.write(self.style.WARNING(
                "\n   This was a dry run. Run without --dry-run to apply changes."
            ))

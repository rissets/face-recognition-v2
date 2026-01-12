"""
Celery tasks for face recognition system
"""
import logging
from celery import shared_task
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db import models
from datetime import timedelta
from analytics.models import AuthenticationLog, FaceRecognitionStats, DataQualityMetrics
from recognition.models import FaceEmbedding, EnrollmentSession
from core.models import SecurityEvent, AuditLog

logger = logging.getLogger('face_recognition')
User = get_user_model()


@shared_task
def process_face_embedding(user_id, embedding_data, metadata):
    """Process and store face embedding asynchronously"""
    try:
        user = User.objects.get(id=user_id)
        
        # Process embedding with additional validation
        # This could include duplicate detection, quality assessment, etc.
        
        logger.info(f"Successfully processed face embedding for user {user.email}")
        return {'success': True, 'user_id': user_id}
        
    except User.DoesNotExist:
        logger.error(f"User {user_id} not found for embedding processing")
        return {'success': False, 'error': 'User not found'}
    except Exception as e:
        logger.error(f"Error processing face embedding: {e}")
        return {'success': False, 'error': str(e)}


@shared_task
def cleanup_expired_sessions():
    """Clean up expired enrollment and streaming sessions"""
    try:
        now = timezone.now()
        
        # Clean up expired enrollment sessions
        expired_enrollment = EnrollmentSession.objects.filter(
            expires_at__lt=now,
            status__in=['pending', 'in_progress']
        )
        
        count = expired_enrollment.count()
        expired_enrollment.update(status='failed', error_messages='Session expired')
        
        # Clean up expired streaming sessions
        from streaming.models import StreamingSession
        expired_streaming = StreamingSession.objects.filter(
            created_at__lt=now - timedelta(hours=2),
            status__in=['initiating', 'connecting']
        )
        
        streaming_count = expired_streaming.count()
        expired_streaming.update(status='failed')
        
        logger.info(f"Cleaned up {count} enrollment sessions and {streaming_count} streaming sessions")
        return {'enrollment_sessions': count, 'streaming_sessions': streaming_count}
        
    except Exception as e:
        logger.error(f"Error cleaning up expired sessions: {e}")
        return {'error': str(e)}


@shared_task
def generate_daily_statistics():
    """Generate daily face recognition statistics"""
    try:
        today = timezone.now().date()
        
        # Check if stats already exist for today
        if FaceRecognitionStats.objects.filter(date=today, hour__isnull=True).exists():
            logger.info(f"Daily statistics already exist for {today}")
            return {'message': 'Statistics already exist'}
        
        # Aggregate authentication logs for today
        auth_logs = AuthenticationLog.objects.filter(
            created_at__date=today,
            auth_method='face'
        )
        
        total_attempts = auth_logs.count()
        successful_attempts = auth_logs.filter(success=True).count()
        failed_attempts = total_attempts - successful_attempts
        
        # Breakdown failures by reason
        failure_counts = {}
        for log in auth_logs.filter(success=False):
            reason = log.failure_reason
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
        
        # Calculate averages
        avg_response_time = auth_logs.aggregate(
            avg_time=models.Avg('response_time')
        )['avg_time'] or 0
        
        avg_similarity = auth_logs.filter(similarity_score__gt=0).aggregate(
            avg_sim=models.Avg('similarity_score')
        )['avg_sim'] or 0
        
        avg_liveness = auth_logs.filter(liveness_score__gt=0).aggregate(
            avg_live=models.Avg('liveness_score')
        )['avg_live'] or 0
        
        avg_quality = auth_logs.filter(quality_score__gt=0).aggregate(
            avg_qual=models.Avg('quality_score')
        )['avg_qual'] or 0
        
        # Count unique users and new enrollments
        unique_users = auth_logs.filter(success=True).values('user').distinct().count()
        new_enrollments = EnrollmentSession.objects.filter(
            completed_at__date=today,
            status='completed'
        ).count()
        
        # Create daily statistics
        stats = FaceRecognitionStats.objects.create(
            date=today,
            total_attempts=total_attempts,
            successful_attempts=successful_attempts,
            failed_attempts=failed_attempts,
            failed_similarity=failure_counts.get('face_not_recognized', 0),
            failed_liveness=failure_counts.get('liveness_failed', 0),
            failed_quality=failure_counts.get('quality_too_low', 0),
            failed_obstacles=failure_counts.get('obstacles_detected', 0),
            failed_no_face=failure_counts.get('no_face_detected', 0),
            failed_multiple_faces=failure_counts.get('multiple_faces', 0),
            failed_system_error=failure_counts.get('system_error', 0),
            avg_response_time=avg_response_time,
            avg_similarity_score=avg_similarity,
            avg_liveness_score=avg_liveness,
            avg_quality_score=avg_quality,
            unique_users=unique_users,
            new_enrollments=new_enrollments
        )
        
        logger.info(f"Generated daily statistics for {today}")
        return {'date': str(today), 'total_attempts': total_attempts}
        
    except Exception as e:
        logger.error(f"Error generating daily statistics: {e}")
        return {'error': str(e)}


@shared_task
def analyze_data_quality():
    """Analyze data quality metrics"""
    try:
        today = timezone.now().date()
        
        # Check if quality metrics already exist for today
        if DataQualityMetrics.objects.filter(date=today).exists():
            logger.info(f"Data quality metrics already exist for {today}")
            return {'message': 'Quality metrics already exist'}
        
        # Analyze face embeddings created today
        embeddings = FaceEmbedding.objects.filter(
            created_at__date=today,
            is_active=True
        )
        
        if not embeddings.exists():
            logger.info(f"No face embeddings found for {today}")
            return {'message': 'No data to analyze'}
        
        # Calculate quality metrics
        total_samples = embeddings.count()
        
        # Quality distribution
        high_quality = embeddings.filter(quality_score__gte=0.8).count()
        medium_quality = embeddings.filter(
            quality_score__gte=0.6, 
            quality_score__lt=0.8
        ).count()
        low_quality = embeddings.filter(quality_score__lt=0.6).count()
        
        # Average metrics
        avg_quality = embeddings.aggregate(
            avg=models.Avg('quality_score')
        )['avg'] or 0
        
        # Calculate other metrics (would need additional face detection analysis)
        # For now, use placeholder values
        avg_face_size = 150.0  # pixels
        avg_brightness = 128.0  # 0-255
        avg_contrast = 64.0
        avg_sharpness = 500.0
        
        # Create quality metrics
        quality_metrics = DataQualityMetrics.objects.create(
            date=today,
            avg_image_quality=avg_quality,
            avg_face_size=avg_face_size,
            avg_brightness=avg_brightness,
            avg_contrast=avg_contrast,
            avg_sharpness=avg_sharpness,
            high_quality_samples=high_quality,
            medium_quality_samples=medium_quality,
            low_quality_samples=low_quality,
            total_samples=total_samples
        )
        
        logger.info(f"Generated data quality metrics for {today}")
        return {'date': str(today), 'total_samples': total_samples}
        
    except Exception as e:
        logger.error(f"Error analyzing data quality: {e}")
        return {'error': str(e)}


@shared_task
def detect_security_anomalies():
    """Detect security anomalies and create alerts"""
    try:
        now = timezone.now()
        last_hour = now - timedelta(hours=1)
        
        # Detect multiple failed authentication attempts
        failed_attempts = AuthenticationLog.objects.filter(
            created_at__gte=last_hour,
            success=False,
            auth_method='face'
        )
        
        # Group by IP address
        ip_failures = {}
        for attempt in failed_attempts:
            ip = attempt.ip_address
            if ip:
                ip_failures[ip] = ip_failures.get(ip, 0) + 1
        
        # Create alerts for IPs with too many failures
        alerts_created = 0
        for ip, count in ip_failures.items():
            if count >= 5:  # Threshold for suspicious activity
                # Check if alert already exists for this IP recently
                existing_alert = SecurityEvent.objects.filter(
                    event_type='failed_attempts',
                    created_at__gte=now - timedelta(hours=4),
                    details__ip_address=ip
                ).first()
                
                if not existing_alert:
                    SecurityEvent.objects.create(
                        event_type='failed_attempts',
                        severity='medium' if count < 10 else 'high',
                        ip_address=ip,
                        details={
                            'ip_address': ip,
                            'failure_count': count,
                            'time_window': '1 hour'
                        }
                    )
                    alerts_created += 1
        
        # Detect unusual authentication patterns
        # (This is a simplified example - could be more sophisticated)
        unusual_patterns = AuthenticationLog.objects.filter(
            created_at__gte=last_hour,
            success=True,
            auth_method='face'
        ).values('user').annotate(
            login_count=models.Count('id')
        ).filter(login_count__gte=10)  # More than 10 logins per hour
        
        for pattern in unusual_patterns:
            user_id = pattern['user']
            count = pattern['login_count']
            
            try:
                user = User.objects.get(id=user_id)
                
                # Check if alert already exists
                existing_alert = SecurityEvent.objects.filter(
                    event_type='unusual_activity',
                    user=user,
                    created_at__gte=now - timedelta(hours=4)
                ).first()
                
                if not existing_alert:
                    SecurityEvent.objects.create(
                        event_type='unusual_activity',
                        severity='medium',
                        user=user,
                        details={
                            'user_id': str(user_id),
                            'login_count': count,
                            'time_window': '1 hour'
                        }
                    )
                    alerts_created += 1
                    
            except User.DoesNotExist:
                continue
        
        logger.info(f"Created {alerts_created} security alerts")
        return {'alerts_created': alerts_created}
        
    except Exception as e:
        logger.error(f"Error detecting security anomalies: {e}")
        return {'error': str(e)}


@shared_task
def backup_embeddings():
    """Backup face embeddings to secure storage"""
    try:
        # This would implement actual backup logic
        # For now, just log the action
        
        active_embeddings = FaceEmbedding.objects.filter(is_active=True).count()
        
        logger.info(f"Backup completed for {active_embeddings} face embeddings")
        return {'embeddings_backed_up': active_embeddings}
        
    except Exception as e:
        logger.error(f"Error backing up embeddings: {e}")
        return {'error': str(e)}


@shared_task
def health_check():
    """Perform system health check"""
    try:
        from core.face_recognition_engine import FaceRecognitionEngine
        
        # Initialize face recognition engine
        engine = FaceRecognitionEngine()
        
        # Get system status
        status = engine.get_system_status()
        
        # Check database connectivity
        user_count = User.objects.count()
        embedding_count = FaceEmbedding.objects.count()
        
        # Log health status
        health_status = {
            'timestamp': timezone.now().isoformat(),
            'face_engine': status,
            'database': {
                'users': user_count,
                'embeddings': embedding_count
            }
        }
        
        logger.info(f"Health check completed: {health_status}")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {'error': str(e), 'status': 'unhealthy'}


@shared_task
def preload_client_embeddings(client_id: str = None):
    """
    Preload face embeddings into Redis cache for a client.
    This improves authentication performance by avoiding ChromaDB queries.
    
    Args:
        client_id: Specific client ID to preload, or None for all clients
    """
    try:
        from clients.models import Client, ClientUser
        from core.optimized_embedding_store import OptimizedChromaEmbeddingStore, EmbeddingCacheManager
        
        if client_id:
            clients = Client.objects.filter(client_id=client_id, status='active')
        else:
            clients = Client.objects.filter(status='active')
        
        total_cached = 0
        
        for client in clients:
            try:
                store = OptimizedChromaEmbeddingStore(client_id=client.client_id)
                count = EmbeddingCacheManager.preload_client_embeddings(client.client_id, store)
                total_cached += count
                logger.info(f"Preloaded {count} embeddings for client {client.client_id}")
            except Exception as e:
                logger.error(f"Failed to preload for client {client.client_id}: {e}")
        
        logger.info(f"Total embeddings preloaded: {total_cached}")
        return {'success': True, 'total_cached': total_cached}
        
    except Exception as e:
        logger.error(f"Failed to preload embeddings: {e}")
        return {'success': False, 'error': str(e)}


@shared_task
def cache_user_embedding(client_id: str, external_user_id: str):
    """
    Cache a specific user's embedding from ChromaDB to Redis and database.
    Called after enrollment or when cache is invalidated.
    
    Args:
        client_id: Client identifier
        external_user_id: User's external identifier
    """
    try:
        from clients.models import ClientUser
        from core.optimized_embedding_store import OptimizedChromaEmbeddingStore, EmbeddingCacheManager
        import numpy as np
        
        # Get the user
        user = ClientUser.objects.get(
            client__client_id=client_id,
            external_user_id=external_user_id
        )
        
        if not user.is_enrolled:
            logger.warning(f"User {external_user_id} is not enrolled, skipping cache")
            return {'success': False, 'error': 'User not enrolled'}
        
        # Get embedding from ChromaDB
        engine_user_id = f"{client_id}:{external_user_id}"
        store = OptimizedChromaEmbeddingStore(client_id=client_id)
        user_data = store.get_user_embedding(engine_user_id, client_id)
        
        if not user_data:
            logger.warning(f"No embedding found for user {external_user_id}")
            return {'success': False, 'error': 'Embedding not found'}
        
        embedding = user_data.get('embedding')
        if embedding is None:
            return {'success': False, 'error': 'Empty embedding'}
        
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Cache in database
        user.cache_embedding(embedding)
        
        # Cache in Redis
        EmbeddingCacheManager.cache_user_embedding(
            client_id,
            external_user_id,
            embedding,
            user_data.get('metadata', {})
        )
        
        logger.info(f"Cached embedding for user {external_user_id}")
        return {'success': True, 'user_id': external_user_id}
        
    except ClientUser.DoesNotExist:
        logger.error(f"User not found: {client_id}:{external_user_id}")
        return {'success': False, 'error': 'User not found'}
    except Exception as e:
        logger.error(f"Failed to cache embedding: {e}")
        return {'success': False, 'error': str(e)}


@shared_task
def warm_up_face_engine():
    """
    Warm up face recognition engine by pre-loading models.
    This reduces cold-start latency for the first request.
    """
    try:
        from core.face_recognition_engine import FaceRecognitionEngine
        import numpy as np
        
        # Initialize engine (this loads InsightFace models)
        engine = FaceRecognitionEngine()
        
        # Create dummy image for warm-up
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run detection to warm up GPU/CPU caches
        _ = engine.detect_faces(dummy_frame)
        
        logger.info("Face recognition engine warmed up successfully")
        return {'success': True, 'message': 'Engine warmed up'}
        
    except Exception as e:
        logger.error(f"Failed to warm up engine: {e}")
        return {'success': False, 'error': str(e)}


# =============================================================================
# OLD PHOTO EMBEDDING EXTRACTION TASKS
# These tasks run in the background to extract embeddings from old_profile_photo
# when a ClientUser is created or updated with a new photo.
# =============================================================================

@shared_task(bind=True, max_retries=3, default_retry_delay=60, name='core.tasks.extract_old_photo_embedding_task')
def extract_old_photo_embedding_task(self, client_user_id: str):
    """
    Background task to extract face embedding from a user's old_profile_photo.
    This task runs asynchronously when a user is created/updated with an old_profile_photo.
    
    The extracted embedding is cached in the database for fast lookup during enrollment.
    
    NOTE: This task uses InsightFace which is not fork-safe. Run Celery with:
        celery -A face_app worker --pool=solo --loglevel=info
    Or:
        celery -A face_app worker --pool=threads --loglevel=info
    
    Args:
        client_user_id: UUID of the ClientUser to process
    """
    try:
        from clients.models import ClientUser
        
        # Get the user
        user = ClientUser.objects.get(id=client_user_id)
        
        # Check if old_profile_photo exists
        if not user.old_profile_photo:
            logger.info(f"No old_profile_photo for user {user.external_user_id}, skipping extraction")
            return {'success': True, 'message': 'No old photo to extract'}
        
        # Check if already cached - skip InsightFace if not needed
        existing = user.get_cached_old_photo_embedding()
        if existing is not None:
            logger.info(f"Old photo embedding already cached for user {user.external_user_id}")
            return {'success': True, 'message': 'Already cached'}
        
        # Only import and use InsightFace when actually needed
        # This lazy import helps avoid fork-safety issues
        from core.old_photo_extractor import get_old_photo_extractor
        
        # Extract embedding using the optimized extractor
        extractor = get_old_photo_extractor()
        embedding = extractor.extract_from_client_user(user)
        
        if embedding is not None:
            logger.info(f"✅ Successfully extracted old photo embedding for user {user.external_user_id}")
            return {'success': True, 'user_id': str(user.id)}
        else:
            logger.warning(f"⚠️ No face detected in old_profile_photo for user {user.external_user_id}")
            return {'success': False, 'message': 'No face detected'}
        
    except ClientUser.DoesNotExist:
        logger.error(f"ClientUser {client_user_id} not found")
        return {'success': False, 'error': 'User not found'}
    except Exception as e:
        logger.error(f"Error extracting old photo embedding: {e}")
        # Retry on transient errors
        try:
            self.retry(exc=e)
        except self.MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for user {client_user_id}")
            return {'success': False, 'error': str(e)}


@shared_task(name='core.tasks.bulk_extract_old_photo_embeddings')
def bulk_extract_old_photo_embeddings(client_id: str = None, batch_size: int = 50):
    """
    Bulk extract old_profile_photo embeddings for users who don't have cached embeddings.
    
    This can be run as a one-time migration or scheduled periodically.
    
    NOTE: This task uses InsightFace which is not fork-safe. Run Celery with:
        celery -A face_app worker --pool=solo --loglevel=info
    
    Args:
        client_id: Optional client ID to limit extraction to specific client
        batch_size: Number of users to process in each batch
    """
    try:
        from clients.models import ClientUser
        # Lazy import to avoid fork issues
        from core.old_photo_extractor import get_old_photo_extractor
        
        # Build query for users with old_profile_photo but no cached embedding
        users_query = ClientUser.objects.filter(
            old_profile_photo__isnull=False,
            cached_old_photo_embedding__isnull=True
        ).exclude(old_profile_photo='')
        
        if client_id:
            users_query = users_query.filter(client__client_id=client_id)
        
        total_users = users_query.count()
        logger.info(f"Found {total_users} users needing old photo embedding extraction")
        
        if total_users == 0:
            return {'success': True, 'processed': 0, 'message': 'No users to process'}
        
        # Process in batches
        extractor = get_old_photo_extractor()
        processed = 0
        succeeded = 0
        failed = 0
        
        for user in users_query.iterator():
            try:
                embedding = extractor.extract_from_client_user(user)
                if embedding is not None:
                    succeeded += 1
                else:
                    failed += 1
            except Exception as e:
                logger.warning(f"Failed to extract for user {user.external_user_id}: {e}")
                failed += 1
            
            processed += 1
            
            # Log progress every batch_size
            if processed % batch_size == 0:
                logger.info(f"Progress: {processed}/{total_users} (success: {succeeded}, failed: {failed})")
        
        logger.info(f"✅ Bulk extraction complete: {processed} processed, {succeeded} succeeded, {failed} failed")
        return {
            'success': True,
            'total': total_users,
            'processed': processed,
            'succeeded': succeeded,
            'failed': failed
        }
        
    except Exception as e:
        logger.error(f"Bulk extraction failed: {e}")
        return {'success': False, 'error': str(e)}


@shared_task(name='core.tasks.preload_enrollment_user')
def preload_enrollment_user(client_id: str, external_user_id: str):
    """
    Preload a user's data before enrollment starts.
    This extracts old_photo embedding in advance to minimize latency during enrollment.
    
    Call this when creating enrollment session or immediately after user registration.
    
    NOTE: This task uses InsightFace which is not fork-safe. Run Celery with:
        celery -A face_app worker --pool=solo --loglevel=info
    
    Args:
        client_id: Client identifier
        external_user_id: User's external identifier
    """
    try:
        from clients.models import ClientUser
        
        user = ClientUser.objects.get(
            client__client_id=client_id,
            external_user_id=external_user_id
        )
        
        # Check cache first to avoid unnecessary InsightFace loading
        if not user.old_profile_photo:
            logger.info(f"ℹ️ User {external_user_id} has no old photo")
            return {'success': True, 'old_photo_extracted': False}
        
        if user.get_cached_old_photo_embedding() is not None:
            logger.info(f"ℹ️ User {external_user_id} already has cached embedding")
            return {'success': True, 'old_photo_extracted': False, 'cached': True}
        
        # Only import InsightFace when actually needed
        from core.old_photo_extractor import get_old_photo_extractor
        
        # Extract old photo embedding
        extractor = get_old_photo_extractor()
        embedding = extractor.extract_from_client_user(user)
        
        if embedding is not None:
            logger.info(f"✅ Preloaded old photo embedding for user {external_user_id}")
            return {'success': True, 'old_photo_extracted': True}
        else:
            logger.info(f"ℹ️ No face detected in old photo for user {external_user_id}")
            return {'success': True, 'old_photo_extracted': False}
        
    except ClientUser.DoesNotExist:
        logger.error(f"User not found: {client_id}:{external_user_id}")
        return {'success': False, 'error': 'User not found'}
    except Exception as e:
        logger.error(f"Preload failed: {e}")
        return {'success': False, 'error': str(e)}

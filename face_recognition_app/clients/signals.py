"""
Django signals for ClientUser model.

These signals handle:
1. Automatic extraction of old_profile_photo embeddings when a user is created/updated
2. Cache invalidation when embeddings are updated
"""

import logging
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.conf import settings

logger = logging.getLogger("clients.signals")


@receiver(pre_save, sender='clients.ClientUser')
def track_old_photo_changes(sender, instance, **kwargs):
    """
    Track if old_profile_photo is being set/changed.
    We need to detect this before save to compare with existing value.
    """
    if not instance.pk:
        # New instance - mark for extraction if old_profile_photo is set
        instance._old_photo_changed = bool(instance.old_profile_photo)
        return
    
    try:
        from clients.models import ClientUser
        existing = ClientUser.objects.get(pk=instance.pk)
        
        # Check if old_profile_photo changed
        old_name = existing.old_profile_photo.name if existing.old_profile_photo else ''
        new_name = instance.old_profile_photo.name if instance.old_profile_photo else ''
        
        instance._old_photo_changed = old_name != new_name
        
        # If photo changed, clear the cached embedding
        if instance._old_photo_changed:
            instance.cached_old_photo_embedding = None
            logger.info(f"Old profile photo changed for user {instance.external_user_id}, clearing cached embedding")
            
    except sender.DoesNotExist:
        instance._old_photo_changed = bool(instance.old_profile_photo)


@receiver(post_save, sender='clients.ClientUser')
def trigger_old_photo_extraction(sender, instance, created, **kwargs):
    """
    Trigger background extraction of old_profile_photo embedding when:
    1. A new user is created with an old_profile_photo
    2. An existing user's old_profile_photo is updated
    
    The extraction runs as a Celery task to avoid blocking the API.
    """
    # Check if we should trigger extraction
    should_extract = getattr(instance, '_old_photo_changed', False)
    
    if not should_extract:
        return
    
    if not instance.old_profile_photo:
        return
    
    # Don't trigger if running in test mode or Celery is disabled
    if getattr(settings, 'TESTING', False):
        logger.debug(f"Skipping background extraction in test mode for user {instance.external_user_id}")
        return
    
    try:
        from core.tasks import extract_old_photo_embedding_task
        
        # Trigger async extraction
        extract_old_photo_embedding_task.delay(str(instance.id))
        
        logger.info(f"📤 Triggered background old photo embedding extraction for user {instance.external_user_id}")
        
    except Exception as e:
        logger.warning(f"Failed to trigger background extraction for user {instance.external_user_id}: {e}")
        # Don't fail the save operation - extraction can be retried later


@receiver(post_save, sender='clients.ClientUser')
def invalidate_embedding_cache_on_reenrollment(sender, instance, created, **kwargs):
    """
    When a user re-enrolls (is_enrolled changes from False to True),
    ensure the embedding cache is updated.
    """
    if created:
        return
    
    # Check if this is a re-enrollment (user was already enrolled before)
    if not instance.is_enrolled:
        return
    
    # Get update_fields from kwargs if available
    update_fields = kwargs.get('update_fields')
    
    # If is_enrolled was explicitly set, trigger cache refresh
    if update_fields and 'is_enrolled' in update_fields:
        try:
            from core.tasks import cache_user_embedding
            
            # Trigger async cache update
            cache_user_embedding.delay(
                instance.client.client_id,
                instance.external_user_id
            )
            
            logger.info(f"📤 Triggered embedding cache update for re-enrolled user {instance.external_user_id}")
            
        except Exception as e:
            logger.warning(f"Failed to trigger cache update for user {instance.external_user_id}: {e}")

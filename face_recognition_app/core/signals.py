"""
Core signals for face recognition system
"""
from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.contrib.auth import get_user_model
from django.utils import timezone
from recognition.models import EnrollmentSession
from analytics.models import AuthenticationLog

User = get_user_model()


@receiver(post_save, sender=User)
def user_created_handler(sender, instance, created, **kwargs):
    """Handle user creation"""
    if created:
        # Log user registration
        AuthenticationLog.objects.create(
            user=instance,
            attempted_email=instance.email,
            auth_method='password',
            success=True,
            ip_address='127.0.0.1',  # Default, should be updated with actual IP
            user_agent='System',
            timestamp=timezone.now()
        )


@receiver(post_save, sender=EnrollmentSession)
def enrollment_session_handler(sender, instance, created, **kwargs):
    """Handle enrollment session updates"""
    if not created and instance.status == 'completed':
        # Update user face enrollment status
        if instance.user:
            instance.user.face_enrolled = True
            instance.user.enrollment_completed_at = timezone.now()
            instance.user.save(update_fields=['face_enrolled', 'enrollment_completed_at'])


@receiver(pre_delete, sender=User)
def user_deletion_handler(sender, instance, **kwargs):
    """Handle user deletion - cleanup related data"""
    # Log user deletion
    AuthenticationLog.objects.create(
        attempted_email=instance.email,
        auth_method='system',
        success=True,
        ip_address='127.0.0.1',
        user_agent='System - User Deletion',
        timestamp=timezone.now()
    )
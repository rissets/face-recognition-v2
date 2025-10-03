"""
User signals for face recognition system
"""
from django.db.models.signals import post_save, pre_save, post_delete
from django.dispatch import receiver
from django.contrib.auth import get_user_model
from django.utils import timezone
from .models import UserProfile, UserDevice

User = get_user_model()


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Create UserProfile when a new user is created"""
    if created:
        UserProfile.objects.create(
            user=instance,
            language='en',
            timezone='UTC',
            email_notifications=True,
            security_alerts=True
        )


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Save UserProfile when user is saved"""
    if hasattr(instance, 'profile'):
        instance.profile.save()


@receiver(pre_save, sender=User)
def user_pre_save_handler(sender, instance, **kwargs):
    """Handle user updates before saving"""
    if instance.pk:  # Existing user
        try:
            old_instance = User.objects.get(pk=instance.pk)
            # Check if face enrollment status changed
            if old_instance.face_enrolled != instance.face_enrolled:
                if instance.face_enrolled:
                    instance.enrollment_completed_at = timezone.now()
                else:
                    instance.enrollment_completed_at = None
        except User.DoesNotExist:
            pass


@receiver(post_delete, sender=User)
def user_post_delete_handler(sender, instance, **kwargs):
    """Handle cleanup after user deletion"""
    # Clean up any orphaned UserDevices (though they should cascade delete)
    UserDevice.objects.filter(user=instance).delete()
    
    # Log the deletion (could be extended to create audit log)
    pass


@receiver(post_save, sender=UserDevice)
def user_device_created(sender, instance, created, **kwargs):
    """Handle new device registration"""
    if created:
        # Could trigger security alerts for new device
        # Could send notification to user about new device
        pass
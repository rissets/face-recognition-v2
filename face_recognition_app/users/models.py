"""
Custom User Model for Face Recognition System
"""
import uuid
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.validators import RegexValidator
from encrypted_model_fields.fields import EncryptedTextField
from PIL import Image
import os


class CustomUser(AbstractUser):
    """Extended user model with face recognition capabilities"""
    
    # Override id to use UUID
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Enhanced user information
    email = models.EmailField(unique=True)
    phone_regex = RegexValidator(
        regex=r'^\+?1?\d{9,15}$',
        message="Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed."
    )
    phone_number = models.CharField(
        validators=[phone_regex], 
        max_length=17, 
        blank=True,
        help_text="Phone number in international format"
    )
    
    # Profile information
    date_of_birth = models.DateField(null=True, blank=True)
    profile_picture = models.ImageField(
        upload_to='profiles/', 
        null=True, 
        blank=True,
        help_text="Profile picture for identification"
    )
    bio = models.TextField(max_length=500, blank=True)
    
    # Face recognition status
    face_enrolled = models.BooleanField(
        default=False,
        help_text="Whether user has completed face enrollment"
    )
    enrollment_completed_at = models.DateTimeField(null=True, blank=True)
    
    # Security settings
    two_factor_enabled = models.BooleanField(default=False)
    face_auth_enabled = models.BooleanField(
        default=True,
        help_text="Enable face authentication for this user"
    )
    
    # Account status
    is_verified = models.BooleanField(
        default=False,
        help_text="Email verification status"
    )
    verification_token = models.CharField(max_length=255, blank=True)
    
    # Privacy settings
    allow_analytics = models.BooleanField(
        default=True,
        help_text="Allow collection of analytics data"
    )
    
    # Timestamps
    last_face_auth = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="Last successful face authentication"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    timestamp = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'first_name', 'last_name']

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"
        indexes = [
            models.Index(fields=['email']),
            models.Index(fields=['face_enrolled']),
            models.Index(fields=['is_active', 'face_enrolled']),
        ]

    def __str__(self):
        return f"{self.get_full_name()} ({self.email})"

    def get_full_name(self):
        """Return the first_name plus the last_name, with a space in between."""
        full_name = f"{self.first_name} {self.last_name}"
        return full_name.strip() or self.email

    def get_short_name(self):
        """Return the short name for the user."""
        return self.first_name or self.email.split('@')[0]

    def save(self, *args, **kwargs):
        """Override save to handle profile picture processing"""
        super().save(*args, **kwargs)
        
        # Process profile picture if it exists
        if self.profile_picture:
            self._process_profile_picture()

    def _process_profile_picture(self):
        """Process and optimize profile picture"""
        try:
            img = Image.open(self.profile_picture.path)
            
            # Resize if too large
            if img.height > 300 or img.width > 300:
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                img.save(self.profile_picture.path, optimize=True, quality=85)
                
        except Exception as e:
            # Log error but don't fail the save
            import logging
            logger = logging.getLogger('users')
            logger.error(f"Error processing profile picture for user {self.id}: {e}")

    @property
    def has_face_data(self):
        """Check if user has face embedding data"""
        return hasattr(self, 'face_embeddings') and self.face_embeddings.exists()

    @property
    def enrollment_progress(self):
        """Get enrollment progress percentage"""
        if not hasattr(self, 'face_embeddings'):
            return 0
        
        total_required = 5  # Minimum samples needed
        current_count = self.face_embeddings.filter(is_active=True).count()
        return min(100, (current_count / total_required) * 100)

    def can_authenticate_with_face(self):
        """Check if user can use face authentication"""
        return (
            self.is_active and 
            self.face_auth_enabled and 
            self.face_enrolled and 
            self.has_face_data
        )

    def get_recent_authentications(self, days=30):
        """Get recent authentication attempts"""
        from datetime import datetime, timedelta
        from analytics.models import AuthenticationLog
        
        since = datetime.now() - timedelta(days=days)
        return AuthenticationLog.objects.filter(
            user=self,
            created_at__gte=since
        ).order_by('-created_at')


class UserProfile(models.Model):
    """Extended profile information"""
    user = models.OneToOneField(
        CustomUser, 
        on_delete=models.CASCADE, 
        related_name='profile'
    )
    
    # Additional profile fields
    company = models.CharField(max_length=100, blank=True)
    position = models.CharField(max_length=100, blank=True)
    address = models.TextField(blank=True)
    city = models.CharField(max_length=50, blank=True)
    country = models.CharField(max_length=50, blank=True)
    
    # Preferences
    language = models.CharField(
        max_length=10, 
        default='en',
        choices=[
            ('en', 'English'),
            ('id', 'Indonesian'),
            ('es', 'Spanish'),
            ('fr', 'French'),
        ]
    )
    timezone = models.CharField(max_length=50, default='UTC')
    
    # Notification preferences
    email_notifications = models.BooleanField(default=True)
    security_alerts = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"

    def __str__(self):
        return f"{self.user.get_full_name()}'s Profile"


class UserDevice(models.Model):
    """Track user devices for security"""
    user = models.ForeignKey(
        CustomUser, 
        on_delete=models.CASCADE, 
        related_name='devices'
    )
    
    device_id = models.CharField(max_length=255, unique=True)
    device_name = models.CharField(max_length=100)
    device_type = models.CharField(
        max_length=20,
        choices=[
            ('web', 'Web Browser'),
            ('mobile', 'Mobile App'),
            ('desktop', 'Desktop App'),
        ]
    )
    
    # Device information
    operating_system = models.CharField(max_length=50, blank=True)
    browser = models.CharField(max_length=250, blank=True)
    user_agent = models.TextField(blank=True)
    
    # Security
    is_trusted = models.BooleanField(default=False)
    last_ip = models.GenericIPAddressField(null=True, blank=True)
    last_location = models.CharField(max_length=100, blank=True)
    
    # Activity
    first_seen = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)
    login_count = models.PositiveIntegerField(default=0)
    
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "User Device"
        verbose_name_plural = "User Devices"
        unique_together = ['user', 'device_id']
        indexes = [
            models.Index(fields=['user', 'is_active']),
            models.Index(fields=['device_id']),
        ]

    def __str__(self):
        return f"{self.user.get_full_name()}'s {self.device_name}"
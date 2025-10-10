"""
Custom User Model for Face Recognition System - Admin Only
This model is used for Django admin authentication only.
Client users are handled by the ClientUser model in the clients app.
"""
import uuid
from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    """Extended user model for admin authentication"""
    
    # Use UUID for primary key
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Email as primary identifier
    email = models.EmailField(unique=True)
    
    # Additional fields that might exist in the table
    phone_number = models.CharField(max_length=20, blank=True)
    profile_picture = models.ImageField(upload_to='admin_profiles/', blank=True, null=True)
    
    # Face recognition fields (for admin)
    face_enrolled = models.BooleanField(default=False)
    enrollment_completed_at = models.DateTimeField(null=True, blank=True)
    
    # Security settings
    two_factor_enabled = models.BooleanField(default=False)
    face_auth_enabled = models.BooleanField(default=True)
    
    # Account status
    is_verified = models.BooleanField(default=False)
    verification_token = models.CharField(max_length=255, blank=True)
    
    # Privacy settings
    allow_analytics = models.BooleanField(default=True)
    
    # Timestamps
    last_face_auth = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    timestamp = models.DateTimeField(auto_now=True)
    
    # Additional fields for date_of_birth and bio if they exist
    date_of_birth = models.DateField(null=True, blank=True)
    bio = models.TextField(blank=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'first_name', 'last_name']

    class Meta:
        verbose_name = "Admin User"
        verbose_name_plural = "Admin Users"
        db_table = 'users_customuser'  # Use existing table

    def __str__(self):
        return f"{self.get_full_name()} ({self.email})"

    def get_full_name(self):
        """Return the first_name plus the last_name, with a space in between."""
        full_name = f"{self.first_name} {self.last_name}"
        return full_name.strip() or self.email

    def get_short_name(self):
        """Return the short name for the user."""
        return self.first_name or self.email.split('@')[0]


class UserProfile(models.Model):
    """Extended profile information for admin users"""
    
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='profile')
    
    # Professional Info
    company = models.CharField(max_length=100, blank=True)
    position = models.CharField(max_length=100, blank=True)
    
    # Contact Info  
    address = models.TextField(blank=True)
    city = models.CharField(max_length=50, blank=True)
    country = models.CharField(max_length=50, blank=True)
    
    # Preferences
    language = models.CharField(max_length=10, default='en')
    timezone = models.CharField(max_length=50, default='UTC')
    
    # Notification Settings
    email_notifications = models.BooleanField(default=True)
    security_alerts = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"
        db_table = 'users_userprofile'  # Use existing table

    def __str__(self):
        return f"{self.user.get_full_name()}'s Profile"


class UserDevice(models.Model):
    """Track admin user devices for security"""
    
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='devices')
    
    # Device identification
    device_id = models.CharField(max_length=255, unique=True)
    device_name = models.CharField(max_length=100, blank=True)
    device_type = models.CharField(max_length=50, blank=True)  # desktop, mobile, tablet
    
    # Browser/App info
    operating_system = models.CharField(max_length=100, blank=True)
    browser = models.CharField(max_length=250, blank=True)
    user_agent = models.TextField(blank=True)
    
    # Security
    is_trusted = models.BooleanField(default=False)
    
    # Network info
    last_ip = models.GenericIPAddressField(blank=True, null=True)
    last_location = models.CharField(max_length=100, blank=True)
    
    # Usage tracking
    first_seen = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)
    login_count = models.PositiveIntegerField(default=0)
    
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "User Device"
        verbose_name_plural = "User Devices"
        db_table = 'users_userdevice'  # Use existing table
        unique_together = ['user', 'device_id']

    def __str__(self):
        return f"{self.user.get_full_name()}'s {self.device_name}"
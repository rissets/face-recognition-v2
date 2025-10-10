"""
Core models for face recognition system
"""
import uuid
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth import get_user_model
from encrypted_model_fields.fields import EncryptedTextField
import json


class BaseModel(models.Model):
    """Base model with common fields"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        abstract = True


class SystemConfiguration(BaseModel):
    """System-wide configuration"""
    key = models.CharField(max_length=100, unique=True)
    value = EncryptedTextField()
    description = models.TextField(blank=True)
    is_encrypted = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = "System Configuration"
        verbose_name_plural = "System Configurations"
        
    def __str__(self):
        return f"{self.key}: {self.description[:50]}"
    
    def set_value(self, value):
        """Set configuration value (handles JSON serialization)"""
        if isinstance(value, (dict, list)):
            self.value = json.dumps(value)
        else:
            self.value = str(value)
    
    def get_value(self):
        """Get configuration value (handles JSON deserialization)"""
        try:
            return json.loads(self.value)
        except (json.JSONDecodeError, TypeError):
            return self.value


class AuditLog(BaseModel):
    """Audit log for tracking system activities"""
    # User field migrated to use client system
    client = models.ForeignKey(
        'clients.Client',
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.CASCADE, 
        null=True, 
        blank=True
    )
    action = models.CharField(max_length=100)
    resource_type = models.CharField(max_length=50)
    resource_id = models.CharField(max_length=100, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    details = models.JSONField(default=dict)
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True)

    class Meta:
        verbose_name = "Audit Log"
        verbose_name_plural = "Audit Logs"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['client_user', 'created_at']),
            models.Index(fields=['client', 'created_at']),
            models.Index(fields=['action', 'created_at']),
            models.Index(fields=['resource_type', 'created_at']),
        ]

    def __str__(self):
        user_info = self.client_user.username if self.client_user else f"Client: {self.client.name}" if self.client else "Unknown"
        return f"{self.action} by {user_info} at {self.created_at}"


class SecurityEvent(BaseModel):
    """Security events and incidents"""
    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    EVENT_TYPES = [
        ('login_attempt', 'Login Attempt'),
        ('failed_authentication', 'Failed Authentication'),
        ('multiple_faces_detected', 'Multiple Faces Detected'),
        ('liveness_check_failed', 'Liveness Check Failed'),
        ('obstacle_detected', 'Obstacle Detected'),
        ('spoofing_attempt', 'Spoofing Attempt'),
        ('unusual_activity', 'Unusual Activity'),
        ('system_error', 'System Error'),
    ]
    
    event_type = models.CharField(max_length=50, choices=EVENT_TYPES)
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='medium')
    # User field migrated to use client system
    client = models.ForeignKey(
        'clients.Client',
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    details = models.JSONField(default=dict)
    resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)
    # Resolved by field migrated to use client system
    resolved_by_client = models.ForeignKey(
        'clients.Client',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='resolved_security_events'
    )
    resolved_by_client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='resolved_security_events_users'
    )

    class Meta:
        verbose_name = "Security Event"
        verbose_name_plural = "Security Events"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['event_type', 'created_at']),
            models.Index(fields=['severity', 'created_at']),
            models.Index(fields=['resolved', 'created_at']),
        ]

    def __str__(self):
        return f"{self.get_event_type_display()} - {self.get_severity_display()}"


class HealthCheck(BaseModel):
    """System health monitoring"""
    service_name = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=[
        ('healthy', 'Healthy'),
        ('warning', 'Warning'),
        ('critical', 'Critical'),
        ('down', 'Down'),
    ])
    response_time = models.FloatField(null=True, blank=True)  # in milliseconds
    error_message = models.TextField(blank=True)
    details = models.JSONField(default=dict)

    class Meta:
        verbose_name = "Health Check"
        verbose_name_plural = "Health Checks"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['service_name', 'created_at']),
            models.Index(fields=['status', 'created_at']),
        ]

    def __str__(self):
        return f"{self.service_name}: {self.status}"
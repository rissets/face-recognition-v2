"""
Authentication Service Models for Face Recognition Third-Party Service
"""
import uuid
import jwt
import secrets
from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.core.validators import MinValueValidator, MaxValueValidator
from encrypted_model_fields.fields import EncryptedTextField, EncryptedCharField
from django.utils import timezone
import json


class AuthenticationSession(models.Model):
    """
    Face authentication sessions for clients
    """
    
    SESSION_TYPE_CHOICES = [
        ('enrollment', 'Enrollment'),
        ('recognition', 'Recognition'), 
        ('verification', 'Verification'),
        ('identification', 'Identification'),
    ]
    
    SESSION_STATUS_CHOICES = [
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('expired', 'Expired'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_token = models.CharField(max_length=100, unique=True, db_index=True)
    
    # Client and user context
    client = models.ForeignKey('clients.Client', on_delete=models.CASCADE, related_name='auth_sessions')
    client_user = models.ForeignKey(
        'clients.ClientUser', 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True,
        related_name='auth_sessions'
    )
    
    # Session details
    session_type = models.CharField(max_length=20, choices=SESSION_TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=SESSION_STATUS_CHOICES, default='active')
    
    # Configuration
    liveness_required = models.BooleanField(default=False)
    anti_spoofing_required = models.BooleanField(default=False)
    max_attempts = models.PositiveIntegerField(default=3)
    current_attempts = models.PositiveIntegerField(default=0)
    
    # Device and security info
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    device_fingerprint = models.CharField(max_length=255, blank=True)
    
    # Session results
    is_successful = models.BooleanField(default=False)
    confidence_score = models.FloatField(
        null=True, 
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    failure_reason = models.CharField(max_length=100, blank=True)
    
    # Processing data
    frames_processed = models.PositiveIntegerField(default=0)
    processing_time_ms = models.FloatField(null=True, blank=True)
    
    # Metadata
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Authentication Session"
        verbose_name_plural = "Authentication Sessions"
        indexes = [
            models.Index(fields=['client', 'session_type']),
            models.Index(fields=['session_token']),
            models.Index(fields=['status', 'expires_at']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.client.client_id} - {self.session_type} - {self.status}"
    
    def save(self, *args, **kwargs):
        if not self.session_token:
            self.session_token = self.generate_session_token()
        
        if not self.expires_at:
            # Default session expiry: 10 minutes
            from datetime import timedelta
            self.expires_at = timezone.now() + timedelta(minutes=10)
        
        super().save(*args, **kwargs)
    
    def generate_session_token(self):
        """Generate unique session token"""
        return f"sess_{secrets.token_urlsafe(32)}"
    
    @property
    def is_expired(self):
        """Check if session is expired"""
        return timezone.now() > self.expires_at
    
    @property
    def is_active(self):
        """Check if session is active and not expired"""
        return self.status == 'active' and not self.is_expired
    
    def expire_session(self):
        """Expire the session"""
        self.status = 'expired'
        self.completed_at = timezone.now()
        self.save(update_fields=['status', 'completed_at'])
    
    def complete_session(self, success=True, confidence_score=None, failure_reason=None):
        """Complete the session with result"""
        self.status = 'completed' if success else 'failed'
        self.is_successful = success
        self.confidence_score = confidence_score
        self.failure_reason = failure_reason or ''
        self.completed_at = timezone.now()
        self.save(update_fields=[
            'status', 'is_successful', 'confidence_score', 
            'failure_reason', 'completed_at'
        ])


class FaceEnrollment(models.Model):
    """
    Face enrollment records for client users
    """
    
    ENROLLMENT_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('expired', 'Expired'),
        ('completed', 'Completed'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Client and user context
    client = models.ForeignKey('clients.Client', on_delete=models.CASCADE, related_name='enrollments')
    client_user = models.ForeignKey('clients.ClientUser', on_delete=models.CASCADE, related_name='enrollments')
    
    # Enrollment details
    enrollment_session = models.ForeignKey(
        AuthenticationSession, 
        on_delete=models.CASCADE, 
        related_name='enrollments'
    )
    
    status = models.CharField(max_length=20, choices=ENROLLMENT_STATUS_CHOICES, default='pending')
    
    # Face embedding data (encrypted)
    embedding_vector = EncryptedTextField(help_text="Encrypted face embedding vector")
    embedding_dimension = models.PositiveIntegerField(default=512)
    
    # Quality metrics
    face_quality_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Face image quality score"
    )
    liveness_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        default=0.0,
        help_text="Liveness detection score"
    )
    anti_spoofing_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        default=0.0,
        help_text="Anti-spoofing detection score"
    )
    
    # Face detection data
    face_landmarks = models.JSONField(
        null=True, 
        blank=True,
        help_text="Facial landmarks coordinates"
    )
    face_bbox = models.JSONField(
        null=True,
        blank=True, 
        help_text="Face bounding box coordinates"
    )
    
    # Sample information
    sample_number = models.PositiveIntegerField(help_text="Sample number in enrollment session")
    total_samples = models.PositiveIntegerField(default=1)
    
    # Image reference (stored in MinIO)
    face_image_path = models.CharField(max_length=500, blank=True)
    
    # Metadata
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Face Enrollment"
        verbose_name_plural = "Face Enrollments"
        unique_together = [['client_user', 'sample_number']]
        indexes = [
            models.Index(fields=['client', 'status']),
            models.Index(fields=['client_user']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.client_user} - Sample {self.sample_number}"
    
    @property
    def is_active(self):
        """Check if enrollment is active"""
        if self.expires_at and timezone.now() > self.expires_at:
            return False
        return self.status == 'active'


class FaceRecognitionAttempt(models.Model):
    """
    Face recognition attempts and results
    """
    
    RESULT_CHOICES = [
        ('success', 'Success'),
        ('failed', 'Failed'),
        ('no_match', 'No Match'),
        ('multiple_matches', 'Multiple Matches'),
        ('liveness_failed', 'Liveness Failed'),
        ('spoofing_detected', 'Spoofing Detected'),
        ('quality_too_low', 'Quality Too Low'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Session context
    client = models.ForeignKey('clients.Client', on_delete=models.CASCADE, related_name='recognition_attempts')
    session = models.ForeignKey(AuthenticationSession, on_delete=models.CASCADE, related_name='recognition_attempts')
    
    # Recognition details
    result = models.CharField(max_length=20, choices=RESULT_CHOICES)
    
    # Matched user (if successful)
    matched_user = models.ForeignKey(
        'clients.ClientUser', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='recognition_matches'
    )
    matched_enrollment = models.ForeignKey(
        FaceEnrollment,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='recognition_matches'
    )
    
    # Similarity and confidence scores
    similarity_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Quality metrics for submitted face
    face_quality_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    liveness_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        default=0.0
    )
    anti_spoofing_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        default=0.0
    )
    
    # Face data
    submitted_embedding = EncryptedTextField(help_text="Encrypted submitted face embedding")
    face_landmarks = models.JSONField(null=True, blank=True)
    face_bbox = models.JSONField(null=True, blank=True)
    
    # Performance metrics
    processing_time_ms = models.FloatField(null=True, blank=True)
    
    # Device and security info
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    device_fingerprint = models.CharField(max_length=255, blank=True)
    
    # Additional data
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Face Recognition Attempt"
        verbose_name_plural = "Face Recognition Attempts"
        indexes = [
            models.Index(fields=['client', 'result']),
            models.Index(fields=['session']),
            models.Index(fields=['matched_user', 'created_at']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.client.client_id} - {self.result} - {self.created_at}"
    
    @property
    def is_successful(self):
        """Check if recognition was successful"""
        return self.result == 'success'


class LivenessDetectionResult(models.Model):
    """
    Liveness detection results for enhanced security
    """
    
    LIVENESS_STATUS_CHOICES = [
        ('live', 'Live Person'),
        ('spoof', 'Spoof Detected'),
        ('uncertain', 'Uncertain'),
        ('error', 'Detection Error'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Context
    client = models.ForeignKey('clients.Client', on_delete=models.CASCADE, related_name='liveness_results')
    session = models.ForeignKey(AuthenticationSession, on_delete=models.CASCADE, related_name='liveness_results')
    
    # Liveness detection result
    status = models.CharField(max_length=20, choices=LIVENESS_STATUS_CHOICES)
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Detection methods used
    methods_used = ArrayField(
        models.CharField(max_length=50),
        default=list,
        help_text="Methods used: blink_detection, motion_detection, texture_analysis, etc."
    )
    
    # Detection metrics
    blink_detected = models.BooleanField(default=False)
    motion_detected = models.BooleanField(default=False)
    texture_score = models.FloatField(null=True, blank=True)
    
    # Face movement data
    face_movements = models.JSONField(
        default=list,
        help_text="Sequence of face movement data"
    )
    
    # Processing details
    frames_analyzed = models.PositiveIntegerField(default=1)
    processing_time_ms = models.FloatField(null=True, blank=True)
    
    # Metadata
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Liveness Detection Result"
        verbose_name_plural = "Liveness Detection Results"
        indexes = [
            models.Index(fields=['client', 'status']),
            models.Index(fields=['session']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.client.client_id} - {self.status} - {self.confidence_score}"
    
    @property
    def is_live(self):
        """Check if result indicates live person"""
        return self.status == 'live'


class SystemMetrics(models.Model):
    """
    System performance and usage metrics
    """
    
    METRIC_TYPE_CHOICES = [
        ('performance', 'Performance'),
        ('usage', 'Usage'),
        ('error', 'Error'),
        ('security', 'Security'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Metric details
    metric_type = models.CharField(max_length=20, choices=METRIC_TYPE_CHOICES)
    metric_name = models.CharField(max_length=100, db_index=True)
    metric_value = models.FloatField()
    
    # Context
    client = models.ForeignKey(
        'clients.Client', 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True,
        related_name='metrics'
    )
    
    # Additional data
    dimensions = models.JSONField(default=dict, help_text="Metric dimensions and tags")
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        verbose_name = "System Metric"
        verbose_name_plural = "System Metrics"
        indexes = [
            models.Index(fields=['metric_type', 'metric_name']),
            models.Index(fields=['client', 'timestamp']),
            models.Index(fields=['timestamp']),
        ]
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.metric_name}: {self.metric_value}"

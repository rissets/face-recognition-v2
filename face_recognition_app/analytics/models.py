"""
Analytics Models for Face Recognition System
"""
import uuid
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta
import json

User = get_user_model()


class AuthenticationLog(models.Model):
    """Comprehensive authentication logging"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client = models.ForeignKey(
        'clients.Client',
        on_delete=models.CASCADE,
        related_name='authentication_logs',
        null=True,
        blank=True,
        help_text="Client tenant associated with the authentication attempt",
    )
    
    # User information
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='auth_logs',
        null=True,
        blank=True
    )
    attempted_email = models.EmailField(blank=True, null=True)  # For failed attempts
    
    # Authentication details
    auth_method = models.CharField(
        max_length=20,
        choices=[
            ('face', 'Face Recognition'),
            ('password', 'Password'),
            ('2fa', 'Two Factor'),
            ('social', 'Social Login'),
        ],
        default='face'
    )
    
    # Result
    success = models.BooleanField(default=False)
    failure_reason = models.CharField(
        max_length=50,
        blank=True,
        choices=[
            ('invalid_credentials', 'Invalid Credentials'),
            ('face_not_recognized', 'Face Not Recognized'),
            ('liveness_failed', 'Liveness Check Failed'),
            ('quality_too_low', 'Image Quality Too Low'),
            ('multiple_faces', 'Multiple Faces Detected'),
            ('no_face_detected', 'No Face Detected'),
            ('obstacles_detected', 'Obstacles Detected'),
            ('account_disabled', 'Account Disabled'),
            ('rate_limited', 'Rate Limited'),
            ('system_error', 'System Error'),
        ]
    )
    
    # Context information
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    device_fingerprint = models.CharField(max_length=255, blank=True)
    location = models.CharField(max_length=100, blank=True)  # City, Country
    
    # Performance metrics
    response_time = models.FloatField(null=True, blank=True)  # in milliseconds
    
    # Face recognition specific data
    similarity_score = models.FloatField(null=True, blank=True)
    liveness_score = models.FloatField(null=True, blank=True)
    quality_score = models.FloatField(null=True, blank=True)
    
    # Risk assessment
    risk_score = models.FloatField(
        default=0.0,
        help_text="Calculated risk score for this authentication attempt"
    )
    risk_factors = models.JSONField(
        default=list,
        help_text="List of risk factors identified"
    )
    
    # Session information
    session_id = models.CharField(max_length=255, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Authentication Log"
        verbose_name_plural = "Authentication Logs"
        indexes = [
            models.Index(fields=['client', 'created_at']),
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['success', 'created_at']),
            models.Index(fields=['auth_method', 'created_at']),
            models.Index(fields=['ip_address', 'created_at']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        user_info = self.user.get_full_name() if self.user else self.attempted_email
        status = "Success" if self.success else f"Failed ({self.failure_reason})"
        return f"{user_info} - {status}"


class SystemMetrics(models.Model):
    """System-wide performance metrics"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client = models.ForeignKey(
        'clients.Client',
        on_delete=models.CASCADE,
        related_name='system_metrics',
        null=True,
        blank=True,
        help_text="Client tenant for the recorded metric",
    )
    
    # Metric information
    metric_name = models.CharField(max_length=100)
    metric_type = models.CharField(
        max_length=20,
        choices=[
            ('counter', 'Counter'),
            ('gauge', 'Gauge'),
            ('histogram', 'Histogram'),
            ('timer', 'Timer'),
        ]
    )
    
    # Values
    value = models.FloatField()
    unit = models.CharField(max_length=20, blank=True)  # ms, %, count, etc.
    
    # Context
    tags = models.JSONField(default=dict)  # Additional metadata
    
    # Time series data
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "System Metric"
        verbose_name_plural = "System Metrics"
        indexes = [
            models.Index(fields=['client', 'timestamp']),
            models.Index(fields=['metric_name', 'timestamp']),
            models.Index(fields=['metric_type', 'timestamp']),
        ]
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.metric_name}: {self.value} {self.unit}"


class UserBehaviorAnalytics(models.Model):
    """Track user behavior patterns"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client = models.ForeignKey(
        'clients.Client',
        on_delete=models.CASCADE,
        related_name='behavior_analytics',
        null=True,
        blank=True,
        help_text="Client tenant associated with this behavioral profile",
    )
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='behavior_analytics'
    )
    
    # Behavioral metrics
    avg_login_time = models.TimeField(null=True, blank=True)
    common_locations = models.JSONField(default=list)
    device_preferences = models.JSONField(default=dict)
    
    # Authentication patterns
    auth_success_rate = models.FloatField(default=0.0)
    avg_similarity_score = models.FloatField(default=0.0)
    avg_liveness_score = models.FloatField(default=0.0)
    
    # Activity patterns
    login_frequency = models.FloatField(default=0.0)  # logins per day
    peak_activity_hours = models.JSONField(default=list)
    
    # Risk indicators
    suspicious_activity_count = models.PositiveIntegerField(default=0)
    last_risk_assessment = models.DateTimeField(null=True, blank=True)
    risk_level = models.CharField(
        max_length=10,
        choices=[
            ('low', 'Low'),
            ('medium', 'Medium'),
            ('high', 'High'),
            ('critical', 'Critical'),
        ],
        default='low'
    )
    
    # Analysis period
    analysis_start = models.DateTimeField()
    analysis_end = models.DateTimeField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "User Behavior Analytics"
        verbose_name_plural = "User Behavior Analytics"
        unique_together = ['user', 'analysis_start', 'analysis_end']

    def __str__(self):
        return f"Behavior analysis for {self.user.get_full_name()}"


class SecurityAlert(models.Model):
    """Security alerts and notifications"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    client = models.ForeignKey(
        'clients.Client',
        on_delete=models.CASCADE,
        related_name='security_alerts',
        null=True,
        blank=True,
        help_text="Client tenant that owns this alert",
    )
    
    # Alert information
    alert_type = models.CharField(
        max_length=30,
        choices=[
            ('failed_attempts', 'Multiple Failed Attempts'),
            ('new_device', 'New Device Login'),
            ('location_anomaly', 'Unusual Location'),
            ('time_anomaly', 'Unusual Time'),
            ('quality_degradation', 'Quality Degradation'),
            ('system_breach', 'System Breach Attempt'),
            ('data_anomaly', 'Data Anomaly'),
        ]
    )
    
    severity = models.CharField(
        max_length=10,
        choices=[
            ('info', 'Info'),
            ('low', 'Low'),
            ('medium', 'Medium'),
            ('high', 'High'),
            ('critical', 'Critical'),
        ],
        default='medium'
    )
    
    # Affected entities
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True,
        related_name='security_alerts'
    )
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    # Alert details
    title = models.CharField(max_length=200)
    description = models.TextField()
    context_data = models.JSONField(default=dict)
    
    # Status
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='acknowledged_alerts'
    )
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    resolved = models.BooleanField(default=False)
    resolved_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='resolved_alerts'
    )
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolution_notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Security Alert"
        verbose_name_plural = "Security Alerts"
        indexes = [
            models.Index(fields=['client', 'created_at']),
            models.Index(fields=['severity', 'created_at']),
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['acknowledged', 'resolved']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.get_severity_display()}: {self.title}"


class FaceRecognitionStats(models.Model):
    """Aggregated face recognition statistics"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client = models.ForeignKey(
        'clients.Client',
        on_delete=models.CASCADE,
        related_name='recognition_stats',
        null=True,
        blank=True,
        help_text="Client tenant for aggregated recognition statistics",
    )
    
    # Time period
    date = models.DateField()
    hour = models.PositiveIntegerField(null=True, blank=True)  # For hourly stats
    
    # Authentication stats
    total_attempts = models.PositiveIntegerField(default=0)
    successful_attempts = models.PositiveIntegerField(default=0)
    failed_attempts = models.PositiveIntegerField(default=0)
    
    # Failure breakdown
    failed_similarity = models.PositiveIntegerField(default=0)
    failed_liveness = models.PositiveIntegerField(default=0)
    failed_quality = models.PositiveIntegerField(default=0)
    failed_obstacles = models.PositiveIntegerField(default=0)
    failed_no_face = models.PositiveIntegerField(default=0)
    failed_multiple_faces = models.PositiveIntegerField(default=0)
    failed_system_error = models.PositiveIntegerField(default=0)
    
    # Performance metrics
    avg_response_time = models.FloatField(null=True, blank=True)
    avg_similarity_score = models.FloatField(null=True, blank=True)
    avg_liveness_score = models.FloatField(null=True, blank=True)
    avg_quality_score = models.FloatField(null=True, blank=True)
    
    # User metrics
    unique_users = models.PositiveIntegerField(default=0)
    new_enrollments = models.PositiveIntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Face Recognition Stats"
        verbose_name_plural = "Face Recognition Stats"
        unique_together = ['client', 'date', 'hour']
        indexes = [
            models.Index(fields=['client', 'date']),
            models.Index(fields=['date']),
            models.Index(fields=['date', 'hour']),
        ]
        ordering = ['-date', '-hour']

    def __str__(self):
        period = f"{self.date}"
        if self.hour is not None:
            period += f" {self.hour:02d}:00"
        return f"Stats for {period}"

    @property
    def success_rate(self):
        """Calculate success rate percentage"""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_attempts / self.total_attempts) * 100


class ModelPerformance(models.Model):
    """Track model performance over time"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    model = models.ForeignKey(
        'recognition.FaceRecognitionModel',
        on_delete=models.CASCADE,
        related_name='performance_metrics'
    )
    
    # Performance metrics
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    
    # Specific metrics
    false_acceptance_rate = models.FloatField(null=True, blank=True)
    false_rejection_rate = models.FloatField(null=True, blank=True)
    
    # Test set information
    test_set_size = models.PositiveIntegerField()
    test_conditions = models.JSONField(default=dict)
    
    # Performance context
    environment = models.CharField(
        max_length=20,
        choices=[
            ('development', 'Development'),
            ('staging', 'Staging'),
            ('production', 'Production'),
        ]
    )
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Model Performance"
        verbose_name_plural = "Model Performance"
        indexes = [
            models.Index(fields=['model', 'created_at']),
            models.Index(fields=['environment', 'created_at']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.model.name} - Accuracy: {self.accuracy:.2%}"


class DataQualityMetrics(models.Model):
    """Track data quality over time"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Time period
    date = models.DateField()
    
    # Quality metrics
    avg_image_quality = models.FloatField()
    avg_face_size = models.FloatField()
    avg_brightness = models.FloatField()
    avg_contrast = models.FloatField()
    avg_sharpness = models.FloatField()
    
    # Quality distribution
    high_quality_samples = models.PositiveIntegerField(default=0)
    medium_quality_samples = models.PositiveIntegerField(default=0)
    low_quality_samples = models.PositiveIntegerField(default=0)
    
    # Issues detected
    blurry_images = models.PositiveIntegerField(default=0)
    over_exposed = models.PositiveIntegerField(default=0)
    under_exposed = models.PositiveIntegerField(default=0)
    obstacles_present = models.PositiveIntegerField(default=0)
    
    total_samples = models.PositiveIntegerField()
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Data Quality Metrics"
        verbose_name_plural = "Data Quality Metrics"
        unique_together = ['date']
        ordering = ['-date']

    def __str__(self):
        return f"Quality metrics for {self.date}"

    @property
    def quality_score(self):
        """Calculate overall quality score"""
        if self.total_samples == 0:
            return 0.0
        
        quality_weighted = (
            (self.high_quality_samples * 3) +
            (self.medium_quality_samples * 2) +
            (self.low_quality_samples * 1)
        )
        
        max_possible = self.total_samples * 3
        return (quality_weighted / max_possible) * 100

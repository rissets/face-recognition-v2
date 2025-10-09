"""
Streaming Models for WebRTC Sessions
"""
import uuid
import secrets
from django.db import models
from django.conf import settings


def generate_session_token():
    """Generate a unique session token"""
    return f"session_{secrets.token_urlsafe(32)}"


class StreamingSession(models.Model):
    """WebRTC streaming sessions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Session info
    session_token = models.CharField(max_length=255, unique=True, default=generate_session_token)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='streaming_sessions',
        null=True,
        blank=True
    )
    
    # Session type
    session_type = models.CharField(
        max_length=20,
        choices=[
            ('enrollment', 'Enrollment'),
            ('authentication', 'Authentication'),
            ('verification', 'Verification'),
        ]
    )
    
    # WebRTC configuration
    ice_servers = models.JSONField(default=list)
    constraints = models.JSONField(default=dict)
    
    # Connection details
    peer_connection_id = models.CharField(max_length=255, blank=True)
    remote_address = models.GenericIPAddressField(null=True, blank=True)
    
    # Status
    STATUS_CHOICES = [
        ('initiating', 'Initiating'),
        ('connecting', 'Connecting'),
        ('connected', 'Connected'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('disconnected', 'Disconnected'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='initiating')
    
    # Quality metrics
    video_quality = models.CharField(max_length=10, blank=True)  # e.g., "720p"
    frame_rate = models.PositiveIntegerField(null=True, blank=True)
    bitrate = models.PositiveIntegerField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    connected_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Session data
    session_data = models.JSONField(default=dict)
    error_log = models.TextField(blank=True)

    class Meta:
        verbose_name = "Streaming Session"
        verbose_name_plural = "Streaming Sessions"
        indexes = [
            models.Index(fields=['session_token']),
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['status']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        user_info = self.user.get_full_name() if self.user else 'Anonymous'
        return f"{self.session_type} session for {user_info}"


class WebRTCSignal(models.Model):
    """Store WebRTC signaling messages"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    session = models.ForeignKey(
        StreamingSession,
        on_delete=models.CASCADE,
        related_name='signals'
    )
    
    # Signal info
    signal_type = models.CharField(
        max_length=20,
        choices=[
            ('offer', 'Offer'),
            ('answer', 'Answer'),
            ('ice_candidate', 'ICE Candidate'),
            ('ice_gathering', 'ICE Gathering Complete'),
        ]
    )
    
    # Signal data
    signal_data = models.JSONField()
    
    # Direction
    direction = models.CharField(
        max_length=10,
        choices=[
            ('outbound', 'Outbound'),
            ('inbound', 'Inbound'),
        ]
    )
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "WebRTC Signal"
        verbose_name_plural = "WebRTC Signals"
        ordering = ['created_at']

    def __str__(self):
        return f"{self.signal_type} - {self.direction}"
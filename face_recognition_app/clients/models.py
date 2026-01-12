"""
Client Management Models for Face Recognition Third-Party Service
"""
import uuid
import secrets
from django.db import models
from django.core.validators import URLValidator
from django.contrib.postgres.fields import ArrayField
from encrypted_model_fields.fields import EncryptedTextField, EncryptedCharField
from django.conf import settings
import json


def get_media_storage():
    """Get the appropriate storage backend for media files"""
    if settings.USE_MINIO:
        from core.storage import MinIOMediaStorage
        return MinIOMediaStorage()
    return None  # Use default storage


class Client(models.Model):
    """
    Client applications that use face recognition service
    Each client has their own isolated data space
    """
    
    CLIENT_STATUS_CHOICES = [
        ('active', 'Active'),
        ('suspended', 'Suspended'),
        ('inactive', 'Inactive'),
        ('trial', 'Trial'),
    ]
    
    CLIENT_TIER_CHOICES = [
        ('basic', 'Basic'),
        ('premium', 'Premium'), 
        ('enterprise', 'Enterprise'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client_id = models.CharField(max_length=50, unique=True, db_index=True)
    
    # Client information
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    domain = models.URLField(help_text="Primary domain for CORS and webhook validation")
    
    # Authentication credentials
    api_key = EncryptedCharField(max_length=100, unique=True, db_index=True)
    secret_key = EncryptedTextField()
    
    # Client status and tier
    status = models.CharField(max_length=20, choices=CLIENT_STATUS_CHOICES, default='trial')
    tier = models.CharField(max_length=20, choices=CLIENT_TIER_CHOICES, default='basic')
    
    # Service configuration
    webhook_url = models.URLField(null=True, blank=True, validators=[URLValidator()])
    webhook_secret = EncryptedCharField(max_length=100, null=True, blank=True)
    allowed_domains = ArrayField(
        models.CharField(max_length=100),
        default=list,
        help_text="Allowed domains for CORS"
    )
    
    # Rate limiting
    rate_limit_per_hour = models.PositiveIntegerField(default=1000)
    rate_limit_per_day = models.PositiveIntegerField(default=10000)
    
    # Feature flags - what services this client can access
    features = models.JSONField(
        default=dict,
        help_text="Feature configuration: enrollment, recognition, liveness, etc."
    )
    
    # Metadata
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_activity = models.DateTimeField(null=True, blank=True)
    
    # Contact information
    contact_email = models.EmailField()
    contact_name = models.CharField(max_length=100)
    
    class Meta:
        verbose_name = "Client"
        verbose_name_plural = "Clients"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['client_id']),
            models.Index(fields=['api_key']),
            models.Index(fields=['status']),
            models.Index(fields=['tier']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.client_id})"
    
    def save(self, *args, **kwargs):
        if not self.client_id:
            self.client_id = self.generate_client_id()
        if not self.api_key:
            self.api_key = self.generate_api_key()
        if not self.secret_key:
            self.secret_key = self.generate_secret_key()
        if not self.webhook_secret and self.webhook_url:
            self.webhook_secret = self.generate_webhook_secret()
        
        # Set default features based on tier
        if not self.features:
            self.features = self.get_default_features()
            
        super().save(*args, **kwargs)
    
    def generate_client_id(self):
        """Generate unique client ID"""
        prefix = "FR"  # Face Recognition
        return f"{prefix}_{secrets.token_hex(8).upper()}"
    
    def generate_api_key(self):
        """Generate API key"""
        return f"frapi_{secrets.token_urlsafe(32)}"
    
    def generate_secret_key(self):
        """Generate secret key for JWT signing"""
        return secrets.token_urlsafe(64)
    
    def generate_webhook_secret(self):
        """Generate webhook secret for signature verification"""
        return f"whsec_{secrets.token_urlsafe(32)}"
    
    def get_default_features(self):
        """Get default features based on client tier"""
        features = {
            'enrollment': True,
            'recognition': True,
            'liveness_detection': False,
            'anti_spoofing': False,
            'batch_processing': False,
            'analytics': False,
            'webhook_events': ['enrollment.completed', 'recognition.success', 'recognition.failed'],
            'max_users_per_client': 1000,
            'max_embeddings_per_user': 5,
        }
        
        if self.tier == 'premium':
            features.update({
                'liveness_detection': True,
                'analytics': True,
                'max_users_per_client': 10000,
            })
        elif self.tier == 'enterprise':
            features.update({
                'liveness_detection': True,
                'anti_spoofing': True,
                'batch_processing': True,
                'analytics': True,
                'webhook_events': [
                    'enrollment.started', 'enrollment.completed', 'enrollment.failed',
                    'recognition.success', 'recognition.failed', 'recognition.attempt',
                    'liveness.failed', 'spoofing.detected'
                ],
                'max_users_per_client': 100000,
                'max_embeddings_per_user': 10,
            })
        
        return features
    
    def is_feature_enabled(self, feature_name):
        """Check if a feature is enabled for this client"""
        return self.features.get(feature_name, False)
    
    def get_webhook_events(self):
        """Get enabled webhook events"""
        return self.features.get('webhook_events', [])
    
    @property
    def is_active(self):
        """Check if client is active"""
        return self.status == 'active'
    
    def update_last_activity(self):
        """Update last activity timestamp"""
        from django.utils import timezone
        self.last_activity = timezone.now()
        self.save(update_fields=['last_activity'])

    def regenerate_api_key(self):
        """Rotate API key and persist the change"""
        self.api_key = self.generate_api_key()
        self.save(update_fields=['api_key'])
        return self.api_key

    def set_api_secret(self, secret):
        """Persist a new secret key for token signing"""
        if not secret:
            raise ValueError("Secret cannot be empty")
        self.secret_key = secret
        self.save(update_fields=['secret_key'])

    def check_api_secret(self, candidate):
        """Constant-time comparison between stored secret and candidate"""
        if not candidate or not self.secret_key:
            return False
        return secrets.compare_digest(str(self.secret_key), str(candidate))

    def rotate_webhook_secret(self):
        """Generate and persist a new webhook secret"""
        self.webhook_secret = self.generate_webhook_secret()
        self.save(update_fields=['webhook_secret'])
        return self.webhook_secret

    @classmethod
    def find_active_by_api_key(cls, api_key):
        """Perform application-level lookup for encrypted API keys."""
        if not api_key:
            return None
        for candidate in cls.objects.filter(status='active'):
            if candidate.api_key == api_key:
                return candidate
        return None


class ClientUser(models.Model):
    """
    Dynamic user model for each client
    Each client has their own user space
    """
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client = models.ForeignKey(Client, on_delete=models.CASCADE, related_name='users')
    
    # User identification from client system
    external_user_id = models.CharField(max_length=100, db_index=True)
    external_user_uuid = models.UUIDField(null=True, blank=True, db_index=True)
    
    # User profile data (flexible JSON)
    profile = models.JSONField(default=dict, help_text="User profile from client system")
    
    # Face recognition status
    is_enrolled = models.BooleanField(default=False)
    enrollment_completed_at = models.DateTimeField(null=True, blank=True)
    
    # Face auth settings
    face_auth_enabled = models.BooleanField(default=True)
    
    # Profile image (stored in MinIO/S3)
    profile_image = models.ImageField(
        upload_to='client_users/profiles/%Y/%m/%d/',
        storage=get_media_storage,
        null=True,
        blank=True,
        help_text="User profile image from enrollment"
    )
    
    # Old profile photo for comparison
    old_profile_photo = models.ImageField(
        upload_to='client_users/old_profiles/%Y/%m/%d/',
        storage=get_media_storage,
        null=True,
        blank=True,
        help_text="Previous profile photo for similarity comparison"
    )
    
    # Similarity score with old photo
    similarity_with_old_photo = models.FloatField(
        null=True,
        blank=True,
        help_text="Similarity score between current and old profile photo (0.0 - 1.0)"
    )
    
    # OPTIMIZATION: Cached embeddings for fast authentication
    # Store embeddings as binary to avoid ChromaDB query overhead
    cached_embedding = models.BinaryField(
        null=True,
        blank=True,
        help_text="Cached face embedding (512-dim float32 array as bytes)"
    )
    
    cached_old_photo_embedding = models.BinaryField(
        null=True,
        blank=True,
        help_text="Cached embedding from old profile photo"
    )
    
    embedding_cached_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the embedding was last cached"
    )
    
    # Metadata
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_recognition_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Client User"
        verbose_name_plural = "Client Users"
        unique_together = [['client', 'external_user_id']]
        indexes = [
            models.Index(fields=['client', 'external_user_id']),
            models.Index(fields=['external_user_uuid']),
            models.Index(fields=['is_enrolled']),
            models.Index(fields=['embedding_cached_at']),  # Index for cache invalidation queries
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.client.name} - {self.external_user_id}"
    
    def cache_embedding(self, embedding):
        """Cache a face embedding for fast retrieval"""
        import numpy as np
        from django.utils import timezone
        
        if embedding is None:
            return
        
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        else:
            embedding = embedding.astype(np.float32)
        
        self.cached_embedding = embedding.tobytes()
        self.embedding_cached_at = timezone.now()
        self.save(update_fields=['cached_embedding', 'embedding_cached_at'])
    
    def get_cached_embedding(self):
        """Get cached embedding as numpy array"""
        import numpy as np
        
        if not self.cached_embedding:
            return None
        
        return np.frombuffer(self.cached_embedding, dtype=np.float32)
    
    def cache_old_photo_embedding(self, embedding):
        """Cache the old profile photo embedding"""
        import numpy as np
        
        if embedding is None:
            return
        
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        else:
            embedding = embedding.astype(np.float32)
        
        self.cached_old_photo_embedding = embedding.tobytes()
        self.save(update_fields=['cached_old_photo_embedding'])
    
    def get_cached_old_photo_embedding(self):
        """Get cached old photo embedding as numpy array"""
        import numpy as np
        
        if not self.cached_old_photo_embedding:
            return None
        
        return np.frombuffer(self.cached_old_photo_embedding, dtype=np.float32)
    
    def invalidate_embedding_cache(self):
        """Invalidate cached embeddings (e.g., after re-enrollment)"""
        self.cached_embedding = None
        self.embedding_cached_at = None
        self.save(update_fields=['cached_embedding', 'embedding_cached_at'])
    
    @property
    def display_name(self):
        """Get display name from profile"""
        profile = self.profile or {}
        return (
            profile.get('display_name') or 
            profile.get('full_name') or 
            profile.get('name') or 
            self.external_user_id
        )
    
    def update_recognition_timestamp(self):
        """Update last recognition timestamp"""
        from django.utils import timezone
        self.last_recognition_at = timezone.now()
        self.save(update_fields=['last_recognition_at'])


class ClientAPIUsage(models.Model):
    """
    Track API usage per client for billing and rate limiting
    """
    
    ENDPOINT_CHOICES = [
        ('enrollment', 'Enrollment'),
        ('recognition', 'Recognition'),
        ('liveness', 'Liveness Detection'),
        ('webhook', 'Webhook'),
        ('analytics', 'Analytics'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client = models.ForeignKey(Client, on_delete=models.CASCADE, related_name='api_usage')
    
    # Usage details
    endpoint = models.CharField(max_length=50, choices=ENDPOINT_CHOICES)
    method = models.CharField(max_length=10)
    status_code = models.IntegerField()
    
    # Request details
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    
    # Performance metrics
    response_time_ms = models.FloatField(null=True, blank=True)
    
    # Additional data
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        verbose_name = "API Usage"
        verbose_name_plural = "API Usage"
        indexes = [
            models.Index(fields=['client', 'created_at']),
            models.Index(fields=['endpoint', 'created_at']),
            models.Index(fields=['status_code']),
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.client.client_id} - {self.endpoint} - {self.created_at}"


class ClientWebhookLog(models.Model):
    """
    Log webhook delivery attempts
    """
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('success', 'Success'),
        ('failed', 'Failed'),
        ('retrying', 'Retrying'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client = models.ForeignKey(Client, on_delete=models.CASCADE, related_name='webhook_logs')
    
    # Webhook details
    event_type = models.CharField(max_length=100, db_index=True)
    payload = models.JSONField()
    
    # Delivery status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    response_status_code = models.IntegerField(null=True, blank=True)
    response_body = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    
    # Retry mechanism
    attempt_count = models.PositiveIntegerField(default=0)
    max_attempts = models.PositiveIntegerField(default=3)
    next_retry_at = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Webhook Log"
        verbose_name_plural = "Webhook Logs"
        indexes = [
            models.Index(fields=['client', 'event_type']),
            models.Index(fields=['status', 'next_retry_at']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.client.client_id} - {self.event_type} - {self.status}"
    
    @property
    def should_retry(self):
        """Check if webhook should be retried"""
        return (
            self.status in ['failed', 'retrying'] and 
            self.attempt_count < self.max_attempts
        )

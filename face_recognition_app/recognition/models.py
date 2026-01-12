"""
Face Recognition Models - Legacy (will be migrated to auth_service)
"""
import uuid
import numpy as np
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from encrypted_model_fields.fields import EncryptedTextField, EncryptedCharField
from PIL import Image
import json
import base64


class FaceEmbedding(models.Model):
    """Store face embeddings for users - DEPRECATED: Use auth_service.FaceEnrollment"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Legacy field - will be migrated to client_user
    # user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='face_embeddings')
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.CASCADE,
        related_name='legacy_embeddings',
        null=True,
        blank=True
    )
    
    # Embedding data (encrypted)
    embedding_vector = EncryptedTextField()  # Stores numpy array as binary
    embedding_hash = models.CharField(max_length=64, unique=True)  # Hash for deduplication
    
    # Quality metrics
    quality_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Quality score of the face capture (0-1)"
    )
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Detection confidence score (0-1)"
    )
    
    # Face metrics
    face_bbox = models.JSONField(
        help_text="Face bounding box coordinates [x1, y1, x2, y2]"
    )
    face_landmarks = models.JSONField(
        null=True, 
        blank=True,
        help_text="Facial landmarks coordinates"
    )
    
    # Capture context
    enrollment_session = models.ForeignKey(
        'EnrollmentSession', 
        on_delete=models.CASCADE, 
        related_name='embeddings'
    )
    sample_number = models.PositiveIntegerField()  # 1-5 for each session
    
    # Image data (optional, for audit trail)
    face_image = models.ImageField(
        upload_to='faces/%Y/%m/%d/', 
        null=True, 
        blank=True,
        help_text="Cropped face image (encrypted storage)"
    )
    
    # Verification data
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
    
    # Metadata
    capture_device = models.CharField(max_length=100, blank=True)
    capture_resolution = models.CharField(max_length=20, blank=True)  # e.g., "640x480"
    
    # Status
    is_active = models.BooleanField(default=True)
    is_verified = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Face Embedding"
        verbose_name_plural = "Face Embeddings"
        unique_together = ['enrollment_session', 'sample_number']
        indexes = [
            models.Index(fields=['client_user', 'is_active']),
            models.Index(fields=['embedding_hash']),
            models.Index(fields=['quality_score']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"Embedding #{self.sample_number} for {self.client_user.display_name if self.client_user else 'Unknown'}"

    def set_embedding_vector(self, vector):
        """Set embedding vector from numpy array"""
        if isinstance(vector, np.ndarray):
            # Normalize and convert to raw bytes
            normalized_vector = vector / np.linalg.norm(vector)
            raw_bytes = normalized_vector.astype(np.float32).tobytes()

            # Persist as base64-encoded string so EncryptedTextField can handle it
            encoded = base64.b64encode(raw_bytes).decode('ascii')
            self.embedding_vector = encoded

            # Create hash for deduplication using the raw bytes
            import hashlib
            self.embedding_hash = hashlib.sha256(raw_bytes).hexdigest()
        else:
            raise ValueError("Vector must be a numpy array")

    def get_embedding_vector(self):
        """Get embedding vector as numpy array"""
        if self.embedding_vector:
            raw_bytes = base64.b64decode(self.embedding_vector)
            return np.frombuffer(raw_bytes, dtype=np.float32)
        return None

    def calculate_similarity(self, other_embedding):
        """Calculate cosine similarity with another embedding"""
        vec1 = self.get_embedding_vector()
        vec2 = other_embedding.get_embedding_vector() if hasattr(other_embedding, 'get_embedding_vector') else other_embedding
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)


class EnrollmentSession(models.Model):
    """Track face enrollment sessions - DEPRECATED: Use auth_service.AuthenticationSession"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Legacy - migrated to client_user
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.CASCADE,
        related_name='legacy_enrollment_sessions',
        null=True,
        blank=True
    )
    
    # Session info
    session_token = models.CharField(max_length=255, unique=True)
    device_info = models.JSONField(default=dict)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    # Progress tracking
    target_samples = models.PositiveIntegerField(default=5)
    completed_samples = models.PositiveIntegerField(default=0)
    
    # Quality metrics
    average_quality = models.FloatField(default=0.0)
    min_quality_threshold = models.FloatField(default=0.7)
    
    # Status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Timestamps
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField()  # Session timeout
    
    # Logs
    session_log = models.JSONField(default=list)
    error_messages = models.TextField(blank=True)

    class Meta:
        verbose_name = "Enrollment Session"
        verbose_name_plural = "Enrollment Sessions"
        indexes = [
            models.Index(fields=['client_user', 'status']),
            models.Index(fields=['session_token']),
            models.Index(fields=['started_at']),
        ]
        ordering = ['-started_at']

    def __str__(self):
        return f"Enrollment for {self.client_user.display_name if self.client_user else 'Unknown'} - {self.status}"

    @property
    def progress_percentage(self):
        """Calculate progress percentage"""
        if self.target_samples == 0:
            return 0
        return min(100, (self.completed_samples / self.target_samples) * 100)

    @property
    def is_expired(self):
        """Check if session is expired"""
        from django.utils import timezone
        return timezone.now() > self.expires_at

    def add_log_entry(self, message, level='info', details=None):
        """Add entry to session log"""
        from django.utils import timezone
        
        log_entry = {
            'timestamp': timezone.now().isoformat(),
            'level': level,
            'message': message,
            'details': details or {}
        }
        
        if not isinstance(self.session_log, list):
            self.session_log = []
        
        self.session_log.append(log_entry)
        self.save(update_fields=['session_log'])


class AuthenticationAttempt(models.Model):
    """Track face authentication attempts - DEPRECATED: Use auth_service.FaceRecognitionAttempt"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Legacy - migrated to client_user
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.CASCADE,
        related_name='legacy_auth_attempts',
        null=True,
        blank=True
    )
    
    # Attempt info
    session_id = models.CharField(max_length=255)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    device_fingerprint = models.CharField(max_length=255, blank=True)
    
    # Authentication data
    submitted_embedding = EncryptedTextField(null=True, blank=True)
    matched_embedding = models.ForeignKey(
        FaceEmbedding,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='auth_matches'
    )
    
    # Scores and metrics
    similarity_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        null=True,
        blank=True
    )
    liveness_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        default=0.0
    )
    quality_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        default=0.0
    )
    
    # Obstacle detection results
    obstacles_detected = models.JSONField(default=list)
    
    # Result
    SUCCESS_STATUS = [
        ('success', 'Success'),
        ('failed_similarity', 'Failed - Low Similarity'),
        ('failed_liveness', 'Failed - Liveness Check'),
        ('failed_quality', 'Failed - Poor Quality'),
        ('failed_obstacles', 'Failed - Obstacles Detected'),
        ('failed_multiple_faces', 'Failed - Multiple Faces'),
        ('failed_no_face', 'Failed - No Face Detected'),
        ('failed_system_error', 'Failed - System Error'),
    ]
    result = models.CharField(max_length=30, choices=SUCCESS_STATUS)
    
    # Performance metrics
    processing_time = models.FloatField(null=True, blank=True)  # in milliseconds
    
    # Additional data
    face_bbox = models.JSONField(null=True, blank=True)
    metadata = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Authentication Attempt"
        verbose_name_plural = "Authentication Attempts"
        indexes = [
            models.Index(fields=['client_user', 'created_at']),
            models.Index(fields=['session_id']),
            models.Index(fields=['result', 'created_at']),
            models.Index(fields=['ip_address', 'created_at']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        user_name = self.client_user.display_name if self.client_user else 'Unknown'
        return f"Auth attempt by {user_name} - {self.result}"

    @property
    def is_successful(self):
        """Check if authentication was successful"""
        return self.result == 'success'


class LivenessDetection(models.Model):
    """Store liveness detection results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    authentication_attempt = models.OneToOneField(
        AuthenticationAttempt,
        on_delete=models.CASCADE,
        related_name='liveness_detection'
    )
    
    # Blink detection
    blinks_detected = models.PositiveIntegerField(default=0)
    blink_quality_scores = models.JSONField(default=list)
    
    # Eye aspect ratios
    ear_history = models.JSONField(default=list)
    ear_baseline = models.FloatField(null=True, blank=True)
    
    # Frame analysis
    frames_processed = models.PositiveIntegerField(default=0)
    valid_frames = models.PositiveIntegerField(default=0)
    
    # Challenge response (if implemented)
    challenge_type = models.CharField(
        max_length=20,
        choices=[
            ('blink', 'Blink Detection'),
            ('head_turn', 'Head Movement'),
            ('smile', 'Smile Detection'),
            ('random', 'Random Challenge'),
        ],
        default='blink'
    )
    challenge_completed = models.BooleanField(default=False)
    
    # Results
    liveness_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        default=0.0
    )
    is_live = models.BooleanField(default=False)
    
    # Debug information
    debug_data = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Liveness Detection"
        verbose_name_plural = "Liveness Detections"

    def __str__(self):
        return f"Liveness check - Score: {self.liveness_score:.2f}"


class ObstacleDetection(models.Model):
    """Store obstacle detection results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    authentication_attempt = models.OneToOneField(
        AuthenticationAttempt,
        on_delete=models.CASCADE,
        related_name='obstacle_detection'
    )
    
    # Detected obstacles
    glasses_detected = models.BooleanField(default=False)
    glasses_confidence = models.FloatField(default=0.0)
    
    mask_detected = models.BooleanField(default=False)
    mask_confidence = models.FloatField(default=0.0)
    
    hat_detected = models.BooleanField(default=False)
    hat_confidence = models.FloatField(default=0.0)
    
    hand_covering = models.BooleanField(default=False)
    hand_confidence = models.FloatField(default=0.0)
    
    # Overall assessment
    has_obstacles = models.BooleanField(default=False)
    obstacle_score = models.FloatField(default=0.0)
    
    # Details
    detection_details = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Obstacle Detection"
        verbose_name_plural = "Obstacle Detections"

    def __str__(self):
        obstacles = []
        if self.glasses_detected: obstacles.append('glasses')
        if self.mask_detected: obstacles.append('mask')
        if self.hat_detected: obstacles.append('hat')
        if self.hand_covering: obstacles.append('hand')
        
        if obstacles:
            return f"Obstacles: {', '.join(obstacles)}"
        return "No obstacles detected"


class FaceRecognitionModel(models.Model):
    """Store model configurations and versions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    model_type = models.CharField(
        max_length=20,
        choices=[
            ('detection', 'Face Detection'),
            ('recognition', 'Face Recognition'),
            ('liveness', 'Liveness Detection'),
            ('anti_spoofing', 'Anti-Spoofing'),
        ]
    )
    
    # Model configuration
    configuration = models.JSONField(default=dict)
    
    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    is_default = models.BooleanField(default=False)
    
    # Metadata
    description = models.TextField(blank=True)
    created_by_client = models.ForeignKey(
        'clients.Client',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_models'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Face Recognition Model"
        verbose_name_plural = "Face Recognition Models"
        unique_together = ['name', 'version']

    def __str__(self):
        return f"{self.name} v{self.version}"

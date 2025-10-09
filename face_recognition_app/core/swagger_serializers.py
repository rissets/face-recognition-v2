"""
Core app request/response serializers for Swagger documentation
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model
from drf_spectacular.utils import extend_schema_field
from drf_spectacular.types import OpenApiTypes
import base64
import uuid

User = get_user_model()


class UserRegistrationSerializer(serializers.ModelSerializer):
    """
    User registration serializer with comprehensive validation
    
    This serializer handles new user registration with face recognition capabilities.
    It validates email uniqueness, password strength, and creates associated profile.
    """
    password = serializers.CharField(
        write_only=True, 
        min_length=8,
        help_text="Password must be at least 8 characters long",
        style={'input_type': 'password'}
    )
    password_confirm = serializers.CharField(
        write_only=True,
        help_text="Confirm password (must match password field)",
        style={'input_type': 'password'}
    )
    
    class Meta:
        model = User
        fields = (
            'email', 'username', 'first_name', 'last_name', 
            'password', 'password_confirm', 'phone_number'
        )
        extra_kwargs = {
            'email': {
                'help_text': 'Valid email address (will be used as username)',
                'required': True
            },
            'username': {
                'help_text': 'Unique username (3-150 characters)',
                'required': True
            },
            'first_name': {
                'help_text': 'User first name',
                'required': True
            },
            'last_name': {
                'help_text': 'User last name',
                'required': True
            },
            'phone_number': {
                'help_text': 'Phone number in international format (+1234567890)',
                'required': False
            }
        }
    
    def validate_email(self, value):
        """Validate email uniqueness"""
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email already registered")
        return value
    
    def validate_username(self, value):
        """Validate username uniqueness"""
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("Username already taken")
        return value
    
    def validate(self, attrs):
        """Cross-field validation"""
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({
                "password_confirm": "Passwords don't match"
            })
        return attrs
    
    def create(self, validated_data):
        """Create user with encrypted password"""
        validated_data.pop('password_confirm')
        password = validated_data.pop('password')
        
        user = User.objects.create_user(**validated_data)
        user.set_password(password)
        user.save()
        
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    """
    User profile serializer with computed fields
    
    Provides comprehensive user information including face enrollment status,
    authentication history, and security settings.
    """
    full_name = serializers.CharField(
        source='get_full_name', 
        read_only=True,
        help_text="User's full name (first + last name)"
    )
    
    @extend_schema_field(OpenApiTypes.FLOAT)
    def get_enrollment_progress(self, obj):
        """Calculate enrollment progress percentage"""
        if not hasattr(obj, 'enrollment_sessions'):
            return 0.0
        
        latest_session = obj.enrollment_sessions.filter(
            status__in=['in_progress', 'completed']
        ).first()
        
        if not latest_session:
            return 0.0
            
        if latest_session.status == 'completed':
            return 100.0
            
        return (latest_session.completed_samples / latest_session.target_samples) * 100
    
    enrollment_progress = serializers.SerializerMethodField(
        help_text="Face enrollment progress percentage (0-100)"
    )
    
    @extend_schema_field(OpenApiTypes.INT)
    def get_total_authentications(self, obj):
        """Get total authentication attempts"""
        return obj.authentication_attempts.count()
    
    total_authentications = serializers.SerializerMethodField(
        help_text="Total number of authentication attempts"
    )
    
    @extend_schema_field(OpenApiTypes.INT)
    def get_successful_authentications(self, obj):
        """Get successful authentication count"""
        return obj.authentication_attempts.filter(is_successful=True).count()
    
    successful_authentications = serializers.SerializerMethodField(
        help_text="Number of successful authentications"
    )
    
    class Meta:
        model = User
        fields = (
            'id', 'email', 'username', 'first_name', 'last_name',
            'full_name', 'phone_number', 'date_of_birth', 'profile_picture',
            'bio', 'face_enrolled', 'enrollment_progress', 'face_auth_enabled',
            'two_factor_enabled', 'is_verified', 'last_face_auth', 
            'total_authentications', 'successful_authentications',
            'created_at', 'updated_at'
        )
        read_only_fields = (
            'id', 'email', 'created_at', 'updated_at', 'enrollment_progress',
            'face_enrolled', 'total_authentications', 'successful_authentications'
        )
        extra_kwargs = {
            'phone_number': {
                'help_text': 'Phone number in international format'
            },
            'date_of_birth': {
                'help_text': 'Date of birth (YYYY-MM-DD format)'
            },
            'bio': {
                'help_text': 'User biography (max 500 characters)'
            },
            'face_auth_enabled': {
                'help_text': 'Enable/disable face authentication for this user'
            },
            'two_factor_enabled': {
                'help_text': 'Enable/disable two-factor authentication'
            }
        }


class EnrollmentFrameSerializer(serializers.Serializer):
    """
    Serializer for processing enrollment frames
    
    Handles individual frame processing during face enrollment,
    including image data validation and quality assessment.
    """
    session_token = serializers.CharField(
        help_text="Enrollment session token (UUID format)"
    )
    frame_data = serializers.CharField(
        help_text="Base64 encoded image data (JPEG/PNG format)"
    )
    frame_number = serializers.IntegerField(
        min_value=1,
        help_text="Sequential frame number in the session"
    )
    timestamp = serializers.DateTimeField(
        required=False,
        help_text="Frame capture timestamp (ISO 8601 format)"
    )
    device_info = serializers.JSONField(
        required=False,
        help_text="Device information (camera specs, resolution, etc.)"
    )
    
    def validate_frame_data(self, value):
        """Validate base64 image data"""
        try:
            # Remove data URL prefix if present
            if ',' in value:
                value = value.split(',')[1]
            
            # Validate base64 encoding
            base64.b64decode(value)
            return value
        except Exception:
            raise serializers.ValidationError("Invalid base64 image data")
    
    def validate_session_token(self, value):
        """Validate session token format"""
        try:
            uuid.UUID(value)
            return value
        except ValueError:
            raise serializers.ValidationError("Invalid session token format")


class AuthenticationFrameSerializer(serializers.Serializer):
    """
    Serializer for processing authentication frames
    
    Handles real-time face authentication frame processing,
    including liveness detection and anti-spoofing validation.
    """
    session_id = serializers.CharField(
        help_text="Authentication session ID (UUID format)"
    )
    frame_data = serializers.CharField(
        help_text="Base64 encoded image data (JPEG/PNG format)"
    )
    sequence_number = serializers.IntegerField(
        min_value=1,
        help_text="Frame sequence number for temporal analysis"
    )
    timestamp = serializers.DateTimeField(
        required=False,
        help_text="Frame capture timestamp (ISO 8601 format)"
    )
    liveness_required = serializers.BooleanField(
        default=True,
        help_text="Whether liveness detection is required"
    )
    anti_spoofing_enabled = serializers.BooleanField(
        default=True,
        help_text="Whether anti-spoofing detection is enabled"
    )
    
    def validate_frame_data(self, value):
        """Validate base64 image data"""
        try:
            if ',' in value:
                value = value.split(',')[1]
            base64.b64decode(value)
            return value
        except Exception:
            raise serializers.ValidationError("Invalid base64 image data")


class WebRTCSignalingSerializer(serializers.Serializer):
    """
    Serializer for WebRTC signaling messages
    
    Handles WebRTC offer, answer, and ICE candidate exchange
    for real-time video streaming during face recognition.
    """
    MESSAGE_TYPES = [
        ('offer', 'SDP Offer'),
        ('answer', 'SDP Answer'),
        ('ice_candidate', 'ICE Candidate'),
        ('bye', 'Session Termination'),
    ]
    
    session_id = serializers.CharField(
        help_text="WebRTC session ID (UUID format)"
    )
    message_type = serializers.ChoiceField(
        choices=MESSAGE_TYPES,
        help_text="Type of WebRTC signaling message"
    )
    payload = serializers.JSONField(
        help_text="WebRTC signaling payload (SDP or ICE candidate data)"
    )
    timestamp = serializers.DateTimeField(
        required=False,
        help_text="Message timestamp (ISO 8601 format)"
    )
    
    def validate_payload(self, value):
        """Validate WebRTC payload structure"""
        message_type = self.initial_data.get('message_type')
        
        if message_type in ['offer', 'answer']:
            if 'sdp' not in value or 'type' not in value:
                raise serializers.ValidationError(
                    "SDP payload must contain 'sdp' and 'type' fields"
                )
        elif message_type == 'ice_candidate':
            required_fields = ['candidate', 'sdpMid', 'sdpMLineIndex']
            if not all(field in value for field in required_fields):
                raise serializers.ValidationError(
                    f"ICE candidate payload must contain: {required_fields}"
                )
        
        return value


class AuthenticationResultSerializer(serializers.Serializer):
    """
    Serializer for authentication results
    
    Returns comprehensive authentication results including
    confidence scores, liveness detection, and security metrics.
    """
    is_authenticated = serializers.BooleanField(
        help_text="Whether authentication was successful"
    )
    confidence_score = serializers.FloatField(
        min_value=0.0, max_value=1.0,
        help_text="Authentication confidence score (0.0-1.0)"
    )
    liveness_score = serializers.FloatField(
        min_value=0.0, max_value=1.0,
        help_text="Liveness detection score (0.0-1.0)"
    )
    anti_spoofing_score = serializers.FloatField(
        min_value=0.0, max_value=1.0,
        help_text="Anti-spoofing detection score (0.0-1.0)"
    )
    match_user_id = serializers.UUIDField(
        allow_null=True,
        help_text="ID of the matched user (if authenticated)"
    )
    processing_time = serializers.FloatField(
        help_text="Processing time in milliseconds"
    )
    quality_metrics = serializers.JSONField(
        help_text="Image quality metrics and face detection details"
    )
    security_alerts = serializers.ListField(
        child=serializers.CharField(),
        help_text="List of security alerts or warnings"
    )
    session_id = serializers.CharField(
        help_text="Authentication session ID"
    )
    timestamp = serializers.DateTimeField(
        help_text="Authentication timestamp"
    )


class EnrollmentResultSerializer(serializers.Serializer):
    """
    Serializer for enrollment results
    
    Returns enrollment progress and quality metrics
    for each processed frame during face enrollment.
    """
    is_successful = serializers.BooleanField(
        help_text="Whether frame processing was successful"
    )
    quality_score = serializers.FloatField(
        min_value=0.0, max_value=1.0,
        help_text="Face image quality score (0.0-1.0)"
    )
    face_detected = serializers.BooleanField(
        help_text="Whether a face was detected in the frame"
    )
    face_bbox = serializers.JSONField(
        help_text="Face bounding box coordinates [x1, y1, x2, y2]"
    )
    enrollment_progress = serializers.FloatField(
        min_value=0.0, max_value=100.0,
        help_text="Overall enrollment progress percentage (0-100)"
    )
    samples_collected = serializers.IntegerField(
        help_text="Number of valid samples collected"
    )
    samples_required = serializers.IntegerField(
        help_text="Total number of samples required"
    )
    feedback_message = serializers.CharField(
        help_text="User feedback message for enrollment guidance"
    )
    session_token = serializers.CharField(
        help_text="Enrollment session token"
    )
    is_enrollment_complete = serializers.BooleanField(
        help_text="Whether enrollment is complete"
    )
    processing_time = serializers.FloatField(
        help_text="Frame processing time in milliseconds"
    )


class SystemStatusSerializer(serializers.Serializer):
    """
    Serializer for system status information
    
    Provides comprehensive system health and performance metrics
    for monitoring and debugging purposes.
    """
    status = serializers.CharField(
        help_text="Overall system status (healthy, degraded, down)"
    )
    timestamp = serializers.DateTimeField(
        help_text="Status check timestamp"
    )
    version = serializers.CharField(
        help_text="API version"
    )
    uptime = serializers.IntegerField(
        help_text="System uptime in seconds"
    )
    services = serializers.JSONField(
        help_text="Individual service status and metrics"
    )
    performance_metrics = serializers.JSONField(
        help_text="System performance metrics (CPU, memory, etc.)"
    )
    active_sessions = serializers.IntegerField(
        help_text="Number of active user sessions"
    )
    database_status = serializers.CharField(
        help_text="Database connection status"
    )
    redis_status = serializers.CharField(
        help_text="Redis connection status"
    )
    face_recognition_model_status = serializers.CharField(
        help_text="Face recognition model status"
    )


class ErrorResponseSerializer(serializers.Serializer):
    """
    Standard error response serializer
    
    Provides consistent error response format across all API endpoints
    with detailed error information and debugging context.
    """
    error = serializers.CharField(
        help_text="Error type or category"
    )
    message = serializers.CharField(
        help_text="Human-readable error message"
    )
    details = serializers.JSONField(
        required=False,
        help_text="Additional error details and context"
    )
    code = serializers.CharField(
        required=False,
        help_text="Internal error code for debugging"
    )
    timestamp = serializers.DateTimeField(
        help_text="Error occurrence timestamp"
    )
    request_id = serializers.CharField(
        required=False,
        help_text="Unique request identifier for tracking"
    )


class PaginatedResponseSerializer(serializers.Serializer):
    """
    Standard paginated response serializer
    
    Provides consistent pagination format for list endpoints
    with navigation links and result metadata.
    """
    count = serializers.IntegerField(
        help_text="Total number of items"
    )
    next = serializers.URLField(
        allow_null=True,
        help_text="URL to next page of results"
    )
    previous = serializers.URLField(
        allow_null=True,
        help_text="URL to previous page of results"
    )
    results = serializers.ListField(
        help_text="Array of result items"
    )
"""
Serializers for third-party face authentication service
"""
from rest_framework import serializers
from django.utils import timezone
from .models import (
    AuthenticationSession, FaceEnrollment, FaceRecognitionAttempt,
    LivenessDetectionResult, SystemMetrics
)
from clients.models import ClientUser
import uuid
import base64
from django.core.files.base import ContentFile


class HybridImageField(serializers.Field):
    """
    Image field that accepts either uploaded files or base64-encoded strings.
    """

    def to_internal_value(self, data):
        # Handle empty or None data
        if not data:
            raise serializers.ValidationError('No image data provided.')
        
        # Handle list/array input (take first item)
        if isinstance(data, (list, tuple)):
            data = data[0] if data else None
            if not data:
                raise serializers.ValidationError('No image data provided.')

        # Handle dict input with base64 or data keys
        if isinstance(data, dict):
            if 'base64' in data:
                data = data.get('base64')
            elif 'data' in data:
                data = data.get('data')
            elif 'image' in data:
                data = data.get('image')
            else:
                raise serializers.ValidationError('Invalid image data format. Expected base64, data, or image key.')

        # Handle string input (base64)
        if isinstance(data, str):
            # Remove whitespace
            image_data = data.strip()
            
            # Default extension
            extension = 'jpg'
            
            # Handle data URL format (data:image/jpeg;base64,...)
            if image_data.startswith('data:'):
                if ';base64,' in image_data:
                    header, image_data = image_data.split(';base64,', 1)
                    # Extract file extension from MIME type
                    if 'image/' in header:
                        extension = header.split('image/')[-1].split(';')[0]
                        # Handle common MIME types
                        if extension == 'jpeg':
                            extension = 'jpg'
                else:
                    raise serializers.ValidationError('Invalid data URL format. Expected base64 encoding.')
            
            # Validate base64 format
            if not image_data:
                raise serializers.ValidationError('Empty image data.')
            
            try:
                # Validate base64 and decode
                decoded_file = base64.b64decode(image_data, validate=True)
                
                # Check if decoded data is not empty
                if len(decoded_file) == 0:
                    raise serializers.ValidationError('Decoded image data is empty.')
                
                # Validate minimum file size (at least 100 bytes for a valid image)
                if len(decoded_file) < 100:
                    raise serializers.ValidationError('Image data too small to be a valid image.')
                    
            except (TypeError, ValueError, base64.binascii.Error) as exc:
                raise serializers.ValidationError(f'Invalid base64 image data: {str(exc)}')

            # Create Django file object
            file_name = f"face-{uuid.uuid4().hex[:8]}.{extension}"
            return ContentFile(decoded_file, name=file_name)
        
        # Handle Django UploadedFile objects (from multipart form data)
        elif hasattr(data, 'read'):
            # This is likely an uploaded file
            from django.core.files.uploadedfile import UploadedFile
            if isinstance(data, UploadedFile):
                return data
            
            # Handle other file-like objects
            try:
                content = data.read()
                if hasattr(data, 'name'):
                    file_name = data.name
                else:
                    file_name = f"upload-{uuid.uuid4().hex[:8]}.jpg"
                return ContentFile(content, name=file_name)
            except Exception as exc:
                raise serializers.ValidationError(f'Error reading uploaded file: {str(exc)}')
        
        # Unknown data type
        else:
            raise serializers.ValidationError(
                f'Invalid image data type: {type(data).__name__}. '
                'Expected string (base64), file upload, or dict with base64 data.'
            )

    def to_representation(self, value):
        """Return file URL or None"""
        if not value:
            return None
        
        # If value has url attribute, return it
        if hasattr(value, 'url'):
            return value.url
        
        # If value is string (path), return as-is
        if isinstance(value, str):
            return value
            
        return None


class AuthenticationSessionSerializer(serializers.ModelSerializer):
    """Serializer for authentication sessions"""
    
    client_user_info = serializers.SerializerMethodField()
    
    class Meta:
        model = AuthenticationSession
        fields = [
            'id', 'session_token', 'client', 'client_user', 'client_user_info',
            'session_type', 'status', 'ip_address', 'user_agent', 
            'metadata', 'created_at', 'expires_at', 'completed_at'
        ]
        read_only_fields = ['id', 'session_token', 'created_at', 'client_user_info']

    def get_client_user_info(self, obj):
        """Get client user information if available"""
        if obj.client_user:
            return {
                'id': str(obj.client_user.id),
                'external_user_id': obj.client_user.external_user_id,
                'display_name': obj.client_user.display_name,
                'is_enrolled': obj.client_user.is_enrolled,
                'profile_image_url': obj.client_user.profile_image.url if obj.client_user.profile_image else None
            }
        return None
    
    def create(self, validated_data):
        """Create session with auto-generated token"""
        validated_data['session_token'] = str(uuid.uuid4())
        if 'expires_at' not in validated_data:
            validated_data['expires_at'] = timezone.now() + timezone.timedelta(minutes=10)
        return super().create(validated_data)


class FaceEnrollmentSerializer(serializers.ModelSerializer):
    """Serializer for face enrollment"""
    
    class Meta:
        model = FaceEnrollment
        fields = [
            'id', 'client', 'client_user', 'enrollment_session', 'status',
            'embedding_vector', 'embedding_dimension', 'face_quality_score', 
            'liveness_score', 'anti_spoofing_score', 'face_landmarks', 'face_bbox',
            'sample_number', 'total_samples', 'face_image_path', 'metadata',
            'created_at', 'updated_at', 'expires_at'
        ]
        read_only_fields = [
            'id', 'embedding_vector', 'embedding_dimension', 'face_quality_score',
            'liveness_score', 'anti_spoofing_score', 'face_landmarks', 'face_bbox', 
            'created_at', 'updated_at'
        ]


class FaceRecognitionAttemptSerializer(serializers.ModelSerializer):
    """Serializer for face recognition attempts"""
    
    class Meta:
        model = FaceRecognitionAttempt
        fields = [
            'id', 'session', 'client', 'matched_user', 'matched_enrollment',
            'result', 'similarity_score', 'confidence_score', 'face_quality_score',
            'liveness_score', 'anti_spoofing_score', 'submitted_embedding',
            'face_landmarks', 'face_bbox', 'processing_time_ms',
            'ip_address', 'user_agent', 'device_fingerprint',
            'metadata', 'created_at'
        ]
        read_only_fields = [
            'id', 'similarity_score', 'confidence_score', 'face_quality_score',
            'liveness_score', 'anti_spoofing_score', 'submitted_embedding',
            'face_landmarks', 'face_bbox', 'processing_time_ms', 'created_at'
        ]


class LivenessDetectionResultSerializer(serializers.ModelSerializer):
    """Serializer for liveness detection results"""
    
    class Meta:
        model = LivenessDetectionResult
        fields = [
            'id', 'session', 'client', 'status', 'confidence_score',
            'blink_detected', 'blink_count', 'head_movement_detected',
            'face_quality_score', 'spoofing_detected', 'processing_time',
            'frame_analysis', 'metadata', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class SystemMetricsSerializer(serializers.ModelSerializer):
    """Serializer for system metrics"""
    
    class Meta:
        model = SystemMetrics
        fields = [
            'id', 'client', 'metric_type', 'metric_name', 'metric_value',
            'dimensions', 'metadata', 'timestamp'
        ]
        read_only_fields = ['id', 'timestamp']


# Request/Response serializers for API operations

class EnrollmentRequestSerializer(serializers.Serializer):
    """Serializer for enrollment request"""
    user_id = serializers.CharField(max_length=100)
    session_type = serializers.ChoiceField(choices=['webcam', 'mobile', 'api'])
    metadata = serializers.JSONField(required=False, default=dict)
    
    def validate_user_id(self, value):
        """Validate user exists for the client"""
        client = self.context.get('client')
        if not client:
            raise serializers.ValidationError("Client context required")
        
        try:
            ClientUser.objects.get(client=client, external_user_id=value)
        except ClientUser.DoesNotExist:
            raise serializers.ValidationError("User not found or inactive")
        return value


class AuthenticationRequestSerializer(serializers.Serializer):
    """Serializer for authentication request"""
    user_id = serializers.CharField(max_length=100, required=False)
    session_type = serializers.ChoiceField(choices=['webcam', 'mobile', 'api'])
    require_liveness = serializers.BooleanField(default=True)
    metadata = serializers.JSONField(required=False, default=dict)


class FaceImageUploadSerializer(serializers.Serializer):
    """Serializer for face image upload"""
    image = HybridImageField(required=True)
    session_token = serializers.CharField(max_length=100)
    frame_number = serializers.IntegerField(required=False, default=1)
    timestamp = serializers.DateTimeField(required=False)

    def validate_image(self, value):
        """Additional validation for image"""
        if not value:
            raise serializers.ValidationError('Image is required.')
        
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if hasattr(value, 'size') and value.size > max_size:
            raise serializers.ValidationError(
                f'Image file too large. Maximum size is {max_size // (1024*1024)}MB.'
            )
        
        # Validate file extension
        if hasattr(value, 'name'):
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            import os
            ext = os.path.splitext(value.name)[1].lower()
            if ext and ext not in valid_extensions:
                raise serializers.ValidationError(
                    f'Invalid file extension {ext}. Allowed: {", ".join(valid_extensions)}'
                )
        
        return value

    def validate(self, attrs):
        """Ensure image data is valid and session token is provided"""
        if not attrs.get('image'):
            raise serializers.ValidationError({
                'image': 'Image is required. Provide either an uploaded file or base64 image data.'
            })
        
        if not attrs.get('session_token'):
            raise serializers.ValidationError({
                'session_token': 'Session token is required.'
            })
            
        return attrs


class WebRTCSignalingSerializer(serializers.Serializer):
    """Serializer for WebRTC signaling data"""
    session_token = serializers.CharField(max_length=100)
    type = serializers.ChoiceField(choices=['offer', 'answer', 'ice-candidate'])
    data = serializers.JSONField()


class AuthenticationResponseSerializer(serializers.Serializer):
    """Serializer for authentication session creation response."""

    session_token = serializers.CharField(help_text="Authentication session token.")
    status = serializers.CharField(help_text="Initial session status.")
    session_type = serializers.CharField(help_text="Authentication session type.")
    expires_at = serializers.DateTimeField(help_text="Session expiration timestamp.")
    message = serializers.CharField(help_text="Human readable guidance.", required=False)


class EnrollmentCreateResponseSerializer(serializers.Serializer):
    """Serializer for the response after creating an enrollment session."""
    session_token = serializers.CharField(help_text="The token for the created enrollment session.")
    enrollment_id = serializers.CharField(help_text="The unique identifier for the new enrollment record.")
    status = serializers.CharField(help_text="The initial status of the enrollment, typically 'pending'.")
    target_samples = serializers.IntegerField(help_text="The number of successful frames required to complete the enrollment.")
    expires_at = serializers.DateTimeField(help_text="The timestamp when the session will expire.")
    message = serializers.CharField(help_text="A human-readable message providing guidance for the next steps.")
    websocket_url = serializers.CharField(help_text="WebSocket URL for real-time image processing.", required=False)


class AuthSessionCreateResponseSerializer(serializers.Serializer):
    """Serializer for the response after creating an authentication session."""
    session_token = serializers.CharField(help_text="The token for the created authentication session.")
    status = serializers.CharField(help_text="The initial status of the session, typically 'active'.")
    expires_at = serializers.DateTimeField(help_text="The timestamp when the session will expire.")
    session_type = serializers.CharField(help_text="The type of session, either 'verification' or 'identification'.")
    message = serializers.CharField(help_text="A human-readable message providing guidance for the next steps.")
    websocket_url = serializers.CharField(help_text="WebSocket URL for real-time image processing.", required=False)


class EnrollmentResponseSerializer(serializers.Serializer):
    """Serializer for enrollment session creation response."""

    session_token = serializers.CharField(help_text="Enrollment session token.")
    enrollment_id = serializers.CharField(help_text="Identifier of the created enrollment.")
    status = serializers.CharField(help_text="Current enrollment session status.")
    target_samples = serializers.IntegerField(
        help_text="Number of frames required to complete enrollment."
    )
    expires_at = serializers.DateTimeField(help_text="Enrollment session expiry timestamp.")
    message = serializers.CharField(help_text="Human readable guidance.", required=False)


class SessionStatusSerializer(serializers.Serializer):
    """Serializer for session status"""
    session_token = serializers.CharField()
    status = serializers.CharField()
    session_type = serializers.CharField()
    created_at = serializers.DateTimeField()
    expires_at = serializers.DateTimeField()
    completed_at = serializers.DateTimeField(required=False, allow_null=True)
    result = serializers.CharField(required=False)
    metadata = serializers.JSONField(required=False)
    enrollment_status = serializers.CharField(required=False, help_text="Status of the enrollment if session_type is 'enrollment'")


class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses"""
    error = serializers.CharField()
    message = serializers.CharField(required=False)
    details = serializers.JSONField(required=False)
    error_code = serializers.CharField(required=False)
    timestamp = serializers.DateTimeField(required=False)


class FaceFrameResponseSerializer(serializers.Serializer):
    """
    Serializer describing the payload returned when processing enrollment or authentication frames.
    All fields are optional except `success` - responses vary based on workflow stage.
    """

    success = serializers.BooleanField(help_text="Indicates if the frame was processed successfully.")
    session_status = serializers.CharField(required=False, help_text="The current status of the session.")
    session_token = serializers.CharField(required=False, help_text="The session token for context.")
    message = serializers.CharField(required=False, help_text="A human-readable message about the result.")
    error = serializers.CharField(required=False, help_text="An error message if processing failed.")
    requires_more_frames = serializers.BooleanField(required=False, help_text="True if the session requires more frames to complete.")
    frame_accepted = serializers.BooleanField(required=False, help_text="True if this specific frame was accepted for processing.")
    frame_rejected = serializers.BooleanField(required=False, help_text="True if this frame was rejected.")
    attempted_frame = serializers.IntegerField(required=False, help_text="The sequence number of the frame that was attempted.")
    frames_processed = serializers.IntegerField(required=False, help_text="The total number of frames successfully processed so far.")
    completed_samples = serializers.IntegerField(required=False, help_text="Number of accepted samples for enrollment.")
    target_samples = serializers.IntegerField(required=False, help_text="Total number of samples required for enrollment.")
    enrollment_progress = serializers.FloatField(required=False, help_text="The progress of the enrollment as a percentage (0-100).")
    enrollment_complete = serializers.BooleanField(required=False, help_text="True if the enrollment process is complete.")
    liveness_verified = serializers.BooleanField(required=False, help_text="True if liveness has been verified.")
    liveness_score = serializers.FloatField(required=False, help_text="A score indicating the confidence of liveness detection.")
    liveness_data = serializers.JSONField(required=False, help_text="Detailed data from the liveness detection process.")
    liveness_blinks = serializers.IntegerField(required=False, help_text="Number of blinks detected during the session.")
    liveness_motion_events = serializers.IntegerField(required=False, help_text="Number of motion events detected.")
    similarity_score = serializers.FloatField(required=False, help_text="The similarity score in an authentication attempt.")
    quality_score = serializers.FloatField(required=False, help_text="A score representing the quality of the face image.")
    anti_spoofing_score = serializers.FloatField(required=False, help_text="A score indicating the confidence against spoofing attempts.")
    preview_image = serializers.CharField(required=False, help_text="A base64-encoded data URL of a preview image.")
    obstacles = serializers.ListField(child=serializers.CharField(), required=False, help_text="A list of detected obstacles (e.g., 'face_too_small').")
    obstacles_detected = serializers.ListField(child=serializers.CharField(), required=False, help_text="Deprecated. Use 'obstacles'.")
    obstacle_confidence = serializers.JSONField(required=False, help_text="A dictionary with confidence scores for detected obstacles.")
    match_fallback_used = serializers.BooleanField(required=False, help_text="Indicates if a fallback matching mechanism was used.")
    match_fallback_explanation = serializers.CharField(required=False, help_text="Explanation for why a fallback was used.")
    authentication_metadata = serializers.JSONField(required=False, help_text="Additional metadata related to the authentication attempt.")
    matched_user = serializers.DictField(required=False, help_text="Information about the matched user.")
    liveness_reason = serializers.CharField(required=False, help_text="The reason for the liveness verification result.")
    result = serializers.CharField(required=False, help_text="The final result of the operation.")
    processing_time_ms = serializers.FloatField(required=False, help_text="The time taken to process the frame in milliseconds.")
    rejection_reason = serializers.CharField(required=False, help_text="Reason for frame rejection.")


class ImageValidationErrorSerializer(serializers.Serializer):
    """Serializer for image validation errors with helpful messages"""
    error = serializers.CharField()
    message = serializers.CharField()
    suggestions = serializers.ListField(child=serializers.CharField(), required=False)
    error_code = serializers.CharField(default='IMAGE_VALIDATION_ERROR')
    
    @classmethod
    def create_response(cls, error_message, suggestions=None):
        """Create standardized image validation error response"""
        return {
            'error': 'Image validation failed',
            'message': error_message,
            'suggestions': suggestions or [
                'Ensure image is in base64 format or valid file upload',
                'Supported formats: JPG, PNG, WebP, BMP',
                'Maximum file size: 10MB',
                'Use proper data URL format: data:image/jpeg;base64,...'
            ],
            'error_code': 'IMAGE_VALIDATION_ERROR'
        }


class AnalyticsPeriodSerializer(serializers.Serializer):
    """Serializer describing a time range used for analytics aggregations."""

    start = serializers.DateTimeField()
    end = serializers.DateTimeField()


class EnrollmentAnalyticsSerializer(serializers.Serializer):
    total = serializers.IntegerField()
    active = serializers.IntegerField()
    pending = serializers.IntegerField()


class AttemptAnalyticsSerializer(serializers.Serializer):
    total = serializers.IntegerField()
    success = serializers.IntegerField()
    failed = serializers.IntegerField()
    avg_similarity = serializers.FloatField(allow_null=True, required=False)


class SessionAnalyticsSerializer(serializers.Serializer):
    total = serializers.IntegerField()
    completed = serializers.IntegerField()
    failed = serializers.IntegerField()


class ClientAnalyticsResponseSerializer(serializers.Serializer):
    """Serializer for the aggregate analytics payload exposed to clients."""

    period = AnalyticsPeriodSerializer()
    enrollments = EnrollmentAnalyticsSerializer()
    authentication_attempts = AttemptAnalyticsSerializer()
    sessions = SessionAnalyticsSerializer()

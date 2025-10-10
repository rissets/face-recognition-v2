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
    
    class Meta:
        model = AuthenticationSession
        fields = [
            'id', 'session_token', 'client', 'session_type', 'status',
            'ip_address', 'user_agent', 'metadata', 'created_at',
            'expires_at', 'completed_at'
        ]
        read_only_fields = ['id', 'session_token', 'created_at']
    
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
    """Serializer for authentication response"""
    session_token = serializers.CharField()
    status = serializers.CharField()
    result = serializers.CharField(required=False)
    user_id = serializers.CharField(required=False)
    confidence_score = serializers.FloatField(required=False)
    liveness_score = serializers.FloatField(required=False)
    processing_time = serializers.FloatField(required=False)
    message = serializers.CharField(required=False)
    next_step = serializers.CharField(required=False)


class EnrollmentResponseSerializer(serializers.Serializer):
    """Serializer for enrollment response"""
    session_token = serializers.CharField()
    enrollment_id = serializers.CharField(required=False)
    status = serializers.CharField()
    quality_score = serializers.FloatField(required=False)
    liveness_score = serializers.FloatField(required=False)
    message = serializers.CharField(required=False)
    next_step = serializers.CharField(required=False)


class SessionStatusSerializer(serializers.Serializer):
    """Serializer for session status"""
    session_token = serializers.CharField()
    status = serializers.CharField()
    session_type = serializers.CharField()
    created_at = serializers.DateTimeField()
    expires_at = serializers.DateTimeField()
    completed_at = serializers.DateTimeField(required=False)
    result = serializers.CharField(required=False)
    metadata = serializers.JSONField(required=False)


class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses"""
    error = serializers.CharField()
    message = serializers.CharField(required=False)
    details = serializers.JSONField(required=False)
    error_code = serializers.CharField(required=False)
    timestamp = serializers.DateTimeField(required=False)


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

"""
Serializers for Face Recognition API
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model
from recognition.models import (
    FaceEmbedding, EnrollmentSession, AuthenticationAttempt,
    LivenessDetection, ObstacleDetection
)
from analytics.models import AuthenticationLog, SecurityAlert
from users.models import UserProfile, UserDevice
from streaming.models import StreamingSession
import base64
import uuid

User = get_user_model()


class UserRegistrationSerializer(serializers.ModelSerializer):
    """User registration serializer"""
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = (
            'email', 'username', 'first_name', 'last_name', 
            'password', 'password_confirm', 'phone_number'
        )
    
    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError("Passwords don't match")
        return attrs
    
    def create(self, validated_data):
        validated_data.pop('password_confirm')
        password = validated_data.pop('password')
        
        user = User.objects.create_user(**validated_data)
        user.set_password(password)
        user.save()
        
        # Create user profile
        UserProfile.objects.create(user=user)
        
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    """User profile serializer"""
    full_name = serializers.CharField(source='get_full_name', read_only=True)
    enrollment_progress = serializers.FloatField(read_only=True)
    face_enrolled = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = User
        fields = (
            'id', 'email', 'username', 'first_name', 'last_name',
            'full_name', 'phone_number', 'date_of_birth', 'profile_picture',
            'bio', 'face_enrolled', 'enrollment_progress', 'face_auth_enabled',
            'two_factor_enabled', 'is_verified', 'last_face_auth', 'created_at'
        )
        read_only_fields = ('id', 'email', 'created_at', 'enrollment_progress', 'face_enrolled')


class EnrollmentSessionSerializer(serializers.ModelSerializer):
    """Enrollment session serializer"""
    progress_percentage = serializers.FloatField(read_only=True)
    is_expired = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = EnrollmentSession
        fields = (
            'id', 'session_token', 'target_samples', 'completed_samples',
            'progress_percentage', 'average_quality', 'status', 'is_expired',
            'started_at', 'completed_at', 'expires_at', 'device_info'
        )
        read_only_fields = (
            'id', 'session_token', 'progress_percentage', 'is_expired',
            'started_at', 'completed_at'
        )


class FaceEmbeddingSerializer(serializers.ModelSerializer):
    """Face embedding serializer"""
    
    class Meta:
        model = FaceEmbedding
        fields = (
            'id', 'quality_score', 'confidence_score', 'face_bbox',
            'sample_number', 'liveness_score', 'anti_spoofing_score',
            'is_active', 'is_verified', 'created_at'
        )
        read_only_fields = ('id', 'created_at')


class AuthenticationAttemptSerializer(serializers.ModelSerializer):
    """Authentication attempt serializer"""
    is_successful = serializers.BooleanField(read_only=True)
    user_name = serializers.CharField(source='user.get_full_name', read_only=True)
    
    class Meta:
        model = AuthenticationAttempt
        fields = (
            'id', 'user', 'user_name', 'session_id', 'similarity_score',
            'liveness_score', 'quality_score', 'result', 'is_successful',
            'processing_time', 'obstacles_detected', 'created_at'
        )
        read_only_fields = ('id', 'created_at', 'is_successful', 'user_name')


class LivenessDetectionSerializer(serializers.ModelSerializer):
    """Liveness detection serializer"""
    
    class Meta:
        model = LivenessDetection
        fields = (
            'id', 'blinks_detected', 'frames_processed', 'valid_frames',
            'challenge_type', 'challenge_completed', 'liveness_score',
            'is_live', 'created_at'
        )
        read_only_fields = ('id', 'created_at')


class ObstacleDetectionSerializer(serializers.ModelSerializer):
    """Obstacle detection serializer"""
    
    class Meta:
        model = ObstacleDetection
        fields = (
            'id', 'glasses_detected', 'glasses_confidence', 'mask_detected',
            'mask_confidence', 'hat_detected', 'hat_confidence',
            'hand_covering', 'hand_confidence', 'has_obstacles',
            'obstacle_score', 'created_at'
        )
        read_only_fields = ('id', 'created_at')


class FrameDataSerializer(serializers.Serializer):
    """Serializer for frame data sent from client"""
    frame_data = serializers.CharField(help_text="Base64 encoded image data")
    session_token = serializers.CharField()
    timestamp = serializers.FloatField(required=False)
    device_info = serializers.JSONField(required=False, default=dict)
    
    def validate_frame_data(self, value):
        """Validate and decode base64 frame data"""
        try:
            # Remove data URL prefix if present
            if value.startswith('data:image'):
                value = value.split(',')[1]
            
            # Decode base64
            frame_bytes = base64.b64decode(value)
            
            # Basic validation - check if it's likely an image
            if len(frame_bytes) < 100:
                raise serializers.ValidationError("Frame data too small")
            
            return value
        except Exception as e:
            raise serializers.ValidationError(f"Invalid frame data: {str(e)}")


class EnrollmentRequestSerializer(serializers.Serializer):
    """Serializer for enrollment request"""
    device_info = serializers.JSONField(default=dict)
    target_samples = serializers.IntegerField(default=5, min_value=3, max_value=10)
    
    def create(self, validated_data):
        """Create new enrollment session"""
        user = self.context['request'].user
        
        # Generate session token
        session_token = str(uuid.uuid4())
        
        # Calculate expiry (30 minutes from now)
        from django.utils import timezone
        from datetime import timedelta
        expires_at = timezone.now() + timedelta(minutes=30)
        
        session = EnrollmentSession.objects.create(
            user=user,
            session_token=session_token,
            device_info=validated_data['device_info'],
            target_samples=validated_data['target_samples'],
            expires_at=expires_at,
            ip_address=self.context['request'].META.get('REMOTE_ADDR')
        )
        
        return session


class AuthenticationRequestSerializer(serializers.Serializer):
    """Serializer for authentication request"""
    email = serializers.EmailField(required=False)
    device_info = serializers.JSONField(default=dict)
    session_type = serializers.ChoiceField(
        choices=['verification', 'identification'],
        default='identification'
    )
    
    def validate(self, attrs):
        if attrs['session_type'] == 'verification' and not attrs.get('email'):
            raise serializers.ValidationError("Email required for verification mode")
        return attrs


class AuthenticationLogSerializer(serializers.ModelSerializer):
    """Authentication log serializer"""
    user_name = serializers.CharField(source='user.get_full_name', read_only=True)
    
    class Meta:
        model = AuthenticationLog
        fields = (
            'id', 'user', 'user_name', 'attempted_email', 'auth_method',
            'success', 'failure_reason', 'ip_address', 'location',
            'response_time', 'similarity_score', 'liveness_score',
            'quality_score', 'risk_score', 'risk_factors', 'created_at'
        )
        read_only_fields = ('id', 'created_at', 'user_name')


class SecurityAlertSerializer(serializers.ModelSerializer):
    """Security alert serializer"""
    user_name = serializers.CharField(source='user.get_full_name', read_only=True)
    
    class Meta:
        model = SecurityAlert
        fields = (
            'id', 'alert_type', 'severity', 'user', 'user_name', 'title',
            'description', 'acknowledged', 'resolved', 'created_at'
        )
        read_only_fields = ('id', 'created_at', 'user_name')


class UserDeviceSerializer(serializers.ModelSerializer):
    """User device serializer"""
    
    class Meta:
        model = UserDevice
        fields = (
            'id', 'device_id', 'device_name', 'device_type',
            'operating_system', 'browser', 'is_trusted',
            'last_ip', 'last_location', 'first_seen', 'last_seen',
            'login_count', 'is_active'
        )
        read_only_fields = (
            'id', 'first_seen', 'last_seen', 'login_count'
        )


class SystemStatusSerializer(serializers.Serializer):
    """System status serializer"""
    insightface_ready = serializers.BooleanField()
    chromadb_ready = serializers.BooleanField()
    faiss_embeddings = serializers.IntegerField()
    liveness_detector_ready = serializers.BooleanField()
    obstacle_detector_ready = serializers.BooleanField()
    total_users = serializers.IntegerField()
    total_embeddings = serializers.IntegerField()
    active_sessions = serializers.IntegerField()


class WebRTCSignalSerializer(serializers.Serializer):
    """WebRTC signaling serializer"""
    session_token = serializers.CharField()
    signal_type = serializers.ChoiceField(
        choices=['offer', 'answer', 'ice_candidate', 'ice_gathering']
    )
    signal_data = serializers.JSONField()
    
    def validate_session_token(self, value):
        """Validate session token exists and is active"""
        from streaming.models import StreamingSession
        
        try:
            session = StreamingSession.objects.get(
                session_token=value,
                status__in=['initiating', 'connecting', 'connected']
            )
            return value
        except StreamingSession.DoesNotExist:
            raise serializers.ValidationError("Invalid or expired session token")


class StreamingSessionSerializer(serializers.ModelSerializer):
    """Streaming session serializer"""
    
    class Meta:
        model = StreamingSession
        fields = (
            'id', 'session_token', 'session_type', 'status',
            'video_quality', 'frame_rate', 'bitrate',
            'created_at', 'connected_at', 'completed_at'
        )
        read_only_fields = (
            'id', 'session_token', 'created_at', 'connected_at', 'completed_at'
        )
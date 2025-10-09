"""
Recognition App Serializers with Swagger Documentation
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model
try:
    from drf_spectacular.utils import extend_schema_field
    from drf_spectacular.types import OpenApiTypes
except ImportError:
    def extend_schema_field(field_type):
        def decorator(func):
            return func
        return decorator
    
    class OpenApiTypes:
        FLOAT = float
        INT = int
        STR = str
        BOOL = bool

from recognition.models import (
    FaceEmbedding, EnrollmentSession, AuthenticationAttempt,
    LivenessDetection, ObstacleDetection
)

User = get_user_model()


class FaceEmbeddingSerializer(serializers.ModelSerializer):
    """
    Face Embedding Serializer
    
    Represents stored face embeddings with quality metrics
    and security validation information.
    """
    user_email = serializers.CharField(
        source='user.email', 
        read_only=True,
        help_text="Email of the user who owns this embedding"
    )
    
    enrollment_session_id = serializers.UUIDField(
        source='enrollment_session.id',
        read_only=True,
        help_text="ID of the enrollment session that created this embedding"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_quality_grade(self, obj):
        """Convert quality score to letter grade"""
        if obj.quality_score >= 0.9:
            return "A"
        elif obj.quality_score >= 0.8:
            return "B"
        elif obj.quality_score >= 0.7:
            return "C"
        elif obj.quality_score >= 0.6:
            return "D"
        else:
            return "F"
    
    quality_grade = serializers.SerializerMethodField(
        help_text="Quality grade (A-F) based on quality score"
    )
    
    @extend_schema_field(OpenApiTypes.BOOL)
    def get_is_high_quality(self, obj):
        """Check if embedding meets high quality standards"""
        return (obj.quality_score >= 0.8 and 
                obj.confidence_score >= 0.9 and
                obj.liveness_score >= 0.7)
    
    is_high_quality = serializers.SerializerMethodField(
        help_text="Whether the embedding meets high quality standards"
    )
    
    class Meta:
        model = FaceEmbedding
        fields = (
            'id', 'user_email', 'enrollment_session_id', 'quality_score',
            'confidence_score', 'face_bbox', 'face_landmarks', 'sample_number',
            'liveness_score', 'anti_spoofing_score', 'quality_grade',
            'is_high_quality', 'capture_device', 'capture_resolution',
            'is_active', 'is_verified', 'created_at', 'updated_at'
        )
        read_only_fields = (
            'id', 'user_email', 'enrollment_session_id', 'quality_grade',
            'is_high_quality', 'created_at', 'updated_at'
        )
        extra_kwargs = {
            'quality_score': {
                'help_text': 'Face image quality score (0.0-1.0, higher is better)'
            },
            'confidence_score': {
                'help_text': 'Face detection confidence (0.0-1.0, higher is better)'
            },
            'face_bbox': {
                'help_text': 'Face bounding box coordinates [x1, y1, x2, y2]'
            },
            'face_landmarks': {
                'help_text': 'Facial landmark coordinates for key points'
            },
            'sample_number': {
                'help_text': 'Sample number within the enrollment session (1-based)'
            },
            'liveness_score': {
                'help_text': 'Liveness detection score (0.0-1.0, higher indicates more likely to be live)'
            },
            'anti_spoofing_score': {
                'help_text': 'Anti-spoofing detection score (0.0-1.0, higher indicates less likely to be spoofed)'
            },
            'capture_device': {
                'help_text': 'Device used for face capture (camera model/type)'
            },
            'capture_resolution': {
                'help_text': 'Capture resolution in WxH format (e.g., "640x480")'
            }
        }


class EnrollmentSessionDetailSerializer(serializers.ModelSerializer):
    """
    Detailed Enrollment Session Serializer
    
    Comprehensive enrollment session information including
    progress tracking and embedded face samples.
    """
    user_email = serializers.CharField(
        source='user.email',
        read_only=True,
        help_text="Email of the user enrolling"
    )
    
    user_full_name = serializers.CharField(
        source='user.get_full_name',
        read_only=True,
        help_text="Full name of the user enrolling"
    )
    
    embeddings = FaceEmbeddingSerializer(
        many=True,
        read_only=True,
        help_text="Face embeddings collected during this session"
    )
    
    @extend_schema_field(OpenApiTypes.FLOAT)
    def get_progress_percentage(self, obj):
        """Calculate enrollment progress percentage"""
        if obj.target_samples == 0:
            return 0.0
        return min((obj.completed_samples / obj.target_samples) * 100, 100.0)
    
    progress_percentage = serializers.SerializerMethodField(
        help_text="Enrollment progress as percentage (0-100)"
    )
    
    @extend_schema_field(OpenApiTypes.BOOL)
    def get_is_expired(self, obj):
        """Check if session has expired"""
        from django.utils import timezone
        return timezone.now() > obj.expires_at
    
    is_expired = serializers.SerializerMethodField(
        help_text="Whether the enrollment session has expired"
    )
    
    @extend_schema_field(OpenApiTypes.FLOAT)
    def get_average_quality(self, obj):
        """Calculate average quality of collected samples"""
        embeddings = obj.embeddings.filter(is_active=True)
        if not embeddings.exists():
            return 0.0
        
        total_quality = sum(emb.quality_score for emb in embeddings)
        return total_quality / embeddings.count()
    
    average_quality = serializers.SerializerMethodField(
        help_text="Average quality score of collected face samples"
    )
    
    @extend_schema_field(OpenApiTypes.INT)
    def get_time_remaining_minutes(self, obj):
        """Calculate remaining time in minutes"""
        from django.utils import timezone
        if obj.expires_at <= timezone.now():
            return 0
        
        delta = obj.expires_at - timezone.now()
        return int(delta.total_seconds() / 60)
    
    time_remaining_minutes = serializers.SerializerMethodField(
        help_text="Remaining time for session completion in minutes"
    )
    
    class Meta:
        model = EnrollmentSession
        fields = (
            'id', 'user_email', 'user_full_name', 'session_token',
            'target_samples', 'completed_samples', 'progress_percentage',
            'average_quality', 'status', 'is_expired', 'time_remaining_minutes',
            'started_at', 'completed_at', 'expires_at', 'device_info',
            'failure_reason', 'embeddings', 'created_at', 'updated_at'
        )
        read_only_fields = (
            'id', 'user_email', 'user_full_name', 'session_token',
            'progress_percentage', 'average_quality', 'is_expired',
            'time_remaining_minutes', 'embeddings', 'started_at',
            'completed_at', 'created_at', 'updated_at'
        )
        extra_kwargs = {
            'target_samples': {
                'help_text': 'Target number of face samples to collect (typically 3-10)'
            },
            'completed_samples': {
                'help_text': 'Number of valid face samples collected so far'
            },
            'status': {
                'help_text': 'Current enrollment session status'
            },
            'expires_at': {
                'help_text': 'Session expiration timestamp (ISO 8601 format)'
            },
            'device_info': {
                'help_text': 'Information about the device used for enrollment'
            },
            'failure_reason': {
                'help_text': 'Reason for enrollment failure (if status is failed)'
            }
        }


class AuthenticationAttemptSerializer(serializers.ModelSerializer):
    """
    Authentication Attempt Serializer
    
    Detailed information about face authentication attempts
    including security metrics and result analysis.
    """
    user_email = serializers.CharField(
        source='user.email',
        read_only=True,
        help_text="Email of the user attempting authentication"
    )
    
    attempted_user_email = serializers.CharField(
        source='attempted_user.email',
        read_only=True,
        help_text="Email of the user matched during authentication (if any)"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_result_description(self, obj):
        """Get human-readable result description"""
        if obj.is_successful:
            return f"Successfully authenticated with {obj.confidence_score:.1%} confidence"
        else:
            reasons = []
            if obj.confidence_score < 0.7:
                reasons.append("low confidence")
            if obj.liveness_score < 0.5:
                reasons.append("failed liveness check")
            if obj.anti_spoofing_score < 0.7:
                reasons.append("potential spoofing detected")
            
            if not reasons:
                reasons.append("unknown reason")
            
            return f"Authentication failed: {', '.join(reasons)}"
    
    result_description = serializers.SerializerMethodField(
        help_text="Human-readable description of the authentication result"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_security_level(self, obj):
        """Determine security level of the authentication"""
        if not obj.is_successful:
            return "failed"
        
        # Calculate overall security score
        security_score = (
            obj.confidence_score * 0.4 +
            obj.liveness_score * 0.3 +
            obj.anti_spoofing_score * 0.3
        )
        
        if security_score >= 0.9:
            return "very_high"
        elif security_score >= 0.8:
            return "high"
        elif security_score >= 0.7:
            return "medium"
        else:
            return "low"
    
    security_level = serializers.SerializerMethodField(
        help_text="Overall security level of the authentication (very_high, high, medium, low, failed)"
    )
    
    @extend_schema_field(OpenApiTypes.INT)
    def get_processing_time_ms(self, obj):
        """Convert processing time to milliseconds"""
        return int(obj.processing_time * 1000) if obj.processing_time else 0
    
    processing_time_ms = serializers.SerializerMethodField(
        help_text="Processing time in milliseconds"
    )
    
    class Meta:
        model = AuthenticationAttempt
        fields = (
            'id', 'user_email', 'attempted_user_email', 'is_successful',
            'confidence_score', 'liveness_score', 'anti_spoofing_score',
            'face_bbox', 'processing_time', 'processing_time_ms',
            'result_description', 'security_level', 'device_info',
            'ip_address', 'user_agent', 'location_info', 'failure_reason',
            'created_at'
        )
        read_only_fields = (
            'id', 'user_email', 'attempted_user_email', 'result_description',
            'security_level', 'processing_time_ms', 'created_at'
        )
        extra_kwargs = {
            'is_successful': {
                'help_text': 'Whether the authentication was successful'
            },
            'confidence_score': {
                'help_text': 'Face matching confidence score (0.0-1.0)'
            },
            'liveness_score': {
                'help_text': 'Liveness detection score (0.0-1.0)'
            },
            'anti_spoofing_score': {
                'help_text': 'Anti-spoofing detection score (0.0-1.0)'
            },
            'face_bbox': {
                'help_text': 'Face bounding box coordinates [x1, y1, x2, y2]'
            },
            'processing_time': {
                'help_text': 'Processing time in seconds'
            },
            'device_info': {
                'help_text': 'Device information used for authentication'
            },
            'ip_address': {
                'help_text': 'IP address of the authentication request'
            },
            'user_agent': {
                'help_text': 'User agent string of the client'
            },
            'location_info': {
                'help_text': 'Geographic location information (if available)'
            },
            'failure_reason': {
                'help_text': 'Detailed reason for authentication failure'
            }
        }


class LivenessDetectionSerializer(serializers.ModelSerializer):
    """
    Liveness Detection Result Serializer
    
    Detailed liveness detection analysis results including
    multiple validation methods and confidence scores.
    """
    authentication_attempt_id = serializers.UUIDField(
        source='authentication_attempt.id',
        read_only=True,
        help_text="ID of the associated authentication attempt"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_liveness_verdict(self, obj):
        """Get overall liveness verdict"""
        if obj.overall_score >= 0.8:
            return "live"
        elif obj.overall_score >= 0.5:
            return "uncertain"
        else:
            return "not_live"
    
    liveness_verdict = serializers.SerializerMethodField(
        help_text="Overall liveness verdict (live, uncertain, not_live)"
    )
    
    @extend_schema_field(OpenApiTypes.BOOL)
    def get_passed_all_tests(self, obj):
        """Check if all liveness tests passed"""
        return all([
            obj.blink_detected,
            obj.head_movement_detected,
            obj.texture_analysis_score >= 0.7,
            obj.depth_analysis_score >= 0.7
        ])
    
    passed_all_tests = serializers.SerializerMethodField(
        help_text="Whether all liveness detection tests passed"
    )
    
    class Meta:
        model = LivenessDetection
        fields = (
            'id', 'authentication_attempt_id', 'overall_score',
            'liveness_verdict', 'blink_detected', 'blink_count',
            'head_movement_detected', 'head_movement_score',
            'texture_analysis_score', 'depth_analysis_score',
            'temporal_consistency_score', 'passed_all_tests',
            'processing_details', 'created_at'
        )
        read_only_fields = (
            'id', 'authentication_attempt_id', 'liveness_verdict',
            'passed_all_tests', 'created_at'
        )
        extra_kwargs = {
            'overall_score': {
                'help_text': 'Overall liveness confidence score (0.0-1.0)'
            },
            'blink_detected': {
                'help_text': 'Whether natural blinking was detected'
            },
            'blink_count': {
                'help_text': 'Number of blinks detected during analysis'
            },
            'head_movement_detected': {
                'help_text': 'Whether natural head movement was detected'
            },
            'head_movement_score': {
                'help_text': 'Head movement naturalness score (0.0-1.0)'
            },
            'texture_analysis_score': {
                'help_text': 'Facial texture analysis score (0.0-1.0)'
            },
            'depth_analysis_score': {
                'help_text': 'Depth perception analysis score (0.0-1.0)'
            },
            'temporal_consistency_score': {
                'help_text': 'Temporal consistency across frames (0.0-1.0)'
            },
            'processing_details': {
                'help_text': 'Detailed processing information and intermediate results'
            }
        }


class ObstacleDetectionSerializer(serializers.ModelSerializer):
    """
    Obstacle Detection Result Serializer
    
    Results from obstacle and occlusion detection analysis
    to ensure clear face visibility during authentication.
    """
    authentication_attempt_id = serializers.UUIDField(
        source='authentication_attempt.id',
        read_only=True,
        help_text="ID of the associated authentication attempt"
    )
    
    @extend_schema_field(OpenApiTypes.BOOL)
    def get_face_clearly_visible(self, obj):
        """Check if face is clearly visible without obstacles"""
        return (not obj.sunglasses_detected and
                not obj.mask_detected and
                not obj.hand_obstruction and
                obj.face_visibility_score >= 0.8)
    
    face_clearly_visible = serializers.SerializerMethodField(
        help_text="Whether the face is clearly visible without obstructions"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_visibility_status(self, obj):
        """Get overall visibility status"""
        if obj.face_visibility_score >= 0.9:
            return "excellent"
        elif obj.face_visibility_score >= 0.8:
            return "good"
        elif obj.face_visibility_score >= 0.6:
            return "fair"
        else:
            return "poor"
    
    visibility_status = serializers.SerializerMethodField(
        help_text="Overall face visibility status (excellent, good, fair, poor)"
    )
    
    class Meta:
        model = ObstacleDetection
        fields = (
            'id', 'authentication_attempt_id', 'sunglasses_detected',
            'sunglasses_confidence', 'mask_detected', 'mask_confidence',
            'hat_detected', 'hat_confidence', 'hand_obstruction',
            'hand_confidence', 'face_visibility_score', 'lighting_score',
            'face_clearly_visible', 'visibility_status', 'obstacle_details',
            'created_at'
        )
        read_only_fields = (
            'id', 'authentication_attempt_id', 'face_clearly_visible',
            'visibility_status', 'created_at'
        )
        extra_kwargs = {
            'sunglasses_detected': {
                'help_text': 'Whether sunglasses were detected on the face'
            },
            'sunglasses_confidence': {
                'help_text': 'Confidence score for sunglasses detection (0.0-1.0)'
            },
            'mask_detected': {
                'help_text': 'Whether a face mask was detected'
            },
            'mask_confidence': {
                'help_text': 'Confidence score for mask detection (0.0-1.0)'
            },
            'hat_detected': {
                'help_text': 'Whether a hat or head covering was detected'
            },
            'hat_confidence': {
                'help_text': 'Confidence score for hat detection (0.0-1.0)'
            },
            'hand_obstruction': {
                'help_text': 'Whether hands are obstructing the face'
            },
            'hand_confidence': {
                'help_text': 'Confidence score for hand obstruction detection (0.0-1.0)'
            },
            'face_visibility_score': {
                'help_text': 'Overall face visibility score (0.0-1.0)'
            },
            'lighting_score': {
                'help_text': 'Lighting quality score (0.0-1.0)'
            },
            'obstacle_details': {
                'help_text': 'Detailed information about detected obstacles and occlusions'
            }
        }


# Summary serializers for list views
class FaceEmbeddingSummarySerializer(serializers.ModelSerializer):
    """Simplified face embedding serializer for list views"""
    
    class Meta:
        model = FaceEmbedding
        fields = (
            'id', 'quality_score', 'confidence_score', 'sample_number',
            'is_active', 'is_verified', 'created_at'
        )


class EnrollmentSessionSummarySerializer(serializers.ModelSerializer):
    """Simplified enrollment session serializer for list views"""
    user_email = serializers.CharField(source='user.email', read_only=True)
    progress_percentage = serializers.SerializerMethodField()
    
    def get_progress_percentage(self, obj):
        if obj.target_samples == 0:
            return 0.0
        return min((obj.completed_samples / obj.target_samples) * 100, 100.0)
    
    class Meta:
        model = EnrollmentSession
        fields = (
            'id', 'user_email', 'target_samples', 'completed_samples',
            'progress_percentage', 'status', 'started_at', 'expires_at'
        )


class AuthenticationAttemptSummarySerializer(serializers.ModelSerializer):
    """Simplified authentication attempt serializer for list views"""
    user_email = serializers.CharField(source='user.email', read_only=True)
    
    class Meta:
        model = AuthenticationAttempt
        fields = (
            'id', 'user_email', 'is_successful', 'confidence_score',
            'liveness_score', 'anti_spoofing_score', 'created_at'
        )
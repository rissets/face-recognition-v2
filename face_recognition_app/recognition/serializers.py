"""
Recognition app serializers
"""
from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field
from drf_spectacular.types import OpenApiTypes
from .models import FaceEmbedding, EnrollmentSession, AuthenticationAttempt


class FaceEmbeddingSerializer(serializers.ModelSerializer):
    """Serializer for FaceEmbedding model"""
    
    class Meta:
        model = FaceEmbedding
        fields = [
            'id', 'client_user', 'embedding_hash', 'quality_score', 'confidence_score', 
            'face_bbox', 'face_landmarks', 'sample_number',
            'liveness_score', 'anti_spoofing_score', 'is_active', 
            'is_verified', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'client_user', 'embedding_hash', 'created_at', 'updated_at'
        ]


class EnrollmentSessionSerializer(serializers.ModelSerializer):
    """Serializer for EnrollmentSession model"""
    
    @extend_schema_field(OpenApiTypes.FLOAT)
    def get_progress_percentage(self, obj) -> float:
        """Calculate enrollment progress percentage"""
        if obj.target_samples > 0:
            return (obj.completed_samples / obj.target_samples) * 100
        return 0.0
    
    progress_percentage = serializers.SerializerMethodField()
    
    class Meta:
        model = EnrollmentSession
        fields = [
            'id', 'client_user', 'session_token', 'status', 'target_samples',
            'completed_samples', 'progress_percentage', 'average_quality', 
            'min_quality_threshold', 'device_info', 'ip_address', 'session_log',
            'error_messages', 'started_at', 'completed_at', 'expires_at'
        ]
        read_only_fields = [
            'id', 'client_user', 'session_token', 'progress_percentage', 
            'started_at', 'completed_at'
        ]


class AuthenticationAttemptSerializer(serializers.ModelSerializer):
    """Serializer for AuthenticationAttempt model"""
    
    @extend_schema_field(OpenApiTypes.BOOL)
    def get_is_successful(self, obj) -> bool:
        """Check if authentication attempt was successful"""
        return obj.result == 'success'
    
    is_successful = serializers.SerializerMethodField()
    
    class Meta:
        model = AuthenticationAttempt
        fields = [
            'id', 'client_user', 'session_id', 'result', 'is_successful',
            'similarity_score', 'liveness_score', 'quality_score',
            'obstacles_detected', 'processing_time', 'ip_address', 
            'user_agent', 'device_fingerprint', 'face_bbox', 
            'matched_embedding', 'metadata', 'created_at'
        ]
        read_only_fields = [
            'id', 'client_user', 'is_successful', 'processing_time', 'created_at'
        ]
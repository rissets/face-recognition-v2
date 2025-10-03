"""
Recognition app serializers
"""
from rest_framework import serializers
from .models import FaceEmbedding, EnrollmentSession, AuthenticationAttempt


class FaceEmbeddingSerializer(serializers.ModelSerializer):
    """Serializer for FaceEmbedding model"""
    
    class Meta:
        model = FaceEmbedding
        fields = [
            'id', 'user', 'quality_score', 'confidence_score', 
            'face_landmarks', 'face_area', 'detection_confidence',
            'embedding_version', 'is_active', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'user', 'embedding_version', 'created_at', 'updated_at'
        ]


class EnrollmentSessionSerializer(serializers.ModelSerializer):
    """Serializer for EnrollmentSession model"""
    
    class Meta:
        model = EnrollmentSession
        fields = [
            'id', 'user', 'session_id', 'status', 'frames_captured',
            'frames_processed', 'quality_score', 'enrollment_data',
            'error_message', 'created_at', 'updated_at', 'completed_at'
        ]
        read_only_fields = [
            'id', 'user', 'session_id', 'created_at', 'updated_at', 'completed_at'
        ]


class AuthenticationAttemptSerializer(serializers.ModelSerializer):
    """Serializer for AuthenticationAttempt model"""
    
    class Meta:
        model = AuthenticationAttempt
        fields = [
            'id', 'user', 'status', 'confidence_score', 'quality_score',
            'liveness_passed', 'obstacle_detected', 'processing_time',
            'ip_address', 'user_agent', 'device_info', 'error_message',
            'created_at'
        ]
        read_only_fields = [
            'id', 'user', 'processing_time', 'created_at'
        ]
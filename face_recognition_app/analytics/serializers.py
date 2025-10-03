"""
Analytics app serializers
"""
from rest_framework import serializers
from .models import AuthenticationLog, SecurityAlert, SystemMetrics, UserBehaviorAnalytics


class AuthenticationLogSerializer(serializers.ModelSerializer):
    """Serializer for AuthenticationLog model"""
    
    class Meta:
        model = AuthenticationLog
        fields = [
            'id', 'user', 'attempted_email', 'auth_method', 'success',
            'failure_reason', 'confidence_score', 'quality_score',
            'liveness_passed', 'processing_time', 'ip_address',
            'user_agent', 'device_info', 'timestamp'
        ]
        read_only_fields = [
            'id', 'user', 'processing_time', 'timestamp'
        ]


class SecurityAlertSerializer(serializers.ModelSerializer):
    """Serializer for SecurityAlert model"""
    
    class Meta:
        model = SecurityAlert
        fields = [
            'id', 'user', 'alert_type', 'severity', 'title', 'description',
            'triggered_by', 'ip_address', 'user_agent', 'additional_data',
            'is_resolved', 'resolved_at', 'resolved_by', 'created_at'
        ]
        read_only_fields = [
            'id', 'user', 'created_at'
        ]


class SystemMetricsSerializer(serializers.ModelSerializer):
    """Serializer for SystemMetrics model"""
    
    class Meta:
        model = SystemMetrics
        fields = [
            'id', 'metric_name', 'metric_type', 'metric_value',
            'metric_unit', 'description', 'tags', 'created_at'
        ]
        read_only_fields = [
            'id', 'created_at'
        ]


class UserBehaviorAnalyticsSerializer(serializers.ModelSerializer):
    """Serializer for UserBehaviorAnalytics model"""
    
    class Meta:
        model = UserBehaviorAnalytics
        fields = [
            'id', 'user', 'session_duration', 'pages_visited',
            'authentication_method_preference', 'device_preferences',
            'time_patterns', 'location_patterns', 'success_patterns',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'user', 'created_at', 'updated_at'
        ]
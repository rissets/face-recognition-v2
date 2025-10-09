"""
Analytics app serializers with comprehensive Swagger documentation
"""
from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field
from drf_spectacular.types import OpenApiTypes
from .models import AuthenticationLog, SecurityAlert, SystemMetrics, UserBehaviorAnalytics


class AuthenticationLogSerializer(serializers.ModelSerializer):
    """Comprehensive serializer for AuthenticationLog model with detailed field documentation"""
    
    user_email = serializers.CharField(source='user.email', read_only=True, help_text="User's email address")
    user_full_name = serializers.CharField(source='user.get_full_name', read_only=True, help_text="User's full name")
    
    class Meta:
        model = AuthenticationLog
        fields = [
            'id', 'user', 'user_email', 'user_full_name', 'attempted_email', 
            'auth_method', 'success', 'failure_reason', 'similarity_score', 
            'liveness_score', 'quality_score', 'response_time', 'ip_address', 
            'user_agent', 'device_fingerprint', 'location', 'risk_score', 
            'risk_factors', 'session_id', 'timestamp', 'created_at'
        ]
        read_only_fields = [
            'id', 'user', 'user_email', 'user_full_name', 'response_time', 
            'timestamp', 'created_at'
        ]
        extra_kwargs = {
            'id': {'help_text': 'Unique identifier for the authentication log'},
            'user': {'help_text': 'User who attempted authentication'},
            'attempted_email': {'help_text': 'Email used in authentication attempt'},
            'auth_method': {'help_text': 'Authentication method used (face, password, 2fa, social)'},
            'success': {'help_text': 'Whether the authentication was successful'},
            'failure_reason': {'help_text': 'Reason for authentication failure'},
            'similarity_score': {'help_text': 'Face similarity score (0.0-1.0)'},
            'liveness_score': {'help_text': 'Liveness detection score (0.0-1.0)'},
            'quality_score': {'help_text': 'Image quality score (0.0-1.0)'},
            'response_time': {'help_text': 'Authentication response time in milliseconds'},
            'ip_address': {'help_text': 'IP address of the authentication attempt'},
            'user_agent': {'help_text': 'Browser/client user agent string'},
            'device_fingerprint': {'help_text': 'Unique device fingerprint'},
            'location': {'help_text': 'Geographic location (City, Country)'},
            'risk_score': {'help_text': 'Calculated risk score (0.0-1.0)'},
            'risk_factors': {'help_text': 'List of identified risk factors'},
            'session_id': {'help_text': 'Associated session identifier'},
            'timestamp': {'help_text': 'Timestamp when the authentication occurred'},
        }


class SecurityAlertSerializer(serializers.ModelSerializer):
    """Comprehensive serializer for SecurityAlert model with detailed field documentation"""
    
    user_email = serializers.CharField(source='user.email', read_only=True, help_text="User's email address")
    acknowledged_by_name = serializers.CharField(source='acknowledged_by.get_full_name', read_only=True, help_text="Name of person who acknowledged alert")
    resolved_by_name = serializers.CharField(source='resolved_by.get_full_name', read_only=True, help_text="Name of person who resolved alert")
    
    class Meta:
        model = SecurityAlert
        fields = [
            'id', 'user', 'user_email', 'alert_type', 'severity', 'title', 
            'description', 'ip_address', 'context_data', 'acknowledged', 
            'acknowledged_by', 'acknowledged_by_name', 'acknowledged_at', 
            'resolved', 'resolved_by', 'resolved_by_name', 'resolved_at', 
            'resolution_notes', 'created_at'
        ]
        read_only_fields = [
            'id', 'user', 'user_email', 'acknowledged_by_name', 'resolved_by_name', 'created_at'
        ]
        extra_kwargs = {
            'id': {'help_text': 'Unique identifier for the security alert'},
            'user': {'help_text': 'User associated with this alert'},
            'alert_type': {'help_text': 'Type of security alert'},
            'severity': {'help_text': 'Alert severity level (low, medium, high, critical)'},
            'title': {'help_text': 'Alert title/summary'},
            'description': {'help_text': 'Detailed alert description'},
            'ip_address': {'help_text': 'IP address related to the alert'},
            'context_data': {'help_text': 'Additional context data in JSON format'},
            'acknowledged': {'help_text': 'Whether the alert has been acknowledged'},
            'acknowledged_by': {'help_text': 'User who acknowledged the alert'},
            'acknowledged_at': {'help_text': 'Timestamp when alert was acknowledged'},
            'resolved': {'help_text': 'Whether the alert has been resolved'},
            'resolved_by': {'help_text': 'User who resolved the alert'},
            'resolved_at': {'help_text': 'Timestamp when alert was resolved'},
            'resolution_notes': {'help_text': 'Notes about how the alert was resolved'},
        }


class SystemMetricsSerializer(serializers.ModelSerializer):
    """Comprehensive serializer for SystemMetrics model with detailed field documentation"""
    
    class Meta:
        model = SystemMetrics
        fields = [
            'id', 'metric_name', 'metric_type', 'value', 'unit', 'tags', 'timestamp'
        ]
        read_only_fields = [
            'id', 'timestamp'
        ]
        extra_kwargs = {
            'id': {'help_text': 'Unique identifier for the system metric'},
            'metric_name': {'help_text': 'Name of the metric being measured'},
            'metric_type': {'help_text': 'Type of metric (counter, gauge, histogram, timer)'},
            'value': {'help_text': 'Numeric value of the metric'},
            'unit': {'help_text': 'Unit of measurement (ms, %, count, etc.)'},
            'tags': {'help_text': 'Additional metadata tags in JSON format'},
            'timestamp': {'help_text': 'Timestamp when the metric was recorded'},
        }


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


# Response Serializers for Swagger Documentation

class AuthStatsSerializer(serializers.Serializer):
    """Authentication statistics serializer"""
    total_attempts = serializers.IntegerField(help_text="Total authentication attempts")
    successful_attempts = serializers.IntegerField(help_text="Number of successful attempts")
    avg_similarity = serializers.FloatField(help_text="Average similarity score", allow_null=True)
    avg_liveness = serializers.FloatField(help_text="Average liveness score", allow_null=True)
    avg_quality = serializers.FloatField(help_text="Average quality score", allow_null=True)
    success_rate = serializers.FloatField(help_text="Success rate percentage")


class PerformanceMetricsSerializer(serializers.Serializer):
    """Performance metrics serializer"""
    avg_response_time = serializers.FloatField(help_text="Average response time in milliseconds", allow_null=True)
    min_response_time = serializers.FloatField(help_text="Minimum response time in milliseconds", allow_null=True)
    max_response_time = serializers.FloatField(help_text="Maximum response time in milliseconds", allow_null=True)


class RiskAnalysisSerializer(serializers.Serializer):
    """Risk analysis serializer"""
    avg_risk_score = serializers.FloatField(help_text="Average risk score", allow_null=True)
    high_risk_attempts = serializers.IntegerField(help_text="Number of high risk attempts")


class DateRangeSerializer(serializers.Serializer):
    """Date range serializer"""
    from_date = serializers.DateField(source='from', help_text="Start date of the period")
    to_date = serializers.DateField(source='to', help_text="End date of the period")


class RecentActivitySerializer(serializers.Serializer):
    """Recent activity serializer"""
    timestamp = serializers.DateTimeField(help_text="Activity timestamp")
    success = serializers.BooleanField(help_text="Whether the attempt was successful")
    auth_method = serializers.CharField(help_text="Authentication method used")
    failure_reason = serializers.CharField(help_text="Reason for failure if any", allow_blank=True)


class DashboardResponseSerializer(serializers.Serializer):
    """Dashboard response serializer for comprehensive analytics data"""
    auth_stats = AuthStatsSerializer(help_text="Authentication statistics")
    alert_count = serializers.IntegerField(help_text="Number of security alerts in the period")
    recent_activities = RecentActivitySerializer(many=True, help_text="Recent authentication activities")
    performance_metrics = PerformanceMetricsSerializer(help_text="System performance metrics")
    risk_analysis = RiskAnalysisSerializer(help_text="Risk analysis data")
    period = serializers.CharField(help_text="Time period for the analytics")
    date_range = DateRangeSerializer(help_text="Date range of the analytics period")


class AvgScoresSerializer(serializers.Serializer):
    """Average scores serializer"""
    similarity = serializers.FloatField(help_text="Average similarity score")
    liveness = serializers.FloatField(help_text="Average liveness score")
    quality = serializers.FloatField(help_text="Average quality score")


class SecurityStatusSerializer(serializers.Serializer):
    """Security status serializer"""
    active_alerts = serializers.IntegerField(help_text="Number of active security alerts")
    risk_level = serializers.CharField(help_text="Current risk level (low, medium, high)")
    last_security_scan = serializers.DateTimeField(help_text="Last security scan timestamp")


class StatisticsResponseSerializer(serializers.Serializer):
    """Statistics response serializer for comprehensive user statistics"""
    total_attempts = serializers.IntegerField(help_text="Total authentication attempts")
    successful_attempts = serializers.IntegerField(help_text="Number of successful attempts")
    failed_attempts = serializers.IntegerField(help_text="Number of failed attempts")
    success_rate = serializers.FloatField(help_text="Success rate percentage")
    face_enrolled = serializers.BooleanField(help_text="Whether user has enrolled face biometrics")
    last_login = serializers.DateTimeField(help_text="Last login timestamp", allow_null=True)
    account_created = serializers.DateTimeField(help_text="Account creation timestamp")
    authentication_methods = serializers.ListField(
        child=serializers.CharField(),
        help_text="List of authentication methods used"
    )
    avg_scores = AvgScoresSerializer(help_text="Average biometric scores")
    security_status = SecurityStatusSerializer(help_text="Current security status")


class ErrorResponseSerializer(serializers.Serializer):
    """Standard error response serializer"""
    error = serializers.CharField(help_text="Error message")
    code = serializers.CharField(help_text="Error code", required=False)
    details = serializers.DictField(help_text="Additional error details", required=False)
"""
Analytics App Serializers with Swagger Documentation
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta

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
        OBJECT = dict

from analytics.models import AuthenticationLog, SecurityAlert

User = get_user_model()


class AuthenticationLogSerializer(serializers.ModelSerializer):
    """
    Authentication Log Serializer
    
    Comprehensive logging information for authentication events
    including security context, device information, and risk assessment.
    """
    user_email = serializers.CharField(
        source='user.email',
        read_only=True,
        help_text="Email of the user who attempted authentication"
    )
    
    user_full_name = serializers.CharField(
        source='user.get_full_name',
        read_only=True,
        help_text="Full name of the user who attempted authentication"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_event_type_display(self, obj):
        """Get human-readable event type"""
        event_types = {
            'login_attempt': 'Login Attempt',
            'login_success': 'Successful Login',
            'login_failure': 'Failed Login',
            'logout': 'Logout',
            'enrollment_start': 'Enrollment Started',
            'enrollment_complete': 'Enrollment Completed',
            'enrollment_failed': 'Enrollment Failed',
            'face_auth_success': 'Face Authentication Success',
            'face_auth_failure': 'Face Authentication Failure',
            'security_alert': 'Security Alert',
            'suspicious_activity': 'Suspicious Activity'
        }
        return event_types.get(obj.event_type, obj.event_type.title())
    
    event_type_display = serializers.SerializerMethodField(
        help_text="Human-readable event type description"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_risk_level(self, obj):
        """Calculate risk level based on various factors"""
        risk_score = 0
        
        # Factor in event type
        high_risk_events = ['login_failure', 'face_auth_failure', 'security_alert', 'suspicious_activity']
        if obj.event_type in high_risk_events:
            risk_score += 3
        
        # Factor in geolocation changes
        if obj.location_info and 'unusual_location' in obj.location_info:
            risk_score += 2
        
        # Factor in device changes
        if obj.metadata and obj.metadata.get('new_device'):
            risk_score += 1
        
        # Factor in time-based patterns
        if obj.metadata and obj.metadata.get('unusual_time'):
            risk_score += 1
        
        if risk_score >= 5:
            return "critical"
        elif risk_score >= 3:
            return "high"
        elif risk_score >= 1:
            return "medium"
        else:
            return "low"
    
    risk_level = serializers.SerializerMethodField(
        help_text="Calculated risk level (low, medium, high, critical)"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_time_ago(self, obj):
        """Get human-readable time since event"""
        delta = timezone.now() - obj.timestamp
        
        if delta.days > 0:
            return f"{delta.days} days ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hours ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "Just now"
    
    time_ago = serializers.SerializerMethodField(
        help_text="Human-readable time since the event occurred"
    )
    
    @extend_schema_field(OpenApiTypes.BOOL)
    def get_requires_attention(self, obj):
        """Check if event requires security attention"""
        attention_events = [
            'login_failure', 'face_auth_failure', 'security_alert',
            'suspicious_activity', 'enrollment_failed'
        ]
        
        if obj.event_type in attention_events:
            return True
        
        # Check for repeated failures
        if obj.event_type in ['login_failure', 'face_auth_failure']:
            recent_failures = AuthenticationLog.objects.filter(
                user=obj.user,
                event_type=obj.event_type,
                timestamp__gte=timezone.now() - timedelta(minutes=30)
            ).count()
            
            if recent_failures >= 3:
                return True
        
        return False
    
    requires_attention = serializers.SerializerMethodField(
        help_text="Whether this event requires security attention"
    )
    
    class Meta:
        model = AuthenticationLog
        fields = (
            'id', 'user_email', 'user_full_name', 'event_type',
            'event_type_display', 'success', 'ip_address', 'user_agent',
            'device_info', 'location_info', 'session_id', 'risk_level',
            'time_ago', 'requires_attention', 'metadata', 'timestamp'
        )
        read_only_fields = (
            'id', 'user_email', 'user_full_name', 'event_type_display',
            'risk_level', 'time_ago', 'requires_attention', 'timestamp'
        )
        extra_kwargs = {
            'event_type': {
                'help_text': 'Type of authentication or security event'
            },
            'success': {
                'help_text': 'Whether the authentication event was successful'
            },
            'ip_address': {
                'help_text': 'IP address from which the event originated'
            },
            'user_agent': {
                'help_text': 'User agent string of the client'
            },
            'device_info': {
                'help_text': 'Detailed device information and fingerprinting data'
            },
            'location_info': {
                'help_text': 'Geographic location information (if available)'
            },
            'session_id': {
                'help_text': 'Session identifier for tracking related events'
            },
            'metadata': {
                'help_text': 'Additional event-specific metadata and context'
            }
        }


class SecurityAlertSerializer(serializers.ModelSerializer):
    """
    Security Alert Serializer
    
    Comprehensive security alert information including threat assessment,
    recommended actions, and alert lifecycle management.
    """
    user_email = serializers.CharField(
        source='user.email',
        read_only=True,
        help_text="Email of the user associated with this alert (if any)"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_alert_type_display(self, obj):
        """Get human-readable alert type"""
        alert_types = {
            'failed_authentication': 'Failed Authentication Attempt',
            'suspicious_login': 'Suspicious Login Activity',
            'multiple_failures': 'Multiple Authentication Failures',
            'unusual_location': 'Login from Unusual Location',
            'new_device': 'Login from New Device',
            'face_spoofing_detected': 'Face Spoofing Attempt Detected',
            'liveness_check_failed': 'Liveness Check Failed',
            'enrollment_anomaly': 'Enrollment Process Anomaly',
            'system_intrusion': 'System Intrusion Attempt',
            'data_breach_attempt': 'Data Breach Attempt',
            'rate_limit_exceeded': 'Rate Limit Exceeded',
            'malicious_payload': 'Malicious Payload Detected'
        }
        return alert_types.get(obj.alert_type, obj.alert_type.replace('_', ' ').title())
    
    alert_type_display = serializers.SerializerMethodField(
        help_text="Human-readable alert type description"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_severity_display(self, obj):
        """Get human-readable severity level"""
        severity_map = {
            'low': 'Low Priority',
            'medium': 'Medium Priority',
            'high': 'High Priority',
            'critical': 'Critical Priority'
        }
        return severity_map.get(obj.severity, obj.severity.title())
    
    severity_display = serializers.SerializerMethodField(
        help_text="Human-readable severity level"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_status_display(self, obj):
        """Get human-readable status"""
        status_map = {
            'open': 'Open - Requires Investigation',
            'investigating': 'Under Investigation',
            'resolved': 'Resolved',
            'false_positive': 'False Positive',
            'suppressed': 'Suppressed'
        }
        return status_map.get(obj.status, obj.status.title())
    
    status_display = serializers.SerializerMethodField(
        help_text="Human-readable alert status"
    )
    
    @extend_schema_field(OpenApiTypes.OBJECT)
    def get_recommended_actions(self, obj):
        """Get recommended actions based on alert type and severity"""
        actions = []
        
        if obj.alert_type == 'failed_authentication' and obj.severity in ['high', 'critical']:
            actions.extend([
                "Review user account for compromise",
                "Consider temporary account suspension",
                "Verify user identity through alternate means"
            ])
        
        if obj.alert_type == 'suspicious_login':
            actions.extend([
                "Contact user to verify login attempt",
                "Review device and location information",
                "Enable additional security measures"
            ])
        
        if obj.alert_type == 'face_spoofing_detected':
            actions.extend([
                "Immediately suspend face authentication",
                "Investigate spoofing attempt details",
                "Review and strengthen anti-spoofing measures"
            ])
        
        if obj.alert_type == 'system_intrusion':
            actions.extend([
                "Immediate system security review",
                "Check for data compromise",
                "Implement additional security controls"
            ])
        
        if not actions:
            actions = ["Review alert details and determine appropriate response"]
        
        return {
            "immediate": actions[:2] if len(actions) > 2 else actions,
            "follow_up": actions[2:] if len(actions) > 2 else [],
            "priority": "high" if obj.severity in ['high', 'critical'] else "normal"
        }
    
    recommended_actions = serializers.SerializerMethodField(
        help_text="Recommended actions based on alert type and severity"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_time_since_created(self, obj):
        """Get time since alert was created"""
        delta = timezone.now() - obj.created_at
        
        if delta.days > 0:
            return f"{delta.days} days ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hours ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "Just now"
    
    time_since_created = serializers.SerializerMethodField(
        help_text="Time since the alert was created"
    )
    
    @extend_schema_field(OpenApiTypes.BOOL)
    def get_is_overdue(self, obj):
        """Check if alert response is overdue"""
        if obj.status in ['resolved', 'false_positive', 'suppressed']:
            return False
        
        # Define SLA based on severity
        sla_hours = {
            'critical': 1,
            'high': 4,
            'medium': 24,
            'low': 72
        }
        
        max_hours = sla_hours.get(obj.severity, 24)
        deadline = obj.created_at + timedelta(hours=max_hours)
        
        return timezone.now() > deadline
    
    is_overdue = serializers.SerializerMethodField(
        help_text="Whether the alert response is overdue based on SLA"
    )
    
    class Meta:
        model = SecurityAlert
        fields = (
            'id', 'user_email', 'alert_type', 'alert_type_display',
            'severity', 'severity_display', 'status', 'status_display',
            'title', 'description', 'recommended_actions', 'time_since_created',
            'is_overdue', 'source_ip', 'user_agent', 'additional_data',
            'resolved_at', 'resolved_by', 'resolution_notes', 'created_at', 'updated_at'
        )
        read_only_fields = (
            'id', 'user_email', 'alert_type_display', 'severity_display',
            'status_display', 'recommended_actions', 'time_since_created',
            'is_overdue', 'created_at', 'updated_at'
        )
        extra_kwargs = {
            'alert_type': {
                'help_text': 'Type of security alert or threat detected'
            },
            'severity': {
                'help_text': 'Severity level of the security alert'
            },
            'status': {
                'help_text': 'Current status of the alert investigation'
            },
            'title': {
                'help_text': 'Brief title describing the security alert'
            },
            'description': {
                'help_text': 'Detailed description of the security alert and context'
            },
            'source_ip': {
                'help_text': 'IP address associated with the security event'
            },
            'user_agent': {
                'help_text': 'User agent string associated with the event'
            },
            'additional_data': {
                'help_text': 'Additional metadata and context about the alert'
            },
            'resolved_at': {
                'help_text': 'Timestamp when the alert was resolved'
            },
            'resolved_by': {
                'help_text': 'Administrator who resolved the alert'
            },
            'resolution_notes': {
                'help_text': 'Notes about how the alert was resolved'
            }
        }


class AnalyticsDashboardSerializer(serializers.Serializer):
    """
    Analytics Dashboard Serializer
    
    Comprehensive dashboard data including system metrics,
    security statistics, and performance indicators.
    """
    period = serializers.CharField(
        help_text="Time period for the analytics data"
    )
    
    start_date = serializers.DateTimeField(
        help_text="Start date of the analytics period"
    )
    
    end_date = serializers.DateTimeField(
        help_text="End date of the analytics period"
    )
    
    # User metrics
    total_users = serializers.IntegerField(
        help_text="Total number of registered users"
    )
    
    enrolled_users = serializers.IntegerField(
        help_text="Number of users with completed face enrollment"
    )
    
    enrollment_rate = serializers.FloatField(
        help_text="Percentage of users who have completed enrollment"
    )
    
    # Authentication metrics
    total_authentications = serializers.IntegerField(
        help_text="Total number of authentication attempts"
    )
    
    successful_authentications = serializers.IntegerField(
        help_text="Number of successful authentication attempts"
    )
    
    authentication_success_rate = serializers.FloatField(
        help_text="Percentage of successful authentications"
    )
    
    average_confidence_score = serializers.FloatField(
        help_text="Average confidence score for successful authentications"
    )
    
    # Security metrics
    security_alerts = serializers.IntegerField(
        help_text="Number of security alerts generated"
    )
    
    critical_alerts = serializers.IntegerField(
        help_text="Number of critical security alerts"
    )
    
    resolved_alerts = serializers.IntegerField(
        help_text="Number of resolved security alerts"
    )
    
    # Performance metrics
    average_processing_time = serializers.FloatField(
        help_text="Average authentication processing time in milliseconds"
    )
    
    system_uptime = serializers.FloatField(
        help_text="System uptime percentage"
    )
    
    # Trend data
    daily_stats = serializers.JSONField(
        help_text="Daily statistics for trend analysis"
    )
    
    user_activity = serializers.JSONField(
        help_text="User activity patterns and distribution"
    )
    
    device_analytics = serializers.JSONField(
        help_text="Device and platform usage statistics"
    )
    
    geographic_distribution = serializers.JSONField(
        help_text="Geographic distribution of authentication attempts"
    )


class StatisticsSerializer(serializers.Serializer):
    """
    System Statistics Serializer
    
    Detailed system statistics and performance metrics
    for comprehensive system monitoring and analysis.
    """
    timestamp = serializers.DateTimeField(
        help_text="Timestamp when statistics were calculated"
    )
    
    # System health
    system_status = serializers.CharField(
        help_text="Overall system health status"
    )
    
    api_response_time = serializers.FloatField(
        help_text="Average API response time in milliseconds"
    )
    
    database_performance = serializers.JSONField(
        help_text="Database performance metrics"
    )
    
    # User statistics
    user_statistics = serializers.JSONField(
        help_text="Comprehensive user-related statistics"
    )
    
    # Authentication statistics
    authentication_statistics = serializers.JSONField(
        help_text="Detailed authentication performance and security metrics"
    )
    
    # Security statistics
    security_statistics = serializers.JSONField(
        help_text="Security-related statistics and threat analysis"
    )
    
    # Performance statistics
    performance_statistics = serializers.JSONField(
        help_text="System performance and resource utilization metrics"
    )
    
    # Quality metrics
    quality_metrics = serializers.JSONField(
        help_text="Face recognition quality and accuracy metrics"
    )


# Summary serializers for list views
class AuthenticationLogSummarySerializer(serializers.ModelSerializer):
    """Simplified authentication log serializer for list views"""
    user_email = serializers.CharField(source='user.email', read_only=True)
    event_type_display = serializers.SerializerMethodField()
    time_ago = serializers.SerializerMethodField()
    
    def get_event_type_display(self, obj):
        return obj.event_type.replace('_', ' ').title()
    
    def get_time_ago(self, obj):
        delta = timezone.now() - obj.timestamp
        if delta.days > 0:
            return f"{delta.days} days ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hours ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "Just now"
    
    class Meta:
        model = AuthenticationLog
        fields = (
            'id', 'user_email', 'event_type', 'event_type_display',
            'success', 'ip_address', 'time_ago', 'timestamp'
        )


class SecurityAlertSummarySerializer(serializers.ModelSerializer):
    """Simplified security alert serializer for list views"""
    user_email = serializers.CharField(source='user.email', read_only=True)
    alert_type_display = serializers.SerializerMethodField()
    time_since_created = serializers.SerializerMethodField()
    
    def get_alert_type_display(self, obj):
        return obj.alert_type.replace('_', ' ').title()
    
    def get_time_since_created(self, obj):
        delta = timezone.now() - obj.created_at
        if delta.days > 0:
            return f"{delta.days} days ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hours ago"
        else:
            return "Recently"
    
    class Meta:
        model = SecurityAlert
        fields = (
            'id', 'user_email', 'alert_type', 'alert_type_display',
            'severity', 'status', 'title', 'time_since_created', 'created_at'
        )
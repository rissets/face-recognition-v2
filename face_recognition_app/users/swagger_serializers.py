"""
Users App Serializers with Swagger Documentation
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model
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

from users.models import UserProfile, UserDevice

User = get_user_model()


class UserDeviceSerializer(serializers.ModelSerializer):
    """
    User Device Serializer
    
    Device information and fingerprinting data for security
    tracking and authentication context.
    """
    user_email = serializers.CharField(
        source='user.email',
        read_only=True,
        help_text="Email of the user who owns this device"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_device_type_display(self, obj):
        """Get human-readable device type"""
        device_types = {
            'mobile': 'Mobile Device',
            'tablet': 'Tablet',
            'desktop': 'Desktop Computer',
            'laptop': 'Laptop Computer',
            'unknown': 'Unknown Device'
        }
        return device_types.get(obj.device_type, 'Unknown Device')
    
    device_type_display = serializers.SerializerMethodField(
        help_text="Human-readable device type"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_trust_level(self, obj):
        """Calculate device trust level"""
        score = 0
        
        # Factor in device age
        if obj.first_seen:
            age_days = (timezone.now() - obj.first_seen).days
            if age_days > 30:
                score += 2
            elif age_days > 7:
                score += 1
        
        # Factor in usage frequency
        if obj.last_used and obj.first_seen:
            usage_frequency = (timezone.now() - obj.last_used).days
            if usage_frequency < 7:
                score += 2
            elif usage_frequency < 30:
                score += 1
        
        # Factor in verification status
        if obj.is_verified:
            score += 2
        
        # Factor in successful authentications
        if obj.device_fingerprint and 'successful_auths' in obj.device_fingerprint:
            auths = obj.device_fingerprint.get('successful_auths', 0)
            if auths > 10:
                score += 2
            elif auths > 5:
                score += 1
        
        if score >= 6:
            return "high"
        elif score >= 4:
            return "medium"
        elif score >= 2:
            return "low"
        else:
            return "unknown"
    
    trust_level = serializers.SerializerMethodField(
        help_text="Calculated device trust level (high, medium, low, unknown)"
    )
    
    @extend_schema_field(OpenApiTypes.BOOL)
    def get_is_current_device(self, obj):
        """Check if this is the current request device"""
        request = self.context.get('request')
        if not request:
            return False
        
        current_user_agent = request.META.get('HTTP_USER_AGENT', '')
        return obj.device_name in current_user_agent if obj.device_name else False
    
    is_current_device = serializers.SerializerMethodField(
        help_text="Whether this is the device making the current request"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_time_since_last_used(self, obj):
        """Get time since device was last used"""
        if not obj.last_used:
            return "Never"
        
        delta = timezone.now() - obj.last_used
        
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
    
    time_since_last_used = serializers.SerializerMethodField(
        help_text="Time since the device was last used"
    )
    
    @extend_schema_field(OpenApiTypes.OBJECT)
    def get_security_assessment(self, obj):
        """Get comprehensive security assessment"""
        assessment = {
            "trust_score": 0,
            "risk_factors": [],
            "recommendations": []
        }
        
        # Calculate trust score (0-100)
        trust_level = self.get_trust_level(obj)
        trust_scores = {"high": 90, "medium": 70, "low": 40, "unknown": 20}
        assessment["trust_score"] = trust_scores.get(trust_level, 20)
        
        # Identify risk factors
        if not obj.is_verified:
            assessment["risk_factors"].append("Device not verified")
            assessment["recommendations"].append("Verify device through additional authentication")
        
        if obj.last_used and (timezone.now() - obj.last_used).days > 90:
            assessment["risk_factors"].append("Device not used recently")
            assessment["recommendations"].append("Re-verify device identity")
        
        if obj.device_fingerprint and obj.device_fingerprint.get('failed_auths', 0) > 3:
            assessment["risk_factors"].append("Multiple authentication failures")
            assessment["recommendations"].append("Monitor device for suspicious activity")
        
        # Add recommendations based on device type
        if obj.device_type == 'mobile' and not obj.device_fingerprint.get('biometric_enabled'):
            assessment["recommendations"].append("Enable biometric authentication for enhanced security")
        
        return assessment
    
    security_assessment = serializers.SerializerMethodField(
        help_text="Comprehensive security assessment of the device"
    )
    
    class Meta:
        model = UserDevice
        fields = (
            'id', 'user_email', 'device_id', 'device_name', 'device_type',
            'device_type_display', 'operating_system', 'browser_info',
            'trust_level', 'is_current_device', 'time_since_last_used',
            'security_assessment', 'device_fingerprint', 'is_verified',
            'is_blocked', 'location_info', 'first_seen', 'last_used',
            'created_at', 'updated_at'
        )
        read_only_fields = (
            'id', 'user_email', 'device_type_display', 'trust_level',
            'is_current_device', 'time_since_last_used', 'security_assessment',
            'first_seen', 'last_used', 'created_at', 'updated_at'
        )
        extra_kwargs = {
            'device_id': {
                'help_text': 'Unique device identifier or fingerprint'
            },
            'device_name': {
                'help_text': 'Human-readable device name or model'
            },
            'device_type': {
                'help_text': 'Type of device (mobile, tablet, desktop, laptop)'
            },
            'operating_system': {
                'help_text': 'Operating system information'
            },
            'browser_info': {
                'help_text': 'Browser type and version information'
            },
            'device_fingerprint': {
                'help_text': 'Detailed device fingerprinting data for security analysis'
            },
            'is_verified': {
                'help_text': 'Whether the device has been verified by the user'
            },
            'is_blocked': {
                'help_text': 'Whether the device has been blocked for security reasons'
            },
            'location_info': {
                'help_text': 'Geographic location information when device was registered'
            }
        }


class UserProfileDetailSerializer(serializers.ModelSerializer):
    """
    Detailed User Profile Serializer
    
    Comprehensive user profile information including statistics,
    security settings, and activity metrics.
    """
    user_email = serializers.CharField(
        source='user.email',
        read_only=True,
        help_text="User's email address"
    )
    
    user_full_name = serializers.CharField(
        source='user.get_full_name',
        read_only=True,
        help_text="User's full name"
    )
    
    @extend_schema_field(OpenApiTypes.INT)
    def get_total_authentications(self, obj):
        """Get total authentication attempts"""
        return obj.user.authentication_attempts.count()
    
    total_authentications = serializers.SerializerMethodField(
        help_text="Total number of authentication attempts"
    )
    
    @extend_schema_field(OpenApiTypes.INT)
    def get_successful_authentications(self, obj):
        """Get successful authentication count"""
        return obj.user.authentication_attempts.filter(is_successful=True).count()
    
    successful_authentications = serializers.SerializerMethodField(
        help_text="Number of successful authentications"
    )
    
    @extend_schema_field(OpenApiTypes.FLOAT)
    def get_success_rate(self, obj):
        """Calculate authentication success rate"""
        total = self.get_total_authentications(obj)
        if total == 0:
            return 0.0
        
        successful = self.get_successful_authentications(obj)
        return round((successful / total) * 100, 1)
    
    success_rate = serializers.SerializerMethodField(
        help_text="Authentication success rate percentage"
    )
    
    @extend_schema_field(OpenApiTypes.INT)
    def get_enrolled_faces(self, obj):
        """Get number of enrolled face embeddings"""
        return obj.user.face_embeddings.filter(is_active=True).count()
    
    enrolled_faces = serializers.SerializerMethodField(
        help_text="Number of active face embeddings"
    )
    
    @extend_schema_field(OpenApiTypes.FLOAT)
    def get_average_confidence(self, obj):
        """Get average confidence score"""
        attempts = obj.user.authentication_attempts.filter(is_successful=True)
        if not attempts.exists():
            return 0.0
        
        total_confidence = sum(attempt.confidence_score for attempt in attempts)
        return round(total_confidence / attempts.count(), 3)
    
    average_confidence = serializers.SerializerMethodField(
        help_text="Average confidence score for successful authentications"
    )
    
    @extend_schema_field(OpenApiTypes.INT)
    def get_registered_devices(self, obj):
        """Get number of registered devices"""
        return obj.user.devices.count()
    
    registered_devices = serializers.SerializerMethodField(
        help_text="Number of registered devices"
    )
    
    @extend_schema_field(OpenApiTypes.INT)
    def get_active_sessions(self, obj):
        """Get number of active streaming sessions"""
        return obj.user.streaming_sessions.filter(
            status='active',
            ended_at__isnull=True
        ).count()
    
    active_sessions = serializers.SerializerMethodField(
        help_text="Number of active streaming sessions"
    )
    
    @extend_schema_field(OpenApiTypes.OBJECT)
    def get_security_summary(self, obj):
        """Get security summary"""
        recent_date = timezone.now() - timedelta(days=30)
        
        return {
            "two_factor_enabled": obj.user.two_factor_enabled,
            "face_auth_enabled": obj.user.face_auth_enabled,
            "recent_failed_attempts": obj.user.authentication_attempts.filter(
                is_successful=False,
                created_at__gte=recent_date
            ).count(),
            "last_password_change": None,  # Would be tracked if implemented
            "security_alerts": obj.user.security_alerts.filter(
                status__in=['open', 'investigating']
            ).count(),
            "trusted_devices": obj.user.devices.filter(is_verified=True).count()
        }
    
    security_summary = serializers.SerializerMethodField(
        help_text="Comprehensive security summary and status"
    )
    
    @extend_schema_field(OpenApiTypes.OBJECT)
    def get_activity_summary(self, obj):
        """Get activity summary"""
        now = timezone.now()
        last_week = now - timedelta(days=7)
        last_month = now - timedelta(days=30)
        
        return {
            "last_login": obj.user.last_login.isoformat() if obj.user.last_login else None,
            "last_face_auth": obj.user.last_face_auth.isoformat() if obj.user.last_face_auth else None,
            "logins_last_week": obj.user.authentication_logs.filter(
                event_type='login_success',
                timestamp__gte=last_week
            ).count(),
            "logins_last_month": obj.user.authentication_logs.filter(
                event_type='login_success',
                timestamp__gte=last_month
            ).count(),
            "enrollment_date": obj.user.enrollment_completed_at.isoformat() 
                if obj.user.enrollment_completed_at else None
        }
    
    activity_summary = serializers.SerializerMethodField(
        help_text="User activity summary and statistics"
    )
    
    class Meta:
        model = UserProfile
        fields = (
            'id', 'user_email', 'user_full_name', 'avatar', 'bio',
            'date_of_birth', 'phone_number', 'timezone', 'language',
            'notification_preferences', 'privacy_settings',
            'total_authentications', 'successful_authentications',
            'success_rate', 'enrolled_faces', 'average_confidence',
            'registered_devices', 'active_sessions', 'security_summary',
            'activity_summary', 'created_at', 'updated_at'
        )
        read_only_fields = (
            'id', 'user_email', 'user_full_name', 'total_authentications',
            'successful_authentications', 'success_rate', 'enrolled_faces',
            'average_confidence', 'registered_devices', 'active_sessions',
            'security_summary', 'activity_summary', 'created_at', 'updated_at'
        )
        extra_kwargs = {
            'avatar': {
                'help_text': 'User profile avatar image'
            },
            'bio': {
                'help_text': 'User biography or description'
            },
            'date_of_birth': {
                'help_text': 'User date of birth (YYYY-MM-DD format)'
            },
            'phone_number': {
                'help_text': 'User phone number in international format'
            },
            'timezone': {
                'help_text': 'User timezone preference'
            },
            'language': {
                'help_text': 'User language preference'
            },
            'notification_preferences': {
                'help_text': 'User notification preferences and settings'
            },
            'privacy_settings': {
                'help_text': 'User privacy settings and data sharing preferences'
            }
        }


class UserDeviceUpdateSerializer(serializers.ModelSerializer):
    """
    User Device Update Serializer
    
    Serializer for updating device information and security settings.
    """
    
    class Meta:
        model = UserDevice
        fields = (
            'device_name', 'is_verified', 'is_blocked'
        )
        extra_kwargs = {
            'device_name': {
                'help_text': 'Custom name for the device',
                'required': False
            },
            'is_verified': {
                'help_text': 'Mark device as verified/trusted',
                'required': False
            },
            'is_blocked': {
                'help_text': 'Block device from accessing the account',
                'required': False
            }
        }
    
    def validate(self, attrs):
        """Validate device update"""
        if attrs.get('is_verified') and attrs.get('is_blocked'):
            raise serializers.ValidationError(
                "Device cannot be both verified and blocked"
            )
        return attrs


# Summary serializers for list views
class UserDeviceSummarySerializer(serializers.ModelSerializer):
    """Simplified user device serializer for list views"""
    device_type_display = serializers.SerializerMethodField()
    trust_level = serializers.SerializerMethodField()
    time_since_last_used = serializers.SerializerMethodField()
    
    def get_device_type_display(self, obj):
        device_types = {
            'mobile': 'Mobile',
            'tablet': 'Tablet', 
            'desktop': 'Desktop',
            'laptop': 'Laptop',
            'unknown': 'Unknown'
        }
        return device_types.get(obj.device_type, 'Unknown')
    
    def get_trust_level(self, obj):
        # Simplified trust level calculation
        if obj.is_verified:
            return "high"
        elif obj.last_used and (timezone.now() - obj.last_used).days < 30:
            return "medium"
        else:
            return "low"
    
    def get_time_since_last_used(self, obj):
        if not obj.last_used:
            return "Never"
        
        delta = timezone.now() - obj.last_used
        if delta.days > 0:
            return f"{delta.days} days ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours}h ago"
        else:
            return "Recently"
    
    class Meta:
        model = UserDevice
        fields = (
            'id', 'device_name', 'device_type', 'device_type_display',
            'trust_level', 'is_verified', 'is_blocked',
            'time_since_last_used', 'last_used'
        )
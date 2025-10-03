"""
User serializers for profile and device management
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import UserProfile, UserDevice

User = get_user_model()


class CustomUserSerializer(serializers.ModelSerializer):
    """Serializer for custom user model"""
    
    class Meta:
        model = User
        fields = [
            'id', 'email', 'username', 'first_name', 'last_name',
            'phone_number', 'date_of_birth', 'bio', 'profile_picture',
            'face_enrolled', 'face_auth_enabled', 'two_factor_enabled',
            'is_verified', 'allow_analytics', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'email', 'face_enrolled', 'enrollment_completed_at',
            'last_face_auth', 'created_at', 'updated_at'
        ]
        extra_kwargs = {
            'phone_number': {'required': False},
            'date_of_birth': {'required': False},
            'bio': {'required': False},
        }


class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for user profile"""
    user = serializers.StringRelatedField(read_only=True)
    
    class Meta:
        model = UserProfile
        fields = [
            'id', 'user', 'company', 'position', 'address', 'city', 'country',
            'language', 'timezone', 'email_notifications', 'security_alerts',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'user', 'created_at', 'updated_at']


class UserDeviceSerializer(serializers.ModelSerializer):
    """Serializer for user devices"""
    user = serializers.StringRelatedField(read_only=True)
    duration_since_last_seen = serializers.SerializerMethodField()
    
    class Meta:
        model = UserDevice
        fields = [
            'id', 'user', 'device_id', 'device_name', 'device_type',
            'operating_system', 'browser', 'user_agent', 'is_trusted',
            'last_ip', 'last_location', 'first_seen', 'last_seen',
            'login_count', 'is_active', 'duration_since_last_seen'
        ]
        read_only_fields = [
            'id', 'user', 'device_id', 'first_seen', 'last_seen', 'login_count'
        ]
    
    def get_duration_since_last_seen(self, obj):
        """Calculate time since last seen"""
        if obj.last_seen:
            from django.utils import timezone
            delta = timezone.now() - obj.last_seen
            return int(delta.total_seconds())
        return None
    
    def validate_device_name(self, value):
        """Validate device name"""
        if len(value.strip()) < 2:
            raise serializers.ValidationError("Device name must be at least 2 characters long")
        return value.strip()


class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for user registration"""
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = [
            'email', 'username', 'first_name', 'last_name', 
            'password', 'password_confirm'
        ]
    
    def validate(self, attrs):
        """Validate password confirmation"""
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError("Passwords don't match")
        return attrs
    
    def create(self, validated_data):
        """Create user with encrypted password"""
        validated_data.pop('password_confirm')
        password = validated_data.pop('password')
        
        user = User.objects.create_user(
            password=password,
            **validated_data
        )
        return user
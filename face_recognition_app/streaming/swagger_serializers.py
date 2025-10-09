"""
Streaming App Serializers with Swagger Documentation
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.utils import timezone

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

from streaming.models import StreamingSession, WebRTCSignal

User = get_user_model()


class StreamingSessionSerializer(serializers.ModelSerializer):
    """
    Streaming Session Serializer
    
    WebRTC streaming session information for real-time video
    communication during face recognition processes.
    """
    user_email = serializers.CharField(
        source='user.email',
        read_only=True,
        help_text="Email of the user who owns this streaming session"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_session_duration(self, obj):
        """Calculate session duration"""
        if obj.ended_at:
            delta = obj.ended_at - obj.started_at
        else:
            delta = timezone.now() - obj.started_at
        
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    session_duration = serializers.SerializerMethodField(
        help_text="Duration of the streaming session"
    )
    
    @extend_schema_field(OpenApiTypes.BOOL)
    def get_is_active(self, obj):
        """Check if session is currently active"""
        return obj.status == 'active' and obj.ended_at is None
    
    is_active = serializers.SerializerMethodField(
        help_text="Whether the streaming session is currently active"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_connection_quality(self, obj):
        """Determine connection quality based on metrics"""
        if obj.quality_metrics:
            bandwidth = obj.quality_metrics.get('bandwidth', 0)
            latency = obj.quality_metrics.get('latency', 0)
            packet_loss = obj.quality_metrics.get('packet_loss', 0)
            
            if bandwidth > 1000000 and latency < 50 and packet_loss < 0.01:
                return "excellent"
            elif bandwidth > 500000 and latency < 100 and packet_loss < 0.02:
                return "good"
            elif bandwidth > 250000 and latency < 200 and packet_loss < 0.05:
                return "fair"
            else:
                return "poor"
        
        return "unknown"
    
    connection_quality = serializers.SerializerMethodField(
        help_text="Overall connection quality assessment"
    )
    
    class Meta:
        model = StreamingSession
        fields = (
            'id', 'user_email', 'session_type', 'status', 'peer_connection_id',
            'ice_servers', 'session_duration', 'is_active', 'connection_quality',
            'quality_metrics', 'error_details', 'started_at', 'ended_at',
            'created_at', 'updated_at'
        )
        read_only_fields = (
            'id', 'user_email', 'session_duration', 'is_active',
            'connection_quality', 'started_at', 'ended_at', 'created_at', 'updated_at'
        )
        extra_kwargs = {
            'session_type': {
                'help_text': 'Type of streaming session (enrollment, authentication, monitoring)'
            },
            'status': {
                'help_text': 'Current status of the streaming session'
            },
            'peer_connection_id': {
                'help_text': 'WebRTC peer connection identifier'
            },
            'ice_servers': {
                'help_text': 'ICE servers configuration for WebRTC connection'
            },
            'quality_metrics': {
                'help_text': 'Real-time quality metrics (bandwidth, latency, packet loss)'
            },
            'error_details': {
                'help_text': 'Details about any errors that occurred during the session'
            }
        }


class WebRTCSignalSerializer(serializers.ModelSerializer):
    """
    WebRTC Signal Serializer
    
    WebRTC signaling messages for connection establishment
    and maintenance during streaming sessions.
    """
    session_user_email = serializers.CharField(
        source='session.user.email',
        read_only=True,
        help_text="Email of the user who owns the streaming session"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_signal_type_display(self, obj):
        """Get human-readable signal type"""
        signal_types = {
            'offer': 'SDP Offer',
            'answer': 'SDP Answer',
            'candidate': 'ICE Candidate',
            'bye': 'Session Termination',
            'renegotiation': 'Connection Renegotiation',
            'error': 'Error Signal'
        }
        return signal_types.get(obj.signal_type, obj.signal_type.title())
    
    signal_type_display = serializers.SerializerMethodField(
        help_text="Human-readable signal type description"
    )
    
    @extend_schema_field(OpenApiTypes.BOOL)
    def get_is_processed(self, obj):
        """Check if signal has been processed"""
        return obj.processed_at is not None
    
    is_processed = serializers.SerializerMethodField(
        help_text="Whether the signal has been processed"
    )
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_time_since_created(self, obj):
        """Get time since signal was created"""
        delta = timezone.now() - obj.created_at
        
        if delta.seconds < 60:
            return f"{delta.seconds}s ago"
        elif delta.seconds < 3600:
            minutes = delta.seconds // 60
            return f"{minutes}m ago"
        else:
            hours = delta.seconds // 3600
            return f"{hours}h ago"
    
    time_since_created = serializers.SerializerMethodField(
        help_text="Time since the signal was created"
    )
    
    class Meta:
        model = WebRTCSignal
        fields = (
            'id', 'session_user_email', 'signal_type', 'signal_type_display',
            'signal_data', 'is_processed', 'time_since_created', 'processed_at',
            'error_message', 'created_at'
        )
        read_only_fields = (
            'id', 'session_user_email', 'signal_type_display', 'is_processed',
            'time_since_created', 'processed_at', 'created_at'
        )
        extra_kwargs = {
            'signal_type': {
                'help_text': 'Type of WebRTC signaling message'
            },
            'signal_data': {
                'help_text': 'WebRTC signaling payload (SDP or ICE candidate data)'
            },
            'error_message': {
                'help_text': 'Error message if signal processing failed'
            }
        }


class StreamingSessionCreateSerializer(serializers.ModelSerializer):
    """
    Streaming Session Creation Serializer
    
    Serializer for creating new WebRTC streaming sessions
    with proper validation and configuration.
    """
    
    class Meta:
        model = StreamingSession
        fields = (
            'session_type', 'ice_servers'
        )
        extra_kwargs = {
            'session_type': {
                'help_text': 'Type of streaming session to create',
                'required': True
            },
            'ice_servers': {
                'help_text': 'Custom ICE servers configuration (optional)',
                'required': False
            }
        }
    
    def validate_session_type(self, value):
        """Validate session type"""
        valid_types = ['enrollment', 'authentication', 'monitoring']
        if value not in valid_types:
            raise serializers.ValidationError(
                f"Invalid session type. Must be one of: {', '.join(valid_types)}"
            )
        return value
    
    def create(self, validated_data):
        """Create streaming session with user context"""
        user = self.context['request'].user
        
        # Set default ICE servers if not provided
        if 'ice_servers' not in validated_data:
            validated_data['ice_servers'] = [
                {"urls": "stun:stun.l.google.com:19302"}
            ]
        
        return StreamingSession.objects.create(
            user=user,
            status='initializing',
            **validated_data
        )


class WebRTCSignalCreateSerializer(serializers.Serializer):
    """
    WebRTC Signal Creation Serializer
    
    Serializer for creating WebRTC signaling messages
    with proper validation and type checking.
    """
    session_id = serializers.UUIDField(
        help_text="ID of the streaming session"
    )
    
    signal_type = serializers.ChoiceField(
        choices=[
            ('offer', 'SDP Offer'),
            ('answer', 'SDP Answer'),
            ('candidate', 'ICE Candidate'),
            ('bye', 'Session Termination'),
            ('renegotiation', 'Connection Renegotiation'),
            ('error', 'Error Signal')
        ],
        help_text="Type of WebRTC signaling message"
    )
    
    signal_data = serializers.JSONField(
        help_text="WebRTC signaling payload"
    )
    
    def validate_signal_data(self, value):
        """Validate signal data based on signal type"""
        signal_type = self.initial_data.get('signal_type', '')
        
        if signal_type in ['offer', 'answer']:
            required_fields = ['sdp', 'type']
            if not all(field in value for field in required_fields):
                raise serializers.ValidationError(
                    f"SDP payload must contain: {required_fields}"
                )
        
        elif signal_type == 'candidate':
            required_fields = ['candidate', 'sdpMid', 'sdpMLineIndex']
            if not all(field in value for field in required_fields):
                raise serializers.ValidationError(
                    f"ICE candidate payload must contain: {required_fields}"
                )
        
        return value
    
    def validate(self, attrs):
        """Cross-field validation"""
        session_id = attrs['session_id']
        user = self.context['request'].user
        
        # Verify session exists and belongs to user
        try:
            session = StreamingSession.objects.get(
                id=session_id,
                user=user
            )
            if session.status == 'ended':
                raise serializers.ValidationError(
                    "Cannot send signals to ended session"
                )
        except StreamingSession.DoesNotExist:
            raise serializers.ValidationError(
                "Streaming session not found"
            )
        
        attrs['session'] = session
        return attrs
    
    def create(self, validated_data):
        """Create WebRTC signal"""
        session = validated_data.pop('session')
        
        return WebRTCSignal.objects.create(
            session=session,
            **validated_data
        )


# Summary serializers for list views
class StreamingSessionSummarySerializer(serializers.ModelSerializer):
    """Simplified streaming session serializer for list views"""
    user_email = serializers.CharField(source='user.email', read_only=True)
    is_active = serializers.SerializerMethodField()
    session_duration = serializers.SerializerMethodField()
    
    def get_is_active(self, obj):
        return obj.status == 'active' and obj.ended_at is None
    
    def get_session_duration(self, obj):
        if obj.ended_at:
            delta = obj.ended_at - obj.started_at
        else:
            delta = timezone.now() - obj.started_at
        
        total_seconds = int(delta.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes}m"
        else:
            hours = total_seconds // 3600
            return f"{hours}h"
    
    class Meta:
        model = StreamingSession
        fields = (
            'id', 'user_email', 'session_type', 'status',
            'is_active', 'session_duration', 'started_at'
        )


class WebRTCSignalSummarySerializer(serializers.ModelSerializer):
    """Simplified WebRTC signal serializer for list views"""
    session_user_email = serializers.CharField(source='session.user.email', read_only=True)
    signal_type_display = serializers.SerializerMethodField()
    
    def get_signal_type_display(self, obj):
        return obj.signal_type.title()
    
    class Meta:
        model = WebRTCSignal
        fields = (
            'id', 'session_user_email', 'signal_type',
            'signal_type_display', 'processed_at', 'created_at'
        )
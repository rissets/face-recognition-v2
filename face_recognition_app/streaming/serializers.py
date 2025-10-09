"""
Streaming serializers for WebRTC and real-time communication
"""
from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field
from drf_spectacular.types import OpenApiTypes
from .models import StreamingSession, WebRTCSignal


class StreamingSessionSerializer(serializers.ModelSerializer):
    """Serializer for streaming sessions"""
    user = serializers.StringRelatedField(read_only=True)
    duration = serializers.SerializerMethodField()
    
    class Meta:
        model = StreamingSession
        fields = [
            'id', 'user', 'session_token', 'session_type', 'status',
            'ice_servers', 'constraints', 'peer_connection_id', 'remote_address',
            'video_quality', 'frame_rate', 'bitrate', 'session_data', 'error_log',
            'created_at', 'connected_at', 'completed_at', 'duration'
        ]
        read_only_fields = ['id', 'user', 'session_token', 'created_at']
    
    @extend_schema_field(OpenApiTypes.INT)
    def get_duration(self, obj) -> int:
        """Calculate session duration in seconds"""
        if obj.completed_at:
            duration = obj.completed_at - obj.created_at
            return int(duration.total_seconds())
        return 0


class WebRTCSignalSerializer(serializers.ModelSerializer):
    """Serializer for WebRTC signaling data"""
    session = serializers.StringRelatedField(read_only=True)
    
    class Meta:
        model = WebRTCSignal
        fields = [
            'id', 'session', 'signal_type', 'signal_data', 'direction', 'created_at'
        ]
        read_only_fields = ['id', 'session', 'created_at']
    
    def validate_signal_type(self, value):
        """Validate signal type"""
        valid_types = ['offer', 'answer', 'ice_candidate']
        if value not in valid_types:
            raise serializers.ValidationError(
                f"Invalid signal type. Must be one of: {', '.join(valid_types)}"
            )
        return value
    
    def validate_signal_data(self, value):
        """Validate signal data structure"""
        if not isinstance(value, dict):
            raise serializers.ValidationError("Signal data must be a JSON object")
        
        signal_type = self.initial_data.get('signal_type')
        
        if signal_type == 'offer' and 'sdp' not in value:
            raise serializers.ValidationError("Offer signal must contain 'sdp' field")
        
        if signal_type == 'answer' and 'sdp' not in value:
            raise serializers.ValidationError("Answer signal must contain 'sdp' field")
        
        if signal_type == 'ice_candidate' and 'candidate' not in value:
            raise serializers.ValidationError("ICE candidate signal must contain 'candidate' field")
        
        return value


class StreamingSessionCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating streaming sessions"""
    
    class Meta:
        model = StreamingSession
        fields = ['session_type', 'stream_config']
    
    def validate_session_type(self, value):
        """Validate session type"""
        valid_types = ['enrollment', 'authentication', 'verification', 'identification']
        if value not in valid_types:
            raise serializers.ValidationError(
                f"Invalid session type. Must be one of: {', '.join(valid_types)}"
            )
        return value

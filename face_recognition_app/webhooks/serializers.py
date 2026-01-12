"""
Serializers for webhook management
"""
from rest_framework import serializers
from .models import WebhookEvent, WebhookEndpoint, WebhookDelivery, WebhookEventLog


class WebhookEventSerializer(serializers.ModelSerializer):
    """Serializer for webhook events"""
    
    class Meta:
        model = WebhookEvent
        fields = [
            'id', 'event_name', 'description', 'is_active',
            'payload_schema', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class WebhookEndpointSerializer(serializers.ModelSerializer):
    """Serializer for webhook endpoints"""
    
    class Meta:
        model = WebhookEndpoint
        fields = [
            'id', 'client', 'name', 'url', 'status', 'subscribed_events',
            'secret_token', 'max_retries', 'retry_delay_seconds',
            'total_deliveries', 'successful_deliveries', 'failed_deliveries',
            'metadata', 'created_at', 'updated_at', 'last_delivery_at'
        ]
        read_only_fields = [
            'id', 'total_deliveries', 'successful_deliveries', 'failed_deliveries',
            'created_at', 'updated_at', 'last_delivery_at'
        ]
        extra_kwargs = {
            'secret_token': {'write_only': True, 'required': False, 'allow_null': True, 'allow_blank': True}
        }


class WebhookDeliverySerializer(serializers.ModelSerializer):
    """Serializer for webhook deliveries"""
    
    class Meta:
        model = WebhookDelivery
        fields = [
            'id', 'endpoint', 'event_name', 'event_data', 'status',
            'http_method', 'headers', 'response_status_code', 'response_headers',
            'response_body', 'response_time_ms', 'attempt_number', 'max_attempts',
            'next_retry_at', 'error_message', 'error_code',
            'created_at', 'delivered_at', 'completed_at'
        ]
        read_only_fields = [
            'id', 'response_status_code', 'response_headers', 'response_body',
            'response_time_ms', 'error_message', 'error_code',
            'created_at', 'delivered_at', 'completed_at'
        ]


class WebhookEventLogSerializer(serializers.ModelSerializer):
    """Serializer for webhook event logs"""

    success_rate = serializers.FloatField(read_only=True)
    
    class Meta:
        model = WebhookEventLog
        fields = [
            'id',
            'client',
            'event_name',
            'event_source',
            'event_data',
            'total_endpoints',
            'successful_deliveries',
            'failed_deliveries',
            'metadata',
            'success_rate',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class WebhookTestSerializer(serializers.Serializer):
    """Serializer for webhook testing"""
    test_event_type = serializers.ChoiceField(choices=[
        ('user.enrolled', 'User Enrolled'),
        ('user.authentication.success', 'Authentication Success'),
        ('user.authentication.failed', 'Authentication Failed'),
        ('enrollment.completed', 'Enrollment Completed'),
        ('enrollment.failed', 'Enrollment Failed'),
        ('session.created', 'Session Created'),
        ('session.expired', 'Session Expired'),
    ])
    test_data = serializers.JSONField(required=False, default=dict)


class WebhookStatsSerializer(serializers.Serializer):
    """Serializer for webhook statistics"""
    total_events = serializers.IntegerField()
    total_deliveries = serializers.IntegerField()
    successful_deliveries = serializers.IntegerField()
    failed_deliveries = serializers.IntegerField()
    pending_deliveries = serializers.IntegerField()
    success_rate = serializers.FloatField()
    avg_delivery_time = serializers.FloatField()
    events_by_type = serializers.DictField()
    recent_failures = serializers.ListField()

"""
Client management serializers aligned with the third-party architecture.
"""
from rest_framework import serializers

from .models import Client, ClientUser, ClientAPIUsage, ClientWebhookLog


class ClientSerializer(serializers.ModelSerializer):
    """Expose client configuration without leaking sensitive secrets."""

    api_secret = serializers.CharField(write_only=True, required=False)
    webhook_secret = serializers.CharField(write_only=True, required=False, allow_blank=True)

    class Meta:
        model = Client
        fields = [
            'id',
            'client_id',
            'name',
            'description',
            'domain',
            'api_key',
            'secret_key',
            'api_secret',
            'status',
            'tier',
            'webhook_url',
            'webhook_secret',
            'allowed_domains',
            'rate_limit_per_hour',
            'rate_limit_per_day',
            'features',
            'metadata',
            'contact_email',
            'contact_name',
            'created_at',
            'updated_at',
            'last_activity',
        ]
        read_only_fields = [
            'id',
            'client_id',
            'api_key',
            'secret_key',
            'created_at',
            'updated_at',
            'last_activity',
        ]

    def create(self, validated_data):
        """Allow optional secret rotation during creation."""
        api_secret = validated_data.pop('api_secret', None)
        webhook_secret = validated_data.pop('webhook_secret', None)
        client = super().create(validated_data)

        if api_secret:
            client.set_api_secret(api_secret)
        if webhook_secret:
            client.webhook_secret = webhook_secret
            client.save(update_fields=['webhook_secret'])

        return client

    def update(self, instance, validated_data):
        """Support rotating credentials via serializer."""
        api_secret = validated_data.pop('api_secret', None)
        webhook_secret = validated_data.pop('webhook_secret', None)
        instance = super().update(instance, validated_data)

        if api_secret:
            instance.set_api_secret(api_secret)
        if webhook_secret:
            instance.webhook_secret = webhook_secret
            instance.save(update_fields=['webhook_secret'])

        return instance


class ClientUserSerializer(serializers.ModelSerializer):
    """Serialise client user profiles from upstream systems."""

    display_name = serializers.CharField(read_only=True)
    profile_image_url = serializers.SerializerMethodField()

    class Meta:
        model = ClientUser
        fields = [
            'id',
            'client',
            'external_user_id',
            'external_user_uuid',
            'profile',
            'is_enrolled',
            'enrollment_completed_at',
            'face_auth_enabled',
            'metadata',
            'last_recognition_at',
            'created_at',
            'updated_at',
            'display_name',
            'profile_image_url',
        ]
        read_only_fields = [
            'id',
            'client',
            'is_enrolled',
            'enrollment_completed_at',
            'last_recognition_at',
            'created_at',
            'updated_at',
            'profile_image_url',
        ]

    def get_profile_image_url(self, obj):
        """Get the profile image URL if it exists"""
        if obj.profile_image:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.profile_image.url)
            return obj.profile_image.url
        return None


class ClientAPIUsageSerializer(serializers.ModelSerializer):
    """Expose granular API usage entries."""

    class Meta:
        model = ClientAPIUsage
        fields = [
            'id',
            'client',
            'endpoint',
            'method',
            'status_code',
            'ip_address',
            'user_agent',
            'response_time_ms',
            'metadata',
            'created_at',
        ]
        read_only_fields = fields


class ClientWebhookLogSerializer(serializers.ModelSerializer):
    """Serialise webhook delivery attempts and states."""

    class Meta:
        model = ClientWebhookLog
        fields = [
            'id',
            'client',
            'event_type',
            'payload',
            'status',
            'response_status_code',
            'response_body',
            'error_message',
            'attempt_count',
            'max_attempts',
            'next_retry_at',
            'created_at',
            'delivered_at',
        ]
        read_only_fields = fields


class ClientStatsSerializer(serializers.Serializer):
    """Embed aggregated client statistics."""

    total_users = serializers.IntegerField()
    enrolled_users = serializers.IntegerField()
    active_face_auth = serializers.IntegerField()
    total_enrollments = serializers.IntegerField()
    successful_authentications = serializers.IntegerField()
    failed_authentications = serializers.IntegerField()
    api_calls_last_24h = serializers.IntegerField()
    api_calls_last_7d = serializers.IntegerField()
    webhook_success_rate = serializers.FloatField()


class ClientCredentialsResetSerializer(serializers.Serializer):
    """Payload for credential rotation."""

    reset_api_key = serializers.BooleanField(default=False)
    reset_secret_key = serializers.BooleanField(default=False)
    reset_webhook_secret = serializers.BooleanField(default=False)

    def validate(self, attrs):
        if not any(attrs.values()):
            raise serializers.ValidationError("Select at least one credential to rotate.")
        return attrs


class ClientCredentialsResetResponseSerializer(serializers.Serializer):
    """Response body when credentials are rotated."""

    message = serializers.CharField()
    credentials = serializers.DictField(
        child=serializers.CharField(),
        required=False,
        help_text="Map of rotated credential identifiers to their new values.",
    )


class ClientUserWriteSerializer(serializers.ModelSerializer):
    """Request payload for creating or updating client users."""

    profile = serializers.JSONField(required=False, default=dict)
    metadata = serializers.JSONField(required=False, default=dict)

    class Meta:
        model = ClientUser
        fields = [
            'external_user_id',
            'external_user_uuid',
            'profile',
            'metadata',
            'face_auth_enabled',
        ]
        extra_kwargs = {
            'external_user_id': {'required': True},
            'face_auth_enabled': {'required': False, 'default': True},
        }


class ClientUserToggleResponseSerializer(serializers.Serializer):
    """Response envelope after toggling face authentication."""

    message = serializers.CharField()
    user = ClientUserSerializer()


class ClientUserEnrollmentSerializer(serializers.Serializer):
    """Minimal enrollment summary for a client user."""

    id = serializers.UUIDField()
    status = serializers.CharField()
    quality_score = serializers.FloatField(allow_null=True)
    created_at = serializers.DateTimeField()


class ClientUserEnrollmentListSerializer(serializers.Serializer):
    """Response payload for listing user enrollments."""

    count = serializers.IntegerField()
    enrollments = ClientUserEnrollmentSerializer(many=True)

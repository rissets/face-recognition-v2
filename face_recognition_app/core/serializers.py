"""
Serializers supporting the third-party face recognition API surface.
"""
from typing import Optional

from rest_framework import serializers

from .models import AuditLog, HealthCheck, SecurityEvent, SystemConfiguration
from clients.models import Client, ClientUser


class SystemConfigurationSerializer(serializers.ModelSerializer):
    """Expose key/value configuration entries."""

    class Meta:
        model = SystemConfiguration
        fields = [
            "id",
            "key",
            "value",
            "description",
            "is_encrypted",
            "is_active",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class AuditLogSerializer(serializers.ModelSerializer):
    """Tenant-aware audit log serialisation."""

    client_name = serializers.CharField(source="client.name", read_only=True)
    client_user_name = serializers.CharField(
        source="client_user.display_name", read_only=True
    )

    class Meta:
        model = AuditLog
        fields = [
            "id",
            "client",
            "client_user",
            "client_name",
            "client_user_name",
            "action",
            "resource_type",
            "resource_id",
            "ip_address",
            "user_agent",
            "details",
            "success",
            "error_message",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class SecurityEventSerializer(serializers.ModelSerializer):
    """Serialise security events per client."""

    client_name = serializers.CharField(source="client.name", read_only=True)
    client_user_name = serializers.CharField(
        source="client_user.display_name", read_only=True
    )
    resolved_by_name = serializers.CharField(
        source="resolved_by_client_user.display_name", read_only=True
    )

    class Meta:
        model = SecurityEvent
        fields = [
            "id",
            "event_type",
            "severity",
            "client",
            "client_user",
            "client_name",
            "client_user_name",
            "ip_address",
            "user_agent",
            "details",
            "resolved",
            "resolved_at",
            "resolved_by_client",
            "resolved_by_client_user",
            "resolved_by_name",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class HealthCheckSerializer(serializers.ModelSerializer):
    """Expose system health probe information."""

    class Meta:
        model = HealthCheck
        fields = [
            "id",
            "service_name",
            "status",
            "response_time",
            "error_message",
            "details",
            "created_at",
        ]
        read_only_fields = fields


class ClientTokenSerializer(serializers.Serializer):
    """Validate credentials when clients request API tokens."""

    api_key = serializers.CharField(required=False, allow_blank=False)
    client_id = serializers.CharField(required=False, allow_blank=False)
    api_secret = serializers.CharField(write_only=True)

    default_error_messages = {
        "invalid_credentials": "Invalid credentials",
        "missing_api_key": "API key is required when multiple clients match the query.",
    }

    def _filter_clients(self, client_id: Optional[str]):
        qs = Client.objects.filter(status="active")
        if client_id:
            qs = qs.filter(client_id=client_id)
        return qs

    def _match_by_api_key(self, queryset, api_key: str):
        for candidate in queryset:
            if candidate.api_key == api_key:
                return candidate
        return None

    def validate(self, attrs):
        api_key = attrs.get("api_key")
        api_secret = attrs.get("api_secret")
        client_id = attrs.get("client_id")

        if not api_secret:
            raise serializers.ValidationError(self.default_error_messages["invalid_credentials"])

        candidates = self._filter_clients(client_id)

        client = None
        if api_key:
            client = Client.find_active_by_api_key(api_key)
            if client_id and client and client.client_id != client_id:
                client = None
        else:
            if candidates.count() == 1:
                client = candidates.first()
            else:
                raise serializers.ValidationError(self.default_error_messages["missing_api_key"])

        if not client or not client.check_api_secret(api_secret):
            raise serializers.ValidationError(self.default_error_messages["invalid_credentials"])

        attrs["client"] = client
        return attrs


class ClientUserAuthSerializer(serializers.Serializer):
    """Validate client-side user references for face workflows."""

    client_id = serializers.CharField()
    external_user_id = serializers.CharField()

    def validate(self, attrs):
        client_identifier = attrs.get("client_id")

        try:
            client = Client.objects.get(client_id=client_identifier, status="active")
        except Client.DoesNotExist as exc:
            raise serializers.ValidationError("Invalid client credentials") from exc

        try:
            user = ClientUser.objects.get(
                client=client, external_user_id=attrs["external_user_id"]
            )
        except ClientUser.DoesNotExist as exc:
            raise serializers.ValidationError("Client user not found") from exc

        attrs["client"] = client
        attrs["client_user"] = user
        return attrs


class SystemStatusSerializer(serializers.Serializer):
    """Aggregate status payload for observability endpoints."""

    status = serializers.CharField()
    uptime = serializers.CharField()
    version = serializers.CharField()
    database_status = serializers.CharField()
    redis_status = serializers.CharField()
    celery_status = serializers.CharField()
    face_processing_status = serializers.CharField()
    total_clients = serializers.IntegerField()
    active_sessions = serializers.IntegerField()
    total_enrollments = serializers.IntegerField()
    total_authentications = serializers.IntegerField()


class ErrorResponseSerializer(serializers.Serializer):
    """Canonical error envelope."""

    error = serializers.CharField()
    message = serializers.CharField()
    code = serializers.CharField(required=False)
    details = serializers.JSONField(required=False)
    timestamp = serializers.DateTimeField()


class SuccessResponseSerializer(serializers.Serializer):
    """Canonical success envelope."""

    success = serializers.BooleanField(default=True)
    message = serializers.CharField()
    data = serializers.JSONField(required=False)
    timestamp = serializers.DateTimeField()

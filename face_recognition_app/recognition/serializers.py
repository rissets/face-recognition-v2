"""
Recognition data serializers backed by the enhanced auth_service models.
"""
from rest_framework import serializers
from auth_service.models import (
    FaceEnrollment,
    AuthenticationSession,
    FaceRecognitionAttempt,
)


class FaceEnrollmentSummarySerializer(serializers.ModelSerializer):
    """Lightweight serializer for enrollment records."""

    client_user_id = serializers.UUIDField(source="client_user.id", read_only=True)
    external_user_id = serializers.CharField(
        source="client_user.external_user_id", read_only=True
    )
    display_name = serializers.CharField(
        source="client_user.display_name", read_only=True
    )

    class Meta:
        model = FaceEnrollment
        fields = [
            "id",
            "client_user_id",
            "external_user_id",
            "display_name",
            "status",
            "face_quality_score",
            "liveness_score",
            "anti_spoofing_score",
            "sample_number",
            "total_samples",
            "face_image_path",
            "metadata",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields


class AuthenticationSessionSummarySerializer(serializers.ModelSerializer):
    """Expose key details about authentication/enrollment sessions."""

    client_user_id = serializers.UUIDField(
        source="client_user.id", read_only=True, allow_null=True
    )
    external_user_id = serializers.CharField(
        source="client_user.external_user_id", read_only=True, allow_blank=True
    )

    class Meta:
        model = AuthenticationSession
        fields = [
            "id",
            "session_token",
            "session_type",
            "status",
            "client_user_id",
            "external_user_id",
            "ip_address",
            "user_agent",
            "metadata",
            "created_at",
            "completed_at",
            "expires_at",
        ]
        read_only_fields = fields


class FaceRecognitionAttemptSummarySerializer(serializers.ModelSerializer):
    """Summarise recognition attempts for troubleshooting."""

    matched_user_id = serializers.UUIDField(
        source="matched_user.id", read_only=True, allow_null=True
    )
    matched_external_user_id = serializers.CharField(
        source="matched_user.external_user_id", read_only=True, allow_blank=True
    )
    session_token = serializers.CharField(
        source="session.session_token", read_only=True
    )

    class Meta:
        model = FaceRecognitionAttempt
        fields = [
            "id",
            "session_token",
            "result",
            "similarity_score",
            "confidence_score",
            "face_quality_score",
            "liveness_score",
            "anti_spoofing_score",
            "matched_user_id",
            "matched_external_user_id",
            "ip_address",
            "user_agent",
            "metadata",
            "created_at",
        ]
        read_only_fields = fields

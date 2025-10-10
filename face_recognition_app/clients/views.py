"""
Client management API views for the third-party face recognition service.
"""
from datetime import timedelta

from django.utils import timezone
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

try:
    from drf_spectacular.utils import extend_schema, extend_schema_view
    SPECTACULAR_AVAILABLE = True
except ImportError:
    def extend_schema(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def extend_schema_view(**kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    SPECTACULAR_AVAILABLE = False

from auth_service.authentication import APIKeyAuthentication, JWTClientAuthentication
from .models import Client, ClientAPIUsage, ClientUser, ClientWebhookLog
from .serializers import (
    ClientAPIUsageSerializer,
    ClientCredentialsResetSerializer,
    ClientSerializer,
    ClientStatsSerializer,
    ClientUserSerializer,
    ClientWebhookLogSerializer,
)



class ClientViewSet(viewsets.ModelViewSet):
    """CRUD and analytics for clients authenticated via API key."""

    queryset = Client.objects.all()
    serializer_class = ClientSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Limit non-admin clients to their own record."""
        if hasattr(self.request, "client"):
            return self.queryset.filter(id=self.request.client.id)
        return self.queryset.none()

    @action(detail=True, methods=["get"])
    def stats(self, request, pk=None):
        """Return aggregated usage metrics for the client."""
        client = self.get_object()

        users_qs = client.users.all()
        enrollments_qs = client.enrollments.all()
        attempts_qs = client.recognition_attempts.all()
        usage_qs = client.api_usage.all()
        webhook_qs = client.webhook_logs.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        )

        stats_payload = {
            "total_users": users_qs.count(),
            "enrolled_users": users_qs.filter(is_enrolled=True).count(),
            "active_face_auth": users_qs.filter(face_auth_enabled=True).count(),
            "total_enrollments": enrollments_qs.count(),
            "successful_authentications": attempts_qs.filter(result="success").count(),
            "failed_authentications": attempts_qs.exclude(result="success").count(),
            "api_calls_last_24h": usage_qs.filter(
                created_at__gte=timezone.now() - timedelta(days=1)
            ).count(),
            "api_calls_last_7d": usage_qs.filter(
                created_at__gte=timezone.now() - timedelta(days=7)
            ).count(),
            "webhook_success_rate": self._compute_webhook_success_rate(webhook_qs),
        }

        serializer = ClientStatsSerializer(stats_payload)
        return Response(serializer.data)

    @staticmethod
    def _compute_webhook_success_rate(webhook_qs):
        total = webhook_qs.count()
        if total == 0:
            return 0.0
        success = webhook_qs.filter(status="success").count()
        return round((success / total) * 100, 2)

    @action(detail=True, methods=["post"])
    def reset_credentials(self, request, pk=None):
        """Rotate client secrets or API keys."""
        client = self.get_object()
        serializer = ClientCredentialsResetSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        new_values = {}
        if serializer.validated_data["reset_api_key"]:
            new_values["api_key"] = client.regenerate_api_key()

        if serializer.validated_data["reset_secret_key"]:
            from secrets import token_urlsafe

            new_secret = token_urlsafe(48)
            client.set_api_secret(new_secret)
            new_values["api_secret"] = new_secret

        if serializer.validated_data["reset_webhook_secret"]:
            new_values["webhook_secret"] = client.rotate_webhook_secret()

        return Response(
            {
                "message": "Credentials rotated successfully",
                "credentials": new_values,
            }
        )


class ClientUserViewSet(viewsets.ModelViewSet):
    """Manage client users within the authenticated tenant."""

    queryset = ClientUser.objects.all()
    serializer_class = ClientUserSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if hasattr(self.request, "client"):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()

    def perform_create(self, serializer):
        serializer.save(client=self.request.client)

    @action(detail=True, methods=["post"])
    def activate(self, request, pk=None):
        user = self.get_object()
        user.face_auth_enabled = True
        user.save(update_fields=["face_auth_enabled"])
        return Response({"message": "User face authentication enabled"})

    @action(detail=True, methods=["post"])
    def deactivate(self, request, pk=None):
        user = self.get_object()
        user.face_auth_enabled = False
        user.save(update_fields=["face_auth_enabled"])
        return Response({"message": "User face authentication disabled"})

    @action(detail=True, methods=["get"])
    def enrollments(self, request, pk=None):
        user = self.get_object()
        enrollments = user.enrollments.all()
        return Response(
            {
                "count": enrollments.count(),
                "enrollments": [
                    {
                        "id": enrollment.id,
                        "status": enrollment.status,
                        "quality_score": enrollment.face_quality_score,
                        "created_at": enrollment.created_at,
                    }
                    for enrollment in enrollments
                ],
            }
        )


class ClientAPIUsageViewSet(viewsets.ReadOnlyModelViewSet):
    """Per-client API usage history."""

    queryset = ClientAPIUsage.objects.all()
    serializer_class = ClientAPIUsageSerializer
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if hasattr(self.request, "client"):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()


class ClientWebhookLogViewSet(viewsets.ReadOnlyModelViewSet):
    """Expose webhook delivery logs for observability."""

    queryset = ClientWebhookLog.objects.all()
    serializer_class = ClientWebhookLogSerializer
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if hasattr(self.request, "client"):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()

    @action(detail=False, methods=["get"])
    def failed(self, request):
        """Shortcut to inspect failed deliveries."""
        failed_logs = self.get_queryset().filter(status="failed").order_by("-created_at")[
            :50
        ]
        serializer = self.get_serializer(failed_logs, many=True)
        return Response(serializer.data)

"""
Client management API views for the third-party face recognition service.
"""
import logging
from datetime import timedelta

from django.utils import timezone
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

try:
    from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiResponse
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
    
    def OpenApiResponse(*args, **kwargs):
        return None
    
    SPECTACULAR_AVAILABLE = False

logger = logging.getLogger("clients.views")

from auth_service.authentication import APIKeyAuthentication, JWTClientAuthentication
from .models import Client, ClientAPIUsage, ClientUser, ClientWebhookLog
from .serializers import (
    ClientAPIUsageSerializer,
    ClientCredentialsResetResponseSerializer,
    ClientCredentialsResetSerializer,
    ClientSerializer,
    ClientStatsSerializer,
    ClientUserEnrollmentListSerializer,
    ClientUserSerializer,
    ClientUserToggleResponseSerializer,
    ClientUserWriteSerializer,
    ClientWebhookLogSerializer,
)


@extend_schema_view(
    list=extend_schema(
        tags=["Client Management"],
        summary="List Clients",
        description="Retrieve the client record for the authenticated API key. Non-admin keys can only see their own client.",
    ),
    retrieve=extend_schema(
        tags=["Client Management"],
        summary="Retrieve Client Details",
        description="Get detailed information about the authenticated client.",
    ),
    create=extend_schema(
        tags=["Client Management"],
        summary="Create a Client (Admin Only)",
        description="Create a new client. This endpoint is typically restricted to system administrators.",
    ),
    update=extend_schema(
        tags=["Client Management"],
        summary="Update Client Details",
        description="Update the configuration of the authenticated client.",
    ),
    partial_update=extend_schema(
        tags=["Client Management"],
        summary="Partially Update Client Details",
        description="Partially update the configuration of the authenticated client.",
    ),
    destroy=extend_schema(
        tags=["Client Management"],
        summary="Delete a Client (Admin Only)",
        description="Permanently delete a client. This is a destructive action and typically restricted.",
    ),
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

    @extend_schema(
        tags=["Client Management", "Analytics"],
        summary="Get Client Statistics",
        description="Retrieve aggregated usage metrics and statistics for the authenticated client, such as user counts, authentication attempts, and API call volume.",
        responses=ClientStatsSerializer,
    )
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

    @extend_schema(
        tags=["Client Management"],
        summary="Reset Client Credentials",
        description="Rotate API keys, secret keys, or webhook secrets for the authenticated client. The old credentials will be invalidated.",
        request=ClientCredentialsResetSerializer,
        responses=ClientCredentialsResetResponseSerializer,
    )
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


@extend_schema_view(
    list=extend_schema(
        tags=["Client Management"],
        summary="List Client Users",
        description="Retrieve a paginated list of users associated with the authenticated client.",
        responses=ClientUserSerializer,
    ),
    retrieve=extend_schema(
        tags=["Client Management"],
        summary="Retrieve a Client User",
        description="Get detailed information about a specific user by their external_user_id.",
        responses=ClientUserSerializer,
    ),
    create=extend_schema(
        tags=["Client Management"],
        summary="Create a Client User",
        description="Create a new user associated with the authenticated client.",
        request=ClientUserWriteSerializer,
        responses=ClientUserSerializer,
    ),
    update=extend_schema(
        tags=["Client Management"],
        summary="Update a Client User",
        description="Update the details of an existing client user by their external_user_id.",
        request=ClientUserWriteSerializer,
        responses=ClientUserSerializer,
    ),
    partial_update=extend_schema(
        tags=["Client Management"],
        summary="Partially Update a Client User",
        description="Partially update the details of an existing client user by their external_user_id.",
        request=ClientUserWriteSerializer,
        responses=ClientUserSerializer,
    ),
    destroy=extend_schema(
        tags=["Client Management"],
        summary="Delete a Client User",
        description="Permanently delete a client user and all their associated data, including enrollments.",
    ),
)
class ClientUserViewSet(viewsets.ModelViewSet):
    """Manage client users within the authenticated tenant."""

    queryset = ClientUser.objects.all()
    serializer_class = ClientUserSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    lookup_field = 'external_user_id'
    lookup_url_kwarg = 'external_user_id'

    def get_queryset(self):
        if hasattr(self.request, "client"):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()

    def get_serializer_class(self):
        """Use different serializers for read and write operations."""
        if self.action in ['create', 'update', 'partial_update']:
            return ClientUserWriteSerializer
        return ClientUserSerializer

    def perform_create(self, serializer):
        """
        Create a new ClientUser.
        The signal will automatically trigger old_profile_photo embedding extraction.
        """
        serializer.save(client=self.request.client)
    
    def perform_update(self, serializer):
        """
        Update a ClientUser.
        The signal will automatically trigger old_profile_photo embedding extraction if changed.
        """
        serializer.save()
    
    @extend_schema(
        tags=["Client Management"],
        summary="Preload User for Enrollment",
        description="Pre-extract old_profile_photo embedding in the background. Call this before creating an enrollment session to reduce latency.",
        responses={
            200: {"type": "object", "properties": {"message": {"type": "string"}, "preload_started": {"type": "boolean"}}},
            404: OpenApiResponse(description="User not found"),
        },
    )
    @action(detail=True, methods=["post"])
    def preload(self, request, external_user_id=None):
        """
        Preload user data for enrollment.
        This triggers background extraction of old_profile_photo embedding.
        """
        user = self.get_object()
        
        preload_started = False
        message = "User already preloaded or no old profile photo"
        
        if user.old_profile_photo and not user.get_cached_old_photo_embedding():
            try:
                from core.tasks import extract_old_photo_embedding_task
                extract_old_photo_embedding_task.delay(str(user.id))
                preload_started = True
                message = "Preload started for old profile photo embedding extraction"
            except Exception as e:
                message = f"Failed to start preload: {str(e)}"
        elif user.get_cached_old_photo_embedding():
            message = "Old profile photo embedding already cached"
        else:
            message = "No old profile photo to preload"
        
        return Response({
            "message": message,
            "preload_started": preload_started,
            "user_id": user.external_user_id,
        })

    @extend_schema(
        tags=["Client Management"],
        summary="Activate Face Authentication",
        description="Enable the face authentication feature for a specific client user.",
        responses=ClientUserToggleResponseSerializer,
    )
    @action(detail=True, methods=["post"])
    def activate(self, request, external_user_id=None):
        user = self.get_object()
        user.face_auth_enabled = True
        user.save(update_fields=["face_auth_enabled"])
        serialized = ClientUserSerializer(user, context={"request": request})
        return Response(
            {
                "message": "User face authentication enabled",
                "user": serialized.data,
            }
        )

    @extend_schema(
        tags=["Client Management"],
        summary="Deactivate Face Authentication",
        description="Disable the face authentication feature for a specific client user.",
        responses=ClientUserToggleResponseSerializer,
    )
    @action(detail=True, methods=["post"])
    def deactivate(self, request, external_user_id=None):
        user = self.get_object()
        user.face_auth_enabled = False
        user.save(update_fields=["face_auth_enabled"])
        serialized = ClientUserSerializer(user, context={"request": request})
        return Response(
            {
                "message": "User face authentication disabled",
                "user": serialized.data,
            }
        )

    @extend_schema(
        tags=["Client Management", "Face Enrollment"],
        summary="List User Enrollments",
        description="Retrieve a list of all face enrollment records for a specific client user.",
        responses=ClientUserEnrollmentListSerializer,
    )
    @action(detail=True, methods=["get"])
    def enrollments(self, request, external_user_id=None):
        user = self.get_object()
        enrollments = user.enrollments.all()
        payload = {
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
        serializer = ClientUserEnrollmentListSerializer(payload)
        return Response(serializer.data)


@extend_schema_view(
    list=extend_schema(
        tags=["Client Management", "Analytics"],
        summary="List API Usage Logs",
        description="Retrieve a paginated list of API usage logs for the authenticated client.",
    ),
    retrieve=extend_schema(
        tags=["Client Management", "Analytics"],
        summary="Retrieve API Usage Log",
        description="Get detailed information about a specific API usage log entry.",
    ),
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


@extend_schema_view(
    list=extend_schema(
        tags=["Client Management", "Webhooks"],
        summary="List Webhook Logs",
        description="Retrieve a paginated list of webhook delivery logs for the authenticated client.",
    ),
    retrieve=extend_schema(
        tags=["Client Management", "Webhooks"],
        summary="Retrieve Webhook Log",
        description="Get detailed information about a specific webhook delivery log entry.",
    ),
)
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
    @extend_schema(
        tags=["Client Management", "Webhooks"],
        summary="List Failed Webhook Logs",
        description="A convenient shortcut to retrieve a list of the 50 most recent failed webhook deliveries.",
    )
    def failed(self, request):
        """Shortcut to inspect failed deliveries."""
        failed_logs = self.get_queryset().filter(status="failed").order_by("-created_at")[
            :50
        ]
        serializer = self.get_serializer(failed_logs, many=True)
        return Response(serializer.data)

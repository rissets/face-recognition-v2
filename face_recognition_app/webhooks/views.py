"""
API views for webhook management.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from django.db.models import Avg, Count, Q
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action, api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, extend_schema_view

from auth_service.authentication import APIKeyAuthentication, JWTClientAuthentication
from .models import WebhookDelivery, WebhookEndpoint, WebhookEvent, WebhookEventLog
from .serializers import (
    WebhookDeliverySerializer,
    WebhookEndpointSerializer,
    WebhookEventLogSerializer,
    WebhookEventSerializer,
    WebhookStatsSerializer,
)


def _parse_iso_date(date_string, default_value):
    """Parse ISO date string, return default if parsing fails"""
    if not date_string:
        return default_value
    
    try:
        parsed = parse_datetime(date_string)
        if parsed:
            return parsed
    except (ValueError, TypeError):
        pass
    
    return default_value
from .services import WebhookService


@extend_schema_view(
    list=extend_schema(
        tags=["Webhooks"],
        summary="List Webhook Events",
        description="Retrieve a list of all possible webhook events that can be subscribed to.",
    ),
    retrieve=extend_schema(
        tags=["Webhooks"],
        summary="Retrieve a Webhook Event",
        description="Get details of a specific webhook event.",
    ),
)
class WebhookEventViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Expose registered webhook event definitions.
    """

    queryset = WebhookEvent.objects.all().order_by('event_name')
    serializer_class = WebhookEventSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = super().get_queryset()
        is_active = self.request.query_params.get('is_active')
        if is_active is not None:
            if is_active.lower() == 'true':
                queryset = queryset.filter(is_active=True)
            elif is_active.lower() == 'false':
                queryset = queryset.filter(is_active=False)
        return queryset


@extend_schema_view(
    list=extend_schema(
        tags=["Webhooks"],
        summary="List Webhook Endpoints",
        description="Retrieve a list of all webhook endpoints configured for the client.",
    ),
    retrieve=extend_schema(
        tags=["Webhooks"],
        summary="Retrieve a Webhook Endpoint",
        description="Get detailed information about a specific webhook endpoint.",
    ),
    create=extend_schema(
        tags=["Webhooks"],
        summary="Create a Webhook Endpoint",
        description="Create a new endpoint to receive webhook notifications.",
    ),
    update=extend_schema(
        tags=["Webhooks"],
        summary="Update a Webhook Endpoint",
        description="Update the configuration of an existing webhook endpoint.",
    ),
    partial_update=extend_schema(
        tags=["Webhooks"],
        summary="Partially Update a Webhook Endpoint",
        description="Partially update the configuration of an existing webhook endpoint.",
    ),
    destroy=extend_schema(
        tags=["Webhooks"],
        summary="Delete a Webhook Endpoint",
        description="Permanently delete a webhook endpoint.",
    ),
)
class WebhookEndpointViewSet(viewsets.ModelViewSet):
    """
    Manage webhook endpoints for the authenticated client.
    """

    queryset = WebhookEndpoint.objects.all()
    serializer_class = WebhookEndpointSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if hasattr(self.request, "client"):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()

    def perform_create(self, serializer):
        secret = serializer.validated_data.get("secret_token") or secrets.token_urlsafe(32)
        serializer.save(client=self.request.client, secret_token=secret)

    @extend_schema(
        tags=["Webhooks"],
        summary="Regenerate Webhook Secret",
        description="Generates a new secret token for a webhook endpoint, invalidating the old one. This is used to sign outgoing webhook payloads.",
    )
    @action(detail=True, methods=["post"])
    def regenerate_secret(self, request, pk=None):
        """Rotate the secret used for signing webhook payloads."""
        endpoint = self.get_object()
        endpoint.secret_token = secrets.token_urlsafe(32)
        endpoint.save(update_fields=["secret_token"])
        return Response(
            {
                "message": "Webhook secret regenerated successfully",
                "secret_token": endpoint.secret_token,
            }
        )

    @extend_schema(
        tags=["Webhooks"],
        summary="Test a Webhook Endpoint",
        description="Sends a test event to the specified webhook endpoint to verify its configuration and connectivity.",
    )
    @action(detail=True, methods=["post"])
    def test(self, request, pk=None):
        """Trigger a test webhook delivery."""
        endpoint = self.get_object()
        serializer = WebhookTestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        event_name = f"test.{serializer.validated_data['test_event_type']}"
        payload = {
            "test": True,
            "timestamp": timezone.now().isoformat(),
            **serializer.validated_data.get("test_data", {}),
        }

        webhook_service = WebhookService()
        webhook_service.send_webhook(
            client_id=request.client.client_id,
            event_name=event_name,
            event_data=payload,
            source="test_interface",
            endpoint_ids=[endpoint.id],
        )

        return Response(
            {
                "message": "Test webhook dispatched",
                "event": event_name,
                "endpoint": endpoint.url,
            }
        )

    @extend_schema(
        tags=["Webhooks", "Analytics"],
        summary="Get Endpoint Statistics",
        description="Retrieve delivery statistics for a specific webhook endpoint, including success rate and average delivery time.",
    )
    @action(detail=True, methods=["get"])
    def stats(self, request, pk=None):
        """Return delivery statistics for an endpoint."""
        endpoint = self.get_object()
        deliveries = endpoint.deliveries.all()

        total = deliveries.count()
        successes = deliveries.filter(status="success").count()
        failures = deliveries.filter(status__in=["failed", "abandoned"]).count()
        pending = deliveries.filter(status__in=["pending", "retrying"]).count()

        success_rate = (successes / total * 100) if total else 0.0
        average_time = (
            deliveries.filter(status="success").aggregate(avg=Avg("response_time_ms"))["avg"] or 0.0
        )

        recent_failures = list(
            deliveries.filter(status__in=["failed", "abandoned"])
            .order_by("-created_at")[:10]
            .values(
                "event_name",
                "response_status_code",
                "error_message",
                "created_at",
            )
        )

        return Response(
            {
                "total_deliveries": total,
                "successful_deliveries": successes,
                "failed_deliveries": failures,
                "pending_deliveries": pending,
                "success_rate": success_rate,
                "avg_delivery_time_ms": average_time,
                "recent_failures": recent_failures,
            }
        )


@extend_schema_view(
    list=extend_schema(
        tags=["Webhooks", "Analytics"],
        summary="List Webhook Event Logs",
        description="Retrieve a list of historical webhook event logs for the client.",
    ),
    retrieve=extend_schema(
        tags=["Webhooks", "Analytics"],
        summary="Retrieve a Webhook Event Log",
        description="Get detailed information about a specific webhook event log.",
    ),
)
class WebhookEventLogViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Inspect historical webhook event logs for the authenticated client.
    """

    queryset = WebhookEventLog.objects.all()
    serializer_class = WebhookEventLogSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if hasattr(self.request, "client"):
            queryset = self.queryset.filter(client=self.request.client)
        else:
            queryset = self.queryset.none()

        event_name = self.request.query_params.get("event_name")
        if event_name:
            queryset = queryset.filter(event_name=event_name)
        return queryset.order_by("-created_at")


@extend_schema_view(
    list=extend_schema(
        tags=["Webhooks", "Analytics"],
        summary="List Webhook Deliveries",
        description="Retrieve a list of all webhook delivery attempts for the client.",
    ),
    retrieve=extend_schema(
        tags=["Webhooks", "Analytics"],
        summary="Retrieve a Webhook Delivery",
        description="Get detailed information about a specific webhook delivery attempt.",
    ),
)
class WebhookDeliveryViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Read-only access to webhook delivery history.
    """

    queryset = WebhookDelivery.objects.all()
    serializer_class = WebhookDeliverySerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if hasattr(self.request, "client"):
            return self.queryset.filter(endpoint__client=self.request.client).order_by("-created_at")
        return self.queryset.none()

    @extend_schema(
        tags=["Webhooks", "Analytics"],
        summary="List Failed Deliveries",
        description="A convenient shortcut to retrieve a list of the 100 most recent failed webhook deliveries.",
    )
    @action(detail=False, methods=["get"])
    def failed(self, request):
        """Return recent failed deliveries."""
        failed_deliveries = self.get_queryset().filter(status__in=["failed", "abandoned"])[:100]
        serializer = self.get_serializer(failed_deliveries, many=True)
        return Response(serializer.data)

    @extend_schema(
        tags=["Webhooks"],
        summary="Retry a Webhook Delivery",
        description="Schedules a retry for a single failed webhook delivery.",
    )
    @action(detail=True, methods=["post"])
    def retry(self, request, pk=None):
        """Retry an individual delivery."""
        delivery = self.get_object()
        if delivery.status not in ["failed", "abandoned", "retrying"]:
            return Response(
                {"error": "Only failed deliveries can be retried"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        webhook_service = WebhookService()
        retried = webhook_service.retry_webhook(delivery)
        serializer = self.get_serializer(retried)
        return Response({"message": "Webhook retry scheduled", "delivery": serializer.data})


# ---------------------------------------------------------------------------#
# Function-based utilities
# ---------------------------------------------------------------------------#


@extend_schema(
    tags=["Webhooks", "Analytics"],
    summary="Get Webhook Statistics",
    description="Retrieve aggregate webhook statistics for the client over a specified time period (default is 30 days).",
)
@api_view(["GET"])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_webhook_stats(request):
    """
    Aggregate webhook statistics for the authenticated client.
    """
    client = getattr(request, "client", None)
    if client is None:
        return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)

    end_date = timezone.now()
    start_date = end_date - timedelta(days=30)

    start_param = request.query_params.get("start_date")
    end_param = request.query_params.get("end_date")
    start_date = _parse_iso_date(start_param, start_date)
    end_date = _parse_iso_date(end_param, end_date)

    events = WebhookEventLog.objects.filter(
        client=client,
        created_at__gte=start_date,
        created_at__lte=end_date,
    )
    events_by_type = events.values("event_name").annotate(count=Count("id")).order_by("-count")

    deliveries = WebhookDelivery.objects.filter(
        endpoint__client=client,
        created_at__gte=start_date,
        created_at__lte=end_date,
    )
    delivery_stats = deliveries.aggregate(
        total=Count("id"),
        successful=Count("id", filter=Q(status="success")),
        failed=Count("id", filter=Q(status__in=["failed", "abandoned"])),
        pending=Count("id", filter=Q(status__in=["pending", "retrying"])),
        avg_time=Avg("response_time_ms"),
    )

    total = delivery_stats["total"] or 0
    success_rate = (delivery_stats["successful"] / total * 100) if total else 0.0

    recent_failures = list(
        deliveries.filter(status__in=["failed", "abandoned"])
        .order_by("-created_at")[:20]
        .values(
            "event_name",
            "endpoint__url",
            "response_status_code",
            "error_message",
            "created_at",
        )
    )

    payload = {
        "total_events": events.count(),
        "total_deliveries": total,
        "successful_deliveries": delivery_stats["successful"] or 0,
        "failed_deliveries": delivery_stats["failed"] or 0,
        "pending_deliveries": delivery_stats["pending"] or 0,
        "success_rate": success_rate,
        "avg_delivery_time": delivery_stats["avg_time"] or 0.0,
        "events_by_type": {item["event_name"]: item["count"] for item in events_by_type},
        "recent_failures": recent_failures,
    }
    serializer = WebhookStatsSerializer(payload)
    return Response(serializer.data)


@extend_schema(
    tags=["Webhooks"],
    summary="Retry All Failed Webhooks",
    description="Initiates a retry process for the 50 oldest failed webhook deliveries for the client.",
)
@api_view(["POST"])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def retry_failed_webhooks(request):
    """
    Retry failed webhook deliveries for the authenticated client.
    """
    client = getattr(request, "client", None)
    if client is None:
        return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)

    failed_deliveries = WebhookDelivery.objects.filter(
        endpoint__client=client,
        status__in=['failed', 'abandoned'],
    ).order_by("created_at")[:50]

    webhook_service = WebhookService()
    retry_count = 0
    for delivery in failed_deliveries:
        webhook_service.retry_webhook(delivery)
        retry_count += 1

    return Response(
        {
            "message": f"Initiated retry for {retry_count} deliveries",
            "retry_count": retry_count,
        }
    )


@extend_schema(
    tags=["Webhooks"],
    summary="Clear Old Webhook Logs",
    description="Permanently deletes webhook event and delivery logs older than a specified number of days (default is 90).",
)
@api_view(["DELETE"])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def clear_old_webhook_logs(request):
    """
    Remove webhook logs older than the specified number of days.
    """
    client = getattr(request, "client", None)
    if client is None:
        return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)

    try:
        days = int(request.query_params.get("days", 90))
    except ValueError:
        return Response({"error": "days must be an integer"}, status=status.HTTP_400_BAD_REQUEST)

    cutoff = timezone.now() - timedelta(days=days)
    deleted_events = WebhookEventLog.objects.filter(client=client, created_at__lt=cutoff).delete()[0]
    deleted_deliveries = WebhookDelivery.objects.filter(
        endpoint__client=client,
        created_at__lt=cutoff,
    ).delete()[0]

    return Response(
        {
            "message": f"Cleared webhook logs older than {days} days",
            "deleted_event_logs": deleted_events,
            "deleted_deliveries": deleted_deliveries,
        }
    )

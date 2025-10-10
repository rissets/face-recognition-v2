"""
API views for webhook management.
"""
from __future__ import annotations

import secrets
from datetime import timedelta

from django.db.models import Avg, Count, Q
from django.utils import timezone
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action, api_view, authentication_classes, permission_classes
from rest_framework.response import Response

from auth_service.authentication import APIKeyAuthentication, JWTClientAuthentication
from .models import WebhookDelivery, WebhookEndpoint, WebhookEvent, WebhookEventLog
from .serializers import (
    WebhookDeliverySerializer,
    WebhookEndpointSerializer,
    WebhookEventLogSerializer,
    WebhookEventSerializer,
    WebhookStatsSerializer,
    WebhookTestSerializer,
)
from .services import WebhookService


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

    @action(detail=False, methods=["get"])
    def failed(self, request):
        """Return recent failed deliveries."""
        failed_deliveries = self.get_queryset().filter(status__in=["failed", "abandoned"])[:100]
        serializer = self.get_serializer(failed_deliveries, many=True)
        return Response(serializer.data)

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


def _parse_iso_date(value, default):
    if not value:
        return default
    try:
        parsed = timezone.datetime.fromisoformat(value)
        if timezone.is_naive(parsed):
            parsed = timezone.make_aware(parsed)
        return parsed
    except ValueError:
        return default


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

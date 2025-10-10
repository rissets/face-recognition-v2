"""from django.shortcuts import render

API views for webhook management

"""# Create your views here.

from rest_framework import viewsets, status, permissions
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from django.db.models import Count, Q, Avg
from django.utils import timezone
from datetime import timedelta
import secrets

from .models import WebhookEvent, WebhookEndpoint, WebhookDelivery, WebhookEventLog
from .serializers import (
    WebhookEventSerializer, WebhookEndpointSerializer, WebhookDeliverySerializer,
    WebhookEventLogSerializer, WebhookTestSerializer, WebhookStatsSerializer
)
from .services import WebhookService
from auth_service.authentication import APIKeyAuthentication, JWTClientAuthentication


class WebhookEventViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for webhook events
    Read-only access to event history
    """
    queryset = WebhookEvent.objects.all()
    serializer_class = WebhookEventSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter events by client"""
        if hasattr(self.request, 'client'):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()


class WebhookEndpointViewSet(viewsets.ModelViewSet):
    """
    ViewSet for webhook endpoint management
    """
    queryset = WebhookEndpoint.objects.all()
    serializer_class = WebhookEndpointSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter endpoints by client"""
        if hasattr(self.request, 'client'):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()
    
    def perform_create(self, serializer):
        """Set client and generate secret when creating endpoint"""
        if not serializer.validated_data.get('secret'):
            serializer.validated_data['secret'] = secrets.token_urlsafe(32)
        serializer.save(client=self.request.client)
    
    @action(detail=True, methods=['post'])
    def test(self, request, pk=None):
        """Test webhook endpoint with a sample event"""
        endpoint = self.get_object()
        serializer = WebhookTestSerializer(data=request.data)
        
        if serializer.is_valid():
            # Create a test event
            test_event = WebhookEvent.objects.create(
                event_type=f"test.{serializer.validated_data['test_event_type']}",
                event_data={
                    'test': True,
                    'timestamp': timezone.now().isoformat(),
                    **serializer.validated_data.get('test_data', {})
                },
                client=self.request.client,
                resource_id='test_resource'
            )
            
            # Send webhook
            webhook_service = WebhookService()
            delivery = webhook_service.send_webhook(test_event, endpoint)
            
            return Response({
                'message': 'Test webhook sent',
                'event_id': test_event.id,
                'delivery_id': delivery.id,
                'status': delivery.status,
                'http_status': delivery.http_status,
                'response': delivery.response_body
            })
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def regenerate_secret(self, request, pk=None):
        """Regenerate webhook secret"""
        endpoint = self.get_object()
        endpoint.secret = secrets.token_urlsafe(32)
        endpoint.save()
        
        return Response({
            'message': 'Webhook secret regenerated successfully',
            'secret': endpoint.secret
        })
    
    @action(detail=True, methods=['get'])
    def stats(self, request, pk=None):
        """Get statistics for this endpoint"""
        endpoint = self.get_object()
        
        # Calculate delivery statistics
        deliveries = endpoint.deliveries.all()
        total_deliveries = deliveries.count()
        
        if total_deliveries > 0:
            successful_deliveries = deliveries.filter(status='delivered').count()
            failed_deliveries = deliveries.filter(status='failed').count()
            pending_deliveries = deliveries.filter(status='pending').count()
            success_rate = (successful_deliveries / total_deliveries) * 100
            avg_delivery_time = deliveries.filter(
                status='delivered'
            ).aggregate(
                avg_time=Avg('processing_time')
            )['avg_time'] or 0.0
        else:
            successful_deliveries = failed_deliveries = pending_deliveries = 0
            success_rate = avg_delivery_time = 0.0
        
        # Get recent failures
        recent_failures = list(
            endpoint.deliveries.filter(
                status='failed',
                created_at__gte=timezone.now() - timedelta(days=7)
            ).values(
                'event__event_type', 'http_status', 'response_body', 'created_at'
            ).order_by('-created_at')[:10]
        )
        
        stats_data = {
            'total_deliveries': total_deliveries,
            'successful_deliveries': successful_deliveries,
            'failed_deliveries': failed_deliveries,
            'pending_deliveries': pending_deliveries,
            'success_rate': success_rate,
            'avg_delivery_time': avg_delivery_time,
            'recent_failures': recent_failures
        }
        
        return Response(stats_data)


class WebhookDeliveryViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for webhook delivery logs
    Read-only access to delivery history
    """
    queryset = WebhookDelivery.objects.all()
    serializer_class = WebhookDeliverySerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter deliveries by client"""
        if hasattr(self.request, 'client'):
            return self.queryset.filter(
                endpoint__client=self.request.client
            )
        return self.queryset.none()
    
    @action(detail=False, methods=['get'])
    def failed(self, request):
        """Get failed deliveries for retry"""
        failed_deliveries = self.get_queryset().filter(
            status='failed'
        ).order_by('-created_at')[:100]
        
        serializer = self.get_serializer(failed_deliveries, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def retry(self, request, pk=None):
        """Retry a failed webhook delivery"""
        delivery = self.get_object()
        
        if delivery.status != 'failed':
            return Response({
                'error': 'Can only retry failed deliveries'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Retry the webhook
        webhook_service = WebhookService()
        new_delivery = webhook_service.retry_webhook(delivery)
        
        serializer = self.get_serializer(new_delivery)
        return Response({
            'message': 'Webhook retry initiated',
            'delivery': serializer.data
        })


@api_view(['GET'])
def get_webhook_stats(request):
    """
    Get comprehensive webhook statistics for the client
    """
    client = request.client
    
    # Date range (last 30 days by default)
    end_date = timezone.now()
    start_date = end_date - timedelta(days=30)
    
    # Get query params for date filtering
    start_param = request.GET.get('start_date')
    end_param = request.GET.get('end_date')
    
    if start_param:
        start_date = timezone.datetime.fromisoformat(start_param)
    if end_param:
        end_date = timezone.datetime.fromisoformat(end_param)
    
    # Calculate event statistics
    events = client.webhook_events.filter(
        created_at__range=[start_date, end_date]
    )
    
    events_by_type = events.values('event_type').annotate(
        count=Count('id')
    ).order_by('-count')
    
    # Calculate delivery statistics
    deliveries = WebhookDelivery.objects.filter(
        endpoint__client=client,
        created_at__range=[start_date, end_date]
    )
    
    delivery_stats = deliveries.aggregate(
        total=Count('id'),
        successful=Count('id', filter=Q(status='delivered')),
        failed=Count('id', filter=Q(status='failed')),
        pending=Count('id', filter=Q(status='pending')),
        avg_delivery_time=Avg('processing_time')
    )
    
    # Calculate success rate
    success_rate = 0.0
    if delivery_stats['total'] > 0:
        success_rate = (delivery_stats['successful'] / delivery_stats['total']) * 100
    
    # Get recent failures
    recent_failures = list(
        deliveries.filter(status='failed').order_by('-created_at')[:20].values(
            'event__event_type', 'endpoint__url', 'http_status', 
            'response_body', 'created_at'
        )
    )
    
    stats_data = {
        'total_events': events.count(),
        'total_deliveries': delivery_stats['total'] or 0,
        'successful_deliveries': delivery_stats['successful'] or 0,
        'failed_deliveries': delivery_stats['failed'] or 0,
        'pending_deliveries': delivery_stats['pending'] or 0,
        'success_rate': success_rate,
        'avg_delivery_time': delivery_stats['avg_delivery_time'] or 0.0,
        'events_by_type': {
            item['event_type']: item['count'] 
            for item in events_by_type
        },
        'recent_failures': recent_failures
    }
    
    serializer = WebhookStatsSerializer(stats_data)
    return Response(serializer.data)


@api_view(['POST'])
def retry_failed_webhooks(request):
    """
    Retry all failed webhook deliveries for the client
    """
    client = request.client
    
    # Get all failed deliveries
    failed_deliveries = WebhookDelivery.objects.filter(
        endpoint__client=client,
        status='failed'
    )
    
    webhook_service = WebhookService()
    retry_count = 0
    
    for delivery in failed_deliveries[:50]:  # Limit to 50 retries per request
        webhook_service.retry_webhook(delivery)
        retry_count += 1
    
    return Response({
        'message': f'Initiated retry for {retry_count} failed webhook deliveries',
        'retry_count': retry_count
    })


@api_view(['DELETE'])
def clear_old_webhook_logs(request):
    """
    Clear old webhook logs (older than specified days)
    """
    client = request.client
    days = int(request.query_params.get('days', 90))
    
    cutoff_date = timezone.now() - timedelta(days=days)
    
    # Delete old events and deliveries
    deleted_events = client.webhook_events.filter(
        created_at__lt=cutoff_date
    ).delete()[0]
    
    deleted_deliveries = WebhookDelivery.objects.filter(
        endpoint__client=client,
        created_at__lt=cutoff_date
    ).delete()[0]
    
    return Response({
        'message': f'Cleared webhook logs older than {days} days',
        'deleted_events': deleted_events,
        'deleted_deliveries': deleted_deliveries
    })
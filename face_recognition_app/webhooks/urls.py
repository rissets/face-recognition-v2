"""
URL configuration for webhooks app
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'webhooks'

# Create router for ViewSets
router = DefaultRouter()
router.register(r'events', views.WebhookEventViewSet)
router.register(r'endpoints', views.WebhookEndpointViewSet)
router.register(r'deliveries', views.WebhookDeliveryViewSet)

urlpatterns = [
    # ViewSet routes
    path('', include(router.urls)),
    
    # Statistics and management
    path('stats/', views.get_webhook_stats, name='webhook_stats'),
    path('retry-failed/', views.retry_failed_webhooks, name='retry_failed_webhooks'),
    path('clear-logs/', views.clear_old_webhook_logs, name='clear_old_webhook_logs'),
]
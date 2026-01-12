from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import WebhookEvent, WebhookEndpoint, WebhookDelivery

@admin.register(WebhookEvent)
class WebhookEventAdmin(ModelAdmin):
    list_display = ['event_type', 'client', 'created_at', 'processed']
    list_filter = ['event_type', 'processed', 'created_at']
    search_fields = ['event_type', 'client__name']
    readonly_fields = ['id', 'created_at']

@admin.register(WebhookEndpoint)
class WebhookEndpointAdmin(ModelAdmin):
    list_display = [
        'name', 'client', 'url', 'status', 'total_deliveries', 
        'successful_deliveries', 'failed_deliveries'
    ]
    list_filter = ['status', 'client']
    search_fields = ['name', 'client__name', 'url']
    readonly_fields = [
        'id', 'total_deliveries', 'successful_deliveries', 
        'failed_deliveries', 'created_at', 'updated_at', 'last_delivery_at'
    ]

@admin.register(WebhookDelivery)
class WebhookDeliveryAdmin(ModelAdmin):
    list_display = [
        'webhook_event', 'endpoint', 'status', 'attempt_count', 
        'created_at', 'delivered_at'
    ]
    list_filter = ['status', 'created_at']
    search_fields = ['webhook_event__event_type', 'endpoint__name']
    readonly_fields = ['id', 'created_at', 'delivered_at']

from .models import WebhookEvent, WebhookEndpoint, WebhookDelivery

from django.contrib import admin

from django.utils.html import format_html

from unfold.admin import ModelAdmin
from .models import WebhookEvent, WebhookEndpoint, WebhookDelivery


@admin.register(WebhookEvent)
class WebhookEventAdmin(ModelAdmin):
    """Admin for webhook events"""
    list_display = ('event_name', 'description', 'is_active', 'created_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('event_name', 'description')
    readonly_fields = ('id', 'created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Event Information', {
            'fields': ('event_name', 'description', 'is_active')
        }),
        ('Configuration', {
            'fields': ('payload_schema',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(WebhookEndpoint)
class WebhookEndpointAdmin(ModelAdmin):
    """Admin for webhook endpoints"""
    list_display = ('name', 'client', 'url_short', 'status', 'success_rate_formatted', 'total_deliveries', 'created_at')
    list_filter = ('status', 'created_at', 'client__tier')
    search_fields = ('name', 'url', 'client__name', 'client__client_id')
    readonly_fields = ('created_at', 'updated_at', 'last_delivery_at', 'total_deliveries', 'successful_deliveries', 'failed_deliveries')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Endpoint Information', {
            'fields': ('client', 'name', 'url', 'status')
        }),
        ('Event Subscriptions', {
            'fields': ('subscribed_events',)
        }),
        ('Security Configuration', {
            'fields': ('secret_token',),
            'classes': ('collapse',)
        }),
        ('Retry Configuration', {
            'fields': ('max_retries', 'retry_delay_seconds'),
            'classes': ('collapse',)
        }),
        ('Statistics', {
            'fields': ('total_deliveries', 'successful_deliveries', 'failed_deliveries'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'last_delivery_at'),
            'classes': ('collapse',)
        })
    )
    
    def url_short(self, obj):
        """Display shortened URL"""
        return f"{obj.url[:30]}..." if len(obj.url) > 30 else obj.url
    url_short.short_description = "URL"
    
    def success_rate_formatted(self, obj):
        """Display success rate with color coding"""
        try:
            rate = obj.success_rate
            if rate is None:
                return "-"
            rate_value = float(rate)
        except (ValueError, TypeError, AttributeError):
            return "-"
            
        if rate_value >= 90:
            color = 'green'
        elif rate_value >= 70:
            color = 'orange'
        else:
            color = 'red'
            
        return f"{rate_value:.1f}%"
    success_rate_formatted.short_description = "Success Rate"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('client')


@admin.register(WebhookDelivery)
class WebhookDeliveryAdmin(ModelAdmin):
    """Admin for webhook deliveries"""
    list_display = ('endpoint_name', 'event_name', 'status', 'attempt_number', 'response_status_code', 'response_time_display', 'created_at')
    list_filter = ('status', 'event_name', 'response_status_code', 'created_at')
    search_fields = ('endpoint__name', 'endpoint__client__name', 'event_name')
    readonly_fields = ('created_at', 'delivered_at', 'completed_at', 'response_time_ms')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Delivery Information', {
            'fields': ('endpoint', 'event_name', 'status')
        }),
        ('HTTP Configuration', {
            'fields': ('http_method', 'headers'),
            'classes': ('collapse',)
        }),
        ('Response Details', {
            'fields': ('response_status_code', 'response_headers', 'response_body', 'response_time_ms')
        }),
        ('Retry Configuration', {
            'fields': ('attempt_number', 'max_attempts', 'next_retry_at'),
            'classes': ('collapse',)
        }),
        ('Error Information', {
            'fields': ('error_message', 'error_code'),
            'classes': ('collapse',)
        }),
        ('Event Data', {
            'fields': ('event_data',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'delivered_at', 'completed_at'),
            'classes': ('collapse',)
        })
    )
    
    def endpoint_name(self, obj):
        """Display endpoint name with client"""
        return f"{obj.endpoint.name} ({obj.endpoint.client.name})" if obj.endpoint and obj.endpoint.client else "-"
    endpoint_name.short_description = "Endpoint"
    
    def response_time_display(self, obj):
        """Display response time with formatting"""
        if obj.response_time_ms:
            if obj.response_time_ms < 500:
                color = 'green'
            elif obj.response_time_ms < 2000:
                color = 'orange'
            else:
                color = 'red'
            return format_html(
                '<span style="color: {};">{:.0f}ms</span>',
                color,
                obj.response_time_ms
            )
        return "-"
    response_time_display.short_description = "Response Time"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('endpoint', 'endpoint__client')
from .models import Client, ClientUser, ClientAPIUsage, ClientWebhookLog

from django.contrib import admin

from django.utils.html import format_html

from django.db.models import Count
from unfold.admin import ModelAdmin, TabularInline 


@admin.register(Client)
class ClientAdmin(ModelAdmin):
    list_display = ('name', 'client_id', 'tier', 'status', 'rate_limit_per_hour', 'total_users', 'created_at')
    list_filter = ('tier', 'status', 'created_at')
    search_fields = ('name', 'client_id', 'contact_name', 'contact_email')
    readonly_fields = ('client_id', 'api_key', 'secret_key', 'webhook_secret', 'created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'contact_name', 'contact_email')
        }),
        ('API Configuration', {
            'fields': ('client_id', 'api_key', 'secret_key', 'tier', 'status')
        }),
        ('Rate Limiting', {
            'fields': ('rate_limit_per_hour', 'rate_limit_per_day'),
            'classes': ('collapse',)
        }),
        ('Webhook Configuration', {
            'fields': ('webhook_url', 'webhook_secret'),
            'classes': ('collapse',)
        }),
        ('Domain & Security', {
            'fields': ('domain', 'allowed_domains'),
            'classes': ('collapse',)
        }),
        ('Features & Settings', {
            'fields': ('features',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def total_users(self, obj):
        """Display total users for this client"""
        return obj.users.count()
    total_users.short_description = "Total Users"

    def client_id(self, obj):
        """Display client ID with formatting"""
        return obj.client_id
    client_id.short_description = "Client ID"
    
    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related('users')


@admin.register(ClientUser)
class ClientUserAdmin(ModelAdmin):
    list_display = ('external_user_id', 'client', 'user_profile_name', 'is_enrolled', 'face_auth_enabled', 'created_at')
    list_filter = ('is_enrolled', 'face_auth_enabled', 'created_at', 'client__tier')
    search_fields = ('external_user_id', 'client__name', 'client__client_id')
    readonly_fields = ('id', 'created_at', 'updated_at', 'last_recognition_at', 'enrollment_completed_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('User Identification', {
            'fields': ('client', 'external_user_id', 'external_user_uuid')
        }),
        ('Face Recognition Status', {
            'fields': ('is_enrolled', 'enrollment_completed_at', 'face_auth_enabled', 'last_recognition_at')
        }),
        ('User Profile Data', {
            'fields': ('profile',),
            'classes': ('collapse',)
        }),
        ('Additional Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def user_profile_name(self, obj):
        """Display user name from profile"""
        if obj.profile and obj.profile.get('name'):
            return obj.profile.get('name') or obj.external_user_id
        return "-"
    user_profile_name.short_description = "Profile Name"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('client')


@admin.register(ClientAPIUsage)
class ClientAPIUsageAdmin(ModelAdmin):
    """Admin for API usage tracking"""
    list_display = ('client', 'endpoint', 'method', 'status_code', 'response_time_display', 'created_at')
    list_filter = ('endpoint', 'method', 'status_code', 'created_at')
    search_fields = ('client__name', 'client__client_id', 'ip_address')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Request Information', {
            'fields': ('client', 'endpoint', 'method', 'status_code')
        }),
        ('Network Details', {
            'fields': ('ip_address', 'user_agent'),
            'classes': ('collapse',)
        }),
        ('Performance Metrics', {
            'fields': ('response_time_ms',)
        }),
        ('Additional Data', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',)
        })
    )
    
    def response_time_display(self, obj):
        """Display response time with formatting"""
        if obj.response_time_ms:
            if obj.response_time_ms < 100:
                color = 'green'
            elif obj.response_time_ms < 500:
                color = 'orange'  
            else:
                color = 'red'
            return obj.response_time_ms
        return "-"
    response_time_display.short_description = "Response Time"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('client')


@admin.register(ClientWebhookLog)
class ClientWebhookLogAdmin(ModelAdmin):
    """Admin for webhook delivery logs"""
    list_display = ('client', 'event_type', 'status', 'attempt_count', 'response_status_code', 'created_at')
    list_filter = ('status', 'event_type', 'created_at', 'response_status_code')
    search_fields = ('client__name', 'client__client_id', 'event_type')
    readonly_fields = ('created_at', 'delivered_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Webhook Information', {
            'fields': ('client', 'event_type', 'status')
        }),
        ('Delivery Details', {
            'fields': ('attempt_count', 'max_attempts', 'response_status_code', 'next_retry_at')
        }),
        ('Response Information', {
            'fields': ('response_body', 'error_message'),
            'classes': ('collapse',)
        }),
        ('Event Payload', {
            'fields': ('payload',),
            'classes': ('collapse',)
        }),
        ('Timing Information', {
            'fields': ('created_at', 'delivered_at'),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('client')
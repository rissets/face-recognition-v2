from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline
from .models import Client, ClientUser, ClientAPIUsage, ClientWebhookLog

@admin.register(Client)
class ClientAdmin(ModelAdmin):
    list_display = [
        'name', 'client_id', 'tier', 'status', 'rate_limit_per_hour', 
        'created_at', 'updated_at'
    ]
    list_filter = ['tier', 'status', 'created_at']
    search_fields = ['name', 'client_id', 'contact_email']
    readonly_fields = [
        'client_id', 'api_key', 'secret_key', 'webhook_secret', 
        'created_at', 'updated_at'
    ]
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'tier', 'status', 'contact_name', 'contact_email')
        }),
        ('API Configuration', {
            'fields': ('client_id', 'api_key', 'secret_key', 'rate_limit_per_hour')
        }),
        ('Webhook Configuration', {
            'fields': ('webhook_url', 'webhook_secret'),
            'classes': ['collapse']
        }),
        ('Features & Metadata', {
            'fields': ('features', 'metadata'),
            'classes': ['collapse']
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ['collapse']
        }),
    )

class ClientUserInline(TabularInline):
    model = ClientUser
    extra = 0
    fields = ['external_user_id', 'is_enrolled', 'face_auth_enabled', 'created_at']
    readonly_fields = ['created_at']

@admin.register(ClientUser)  
class ClientUserAdmin(ModelAdmin):
    list_display = [
        'external_user_id', 'client', 'is_enrolled', 'face_auth_enabled', 
        'enrollment_completed_at', 'last_recognition_at'
    ]
    list_filter = ['is_enrolled', 'face_auth_enabled', 'client']
    search_fields = ['external_user_id', 'client__name']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
@admin.register(ClientAPIUsage)
class ClientAPIUsageAdmin(ModelAdmin):
    list_display = ['client', 'endpoint', 'method', 'timestamp', 'response_status']
    list_filter = ['method', 'response_status', 'timestamp']
    search_fields = ['client__name', 'endpoint']
    readonly_fields = ['timestamp']
    
@admin.register(ClientWebhookLog)
class ClientWebhookLogAdmin(ModelAdmin):
    list_display = [
        'client', 'event_type', 'status', 'attempt_count', 
        'created_at', 'last_attempt_at'
    ]
    list_filter = ['event_type', 'status', 'created_at']
    search_fields = ['client__name', 'event_type']
    readonly_fields = ['created_at', 'last_attempt_at']
"""
Core admin configuration for third-party authentication service
Redesigned for multi-client architecture with Django Unfold
"""
from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import SystemConfiguration, AuditLog, SecurityEvent, HealthCheck


@admin.register(SystemConfiguration)
class SystemConfigurationAdmin(ModelAdmin):
    """Admin for system configuration"""
    list_display = ('key', 'description', 'is_encrypted', 'is_active', 'created_at')
    list_filter = ('is_encrypted', 'is_active', 'created_at')
    search_fields = ('key', 'description')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Configuration', {
            'fields': ('key', 'value', 'description', 'is_encrypted')
        }),
        ('Status', {
            'fields': ('is_active',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def has_add_permission(self, request):
        """Only superusers can add system configurations"""
        return request.user.is_superuser
    
    def has_change_permission(self, request, obj=None):
        """Only superusers can change system configurations"""
        return request.user.is_superuser


@admin.register(AuditLog)
class AuditLogAdmin(ModelAdmin):
    """Admin for audit logs with client context"""
    list_display = ('action', 'client', 'client_user', 'resource_type', 'success', 'created_at')
    list_filter = ('action', 'resource_type', 'success', 'created_at', 'client__name')
    search_fields = ('action', 'client__name', 'client_user__username', 'resource_type', 'ip_address')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Action Details', {
            'fields': ('client', 'client_user', 'action', 'resource_type', 'resource_id', 'success')
        }),
        ('Request Info', {
            'fields': ('ip_address', 'user_agent')
        }),
        ('Details', {
            'fields': ('details', 'error_message'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def has_add_permission(self, request):
        """Audit logs are system generated"""
        return False
    
    def has_change_permission(self, request, obj=None):
        """Audit logs are immutable"""
        return False


@admin.register(SecurityEvent)
class SecurityEventAdmin(ModelAdmin):
    """Admin for security events with client context"""
    list_display = ('event_type', 'severity', 'client', 'client_user', 'resolved', 'created_at')
    list_filter = ('event_type', 'severity', 'resolved', 'created_at', 'client__name')
    search_fields = ('event_type', 'client__name', 'client_user__username', 'ip_address')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Event Details', {
            'fields': ('event_type', 'severity', 'client', 'client_user')
        }),
        ('Request Info', {
            'fields': ('ip_address', 'user_agent', 'details')
        }),
        ('Resolution', {
            'fields': ('resolved', 'resolved_at', 'resolved_by_client', 'resolved_by_client_user')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    actions = ['mark_resolved']
    
    def mark_resolved(self, request, queryset):
        """Mark selected security events as resolved"""
        from django.utils import timezone
        updated = queryset.update(
            resolved=True,
            resolved_at=timezone.now()
        )
        self.message_user(request, f'{updated} security events marked as resolved.')
    mark_resolved.short_description = "Mark selected events as resolved"


@admin.register(HealthCheck)
class HealthCheckAdmin(ModelAdmin):
    """Admin for health check logs"""
    list_display = ('service_name', 'status', 'response_time', 'created_at')
    list_filter = ('status', 'service_name', 'created_at')
    search_fields = ('service_name',)
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Service Info', {
            'fields': ('service_name', 'status', 'response_time')
        }),
        ('Error Info', {
            'fields': ('error_message',),
            'classes': ('collapse',)
        }),
        ('Details', {
            'fields': ('details',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def has_add_permission(self, request):
        """Health checks are system generated"""
        return False

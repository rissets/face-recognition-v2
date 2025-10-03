"""
Core admin configuration
"""
from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import SystemConfiguration, AuditLog, SecurityEvent, HealthCheck


@admin.register(SystemConfiguration)
class SystemConfigurationAdmin(ModelAdmin):
    """Admin for system configuration"""
    list_display = ('key', 'description', 'is_encrypted', 'created_at')
    list_filter = ('is_encrypted', 'created_at')
    search_fields = ('key', 'description')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Configuration', {
            'fields': ('key', 'value', 'description', 'is_encrypted')
        }),
        ('Metadata', {
            'fields': ('is_active', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(AuditLog)
class AuditLogAdmin(ModelAdmin):
    """Admin for audit logs"""
    list_display = ('action', 'user', 'resource_type', 'success', 'created_at')
    list_filter = ('action', 'resource_type', 'success', 'created_at')
    search_fields = ('action', 'user__email', 'resource_type', 'ip_address')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Action Details', {
            'fields': ('user', 'action', 'resource_type', 'resource_id', 'success')
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


@admin.register(SecurityEvent)
class SecurityEventAdmin(ModelAdmin):
    """Admin for security events"""
    list_display = ('event_type', 'severity', 'user', 'resolved', 'created_at')
    list_filter = ('event_type', 'severity', 'resolved', 'created_at')
    search_fields = ('user__email', 'ip_address')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    actions = ['mark_resolved']
    
    fieldsets = (
        ('Event Details', {
            'fields': ('event_type', 'severity', 'user', 'ip_address')
        }),
        ('Details', {
            'fields': ('details', 'user_agent')
        }),
        ('Resolution', {
            'fields': ('resolved', 'resolved_at', 'resolved_by', 'resolution_notes')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def mark_resolved(self, request, queryset):
        """Mark security events as resolved"""
        from django.utils import timezone
        
        updated = queryset.update(
            resolved=True,
            resolved_at=timezone.now(),
            resolved_by=request.user
        )
        
        self.message_user(
            request,
            f'{updated} security events marked as resolved.'
        )
    
    mark_resolved.short_description = "Mark selected events as resolved"


@admin.register(HealthCheck)
class HealthCheckAdmin(ModelAdmin):
    """Admin for health checks"""
    list_display = ('service_name', 'status', 'response_time', 'created_at')
    list_filter = ('service_name', 'status', 'created_at')
    search_fields = ('service_name',)
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Service Details', {
            'fields': ('service_name', 'status', 'response_time')
        }),
        ('Details', {
            'fields': ('error_message', 'details'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
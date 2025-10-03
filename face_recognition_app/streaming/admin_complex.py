"""
Streaming admin configuration
"""
from django.contrib import admin
from django.utils.html import format_html
from django.utils import timezone
from unfold.admin import ModelAdmin
from .models import StreamingSession, WebRTCSignal


@admin.register(WebRTCSignal)
class WebRTCSignalAdmin(ModelAdmin):
    """Admin for WebRTC signals"""
    list_display = ('session', 'signal_type', 'created_at')
    list_filter = ('signal_type', 'created_at')
    search_fields = ('session__session_token', 'session__user__email')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Signal Info', {
            'fields': ('session', 'signal_type')
        }),
        ('Signal Data', {
            'fields': ('signal_data',),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('session', 'session__user')


@admin.register(StreamingSession)
class StreamingSessionAdmin(ModelAdmin):
    """Admin for streaming sessions"""
    list_display = ('user', 'session_type', 'status', 'frames_processed', 'created_at', 'quality_display')
    list_filter = ('session_type', 'status', 'created_at')
    search_fields = ('user__email', 'session_id', 'user__first_name', 'user__last_name')
    readonly_fields = ('session_id', 'created_at', 'updated_at', 'ended_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Session Info', {
            'fields': ('user', 'session_id', 'session_type', 'status')
        }),
        ('Processing Stats', {
            'fields': ('frames_processed', 'frames_analyzed', 'average_quality')
        }),
        ('Results', {
            'fields': ('processing_results', 'error_message')
        }),
        ('Configuration', {
            'fields': ('stream_config',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'ended_at'),
            'classes': ('collapse',)
        })
    )
    
    def quality_display(self, obj):
        """Display average quality with color coding"""
        if obj.average_quality is None:
            return "N/A"
        
        quality = obj.average_quality
        if quality >= 0.8:
            color = 'green'
        elif quality >= 0.6:
            color = 'orange'
        else:
            color = 'red'
        
        return format_html(
            '<span style="color: {};">{:.2f}</span>',
            color,
            quality
        )
    quality_display.short_description = 'Avg Quality'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')
    
    actions = ['restart_failed_sessions']
    
    def restart_failed_sessions(self, request, queryset):
        """Restart failed sessions"""
        failed_sessions = queryset.filter(status='failed')
        updated = failed_sessions.update(
            status='pending',
            error_message=None,
            updated_at=timezone.now()
        )
        self.message_user(request, f'{updated} failed sessions restarted.')
    restart_failed_sessions.short_description = "Restart failed sessions"



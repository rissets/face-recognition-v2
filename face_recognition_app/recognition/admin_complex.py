"""
Recognition admin configuration
"""
from django.contrib import admin
from django.utils.html import format_html
from django.utils import timezone
from unfold.admin import ModelAdmin
from .models import (
    FaceEmbedding, EnrollmentSession, AuthenticationAttempt, 
    LivenessDetection, ObstacleDetection, FaceRecognitionModel
)


@admin.register(FaceEmbedding)
class FaceEmbeddingAdmin(ModelAdmin):
    """Admin for face embeddings"""
    list_display = ('user', 'quality_score', 'confidence_score', 'created_at', 'is_active')
    list_filter = ('is_active', 'created_at', 'updated_at')
    search_fields = ('user__email', 'user__first_name', 'user__last_name')
    readonly_fields = ('created_at', 'updated_at', 'embedding_version')
    ordering = ('-created_at',)
    
    fieldsets = (
        ('User', {
            'fields': ('user',)
        }),
        ('Embedding Data', {
            'fields': ('embedding_data', 'embedding_version'),
            'classes': ('collapse',)
        }),
        ('Quality Metrics', {
            'fields': ('quality_score', 'confidence_score')
        }),
        ('Face Details', {
            'fields': ('face_landmarks', 'face_area', 'detection_confidence')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at', 'is_active'),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


@admin.register(EnrollmentSession)
class EnrollmentSessionAdmin(ModelAdmin):
    """Admin for enrollment sessions"""
    list_display = ('user', 'session_id', 'status', 'quality_score', 'created_at', 'completed_at')
    list_filter = ('status', 'created_at', 'completed_at')
    search_fields = ('user__email', 'session_id', 'user__first_name', 'user__last_name')
    readonly_fields = ('session_id', 'created_at', 'updated_at', 'completed_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Session Info', {
            'fields': ('user', 'session_id', 'status')
        }),
        ('Progress', {
            'fields': ('frames_captured', 'frames_processed', 'quality_score')
        }),
        ('Results', {
            'fields': ('enrollment_data', 'error_message')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'completed_at'),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')
    
    actions = ['mark_completed', 'mark_failed']
    
    def mark_completed(self, request, queryset):
        """Mark sessions as completed"""
        updated = queryset.filter(status='in_progress').update(
            status='completed',
            completed_at=timezone.now()
        )
        self.message_user(request, f'{updated} sessions marked as completed.')
    mark_completed.short_description = "Mark selected sessions as completed"
    
    def mark_failed(self, request, queryset):
        """Mark sessions as failed"""
        updated = queryset.filter(status='in_progress').update(
            status='failed',
            completed_at=timezone.now()
        )
        self.message_user(request, f'{updated} sessions marked as failed.')
    mark_failed.short_description = "Mark selected sessions as failed"


@admin.register(AuthenticationAttempt)
class AuthenticationAttemptAdmin(ModelAdmin):
    """Admin for authentication attempts"""
    list_display = ('user', 'status', 'confidence_score', 'ip_address', 'created_at', 'colored_status')
    list_filter = ('status', 'liveness_passed', 'obstacle_detected', 'created_at')
    search_fields = ('user__email', 'ip_address', 'user_agent', 'user__first_name', 'user__last_name')
    readonly_fields = ('created_at', 'processing_time')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Authentication Info', {
            'fields': ('user', 'status', 'confidence_score')
        }),
        ('Quality Checks', {
            'fields': ('quality_score', 'liveness_passed', 'obstacle_detected')
        }),
        ('Request Details', {
            'fields': ('ip_address', 'user_agent', 'device_info')
        }),
        ('Processing', {
            'fields': ('processing_time', 'error_message')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )
    
    def colored_status(self, obj):
        """Display status with color coding"""
        colors = {
            'success': 'green',
            'failed': 'red',
            'rejected': 'orange',
            'error': 'darkred'
        }
        color = colors.get(obj.status, 'black')
        return format_html(
            '<span style="color: {};">{}</span>',
            color,
            obj.status.upper()
        )
    colored_status.short_description = 'Status'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')
    
    actions = ['export_failed_attempts']
    
    def export_failed_attempts(self, request, queryset):
        """Export failed authentication attempts for analysis"""
        failed_attempts = queryset.filter(status__in=['failed', 'rejected', 'error'])
        # This could be extended to actual export functionality
        self.message_user(
            request, 
            f'{failed_attempts.count()} failed attempts selected for export.'
        )
    export_failed_attempts.short_description = "Export failed attempts"

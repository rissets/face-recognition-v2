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
    readonly_fields = ('created_at', 'updated_at', 'embedding_hash')
    ordering = ('-created_at',)
    
    fieldsets = (
        ('User', {
            'fields': ('user',)
        }),
        ('Embedding Data', {
            'fields': ('embedding_vector', 'embedding_hash'),
            'classes': ('collapse',)
        }),
        ('Quality Metrics', {
            'fields': ('quality_score', 'confidence_score', 'liveness_score', 'anti_spoofing_score')
        }),
        ('Face Details', {
            'fields': ('face_bbox', 'face_landmarks', 'capture_device', 'capture_resolution')
        }),
        ('Session Info', {
            'fields': ('enrollment_session', 'sample_number'),
            'classes': ('collapse',)
        }),
        ('Status & Metadata', {
            'fields': ('is_active', 'is_verified', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


@admin.register(EnrollmentSession)
class EnrollmentSessionAdmin(ModelAdmin):
    """Admin for enrollment sessions"""
    list_display = ('user', 'session_token', 'status', 'average_quality', 'started_at', 'completed_at')
    list_filter = ('status', 'started_at', 'completed_at')
    search_fields = ('user__email', 'session_token', 'user__first_name', 'user__last_name')
    readonly_fields = ('session_token', 'started_at', 'completed_at')
    date_hierarchy = 'started_at'
    
    fieldsets = (
        ('Session Info', {
            'fields': ('user', 'session_token', 'status')
        }),
        ('Progress', {
            'fields': ('target_samples', 'completed_samples', 'average_quality', 'min_quality_threshold')
        }),
        ('Device & Network', {
            'fields': ('device_info', 'ip_address'),
            'classes': ('collapse',)
        }),
        ('Logs & Errors', {
            'fields': ('session_log', 'error_messages'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('started_at', 'completed_at', 'expires_at'),
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
    list_display = ('user', 'result', 'similarity_score', 'ip_address', 'created_at', 'colored_result')
    list_filter = ('result', 'created_at')
    search_fields = ('user__email', 'ip_address', 'user_agent', 'user__first_name', 'user__last_name')
    readonly_fields = ('created_at', 'processing_time')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Authentication Info', {
            'fields': ('user', 'result', 'similarity_score')
        }),
        ('Quality Metrics', {
            'fields': ('quality_score', 'liveness_score', 'obstacles_detected')
        }),
        ('Request Details', {
            'fields': ('ip_address', 'user_agent', 'device_fingerprint')
        }),
        ('Processing', {
            'fields': ('processing_time', 'face_bbox', 'metadata')
        }),
        ('Session Info', {
            'fields': ('session_id', 'submitted_embedding', 'matched_embedding'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )
    
    def colored_result(self, obj):
        """Display result with color coding"""
        colors = {
            'success': 'green',
            'failed_similarity': 'red',
            'failed_liveness': 'orange',
            'failed_quality': 'orange',
            'failed_obstacles': 'orange',
            'failed_multiple_faces': 'red',
            'failed_no_face': 'red',
            'failed_system_error': 'darkred'
        }
        color = colors.get(obj.result, 'black')
        return format_html(
            '<span style="color: {};">{}</span>',
            color,
            obj.get_result_display()
        )
    colored_result.short_description = 'Result'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')
    
    actions = ['export_failed_attempts']
    
    def export_failed_attempts(self, request, queryset):
        """Export failed authentication attempts for analysis"""
        failed_attempts = queryset.filter(result__startswith='failed')
        # This could be extended to actual export functionality
        self.message_user(
            request, 
            f'{failed_attempts.count()} failed attempts selected for export.'
        )
    export_failed_attempts.short_description = "Export failed attempts"

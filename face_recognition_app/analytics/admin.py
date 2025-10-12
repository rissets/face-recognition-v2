"""
Analytics admin configuration - Fixed to match actual model fields
"""
from django.contrib import admin
from django.db.models import Avg, Count
from django.utils.html import format_html
from unfold.admin import ModelAdmin
from .models import (
    AuthenticationLog, 
    SystemMetrics, 
    UserBehaviorAnalytics, 
    SecurityAlert, 
    FaceRecognitionStats, 
    ModelPerformance, 
    DataQualityMetrics
)


@admin.register(AuthenticationLog)
class AuthenticationLogAdmin(ModelAdmin):
    """Admin for authentication logs"""
    list_display = ('user', 'attempted_email', 'auth_method', 'success', 'failure_reason', 'created_at')
    list_filter = ('auth_method', 'success', 'failure_reason', 'created_at')
    search_fields = ('user__email', 'attempted_email', 'ip_address', 'user__first_name', 'user__last_name')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Authentication Info', {
            'fields': ('user', 'attempted_email', 'auth_method', 'success', 'failure_reason')
        }),
        ('Request Details', {
            'fields': ('ip_address', 'user_agent', 'device_fingerprint', 'location')
        }),
        ('Face Recognition Details', {
            'fields': ('similarity_score', 'quality_score', 'liveness_score', 'response_time'),
            'classes': ('collapse',)
        }),
        ('Risk Assessment', {
            'fields': ('risk_score', 'risk_factors'),
            'classes': ('collapse',)
        }),
        ('Session', {
            'fields': ('session_id',),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


@admin.register(SystemMetrics)
class SystemMetricsAdmin(ModelAdmin):
    """Admin for system metrics"""
    list_display = ('metric_name', 'value', 'unit', 'metric_type', 'timestamp')
    list_filter = ('metric_type', 'metric_name', 'timestamp')
    search_fields = ('metric_name', 'metric_type')
    readonly_fields = ('timestamp',)
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Metric Info', {
            'fields': ('metric_name', 'metric_type', 'value', 'unit')
        }),
        ('Metadata', {
            'fields': ('tags', 'timestamp'),
            'classes': ('collapse',)
        })
    )


@admin.register(UserBehaviorAnalytics)
class UserBehaviorAnalyticsAdmin(ModelAdmin):
    """Admin for user behavior analytics"""
    list_display = ('user', 'auth_success_rate', 'risk_level', 'created_at')
    list_filter = ('risk_level', 'created_at', 'updated_at')
    search_fields = ('user__email', 'user__first_name', 'user__last_name')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('User Info', {
            'fields': ('user',)
        }),
        ('Behavioral Metrics', {
            'fields': ('avg_login_time', 'common_locations', 'device_preferences')
        }),
        ('Authentication Patterns', {
            'fields': ('auth_success_rate', 'avg_similarity_score', 'avg_liveness_score')
        }),
        ('Activity Patterns', {
            'fields': ('login_frequency', 'peak_activity_hours')
        }),
        ('Risk Assessment', {
            'fields': ('suspicious_activity_count', 'last_risk_assessment', 'risk_level'),
            'classes': ('collapse',)
        }),
        ('Analysis Period', {
            'fields': ('analysis_start', 'analysis_end'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


@admin.register(SecurityAlert)
class SecurityAlertAdmin(ModelAdmin):
    """Admin for security alerts"""
    list_display = ('alert_type', 'severity', 'user', 'acknowledged', 'resolved', 'created_at')
    list_filter = ('alert_type', 'severity', 'acknowledged', 'resolved', 'created_at')
    search_fields = ('alert_type', 'title', 'description', 'user__email', 'user__first_name', 'user__last_name')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Alert Info', {
            'fields': ('alert_type', 'severity', 'title', 'description')
        }),
        ('Affected Entity', {
            'fields': ('user', 'ip_address')
        }),
        ('Context', {
            'fields': ('context_data',),
            'classes': ('collapse',)
        }),
        ('Status', {
            'fields': ('acknowledged', 'acknowledged_by', 'acknowledged_at')
        }),
        ('Resolution', {
            'fields': ('resolved', 'resolved_by', 'resolved_at', 'resolution_notes'),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'acknowledged_by', 'resolved_by')


@admin.register(FaceRecognitionStats)
class FaceRecognitionStatsAdmin(ModelAdmin):
    """Admin for face recognition statistics"""
    list_display = ('date', 'hour', 'total_attempts', 'successful_attempts', 'success_rate_display', 'unique_users')
    list_filter = ('date', 'hour')
    search_fields = ('date',)
    readonly_fields = ('created_at',)
    date_hierarchy = 'date'
    
    fieldsets = (
        ('Time Period', {
            'fields': ('date', 'hour')
        }),
        ('Authentication Stats', {
            'fields': ('total_attempts', 'successful_attempts', 'failed_attempts')
        }),
        ('Failure Breakdown', {
            'fields': (
                'failed_similarity', 'failed_liveness', 'failed_quality', 
                'failed_obstacles', 'failed_no_face', 'failed_multiple_faces', 
                'failed_system_error'
            ),
            'classes': ('collapse',)
        }),
        ('Performance Metrics', {
            'fields': (
                'avg_response_time', 'avg_similarity_score', 
                'avg_liveness_score', 'avg_quality_score'
            ),
            'classes': ('collapse',)
        }),
        ('User Metrics', {
            'fields': ('unique_users', 'new_enrollments')
        }),
        ('Timestamp', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )
    
    def success_rate_display(self, obj):
        """Display success rate with color coding"""
        rate = obj.success_rate
        if rate >= 90:
            color = 'green'
        elif rate >= 70:
            color = 'orange'
        else:
            color = 'red'
        
        return rate
    success_rate_display.short_description = 'Success Rate'


@admin.register(ModelPerformance)
class ModelPerformanceAdmin(ModelAdmin):
    """Admin for model performance metrics"""
    list_display = ('model', 'accuracy', 'precision', 'recall', 'f1_score', 'environment', 'created_at')
    list_filter = ('environment', 'created_at')
    search_fields = ('model__name',)
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Model Info', {
            'fields': ('model', 'environment')
        }),
        ('Performance Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score')
        }),
        ('Specific Metrics', {
            'fields': ('false_acceptance_rate', 'false_rejection_rate'),
            'classes': ('collapse',)
        }),
        ('Test Information', {
            'fields': ('test_set_size', 'test_conditions'),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('model')


@admin.register(DataQualityMetrics)
class DataQualityMetricsAdmin(ModelAdmin):
    """Admin for data quality metrics"""
    list_display = ('date', 'avg_image_quality', 'quality_score_display', 'total_samples', 'created_at')
    list_filter = ('date', 'created_at')
    search_fields = ('date',)
    readonly_fields = ('created_at',)
    date_hierarchy = 'date'
    
    fieldsets = (
        ('Time Period', {
            'fields': ('date',)
        }),
        ('Quality Metrics', {
            'fields': (
                'avg_image_quality', 'avg_face_size', 'avg_brightness', 
                'avg_contrast', 'avg_sharpness'
            )
        }),
        ('Quality Distribution', {
            'fields': ('high_quality_samples', 'medium_quality_samples', 'low_quality_samples')
        }),
        ('Issues Detected', {
            'fields': (
                'blurry_images', 'over_exposed', 'under_exposed', 'obstacles_present'
            ),
            'classes': ('collapse',)
        }),
        ('Summary', {
            'fields': ('total_samples',)
        }),
        ('Timestamp', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )
    
    def quality_score_display(self, obj):
        """Display quality score with color coding"""
        score = obj.quality_score
        if score >= 80:
            color = 'green'
        elif score >= 60:
            color = 'orange'
        else:
            color = 'red'
        
        return score
    quality_score_display.short_description = 'Quality Score'
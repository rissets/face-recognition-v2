# """
# Analytics admin configuration - Simplified working version
# """
# from django.contrib import admin
# from unfold.admin import ModelAdmin
# from .models import AuthenticationLog, SystemMetrics, UserBehaviorAnalytics, SecurityAlert, FaceRecognitionStats, ModelPerformance, DataQualityMetrics


# @admin.register(AuthenticationLog)
# class AuthenticationLogAdmin(ModelAdmin):
#     """Admin for authentication logs"""
#     list_display = ('user', 'attempted_email', 'auth_method', 'success', 'created_at')
#     list_filter = ('auth_method', 'success', 'created_at')
#     search_fields = ('user__email', 'attempted_email')
#     readonly_fields = ('created_at',)


# @admin.register(SystemMetrics) 
# class SystemMetricsAdmin(ModelAdmin):
#     """Admin for system metrics"""
#     list_display = ('metric_name', 'metric_type', 'created_at')
#     list_filter = ('metric_type',)
#     search_fields = ('metric_name',)
#     readonly_fields = ('created_at',)


# @admin.register(UserBehaviorAnalytics)
# class UserBehaviorAnalyticsAdmin(ModelAdmin):
#     """Admin for user behavior analytics"""
#     list_display = ('user', 'created_at')
#     search_fields = ('user__email',)
#     readonly_fields = ('created_at', 'updated_at')


# @admin.register(SecurityAlert)
# class SecurityAlertAdmin(ModelAdmin):
#     """Admin for security alerts"""
#     list_display = ('alert_type', 'severity', 'user', 'created_at')
#     list_filter = ('alert_type', 'severity')
#     search_fields = ('alert_type', 'user__email')
#     readonly_fields = ('created_at',)


# @admin.register(FaceRecognitionStats)
# class FaceRecognitionStatsAdmin(ModelAdmin):
#     """Admin for face recognition statistics"""
#     list_display = ('user', 'created_at')
#     search_fields = ('user__email',)
#     readonly_fields = ('created_at',)


# @admin.register(ModelPerformance)
# class ModelPerformanceAdmin(ModelAdmin):
#     """Admin for model performance metrics"""
#     list_display = ('model_name', 'model_version', 'created_at')
#     search_fields = ('model_name',)
#     readonly_fields = ('created_at',)


# @admin.register(DataQualityMetrics)
# class DataQualityMetricsAdmin(ModelAdmin):
#     """Admin for data quality metrics"""
#     list_display = ('name', 'value', 'created_at')
#     search_fields = ('name',)
#     readonly_fields = ('created_at',)


# @admin.register(AuthenticationLog)
# class AuthenticationLogAdmin(ModelAdmin):
#     """Admin for authentication logs"""
#     list_display = ('user', 'attempted_email', 'auth_method', 'success', 'failure_reason', 'created_at')
#     list_filter = ('auth_method', 'success', 'failure_reason', 'created_at')
#     search_fields = ('user__email', 'attempted_email', 'ip_address', 'user__first_name', 'user__last_name')
#     readonly_fields = ('created_at',)
#     date_hierarchy = 'created_at'
    
#     fieldsets = (
#         ('Authentication Info', {
#             'fields': ('user', 'attempted_email', 'auth_method', 'success', 'failure_reason')
#         }),
#         ('Request Details', {
#             'fields': ('ip_address', 'user_agent', 'device_fingerprint')
#         }),
#         ('Face Recognition Details', {
#             'fields': ('confidence_score', 'quality_score', 'liveness_score', 'processing_time'),
#             'classes': ('collapse',)
#         }),
#         ('Location & Context', {
#             'fields': ('location_data', 'session_context'),
#             'classes': ('collapse',)
#         }),
#         ('Timestamp', {
#             'fields': ('created_at',),
#             'classes': ('collapse',)
#         })
#     )
    
#     def get_queryset(self, request):
#         return super().get_queryset(request).select_related('user')
    
#     actions = ['export_auth_report']
    
#     def export_auth_report(self, request, queryset):
#         """Export authentication report"""
#         total = queryset.count()
#         successful = queryset.filter(success=True).count()
#         self.message_user(
#             request, 
#             f'Auth report: {total} total attempts, {successful} successful ({successful/total*100:.1f}%)'
#         )
#     export_auth_report.short_description = "Generate authentication report"


# @admin.register(SystemMetrics)
# class SystemMetricsAdmin(ModelAdmin):
#     """Admin for system metrics"""
#     list_display = ('metric_name', 'metric_value', 'metric_type', 'created_at')
#     list_filter = ('metric_type', 'metric_name', 'created_at')
#     search_fields = ('metric_name', 'metric_type')
#     readonly_fields = ('created_at',)
#     date_hierarchy = 'created_at'
    
#     fieldsets = (
#         ('Metric Info', {
#             'fields': ('metric_name', 'metric_type', 'metric_value')
#         }),
#         ('Details', {
#             'fields': ('metric_unit', 'description')
#         }),
#         ('Metadata', {
#             'fields': ('tags', 'created_at'),
#             'classes': ('collapse',)
#         })
#     )
    
#     def changelist_view(self, request, extra_context=None):
#         """Add summary statistics to changelist"""
#         extra_context = extra_context or {}
        
#         # Get average values for numeric metrics
#         numeric_metrics = self.get_queryset(request).filter(
#             metric_type__in=['counter', 'gauge', 'histogram']
#         )
        
#         if numeric_metrics.exists():
#             avg_stats = numeric_metrics.values('metric_name').annotate(
#                 avg_value=Avg('metric_value'),
#                 count=Count('id')
#             )
#             extra_context['avg_stats'] = list(avg_stats)
        
#         return super().changelist_view(request, extra_context)


# @admin.register(UserBehaviorAnalytics)
# class UserBehaviorAnalyticsAdmin(ModelAdmin):
#     """Admin for user behavior analytics"""
#     list_display = ('user', 'session_count', 'avg_session_duration', 'total_login_attempts', 'created_at')
#     list_filter = ('created_at', 'updated_at')
#     search_fields = ('user__email', 'user__first_name', 'user__last_name')
#     readonly_fields = ('created_at', 'updated_at')
#     date_hierarchy = 'created_at'
    
#     fieldsets = (
#         ('User Info', {
#             'fields': ('user',)
#         }),
#         ('Session Statistics', {
#             'fields': ('session_count', 'avg_session_duration', 'total_active_time')
#         }),
#         ('Authentication Stats', {
#             'fields': ('total_login_attempts', 'successful_logins', 'failed_attempts')
#         }),
#         ('Face Recognition Stats', {
#             'fields': ('face_auth_attempts', 'face_auth_success_rate', 'avg_recognition_time'),
#             'classes': ('collapse',)
#         }),
#         ('Device & Location', {
#             'fields': ('unique_devices', 'unique_locations', 'preferred_device_type'),
#             'classes': ('collapse',)
#         }),
#         ('Patterns', {
#             'fields': ('usage_patterns', 'peak_hours', 'behavioral_flags'),
#             'classes': ('collapse',)
#         }),
#         ('Timestamps', {
#             'fields': ('created_at', 'updated_at'),
#             'classes': ('collapse',)
#         })
#     )
    
#     def get_queryset(self, request):
#         return super().get_queryset(request).select_related('user')
    
#     def colored_success_rate(self, obj):
#         """Display success rate with color coding"""
#         if obj.face_auth_success_rate >= 0.8:
#             color = 'green'
#         elif obj.face_auth_success_rate >= 0.6:
#             color = 'orange'
#         else:
#             color = 'red'
        
#         return format_html(
#             '<span style="color: {};">{:.1%}</span>',
#             color,
#             obj.face_auth_success_rate
#         )
#     colored_success_rate.short_description = 'Success Rate'


# @admin.register(SecurityAlert)
# class SecurityAlertAdmin(ModelAdmin):
#     """Admin for security alerts"""
#     list_display = ('alert_type', 'severity', 'user', 'status', 'triggered_at', 'colored_severity')
#     list_filter = ('alert_type', 'severity', 'status', 'triggered_at')
#     search_fields = ('alert_type', 'description', 'user__email', 'user__first_name', 'user__last_name')
#     readonly_fields = ('triggered_at', 'resolved_at')
#     date_hierarchy = 'triggered_at'
    
#     fieldsets = (
#         ('Alert Info', {
#             'fields': ('alert_type', 'severity', 'status')
#         }),
#         ('User Context', {
#             'fields': ('user', 'source_ip', 'user_agent')
#         }),
#         ('Alert Details', {
#             'fields': ('description', 'detection_data'),
#             'classes': ('collapse',)
#         }),
#         ('Risk Assessment', {
#             'fields': ('risk_score', 'confidence_level', 'false_positive_probability'),
#             'classes': ('collapse',)
#         }),
#         ('Context & Metadata', {
#             'fields': ('context_data', 'related_alerts'),
#             'classes': ('collapse',)
#         }),
#         ('Response', {
#             'fields': ('auto_response_taken', 'manual_response', 'resolution_notes'),
#             'classes': ('collapse',)
#         }),
#         ('Timestamps', {
#             'fields': ('triggered_at', 'resolved_at'),
#             'classes': ('collapse',)
#         })
#     )
    
#     def colored_severity(self, obj):
#         """Display severity with color coding"""
#         colors = {
#             'low': 'green',
#             'medium': 'orange',
#             'high': 'red',
#             'critical': 'darkred'
#         }
#         color = colors.get(obj.severity, 'black')
#         return format_html(
#             '<span style="color: {}; font-weight: bold;">{}</span>',
#             color,
#             obj.severity.upper()
#         )
#     colored_severity.short_description = 'Severity'
    
#     def get_queryset(self, request):
#         return super().get_queryset(request).select_related('user')
    
#     actions = ['mark_resolved', 'mark_false_positive', 'export_security_report']
    
#     def mark_resolved(self, request, queryset):
#         """Mark alerts as resolved"""
#         from django.utils import timezone
#         updated = queryset.filter(status='active').update(
#             status='resolved',
#             resolved_at=timezone.now()
#         )
#         self.message_user(request, f'{updated} alerts marked as resolved.')
#     mark_resolved.short_description = "Mark as resolved"
    
#     def mark_false_positive(self, request, queryset):
#         """Mark alerts as false positive"""
#         from django.utils import timezone
#         updated = queryset.filter(status='active').update(
#             status='false_positive',
#             resolved_at=timezone.now()
#         )
#         self.message_user(request, f'{updated} alerts marked as false positive.')
#     mark_false_positive.short_description = "Mark as false positive"
    
#     def export_security_report(self, request, queryset):
#         """Export security report for analysis"""
#         alert_summary = queryset.values('alert_type', 'severity').annotate(
#             count=Count('id')
#         ).order_by('-count')
        
#         total_alerts = queryset.count()
#         critical_alerts = queryset.filter(severity='critical').count()
#         active_alerts = queryset.filter(status='active').count()
        
#         self.message_user(
#             request, 
#             f'Security report: {total_alerts} total alerts, {critical_alerts} critical, {active_alerts} active'
#         )
#     export_security_report.short_description = "Generate security report"


# @admin.register(FaceRecognitionStats)
# class FaceRecognitionStatsAdmin(ModelAdmin):
#     """Admin for face recognition statistics"""
#     list_display = ('user', 'total_enrollments', 'total_authentications', 'success_rate', 'created_at')
#     list_filter = ('created_at', 'enrollment_completed')
#     search_fields = ('user__email', 'user__first_name', 'user__last_name')
#     readonly_fields = ('created_at', 'updated_at')
    
#     fieldsets = (
#         ('User Info', {
#             'fields': ('user',)
#         }),
#         ('Enrollment Stats', {
#             'fields': ('total_enrollments', 'enrollment_completed', 'enrollment_attempts', 'avg_enrollment_time')
#         }),
#         ('Authentication Stats', {
#             'fields': ('total_authentications', 'successful_authentications', 'failed_authentications', 'success_rate')
#         }),
#         ('Quality Metrics', {
#             'fields': ('avg_confidence_score', 'avg_quality_score', 'best_quality_score', 'worst_quality_score')
#         }),
#         ('Performance', {
#             'fields': ('avg_processing_time', 'fastest_recognition', 'slowest_recognition'),
#             'classes': ('collapse',)
#         }),
#         ('Timestamps', {
#             'fields': ('created_at', 'updated_at'),
#             'classes': ('collapse',)
#         })
#     )
    
#     def get_queryset(self, request):
#         return super().get_queryset(request).select_related('user')


# @admin.register(ModelPerformance)
# class ModelPerformanceAdmin(ModelAdmin):
#     """Admin for model performance metrics"""
#     list_display = ('model_name', 'version', 'accuracy', 'precision', 'recall', 'created_at')
#     list_filter = ('model_name', 'version', 'created_at')
#     search_fields = ('model_name', 'version')
#     readonly_fields = ('created_at',)
    
#     fieldsets = (
#         ('Model Info', {
#             'fields': ('model_name', 'version', 'model_type')
#         }),
#         ('Performance Metrics', {
#             'fields': ('accuracy', 'precision', 'recall', 'f1_score', 'auc_roc')
#         }),
#         ('Detailed Stats', {
#             'fields': ('true_positives', 'true_negatives', 'false_positives', 'false_negatives')
#         }),
#         ('Test Details', {
#             'fields': ('test_dataset_size', 'test_duration', 'avg_inference_time'),
#             'classes': ('collapse',)
#         }),
#         ('Configuration', {
#             'fields': ('model_config', 'test_config'),
#             'classes': ('collapse',)
#         }),
#         ('Timestamp', {
#             'fields': ('created_at',),
#             'classes': ('collapse',)
#         })
#     )


# @admin.register(DataQualityMetrics)
# class DataQualityMetricsAdmin(ModelAdmin):
#     """Admin for data quality metrics"""
#     list_display = ('metric_name', 'metric_value', 'threshold', 'status', 'created_at')
#     list_filter = ('metric_name', 'status', 'created_at')
#     search_fields = ('metric_name', 'description')
#     readonly_fields = ('created_at',)
    
#     fieldsets = (
#         ('Metric Info', {
#             'fields': ('metric_name', 'metric_value', 'threshold', 'status')
#         }),
#         ('Details', {
#             'fields': ('description', 'measurement_unit')
#         }),
#         ('Quality Assessment', {
#             'fields': ('quality_issues', 'recommendations'),
#             'classes': ('collapse',)
#         }),
#         ('Metadata', {
#             'fields': ('metadata', 'created_at'),
#             'classes': ('collapse',)
#         })
#     )

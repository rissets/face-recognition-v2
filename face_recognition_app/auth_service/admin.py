"""
Auth service admin configuration
"""
import logging
from django.contrib import admin
from django.utils.html import format_html
from unfold.admin import ModelAdmin
from .models import AuthenticationSession, FaceEnrollment, LivenessDetectionResult, FaceRecognitionAttempt

logger = logging.getLogger(__name__)


@admin.register(AuthenticationSession)
class AuthenticationSessionAdmin(ModelAdmin):
    """Admin for authentication sessions"""
    list_display = ('session_token_short', 'client', 'session_type', 'status', 'is_successful', 'created_at', 'expires_at')
    list_filter = ('session_type', 'status', 'liveness_required', 'anti_spoofing_required', 'is_successful', 'created_at')
    search_fields = ('session_token', 'client__name', 'client__client_id', 'client_user__external_user_id')
    readonly_fields = ('id', 'session_token', 'created_at', 'completed_at', 'processing_time_ms', 'frames_processed')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Session Info', {
            'fields': ('id', 'session_token', 'client', 'client_user')
        }),
        ('Configuration', {
            'fields': ('session_type', 'status', 'liveness_required', 'anti_spoofing_required', 'max_attempts', 'current_attempts')
        }),
        ('Device & Security', {
            'fields': ('ip_address', 'user_agent', 'device_fingerprint'),
            'classes': ('collapse',)
        }),
        ('Results & Metrics', {
            'fields': ('is_successful', 'confidence_score', 'failure_reason', 'frames_processed', 'processing_time_ms')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'expires_at', 'completed_at'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        })
    )
    
    def session_token_short(self, obj):
        """Display shortened session token"""
        return f"{obj.session_token[:15]}..." if obj.session_token else "-"
    session_token_short.short_description = "Session Token"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('client', 'client_user')


@admin.register(LivenessDetectionResult)
class LivenessDetectionResultAdmin(ModelAdmin):
    """Admin for liveness detection results"""
    list_display = ('client', 'session', 'status', 'confidence_score', 'blink_detected', 'motion_detected', 'created_at')
    list_filter = ('status', 'blink_detected', 'motion_detected', 'created_at', 'client__tier')
    search_fields = ('client__name', 'client__client_id', 'session__session_token')
    readonly_fields = ('id', 'created_at', 'processing_time_ms', 'frames_analyzed')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Detection Info', {
            'fields': ('id', 'client', 'session', 'status', 'confidence_score')
        }),
        ('Methods & Results', {
            'fields': ('methods_used', 'blink_detected', 'motion_detected', 'texture_score')
        }),
        ('Analysis Data', {
            'fields': ('face_movements', 'frames_analyzed', 'processing_time_ms'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('client', 'session')


@admin.register(FaceRecognitionAttempt)
class FaceRecognitionAttemptAdmin(ModelAdmin):
    """Admin for face recognition attempts"""
    list_display = ('client', 'session', 'result', 'similarity_score', 'confidence_score', 'created_at')
    list_filter = ('result', 'created_at', 'client__tier')
    search_fields = ('client__name', 'client__client_id', 'session__session_token', 'matched_user__external_user_id')
    readonly_fields = ('id', 'created_at', 'processing_time_ms')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Attempt Info', {
            'fields': ('id', 'client', 'session', 'matched_user', 'matched_enrollment')
        }),
        ('Results & Scores', {
            'fields': ('result', 'similarity_score', 'confidence_score', 'face_quality_score', 'liveness_score')
        }),
        ('Technical Details', {
            'fields': ('processing_time_ms', 'error_details'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',)
        })
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('client', 'session', 'matched_user', 'matched_enrollment')


@admin.register(FaceEnrollment)
class FaceEnrollmentAdmin(ModelAdmin):
    """Admin for face enrollments"""
    list_display = ('client_user_display', 'client', 'status', 'face_quality_score', 'liveness_score', 'created_at')
    list_filter = ('status', 'created_at', 'client__tier')
    search_fields = ('client_user__external_user_id', 'client__name', 'client__client_id')
    readonly_fields = ('id', 'created_at', 'embedding_dimension', 'face_landmarks', 'face_bbox')
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Enrollment Info', {
            'fields': ('id', 'client', 'client_user', 'enrollment_session', 'status')
        }),
        ('Quality Metrics', {
            'fields': ('face_quality_score', 'liveness_score', 'anti_spoofing_score')
        }),
        ('Face Data', {
            'fields': ('embedding_dimension', 'face_landmarks', 'face_bbox'),
            'classes': ('collapse',)
        }),
        ('Sample Info', {
            'fields': ('sample_number', 'total_samples', 'face_image_path'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'expires_at'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        })
    )
    
    def client_user_display(self, obj):
        """Display client user info"""
        if obj.client_user and obj.client_user.profile:
            name = obj.client_user.profile.get('name', obj.client_user.external_user_id)
            return format_html(
                '<strong>{}</strong><br><small>{}</small>',
                name,
                obj.client_user.external_user_id
            )
        return obj.client_user.external_user_id if obj.client_user else "-"
    client_user_display.short_description = "Client User"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('client', 'client_user', 'enrollment_session')
    
    def _delete_chroma_embeddings(self, enrollment):
        """Delete embeddings from ChromaDB for the given enrollment"""
        try:
            from core.face_recognition_engine import ChromaEmbeddingStore
            
            if not enrollment.client_user or not enrollment.client:
                logger.warning(f"Cannot delete ChromaDB embeddings: missing client_user or client for enrollment {enrollment.id}")
                return False
            
            # Build the user_id in the same format used during enrollment
            user_id = f"{enrollment.client.client_id}:{enrollment.client_user.external_user_id}"
            
            # Initialize ChromaDB store and delete embeddings
            chroma_store = ChromaEmbeddingStore()
            result = chroma_store.delete_user_embeddings(user_id)
            
            if result:
                logger.info(f"Successfully deleted ChromaDB embeddings for user: {user_id}")
            else:
                logger.warning(f"Failed to delete ChromaDB embeddings for user: {user_id}")
            
            return result
        except Exception as e:
            logger.error(f"Error deleting ChromaDB embeddings for enrollment {enrollment.id}: {e}")
            return False
    
    def delete_model(self, request, obj):
        """Override delete_model to also delete from ChromaDB"""
        # Delete from ChromaDB first
        self._delete_chroma_embeddings(obj)
        
        # Update client_user enrollment status
        if obj.client_user:
            remaining = FaceEnrollment.objects.filter(
                client_user=obj.client_user, 
                status='active'
            ).exclude(id=obj.id).exists()
            
            if not remaining:
                obj.client_user.is_enrolled = False
                obj.client_user.enrollment_completed_at = None
                obj.client_user.save(update_fields=['is_enrolled', 'enrollment_completed_at'])
                logger.info(f"Updated client_user {obj.client_user.external_user_id} enrollment status to False")
        
        # Call parent delete
        super().delete_model(request, obj)
    
    def delete_queryset(self, request, queryset):
        """Override delete_queryset to also delete from ChromaDB for bulk deletes"""
        # Collect unique client_users to update after deletion
        client_users_to_check = set()
        
        for obj in queryset:
            # Delete from ChromaDB
            self._delete_chroma_embeddings(obj)
            if obj.client_user:
                client_users_to_check.add(obj.client_user)
        
        # Call parent delete
        super().delete_queryset(request, queryset)
        
        # Update client_user enrollment status for affected users
        for client_user in client_users_to_check:
            remaining = FaceEnrollment.objects.filter(
                client_user=client_user, 
                status='active'
            ).exists()
            
            if not remaining:
                client_user.is_enrolled = False
                client_user.enrollment_completed_at = None
                client_user.save(update_fields=['is_enrolled', 'enrollment_completed_at'])
                logger.info(f"Updated client_user {client_user.external_user_id} enrollment status to False")

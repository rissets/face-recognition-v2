"""
Auth service admin configuration
"""
import logging
from django.contrib import admin
from django.utils.html import format_html
from unfold.admin import ModelAdmin
from .models import AuthenticationSession, FaceEnrollment, LivenessDetectionResult, FaceRecognitionAttempt

# Import OIDC models
from .oidc.models import (
    OAuthClient, AuthorizationCode, OAuthToken, OIDCSession, UserConsent,
    OIDCAuthorizationLog, OIDCTokenLog
)

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


# ---------------------------------------------------------------------------
# OAuth 2.0 / OpenID Connect Admin
# ---------------------------------------------------------------------------

@admin.register(OAuthClient)
class OAuthClientAdmin(ModelAdmin):
    list_display = ['name', 'client_id', 'client_type', 'is_active', 'require_face_auth', 'created_at']
    list_filter = ['client_type', 'is_active', 'require_face_auth', 'require_liveness']
    search_fields = ['name', 'client_id', 'description']
    readonly_fields = ['id', 'client_id', 'created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'logo_url', 'client_id', 'client_secret', 'client_type')
        }),
        ('OAuth Configuration', {
            'fields': ('redirect_uris', 'grant_types', 'response_types', 'allowed_scopes')
        }),
        ('Token Settings', {
            'fields': ('access_token_lifetime', 'refresh_token_lifetime', 'id_token_lifetime')
        }),
        ('Security', {
            'fields': ('require_pkce', 'require_consent')
        }),
        ('Face Authentication', {
            'fields': ('require_face_auth', 'require_liveness', 'min_confidence_score')
        }),
        ('API Client Link', {
            'fields': ('api_client',),
            'classes': ('collapse',)
        }),
        ('Status', {
            'fields': ('is_active', 'created_at', 'updated_at')
        }),
    )
    
    actions = ['regenerate_secrets']
    
    @admin.action(description="Regenerate client secrets")
    def regenerate_secrets(self, request, queryset):
        for client in queryset:
            client.regenerate_secret()
        self.message_user(request, f"Regenerated secrets for {queryset.count()} clients")


@admin.register(AuthorizationCode)
class AuthorizationCodeAdmin(ModelAdmin):
    list_display = ['code_preview', 'client', 'client_user', 'is_used', 'is_expired_display', 'created_at']
    list_filter = ['is_used', 'client']
    search_fields = ['code', 'client__name', 'client_user__external_user_id']
    readonly_fields = ['id', 'code', 'created_at']
    
    def code_preview(self, obj):
        return f"{obj.code[:16]}..."
    code_preview.short_description = "Code"
    
    def is_expired_display(self, obj):
        return obj.is_expired
    is_expired_display.short_description = "Expired"
    is_expired_display.boolean = True


@admin.register(OAuthToken)
class OAuthTokenAdmin(ModelAdmin):
    list_display = ['token_preview', 'token_type', 'client', 'client_user', 'is_revoked', 'is_valid_display', 'created_at']
    list_filter = ['token_type', 'is_revoked', 'client']
    search_fields = ['token', 'client__name', 'client_user__external_user_id']
    readonly_fields = ['id', 'created_at']
    
    def token_preview(self, obj):
        return f"{obj.token[:16]}..."
    token_preview.short_description = "Token"
    
    def is_valid_display(self, obj):
        return obj.is_valid
    is_valid_display.short_description = "Valid"
    is_valid_display.boolean = True
    
    actions = ['revoke_tokens']
    
    @admin.action(description="Revoke selected tokens")
    def revoke_tokens(self, request, queryset):
        for token in queryset:
            token.revoke()
        self.message_user(request, f"Revoked {queryset.count()} tokens")


@admin.register(OIDCSession)
class OIDCSessionAdmin(ModelAdmin):
    list_display = ['session_id_preview', 'client_user', 'face_auth_confidence', 'liveness_verified', 'is_valid_display', 'created_at']
    list_filter = ['liveness_verified']
    search_fields = ['session_id', 'client_user__external_user_id']
    readonly_fields = ['id', 'session_id', 'created_at']
    
    def session_id_preview(self, obj):
        return f"{obj.session_id[:16]}..."
    session_id_preview.short_description = "Session ID"
    
    def is_valid_display(self, obj):
        return obj.is_valid
    is_valid_display.short_description = "Valid"
    is_valid_display.boolean = True


@admin.register(UserConsent)
class UserConsentAdmin(ModelAdmin):
    list_display = ['client_user', 'client', 'scopes_preview', 'is_valid_display', 'granted_at']
    list_filter = ['client']
    search_fields = ['client__name', 'client_user__external_user_id']
    readonly_fields = ['id', 'granted_at']
    
    def scopes_preview(self, obj):
        if obj.scopes:
            return ", ".join(obj.scopes[:3]) + ("..." if len(obj.scopes) > 3 else "")
        return "-"
    scopes_preview.short_description = "Scopes"
    
    def is_valid_display(self, obj):
        return obj.is_valid
    is_valid_display.short_description = "Valid"
    is_valid_display.boolean = True
    
    actions = ['revoke_consents']
    
    @admin.action(description="Revoke selected consents")
    def revoke_consents(self, request, queryset):
        for consent in queryset:
            consent.revoke()
        self.message_user(request, f"Revoked {queryset.count()} consents")


@admin.register(OIDCAuthorizationLog)
class OIDCAuthorizationLogAdmin(ModelAdmin):
    """Admin for OIDC authorization logs"""
    list_display = ['login_identifier', 'client', 'status', 'authorization_code_display', 'created_at', 'completed_at']
    list_filter = ['status', 'client', 'created_at']
    search_fields = ['login_identifier', 'client__name', 'client_user__external_user_id', 'state', 'nonce']
    readonly_fields = ['id', 'created_at', 'updated_at', 'completed_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Authentication Info', {
            'fields': ('login_identifier', 'client', 'client_user', 'status')
        }),
        ('Authorization Details', {
            'fields': ('redirect_uri', 'scope', 'state', 'nonce')
        }),
        ('Results', {
            'fields': ('authorization_code', 'auth_session', 'error_code', 'error_description')
        }),
        ('Request Metadata', {
            'fields': ('user_agent', 'ip_address'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'completed_at')
        })
    )
    
    def authorization_code_display(self, obj):
        if obj.authorization_code:
            return format_html(
                '<span style="color: green;">✓</span> {}',
                obj.authorization_code.code[:16] + '...'
            )
        return format_html('<span style="color: gray;">-</span>')
    authorization_code_display.short_description = "Auth Code"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'client', 'client_user', 'authorization_code', 'auth_session'
        )


@admin.register(OIDCTokenLog)
class OIDCTokenLogAdmin(ModelAdmin):
    """Admin for OIDC token logs"""
    list_display = ['token_display', 'operation', 'client', 'client_user', 'grant_type', 'created_at']
    list_filter = ['operation', 'grant_type', 'client', 'created_at']
    search_fields = ['client__name', 'client_user__external_user_id', 'token__token']
    readonly_fields = ['id', 'created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Token Info', {
            'fields': ('token', 'operation', 'grant_type')
        }),
        ('Client & User', {
            'fields': ('client', 'client_user')
        }),
        ('Request Metadata', {
            'fields': ('user_agent', 'ip_address'),
            'classes': ('collapse',)
        }),
        ('Timestamp', {
            'fields': ('created_at',)
        })
    )
    
    def token_display(self, obj):
        token_type_colors = {
            'access': '#00c853',
            'refresh': '#2196f3'
        }
        color = token_type_colors.get(obj.token.token_type, '#666')
        return format_html(
            '<span style="color: {};">{}</span> {}',
            color,
            obj.token.token_type.upper(),
            obj.token.token[:16] + '...'
        )
    token_display.short_description = "Token"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'token', 'client', 'client_user'
        )

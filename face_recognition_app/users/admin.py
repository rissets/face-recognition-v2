"""
Users admin configuration
"""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from unfold.admin import ModelAdmin
from .models import CustomUser, UserProfile, UserDevice


@admin.register(CustomUser)
class CustomUserAdmin(BaseUserAdmin, ModelAdmin):
    """Admin for custom user model"""
    list_display = ('email', 'first_name', 'last_name', 'face_enrolled', 'is_active', 'created_at')
    list_filter = ('face_enrolled', 'face_auth_enabled', 'is_verified', 'is_active', 'created_at')
    search_fields = ('email', 'first_name', 'last_name', 'username')
    ordering = ('-created_at',)
    readonly_fields = ('created_at', 'updated_at', 'last_face_auth', 'enrollment_completed_at')
    
    fieldsets = (
        ('Authentication', {
            'fields': ('email', 'username', 'password')
        }),
        ('Personal Info', {
            'fields': ('first_name', 'last_name', 'phone_number', 'date_of_birth', 'bio', 'profile_picture')
        }),
        ('Face Recognition', {
            'fields': (
                'face_enrolled', 'enrollment_completed_at', 'face_auth_enabled', 
                'last_face_auth'
            )
        }),
        ('Security', {
            'fields': ('two_factor_enabled', 'is_verified', 'verification_token')
        }),
        ('Permissions', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions'),
            'classes': ('collapse',)
        }),
        ('Privacy', {
            'fields': ('allow_analytics',),
            'classes': ('collapse',)
        }),
        ('Important dates', {
            'fields': ('date_joined', 'last_login', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'username', 'first_name', 'last_name', 'password1', 'password2'),
        }),
    )


@admin.register(UserProfile)
class UserProfileAdmin(ModelAdmin):
    """Admin for user profiles"""
    list_display = ('user', 'company', 'position', 'language', 'timezone', 'created_at')
    list_filter = ('language', 'timezone', 'email_notifications', 'security_alerts', 'created_at')
    search_fields = ('user__email', 'user__first_name', 'user__last_name', 'company', 'position')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('User', {
            'fields': ('user',)
        }),
        ('Professional Info', {
            'fields': ('company', 'position')
        }),
        ('Location', {
            'fields': ('address', 'city', 'country')
        }),
        ('Preferences', {
            'fields': ('language', 'timezone')
        }),
        ('Notifications', {
            'fields': ('email_notifications', 'security_alerts')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(UserDevice)
class UserDeviceAdmin(ModelAdmin):
    """Admin for user devices"""
    list_display = ('user', 'device_name', 'device_type', 'operating_system', 'is_trusted', 'last_seen', 'is_active')
    list_filter = ('device_type', 'is_trusted', 'is_active', 'first_seen', 'last_seen')
    search_fields = ('user__email', 'device_name', 'browser', 'last_ip', 'operating_system')
    readonly_fields = ('first_seen', 'last_seen', 'login_count', 'user_agent')
    
    fieldsets = (
        ('User', {
            'fields': ('user',)
        }),
        ('Device Info', {
            'fields': ('device_id', 'device_name', 'device_type')
        }),
        ('Technical Details', {
            'fields': ('operating_system', 'browser', 'user_agent')
        }),
        ('Security', {
            'fields': ('is_trusted',)
        }),
        ('Network Info', {
            'fields': ('last_ip', 'last_location')
        }),
        ('Usage Stats', {
            'fields': ('first_seen', 'last_seen', 'login_count', 'is_active'),
            'classes': ('collapse',)
        })
    )
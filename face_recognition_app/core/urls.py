"""
Core API URLs for third-party authentication service
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'core'

# Create router for ViewSets
router = DefaultRouter()
router.register(r'config', views.SystemConfigurationViewSet)
router.register(r'audit-logs', views.AuditLogViewSet)
router.register(r'security-events', views.SecurityEventViewSet)

urlpatterns = [
    # ViewSet routes
    path('', include(router.urls)),
    
    # Authentication endpoints
    path('auth/client/', views.authenticate_client, name='authenticate_client'),
    path('auth/user/', views.authenticate_client_user, name='authenticate_client_user'),
    
    # System endpoints
    path('status/', views.system_status, name='system_status'),
    path('health/', views.health_check, name='health_check'),
    path('info/', views.client_info, name='client_info'),
    
    # Security
    path('security/event/', views.log_security_event, name='log_security_event'),
]

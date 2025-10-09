"""
Core API URLs
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

app_name = 'core'

urlpatterns = [
    # Authentication
    path('auth/register/', views.UserRegistrationView.as_view(), name='register'),
    path('auth/token/', views.CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/profile/', views.UserProfileView.as_view(), name='profile'),
    
    # Face Recognition Enrollment
    path('enrollment/create/', views.EnrollmentSessionCreateView.as_view(), name='enrollment_create'),
    path('enrollment/process-frame/', views.EnrollmentFrameProcessView.as_view(), name='enrollment_process_frame'),
    
    # Face Recognition Authentication
    path('auth/face/create/', views.AuthenticationCreateView.as_view(), name='auth_create'),
    path('auth/face/process-frame/', views.AuthenticationFrameProcessView.as_view(), name='auth_process_frame'),
    path('auth/face/public/create/', views.PublicAuthenticationCreateView.as_view(), name='auth_public_create'),
    
    # WebRTC Signaling
    path('webrtc/signal/', views.WebRTCSignalingView.as_view(), name='webrtc_signal'),
    
    # User Management
    path('user/devices/', views.UserDevicesView.as_view(), name='user_devices'),
    path('user/auth-history/', views.AuthenticationHistoryView.as_view(), name='auth_history'),
    path('user/security-alerts/', views.SecurityAlertsView.as_view(), name='security_alerts'),
    
    # System
    path('system/status/', views.system_status, name='system_status'),
]

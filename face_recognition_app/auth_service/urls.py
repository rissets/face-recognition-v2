"""
URL configuration for auth_service app
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'auth_service'

# Create router for ViewSets
router = DefaultRouter()
router.register(r'sessions', views.AuthenticationSessionViewSet)
router.register(r'enrollments', views.FaceEnrollmentViewSet)
router.register(r'metrics', views.SystemMetricsViewSet)

urlpatterns = [
    # ViewSet routes
    path('', include(router.urls)),
    
    # Session management
    path('enrollment/create/', views.create_enrollment_session, name='create_enrollment_session'),
    path('authentication/create/', views.create_authentication_session, name='create_authentication_session'),
    path('process-image/', views.process_face_image, name='process_face_image'),
    path('session/<str:session_token>/status/', views.get_session_status, name='get_session_status'),
    
    # Analytics
    path('analytics/', views.get_client_analytics, name='get_client_analytics'),
]

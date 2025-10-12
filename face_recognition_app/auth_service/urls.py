"""
URL configuration for auth_service app
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'auth_service'

# Create router for ViewSets
router = DefaultRouter()
router.register(r'sessions', views.AuthenticationSessionViewSet, basename='authenticationsession')
router.register(r'enrollments', views.FaceEnrollmentViewSet)

urlpatterns = [
    # ViewSet routes
    path('', include(router.urls)),
    
    # Session management
    path('enrollment/', views.create_enrollment_session, name='create_enrollment_session'),
    path('authentication/', views.create_authentication_session, name='create_authentication_session'),
    path('process-image/', views.process_face_image, name='process_face_image'),
]

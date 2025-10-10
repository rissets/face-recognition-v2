"""
URL configuration for face_app project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)
from face_app import admin_dashboard  # noqa: F401

urlpatterns = [
    path("admin/", admin.site.urls),
    
    # API Documentation
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
    
    # API Endpoints - New Third-Party Service Architecture
    path('api/core/', include(('core.urls', 'core'), namespace='core-prefixed')),
    path('api/clients/', include('clients.urls')),
    path('api/auth/', include('auth_service.urls')),
    path('api/webhooks/', include('webhooks.urls')),
    
    # Legacy endpoints (temporarily maintained for backward compatibility)
    path('api/', include('core.urls')),
    path('api/recognition/', include('recognition.urls')),
    path('api/analytics/', include('analytics.urls')),
    path('api/streaming/', include('streaming.urls')),
    
    # Test Interface
    # path('test/', include('core.test_urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

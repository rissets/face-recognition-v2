"""
URL configuration for clients app
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'clients'

# Create router for ViewSets
router = DefaultRouter()
router.register(r'clients', views.ClientViewSet)
router.register(r'users', views.ClientUserViewSet)
router.register(r'usage', views.ClientAPIUsageViewSet)
router.register(r'webhook-logs', views.ClientWebhookLogViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
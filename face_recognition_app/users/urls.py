"""
Users app URLs
"""
from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    # User management
    path('profile/', views.UserProfileView.as_view(), name='profile'),
    path('devices/', views.UserDeviceListView.as_view(), name='device_list'),
    path('devices/<uuid:pk>/', views.UserDeviceDetailView.as_view(), name='device_detail'),
]
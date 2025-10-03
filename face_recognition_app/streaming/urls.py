"""
Streaming app URLs
"""
from django.urls import path
from . import views

app_name = 'streaming'

urlpatterns = [
    # WebRTC sessions
    path('sessions/', views.StreamingSessionListView.as_view(), name='session_list'),
    path('sessions/<uuid:pk>/', views.StreamingSessionDetailView.as_view(), name='session_detail'),
    path('sessions/create/', views.StreamingSessionCreateView.as_view(), name='session_create'),
    
    # WebRTC signaling
    path('signaling/', views.WebRTCSignalingView.as_view(), name='signaling'),
]
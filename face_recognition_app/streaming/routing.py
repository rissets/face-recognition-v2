"""
WebSocket routing for streaming app
"""
from django.urls import re_path
from . import consumers
from auth_service import consumers as auth_consumers

websocket_urlpatterns = [
    re_path(r'ws/face-recognition/(?P<session_token>[^/]+)/$', consumers.FaceRecognitionConsumer.as_asgi()),
    re_path(r'ws/auth/process-image/(?P<session_token>[^/]+)/$', auth_consumers.AuthProcessConsumer.as_asgi()),
]
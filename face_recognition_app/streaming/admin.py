"""
Streaming admin configuration - Simplified working version
"""
from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import StreamingSession, WebRTCSignal


admin.site.register(StreamingSession)
admin.site.register(WebRTCSignal)
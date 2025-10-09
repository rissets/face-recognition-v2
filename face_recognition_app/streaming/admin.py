"""
Streaming admin configuration using django-unfold.
"""
from django.contrib import admin
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from unfold.admin import ModelAdmin

from .models import StreamingSession, WebRTCSignal


@admin.register(StreamingSession)
class StreamingSessionAdmin(ModelAdmin):
    """Admin interface for WebRTC streaming sessions."""

    list_display = (
        "session_token",
        "session_type",
        "status",
        "user_display",
        "created_at",
        "completed_at",
    )
    list_filter = (
        "session_type",
        "status",
        ("created_at", admin.DateFieldListFilter),
        ("completed_at", admin.DateFieldListFilter),
    )
    search_fields = ("session_token", "user__email", "user__first_name", "user__last_name")
    readonly_fields = (
        "session_token",
        "created_at",
        "connected_at",
        "completed_at",
        "session_data_pretty",
    )
    ordering = ("-created_at",)
    date_hierarchy = "created_at"

    fieldsets = (
        (_("Session details"), {"fields": ("session_token", "session_type", "status", "user")}),
        (
            _("Connection"),
            {
                "fields": (
                    "remote_address",
                    "peer_connection_id",
                    "video_quality",
                    "frame_rate",
                    "bitrate",
                )
            },
        ),
        (
            _("Configuration"),
            {
                "classes": ("collapse",),
                "fields": ("ice_servers", "constraints"),
            },
        ),
        (
            _("Timeline"),
            {
                "fields": ("created_at", "connected_at", "completed_at"),
            },
        ),
        (
            _("Telemetry"),
            {
                "classes": ("collapse",),
                "fields": ("session_data_pretty", "error_log"),
            },
        ),
    )

    actions = ["mark_completed", "mark_failed"]

    def user_display(self, obj):
        """Return user email or placeholder."""
        if obj.user:
            return obj.user.email
        return _("Anonymous")

    user_display.short_description = _("User")

    def session_data_pretty(self, obj):
        """Provide pretty-printed session data."""
        import json

        if not obj.session_data:
            return "-"
        formatted = json.dumps(obj.session_data, indent=2, sort_keys=True)
        return format_html(
            "<pre style='white-space: pre-wrap;'>{}</pre>",
            formatted,
        )

    session_data_pretty.short_description = _("Session data")

    def mark_completed(self, request, queryset):
        """Bulk mark sessions as completed."""
        updated = queryset.filter(status__in=["processing", "connected"]).update(status="completed")
        self.message_user(request, _("%d sessions marked as completed.") % updated)

    mark_completed.short_description = _("Mark selected sessions as completed")

    def mark_failed(self, request, queryset):
        """Bulk mark sessions as failed."""
        updated = queryset.exclude(status="failed").update(status="failed")
        self.message_user(request, _("%d sessions marked as failed.") % updated)

    mark_failed.short_description = _("Mark selected sessions as failed")


@admin.register(WebRTCSignal)
class WebRTCSignalAdmin(ModelAdmin):
    """Admin interface for WebRTC signalling messages."""

    list_display = ("session", "signal_type", "direction", "created_at")
    list_filter = ("signal_type", "direction", ("created_at", admin.DateFieldListFilter))
    search_fields = ("session__session_token",)
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)

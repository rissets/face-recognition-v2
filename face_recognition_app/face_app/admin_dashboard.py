"""
Custom admin dashboard integration using django-unfold.
"""
from __future__ import annotations
from datetime import timedelta
from typing import Dict, Any

from django.contrib import admin
from django.db.models import Count, Avg, Q, Case, When, IntegerField
from django.db.models.functions import TruncDate
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from analytics.models import (
    AuthenticationLog,
    FaceRecognitionStats,
    SecurityAlert,
    SystemMetrics,
    UserBehaviorAnalytics,
)
from recognition.models import (
    AuthenticationAttempt,
    EnrollmentSession,
    FaceEmbedding,
)
from streaming.models import StreamingSession
from users.models import CustomUser, UserDevice


def _safe_percentage(part: int, whole: int) -> float:
    """Return rounded percentage or 0 when denominator is 0."""
    if not whole:
        return 0.0
    return round((part / whole) * 100, 1)


def _get_auth_result_labels() -> Dict[str, str]:
    """Map authentication result codes to human readable labels."""
    return dict(AuthenticationAttempt.SUCCESS_STATUS)


def get_dashboard_context() -> Dict[str, Any]:
    """Collect metrics for the admin dashboard."""
    now = timezone.now()
    last_24h = now - timedelta(hours=24)
    previous_24h_start = last_24h - timedelta(hours=24)
    last_7d = now - timedelta(days=7)
    last_30d = now - timedelta(days=30)

    # User metrics
    total_users = CustomUser.objects.count()
    face_enrolled = CustomUser.objects.filter(face_enrolled=True).count()
    face_enabled = CustomUser.objects.filter(face_auth_enabled=True).count()
    active_users = CustomUser.objects.filter(last_login__gte=last_7d).count()
    new_users = CustomUser.objects.filter(created_at__gte=last_7d).count()

    # Enrollment and embeddings
    total_embeddings = FaceEmbedding.objects.count()
    active_embeddings = FaceEmbedding.objects.filter(is_active=True).count()
    enrollments_completed = EnrollmentSession.objects.filter(
        status="completed"
    ).count()
    enrollments_active = EnrollmentSession.objects.filter(
        status__in=["pending", "in_progress"]
    ).count()

    # Authentication metrics
    attempts_24h = AuthenticationAttempt.objects.filter(created_at__gte=last_24h)
    auth_success_24h = attempts_24h.filter(result="success").count()
    auth_total_24h = attempts_24h.count()

    previous_window = AuthenticationAttempt.objects.filter(
        created_at__gte=previous_24h_start, created_at__lt=last_24h
    )
    previous_success = previous_window.filter(result="success").count()
    previous_total = previous_window.count()
    previous_rate = _safe_percentage(previous_success, previous_total)
    current_rate = _safe_percentage(auth_success_24h, auth_total_24h)
    success_rate_delta = round(current_rate - previous_rate, 1)

    # Aggregated summary for the last 7 days
    summary_window = AuthenticationAttempt.objects.filter(created_at__gte=last_7d)
    summary_stats = summary_window.aggregate(
        total=Count("id"),
        success=Count("id", filter=Q(result="success")),
        avg_similarity=Avg("similarity_score"),
        avg_liveness=Avg("liveness_score"),
        avg_quality=Avg("quality_score"),
    )

    summary_total = summary_stats["total"] or 0
    summary_success = summary_stats["success"] or 0

    # Failure breakdown
    result_labels = _get_auth_result_labels()
    failure_breakdown = (
        summary_window.exclude(result="success")
        .values("result")
        .annotate(total=Count("id"))
        .order_by("-total")[:5]
    )
    failure_data = [
        {
            "label": result_labels.get(item["result"], item["result"]),
            "code": item["result"],
            "value": item["total"],
        }
        for item in failure_breakdown
    ]

    # Authentication method mix (last 7 days)
    auth_method_breakdown = list(
        AuthenticationLog.objects.filter(created_at__gte=last_7d)
        .values("auth_method")
        .annotate(total=Count("id"))
        .order_by("-total")
    )
    auth_method_labels = [
        (item["auth_method"] or _("Unknown")).title() for item in auth_method_breakdown
    ]
    auth_method_values = [item["total"] for item in auth_method_breakdown]

    # Device mix
    device_breakdown = list(
        UserDevice.objects.values("device_type")
        .annotate(total=Count("id"))
        .order_by("-total")[:6]
    )
    device_labels = [
        (item["device_type"] or _("Unknown")).title() for item in device_breakdown
    ]
    device_values = [item["total"] for item in device_breakdown]

    # Risk distribution
    risk_distribution_qs = list(
        UserBehaviorAnalytics.objects.values("risk_level")
        .annotate(total=Count("id"))
        .order_by()
    )
    risk_level_map = dict(UserBehaviorAnalytics._meta.get_field("risk_level").choices)
    risk_labels = [
        risk_level_map.get(item["risk_level"], item["risk_level"]) for item in risk_distribution_qs
    ]
    risk_values = [item["total"] for item in risk_distribution_qs]

    risk_priority = Case(
        When(risk_level="critical", then=0),
        When(risk_level="high", then=1),
        When(risk_level="medium", then=2),
        default=3,
        output_field=IntegerField(),
    )
    top_risk_profiles_qs = (
        UserBehaviorAnalytics.objects.select_related("user")
        .annotate(priority=risk_priority)
        .order_by("priority", "-suspicious_activity_count", "-auth_success_rate")[:5]
    )
    top_risk_profiles = [
        {
            "user": analytics.user.get_full_name() or analytics.user.email,
            "email": analytics.user.email,
            "risk_level": analytics.get_risk_level_display(),
            "suspicious_activity_count": analytics.suspicious_activity_count,
            "auth_success_rate": (analytics.auth_success_rate or 0.0) * 100,
            "last_assessed": analytics.last_risk_assessment,
        }
        for analytics in top_risk_profiles_qs
        if analytics.user
    ]

    # Success rate trend based on daily aggregates (fallback to attempts when missing)
    stats_window_qs = FaceRecognitionStats.objects.filter(
        date__gte=now.date() - timedelta(days=6), hour__isnull=True
    ).order_by("date")
    stats_window = list(stats_window_qs)
    if stats_window:
        labels = [stat.date.strftime("%d %b") for stat in stats_window]
        success_points = [round(stat.success_rate, 1) for stat in stats_window]
    else:
        fallback_stats_qs = (
            summary_window.annotate(day=TruncDate("created_at"))
            .values("day")
            .annotate(
                total=Count("id"),
                success=Count("id", filter=Q(result="success")),
            )
            .order_by("day")
        )
        fallback_stats = list(fallback_stats_qs)
        labels = [item["day"].strftime("%d %b") for item in fallback_stats]
        success_points = [
            _safe_percentage(item["success"], item["total"]) for item in fallback_stats
        ]

    # Streaming sessions
    streaming_active = StreamingSession.objects.filter(
        status__in=["connecting", "connected", "processing"]
    ).count()
    streaming_total_24h = StreamingSession.objects.filter(
        created_at__gte=last_24h
    ).count()

    # Security alerts
    open_alerts = SecurityAlert.objects.filter(resolved=False).count()
    high_alerts = SecurityAlert.objects.filter(
        resolved=False, severity__in=["high", "critical"]
    ).count()
    recent_alerts = list(
        SecurityAlert.objects.select_related("user")
        .order_by("-created_at")[:6]
        .values(
            "id",
            "title",
            "severity",
            "alert_type",
            "created_at",
            "acknowledged",
            "resolved",
            "user__email",
        )
    )

    # Recent authentication attempts timeline
    recent_attempts_qs = (
        AuthenticationAttempt.objects.select_related("user")
        .order_by("-created_at")[:8]
        .values(
            "id",
            "user__email",
            "result",
            "similarity_score",
            "quality_score",
            "liveness_score",
            "created_at",
        )
    )
    recent_attempts = [
        {
            "id": attempt["id"],
            "user": attempt["user__email"] or _("Anonymous"),
            "result": attempt["result"],
            "result_label": result_labels.get(attempt["result"], attempt["result"]),
            "similarity": round(attempt["similarity_score"] or 0.0, 2),
            "quality": round(attempt["quality_score"] or 0.0, 2),
            "liveness": round(attempt["liveness_score"] or 0.0, 2),
            "created_at": attempt["created_at"],
        }
        for attempt in recent_attempts_qs
    ]

    # Streaming session timeline
    recent_sessions = list(
        StreamingSession.objects.select_related("user")
        .order_by("-created_at")[:5]
        .values(
            "session_token",
            "session_type",
            "status",
            "user__email",
            "created_at",
            "completed_at",
        )
    )

    # System metrics overview
    latest_metrics = list(
        SystemMetrics.objects.order_by("-timestamp")
        .values("metric_name", "metric_type", "value", "unit", "timestamp")[:5]
    )

    # Authentication log trends (request volume)
    log_trail_qs = (
        AuthenticationLog.objects.filter(created_at__gte=last_30d)
        .annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(count=Count("id"))
        .order_by("day")
    )
    log_trail = list(log_trail_qs)
    log_labels = [item["day"].strftime("%d %b") for item in log_trail]
    log_values = [item["count"] for item in log_trail]

    # Derive medians for gauges
    gauge_similarity = round(summary_stats["avg_similarity"] or 0.0, 2)
    gauge_liveness = round(summary_stats["avg_liveness"] or 0.0, 2)
    gauge_quality = round(summary_stats["avg_quality"] or 0.0, 2)

    cards = [
        {
            "title": _("Registered Users"),
            "value": total_users,
            "display": f"{total_users:,}",
            "icon": "👥",
            "delta": {
                "value": new_users,
                "is_positive": new_users >= 0,
                "label": _("joined in last 7 days"),
                "display": "0"
                if new_users == 0
                else f"{'+' if new_users > 0 else ''}{new_users:,}",
            },
            "description": _(
                "{enrolled}/{total} enrolled · {active} active this week"
            ).format(
                enrolled=face_enrolled,
                total=total_users,
                active=active_users,
            ),
        },
        {
            "title": _("Face Authentication Success"),
            "value": current_rate,
            "display": f"{current_rate:.1f}%",
            "icon": "✅",
            "delta": {
                "value": success_rate_delta,
                "is_positive": success_rate_delta >= 0,
                "label": _("vs previous 24h"),
                "display": f"{success_rate_delta:+.1f} pp"
                if success_rate_delta
                else "0.0 pp",
            },
            "description": _("{success}/{total} attempts").format(
                success=auth_success_24h, total=auth_total_24h
            ),
        },
        {
            "title": _("Active Streaming Sessions"),
            "value": streaming_active,
            "display": f"{streaming_active:,}",
            "icon": "📹",
            "delta": {
                "value": streaming_total_24h,
                "is_positive": streaming_total_24h >= 0,
                "label": _("started in last 24h"),
                "display": "0"
                if streaming_total_24h == 0
                else f"{'+' if streaming_total_24h > 0 else ''}{streaming_total_24h:,}",
            },
            "description": _("Latest session status updates in real-time."),
        },
        {
            "title": _("Security Alerts"),
            "value": open_alerts,
            "display": f"{open_alerts:,}",
            "icon": "🛡️",
            "delta": {
                "value": high_alerts,
                "is_positive": False,
                "label": _("high priority open"),
                "display": f"{high_alerts:,}",
            },
            "description": _("Stay ahead of suspicious activity."),
        },
    ]

    chart_success_rate = {
        "title": _("Success rate (rolling 7 days)"),
        "subtitle": _("How consistently users pass face verification."),
        "canvas_id": "chart-success-rate",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": _("Success rate"),
                    "data": success_points,
                    "color": "#2563eb",
                    "fill": True,
                }
            ],
            "y_max": 100,
            "y_suffix": "%",
        },
    }

    chart_failures = {
        "title": _("Top failure reasons"),
        "subtitle": _("Focus remediation where users struggle the most."),
        "canvas_id": "chart-failures",
        "data": {
            "labels": [item["label"] for item in failure_data],
            "datasets": [
                {
                    "label": _("Attempts"),
                    "data": [item["value"] for item in failure_data],
                    "color": "#f97316",
                }
            ],
            "y_max": max([item["value"] for item in failure_data] + [1]),
            "y_suffix": "",
        },
    }

    chart_requests = {
        "title": _("Authentication request volume"),
        "subtitle": _("Total API requests reaching the authentication pipeline."),
        "canvas_id": "chart-request-volume",
        "data": {
            "labels": log_labels,
            "datasets": [
                {
                    "label": _("Requests"),
                    "data": log_values,
                    "color": "#0ea5e9",
                    "fill": False,
                }
            ],
            "y_max": max(log_values + [1]),
            "y_suffix": "",
        },
    }

    chart_auth_methods = {
        "title": _("Authentication methods"),
        "subtitle": _("Login attempts split by verification channel (7 days)."),
        "canvas_id": "chart-auth-methods",
        "data": {
            "labels": auth_method_labels or [_("No data")],
            "datasets": [
                {
                    "label": _("Attempts"),
                    "data": auth_method_values or [0],
                    "color": "#14b8a6",
                }
            ],
            "y_max": max((auth_method_values or [0]) + [1]),
            "y_suffix": "",
        },
    }

    chart_device_types = {
        "title": _("Trusted device mix"),
        "subtitle": _("Top devices recently used for login."),
        "canvas_id": "chart-device-types",
        "data": {
            "labels": device_labels or [_("No data")],
            "datasets": [
                {
                    "label": _("Devices"),
                    "data": device_values or [0],
                    "color": "#6366f1",
                }
            ],
            "y_max": max((device_values or [0]) + [1]),
            "y_suffix": "",
        },
    }

    chart_risk_levels = {
        "title": _("Risk distribution"),
        "subtitle": _("Current behavioral risk assessment across users."),
        "canvas_id": "chart-risk-levels",
        "data": {
            "labels": risk_labels or [_("No data")],
            "datasets": [
                {
                    "label": _("Profiles"),
                    "data": risk_values or [0],
                    "color": "#f97316",
                }
            ],
            "y_max": max((risk_values or [0]) + [1]),
            "y_suffix": "",
        },
    }

    gauges = [
        {
            "label": _("Avg similarity"),
            "value": gauge_similarity,
            "max": 1.0,
            "percentage": round(min(max(gauge_similarity / 1.0, 0.0), 1.0) * 100, 1),
        },
        {
            "label": _("Avg liveness"),
            "value": gauge_liveness,
            "max": 1.0,
            "percentage": round(min(max(gauge_liveness / 1.0, 0.0), 1.0) * 100, 1),
        },
        {
            "label": _("Avg quality"),
            "value": gauge_quality,
            "max": 1.0,
            "percentage": round(min(max(gauge_quality / 1.0, 0.0), 1.0) * 100, 1),
        },
    ]

    engine_overview = {
        "enrolled_users": face_enrolled,
        "face_enabled": face_enabled,
        "active_embeddings": active_embeddings,
        "total_embeddings": total_embeddings,
        "enrollments_active": enrollments_active,
        "enrollments_completed": enrollments_completed,
        "recent_metrics": latest_metrics,
    }

    return {
        "dashboard_cards": cards,
        "chart_success_rate": chart_success_rate,
        "chart_failures": chart_failures,
        "chart_requests": chart_requests,
        "chart_auth_methods": chart_auth_methods,
        "chart_device_types": chart_device_types,
        "chart_risk_levels": chart_risk_levels,
        "gauges": gauges,
        "auth_summary": {
            "total_attempts": summary_total,
            "success_rate": _safe_percentage(summary_success, summary_total),
            "average_similarity": gauge_similarity,
            "average_liveness": gauge_liveness,
            "average_quality": gauge_quality,
            "failures": failure_data,
        },
        "recent_attempts": recent_attempts,
        "recent_sessions": recent_sessions,
        "recent_alerts": recent_alerts,
        "engine_overview": engine_overview,
        "top_risk_profiles": top_risk_profiles,
    }


def _inject_dashboard_context(request, extra_context=None):
    """Wrap the default admin index view with additional context."""
    if extra_context is None:
        extra_context = {}
    extra_context.update(get_dashboard_context())
    return extra_context


def _wrap_admin_index():
    """Attach the dashboard template and context builder to the default admin site."""
    original_index = admin.site.index

    def custom_index(request, extra_context=None):
        context = _inject_dashboard_context(request, extra_context)
        return original_index(request, extra_context=context)

    custom_index.__name__ = original_index.__name__
    custom_index.__doc__ = original_index.__doc__
    admin.site.index = custom_index
    admin.site.index_template = "admin/dashboard.html"


# Perform the patch as soon as the module is imported.
_wrap_admin_index()

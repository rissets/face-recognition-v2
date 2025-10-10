"""
Custom admin dashboard context and view registration.
"""
from datetime import timedelta
from types import MethodType

from django.contrib import admin
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDay
from django.template.response import TemplateResponse
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from analytics.models import (
    AuthenticationLog,
    SecurityAlert,
    UserBehaviorAnalytics,
    SystemMetrics as AnalyticsSystemMetric,
)
from auth_service.models import (
    AuthenticationSession,
    FaceEnrollment,
    FaceRecognitionAttempt,
    LivenessDetectionResult,
)
from clients.models import Client, ClientAPIUsage, ClientUser
from users.models import UserDevice


def _format_number(value):
    return f"{value:,}" if isinstance(value, int) else f"{value:.2f}"


def _format_delta(value):
    if value > 0:
        return f"+{value}"
    if value < 0:
        return str(value)
    return "0"


def _chart_payload(title, subtitle, canvas_id, labels, dataset, y_max=None):
    return {
        "title": title,
        "subtitle": subtitle,
        "canvas_id": canvas_id,
        "data": {
            "labels": labels,
            "datasets": [dataset],
            **({"y_max": y_max} if y_max is not None else {}),
        },
    }


def _build_dashboard_cards(now, seven_days_ago):
    this_week = Client.objects.filter(created_at__gte=seven_days_ago).count()
    last_week = Client.objects.filter(
        created_at__lt=seven_days_ago, created_at__gte=seven_days_ago - timedelta(days=7)
    ).count()
    active_clients = Client.objects.filter(status="active").count()

    calls_today = ClientAPIUsage.objects.filter(created_at__gte=now - timedelta(days=1)).count()
    calls_yesterday = ClientAPIUsage.objects.filter(
        created_at__lt=now - timedelta(days=1),
        created_at__gte=now - timedelta(days=2),
    ).count()

    attempts_week = FaceRecognitionAttempt.objects.filter(created_at__gte=seven_days_ago)
    total_attempts = attempts_week.count()
    success_attempts = attempts_week.filter(result="success").count()
    success_rate = round((success_attempts / total_attempts) * 100, 2) if total_attempts else 0.0

    liveness_week = LivenessDetectionResult.objects.filter(created_at__gte=seven_days_ago)
    live_results = liveness_week.filter(status="live").count()
    liveness_rate = round((live_results / liveness_week.count()) * 100, 2) if liveness_week.exists() else 0.0

    return [
        {
            "title": _("Active clients"),
            "icon": "👥",
            "display": _format_number(active_clients),
            "description": _("Tenant with status aktif"),
            "delta": {
                "value": this_week - last_week,
                "display": _format_delta(this_week - last_week),
                "label": _("vs 7 hari lalu"),
            },
        },
        {
            "title": _("API calls (24h)"),
            "icon": "⚡️",
            "display": _format_number(calls_today),
            "description": _("Permintaan yang tercatat dalam 24 jam"),
            "delta": {
                "value": calls_today - calls_yesterday,
                "display": _format_delta(calls_today - calls_yesterday),
                "label": _("dibanding hari sebelumnya"),
            },
        },
        {
            "title": _("Auth success rate"),
            "icon": "✅",
            "display": f"{success_rate:.1f}%",
            "description": _("Rasio keberhasilan autentikasi 7 hari"),
            "delta": {
                "value": success_rate,
                "display": _("Stabil") if success_rate else _("Belum ada data"),
                "label": _("agregasi mingguan"),
            },
        },
        {
            "title": _("Liveness pass rate"),
            "icon": "🛡️",
            "display": f"{liveness_rate:.1f}%",
            "description": _("Persentase deteksi hidup berhasil"),
            "delta": {
                "value": liveness_rate,
                "display": _("Stabil") if liveness_rate else _("Belum ada data"),
                "label": _("agregasi mingguan"),
            },
        },
    ]


def _build_success_rate_chart(seven_days_ago):
    attempts = (
        FaceRecognitionAttempt.objects.filter(created_at__gte=seven_days_ago)
        .annotate(day=TruncDay("created_at"))
        .values("day")
        .annotate(
            total=Count("id"),
            success=Count("id", filter=Q(result="success")),
        )
        .order_by("day")
    )
    labels, values = [], []
    for entry in attempts:
        labels.append(entry["day"].strftime("%d %b"))
        total = entry["total"]
        success = entry["success"]
        values.append(round((success / total) * 100, 2) if total else 0)

    dataset = {
        "label": _("Success rate"),
        "data": values,
        "color": "#22c55e",
        "fill": True,
    }
    return _chart_payload(
        _("Authentication success rate"),
        _("Rasio harian dalam 7 hari terakhir"),
        "chart-success-rate",
        labels,
        dataset,
        y_max=100,
    )


def _build_failure_chart(seven_days_ago):
    attempts = (
        FaceRecognitionAttempt.objects.filter(created_at__gte=seven_days_ago)
        .values("result")
        .annotate(count=Count("id"))
        .exclude(result="success")
        .order_by("-count")[:6]
    )
    result_labels = dict(FaceRecognitionAttempt.RESULT_CHOICES)
    labels = [result_labels.get(entry["result"], entry["result"]) for entry in attempts]
    values = [entry["count"] for entry in attempts]
    dataset = {
        "label": _("Failure count"),
        "data": values,
        "color": "#ef4444",
    }
    return _chart_payload(
        _("Dominant failure reasons"),
        _("Distribusi kegagalan autentikasi"),
        "chart-failures",
        labels,
        dataset,
    )


def _build_request_chart(seven_days_ago):
    usage = (
        ClientAPIUsage.objects.filter(created_at__gte=seven_days_ago)
        .annotate(day=TruncDay("created_at"))
        .values("day")
        .annotate(count=Count("id"))
        .order_by("day")
    )
    labels = [entry["day"].strftime("%d %b") for entry in usage]
    values = [entry["count"] for entry in usage]
    dataset = {
        "label": _("API calls"),
        "data": values,
        "color": "#2563eb",
        "fill": True,
    }
    return _chart_payload(
        _("API request volume"),
        _("Jumlah permintaan per hari"),
        "chart-request-volume",
        labels,
        dataset,
    )


def _build_auth_method_chart(thirty_days_ago):
    logs = (
        AuthenticationLog.objects.filter(created_at__gte=thirty_days_ago)
        .values("auth_method")
        .annotate(count=Count("id"))
        .order_by("-count")[:6]
    )
    method_labels = dict(AuthenticationLog._meta.get_field("auth_method").choices)
    labels = [method_labels.get(entry["auth_method"], entry["auth_method"]) for entry in logs]
    values = [entry["count"] for entry in logs]
    dataset = {
        "label": _("Authentications"),
        "data": values,
        "color": "#f97316",
    }
    return _chart_payload(
        _("Authentication methods"),
        _("Sebaran metode autentikasi 30 hari"),
        "chart-auth-methods",
        labels,
        dataset,
    )


def _build_device_chart(thirty_days_ago):
    devices = (
        UserDevice.objects.filter(last_seen__gte=thirty_days_ago, device_type__isnull=False)
        .exclude(device_type="")
        .values("device_type")
        .annotate(count=Count("id"))
        .order_by("-count")[:6]
    )
    labels = [entry["device_type"].title() for entry in devices]
    values = [entry["count"] for entry in devices]
    dataset = {
        "label": _("Devices"),
        "data": values,
        "color": "#0ea5e9",
    }
    return _chart_payload(
        _("Device distribution"),
        _("Perangkat admin yang terdaftar"),
        "chart-device-types",
        labels,
        dataset,
    )


def _build_risk_chart(thirty_days_ago):
    analytics = (
        UserBehaviorAnalytics.objects.filter(updated_at__gte=thirty_days_ago)
        .values("risk_level")
        .annotate(count=Count("id"))
        .order_by("risk_level")
    )
    labels = [entry["risk_level"].title() for entry in analytics]
    values = [entry["count"] for entry in analytics]
    dataset = {
        "label": _("Profiles"),
        "data": values,
        "color": "#a855f7",
    }
    return _chart_payload(
        _("Risk level distribution"),
        _("Profil perilaku pengguna"),
        "chart-risk-levels",
        labels,
        dataset,
    )


def _build_gauges(seven_days_ago):
    attempts = FaceRecognitionAttempt.objects.filter(created_at__gte=seven_days_ago)
    total = attempts.count()
    success_count = attempts.filter(result="success").count()
    success_rate = (success_count / total) * 100 if total else 0

    averages = attempts.aggregate(
        avg_liveness=Avg("liveness_score"),
        avg_quality=Avg("face_quality_score"),
        avg_similarity=Avg("similarity_score"),
    )

    return [
        {
            "label": _("Authentication success"),
            "percentage": round(success_rate, 2),
            "value": success_rate,
        },
        {
            "label": _("Average liveness score"),
            "percentage": round((averages["avg_liveness"] or 0) * 100, 2),
            "value": averages["avg_liveness"] or 0,
        },
        {
            "label": _("Average face quality"),
            "percentage": round((averages["avg_quality"] or 0) * 100, 2),
            "value": averages["avg_quality"] or 0,
        },
    ]


def _build_auth_summary(seven_days_ago):
    attempts = FaceRecognitionAttempt.objects.filter(created_at__gte=seven_days_ago)
    total = attempts.count()
    success = attempts.filter(result="success").count()

    averages = attempts.aggregate(
        avg_similarity=Avg("similarity_score"),
        avg_liveness=Avg("liveness_score"),
        avg_quality=Avg("face_quality_score"),
    )

    failures = (
        attempts.values("result")
        .annotate(count=Count("id"))
        .exclude(result="success")
        .order_by("-count")[:5]
    )
    result_labels = dict(FaceRecognitionAttempt.RESULT_CHOICES)
    failure_data = [
        {"label": result_labels.get(entry["result"], entry["result"]), "value": entry["count"]}
        for entry in failures
    ]

    success_rate = (success / total) * 100 if total else 0

    return {
        "total_attempts": total,
        "success_rate": round(success_rate, 2),
        "average_similarity": averages["avg_similarity"] or 0,
        "average_liveness": averages["avg_liveness"] or 0,
        "average_quality": averages["avg_quality"] or 0,
        "failures": failure_data,
    }


def _build_engine_overview(seven_days_ago):
    return {
        "enrolled_users": ClientUser.objects.filter(is_enrolled=True).count(),
        "face_enabled": ClientUser.objects.filter(face_auth_enabled=True).count(),
        "active_embeddings": FaceEnrollment.objects.filter(status="active").count(),
        "total_embeddings": FaceEnrollment.objects.count(),
        "enrollments_active": AuthenticationSession.objects.filter(
            session_type="enrollment", status="active"
        ).count(),
        "enrollments_completed": AuthenticationSession.objects.filter(
            session_type="enrollment", status="completed"
        ).count(),
        "recent_metrics": list(
            AnalyticsSystemMetric.objects.order_by("-timestamp")
            .values("metric_name", "value", "unit", "timestamp")[:5]
        ),
    }


def _build_recent_attempts():
    result_labels = dict(FaceRecognitionAttempt.RESULT_CHOICES)
    attempts = (
        FaceRecognitionAttempt.objects.select_related("matched_user")
        .order_by("-created_at")[:5]
    )
    data = []
    for attempt in attempts:
        matched = attempt.matched_user
        data.append(
            {
                "user": matched.display_name if matched else _("Unknown user"),
                "result": attempt.result,
                "result_label": result_labels.get(attempt.result, attempt.result),
                "similarity": attempt.similarity_score or 0,
                "quality": attempt.face_quality_score or 0,
                "liveness": attempt.liveness_score or 0,
                "created_at": attempt.created_at,
            }
        )
    return data


def _build_recent_sessions():
    sessions = (
        AuthenticationSession.objects.select_related("client_user")
        .order_by("-created_at")[:5]
    )
    data = []
    for session in sessions:
        client_user = session.client_user
        data.append(
            {
                "session_token": session.session_token,
                "session_type": session.session_type,
                "status": session.status,
                "user__email": (
                    client_user.display_name
                    if client_user
                    else _("Anonymous")
                ),
                "created_at": session.created_at,
                "completed_at": session.completed_at,
            }
        )
    return data


def _build_risk_profiles(seven_days_ago):
    profiles = (
        UserBehaviorAnalytics.objects.filter(
            updated_at__gte=seven_days_ago,
            risk_level__in=["high", "critical"],
        )
        .select_related("user")
        .order_by("-risk_level", "-suspicious_activity_count")[:5]
    )
    data = []
    for profile in profiles:
        user = profile.user
        data.append(
            {
                "user": user.get_full_name() or user.email,
                "email": user.email,
                "risk_level": profile.risk_level.title(),
                "suspicious_activity_count": profile.suspicious_activity_count,
                "auth_success_rate": (profile.auth_success_rate or 0) * 100,
                "last_assessed": profile.last_risk_assessment,
            }
        )
    return data


def _build_security_alerts():
    alerts = (
        SecurityAlert.objects.filter(resolved=False)
        .select_related("user")
        .order_by("-created_at")[:5]
    )
    data = []
    for alert in alerts:
        data.append(
            {
                "severity": alert.severity,
                "created_at": alert.created_at,
                "title": alert.title,
                "alert_type": alert.alert_type,
                "user__email": alert.user.email if alert.user else None,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved,
            }
        )
    return data


def build_dashboard_context():
    now = timezone.now()
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    return {
        "dashboard_cards": _build_dashboard_cards(now, seven_days_ago),
        "chart_success_rate": _build_success_rate_chart(seven_days_ago),
        "chart_failures": _build_failure_chart(seven_days_ago),
        "chart_requests": _build_request_chart(seven_days_ago),
        "chart_auth_methods": _build_auth_method_chart(thirty_days_ago),
        "chart_device_types": _build_device_chart(thirty_days_ago),
        "chart_risk_levels": _build_risk_chart(thirty_days_ago),
        "gauges": _build_gauges(seven_days_ago),
        "auth_summary": _build_auth_summary(seven_days_ago),
        "engine_overview": _build_engine_overview(seven_days_ago),
        "recent_attempts": _build_recent_attempts(),
        "recent_sessions": _build_recent_sessions(),
        "top_risk_profiles": _build_risk_profiles(seven_days_ago),
        "recent_alerts": _build_security_alerts(),
    }


def _dashboard_index(self, request, extra_context=None):
    context = self.each_context(request)
    context.update({"title": _("Dashboard"), "app_list": self.get_app_list(request)})
    if extra_context:
        context.update(extra_context)
    context.update(build_dashboard_context())
    return TemplateResponse(request, "admin/dashboard.html", context)


def register_dashboard():
    admin.site.index_template = "admin/dashboard.html"
    admin.site.index = MethodType(_dashboard_index, admin.site)


register_dashboard()

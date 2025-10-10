"""
Analytics helpers for tracking face recognition metrics and events
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

from django.db.models import Avg
from django.utils import timezone

from analytics.models import SystemMetrics, FaceRecognitionStats
from auth_service.models import (
    AuthenticationSession,
    FaceEnrollment,
    FaceRecognitionAttempt,
)
from clients.models import ClientAPIUsage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
# Internal helpers
# ---------------------------------------------------------------------------#


def _make_aware(dt: datetime) -> datetime:
    if timezone.is_naive(dt):
        return timezone.make_aware(dt)
    return dt


def _day_range(target_date: datetime.date) -> Tuple[datetime, datetime]:
    start = _make_aware(datetime.combine(target_date, datetime.min.time()))
    end = start + timedelta(days=1)
    return start, end


def _safe_metadata_lookup(obj, key, default=0):
    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


# ---------------------------------------------------------------------------#
# Tracking entry points
# ---------------------------------------------------------------------------#


def track_enrollment_metrics(client, enrollment, session):
    """Track enrollment completions as individual metrics."""
    try:
        outcome = "enrollment.completed" if enrollment.status == "active" else "enrollment.failed"
        SystemMetrics.objects.create(
            client=client,
            metric_type="counter",
            metric_name=outcome,
            value=1,
            unit="count",
            tags={
                "client_id": client.client_id,
                "enrollment_id": str(enrollment.id),
                "session_id": str(session.id),
                "status": enrollment.status,
            },
        )

        SystemMetrics.objects.create(
            client=client,
            metric_type="gauge",
            metric_name="enrollment.quality_score",
            value=enrollment.face_quality_score,
            unit="score",
            tags={"client_id": client.client_id},
        )

        SystemMetrics.objects.create(
            client=client,
            metric_type="gauge",
            metric_name="enrollment.liveness_score",
            value=enrollment.liveness_score,
            unit="score",
            tags={"client_id": client.client_id},
        )

        SystemMetrics.objects.create(
            client=client,
            metric_type="gauge",
            metric_name="enrollment.anti_spoofing_score",
            value=enrollment.anti_spoofing_score,
            unit="score",
            tags={"client_id": client.client_id},
        )

        frames_processed = _safe_metadata_lookup(session, "frames_processed", 0)
        if frames_processed:
            SystemMetrics.objects.create(
                client=client,
                metric_type="counter",
                metric_name="enrollment.frames_processed",
                value=frames_processed,
                unit="frames",
                tags={"client_id": client.client_id},
            )
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to track enrollment metrics")


def track_authentication_metrics(client, session, success=True, similarity_score=0.0):
    """Track authentication attempts as standalone metrics."""
    try:
        outcome = "authentication.success" if success else "authentication.failure"
        SystemMetrics.objects.create(
            client=client,
            metric_type="counter",
            metric_name=outcome,
            value=1,
            unit="count",
            tags={
                "client_id": client.client_id,
                "session_id": str(session.id),
            },
        )

        SystemMetrics.objects.create(
            client=client,
            metric_type="gauge",
            metric_name="authentication.similarity_score",
            value=similarity_score,
            unit="score",
            tags={"client_id": client.client_id},
        )

        frames_processed = _safe_metadata_lookup(session, "frames_processed", 0)
        if frames_processed:
            SystemMetrics.objects.create(
                client=client,
                metric_type="counter",
                metric_name="authentication.frames_processed",
                value=frames_processed,
                unit="frames",
                tags={"client_id": client.client_id},
            )
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to track authentication metrics")


def track_security_event(client, session, event_type, event_data):
    """Track security events (spoofing attempts, obstacle detection, etc.)."""
    try:
        SystemMetrics.objects.create(
            client=client,
            metric_type="counter",
            metric_name="security.event",
            value=1,
            unit="count",
            tags={
                "client_id": client.client_id,
                "session_id": str(session.id) if session else None,
                "event_type": event_type,
            },
            metadata=event_data,
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to track security event")


def update_face_recognition_stats(client, attempt):
    """Update aggregated FaceRecognitionStats for a given client/day."""
    try:
        date_key = attempt.created_at.date()
        stats_obj, _ = FaceRecognitionStats.objects.get_or_create(
            client=client,
            date=date_key,
            hour=None,
            defaults={
                "total_attempts": 0,
                "successful_attempts": 0,
                "failed_attempts": 0,
                "failed_similarity": 0,
                "failed_liveness": 0,
                "failed_quality": 0,
                "failed_obstacles": 0,
                "failed_no_face": 0,
                "failed_multiple_faces": 0,
                "failed_system_error": 0,
                "avg_response_time": 0.0,
                "avg_similarity_score": 0.0,
                "avg_liveness_score": 0.0,
                "avg_quality_score": 0.0,
                "unique_users": 0,
                "new_enrollments": 0,
            },
        )

        daily_attempts = FaceRecognitionAttempt.objects.filter(
            client=client,
            created_at__date=date_key,
        )

        stats_obj.total_attempts = daily_attempts.count()
        stats_obj.successful_attempts = daily_attempts.filter(result="success").count()
        stats_obj.failed_attempts = stats_obj.total_attempts - stats_obj.successful_attempts
        stats_obj.failed_similarity = daily_attempts.filter(result="no_match").count()
        stats_obj.failed_liveness = daily_attempts.filter(result="liveness_failed").count()
        stats_obj.failed_quality = daily_attempts.filter(result="quality_too_low").count()
        stats_obj.failed_obstacles = daily_attempts.filter(result="spoofing_detected").count()
        stats_obj.failed_multiple_faces = daily_attempts.filter(result="multiple_matches").count()
        stats_obj.failed_system_error = daily_attempts.filter(result="failed").count()

        averages = daily_attempts.aggregate(
            avg_response_time=Avg("processing_time_ms"),
            avg_similarity=Avg("similarity_score"),
            avg_liveness=Avg("liveness_score"),
            avg_quality=Avg("face_quality_score"),
        )
        stats_obj.avg_response_time = averages.get("avg_response_time") or 0.0
        stats_obj.avg_similarity_score = averages.get("avg_similarity") or 0.0
        stats_obj.avg_liveness_score = averages.get("avg_liveness") or 0.0
        stats_obj.avg_quality_score = averages.get("avg_quality") or 0.0

        stats_obj.unique_users = (
            daily_attempts.exclude(matched_user__isnull=True)
            .values("matched_user")
            .distinct()
            .count()
        )
        stats_obj.new_enrollments = FaceEnrollment.objects.filter(
            client=client,
            created_at__date=date_key,
        ).count()

        stats_obj.save()
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to update face recognition stats")


# ---------------------------------------------------------------------------#
# Reporting helpers
# ---------------------------------------------------------------------------#


def generate_daily_report(client, date=None) -> Dict:
    """Generate daily usage report for a client."""
    try:
        report_date = date or timezone.now().date()
        start, end = _day_range(report_date)

        daily_enrollments = FaceEnrollment.objects.filter(
            client=client,
            created_at__gte=start,
            created_at__lt=end,
        )
        daily_attempts = FaceRecognitionAttempt.objects.filter(
            client=client,
            created_at__gte=start,
            created_at__lt=end,
        )
        daily_sessions = AuthenticationSession.objects.filter(
            client=client,
            created_at__gte=start,
            created_at__lt=end,
        )
        daily_usage = ClientAPIUsage.objects.filter(
            client=client,
            created_at__gte=start,
            created_at__lt=end,
        )

        completed_enrollments = daily_enrollments.filter(status="active").count()
        failed_enrollments = daily_enrollments.exclude(status="active").count()
        successful_attempts = daily_attempts.filter(result="success").count()
        failed_attempts = daily_attempts.exclude(result="success").count()

        quality_metrics = daily_enrollments.aggregate(
            avg_quality=Avg("face_quality_score"),
            avg_liveness=Avg("liveness_score"),
            avg_anti_spoofing=Avg("anti_spoofing_score"),
        )
        similarity_metrics = daily_attempts.aggregate(
            avg_similarity=Avg("similarity_score"),
        )

        frames_processed = sum(
            _safe_metadata_lookup(session, "frames_processed", 0)
            for session in daily_sessions
        )

        enrollment_total = completed_enrollments + failed_enrollments
        auth_total = successful_attempts + failed_attempts

        return {
            "date": report_date.isoformat(),
            "client_id": client.client_id,
            "enrollments": {
                "completed": completed_enrollments,
                "failed": failed_enrollments,
                "total": enrollment_total,
            },
            "authentications": {
                "successful": successful_attempts,
                "failed": failed_attempts,
                "total": auth_total,
            },
            "quality_metrics": {
                "avg_enrollment_quality": round(quality_metrics["avg_quality"] or 0, 3),
                "avg_liveness_score": round(quality_metrics["avg_liveness"] or 0, 3),
                "avg_anti_spoofing_score": round(quality_metrics["avg_anti_spoofing"] or 0, 3),
                "avg_authentication_similarity": round(similarity_metrics["avg_similarity"] or 0, 3),
            },
            "total_frames_processed": frames_processed,
            "success_rates": {
                "enrollment_success_rate": round(
                    (completed_enrollments / enrollment_total) * 100 if enrollment_total else 0.0, 2
                ),
                "authentication_success_rate": round(
                    (successful_attempts / auth_total) * 100 if auth_total else 0.0, 2
                ),
            },
            "api_usage": {
                "calls": daily_usage.count(),
                "endpoints": list(daily_usage.values_list("endpoint", flat=True).distinct()),
            },
        }
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to generate daily report")
        return {}


def get_client_summary_stats(client, days=30) -> Dict:
    """Get summary statistics for a client over the last N days."""
    try:
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)

        enrollments = FaceEnrollment.objects.filter(
            client=client,
            created_at__gte=start_date,
            created_at__lte=end_date,
        )
        attempts = FaceRecognitionAttempt.objects.filter(
            client=client,
            created_at__gte=start_date,
            created_at__lte=end_date,
        )
        sessions = AuthenticationSession.objects.filter(
            client=client,
            created_at__gte=start_date,
            created_at__lte=end_date,
        )
        api_usage = ClientAPIUsage.objects.filter(
            client=client,
            created_at__gte=start_date,
            created_at__lte=end_date,
        )

        enrollments_completed = enrollments.filter(status="active").count()
        enrollments_failed = enrollments.exclude(status="active").count()
        auth_success = attempts.filter(result="success").count()
        auth_failed = attempts.exclude(result="success").count()

        frames_processed = sum(
            _safe_metadata_lookup(session, "frames_processed", 0)
            for session in sessions
        )

        aggregates = {
            "avg_enrollment_quality": enrollments.aggregate(avg=Avg("face_quality_score"))["avg"] or 0.0,
            "avg_liveness_score": enrollments.aggregate(avg=Avg("liveness_score"))["avg"] or 0.0,
            "avg_anti_spoofing_score": enrollments.aggregate(avg=Avg("anti_spoofing_score"))["avg"] or 0.0,
            "avg_similarity_score": attempts.aggregate(avg=Avg("similarity_score"))["avg"] or 0.0,
        }

        return {
            "period": f"Last {days} days",
            "date_range": {
                "start": start_date.date().isoformat(),
                "end": end_date.date().isoformat(),
            },
            "enrollments": {
                "completed": enrollments_completed,
                "failed": enrollments_failed,
                "total": enrollments.count(),
            },
            "authentications": {
                "successful": auth_success,
                "failed": auth_failed,
                "total": attempts.count(),
            },
            "quality_metrics": {key: round(value, 3) for key, value in aggregates.items()},
            "frames_processed": frames_processed,
            "api_usage": {
                "calls": api_usage.count(),
                "unique_endpoints": api_usage.values("endpoint").distinct().count(),
            },
        }
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to compute client summary statistics")
        return {}

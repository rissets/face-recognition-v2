"""
Analytics helpers for tracking face recognition metrics and events
"""
import logging
from django.utils import timezone
from django.db.models import Count, Avg, F, Q
from django.db import transaction
from analytics.models import SystemMetrics
from auth_service.models import AuthenticationSession, FaceEnrollment, FaceRecognitionAttempt

logger = logging.getLogger(__name__)


def track_enrollment_metrics(client, enrollment, session):
    """Track metrics for enrollment completion"""
    try:
        with transaction.atomic():
            # Update or create daily metrics
            today = timezone.now().date()
            metrics, created = SystemMetrics.objects.get_or_create(
                date=today,
                client=client,
                defaults={
                    'enrollments_completed': 0,
                    'enrollments_failed': 0,
                    'authentications_successful': 0,
                    'authentications_failed': 0,
                    'total_frame_processed': 0,
                    'avg_enrollment_quality': 0.0,
                    'avg_authentication_similarity': 0.0,
                    'avg_liveness_score': 0.0,
                    'avg_anti_spoofing_score': 0.0,
                }
            )
            
            # Update enrollment metrics
            if enrollment.status == 'active':
                metrics.enrollments_completed = F('enrollments_completed') + 1
                
                # Update quality averages
                current_quality = enrollment.face_quality_score or 0.0
                current_liveness = enrollment.liveness_score or 0.0
                current_anti_spoofing = enrollment.anti_spoofing_score or 0.0
                
                # Simple moving average update (can be improved with proper weighted average)
                total_enrollments = metrics.enrollments_completed
                if total_enrollments > 0:
                    metrics.avg_enrollment_quality = (
                        (metrics.avg_enrollment_quality * (total_enrollments - 1) + current_quality) / total_enrollments
                    )
                    metrics.avg_liveness_score = (
                        (metrics.avg_liveness_score * (total_enrollments - 1) + current_liveness) / total_enrollments
                    )
                    metrics.avg_anti_spoofing_score = (
                        (metrics.avg_anti_spoofing_score * (total_enrollments - 1) + current_anti_spoofing) / total_enrollments
                    )
            else:
                metrics.enrollments_failed = F('enrollments_failed') + 1
            
            # Update frame processing count
            frames_processed = session.metadata.get('frames_processed', 0)
            metrics.total_frame_processed = F('total_frame_processed') + frames_processed
            
            metrics.save(update_fields=[
                'enrollments_completed', 'enrollments_failed', 'total_frame_processed',
                'avg_enrollment_quality', 'avg_liveness_score', 'avg_anti_spoofing_score'
            ])
            
    except Exception as e:
        logger.error(f"Failed to track enrollment metrics: {e}")


def track_authentication_metrics(client, session, success=True, similarity_score=0.0):
    """Track metrics for authentication attempts"""
    try:
        with transaction.atomic():
            # Update or create daily metrics
            today = timezone.now().date()
            metrics, created = SystemMetrics.objects.get_or_create(
                date=today,
                client=client,
                defaults={
                    'enrollments_completed': 0,
                    'enrollments_failed': 0,
                    'authentications_successful': 0,
                    'authentications_failed': 0,
                    'total_frame_processed': 0,
                    'avg_enrollment_quality': 0.0,
                    'avg_authentication_similarity': 0.0,
                    'avg_liveness_score': 0.0,
                    'avg_anti_spoofing_score': 0.0,
                }
            )
            
            # Update authentication metrics
            if success:
                metrics.authentications_successful = F('authentications_successful') + 1
                
                # Update similarity average
                total_auths = metrics.authentications_successful
                if total_auths > 0:
                    metrics.avg_authentication_similarity = (
                        (metrics.avg_authentication_similarity * (total_auths - 1) + similarity_score) / total_auths
                    )
            else:
                metrics.authentications_failed = F('authentications_failed') + 1
            
            # Update frame processing count
            frames_processed = session.metadata.get('frames_processed', 0)
            metrics.total_frame_processed = F('total_frame_processed') + frames_processed
            
            metrics.save(update_fields=[
                'authentications_successful', 'authentications_failed', 'total_frame_processed',
                'avg_authentication_similarity'
            ])
            
    except Exception as e:
        logger.error(f"Failed to track authentication metrics: {e}")


def track_security_event(client, session, event_type, event_data):
    """Track security events (spoofing attempts, obstacle detection, etc.)"""
    try:
        with transaction.atomic():
            today = timezone.now().date()
            
            # You might want to create a SecurityEvent model for this
            # For now, we'll log it and potentially add to metrics
            logger.warning(f"Security event for client {client.client_id}: {event_type} - {event_data}")
            
            # Could increment security event counters in metrics
            # metrics.security_alerts = F('security_alerts') + 1
            
    except Exception as e:
        logger.error(f"Failed to track security event: {e}")


def generate_daily_report(client, date=None):
    """Generate daily usage report for a client"""
    try:
        if date is None:
            date = timezone.now().date()
        
        # Get metrics for the date
        try:
            metrics = SystemMetrics.objects.get(date=date, client=client)
        except SystemMetrics.DoesNotExist:
            # No activity for this date
            return {
                'date': date.isoformat(),
                'client_id': client.client_id,
                'enrollments': {'completed': 0, 'failed': 0},
                'authentications': {'successful': 0, 'failed': 0},
                'quality_metrics': {
                    'avg_enrollment_quality': 0.0,
                    'avg_authentication_similarity': 0.0,
                    'avg_liveness_score': 0.0,
                    'avg_anti_spoofing_score': 0.0,
                },
                'total_frames_processed': 0,
                'success_rates': {
                    'enrollment_success_rate': 0.0,
                    'authentication_success_rate': 0.0,
                }
            }
        
        # Calculate success rates
        total_enrollments = metrics.enrollments_completed + metrics.enrollments_failed
        total_authentications = metrics.authentications_successful + metrics.authentications_failed
        
        enrollment_success_rate = (
            (metrics.enrollments_completed / total_enrollments * 100) if total_enrollments > 0 else 0.0
        )
        authentication_success_rate = (
            (metrics.authentications_successful / total_authentications * 100) if total_authentications > 0 else 0.0
        )
        
        return {
            'date': date.isoformat(),
            'client_id': client.client_id,
            'enrollments': {
                'completed': metrics.enrollments_completed,
                'failed': metrics.enrollments_failed,
                'total': total_enrollments,
            },
            'authentications': {
                'successful': metrics.authentications_successful,
                'failed': metrics.authentications_failed,
                'total': total_authentications,
            },
            'quality_metrics': {
                'avg_enrollment_quality': round(metrics.avg_enrollment_quality, 3),
                'avg_authentication_similarity': round(metrics.avg_authentication_similarity, 3),
                'avg_liveness_score': round(metrics.avg_liveness_score, 3),
                'avg_anti_spoofing_score': round(metrics.avg_anti_spoofing_score, 3),
            },
            'total_frames_processed': metrics.total_frame_processed,
            'success_rates': {
                'enrollment_success_rate': round(enrollment_success_rate, 2),
                'authentication_success_rate': round(authentication_success_rate, 2),
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        return {}


def get_client_summary_stats(client, days=30):
    """Get summary statistics for a client over the last N days"""
    try:
        end_date = timezone.now().date()
        start_date = end_date - timezone.timedelta(days=days)
        
        metrics = SystemMetrics.objects.filter(
            client=client,
            date__gte=start_date,
            date__lte=end_date
        )
        
        if not metrics.exists():
            return {
                'period': f'Last {days} days',
                'total_enrollments': 0,
                'total_authentications': 0,
                'total_frames_processed': 0,
                'avg_success_rates': {'enrollment': 0.0, 'authentication': 0.0},
                'quality_trends': {},
            }
        
        # Aggregate metrics
        totals = metrics.aggregate(
            total_enrollments_completed=Count('enrollments_completed'),
            total_enrollments_failed=Count('enrollments_failed'),
            total_authentications_successful=Count('authentications_successful'),
            total_authentications_failed=Count('authentications_failed'),
            total_frames=Count('total_frame_processed'),
            avg_quality=Avg('avg_enrollment_quality'),
            avg_similarity=Avg('avg_authentication_similarity'),
            avg_liveness=Avg('avg_liveness_score'),
            avg_anti_spoofing=Avg('avg_anti_spoofing_score'),
        )
        
        return {
            'period': f'Last {days} days',
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
            },
            'totals': totals,
            'daily_metrics_count': metrics.count(),
        }
        
    except Exception as e:
        logger.error(f"Failed to get client summary stats: {e}")
        return {}
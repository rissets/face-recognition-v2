"""
Enhanced webhook helpers for enrollment and authentication events
"""
import logging
from django.utils import timezone
from .services import webhook_service

logger = logging.getLogger(__name__)


def send_enrollment_completed_webhook(client, client_user, session, enrollment):
    """Send webhook notification when enrollment is completed successfully"""
    try:
        event_data = {
            'user_id': client_user.external_user_id,
            'client_user_id': str(client_user.id),
            'enrollment_id': str(enrollment.id),
            'session_token': session.session_token,
            'enrollment_data': {
                'status': enrollment.status,
                'quality_score': float(enrollment.face_quality_score or 0.0),
                'liveness_score': float(enrollment.liveness_score or 0.0),
                'anti_spoofing_score': float(enrollment.anti_spoofing_score or 0.0),
                'sample_number': enrollment.sample_number,
                'total_samples': enrollment.total_samples,
                'face_image_url': enrollment.face_image_path,
                'completed_at': enrollment.updated_at.isoformat() if enrollment.updated_at else None,
            },
            'session_data': {
                'session_type': session.session_type,
                'started_at': session.created_at.isoformat(),
                'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                'frames_processed': session.metadata.get('frames_processed', 0),
                'liveness_blinks': session.metadata.get('liveness_blinks', 0),
                'liveness_motion_events': session.metadata.get('liveness_motion_events', 0),
            }
        }
        
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='enrollment.completed',
            event_data=event_data,
            source='enrollment_processor'
        )
    except Exception as e:
        logger.error(f"Failed to send enrollment completed webhook: {e}")
        return False


def send_authentication_success_webhook(client, client_user, session, similarity_score, auth_result):
    """Send webhook notification when authentication succeeds"""
    try:
        event_data = {
            'user_id': client_user.external_user_id,
            'client_user_id': str(client_user.id),
            'session_token': session.session_token,
            'authentication_data': {
                'similarity_score': similarity_score,
                'confidence': auth_result.get('confidence', 0.0),
                'quality_score': auth_result.get('quality_score', 0.0),
                'liveness_score': auth_result.get('liveness_data', {}).get('liveness_score', 0.0),
                'anti_spoofing_score': auth_result.get('anti_spoofing_score', 0.0),
                'authentication_level': auth_result.get('authentication_level', 'medium'),
                'liveness_verified': auth_result.get('liveness_verified', False),
            },
            'user_data': {
                'external_user_id': client_user.external_user_id,
                'is_enrolled': client_user.is_enrolled,
                'profile_image_url': client_user.profile_image.url if client_user.profile_image else None,
            },
            'session_data': {
                'session_type': session.session_type,
                'started_at': session.created_at.isoformat(),
                'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                'frames_processed': session.metadata.get('frames_processed', 0),
            }
        }
        
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='authentication.success',
            event_data=event_data,
            source='authentication_processor'
        )
    except Exception as e:
        logger.error(f"Failed to send authentication success webhook: {e}")
        return False


def send_authentication_failed_webhook(client, session, error_message, auth_result=None):
    """Send webhook notification when authentication fails"""
    try:
        event_data = {
            'session_token': session.session_token,
            'error': error_message,
            'authentication_data': {
                'similarity_score': auth_result.get('similarity_score', 0.0) if auth_result else 0.0,
                'quality_score': auth_result.get('quality_score', 0.0) if auth_result else 0.0,
                'liveness_score': auth_result.get('liveness_data', {}).get('liveness_score', 0.0) if auth_result else 0.0,
                'anti_spoofing_score': auth_result.get('anti_spoofing_score', 0.0) if auth_result else 0.0,
            },
            'session_data': {
                'session_type': session.session_type,
                'started_at': session.created_at.isoformat(),
                'frames_processed': session.metadata.get('frames_processed', 0),
            }
        }
        
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='authentication.failed',
            event_data=event_data,
            source='authentication_processor'
        )
    except Exception as e:
        logger.error(f"Failed to send authentication failed webhook: {e}")
        return False


def send_security_alert_webhook(client, session, alert_type, alert_data):
    """Send webhook notification for security alerts (anti-spoofing, obstacles, etc.)"""
    try:
        event_data = {
            'session_token': session.session_token,
            'alert_type': alert_type,
            'alert_data': alert_data,
            'session_data': {
                'session_type': session.session_type,
                'ip_address': session.ip_address,
                'user_agent': session.user_agent,
                'started_at': session.created_at.isoformat(),
            },
            'timestamp': timezone.now().isoformat(),
        }
        
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='security.alert',
            event_data=event_data,
            source='security_monitor'
        )
    except Exception as e:
        logger.error(f"Failed to send security alert webhook: {e}")
        return False
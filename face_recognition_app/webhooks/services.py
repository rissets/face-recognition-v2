"""
Webhook Service for Face Recognition Third-Party Service
"""
import json
import logging
import hashlib
import hmac
import requests
from datetime import datetime, timedelta
from django.conf import settings
from django.utils import timezone
from django.db import models
from celery import shared_task
from clients.models import Client, ClientWebhookLog
from webhooks.models import WebhookEndpoint, WebhookDelivery, WebhookEventLog
import time

logger = logging.getLogger(__name__)


class WebhookService:
    """
    Service for managing webhook deliveries
    """
    
    def __init__(self):
        self.config = getattr(settings, 'WEBHOOK_CONFIG', {})
        self.timeout = self.config.get('DEFAULT_TIMEOUT', 30)
        self.max_retries = self.config.get('MAX_RETRIES', 3)
        self.signature_header = self.config.get('SIGNATURE_HEADER', 'X-FR-Signature')
        self.timestamp_header = self.config.get('TIMESTAMP_HEADER', 'X-FR-Timestamp')
        self.event_header = self.config.get('EVENT_HEADER', 'X-FR-Event')
    
    def send_webhook(self, client_id, event_name, event_data, source='system', endpoint_ids=None):
        """
        Send webhook to all subscribed endpoints for a client
        """
        try:
            client = Client.objects.get(client_id=client_id, status='active')
        except Client.DoesNotExist:
            logger.error(f"Client {client_id} not found or inactive")
            return
        
        # Check if client has webhook events enabled
        if not client.is_feature_enabled('webhook_events'):
            logger.info(f"Webhook events not enabled for client {client_id}")
            return
        
        # Check if this event is in client's webhook events list
        webhook_events = client.get_webhook_events()
        if event_name not in webhook_events:
            logger.info(f"Event {event_name} not subscribed by client {client_id}")
            return
        
        # Get active webhook endpoints for this client
        endpoints = WebhookEndpoint.objects.filter(
            client=client,
            status='active',
            subscribed_events__contains=[event_name]
        )

        if endpoint_ids:
            endpoints = endpoints.filter(id__in=endpoint_ids)
        
        client_log = ClientWebhookLog.objects.create(
            client=client,
            event_type=event_name,
            payload=event_data,
            status='pending',
            attempt_count=0,
            max_attempts=self.max_retries,
        )

        if not endpoints.exists():
            logger.info(f"No active webhook endpoints for client {client_id} and event {event_name}")
            client_log.status = 'failed'
            client_log.error_message = 'No active webhook endpoints configured'
            client_log.delivered_at = timezone.now()
            client_log.save(update_fields=['status', 'error_message', 'delivered_at'])
            return
        
        # Create event log
        event_log = WebhookEventLog.objects.create(
            client=client,
            event_name=event_name,
            event_source=source,
            event_data=event_data,
            total_endpoints=endpoints.count()
        )
        
        # Send to each endpoint
        for endpoint in endpoints:
            self._schedule_webhook_delivery(endpoint, event_name, event_data, event_log)
            client_log.attempt_count += 1

        client_log.status = 'success'
        client_log.delivered_at = timezone.now()
        client_log.save(update_fields=['status', 'delivered_at', 'attempt_count'])

        logger.info(f"Scheduled webhook delivery for {endpoints.count()} endpoints")
    
    def _schedule_webhook_delivery(self, endpoint, event_name, event_data, event_log):
        """
        Schedule webhook delivery to specific endpoint
        """
        # Create delivery record
        delivery = WebhookDelivery.objects.create(
            endpoint=endpoint,
            event_name=event_name,
            event_data=event_data,
            max_attempts=self.max_retries
        )
        
        # Schedule async delivery
        deliver_webhook_async.delay(delivery.id, event_log.id)
    
    def deliver_webhook(self, delivery_id, event_log_id=None):
        """
        Actually deliver the webhook
        """
        try:
            delivery = WebhookDelivery.objects.get(id=delivery_id)
        except WebhookDelivery.DoesNotExist:
            logger.error(f"Webhook delivery {delivery_id} not found")
            return
        
        endpoint = delivery.endpoint
        client = endpoint.client
        
        # Prepare payload
        payload = {
            'event': delivery.event_name,
            'client_id': client.client_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': delivery.event_data,
            'delivery_id': str(delivery.id)
        }
        
        payload_json = json.dumps(payload, separators=(',', ':'))
        
        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'FaceRecognition-Webhook/1.0',
            self.event_header: delivery.event_name,
            self.timestamp_header: str(int(time.time()))
        }
        
        # Add signature if endpoint has secret
        if endpoint.secret_token:
            signature = self._generate_signature(
                payload_json, 
                endpoint.secret_token, 
                headers[self.timestamp_header]
            )
            headers[self.signature_header] = signature
        
        # Add custom headers from endpoint metadata
        custom_headers = endpoint.metadata.get('headers', {})
        headers.update(custom_headers)
        
        # Store headers in delivery
        delivery.headers = headers
        delivery.save(update_fields=['headers'])
        
        try:
            start_time = time.time()
            
            # Make HTTP request
            response = requests.post(
                endpoint.url,
                data=payload_json,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=False
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check if successful
            if 200 <= response.status_code < 300:
                # Success
                delivery.mark_as_success({
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                    'body': response.text[:1000],  # Limit body size
                    'response_time_ms': response_time_ms
                })
                
                # Update event log
                if event_log_id:
                    self._update_event_log_success(event_log_id)
                
                logger.info(f"Webhook delivered successfully to {endpoint.url}")
            else:
                # HTTP error
                error_data = {
                    'status_code': response.status_code,
                    'body': response.text[:1000],
                    'response_time_ms': response_time_ms,
                    'message': f'HTTP {response.status_code}: {response.reason}'
                }
                delivery.mark_as_failed(error_data)
                
                # Update event log
                if event_log_id:
                    self._update_event_log_failure(event_log_id)
                
                logger.error(f"Webhook failed with HTTP {response.status_code}: {endpoint.url}")
                
        except requests.exceptions.Timeout:
            error_data = {
                'message': 'Request timeout',
                'code': 'TIMEOUT'
            }
            delivery.mark_as_failed(error_data)
            
            if event_log_id:
                self._update_event_log_failure(event_log_id)
                
            logger.error(f"Webhook timeout: {endpoint.url}")
            
        except requests.exceptions.ConnectionError:
            error_data = {
                'message': 'Connection error',
                'code': 'CONNECTION_ERROR'
            }
            delivery.mark_as_failed(error_data)
            
            if event_log_id:
                self._update_event_log_failure(event_log_id)
                
            logger.error(f"Webhook connection error: {endpoint.url}")
            
        except Exception as e:
            error_data = {
                'message': str(e),
                'code': 'UNKNOWN_ERROR'
            }
            delivery.mark_as_failed(error_data)
            
            if event_log_id:
                self._update_event_log_failure(event_log_id)
                
            logger.error(f"Webhook delivery error: {e}")
    
    def _generate_signature(self, payload, secret, timestamp):
        """
        Generate webhook signature
        """
        sig_payload = f"{timestamp}.{payload}"
        signature = hmac.new(
            secret.encode('utf-8'),
            sig_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"t={timestamp},v1={signature}"
    
    def _update_event_log_success(self, event_log_id):
        """Update event log with successful delivery"""
        try:
            event_log = WebhookEventLog.objects.get(id=event_log_id)
            event_log.successful_deliveries += 1
            event_log.save(update_fields=['successful_deliveries'])
        except WebhookEventLog.DoesNotExist:
            pass
    
    def _update_event_log_failure(self, event_log_id):
        """Update event log with failed delivery"""
        try:
            event_log = WebhookEventLog.objects.get(id=event_log_id)
            event_log.failed_deliveries += 1
            event_log.save(update_fields=['failed_deliveries'])
        except WebhookEventLog.DoesNotExist:
            pass
    
    def retry_webhook(self, delivery: WebhookDelivery):
        """
        Manually requeue a webhook delivery.
        """
        if delivery.status not in ['failed', 'abandoned', 'retrying']:
            return delivery

        delivery.status = 'retrying'
        delivery.attempt_number = min(delivery.attempt_number + 1, delivery.max_attempts)
        delivery.next_retry_at = timezone.now()
        delivery.save(update_fields=['status', 'attempt_number', 'next_retry_at'])

        deliver_webhook_async.delay(delivery.id)
        return delivery

    def retry_failed_webhooks(self):
        """
        Retry failed webhooks that are due for retry
        """
        now = timezone.now()
        
        # Get deliveries that should be retried
        deliveries_to_retry = WebhookDelivery.objects.filter(
            status__in=['failed', 'retrying'],
            next_retry_at__lte=now,
            attempt_number__lt=models.F('max_attempts')
        )
        
        for delivery in deliveries_to_retry:
            logger.info(f"Retrying webhook delivery {delivery.id}")
            self.retry_webhook(delivery)

    @staticmethod
    def send_processing_error(client, session, error_message):
        """Send processing error webhook"""
        event_data = {
            'session_id': session.session_token,
            'session_type': session.session_type,
            'error_message': error_message,
            'timestamp': timezone.now().isoformat(),
            'client_user': session.client_user.external_user_id if session.client_user else None
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='processing.error',
            event_data=event_data,
            source='face_processor'
        )
    
    @staticmethod
    def send_authentication_started(client, session):
        """Send authentication session started webhook"""
        event_data = {
            'session_id': session.session_token,
            'session_type': session.session_type,
            'client_user': session.client_user.external_user_id if session.client_user else None,
            'expires_at': session.expires_at.isoformat(),
            'created_at': session.created_at.isoformat(),
            'ip_address': session.ip_address,
            'user_agent': session.user_agent
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='authentication.session.started',
            event_data=event_data,
            source='authentication_service'
        )
    
    @staticmethod
    def send_enrollment_completed(client, enrollment):
        """Send enrollment completed webhook"""
        event_data = {
            'enrollment_id': enrollment.id,
            'client_user': enrollment.client_user.external_user_id,
            'quality_score': enrollment.face_quality_score,
            'liveness_score': enrollment.liveness_score,
            'total_samples': enrollment.total_samples,
            'completed_at': timezone.now().isoformat()
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='enrollment.completed',
            event_data=event_data,
            source='enrollment_service'
        )
    
    @staticmethod
    def send_enrollment_failed(client, enrollment, reason):
        """Send enrollment failed webhook"""
        event_data = {
            'enrollment_id': enrollment.id if enrollment else None,
            'client_user': enrollment.client_user.external_user_id if enrollment and enrollment.client_user else None,
            'failure_reason': reason,
            'failed_at': timezone.now().isoformat()
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='enrollment.failed',
            event_data=event_data,
            source='enrollment_service'
        )
    
    @staticmethod
    def send_authentication_success(client, attempt):
        """Send authentication success webhook"""
        event_data = {
            'attempt_id': attempt.id,
            'client_user': attempt.matched_user.external_user_id if attempt.matched_user else None,
            'similarity_score': attempt.similarity_score,
            'confidence_score': attempt.confidence_score,
            'processing_time_ms': attempt.processing_time,
            'authenticated_at': timezone.now().isoformat()
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='authentication.success',
            event_data=event_data,
            source='authentication_service'
        )
    
    @staticmethod
    def send_authentication_failed(client, attempt):
        """Send authentication failed webhook"""
        event_data = {
            'attempt_id': attempt.id,
            'failure_reason': attempt.result,
            'similarity_score': attempt.similarity_score,
            'processing_time_ms': attempt.processing_time,
            'failed_at': timezone.now().isoformat()
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='authentication.failed',
            event_data=event_data,
            source='authentication_service'
        )


# Celery tasks
@shared_task(bind=True, max_retries=3)
def deliver_webhook_async(self, delivery_id, event_log_id=None):
    """
    Async task to deliver webhook
    """
    try:
        webhook_service = WebhookService()
        webhook_service.deliver_webhook(delivery_id, event_log_id)
    except Exception as e:
        logger.error(f"Failed to deliver webhook {delivery_id}: {e}")
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@shared_task
def retry_failed_webhooks():
    """
    Periodic task to retry failed webhooks
    """
    webhook_service = WebhookService()
    webhook_service.retry_failed_webhooks()


@shared_task
def cleanup_old_webhook_logs():
    """
    Cleanup old webhook logs and deliveries
    """
    cutoff_date = timezone.now() - timedelta(days=30)
    
    # Delete old deliveries
    deleted_deliveries = WebhookDelivery.objects.filter(
        created_at__lt=cutoff_date,
        status__in=['success', 'abandoned']
    ).delete()
    
    # Delete old event logs
    deleted_logs = WebhookEventLog.objects.filter(
        created_at__lt=cutoff_date
    ).delete()
    
    logger.info(f"Cleaned up {deleted_deliveries[0]} deliveries and {deleted_logs[0]} event logs")


# Convenience function for sending webhooks
webhook_service = WebhookService()

def send_webhook(client_id, event_name, event_data, source='system'):
    """
    Convenience function to send webhook
    """
    return webhook_service.send_webhook(client_id, event_name, event_data, source)


class WebhookStaticMethods:
    """Static methods for webhook operations"""
    
    @staticmethod
    def send_processing_error(client, session, error_message):
        """Send processing error webhook"""
        event_data = {
            'session_id': session.session_token,
            'session_type': session.session_type,
            'error_message': error_message,
            'timestamp': timezone.now().isoformat(),
            'client_user': session.client_user.external_user_id if session.client_user else None
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='processing.error',
            event_data=event_data,
            source='face_processor'
        )
    
    @staticmethod
    def send_authentication_started(client, session):
        """Send authentication session started webhook"""
        event_data = {
            'session_id': session.session_token,
            'session_type': session.session_type,
            'client_user': session.client_user.external_user_id if session.client_user else None,
            'expires_at': session.expires_at.isoformat(),
            'created_at': session.created_at.isoformat(),
            'ip_address': session.ip_address,
            'user_agent': session.user_agent
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='authentication.session.started',
            event_data=event_data,
            source='authentication_service'
        )
    
    @staticmethod
    def send_enrollment_completed(client, enrollment):
        """Send enrollment completed webhook"""
        event_data = {
            'enrollment_id': str(enrollment.id),
            'user_id': enrollment.client_user.external_user_id,
            'session_id': enrollment.enrollment_session.session_token,
            'status': enrollment.status,
            'quality_score': enrollment.face_quality_score,
            'liveness_score': enrollment.liveness_score,
            'timestamp': timezone.now().isoformat()
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='enrollment.completed',
            event_data=event_data,
            source='enrollment_processor'
        )
    
    @staticmethod
    def send_enrollment_failed(client, enrollment, reason):
        """Send enrollment failed webhook"""
        event_data = {
            'enrollment_id': str(enrollment.id) if enrollment.id else None,
            'user_id': enrollment.client_user.external_user_id,
            'session_id': enrollment.enrollment_session.session_token,
            'failure_reason': reason,
            'quality_score': getattr(enrollment, 'face_quality_score', 0.0),
            'timestamp': timezone.now().isoformat()
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='enrollment.failed',
            event_data=event_data,
            source='enrollment_processor'
        )
    
    @staticmethod
    def send_authentication_success(client, attempt):
        """Send authentication success webhook"""
        event_data = {
            'attempt_id': str(attempt.id),
            'user_id': attempt.matched_user.external_user_id,
            'session_id': attempt.session.session_token,
            'confidence_score': attempt.confidence_score,
            'similarity_score': getattr(attempt, 'similarity_score', 0.0),
            'processing_time_ms': getattr(attempt, 'processing_time_ms', 0.0),
            'timestamp': timezone.now().isoformat()
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='authentication.success',
            event_data=event_data,
            source='authentication_processor'
        )
    
    @staticmethod
    def send_authentication_failed(client, attempt):
        """Send authentication failed webhook"""
        event_data = {
            'attempt_id': str(attempt.id),
            'session_id': attempt.session.session_token,
            'result': attempt.result,
            'confidence_score': attempt.confidence_score,
            'processing_time_ms': getattr(attempt, 'processing_time_ms', 0.0),
            'timestamp': timezone.now().isoformat()
        }
        
        webhook_service = WebhookService()
        return webhook_service.send_webhook(
            client_id=client.client_id,
            event_name='authentication.failed',
            event_data=event_data,
            source='authentication_processor'
        )


# Convenience function for sending webhooks
webhook_service = WebhookService()

def send_webhook(client_id, event_name, event_data, source='system'):
    """
    Convenience function to send webhook
    """
    return webhook_service.send_webhook(client_id, event_name, event_data, source)

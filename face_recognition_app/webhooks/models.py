"""
Webhook System Models for Face Recognition Third-Party Service
"""
import uuid
import hashlib
import hmac
from django.db import models
from django.contrib.postgres.fields import ArrayField
import json


class WebhookEvent(models.Model):
    """
    Define webhook event types and their configurations
    """
    
    EVENT_STATUS_CHOICES = [
        ('active', 'Active'),
        ('inactive', 'Inactive'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Event details
    event_name = models.CharField(max_length=100, unique=True, db_index=True)
    description = models.TextField()
    
    # Event configuration
    is_active = models.BooleanField(default=True)
    payload_schema = models.JSONField(
        default=dict,
        help_text="JSON schema for event payload validation"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Webhook Event"
        verbose_name_plural = "Webhook Events"
        ordering = ['event_name']
    
    def __str__(self):
        return self.event_name


class WebhookEndpoint(models.Model):
    """
    Client webhook endpoints configuration
    """
    
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('failed', 'Failed'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client = models.ForeignKey('clients.Client', on_delete=models.CASCADE, related_name='webhook_endpoints')
    
    # Endpoint details
    name = models.CharField(max_length=100)
    url = models.URLField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    
    # Event subscriptions
    subscribed_events = ArrayField(
        models.CharField(max_length=100),
        default=list,
        help_text="List of event names this endpoint subscribes to"
    )
    
    # Security
    secret_token = models.CharField(max_length=100, blank=True)
    
    # Retry configuration
    max_retries = models.PositiveIntegerField(default=3)
    retry_delay_seconds = models.PositiveIntegerField(default=60)
    
    # Statistics
    total_deliveries = models.PositiveIntegerField(default=0)
    successful_deliveries = models.PositiveIntegerField(default=0)
    failed_deliveries = models.PositiveIntegerField(default=0)
    
    # Metadata
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_delivery_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Webhook Endpoint"
        verbose_name_plural = "Webhook Endpoints"
        unique_together = [['client', 'name']]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.client.name} - {self.name}"
    
    @property
    def success_rate(self):
        """Calculate delivery success rate"""
        if self.total_deliveries == 0:
            return 0
        return (self.successful_deliveries / self.total_deliveries) * 100
    
    def is_subscribed_to_event(self, event_name):
        """Check if endpoint is subscribed to an event"""
        return event_name in self.subscribed_events
    
    def generate_signature(self, payload_json, timestamp=None):
        """Generate webhook signature for payload verification"""
        if not self.secret_token:
            return None
        
        if timestamp is None:
            from django.utils import timezone
            timestamp = str(int(timezone.now().timestamp()))
        
        # Create signature payload
        sig_payload = f"{timestamp}.{payload_json}"
        
        # Generate signature
        signature = hmac.new(
            self.secret_token.encode('utf-8'),
            sig_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"t={timestamp},v1={signature}"


class WebhookDelivery(models.Model):
    """
    Track webhook delivery attempts and results
    """
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('success', 'Success'),
        ('failed', 'Failed'),
        ('retrying', 'Retrying'),
        ('abandoned', 'Abandoned'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    endpoint = models.ForeignKey(WebhookEndpoint, on_delete=models.CASCADE, related_name='deliveries')
    
    # Event details
    event_name = models.CharField(max_length=100, db_index=True)
    event_data = models.JSONField()
    
    # Delivery details
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # HTTP details
    http_method = models.CharField(max_length=10, default='POST')
    headers = models.JSONField(default=dict)
    
    # Response details
    response_status_code = models.IntegerField(null=True, blank=True)
    response_headers = models.JSONField(default=dict)
    response_body = models.TextField(blank=True)
    
    # Performance metrics
    response_time_ms = models.FloatField(null=True, blank=True)
    
    # Retry mechanism
    attempt_number = models.PositiveIntegerField(default=1)
    max_attempts = models.PositiveIntegerField(default=3)
    next_retry_at = models.DateTimeField(null=True, blank=True)
    
    # Error tracking
    error_message = models.TextField(blank=True)
    error_code = models.CharField(max_length=50, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Webhook Delivery"
        verbose_name_plural = "Webhook Deliveries"
        indexes = [
            models.Index(fields=['endpoint', 'event_name']),
            models.Index(fields=['status', 'next_retry_at']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.endpoint.name} - {self.event_name} - {self.status}"
    
    @property
    def should_retry(self):
        """Check if delivery should be retried"""
        return (
            self.status in ['failed', 'retrying'] and 
            self.attempt_number < self.max_attempts
        )
    
    @property
    def is_final_state(self):
        """Check if delivery is in final state (no more retries)"""
        return self.status in ['success', 'abandoned']
    
    def mark_as_success(self, response_data=None):
        """Mark delivery as successful"""
        from django.utils import timezone
        
        self.status = 'success'
        self.delivered_at = timezone.now()
        self.completed_at = timezone.now()
        
        if response_data:
            self.response_status_code = response_data.get('status_code')
            self.response_headers = response_data.get('headers', {})
            self.response_body = response_data.get('body', '')
            self.response_time_ms = response_data.get('response_time_ms')
        
        self.save()
        
        # Update endpoint statistics
        self.endpoint.successful_deliveries += 1
        self.endpoint.total_deliveries += 1
        self.endpoint.last_delivery_at = timezone.now()
        self.endpoint.save(update_fields=[
            'successful_deliveries', 'total_deliveries', 'last_delivery_at'
        ])
    
    def mark_as_failed(self, error_data=None):
        """Mark delivery as failed"""
        from django.utils import timezone
        from datetime import timedelta
        
        self.status = 'failed'
        
        if error_data:
            self.error_message = error_data.get('message', '')
            self.error_code = error_data.get('code', '')
            self.response_status_code = error_data.get('status_code')
            self.response_body = error_data.get('body', '')
            self.response_time_ms = error_data.get('response_time_ms')
        
        # Check if should retry
        if self.should_retry:
            self.status = 'retrying'
            self.attempt_number += 1
            # Exponential backoff: 60s, 120s, 300s
            delay_seconds = min(60 * (2 ** (self.attempt_number - 2)), 300)
            self.next_retry_at = timezone.now() + timedelta(seconds=delay_seconds)
        else:
            self.status = 'abandoned'
            self.completed_at = timezone.now()
            
            # Update endpoint statistics
            self.endpoint.failed_deliveries += 1
            self.endpoint.total_deliveries += 1
            self.endpoint.save(update_fields=['failed_deliveries', 'total_deliveries'])
        
        self.save()


class WebhookEventLog(models.Model):
    """
    Comprehensive log of all webhook events for debugging and analytics
    """
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client = models.ForeignKey('clients.Client', on_delete=models.CASCADE, related_name='webhook_event_logs')
    
    # Event details
    event_name = models.CharField(max_length=100, db_index=True)
    event_source = models.CharField(max_length=100)  # enrollment, recognition, etc.
    event_data = models.JSONField()
    
    # Processing details
    total_endpoints = models.PositiveIntegerField(default=0)
    successful_deliveries = models.PositiveIntegerField(default=0)
    failed_deliveries = models.PositiveIntegerField(default=0)
    
    # Metadata
    metadata = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        verbose_name = "Webhook Event Log"
        verbose_name_plural = "Webhook Event Logs"
        indexes = [
            models.Index(fields=['client', 'event_name']),
            models.Index(fields=['event_source', 'created_at']),
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.client.client_id} - {self.event_name}"
    
    @property
    def success_rate(self):
        """Calculate event delivery success rate"""
        if self.total_endpoints == 0:
            return 100  # No endpoints to deliver to
        return (self.successful_deliveries / self.total_endpoints) * 100

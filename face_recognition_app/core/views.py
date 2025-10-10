"""
Core API views for third-party authentication service
Redesigned for multi-client architecture
"""
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.utils import timezone
from django.db.models import Count, Q
from datetime import timedelta
import platform
import time
import psutil
from django.conf import settings

from .models import SystemConfiguration, AuditLog, SecurityEvent, HealthCheck
from .serializers import (
    SystemConfigurationSerializer,
    AuditLogSerializer,
    SecurityEventSerializer,
    HealthCheckSerializer,
    SystemStatusSerializer,
    ClientTokenSerializer,
    ClientUserAuthSerializer,
)
from .services import ThirdPartyIntegrationService
from auth_service.authentication import APIKeyAuthentication
from auth_service.models import AuthenticationSession, FaceEnrollment, FaceRecognitionAttempt
from clients.models import Client, ClientUser


class SystemConfigurationViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for system configuration (admin only)
    """
    queryset = SystemConfiguration.objects.filter(is_active=True)
    serializer_class = SystemConfigurationSerializer
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Only allow access to admin clients"""
        if hasattr(self.request, 'client') and self.request.client.tier == 'enterprise':
            return self.queryset
        return self.queryset.none()


class AuditLogViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for audit logs (filtered by client)
    """
    queryset = AuditLog.objects.all()
    serializer_class = AuditLogSerializer
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter audit logs by client"""
        if hasattr(self.request, 'client'):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()


class SecurityEventViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for security events (filtered by client)
    """
    queryset = SecurityEvent.objects.all()
    serializer_class = SecurityEventSerializer
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Filter security events by client"""
        if hasattr(self.request, 'client'):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()


@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def authenticate_client(request):
    """
    Authenticate client using API key and secret
    Returns JWT token for subsequent requests
    """
    serializer = ClientTokenSerializer(data=request.data)
    
    if serializer.is_valid():
        client = serializer.validated_data['client']
        integration_service = ThirdPartyIntegrationService(client)
        
        # Log authentication attempt
        AuditLog.objects.create(
            client=client,
            action='client_authentication',
            resource_type='client',
            resource_id=str(client.id),
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            success=True,
        )
        
        auth_payload = integration_service.build_auth_response()
        return Response(auth_payload)
    
    # Log failed authentication
    AuditLog.objects.create(
        action='client_authentication_failed',
        resource_type='client',
        ip_address=request.META.get('REMOTE_ADDR'),
        user_agent=request.META.get('HTTP_USER_AGENT', ''),
        success=False,
        error_message='Invalid credentials'
    )
    
    return Response(serializer.errors, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['POST'])
def authenticate_client_user(request):
    """
    Authenticate client user
    """
    serializer = ClientUserAuthSerializer(data=request.data)
    
    if serializer.is_valid():
        client = serializer.validated_data['client']
        user = serializer.validated_data['client_user']
        
        # Update last login
        user.last_login_at = timezone.now()
        user.save()
        
        # Log user authentication
        AuditLog.objects.create(
            client=client,
            client_user=user,
            action='user_authentication',
            resource_type='client_user',
            resource_id=str(user.id),
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            success=True
        )
        
        return Response({
            'success': True,
            'client_id': client.client_id,
            'user_id': user.external_user_id,
            'display_name': user.display_name,
            'profile': user.profile,
            'is_enrolled': user.is_enrolled,
            'face_auth_enabled': user.face_auth_enabled,
            'metadata': user.metadata,
            'last_recognition_at': user.last_recognition_at
        })
    
    return Response(serializer.errors, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['GET'])
def system_status(request):
    """
    Get comprehensive system status
    """
    try:
        # Calculate uptime
        uptime_seconds = time.time() - psutil.boot_time()
        uptime_str = str(timedelta(seconds=int(uptime_seconds)))
        
        # Check database
        try:
            Client.objects.count()
            db_status = 'healthy'
        except Exception:
            db_status = 'error'
        
        # Check Redis (if configured)
        redis_status = 'healthy'  # Implement Redis check
        
        # Check Celery (if configured)
        celery_status = 'healthy'  # Implement Celery check
        
        # Face processing status
        face_processing_status = 'healthy'  # Implement face processing check
        
        # Calculate statistics
        stats = {
            'total_clients': Client.objects.filter(status='active').count(),
            'active_sessions': AuthenticationSession.objects.filter(status='active').count(),
            'total_enrollments': FaceEnrollment.objects.count(),
            'total_authentications': FaceRecognitionAttempt.objects.count(),
        }
        
        status_data = {
            'status': 'healthy' if all([
                db_status == 'healthy',
                redis_status == 'healthy',
                celery_status == 'healthy',
                face_processing_status == 'healthy'
            ]) else 'degraded',
            'uptime': uptime_str,
            'version': getattr(settings, 'VERSION', '1.0.0'),
            'database_status': db_status,
            'redis_status': redis_status,
            'celery_status': celery_status,
            'face_processing_status': face_processing_status,
            **stats
        }
        
        serializer = SystemStatusSerializer(status_data)
        return Response(serializer.data)
        
    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def health_check(request):
    """
    Simple health check endpoint for load balancers
    """
    try:
        # Basic database check
        Client.objects.count()
        
        return Response({
            'status': 'healthy',
            'timestamp': timezone.now(),
            'service': 'face-auth-api'
        })
    except Exception as e:
        return Response({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': timezone.now(),
            'service': 'face-auth-api'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(['POST'])
def log_security_event(request):
    """
    Log security events for monitoring
    """
    if not hasattr(request, 'client'):
        return Response({'error': 'Authentication required'}, 
                       status=status.HTTP_401_UNAUTHORIZED)
    
    event_type = request.data.get('event_type')
    severity = request.data.get('severity', 'medium')
    details = request.data.get('details', {})
    
    if not event_type:
        return Response({'error': 'event_type is required'}, 
                       status=status.HTTP_400_BAD_REQUEST)
    
    SecurityEvent.objects.create(
        client=request.client,
        event_type=event_type,
        severity=severity,
        ip_address=request.META.get('REMOTE_ADDR'),
        user_agent=request.META.get('HTTP_USER_AGENT', ''),
        details=details
    )
    
    return Response({'success': True, 'message': 'Security event logged'})


@api_view(['GET'])
def client_info(request):
    """
    Get current client information
    """
    if not hasattr(request, 'client'):
        return Response({'error': 'Authentication required'}, 
                       status=status.HTTP_401_UNAUTHORIZED)
    
    client = request.client
    
    # Calculate usage statistics
    today = timezone.now().date()
    usage_today = client.api_usage.filter(date=today).aggregate(
        total_requests=Count('total_requests')
    )
    
    return Response({
        'client_id': client.api_key,
        'name': client.name,
        'feature_tier': client.feature_tier,
        'rate_limits': {
            'per_minute': client.rate_limit_per_minute,
            'per_day': client.rate_limit_per_day
        },
        'usage_today': usage_today.get('total_requests', 0),
        'total_users': client.users.count(),
        'active_users': client.users.filter(is_active=True).count(),
        'webhook_url': client.webhook_url,
        'webhook_events': client.webhook_events,
        'permissions': client.permissions,
        'expires_at': client.expires_at
    })

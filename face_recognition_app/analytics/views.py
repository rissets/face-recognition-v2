"""
Analytics app views with comprehensive Swagger documentation
"""
from rest_framework import generics, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Count, Avg, Q, Min, Max
from django.utils import timezone
from datetime import timedelta
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse
from drf_spectacular.types import OpenApiTypes
from rest_framework import serializers
from .models import AuthenticationLog, SecurityAlert, SystemMetrics
from .serializers import (
    AuthenticationLogSerializer, 
    SecurityAlertSerializer,
    SystemMetricsSerializer,
    DashboardResponseSerializer,
    StatisticsResponseSerializer
)


@extend_schema(
    tags=['Analytics'],
    summary='List Authentication Logs',
    description='Retrieve a paginated list of authentication logs for the authenticated user. Includes filtering capabilities.',
    parameters=[
        OpenApiParameter(
            name='success',
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            description='Filter by success status (true/false)'
        ),
        OpenApiParameter(
            name='auth_method',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description='Filter by authentication method',
            enum=['face', 'password', '2fa', 'social']
        ),
        OpenApiParameter(
            name='date_from',
            type=OpenApiTypes.DATE,
            location=OpenApiParameter.QUERY,
            description='Filter logs from this date (YYYY-MM-DD)'
        ),
        OpenApiParameter(
            name='date_to',
            type=OpenApiTypes.DATE,
            location=OpenApiParameter.QUERY,
            description='Filter logs to this date (YYYY-MM-DD)'
        ),
    ],
    responses={
        200: OpenApiResponse(
            response=AuthenticationLogSerializer(many=True),
            description='List of authentication logs'
        ),
        401: OpenApiResponse(description='Authentication required'),
        403: OpenApiResponse(description='Permission denied'),
    }
)
class AuthenticationLogListView(generics.ListAPIView):
    """List authentication logs with filtering capabilities"""
    serializer_class = AuthenticationLogSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = AuthenticationLog.objects.filter(user=self.request.user)
        
        # Apply filters
        success = self.request.query_params.get('success')
        if success is not None:
            queryset = queryset.filter(success=success.lower() == 'true')
            
        auth_method = self.request.query_params.get('auth_method')
        if auth_method:
            queryset = queryset.filter(auth_method=auth_method)
            
        date_from = self.request.query_params.get('date_from')
        if date_from:
            queryset = queryset.filter(timestamp__date__gte=date_from)
            
        date_to = self.request.query_params.get('date_to')
        if date_to:
            queryset = queryset.filter(timestamp__date__lte=date_to)
            
        return queryset.order_by('-timestamp')


@extend_schema(
    tags=['Analytics'],
    summary='Get Authentication Log Details',
    description='Retrieve detailed information about a specific authentication log entry.',
    responses={
        200: OpenApiResponse(
            response=AuthenticationLogSerializer,
            description='Authentication log details'
        ),
        401: OpenApiResponse(description='Authentication required'),
        403: OpenApiResponse(description='Permission denied'),
        404: OpenApiResponse(description='Authentication log not found'),
    }
)
class AuthenticationLogDetailView(generics.RetrieveAPIView):
    """Retrieve detailed authentication log information"""
    serializer_class = AuthenticationLogSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return AuthenticationLog.objects.filter(user=self.request.user)


@extend_schema(
    tags=['Analytics'],
    summary='List Security Alerts',
    description='Retrieve a paginated list of security alerts for the authenticated user with filtering capabilities.',
    parameters=[
        OpenApiParameter(
            name='severity',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description='Filter by alert severity',
            enum=['low', 'medium', 'high', 'critical']
        ),
        OpenApiParameter(
            name='alert_type',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description='Filter by alert type',
            enum=['suspicious_login', 'face_mismatch', 'unusual_activity', 'brute_force', 'device_change']
        ),
        OpenApiParameter(
            name='resolved',
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            description='Filter by resolution status'
        ),
    ],
    responses={
        200: OpenApiResponse(
            response=SecurityAlertSerializer(many=True),
            description='List of security alerts',
        ),
        401: OpenApiResponse(description='Authentication required'),
        403: OpenApiResponse(description='Permission denied'),
    }
)
class SecurityAlertListView(generics.ListAPIView):
    """List security alerts with filtering capabilities"""
    serializer_class = SecurityAlertSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = SecurityAlert.objects.filter(user=self.request.user)
        
        # Apply filters
        severity = self.request.query_params.get('severity')
        if severity:
            queryset = queryset.filter(severity=severity)
            
        alert_type = self.request.query_params.get('alert_type')
        if alert_type:
            queryset = queryset.filter(alert_type=alert_type)
            
        resolved = self.request.query_params.get('resolved')
        if resolved is not None:
            queryset = queryset.filter(resolved=resolved.lower() == 'true')
            
        return queryset.order_by('-created_at')


@extend_schema(
    tags=['Analytics'],
    summary='Get Security Alert Details',
    description='Retrieve detailed information about a specific security alert.',
    responses={
        200: OpenApiResponse(
            response=SecurityAlertSerializer,
            description='Security alert details',
        ),
        401: OpenApiResponse(description='Authentication required'),
        403: OpenApiResponse(description='Permission denied'),
        404: OpenApiResponse(description='Security alert not found'),
    }
)
class SecurityAlertDetailView(generics.RetrieveAPIView):
    """Retrieve detailed security alert information"""
    serializer_class = SecurityAlertSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return SecurityAlert.objects.filter(user=self.request.user)


@extend_schema(
    tags=['Analytics'],
    summary='Analytics Dashboard',
    description='Get comprehensive analytics dashboard data including authentication statistics, security alerts, and performance metrics for the last 30 days.',
    parameters=[
        OpenApiParameter(
            name='period',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            description='Time period for analytics',
            enum=['7_days', '30_days', '90_days', '1_year'],
            default='30_days'
        ),
    ],
    responses={
        200: OpenApiResponse(
            response=DashboardResponseSerializer,
            description='Dashboard analytics data'
        ),
        401: OpenApiResponse(description='Authentication required'),
        403: OpenApiResponse(description='Permission denied'),
    }
)
class AnalyticsDashboardView(APIView):
    """Comprehensive analytics dashboard with authentication stats, alerts, and performance metrics"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        user = request.user
        
        # Get period from query params
        period = request.query_params.get('period', '30_days')
        
        # Calculate date range
        period_mapping = {
            '7_days': 7,
            '30_days': 30,
            '90_days': 90,
            '1_year': 365
        }
        
        days = period_mapping.get(period, 30)
        start_date = timezone.now() - timedelta(days=days)
        
        # Authentication stats
        auth_logs = AuthenticationLog.objects.filter(
            user=user,
            timestamp__gte=start_date
        )
        
        auth_stats = auth_logs.aggregate(
            total_attempts=Count('id'),
            successful_attempts=Count('id', filter=Q(success=True)),
            avg_similarity=Avg('similarity_score'),
            avg_liveness=Avg('liveness_score'),
            avg_quality=Avg('quality_score')
        )
        
        # Calculate success rate
        total = auth_stats['total_attempts'] or 0
        successful = auth_stats['successful_attempts'] or 0
        auth_stats['success_rate'] = (successful / total * 100) if total > 0 else 0
        
        # Security alerts
        alert_count = SecurityAlert.objects.filter(
            user=user,
            created_at__gte=start_date
        ).count()
        
        # Recent activities (last 10)
        recent_activities = auth_logs.order_by('-timestamp')[:10].values(
            'timestamp', 'success', 'auth_method', 'failure_reason'
        )
        
        # Performance metrics
        performance_metrics = auth_logs.aggregate(
            avg_response_time=Avg('response_time'),
            min_response_time=Min('response_time'),
            max_response_time=Max('response_time')
        )
        
        # Risk analysis
        risk_analysis = auth_logs.aggregate(
            avg_risk_score=Avg('risk_score'),
            high_risk_attempts=Count('id', filter=Q(risk_score__gte=0.7))
        )
        
        return Response({
            'auth_stats': auth_stats,
            'alert_count': alert_count,
            'recent_activities': list(recent_activities),
            'performance_metrics': performance_metrics,
            'risk_analysis': risk_analysis,
            'period': period,
            'date_range': {
                'from': start_date.date(),
                'to': timezone.now().date()
            }
        })


@extend_schema(
    tags=['Analytics'],
    summary='User Statistics',
    description='Get comprehensive user statistics including authentication attempts, success rates, and user profile information.',
    responses={
        200: OpenApiResponse(
            response=StatisticsResponseSerializer,
            description='User statistics'
        ),
        401: OpenApiResponse(description='Authentication required'),
        403: OpenApiResponse(description='Permission denied'),
    }
)
class StatisticsView(APIView):
    """Comprehensive user statistics and profile information"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        user = request.user
        
        # Basic user stats
        all_logs = AuthenticationLog.objects.filter(user=user)
        total_auth_attempts = all_logs.count()
        successful_auths = all_logs.filter(success=True).count()
        failed_attempts = total_auth_attempts - successful_auths
        
        success_rate = (successful_auths / total_auth_attempts * 100) if total_auth_attempts > 0 else 0
        
        # Average scores
        avg_scores = all_logs.aggregate(
            similarity=Avg('similarity_score'),
            liveness=Avg('liveness_score'),
            quality=Avg('quality_score')
        )
        
        # Authentication methods used
        auth_methods = list(all_logs.values_list('auth_method', flat=True).distinct())
        
        # Security status
        recent_alerts = SecurityAlert.objects.filter(
            user=user,
            created_at__gte=timezone.now() - timedelta(days=30)
        ).count()
        
        # Calculate risk level
        recent_risk_score = all_logs.filter(
            timestamp__gte=timezone.now() - timedelta(days=7)
        ).aggregate(avg_risk=Avg('risk_score'))['avg_risk'] or 0
        
        risk_level = 'low'
        if recent_risk_score > 0.7:
            risk_level = 'high'
        elif recent_risk_score > 0.4:
            risk_level = 'medium'
        
        return Response({
            'total_attempts': total_auth_attempts,
            'successful_attempts': successful_auths,
            'failed_attempts': failed_attempts,
            'success_rate': round(success_rate, 2),
            'face_enrolled': getattr(user, 'face_enrolled', False),
            'last_login': user.last_login,
            'account_created': user.date_joined,
            'authentication_methods': auth_methods,
            'avg_scores': {
                'similarity': round(avg_scores['similarity'] or 0, 3),
                'liveness': round(avg_scores['liveness'] or 0, 3),
                'quality': round(avg_scores['quality'] or 0, 3)
            },
            'security_status': {
                'active_alerts': recent_alerts,
                'risk_level': risk_level,
                'last_security_scan': timezone.now()
            }
        })

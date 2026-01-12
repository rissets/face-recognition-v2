"""
Analytics app views with comprehensive Swagger documentation
"""
from rest_framework import generics, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from auth_service.authentication import APIKeyAuthentication, JWTClientAuthentication
from django.db.models import Count, Avg, Q, Min, Max
from django.utils import timezone
from datetime import timedelta, datetime
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse
from drf_spectacular.types import OpenApiTypes
from rest_framework import serializers
from .models import (
    AuthenticationLog,
    SecurityAlert,
    SystemMetrics,
    ModelPerformance,
    DataQualityMetrics,
)
from .serializers import (
    AuthenticationLogSerializer, 
    SecurityAlertSerializer,
    SystemMetricsSerializer,
    DashboardResponseSerializer,
    StatisticsResponseSerializer,
    ModelPerformanceSerializer,
    DataQualityMetricsSerializer,
)
from auth_service.models import FaceRecognitionAttempt, FaceEnrollment
from clients.models import ClientUser


def _parse_iso8601(value):
    """Parse ISO 8601 datetime strings safely."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if timezone.is_naive(parsed):
            parsed = timezone.make_aware(parsed)
        return parsed
    except ValueError:
        return None


def _build_behavior_insights(client):
    """Derive behaviour analytics for client users."""
    insights = []
    for user in ClientUser.objects.filter(client=client):
        attempts = FaceRecognitionAttempt.objects.filter(client=client, matched_user=user)
        total_attempts = attempts.count()
        successful_attempts = attempts.filter(result='success').count()
        success_rate = (successful_attempts / total_attempts * 100) if total_attempts else 0.0

        averages = attempts.aggregate(
            avg_similarity=Avg('similarity_score'),
            avg_liveness=Avg('liveness_score'),
            avg_quality=Avg('face_quality_score'),
        )

        last_attempt = attempts.order_by('-created_at').first()

        if success_rate >= 85:
            risk_level = 'low'
        elif success_rate >= 60:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        insights.append({
            'client_user_id': str(user.id),
            'external_user_id': user.external_user_id,
            'display_name': user.display_name,
            'auth_success_rate': round(success_rate, 2),
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'avg_similarity_score': round(averages['avg_similarity'] or 0, 3),
            'avg_liveness_score': round(averages['avg_liveness'] or 0, 3),
            'avg_quality_score': round(averages['avg_quality'] or 0, 3),
            'last_activity': last_attempt.created_at if last_attempt else None,
            'risk_level': risk_level,
        })
    return insights


def _build_recognition_stats(client, date_from=None, date_to=None):
    """Aggregate recognition attempts into daily statistics."""
    attempts = FaceRecognitionAttempt.objects.filter(client=client)
    if date_from:
        attempts = attempts.filter(created_at__date__gte=date_from)
    if date_to:
        attempts = attempts.filter(created_at__date__lte=date_to)

    stats_map = {}

    for attempt in attempts:
        date_key = attempt.created_at.date()
        summary = stats_map.setdefault(
            date_key,
            {
                'date': date_key,
                'total_attempts': 0,
                'successful_attempts': 0,
                'failed_attempts': 0,
                'failed_similarity': 0,
                'failed_liveness': 0,
                'failed_quality': 0,
                'failed_obstacles': 0,
                'failed_no_face': 0,
                'failed_multiple_faces': 0,
                'failed_system_error': 0,
                'sum_response_time': 0.0,
                'sum_similarity': 0.0,
                'sum_liveness': 0.0,
                'sum_quality': 0.0,
                'unique_users': set(),
            },
        )

        summary['total_attempts'] += 1
        summary['sum_response_time'] += attempt.processing_time_ms or 0.0
        summary['sum_similarity'] += attempt.similarity_score or 0.0
        summary['sum_liveness'] += attempt.liveness_score or 0.0
        summary['sum_quality'] += attempt.face_quality_score or 0.0

        if attempt.matched_user_id:
            summary['unique_users'].add(attempt.matched_user_id)

        result = attempt.result or 'failed'
        if result == 'success':
            summary['successful_attempts'] += 1
        elif result == 'no_match':
            summary['failed_similarity'] += 1
        elif result == 'liveness_failed':
            summary['failed_liveness'] += 1
        elif result == 'quality_too_low':
            summary['failed_quality'] += 1
        elif result == 'multiple_matches':
            summary['failed_multiple_faces'] += 1
        elif result == 'spoofing_detected':
            summary['failed_obstacles'] += 1
        else:
            summary['failed_system_error'] += 1

    results = []
    for date_key, summary in stats_map.items():
        total_attempts = summary['total_attempts']
        failed_attempts = total_attempts - summary['successful_attempts']

        results.append({
            'date': date_key,
            'hour': None,
            'total_attempts': total_attempts,
            'successful_attempts': summary['successful_attempts'],
            'failed_attempts': failed_attempts,
            'failed_similarity': summary['failed_similarity'],
            'failed_liveness': summary['failed_liveness'],
            'failed_quality': summary['failed_quality'],
            'failed_obstacles': summary['failed_obstacles'],
            'failed_no_face': summary['failed_no_face'],
            'failed_multiple_faces': summary['failed_multiple_faces'],
            'failed_system_error': summary['failed_system_error'],
            'avg_response_time': (summary['sum_response_time'] / total_attempts) if total_attempts else 0.0,
            'avg_similarity_score': (summary['sum_similarity'] / total_attempts) if total_attempts else 0.0,
            'avg_liveness_score': (summary['sum_liveness'] / total_attempts) if total_attempts else 0.0,
            'avg_quality_score': (summary['sum_quality'] / total_attempts) if total_attempts else 0.0,
            'unique_users': len(summary['unique_users']),
            'new_enrollments': FaceEnrollment.objects.filter(
                client=client,
                created_at__date=date_key
            ).count(),
            'success_rate': (summary['successful_attempts'] / total_attempts * 100) if total_attempts else 0.0,
        })

    results.sort(key=lambda item: item['date'], reverse=True)
    return results


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
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        client = getattr(self.request, 'client', None)
        if not client:
            return AuthenticationLog.objects.none()

        queryset = AuthenticationLog.objects.filter(client=client)
        
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
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        client = getattr(self.request, 'client', None)
        if not client:
            return AuthenticationLog.objects.none()
        return AuthenticationLog.objects.filter(client=client)


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
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        client = getattr(self.request, 'client', None)
        if not client:
            return SecurityAlert.objects.none()

        queryset = SecurityAlert.objects.filter(client=client)
        
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
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        client = getattr(self.request, 'client', None)
        if not client:
            return SecurityAlert.objects.none()
        return SecurityAlert.objects.filter(client=client)


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
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        client = getattr(request, 'client', None)
        if not client:
            return Response(
                {"detail": "Client authentication required"},
                status=status.HTTP_403_FORBIDDEN,
            )
        
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
            client=client,
            timestamp__gte=start_date
        )
        attempts_qs = FaceRecognitionAttempt.objects.filter(
            client=client,
            created_at__gte=start_date
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
            client=client,
            created_at__gte=start_date
        ).count()
        
        # Recent activities (last 10)
        recent_activities = auth_logs.order_by('-timestamp')[:10].values(
            'timestamp', 'success', 'auth_method', 'failure_reason'
        )
        
        # Performance metrics
        performance_metrics = attempts_qs.aggregate(
            avg_response_time=Avg('processing_time_ms'),
            min_response_time=Min('processing_time_ms'),
            max_response_time=Max('processing_time_ms')
        )
        
        # Risk analysis
        risk_analysis = auth_logs.aggregate(
            avg_risk_score=Avg('risk_score'),
            high_risk_attempts=Count('id', filter=Q(risk_score__gte=0.7))
        )
        risk_analysis['avg_risk_score'] = float(risk_analysis.get('avg_risk_score') or 0.0)
        
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
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        client = getattr(request, 'client', None)
        if not client:
            return Response(
                {"detail": "Client authentication required"},
                status=status.HTTP_403_FORBIDDEN,
            )

        attempts_qs = FaceRecognitionAttempt.objects.filter(client=client)
        total_auth_attempts = attempts_qs.count()
        successful_auths = attempts_qs.filter(result='success').count()
        failed_attempts = total_auth_attempts - successful_auths

        success_rate = (successful_auths / total_auth_attempts * 100) if total_auth_attempts else 0.0

        avg_scores = attempts_qs.aggregate(
            similarity=Avg('similarity_score'),
            liveness=Avg('liveness_score'),
            quality=Avg('face_quality_score')
        )

        auth_methods = list(
            AuthenticationLog.objects.filter(client=client)
            .values_list('auth_method', flat=True)
            .distinct()
        ) or ['face']

        recent_alerts = SecurityAlert.objects.filter(
            client=client,
            created_at__gte=timezone.now() - timedelta(days=30)
        ).count()

        recent_risk_score = (
            AuthenticationLog.objects.filter(
                client=client,
                timestamp__gte=timezone.now() - timedelta(days=7)
            ).aggregate(avg_risk=Avg('risk_score'))['avg_risk']
            or 0.0
        )

        if recent_risk_score > 0.7:
            risk_level = 'high'
        elif recent_risk_score > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        latest_attempt = attempts_qs.order_by('-created_at').first()

        enrolled_users = client.users.filter(is_enrolled=True).count()

        return Response({
            'total_attempts': total_auth_attempts,
            'successful_attempts': successful_auths,
            'failed_attempts': failed_attempts,
            'success_rate': round(success_rate, 2),
            'face_enrolled': enrolled_users,
            'last_login': latest_attempt.created_at if latest_attempt else None,
            'account_created': client.created_at,
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


class SystemMetricsListView(generics.ListAPIView):
    """Expose collected system metrics with optional filtering."""

    serializer_class = SystemMetricsSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        client = getattr(self.request, 'client', None)
        if not client:
            return SystemMetrics.objects.none()

        queryset = SystemMetrics.objects.filter(client=client).order_by('-timestamp')

        metric_name = self.request.query_params.get('metric_name')
        metric_type = self.request.query_params.get('metric_type')
        start = _parse_iso8601(self.request.query_params.get('start'))
        end = _parse_iso8601(self.request.query_params.get('end'))

        if metric_name:
            queryset = queryset.filter(metric_name=metric_name)
        if metric_type:
            queryset = queryset.filter(metric_type=metric_type)
        if start:
            queryset = queryset.filter(timestamp__gte=start)
        if end:
            queryset = queryset.filter(timestamp__lte=end)

        try:
            limit = int(self.request.query_params.get('limit', 0))
        except (TypeError, ValueError):
            limit = 0

        if limit > 0:
            queryset = queryset[:limit]
        return queryset


class UserBehaviorAnalyticsListView(APIView):
    """Return behavioural analytics for the authenticated client."""

    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        client = getattr(request, 'client', None)
        if not client:
            return Response({'count': 0, 'results': []})

        risk_level_filter = request.query_params.get('risk_level')
        results = _build_behavior_insights(client)

        if risk_level_filter:
            results = [item for item in results if item['risk_level'] == risk_level_filter]

        return Response({
            'count': len(results),
            'results': results,
        })


class FaceRecognitionStatsListView(APIView):
    """Return aggregated face recognition statistics."""

    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        client = getattr(request, 'client', None)
        if not client:
            return Response({'count': 0, 'results': []})

        date_from = request.query_params.get('date_from')
        date_to = request.query_params.get('date_to')

        results = _build_recognition_stats(client, date_from, date_to)
        return Response({
            'count': len(results),
            'results': results,
        })


class ModelPerformanceListView(generics.ListAPIView):
    """List model performance metrics across environments."""

    serializer_class = ModelPerformanceSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = ModelPerformance.objects.select_related('model').order_by('-created_at')

        environment = self.request.query_params.get('environment')
        if environment:
            queryset = queryset.filter(environment=environment)

        model_id = self.request.query_params.get('model')
        if model_id:
            queryset = queryset.filter(model_id=model_id)

        return queryset


class DataQualityMetricsListView(generics.ListAPIView):
    """Expose captured data quality metrics."""

    serializer_class = DataQualityMetricsSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = DataQualityMetrics.objects.all().order_by('-date')
        date_from = self.request.query_params.get('date_from')
        date_to = self.request.query_params.get('date_to')

        if date_from:
            queryset = queryset.filter(date__gte=date_from)
        if date_to:
            queryset = queryset.filter(date__lte=date_to)

        return queryset


class AnalysisMonitoringView(APIView):
    """
    Provide consolidated monitoring data for dashboards.
    """

    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        client = getattr(request, 'client', None)
        if not client:
            return Response(
                {"detail": "Client authentication required"},
                status=status.HTTP_403_FORBIDDEN,
            )

        latest_metrics = SystemMetrics.objects.filter(client=client).order_by('-timestamp')[:10]
        metrics_data = SystemMetricsSerializer(latest_metrics, many=True).data

        behavior_insights = _build_behavior_insights(client)[:5]
        face_stats = _build_recognition_stats(client)[:7]
        model_performance = ModelPerformance.objects.select_related('model').order_by('-created_at')[:5]
        data_quality = DataQualityMetrics.objects.all().order_by('-date')[:5]

        payload = {
            'system_metrics': metrics_data,
            'behavior_insights': behavior_insights,
            'face_recognition': face_stats,
            'model_performance': ModelPerformanceSerializer(model_performance, many=True).data,
            'data_quality': DataQualityMetricsSerializer(data_quality, many=True).data,
        }

        return Response(payload)

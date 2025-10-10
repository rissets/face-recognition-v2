"""
Analytics API views for face recognition system
"""
import logging
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework import permissions
from auth_service.authentication import APIKeyAuthentication, JWTClientAuthentication
from analytics.helpers import generate_daily_report, get_client_summary_stats

logger = logging.getLogger(__name__)


@api_view(['GET'])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_daily_analytics(request):
    """Get daily analytics report for the authenticated client"""
    try:
        client = request.client
        
        # Get date from query params, default to today
        date_str = request.GET.get('date')
        if date_str:
            try:
                report_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return JsonResponse({
                    'error': 'Invalid date format. Use YYYY-MM-DD'
                }, status=status.HTTP_400_BAD_REQUEST)
        else:
            report_date = timezone.now().date()
        
        # Generate report
        report = generate_daily_report(client, report_date)
        
        return JsonResponse({
            'success': True,
            'report': report,
            'generated_at': timezone.now().isoformat(),
        })
        
    except Exception as e:
        logger.error(f"Failed to get daily analytics: {e}")
        return JsonResponse({
            'error': 'Failed to generate analytics report',
            'success': False
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_summary_analytics(request):
    """Get summary analytics for the authenticated client"""
    try:
        client = request.client
        
        # Get days parameter, default to 30
        days = request.GET.get('days', 30)
        try:
            days = int(days)
            if days < 1 or days > 365:
                raise ValueError("Days must be between 1 and 365")
        except (ValueError, TypeError):
            return JsonResponse({
                'error': 'Invalid days parameter. Must be integer between 1 and 365'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Generate summary stats
        summary = get_client_summary_stats(client, days)
        
        return JsonResponse({
            'success': True,
            'summary': summary,
            'generated_at': timezone.now().isoformat(),
        })
        
    except Exception as e:
        logger.error(f"Failed to get summary analytics: {e}")
        return JsonResponse({
            'error': 'Failed to generate summary analytics',
            'success': False
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_analytics_overview(request):
    """Get analytics overview with multiple time periods"""
    try:
        client = request.client
        
        # Generate reports for different time periods
        today = timezone.now().date()
        yesterday = today - timedelta(days=1)
        week_ago = today - timedelta(days=7)
        
        reports = {
            'today': generate_daily_report(client, today),
            'yesterday': generate_daily_report(client, yesterday),
            'last_7_days': get_client_summary_stats(client, 7),
            'last_30_days': get_client_summary_stats(client, 30),
        }
        
        return JsonResponse({
            'success': True,
            'overview': reports,
            'generated_at': timezone.now().isoformat(),
        })
        
    except Exception as e:
        logger.error(f"Failed to get analytics overview: {e}")
        return JsonResponse({
            'error': 'Failed to generate analytics overview',
            'success': False
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
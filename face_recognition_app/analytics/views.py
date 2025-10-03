"""
Analytics app views
"""
from rest_framework import generics, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Count, Avg, Q
from django.utils import timezone
from datetime import timedelta
from .models import AuthenticationLog, SecurityAlert, SystemMetrics
from .serializers import AuthenticationLogSerializer, SecurityAlertSerializer


class AuthenticationLogListView(generics.ListAPIView):
    """List authentication logs"""
    serializer_class = AuthenticationLogSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return AuthenticationLog.objects.filter(user=self.request.user)


class AuthenticationLogDetailView(generics.RetrieveAPIView):
    """Retrieve authentication log detail"""
    serializer_class = AuthenticationLogSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return AuthenticationLog.objects.filter(user=self.request.user)


class SecurityAlertListView(generics.ListAPIView):
    """List security alerts"""
    serializer_class = SecurityAlertSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return SecurityAlert.objects.filter(user=self.request.user)


class SecurityAlertDetailView(generics.RetrieveAPIView):
    """Retrieve security alert detail"""
    serializer_class = SecurityAlertSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return SecurityAlert.objects.filter(user=self.request.user)


class AnalyticsDashboardView(APIView):
    """Analytics dashboard data"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        user = request.user
        
        # Get last 30 days data
        thirty_days_ago = timezone.now() - timedelta(days=30)
        
        # Authentication stats
        auth_stats = AuthenticationLog.objects.filter(
            user=user,
            timestamp__gte=thirty_days_ago
        ).aggregate(
            total_attempts=Count('id'),
            successful_attempts=Count('id', filter=Q(success=True)),
            avg_confidence=Avg('confidence_score')
        )
        
        # Security alerts
        alert_count = SecurityAlert.objects.filter(
            user=user,
            created_at__gte=thirty_days_ago
        ).count()
        
        return Response({
            'auth_stats': auth_stats,
            'alert_count': alert_count,
            'period': '30_days'
        })


class StatisticsView(APIView):
    """General statistics"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        user = request.user
        
        # Basic user stats
        total_auth_attempts = AuthenticationLog.objects.filter(user=user).count()
        successful_auths = AuthenticationLog.objects.filter(user=user, success=True).count()
        
        success_rate = (successful_auths / total_auth_attempts * 100) if total_auth_attempts > 0 else 0
        
        return Response({
            'total_attempts': total_auth_attempts,
            'successful_attempts': successful_auths,
            'success_rate': round(success_rate, 2),
            'face_enrolled': user.face_enrolled
        })

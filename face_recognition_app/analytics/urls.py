"""
Analytics app URLs
"""
from django.urls import path
from . import views
from .api_views import get_daily_analytics, get_summary_analytics, get_analytics_overview

app_name = 'analytics'

urlpatterns = [
    # Authentication logs
    path('auth-logs/', views.AuthenticationLogListView.as_view(), name='auth_log_list'),
    path('auth-logs/<uuid:pk>/', views.AuthenticationLogDetailView.as_view(), name='auth_log_detail'),
    
    # Security alerts
    path('security-alerts/', views.SecurityAlertListView.as_view(), name='security_alert_list'),
    path('security-alerts/<uuid:pk>/', views.SecurityAlertDetailView.as_view(), name='security_alert_detail'),
    
    # Analytics dashboard
    path('dashboard/', views.AnalyticsDashboardView.as_view(), name='dashboard'),
    path('statistics/', views.StatisticsView.as_view(), name='statistics'),
    
    # API endpoints
    path('api/daily/', get_daily_analytics, name='api_daily_analytics'),
    path('api/summary/', get_summary_analytics, name='api_summary_analytics'),
    path('api/overview/', get_analytics_overview, name='api_analytics_overview'),
]
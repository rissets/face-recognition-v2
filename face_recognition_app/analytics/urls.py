"""
Analytics app URLs
"""
from django.urls import path
from . import views

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
]
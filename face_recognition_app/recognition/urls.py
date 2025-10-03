"""
Recognition app URLs
"""
from django.urls import path
from . import views

app_name = 'recognition'

urlpatterns = [
    # Face embeddings
    path('embeddings/', views.FaceEmbeddingListView.as_view(), name='embedding_list'),
    path('embeddings/<uuid:pk>/', views.FaceEmbeddingDetailView.as_view(), name='embedding_detail'),
    
    # Enrollment sessions
    path('sessions/', views.EnrollmentSessionListView.as_view(), name='session_list'),
    path('sessions/<uuid:pk>/', views.EnrollmentSessionDetailView.as_view(), name='session_detail'),
    
    # Authentication attempts
    path('attempts/', views.AuthenticationAttemptListView.as_view(), name='attempt_list'),
    path('attempts/<uuid:pk>/', views.AuthenticationAttemptDetailView.as_view(), name='attempt_detail'),
]
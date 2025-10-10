"""
Recognition app views
"""
from rest_framework import generics, permissions
from auth_service.authentication import APIKeyAuthentication, JWTClientAuthentication
from auth_service.models import FaceEnrollment, AuthenticationSession, FaceRecognitionAttempt
from .serializers import (
    FaceEnrollmentSummarySerializer,
    AuthenticationSessionSummarySerializer,
    FaceRecognitionAttemptSummarySerializer,
)


class FaceEmbeddingListView(generics.ListAPIView):
    """List all face embeddings for the authenticated user"""
    serializer_class = FaceEnrollmentSummarySerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        if hasattr(self.request, 'client'):
            return FaceEnrollment.objects.filter(client=self.request.client)
        return FaceEnrollment.objects.none()


class FaceEmbeddingDetailView(generics.RetrieveAPIView):
    """Retrieve face embedding detail"""
    serializer_class = FaceEnrollmentSummarySerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        if hasattr(self.request, 'client'):
            return FaceEnrollment.objects.filter(client=self.request.client)
        return FaceEnrollment.objects.none()


class EnrollmentSessionListView(generics.ListAPIView):
    """List enrollment sessions"""
    serializer_class = AuthenticationSessionSummarySerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Filter by client user - assuming we have client context in request
        if hasattr(self.request, 'client'):
            return AuthenticationSession.objects.filter(client=self.request.client)
        return AuthenticationSession.objects.none()


class EnrollmentSessionDetailView(generics.RetrieveAPIView):
    """Retrieve enrollment session detail"""
    serializer_class = AuthenticationSessionSummarySerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Filter by client user - assuming we have client context in request
        if hasattr(self.request, 'client'):
            return AuthenticationSession.objects.filter(client=self.request.client)
        return AuthenticationSession.objects.none()


class AuthenticationAttemptListView(generics.ListAPIView):
    """List authentication attempts"""
    serializer_class = FaceRecognitionAttemptSummarySerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Filter by client user - assuming we have client context in request
        if hasattr(self.request, 'client'):
            return FaceRecognitionAttempt.objects.filter(client=self.request.client)
        return FaceRecognitionAttempt.objects.none()


class AuthenticationAttemptDetailView(generics.RetrieveAPIView):
    """Retrieve authentication attempt detail"""
    serializer_class = FaceRecognitionAttemptSummarySerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Filter by client user - assuming we have client context in request
        if hasattr(self.request, 'client'):
            return FaceRecognitionAttempt.objects.filter(client=self.request.client)
        return FaceRecognitionAttempt.objects.none()

"""
Recognition app views
"""
from rest_framework import generics, permissions
from drf_spectacular.utils import extend_schema, extend_schema_view
from auth_service.authentication import APIKeyAuthentication, JWTClientAuthentication
from auth_service.models import FaceEnrollment, AuthenticationSession, FaceRecognitionAttempt
from .serializers import (
    FaceEnrollmentSummarySerializer,
    AuthenticationSessionSummarySerializer,
    FaceRecognitionAttemptSummarySerializer,
)


@extend_schema_view(
    get=extend_schema(
        tags=["Legacy Recognition"],
        summary="List Face Embeddings (Legacy)",
        description="This endpoint is deprecated. Use the 'Face Enrollment' endpoints instead. It retrieves a list of face enrollments (embeddings).",
        deprecated=True,
    )
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


@extend_schema_view(
    get=extend_schema(
        tags=["Legacy Recognition"],
        summary="Retrieve Face Embedding (Legacy)",
        description="This endpoint is deprecated. Use the 'Face Enrollment' endpoints instead. It retrieves a specific face enrollment (embedding).",
        deprecated=True,
    )
)
class FaceEmbeddingDetailView(generics.RetrieveAPIView):
    """Retrieve face embedding detail"""
    serializer_class = FaceEnrollmentSummarySerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        if hasattr(self.request, 'client'):
            return FaceEnrollment.objects.filter(client=self.request.client)
        return FaceEnrollment.objects.none()


@extend_schema_view(
    get=extend_schema(
        tags=["Legacy Recognition"],
        summary="List Enrollment Sessions (Legacy)",
        description="This endpoint is deprecated. Use the 'Session Management' endpoints instead. It retrieves a list of authentication sessions.",
        deprecated=True,
    )
)
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


@extend_schema_view(
    get=extend_schema(
        tags=["Legacy Recognition"],
        summary="Retrieve Enrollment Session (Legacy)",
        description="This endpoint is deprecated. Use the 'Session Management' endpoints instead. It retrieves a specific authentication session.",
        deprecated=True,
    )
)
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


@extend_schema_view(
    get=extend_schema(
        tags=["Legacy Recognition"],
        summary="List Authentication Attempts (Legacy)",
        description="This endpoint is deprecated. Use analytics or session status endpoints instead. It retrieves a list of face recognition attempts.",
        deprecated=True,
    )
)
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


@extend_schema_view(
    get=extend_schema(
        tags=["Legacy Recognition"],
        summary="Retrieve Authentication Attempt (Legacy)",
        description="This endpoint is deprecated. Use analytics or session status endpoints instead. It retrieves a specific face recognition attempt.",
        deprecated=True,
    )
)
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

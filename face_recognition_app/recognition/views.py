"""
Recognition app views
"""
from rest_framework import generics, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import FaceEmbedding, EnrollmentSession, AuthenticationAttempt
from .serializers import (
    FaceEmbeddingSerializer, 
    EnrollmentSessionSerializer, 
    AuthenticationAttemptSerializer
)


class FaceEmbeddingListView(generics.ListAPIView):
    """List face embeddings"""
    serializer_class = FaceEmbeddingSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return FaceEmbedding.objects.filter(user=self.request.user)


class FaceEmbeddingDetailView(generics.RetrieveAPIView):
    """Retrieve face embedding detail"""
    serializer_class = FaceEmbeddingSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return FaceEmbedding.objects.filter(user=self.request.user)


class EnrollmentSessionListView(generics.ListAPIView):
    """List enrollment sessions"""
    serializer_class = EnrollmentSessionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return EnrollmentSession.objects.filter(user=self.request.user)


class EnrollmentSessionDetailView(generics.RetrieveAPIView):
    """Retrieve enrollment session detail"""
    serializer_class = EnrollmentSessionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return EnrollmentSession.objects.filter(user=self.request.user)


class AuthenticationAttemptListView(generics.ListAPIView):
    """List authentication attempts"""
    serializer_class = AuthenticationAttemptSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return AuthenticationAttempt.objects.filter(user=self.request.user)


class AuthenticationAttemptDetailView(generics.RetrieveAPIView):
    """Retrieve authentication attempt detail"""
    serializer_class = AuthenticationAttemptSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return AuthenticationAttempt.objects.filter(user=self.request.user)

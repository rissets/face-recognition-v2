"""
User views for profile and device management
"""
from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import get_user_model
from .models import UserProfile, UserDevice
from .serializers import UserProfileSerializer, UserDeviceSerializer, CustomUserSerializer

User = get_user_model()


class UserProfileView(APIView):
    """Get and update user profile"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get user profile information"""
        user = request.user
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        user_serializer = CustomUserSerializer(user)
        profile_serializer = UserProfileSerializer(profile)
        
        return Response({
            'user': user_serializer.data,
            'profile': profile_serializer.data
        })
    
    def put(self, request):
        """Update user profile"""
        user = request.user
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        user_data = request.data.get('user', {})
        profile_data = request.data.get('profile', {})
        
        # Update user fields
        user_serializer = CustomUserSerializer(user, data=user_data, partial=True)
        if user_serializer.is_valid():
            user_serializer.save()
        
        # Update profile fields
        profile_serializer = UserProfileSerializer(profile, data=profile_data, partial=True)
        if profile_serializer.is_valid():
            profile_serializer.save()
            
            return Response({
                'user': user_serializer.data,
                'profile': profile_serializer.data
            })
        
        return Response(
            profile_serializer.errors, 
            status=status.HTTP_400_BAD_REQUEST
        )


class UserDeviceListView(generics.ListAPIView):
    """List user's registered devices"""
    serializer_class = UserDeviceSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return UserDevice.objects.filter(user=self.request.user).order_by('-last_seen')


class UserDeviceDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Get, update, or delete a specific user device"""
    serializer_class = UserDeviceSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return UserDevice.objects.filter(user=self.request.user)
    
    def perform_update(self, serializer):
        """Update device with user context"""
        serializer.save(user=self.request.user)
"""
Streaming views for WebRTC and real-time communication
"""
from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone
from auth_service.authentication import APIKeyAuthentication, JWTClientAuthentication
from .models import StreamingSession, WebRTCSignal
from .serializers import StreamingSessionSerializer, WebRTCSignalSerializer


class StreamingSessionListView(generics.ListAPIView):
    """List streaming sessions for the authenticated user"""
    serializer_class = StreamingSessionSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return StreamingSession.objects.filter(user=self.request.user)


class StreamingSessionDetailView(generics.RetrieveAPIView):
    """Get details of a specific streaming session"""
    serializer_class = StreamingSessionSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return StreamingSession.objects.filter(user=self.request.user)


class StreamingSessionCreateView(generics.CreateAPIView):
    """Create a new streaming session"""
    serializer_class = StreamingSessionSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class WebRTCSignalingView(APIView):
    """Handle WebRTC signaling messages"""
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Handle WebRTC signaling (offer, answer, ICE candidates)"""
        try:
            signal_data = request.data
            signal_type = signal_data.get('type')
            
            if signal_type not in ['offer', 'answer', 'ice_candidate']:
                return Response(
                    {'error': 'Invalid signal type'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            session_token = signal_data.get('session_token')
            if not session_token:
                return Response(
                    {'error': 'session_token is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                session = StreamingSession.objects.get(
                    session_token=session_token
                )
            except StreamingSession.DoesNotExist:
                return Response(
                    {'error': 'Session not found'},
                    status=status.HTTP_404_NOT_FOUND
                )

            if session.user and session.user != request.user:
                return Response(
                    {'error': 'Permission denied for this session'},
                    status=status.HTTP_403_FORBIDDEN
                )

            # Create WebRTC signal record
            signal = WebRTCSignal.objects.create(
                session=session,
                signal_type=signal_type,
                signal_data=signal_data,
                direction=signal_data.get('direction', 'inbound')
            )
            
            serializer = WebRTCSignalSerializer(signal)
            
            return Response({
                'success': True,
                'signal_id': signal.id,
                'message': 'Signal processed successfully',
                'data': serializer.data
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def get(self, request):
        """Get recent signaling messages for the user"""
        signals = WebRTCSignal.objects.filter(
            session__user=request.user
        ).order_by('-created_at')[:10]
        
        serializer = WebRTCSignalSerializer(signals, many=True)
        return Response(serializer.data)

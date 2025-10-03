"""
Streaming views for WebRTC and real-time communication
"""
from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone
from .models import StreamingSession, WebRTCSignal
from .serializers import StreamingSessionSerializer, WebRTCSignalSerializer


class StreamingSessionListView(generics.ListAPIView):
    """List streaming sessions for the authenticated user"""
    serializer_class = StreamingSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return StreamingSession.objects.filter(user=self.request.user)


class StreamingSessionDetailView(generics.RetrieveAPIView):
    """Get details of a specific streaming session"""
    serializer_class = StreamingSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return StreamingSession.objects.filter(user=self.request.user)


class StreamingSessionCreateView(generics.CreateAPIView):
    """Create a new streaming session"""
    serializer_class = StreamingSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class WebRTCSignalingView(APIView):
    """Handle WebRTC signaling messages"""
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
            
            # Create WebRTC signal record
            signal = WebRTCSignal.objects.create(
                user=request.user,
                signal_type=signal_type,
                signal_data=signal_data,
                timestamp=timezone.now()
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
            user=request.user
        ).order_by('-timestamp')[:10]
        
        serializer = WebRTCSignalSerializer(signals, many=True)
        return Response(serializer.data)
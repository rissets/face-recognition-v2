"""
API Views for Face Recognition System
"""
import logging
import json
import numpy as np
import cv2
import base64
import uuid
from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db import transaction
from django.core.cache import cache
from django.conf import settings

from core.serializers import (
    UserRegistrationSerializer, UserProfileSerializer,
    EnrollmentSessionSerializer, EnrollmentRequestSerializer,
    FrameDataSerializer, AuthenticationRequestSerializer,
    AuthenticationAttemptSerializer, SystemStatusSerializer,
    WebRTCSignalSerializer, StreamingSessionSerializer
)
from core.face_recognition_engine import FaceRecognitionEngine
from recognition.models import (
    EnrollmentSession, FaceEmbedding, AuthenticationAttempt,
    LivenessDetection, ObstacleDetection
)
from analytics.models import AuthenticationLog, SecurityAlert
from streaming.models import StreamingSession, WebRTCSignal
from users.models import UserDevice

logger = logging.getLogger('face_recognition')
User = get_user_model()

# Initialize face recognition engine
face_engine = FaceRecognitionEngine()


class UserRegistrationView(generics.CreateAPIView):
    """User registration endpoint"""
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user = serializer.save()
        
        # Log registration
        logger.info(f"New user registered: {user.email}")
        
        return Response({
            'message': 'User registered successfully',
            'user_id': user.id,
            'email': user.email
        }, status=status.HTTP_201_CREATED)


class UserProfileView(generics.RetrieveUpdateAPIView):
    """User profile endpoint"""
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user


class EnrollmentSessionCreateView(APIView):
    """Create new enrollment session"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = EnrollmentRequestSerializer(
            data=request.data,
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)
        
        # Check if user already has active enrollment session
        active_session = EnrollmentSession.objects.filter(
            user=request.user,
            status__in=['pending', 'in_progress']
        ).first()
        
        if active_session:
            return Response({
                'error': 'Active enrollment session already exists',
                'session_token': active_session.session_token
            }, status=status.HTTP_400_BAD_REQUEST)
        
        session = serializer.save()
        
        # Reset face recognition engine for new enrollment
        face_engine.reset_liveness_detector()
        
        return Response({
            'session_token': session.session_token,
            'target_samples': session.target_samples,
            'expires_at': session.expires_at
        }, status=status.HTTP_201_CREATED)


class EnrollmentFrameProcessView(APIView):
    """Process frame for enrollment"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = FrameDataSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        session_token = serializer.validated_data['session_token']
        frame_data = serializer.validated_data['frame_data']
        
        # Get enrollment session
        try:
            session = EnrollmentSession.objects.get(
                session_token=session_token,
                user=request.user,
                status__in=['pending', 'in_progress']
            )
        except EnrollmentSession.DoesNotExist:
            return Response({
                'error': 'Invalid or expired enrollment session'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if session is expired
        if session.is_expired:
            session.status = 'failed'
            session.error_messages = 'Session expired'
            session.save()
            return Response({
                'error': 'Enrollment session expired'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Decode frame data
        try:
            frame = self._decode_frame_data(frame_data)
        except Exception as e:
            return Response({
                'error': f'Invalid frame data: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Process frame with face recognition engine
        result, error = face_engine.process_frame_for_enrollment(frame, str(request.user.id))
        
        if error:
            session.add_log_entry(f"Frame processing failed: {error}", 'error')
            return Response({
                'success': False,
                'error': error,
                'session_status': session.status,
                'completed_samples': session.completed_samples
            })
        
        # Save embedding if quality is good
        with transaction.atomic():
            session.status = 'in_progress'
            session.completed_samples += 1
            
            # Create face embedding record
            embedding = FaceEmbedding.objects.create(
                user=request.user,
                enrollment_session=session,
                sample_number=session.completed_samples,
                quality_score=result['quality_score'],
                confidence_score=result['confidence'],
                face_bbox=result['bbox'].tolist(),
                liveness_score=result['liveness_data'].get('blinks_detected', 0) / 5.0,
                anti_spoofing_score=result['quality_score']  # Simplified
            )
            
            # Save embedding vector
            embedding.set_embedding_vector(result['embedding'])
            embedding.save()
            
            # Save to embedding store
            metadata = {
                'user_id': str(request.user.id),
                'sample_number': session.completed_samples,
                'quality_score': result['quality_score'],
                'enrollment_session': str(session.id)
            }
            
            embedding_id = face_engine.save_embedding(
                str(request.user.id),
                result['embedding'],
                metadata
            )
            
            # Update session averages
            session.average_quality = (
                (session.average_quality * (session.completed_samples - 1) + result['quality_score'])
                / session.completed_samples
            )
            
            # Check if enrollment is complete
            if session.completed_samples >= session.target_samples:
                session.status = 'completed'
                session.completed_at = timezone.now()
                
                # Update user face enrollment status
                request.user.face_enrolled = True
                request.user.enrollment_completed_at = timezone.now()
                request.user.save()
                
                session.add_log_entry("Enrollment completed successfully", 'info')
            
            session.save()
        
        return Response({
            'success': True,
            'session_status': session.status,
            'completed_samples': session.completed_samples,
            'target_samples': session.target_samples,
            'progress_percentage': session.progress_percentage,
            'quality_score': result['quality_score'],
            'liveness_data': result['liveness_data']
        })

    def _decode_frame_data(self, frame_data):
        """Decode base64 frame data to OpenCV image"""
        # Remove data URL prefix if present
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        
        # Decode base64
        frame_bytes = base64.b64decode(frame_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Could not decode image data")
        
        return frame


class AuthenticationCreateView(APIView):
    """Create authentication session"""
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = AuthenticationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        session_type = serializer.validated_data['session_type']
        email = serializer.validated_data.get('email')
        device_info = serializer.validated_data['device_info']
        
        # Create streaming session
        session_token = str(uuid.uuid4())
        
        streaming_session = StreamingSession.objects.create(
            session_token=session_token,
            session_type='authentication',
            remote_address=request.META.get('REMOTE_ADDR'),
            session_data={
                'auth_type': session_type,
                'target_email': email,
                'device_info': device_info
            }
        )
        
        # Reset face recognition engine
        face_engine.reset_liveness_detector()
        
        return Response({
            'session_token': session_token,
            'session_type': session_type,
            'webrtc_config': settings.WEBRTC_CONFIG
        }, status=status.HTTP_201_CREATED)


class AuthenticationFrameProcessView(APIView):
    """Process frame for authentication"""
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = FrameDataSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        session_token = serializer.validated_data['session_token']
        frame_data = serializer.validated_data['frame_data']
        
        # Get streaming session
        try:
            streaming_session = StreamingSession.objects.get(
                session_token=session_token,
                status__in=['initiating', 'connecting', 'connected', 'processing']
            )
        except StreamingSession.DoesNotExist:
            return Response({
                'error': 'Invalid or expired session'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Update session status
        streaming_session.status = 'processing'
        streaming_session.save()
        
        # Decode frame
        try:
            frame = self._decode_frame_data(frame_data)
        except Exception as e:
            return Response({
                'error': f'Invalid frame data: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get target user for verification mode
        target_user = None
        if streaming_session.session_data.get('auth_type') == 'verification':
            target_email = streaming_session.session_data.get('target_email')
            if target_email:
                try:
                    target_user = User.objects.get(email=target_email, is_active=True)
                except User.DoesNotExist:
                    pass
        
        # Authenticate with face recognition engine
        user_id = str(target_user.id) if target_user else None
        auth_result = face_engine.authenticate_user(frame, user_id)
        
        # Create authentication attempt record
        attempt = AuthenticationAttempt.objects.create(
            user=target_user if auth_result['success'] else None,
            session_id=session_token,
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            similarity_score=auth_result.get('similarity_score', 0.0),
            liveness_score=auth_result.get('liveness_data', {}).get('blinks_detected', 0) / 5.0,
            quality_score=auth_result.get('quality_score', 0.0),
            result='success' if auth_result['success'] else 'failed_similarity',
            processing_time=0.0,  # Would need to measure this
            obstacles_detected=auth_result.get('obstacles', []),
            face_bbox=auth_result.get('bbox', []) if 'bbox' in auth_result else None,
            metadata=auth_result
        )
        
        # Log authentication attempt
        authenticated_user = None
        if auth_result['success']:
            # Get authenticated user
            user_id_from_match = auth_result.get('user_id')
            if user_id_from_match:
                try:
                    authenticated_user = User.objects.get(id=user_id_from_match)
                    attempt.user = authenticated_user
                    attempt.save()
                    
                    # Update user's last face auth
                    authenticated_user.last_face_auth = timezone.now()
                    authenticated_user.save()
                    
                except User.DoesNotExist:
                    pass
        
        # Create authentication log
        AuthenticationLog.objects.create(
            user=authenticated_user,
            attempted_email=streaming_session.session_data.get('target_email', ''),
            auth_method='face',
            success=auth_result['success'],
            failure_reason=auth_result.get('error', '') if not auth_result['success'] else '',
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            similarity_score=auth_result.get('similarity_score', 0.0),
            liveness_score=auth_result.get('liveness_data', {}).get('blinks_detected', 0) / 5.0,
            quality_score=auth_result.get('quality_score', 0.0),
            session_id=session_token
        )
        
        # Update streaming session
        streaming_session.status = 'completed' if auth_result['success'] else 'failed'
        streaming_session.completed_at = timezone.now()
        streaming_session.save()
        
        response_data = {
            'success': auth_result['success'],
            'similarity_score': auth_result.get('similarity_score', 0.0),
            'quality_score': auth_result.get('quality_score', 0.0),
            'liveness_data': auth_result.get('liveness_data', {}),
        }
        
        if auth_result['success'] and authenticated_user:
            response_data.update({
                'user': {
                    'id': authenticated_user.id,
                    'email': authenticated_user.email,
                    'full_name': authenticated_user.get_full_name(),
                }
            })
        else:
            response_data['error'] = auth_result.get('error', 'Authentication failed')
        
        return Response(response_data)

    def _decode_frame_data(self, frame_data):
        """Decode base64 frame data to OpenCV image"""
        # Remove data URL prefix if present
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        
        # Decode base64
        frame_bytes = base64.b64decode(frame_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Could not decode image data")
        
        return frame


@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def system_status(request):
    """Get system status"""
    engine_status = face_engine.get_system_status()
    
    # Add additional metrics
    engine_status.update({
        'total_users': User.objects.filter(is_active=True).count(),
        'total_embeddings': FaceEmbedding.objects.filter(is_active=True).count(),
        'active_sessions': EnrollmentSession.objects.filter(
            status__in=['pending', 'in_progress']
        ).count()
    })
    
    serializer = SystemStatusSerializer(engine_status)
    return Response(serializer.data)


class WebRTCSignalingView(APIView):
    """WebRTC signaling endpoint"""
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = WebRTCSignalSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        session_token = serializer.validated_data['session_token']
        signal_type = serializer.validated_data['signal_type']
        signal_data = serializer.validated_data['signal_data']
        
        # Get streaming session
        try:
            session = StreamingSession.objects.get(session_token=session_token)
        except StreamingSession.DoesNotExist:
            return Response({
                'error': 'Invalid session token'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Save signal
        WebRTCSignal.objects.create(
            session=session,
            signal_type=signal_type,
            signal_data=signal_data,
            direction='inbound'
        )
        
        # Update session status based on signal
        if signal_type == 'offer':
            session.status = 'connecting'
        elif signal_type == 'answer':
            session.status = 'connected'
            session.connected_at = timezone.now()
        
        session.save()
        
        return Response({
            'success': True,
            'session_status': session.status
        })


class UserDevicesView(generics.ListCreateAPIView):
    """User devices endpoint"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return UserDevice.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class AuthenticationHistoryView(generics.ListAPIView):
    """User authentication history"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return AuthenticationLog.objects.filter(user=self.request.user)


class SecurityAlertsView(generics.ListAPIView):
    """User security alerts"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return SecurityAlert.objects.filter(user=self.request.user, resolved=False)


class CustomTokenObtainPairView(TokenObtainPairView):
    """Custom JWT token view with device tracking"""
    
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        
        # Track device if authentication successful
        if response.status_code == 200:
            email = request.data.get('email') or request.data.get('username')
            if email:
                try:
                    user = User.objects.get(email=email)
                    device_info = request.data.get('device_info', {})
                    
                    # Create or update device record
                    device_id = device_info.get('device_id', 'unknown')
                    device, created = UserDevice.objects.get_or_create(
                        user=user,
                        device_id=device_id,
                        defaults={
                            'device_name': device_info.get('device_name', 'Unknown Device'),
                            'device_type': device_info.get('device_type', 'web'),
                            'operating_system': device_info.get('os', ''),
                            'browser': device_info.get('browser', ''),
                            'user_agent': request.META.get('HTTP_USER_AGENT', ''),
                            'last_ip': request.META.get('REMOTE_ADDR'),
                        }
                    )
                    
                    if not created:
                        device.last_ip = request.META.get('REMOTE_ADDR')
                        device.login_count += 1
                        device.save()
                    
                except User.DoesNotExist:
                    pass
        
        return response
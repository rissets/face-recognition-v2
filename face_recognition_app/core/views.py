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
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db import transaction
from django.core.cache import cache
from django.conf import settings
from django.core.files.base import ContentFile
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse
from drf_spectacular.types import OpenApiTypes
from django.utils.functional import SimpleLazyObject

from core.serializers import (
    UserRegistrationSerializer, UserProfileSerializer,
    EnrollmentSessionSerializer, EnrollmentRequestSerializer,
    FrameDataSerializer, AuthenticationRequestSerializer,
    AuthenticationAttemptSerializer, SystemStatusSerializer,
    WebRTCSignalSerializer, StreamingSessionSerializer,
    SecurityAlertSerializer
)
# Import enhanced detection views
from .enhanced_views import (
    LivenessDetectionHistoryView, ObstacleDetectionHistoryView,
    detection_analytics_view
)
# Create aliases for missing serializers to avoid errors
FrameProcessRequestSerializer = FrameDataSerializer
FrameProcessResponseSerializer = FrameDataSerializer
AuthenticationResponseSerializer = AuthenticationAttemptSerializer
AuthenticationResultSerializer = AuthenticationAttemptSerializer
WebRTCResponseSerializer = WebRTCSignalSerializer
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


MIN_LIVENESS_FRAMES = 2  # Reduce minimum frames for faster verification
MIN_LIVENESS_BLINKS = 0  # Allow liveness via motion without mandatory blink

# Initialize face recognition engine lazily to avoid heavy imports during management commands
face_engine = SimpleLazyObject(lambda: FaceRecognitionEngine())


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


@extend_schema(
    tags=['User Management'],
    summary='User Profile',
    description='Get and update user profile information',
    responses={
        200: OpenApiResponse(
            response=UserProfileSerializer,
            description='User profile information'
        ),
        401: OpenApiResponse(description='Authentication required'),
    }
)
class UserProfileView(generics.RetrieveUpdateAPIView):
    """User profile endpoint"""
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user


@extend_schema(
    tags=['Face Enrollment'],
    summary='Create Enrollment Session',
    description='Create a new face enrollment session for the authenticated user',
    request=EnrollmentRequestSerializer,
    responses={
        201: OpenApiResponse(
            response=EnrollmentSessionSerializer,
            description='Enrollment session created successfully'
        ),
        400: OpenApiResponse(description='Bad request or active session exists'),
        401: OpenApiResponse(description='Authentication required'),
    }
)
class EnrollmentSessionCreateView(APIView):
    """Create new enrollment session"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = EnrollmentRequestSerializer

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
            # Reuse existing session instead of failing so clients can recover
            active_session.add_log_entry("Reusing active enrollment session", 'info')
            return Response({
                'session_token': active_session.session_token,
                'target_samples': active_session.target_samples,
                'completed_samples': active_session.completed_samples,
                'session_status': active_session.status,
                'expires_at': active_session.expires_at,
                'reused': True
            }, status=status.HTTP_200_OK)
        
        session = serializer.save()
        
        # Reset face recognition engine for new enrollment
        face_engine.reset_liveness_detector()
        
        return Response({
            'session_token': session.session_token,
            'target_samples': session.target_samples,
            'expires_at': session.expires_at
        }, status=status.HTTP_201_CREATED)


class WebRTCEnrollmentSessionCreateView(APIView):
    """Create a WebRTC streaming session for enrollment (parallel to HTTP frame mode)"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        # Create a streaming session record dedicated for enrollment WebRTC
        from streaming.models import StreamingSession
        from django.utils import timezone
        import uuid as _uuid

        limits = getattr(settings, 'FACE_STREAMING_LIMITS', {})
        max_active = limits.get('MAX_ACTIVE_STREAMING_SESSIONS_PER_USER', 3)

        active_qs = StreamingSession.objects.filter(
            user=request.user,
            status__in=['initiating', 'connecting', 'connected', 'processing']
        )
        if active_qs.filter(session_type='enrollment').exists():
            existing = active_qs.filter(session_type='enrollment').first()
            return Response({
                'session_token': existing.session_token,
                'session_type': existing.session_type,
                'status': existing.status,
                'webrtc_url': f"/ws/face-recognition/{existing.session_token}/",
                'note': 'Reusing active enrollment session'
            })

        if active_qs.count() >= max_active:
            return Response({'error': 'Active streaming session limit reached'}, status=status.HTTP_429_TOO_MANY_REQUESTS)

        # Rate limiting (simple cache counter per minute)
        from django.core.cache import cache
        key = f"webrtc_enroll_creates_{request.user.id}"
        creates = cache.get(key, 0)
        if creates >= limits.get('MAX_CREATES_PER_MINUTE', 8):
            return Response({'error': 'Too many enrollment session creates, slow down'}, status=status.HTTP_429_TOO_MANY_REQUESTS)
        cache.set(key, creates + 1, timeout=60)

        session_token = str(_uuid.uuid4())
        streaming_session = StreamingSession.objects.create(
            user=request.user,
            session_token=session_token,
            session_type='enrollment',
            status='initiating',
            remote_address=request.META.get('REMOTE_ADDR'),
            session_data={
                'origin': 'webrtc_enrollment',
                'frames_processed': 0,
                'samples_saved': 0,
            }
        )

        face_engine.reset_liveness_detector()
        return Response({
            'session_token': streaming_session.session_token,
            'session_type': streaming_session.session_type,
            'status': streaming_session.status,
            'webrtc_url': f"/ws/face-recognition/{streaming_session.session_token}/"
        }, status=status.HTTP_201_CREATED)


@extend_schema(
    tags=['Face Enrollment'],
    summary='Process Enrollment Frame',
    description='Process a face image frame during enrollment session',
    request=FrameProcessRequestSerializer,
    responses={
        200: OpenApiResponse(
            response=FrameProcessResponseSerializer,
            description='Frame processed successfully'
        ),
        400: OpenApiResponse(description='Bad request or invalid frame'),
        401: OpenApiResponse(description='Authentication required'),
    }
)
class EnrollmentFrameProcessView(APIView):
    """Process frame during enrollment session"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = FrameProcessRequestSerializer

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

        face_snapshot = result.pop('face_snapshot', None)
        preview_image = None
        if face_snapshot:
            preview_image = "data:image/jpeg;base64," + base64.b64encode(face_snapshot).decode('ascii')

        embedding_vector = result['embedding']
        bbox_array = result['bbox']
        bbox_list = bbox_array.tolist() if hasattr(bbox_array, 'tolist') else list(bbox_array)
        
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
                face_bbox=bbox_list,
                liveness_score=result['liveness_data'].get('blinks_detected', 0) / 5.0,
                anti_spoofing_score=result['quality_score']  # Simplified
            )
            
            # Save embedding vector
            embedding.set_embedding_vector(embedding_vector)
            if face_snapshot:
                embedding.face_image.save(
                    f"embedding_{embedding.id}.jpg",
                    ContentFile(face_snapshot),
                    save=False
                )
            embedding.save()

            if face_snapshot:
                user_obj = request.user
                current_picture = getattr(user_obj, 'profile_picture', None)
                if not current_picture or not getattr(current_picture, 'name', ''):
                    user_obj.profile_picture.save(
                        f"profile_{user_obj.id}.jpg",
                        ContentFile(face_snapshot),
                        save=False
                    )
                    user_obj.save(update_fields=['profile_picture', 'updated_at'])
            
            # Save to embedding store
            metadata = {
                'user_id': str(request.user.id),
                'sample_number': session.completed_samples,
                'quality_score': result['quality_score'],
                'enrollment_session': str(session.id)
            }
            
            embedding_id = face_engine.save_embedding(
                str(request.user.id),
                embedding_vector,
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
            'liveness_data': result['liveness_data'],
            'liveness_verified': result.get('liveness_verified', False),
            'obstacles': result.get('obstacles', []),
            'preview_image': preview_image
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


@extend_schema(
    tags=['Face Recognition'],
    summary='Create Authentication Session',
    description='Create a new face authentication session',
    request=AuthenticationRequestSerializer,
    responses={
        200: OpenApiResponse(
            response=AuthenticationResponseSerializer,
            description='Authentication session created'
        ),
        401: OpenApiResponse(description='Authentication required'),
    }
)
class AuthenticationCreateView(APIView):
    """Create new authentication session"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = AuthenticationRequestSerializer

    def post(self, request):
        serializer = AuthenticationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        session_type = serializer.validated_data['session_type']
        email = serializer.validated_data.get('email')
        device_info = serializer.validated_data['device_info']
        
        # Create streaming session
        session_token = str(uuid.uuid4())
        session_data = {
            'auth_type': session_type,
            'device_info': device_info,
            'frames_processed': 0,
            'liveness_blinks': 0,
            'min_frames_required': MIN_LIVENESS_FRAMES,
            'required_blinks': MIN_LIVENESS_BLINKS,
            'session_origin': 'authenticated'
        }
        if email:
            session_data['target_email'] = email
        
        streaming_session = StreamingSession.objects.create(
            user=request.user,  # Add user to session
            session_token=session_token,
            session_type=session_type,
            status='initiating',  # Explicitly set status
            remote_address=request.META.get('REMOTE_ADDR'),
            session_data=session_data
        )
        
        # Reset face recognition engine
        face_engine.reset_liveness_detector()
        
        return Response({
            'session_token': session_token,
            'session_type': session_type,
            'webrtc_config': settings.WEBRTC_CONFIG
        }, status=status.HTTP_201_CREATED)


class WebRTCAuthenticationCreateView(APIView):
    """Create WebRTC streaming session for authentication/verification"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = AuthenticationRequestSerializer

    def post(self, request):
        serializer = AuthenticationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        session_type = serializer.validated_data['session_type']
        email = serializer.validated_data.get('email')
        device_info = serializer.validated_data['device_info']

        from streaming.models import StreamingSession
        import uuid as _uuid

        limits = getattr(settings, 'FACE_STREAMING_LIMITS', {})
        max_active = limits.get('MAX_ACTIVE_STREAMING_SESSIONS_PER_USER', 3)
        active_qs = StreamingSession.objects.filter(
            user=request.user,
            status__in=['initiating', 'connecting', 'connected', 'processing']
        )
        existing = active_qs.filter(session_type=session_type).first()
        if existing:
            return Response({
                'session_token': existing.session_token,
                'session_type': existing.session_type,
                'status': existing.status,
                'webrtc_url': f"/ws/face-recognition/{existing.session_token}/",
                'note': 'Reusing active authentication session'
            })
        if active_qs.count() >= max_active:
            return Response({'error': 'Active streaming session limit reached'}, status=status.HTTP_429_TOO_MANY_REQUESTS)
        from django.core.cache import cache
        key = f"webrtc_auth_creates_{request.user.id}"
        max_creates = limits.get('MAX_CREATES_PER_MINUTE', 8)
        if max_creates and max_creates > 0:
            creates = cache.get(key, 0)
            if creates >= max_creates:
                retry_after = limits.get('RATE_WINDOW_SECONDS', 60) or 60
                response = Response(
                    {'error': 'Too many auth session creates, slow down'},
                    status=status.HTTP_429_TOO_MANY_REQUESTS
                )
                response['Retry-After'] = retry_after
                return response
            cache.set(key, creates + 1, timeout=limits.get('RATE_WINDOW_SECONDS', 60) or 60)

        session_token = str(_uuid.uuid4())
        session_data = {
            'auth_type': session_type,
            'device_info': device_info,
            'frames_processed': 0,
            'liveness_blinks': 0,
            'min_frames_required': MIN_LIVENESS_FRAMES,
            'required_blinks': MIN_LIVENESS_BLINKS,
            'origin': 'webrtc_auth',
            'session_origin': 'authenticated'
        }
        if email:
            session_data['target_email'] = email

        streaming_session = StreamingSession.objects.create(
            user=request.user,
            session_token=session_token,
            session_type=session_type,
            status='initiating',
            remote_address=request.META.get('REMOTE_ADDR'),
            session_data=session_data
        )

        face_engine.reset_liveness_detector()
        return Response({
            'session_token': streaming_session.session_token,
            'session_type': streaming_session.session_type,
            'status': streaming_session.status,
            'webrtc_url': f"/ws/face-recognition/{streaming_session.session_token}/"
        }, status=status.HTTP_201_CREATED)


class PublicWebRTCAuthenticationCreateView(APIView):
    """Create WebRTC streaming session for public face login (no prior JWT)"""
    permission_classes = [permissions.AllowAny]
    serializer_class = AuthenticationRequestSerializer

    def post(self, request):
        serializer = AuthenticationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        session_type = serializer.validated_data['session_type']
        email = serializer.validated_data.get('email')
        device_info = serializer.validated_data['device_info']

        client_ip = (
            request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip()
            or request.META.get('REMOTE_ADDR')
            or ''
        ) or None

        target_user = None
        if email:
            try:
                target_user = User.objects.get(email=email, is_active=True)
            except User.DoesNotExist:
                return Response({'error': 'User not found or inactive'}, status=status.HTTP_404_NOT_FOUND)

            if not target_user.can_authenticate_with_face():
                return Response({'error': 'Face authentication not enabled for this user'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Default to identification when no email provided
            if session_type in ['verification', 'authentication']:
                session_type = 'identification'

        from streaming.models import StreamingSession
        import uuid as _uuid

        filters = {
            'session_type': session_type,
            'status__in': ['initiating', 'connecting', 'connected', 'processing'],
            'session_data__session_origin': 'public_login'
        }
        if target_user:
            filters['user'] = target_user
        else:
            filters['user__isnull'] = True
        if client_ip:
            filters['remote_address'] = client_ip

        existing_session = StreamingSession.objects.filter(**filters).order_by('-created_at').first()
        if existing_session:
            session_data = existing_session.session_data or {}
            session_data.update({
                'auth_type': session_type,
                'device_info': device_info,
                'frames_processed': 0,
                'session_origin': 'public_login',
                'origin': session_data.get('origin', 'webrtc_public_auth'),
                'reused_at': timezone.now().isoformat()
            })
            if email:
                session_data['target_email'] = email
            else:
                session_data.pop('target_email', None)

            existing_session.session_data = session_data
            existing_session.save(update_fields=['session_data'])

            face_engine.reset_liveness_detector()
            return Response({
                'session_token': existing_session.session_token,
                'session_type': existing_session.session_type,
                'status': existing_session.status,
                'webrtc_url': f"/ws/face-recognition/{existing_session.session_token}/",
                'note': 'Reusing active public session'
            }, status=status.HTTP_200_OK)

        session_token = str(_uuid.uuid4())
        limits = getattr(settings, 'FACE_STREAMING_LIMITS', {})
        from django.core.cache import cache
        rate_key_id = None
        if target_user:
            rate_key_id = f"user_{target_user.id}"
        else:
            rate_key_id = client_ip or request.headers.get('X-Forwarded-For')
        rate_key_id = rate_key_id or 'anon'
        key = f"webrtc_public_creates_{rate_key_id}"
        max_creates = limits.get('MAX_CREATES_PER_MINUTE', 8)
        if max_creates and max_creates > 0:
            creates = cache.get(key, 0)
            if creates >= max_creates:
                retry_after = limits.get('RATE_WINDOW_SECONDS', 60) or 60
                response = Response(
                    {'error': 'Too many public auth session creates, slow down'},
                    status=status.HTTP_429_TOO_MANY_REQUESTS
                )
                response['Retry-After'] = retry_after
                return response
            cache.set(key, creates + 1, timeout=limits.get('RATE_WINDOW_SECONDS', 60) or 60)

        session_data = {
            'auth_type': session_type,
            'target_email': email,
            'device_info': device_info,
            'frames_processed': 0,
            'session_origin': 'public_login',
            'origin': 'webrtc_public_auth'
        }
        if not email:
            session_data.pop('target_email', None)

        StreamingSession.objects.create(
            user=target_user,
            session_token=session_token,
            session_type=session_type,
            status='initiating',
            remote_address=client_ip or request.META.get('REMOTE_ADDR'),
            session_data=session_data
        )

        face_engine.reset_liveness_detector()
        return Response({
            'session_token': session_token,
            'session_type': session_type,
            'status': 'initiating',
            'webrtc_url': f"/ws/face-recognition/{session_token}/"
        }, status=status.HTTP_201_CREATED)


class PublicAuthenticationCreateView(APIView):
    """Create authentication session for face login without prior JWT"""
    permission_classes = [permissions.AllowAny]
    serializer_class = AuthenticationRequestSerializer

    def post(self, request):
        serializer = AuthenticationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session_type = serializer.validated_data['session_type']
        email = serializer.validated_data.get('email')
        device_info = serializer.validated_data['device_info']

        target_user = None
        if email:
            try:
                target_user = User.objects.get(email=email, is_active=True)
            except User.DoesNotExist:
                return Response({'error': 'User not found or inactive'}, status=status.HTTP_404_NOT_FOUND)

            if not target_user.can_authenticate_with_face():
                return Response({'error': 'Face authentication not enabled for this user'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            if session_type in ['verification', 'authentication']:
                session_type = 'identification'

        session_token = str(uuid.uuid4())
        session_data = {
            'auth_type': session_type,
            'device_info': device_info,
            'frames_processed': 0,
            'liveness_blinks': 0,
            'min_frames_required': MIN_LIVENESS_FRAMES,
            'required_blinks': MIN_LIVENESS_BLINKS,
            'session_origin': 'public_login'
        }
        if email:
            session_data['target_email'] = email

        StreamingSession.objects.create(
            user=target_user,
            session_token=session_token,
            session_type=session_type,
            status='initiating',
            remote_address=request.META.get('REMOTE_ADDR'),
            session_data=session_data
        )

        face_engine.reset_liveness_detector()

        return Response({
            'session_token': session_token,
            'session_type': session_type,
            'webrtc_config': settings.WEBRTC_CONFIG
        }, status=status.HTTP_201_CREATED)


@extend_schema(
    tags=['Face Recognition'],
    summary='Process Authentication Frame',
    description='Process a face image frame for authentication',
    request=FrameProcessRequestSerializer,
    responses={
        200: OpenApiResponse(
            response=AuthenticationResultSerializer,
            description='Frame processed successfully'
        ),
        400: OpenApiResponse(description='Bad request or invalid frame'),
        401: OpenApiResponse(description='Authentication required'),
    }
)
class AuthenticationFrameProcessView(APIView):
    """Process frame during authentication session"""
    permission_classes = [permissions.AllowAny]
    serializer_class = FrameProcessRequestSerializer
    MIN_FRAMES_FOR_LIVENESS = MIN_LIVENESS_FRAMES
    REQUIRED_BLINKS = MIN_LIVENESS_BLINKS

    def post(self, request):
        serializer = FrameDataSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        session_token = serializer.validated_data['session_token']
        frame_data = serializer.validated_data['frame_data']
        
        # Get streaming session
        lookup_kwargs = {
            'session_token': session_token,
            'status__in': ['initiating', 'connecting', 'connected', 'processing']
        }

        if request.user.is_authenticated:
            lookup_kwargs['user'] = request.user

        try:
            streaming_session = StreamingSession.objects.get(**lookup_kwargs)
        except StreamingSession.DoesNotExist:
            # Log for debugging
            logger.error(f"Authentication session not found: token={session_token}")
            
            # Check if session exists with different status
            existing_sessions = StreamingSession.objects.filter(session_token=session_token)
            if existing_sessions.exists():
                session = existing_sessions.first()
                logger.error(f"Session exists but with status: {session.status}")
                
                # If session is failed or completed, suggest creating new session
                if session.status in ['failed', 'completed', 'disconnected']:
                    # Clean up the failed/ended session using face engine
                    face_engine.cleanup_failed_session(session_token)
                    
                    # Get additional session status from face engine
                    engine_session_status = face_engine.get_session_status(session_token)
                    
                    return Response({
                        'error': 'Session has ended',
                        'status': session.status,
                        'requires_new_session': True,
                        'message': f'Previous session {session.status}. Please create a new authentication session.',
                        'failure_reason': engine_session_status.get('reason', f'Session status: {session.status}')
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                return Response({
                    'error': f'Session exists but has status: {session.status}',
                    'status': session.status,
                    'requires_new_session': False
                }, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({
                'error': 'Invalid or expired session'
            }, status=status.HTTP_400_BAD_REQUEST)

        if not request.user.is_authenticated and streaming_session.session_data.get('session_origin') != 'public_login':
            return Response({'error': 'Authentication required for this session'}, status=status.HTTP_401_UNAUTHORIZED)

        if streaming_session.user and request.user.is_authenticated and streaming_session.user != request.user:
            return Response({'error': 'Session owner mismatch'}, status=status.HTTP_403_FORBIDDEN)
        
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
        
        session_data = streaming_session.session_data or {}
        auth_type = session_data.get('auth_type') or streaming_session.session_type
        session_owner = streaming_session.user

        # Determine target user based on auth type
        target_user = None
        if auth_type == 'verification':
            target_email = session_data.get('target_email')
            if target_email:
                try:
                    target_user = User.objects.get(email=target_email, is_active=True)
                except User.DoesNotExist:
                    target_user = None
        elif auth_type == 'authentication':
            target_user = session_owner
        else:
            target_user = None
        
        # Authenticate with face recognition engine
        user_id = str(target_user.id) if target_user else None
        auth_result = face_engine.authenticate_user(frame, user_id)

        # Track frame metrics for the session
        frames_processed = int(session_data.get('frames_processed', 0) or 0) + 1

        liveness_data = auth_result.get('liveness_data') or {}
        liveness_blinks = int(liveness_data.get('blinks_detected', 0) or 0)
        previous_blinks = int(session_data.get('liveness_blinks', 0) or 0)
        total_blinks = max(previous_blinks, liveness_blinks)

        liveness_motion_events = int(liveness_data.get('motion_events', 0) or 0)
        previous_motion_events = int(session_data.get('liveness_motion_events', 0) or 0)
        total_motion_events = max(previous_motion_events, liveness_motion_events)

        engine_liveness_verified = bool(auth_result.get('liveness_verified'))
        motion_verified = (
            bool(session_data.get('motion_verified')) or
            bool(liveness_data.get('motion_verified')) or
            engine_liveness_verified
        )

        session_data.update({
            'frames_processed': frames_processed,
            'liveness_blinks': total_blinks,
            'liveness_motion_events': total_motion_events,
            'motion_verified': motion_verified,
            'liveness_verified': engine_liveness_verified or motion_verified,
            'last_similarity': auth_result.get('similarity_score', 0.0),
            'last_quality': auth_result.get('quality_score', 0.0),
            'last_error': auth_result.get('error'),
            'last_liveness': liveness_data,
            'last_updated': timezone.now().isoformat(),
            'match_fallback_used': auth_result.get(
                'match_fallback_used',
                session_data.get('match_fallback_used', False)
            ),
            'latest_bbox': auth_result.get('bbox')
        })
        streaming_session.session_data = session_data
        streaming_session.save(update_fields=['session_data'])
        cache.set(f"face_session_{session_token}", session_data, timeout=300)

        min_frames_met = frames_processed >= self.MIN_FRAMES_FOR_LIVENESS
        liveness_met = (
            engine_liveness_verified
            or motion_verified
            or (self.REQUIRED_BLINKS > 0 and total_blinks >= self.REQUIRED_BLINKS)
        )
        failure_reason = auth_result.get('error') or ''
        liveness_failure = 'liveness' in failure_reason.lower()

        if motion_verified or engine_liveness_verified:
            normalized_liveness = 1.0
        elif self.REQUIRED_BLINKS:
            normalized_liveness = min(1.0, total_blinks / max(1, self.REQUIRED_BLINKS))
        else:
            normalized_liveness = min(1.0, total_motion_events / max(1, self.MIN_FRAMES_FOR_LIVENESS))

        # Require minimum frames and blink count before finalising
        if auth_result.get('success') and not (min_frames_met and liveness_met):
            return Response({
                'success': False,
                'requires_more_frames': True,
                'message': 'Lanjutkan streaming dengan kedipan atau gerakan wajah ringan untuk verifikasi.',
                'frames_processed': frames_processed,
                'min_frames_required': self.MIN_FRAMES_FOR_LIVENESS,
                'liveness_blinks': total_blinks,
                'liveness_motion_events': total_motion_events,
                'liveness_required_blinks': self.REQUIRED_BLINKS,
                'liveness_data': liveness_data,
                'liveness_score': normalized_liveness,
                'motion_verified': motion_verified,
                'match_fallback_used': auth_result.get('match_fallback_used', False),
                'session_finalized': False,
                'requires_new_session': False
            })

        if (not auth_result.get('success') and liveness_failure and
                not (min_frames_met and liveness_met)):
            return Response({
                'success': False,
                'requires_more_frames': True,
                'message': 'Deteksi liveness belum terpenuhi. Lanjutkan streaming serta lakukan kedipan atau gerakan wajah.',
                'frames_processed': frames_processed,
                'min_frames_required': self.MIN_FRAMES_FOR_LIVENESS,
                'liveness_blinks': total_blinks,
                'liveness_motion_events': total_motion_events,
                'liveness_required_blinks': self.REQUIRED_BLINKS,
                'liveness_data': liveness_data,
                'liveness_score': normalized_liveness,
                'motion_verified': motion_verified,
                'match_fallback_used': auth_result.get('match_fallback_used', False),
                'error': failure_reason,
                'session_finalized': False,
                'requires_new_session': False
            })

        attempted_email = session_data.get('target_email') or ''

        if auth_result.get('success'):
            authenticated_user = None
            user_id_from_match = auth_result.get('user_id')
            if user_id_from_match:
                try:
                    authenticated_user = User.objects.get(id=user_id_from_match)
                except User.DoesNotExist:
                    authenticated_user = None

            if authenticated_user and not authenticated_user.can_authenticate_with_face():
                failure_reason = 'Face authentication disabled for this user'
                auth_result['success'] = False
                auth_result['error'] = failure_reason
                authenticated_user = None
            else:
                attempt = AuthenticationAttempt.objects.create(
                    user=authenticated_user,
                    session_id=session_token,
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT', ''),
                    similarity_score=auth_result.get('similarity_score', 0.0),
                    liveness_score=normalized_liveness,
                    quality_score=auth_result.get('quality_score', 0.0),
                    result='success',
                    processing_time=0.0,
                    obstacles_detected=auth_result.get('obstacles', []),
                    face_bbox=auth_result.get('bbox', []) if 'bbox' in auth_result else None,
                    metadata=auth_result
                )

                # Create liveness detection record
                LivenessDetection.objects.create(
                    authentication_attempt=attempt,
                    blinks_detected=total_blinks,
                    blink_quality_scores=liveness_data.get('quality_scores', []),
                    ear_history=liveness_data.get('ear_history', []),
                    ear_baseline=liveness_data.get('ear_baseline'),
                    frames_processed=frames_processed,
                    valid_frames=frames_processed,  # Assuming all processed frames are valid
                    challenge_type='blink',
                    challenge_completed=liveness_met,
                    liveness_score=normalized_liveness,
                    is_live=liveness_met,
                    debug_data={
                        'motion_events': total_motion_events,
                        'motion_score': liveness_data.get('motion_score', 0.0),
                        'motion_verified': motion_verified,
                        'engine_liveness_verified': engine_liveness_verified,
                        'liveness_data': liveness_data
                    }
                )

                # Create obstacle detection record
                obstacles = auth_result.get('obstacles', [])
                obstacle_confidence = auth_result.get('obstacle_confidence', {})
                ObstacleDetection.objects.create(
                    authentication_attempt=attempt,
                    glasses_detected='glasses' in obstacles,
                    glasses_confidence=obstacle_confidence.get('glasses', 0.0),
                    mask_detected='mask' in obstacles,
                    mask_confidence=obstacle_confidence.get('mask', 0.0),
                    hat_detected='hat' in obstacles,
                    hat_confidence=obstacle_confidence.get('hat', 0.0),
                    hand_covering='hand_covering' in obstacles,
                    hand_confidence=obstacle_confidence.get('hand_covering', 0.0),
                    has_obstacles=len(obstacles) > 0,
                    obstacle_score=max(obstacle_confidence.values()) if obstacle_confidence else 0.0,
                    detection_details={
                        'detected_obstacles': obstacles,
                        'confidence_scores': obstacle_confidence,
                        'processing_method': 'advanced_mediapipe'
                    }
                )

                if authenticated_user:
                    attempt.user = authenticated_user
                    attempt.save(update_fields=['user'])
                    authenticated_user.last_face_auth = timezone.now()
                    authenticated_user.save(update_fields=['last_face_auth'])
                    if not streaming_session.user:
                        streaming_session.user = authenticated_user
                        streaming_session.save(update_fields=['user'])
                    session_data['recognized_user_id'] = str(authenticated_user.id)
                    session_data['recognized_user_email'] = authenticated_user.email

                AuthenticationLog.objects.create(
                    user=authenticated_user,
                    attempted_email=attempted_email,
                    auth_method='face',
                    success=True,
                    failure_reason='',
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT', ''),
                    similarity_score=auth_result.get('similarity_score', 0.0),
                    liveness_score=normalized_liveness,
                    quality_score=auth_result.get('quality_score', 0.0),
                    session_id=session_token
                )

                session_data['final_status'] = 'success'
                session_data['liveness_verified'] = True
                streaming_session.session_data = session_data
                streaming_session.status = 'completed'
                streaming_session.completed_at = timezone.now()
                streaming_session.save(update_fields=['status', 'completed_at', 'session_data'])
                cache.set(f"face_session_{session_token}", session_data, timeout=300)

                response_data = {
                    'success': True,
                    'similarity_score': auth_result.get('similarity_score', 0.0),
                    'quality_score': auth_result.get('quality_score', 0.0),
                    'liveness_data': liveness_data,
                    'liveness_score': normalized_liveness,
                    'frames_processed': frames_processed,
                    'liveness_blinks': total_blinks,
                    'liveness_motion_events': total_motion_events,
                    'motion_verified': motion_verified,
                    'session_finalized': True,
                    'requires_more_frames': False,
                    'message': 'Authentication successful',
                    'requires_new_session': False,
                    'match_fallback_used': auth_result.get('match_fallback_used', False)
                }

                if authenticated_user:
                    response_data['user'] = {
                        'id': authenticated_user.id,
                        'email': authenticated_user.email,
                        'full_name': authenticated_user.get_full_name(),
                    }

                if authenticated_user and session_data.get('session_origin') == 'public_login':
                    refresh = RefreshToken.for_user(authenticated_user)
                    response_data['refresh_token'] = str(refresh)
                    response_data['access_token'] = str(refresh.access_token)

                return Response(response_data)

        # Final failure (either liveness after requirements or similarity mismatch)
        lower_reason = failure_reason.lower()
        failure_result_code = 'failed_similarity'
        if liveness_failure:
            failure_result_code = 'failed_liveness'
        elif 'quality' in lower_reason:
            failure_result_code = 'failed_quality'
        elif 'obstacle' in lower_reason:
            failure_result_code = 'failed_obstacles'
        elif 'multiple' in lower_reason and 'face' in lower_reason:
            failure_result_code = 'failed_multiple_faces'
        elif 'no face' in lower_reason:
            failure_result_code = 'failed_no_face'
        elif 'system' in lower_reason:
            failure_result_code = 'failed_system_error'
        elif 'disabled' in lower_reason:
            failure_result_code = 'failed_disabled'

        attempt = AuthenticationAttempt.objects.create(
            user=target_user,
            session_id=session_token,
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            similarity_score=auth_result.get('similarity_score', 0.0),
            liveness_score=normalized_liveness,
            quality_score=auth_result.get('quality_score', 0.0),
            result=failure_result_code,
            processing_time=0.0,
            obstacles_detected=auth_result.get('obstacles', []),
            face_bbox=auth_result.get('bbox', []) if 'bbox' in auth_result else None,
            metadata=auth_result
        )

        # Create liveness detection record for failed attempt
        LivenessDetection.objects.create(
            authentication_attempt=attempt,
            blinks_detected=total_blinks,
            blink_quality_scores=liveness_data.get('quality_scores', []),
            ear_history=liveness_data.get('ear_history', []),
            ear_baseline=liveness_data.get('ear_baseline'),
            frames_processed=frames_processed,
            valid_frames=frames_processed,  # Assuming all processed frames are valid
            challenge_type='blink',
            challenge_completed=liveness_met,
            liveness_score=normalized_liveness,
            is_live=liveness_met,
            debug_data={
                'motion_events': total_motion_events,
                'motion_score': liveness_data.get('motion_score', 0.0),
                'motion_verified': motion_verified,
                'engine_liveness_verified': engine_liveness_verified,
                'liveness_data': liveness_data,
                'failure_reason': failure_reason
            }
        )

        # Create obstacle detection record for failed attempt
        obstacles = auth_result.get('obstacles', [])
        obstacle_confidence = auth_result.get('obstacle_confidence', {})
        ObstacleDetection.objects.create(
            authentication_attempt=attempt,
            glasses_detected='glasses' in obstacles,
            glasses_confidence=obstacle_confidence.get('glasses', 0.0),
            mask_detected='mask' in obstacles,
            mask_confidence=obstacle_confidence.get('mask', 0.0),
            hat_detected='hat' in obstacles,
            hat_confidence=obstacle_confidence.get('hat', 0.0),
            hand_covering='hand_covering' in obstacles,
            hand_confidence=obstacle_confidence.get('hand_covering', 0.0),
            has_obstacles=len(obstacles) > 0,
            obstacle_score=max(obstacle_confidence.values()) if obstacle_confidence else 0.0,
            detection_details={
                'detected_obstacles': obstacles,
                'confidence_scores': obstacle_confidence,
                'processing_method': 'advanced_mediapipe',
                'failure_reason': failure_reason
            }
        )

        AuthenticationLog.objects.create(
            user=target_user,
            attempted_email=attempted_email,
            auth_method='face',
            success=False,
            failure_reason=failure_reason or 'Authentication failed',
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            similarity_score=auth_result.get('similarity_score', 0.0),
            liveness_score=normalized_liveness,
            quality_score=auth_result.get('quality_score', 0.0),
            session_id=session_token
        )

        session_data['final_status'] = 'failed'
        session_data['failure_reason'] = failure_reason or 'Authentication failed'
        streaming_session.session_data = session_data
        streaming_session.status = 'failed'
        streaming_session.completed_at = timezone.now()
        streaming_session.save(update_fields=['status', 'completed_at', 'session_data'])
        cache.set(f"face_session_{session_token}", session_data, timeout=300)

        face_engine.mark_session_failed(session_token, failure_reason or 'Authentication failed')

        return Response({
            'success': False,
            'error': failure_reason or 'Authentication failed',
            'message': failure_reason or 'Authentication failed',
            'similarity_score': auth_result.get('similarity_score', 0.0),
            'quality_score': auth_result.get('quality_score', 0.0),
            'liveness_data': liveness_data,
            'liveness_score': normalized_liveness,
            'frames_processed': frames_processed,
            'liveness_blinks': total_blinks,
            'liveness_motion_events': total_motion_events,
            'requires_more_frames': False,
            'requires_new_session': True,
            'session_finalized': True,
            'motion_verified': motion_verified,
            'match_fallback_used': session_data.get('match_fallback_used', False)
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


@extend_schema(
    tags=['System'],
    summary='System Status',
    description='Get system status and health information',
    responses={
        200: OpenApiResponse(
            response=SystemStatusSerializer,
            description='System status information'
        ),
        401: OpenApiResponse(description='Authentication required'),
    }
)
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


@extend_schema(
    tags=['WebRTC'],
    summary='WebRTC Signaling',
    description='Handle WebRTC signaling for video streaming',
    request=WebRTCSignalSerializer,
    responses={
        200: OpenApiResponse(
            response=WebRTCResponseSerializer,
            description='Signal processed successfully'
        ),
        400: OpenApiResponse(description='Invalid signal data'),
        401: OpenApiResponse(description='Authentication required'),
    }
)
class WebRTCSignalingView(APIView):
    """Handle WebRTC signaling"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = WebRTCSignalSerializer

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
    """List and create user devices"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = UserProfileSerializer

    def get_queryset(self):
        return [self.request.user]  # Return user as queryset


class AuthenticationHistoryView(generics.ListAPIView):
    """List authentication history"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = AuthenticationAttemptSerializer

    def get_queryset(self):
        return AuthenticationAttempt.objects.filter(user=self.request.user)
        return AuthenticationAttempt.objects.filter(user=self.request.user)


class SecurityAlertsView(generics.ListAPIView):
    """List security alerts"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = SecurityAlertSerializer
    
    def get_queryset(self):
        """Return security alerts for authenticated user"""
        return SecurityAlert.objects.filter(
            user=self.request.user
        ).order_by('-created_at')[:50]


class CustomTokenObtainPairView(TokenObtainPairView):
    """Custom JWT token view with device tracking"""
    serializer_class = None  # Will be imported later to avoid circular import
    
    def get_serializer_class(self):
        from core.serializers import CustomTokenObtainPairSerializer
        return CustomTokenObtainPairSerializer
    
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
                    device_id = device_info.get('device_id', 'unknown111')
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

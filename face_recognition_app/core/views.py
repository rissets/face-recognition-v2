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
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse
from drf_spectacular.types import OpenApiTypes

from core.serializers import (
    UserRegistrationSerializer, UserProfileSerializer,
    EnrollmentSessionSerializer, EnrollmentRequestSerializer,
    FrameDataSerializer, AuthenticationRequestSerializer,
    AuthenticationAttemptSerializer, SystemStatusSerializer,
    WebRTCSignalSerializer, StreamingSessionSerializer,
    SecurityAlertSerializer
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
            'liveness_data': result['liveness_data'],
            'liveness_verified': result.get('liveness_verified', False),
            'obstacles': result.get('obstacles', [])
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
        
        streaming_session = StreamingSession.objects.create(
            user=request.user,  # Add user to session
            session_token=session_token,
            session_type='authentication',
            status='initiating',  # Explicitly set status
            remote_address=request.META.get('REMOTE_ADDR'),
            session_data={
                'auth_type': session_type,
                'target_email': email,
                'device_info': device_info,
                'frames_processed': 0,
                'liveness_blinks': 0,
                'min_frames_required': MIN_LIVENESS_FRAMES,
                'required_blinks': MIN_LIVENESS_BLINKS,
                'session_origin': 'authenticated'
            }
        )
        
        # Reset face recognition engine
        face_engine.reset_liveness_detector()
        
        return Response({
            'session_token': session_token,
            'session_type': session_type,
            'webrtc_config': settings.WEBRTC_CONFIG
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

        if not email:
            return Response({'error': 'Email is required for face login'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            target_user = User.objects.get(email=email, is_active=True)
        except User.DoesNotExist:
            return Response({'error': 'User not found or inactive'}, status=status.HTTP_404_NOT_FOUND)

        if not target_user.face_auth_enabled or not target_user.face_enrolled:
            return Response({'error': 'Face authentication not enabled for this user'}, status=status.HTTP_400_BAD_REQUEST)

        session_token = str(uuid.uuid4())

        StreamingSession.objects.create(
            user=target_user,
            session_token=session_token,
            session_type='authentication',
            status='initiating',
            remote_address=request.META.get('REMOTE_ADDR'),
            session_data={
                'auth_type': session_type,
                'target_email': email,
                'device_info': device_info,
                'frames_processed': 0,
                'liveness_blinks': 0,
                'min_frames_required': MIN_LIVENESS_FRAMES,
                'required_blinks': MIN_LIVENESS_BLINKS,
                'session_origin': 'public_login'
            }
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
        
        # Get target user for verification mode
        target_user = streaming_session.user
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

        # Track frame metrics for the session
        session_data = streaming_session.session_data or {}
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

            if authenticated_user:
                attempt.user = authenticated_user
                attempt.save(update_fields=['user'])
                authenticated_user.last_face_auth = timezone.now()
                authenticated_user.save(update_fields=['last_face_auth'])

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

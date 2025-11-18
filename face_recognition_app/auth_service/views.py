"""
API views for the third-party face recognition service.

This module exposes enrollment and authentication endpoints that allow
clients to stream multiple frames while we aggregate liveness signals
(blink or natural motion) before finalising an enrollment or login.
"""
from __future__ import annotations

import base64
import json
import logging
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from django.conf import settings
from django.db import transaction
from core.face_recognition_engine import json_serializable
from django.db.models import Avg, Count, Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.functional import SimpleLazyObject
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import (
    action,
    api_view,
    authentication_classes,
    permission_classes,
)
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiResponse, extend_schema_view

from clients.models import ClientUser
from analytics.models import AuthenticationLog, SecurityAlert
from analytics.helpers import (
    track_enrollment_metrics,
    track_authentication_metrics,
    update_face_recognition_stats,
    track_security_event,
)
from core.face_recognition_engine import FaceRecognitionEngine

from .authentication import APIKeyAuthentication, JWTClientAuthentication
from .models import (
    AuthenticationSession,
    FaceEnrollment,
    FaceRecognitionAttempt,
    LivenessDetectionResult,
    SystemMetrics,
)
from .serializers import (
    AuthenticationRequestSerializer,
    AuthenticationSessionSerializer,
    EnrollmentRequestSerializer,
    EnrollmentResponseSerializer,
    FaceEnrollmentSerializer,
    FaceImageUploadSerializer,
    SessionStatusSerializer,
    SystemMetricsSerializer,
    EnrollmentCreateResponseSerializer,
    AuthSessionCreateResponseSerializer,
    ErrorResponseSerializer,
    FaceFrameResponseSerializer,
)

logger = logging.getLogger("face_recognition")
face_engine = SimpleLazyObject(lambda: FaceRecognitionEngine())

MIN_LIVENESS_FRAMES = getattr(settings, "FACE_MIN_LIVENESS_FRAMES", 3)
MIN_LIVENESS_BLINKS = getattr(settings, "FACE_MIN_LIVENESS_BLINKS", 1)
MAX_HISTORY = 25


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ensure_json_serializable_metadata(metadata: dict) -> dict:
    """Ensure metadata can be safely serialized to JSON for database storage"""
    from core.face_recognition_engine import json_serializable
    return json_serializable(metadata)

def _engine_user_id(client_id: str, external_user_id: str) -> str:
    return f"{client_id}:{external_user_id}"


def _decode_image_to_frame(image_file) -> np.ndarray:
    """Decode an uploaded file or content file to an OpenCV BGR frame."""
    if not image_file:
        raise ValueError("No image payload supplied")

    try:
        if hasattr(image_file, "seek"):
            image_file.seek(0)
        data = image_file.read()
        if hasattr(image_file, "seek"):
            image_file.seek(0)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Could not read image data: {exc}") from exc

    if not data:
        raise ValueError("Image data empty")

    buffer = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image payload")
    return frame


def _snapshot_to_data_url(snapshot: Optional[bytes]) -> Optional[str]:
    if not snapshot:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(snapshot).decode("ascii")


def _save_face_image_to_storage(
    face_snapshot: Optional[bytes], 
    client_id: str, 
    external_user_id: str, 
    sample_number: int
) -> Optional[str]:
    """Save face image snapshot to storage and return the path"""
    if not face_snapshot:
        return None
    
    try:
        from django.core.files.base import ContentFile
        from django.core.files.storage import default_storage
        import uuid
        
        # Create unique filename
        timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enrollments/{client_id}/{external_user_id}/sample_{sample_number}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        
        # Save to storage (will use MinIO if configured)
        file_path = default_storage.save(filename, ContentFile(face_snapshot))
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save face image to storage: {e}")
        return None


def _embedding_to_json(embedding: np.ndarray) -> str:
    if embedding is None:
        return "[]"
    if isinstance(embedding, np.ndarray):
        embedding_list = embedding.astype(float).tolist()
    else:
        embedding_list = list(embedding)
    return json.dumps(embedding_list)


def _compute_liveness_score(
    blinks: int,
    motion_events: int,
    required_blinks: int,
    min_frames: int,
    verified: bool,
) -> float:
    if verified:
        return 1.0
    if required_blinks and required_blinks > 0:
        return min(1.0, blinks / float(required_blinks))
    return min(1.0, motion_events / max(1, min_frames))


def _append_history(metadata: Dict[str, Any], key: str, entry: Dict[str, Any]) -> None:
    history = metadata.get(key) or []
    history.append(entry)
    metadata[key] = history[-MAX_HISTORY:]


def _validate_enhanced_liveness(liveness_data: Dict[str, Any], total_blinks: int, total_motion_events: int) -> Tuple[bool, str]:
    """Enhanced liveness validation using multiple detection methods"""
    reasons = []
    
    # Check engine liveness
    engine_verified = bool(liveness_data.get("liveness_verified", False))
    if engine_verified:
        reasons.append("engine_detection")
    
    # Check blink detection
    blink_verified = bool(liveness_data.get("blink_verified", False)) or total_blinks >= MIN_LIVENESS_BLINKS
    if blink_verified:
        reasons.append("blink_detection")
    
    # Check motion detection  
    motion_verified = bool(liveness_data.get("motion_verified", False)) or total_motion_events > 0
    if motion_verified:
        reasons.append("motion_detection")
    
    # Check for head movement indicators
    ear_variation = liveness_data.get("ear_variation", 0.0)
    if ear_variation > 0.05:  # Significant EAR variation indicates natural movement
        reasons.append("ear_variation")
    
    # Quality-based liveness hints
    quality = liveness_data.get("quality", 0.0)
    if quality > 0.8:  # High quality frames often indicate live capture
        reasons.append("high_quality")
    
    # At least one method must verify liveness
    is_live = len(reasons) > 0
    reason_str = ", ".join(reasons) if reasons else "no_liveness_indicators"
    
    return is_live, reason_str


def _resolve_engine_user(client, engine_id: Optional[str]) -> Optional[ClientUser]:
    if not engine_id:
        logger.debug("_resolve_engine_user: engine_id is None or empty")
        return None
    if ":" not in engine_id:
        logger.debug(f"_resolve_engine_user: engine_id '{engine_id}' does not contain ':'")
        return None
    
    client_id, external_user_id = engine_id.split(":", 1)
    logger.debug(f"_resolve_engine_user: parsed client_id='{client_id}', external_user_id='{external_user_id}'")
    logger.debug(f"_resolve_engine_user: current client.client_id='{client.client_id}'")
    
    if client_id != client.client_id:
        logger.debug(f"_resolve_engine_user: client_id mismatch")
        return None
    try:
        user = ClientUser.objects.get(client=client, external_user_id=external_user_id)
        logger.debug(f"_resolve_engine_user: found user {user.id} with external_user_id '{user.external_user_id}'")
        return user
    except ClientUser.DoesNotExist:
        logger.debug(f"_resolve_engine_user: no ClientUser found for external_user_id '{external_user_id}'")
        return None


# ---------------------------------------------------------------------------
# ViewSets
# ---------------------------------------------------------------------------


@extend_schema_view(
    list=extend_schema(
        tags=["Session Management"],
        summary="List Authentication Sessions",
        description="Retrieve a paginated list of authentication sessions for the authenticated client.",
    ),
    retrieve=extend_schema(
        tags=["Session Management"],
        summary="Retrieve an Authentication Session",
        description="Get detailed information about a specific authentication session by its ID.",
    ),
)
class AuthenticationSessionViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only access to authentication sessions for the authenticated client."""

    queryset = AuthenticationSession.objects.all()
    serializer_class = AuthenticationSessionSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if hasattr(self.request, "client"):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()

    @extend_schema(
        tags=["Session Management"],
        summary="Get Session Status",
        description="Check the current status of an authentication or enrollment session.",
        responses={200: SessionStatusSerializer, 404: OpenApiResponse(description="Not Found")}
    )
    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        session = self.get_object()
        enrollment = session.enrollments.first() if session.session_type == "enrollment" else None
        status_payload = {
            "session_token": session.session_token,
            "status": session.status,
            "session_type": session.session_type,
            "created_at": session.created_at,
            "expires_at": session.expires_at,
            "completed_at": session.completed_at,
            "metadata": session.metadata,
        }
        if enrollment:
            status_payload["enrollment_status"] = enrollment.status
        serializer = SessionStatusSerializer(status_payload)
        return Response(serializer.data)


@extend_schema_view(
    list=extend_schema(
        tags=["Face Enrollment"],
        summary="List Face Enrollments",
        description="Retrieve a list of all face enrollments associated with the client.",
    ),
    retrieve=extend_schema(
        tags=["Face Enrollment"],
        summary="Retrieve a Face Enrollment",
        description="Get detailed information about a specific face enrollment by its ID.",
    ),
    create=extend_schema(
        tags=["Face Enrollment"],
        summary="Create a Face Enrollment (Manual)",
        description="Manually create a face enrollment record. This is typically handled by the session-based enrollment flow.",
    ),
    update=extend_schema(
        tags=["Face Enrollment"],
        summary="Update a Face Enrollment",
        description="Update the details of an existing face enrollment.",
    ),
    partial_update=extend_schema(
        tags=["Face Enrollment"],
        summary="Partially Update a Face Enrollment",
        description="Partially update the details of an existing face enrollment.",
    ),
    destroy=extend_schema(
        tags=["Face Enrollment"],
        summary="Delete a Face Enrollment",
        description="Permanently delete a face enrollment record.",
    ),
)
class FaceEnrollmentViewSet(viewsets.ModelViewSet):
    """Manage enrollments for client users."""

    queryset = FaceEnrollment.objects.all()
    serializer_class = FaceEnrollmentSerializer
    authentication_classes = [APIKeyAuthentication, JWTClientAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if hasattr(self.request, "client"):
            return self.queryset.filter(client=self.request.client)
        return self.queryset.none()

    def perform_create(self, serializer):
        serializer.save(client=self.request.client)


# ---------------------------------------------------------------------------
# Enrollment endpoints
# ---------------------------------------------------------------------------


@extend_schema(
    tags=["Face Enrollment"],
    summary="Create Enrollment Session",
    description="Initializes a session to enroll a new user by capturing face frames. A `session_token` is returned, which must be used for subsequent frame uploads.",
    request=EnrollmentRequestSerializer,
    responses={
        201: EnrollmentCreateResponseSerializer,
        400: OpenApiResponse(response=ErrorResponseSerializer, description="Bad Request - Invalid input or user not found."),
        401: OpenApiResponse(response=ErrorResponseSerializer, description="Unauthorized - Invalid API key."),
    },
)
@api_view(["POST"])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def create_enrollment_session(request):
    serializer = EnrollmentRequestSerializer(
        data=request.data,
        context={"client": getattr(request, "client", None)},
    )
    serializer.is_valid(raise_exception=True)

    client = request.client
    user = ClientUser.objects.get(
        client=client,
        external_user_id=serializer.validated_data["user_id"],
    )
    request_metadata = serializer.validated_data.get("metadata") or {}

    target_samples = int(request_metadata.get("target_samples") or MIN_LIVENESS_FRAMES)
    target_samples = max(target_samples, MIN_LIVENESS_FRAMES)

    last_sample = (
        FaceEnrollment.objects.filter(client=client, client_user=user)
        .order_by("-sample_number")
        .first()
    )
    sample_number = (last_sample.sample_number if last_sample else 0) + 1

    session = AuthenticationSession.objects.create(
        client=client,
        client_user=user,  # Add the client user to session
        session_type="enrollment",
        status="active",
        ip_address=request.META.get("REMOTE_ADDR"),
        user_agent=request.META.get("HTTP_USER_AGENT", ""),
        metadata={
            "target_samples": target_samples,
            "frames_processed": 0,
            "liveness_blinks": 0,
            "liveness_motion_events": 0,
            "engine_user_id": _engine_user_id(client.client_id, user.external_user_id),
            "external_user_id": user.external_user_id,
            "client_user_id": str(user.id),
            "device_info": request_metadata.get("device_info", {}),
            "session_origin": request_metadata.get("origin", "http_api"),
        },
    )

    enrollment = FaceEnrollment.objects.create(
        client=client,
        client_user=user,
        enrollment_session=session,
        status="pending",
        embedding_vector="[]",
        embedding_dimension=settings.FACE_RECOGNITION_CONFIG.get("EMBEDDING_DIMENSION", 512),
        face_quality_score=0.0,
        liveness_score=0.0,
        anti_spoofing_score=0.0,
        sample_number=sample_number,
        total_samples=target_samples,
        metadata={
            "frames": [],
            "target_samples": target_samples,
            "session_token": session.session_token,
        },
    )

    session.metadata["enrollment_id"] = str(enrollment.id)
    session.save(update_fields=["metadata"])

    face_engine.reset_liveness_detector()

    # Generate WebSocket URL - check for proxy headers
    is_secure = request.is_secure() or request.META.get('HTTP_X_FORWARDED_PROTO', '').lower() == 'https'
    websocket_scheme = "wss" if is_secure else "ws"
    websocket_host = request.get_host()
    websocket_url = f"{websocket_scheme}://{websocket_host}/ws/auth/process-image/{session.session_token}/"

    response_payload = {
        "session_token": session.session_token,
        "enrollment_id": str(enrollment.id),
        "status": "pending",
        "target_samples": target_samples,
        "expires_at": session.expires_at,
        "websocket_url": websocket_url,
        "message": "Enrollment session created. Stream frames to continue.",
    }
    response_serializer = EnrollmentCreateResponseSerializer(response_payload)
    return Response(response_serializer.data, status=status.HTTP_201_CREATED)


# ---------------------------------------------------------------------------
# Authentication endpoints
# ---------------------------------------------------------------------------


@extend_schema(
    tags=["Face Authentication"],
    summary="Create Authentication Session",
    description="Initializes a session to authenticate a user. For verification, provide a `user_id`. For identification, omit the `user_id`. A `session_token` is returned for use in frame processing.",
    request=AuthenticationRequestSerializer,
    responses={
        201: AuthSessionCreateResponseSerializer,
        400: OpenApiResponse(response=ErrorResponseSerializer, description="Bad Request - Invalid input."),
        401: OpenApiResponse(response=ErrorResponseSerializer, description="Unauthorized - Invalid API key."),
    },
)
@api_view(["POST"])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def create_authentication_session(request):
    serializer = AuthenticationRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    client = request.client
    metadata = serializer.validated_data.get("metadata") or {}
    external_user_id = serializer.validated_data.get("user_id")

    session_type = "verification" if external_user_id else "identification"
    engine_target_id = None
    target_client_user = None
    if external_user_id:
        target_client_user = ClientUser.objects.get(
            client=client,
            external_user_id=external_user_id,
        )
        engine_target_id = _engine_user_id(client.client_id, target_client_user.external_user_id)

    session = AuthenticationSession.objects.create(
        client=client,
        client_user=target_client_user,  # Add the target client user to session
        session_type=session_type,
        status="active",
        ip_address=request.META.get("REMOTE_ADDR"),
        user_agent=request.META.get("HTTP_USER_AGENT", ""),
        metadata={
            "target_user_id": external_user_id,
            "engine_target_id": engine_target_id,
            "client_user_id": str(target_client_user.id) if target_client_user else None,
            "frames_processed": 0,
            "liveness_blinks": 0,
            "liveness_motion_events": 0,
            "min_frames_required": metadata.get("min_frames_required", MIN_LIVENESS_FRAMES),
            "required_blinks": metadata.get("required_blinks", MIN_LIVENESS_BLINKS),
            "require_liveness": serializer.validated_data.get("require_liveness", True),
            "device_info": metadata.get("device_info", {}),
            "session_origin": metadata.get("origin", "http_api"),
        },
    )

    face_engine.reset_liveness_detector()

    # Generate WebSocket URL - check for proxy headers
    is_secure = request.is_secure() or request.META.get('HTTP_X_FORWARDED_PROTO', '').lower() == 'https'
    websocket_scheme = "wss" if is_secure else "ws"
    websocket_host = request.get_host()
    websocket_url = f"{websocket_scheme}://{websocket_host}/ws/auth/process-image/{session.session_token}/"

    response_payload = {
        "session_token": session.session_token,
        "status": "active",
        "expires_at": session.expires_at,
        "session_type": session.session_type,
        "websocket_url": websocket_url,
        "message": "Authentication session created. Stream frames to continue.",
    }
    response_serializer = AuthSessionCreateResponseSerializer(response_payload)
    return Response(response_serializer.data, status=status.HTTP_201_CREATED)


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------


@extend_schema(
    tags=["Face Authentication", "Face Enrollment"],
    summary="Process Face Image Frame",
    description="Upload a single frame (image) for an active enrollment or authentication session. The server processes the frame for face detection, quality checks, and liveness, returning immediate feedback.",
    request=FaceImageUploadSerializer,
    responses={
        200: FaceFrameResponseSerializer,
        400: OpenApiResponse(response=ErrorResponseSerializer, description="Bad Request - Invalid image, expired session, or other processing error."),
        401: OpenApiResponse(response=ErrorResponseSerializer, description="Unauthorized - Invalid API key."),
        404: OpenApiResponse(response=ErrorResponseSerializer, description="Not Found - Session token not found."),
    },
)
@api_view(["POST"])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def process_face_image(request):
    serializer = FaceImageUploadSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    session = get_object_or_404(
        AuthenticationSession,
        session_token=serializer.validated_data["session_token"],
        client=request.client,
        status__in=["active", "processing"],
    )

    if session.expires_at and session.expires_at < timezone.now():
        session.status = "expired"
        session.save(update_fields=["status"])
        return Response(
            {
                "error": "Session expired",
                "session_token": session.session_token,
                "status": session.status,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        frame = _decode_image_to_frame(serializer.validated_data["image"])
    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    frame_number = serializer.validated_data.get("frame_number")
    timestamp = serializer.validated_data.get("timestamp")

    if session.session_type == "enrollment":
        payload, http_status = _handle_enrollment_frame(
            request.client,
            session,
            frame,
            frame_number,
            timestamp,
        )
    else:
        payload, http_status = _handle_authentication_frame(
            request.client,
            session,
            frame,
            frame_number,
            timestamp,
        )

    return Response(payload, status=http_status)


def _handle_enrollment_frame(
    client,
    session: AuthenticationSession,
    frame: np.ndarray,
    frame_number: Optional[int],
    timestamp: Optional[timezone.datetime],
) -> Tuple[Dict[str, Any], int]:
    """Handle enrollment frame with session-based processing"""
    metadata = session.metadata or {}
    enrollment_id = metadata.get("enrollment_id")
    if not enrollment_id:
        return (
            {
                "success": False,
                "error": "Enrollment session misconfigured (missing enrollment_id).",
                "session_status": session.status,
            },
            status.HTTP_400_BAD_REQUEST,
        )

    enrollment = get_object_or_404(FaceEnrollment, id=enrollment_id, client=client)
    client_user = enrollment.client_user
    engine_user_id = metadata.get("engine_user_id") or _engine_user_id(
        client.client_id, client_user.external_user_id
    )
    metadata["engine_user_id"] = engine_user_id

    # Process frame using session-based enrollment with embedding averaging
    current_frames_processed = int(metadata.get("frames_processed", 0))
    target_samples = int(metadata.get("target_samples", enrollment.total_samples))
    attempted_frame_number = current_frames_processed + 1
    
    logger.debug(f"Processing enrollment frame for engine_user_id: {engine_user_id}, session: {session.session_token}")
    result = face_engine.process_enrollment_frame_with_session(
        session_token=session.session_token,
        frame=frame,
        user_id=engine_user_id
    )
    
    if not result.get("success", False):
        error = result.get("error", "Unknown enrollment error")
        metadata["last_error"] = error
        metadata["last_updated"] = timezone.now().isoformat()
        metadata["rejected_frames"] = metadata.get("rejected_frames", 0) + 1
        session.metadata = _ensure_json_serializable_metadata(metadata)
        session.save(update_fields=["metadata"])
        # Log frame rejection with details
        logger.warning(f"❌ Frame {attempted_frame_number} rejected for enrollment - "
                      f"Error: {error}, "
                      f"Quality: {result.get('quality_score', 'N/A')}, "
                      f"Liveness: {result.get('liveness_score', 'N/A')}, "
                      f"Anti-spoofing: {result.get('anti_spoofing_score', 'N/A')}")
        
        return (
            {
                "success": False,
                "error": error,
                "session_status": session.status,
                "completed_samples": current_frames_processed,
                "attempted_frame": attempted_frame_number,
                "target_samples": target_samples,
                "requires_more_frames": True,
                "frame_rejected": not result.get("frame_accepted", False),
                "enrollment_progress": result.get("enrollment_progress", 0),
                "quality_score": result.get("quality_score"),
                "liveness_score": result.get("liveness_score"),
                "liveness_verified": result.get("liveness_verified", False),
                "anti_spoofing_score": result.get("anti_spoofing_score"),
                "obstacles_detected": result.get("obstacles", []),
                "obstacle_confidence": result.get("obstacle_confidence"),
                "rejection_reason": "Frame quality validation failed",
            },
            status.HTTP_200_OK,
        )

    # Frame was successfully processed using session-based enrollment
    frames_processed = result.get("embeddings_collected", current_frames_processed + 1)
    liveness_data = result.get("liveness_data", {})
    quality_score = result.get("quality_score", 0.0)
    liveness_satisfied = result.get("liveness_satisfied", False)
    progress = result.get("progress", {})
    can_complete = result.get("can_complete_enrollment", False)
    
    # Extract frame image for storage (generate from frame)
    face_snapshot = None
    bbox_list = None
    try:
        # Get a reasonable face region from the frame
        h, w = frame.shape[:2]
        # Simple center crop as fallback
        x1, y1 = int(w * 0.25), int(h * 0.25)
        x2, y2 = int(w * 0.75), int(h * 0.75)
        face_crop = frame[y1:y2, x1:x2]
        _, buffer = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        face_snapshot = buffer.tobytes()
        bbox_list = [x1, y1, x2, y2]
    except Exception as e:
        logger.error(f"Failed to extract face snapshot: {e}")
    
    snapshot_preview = _snapshot_to_data_url(face_snapshot)
    
    # Get liveness metrics from session result
    blink_count = int(liveness_data.get("blinks_detected", 0))
    motion_events = int(liveness_data.get("motion_events", 0))
    
    total_blinks = max(int(metadata.get("liveness_blinks", 0)), blink_count)
    total_motion_events = max(int(metadata.get("liveness_motion_events", 0)), motion_events)
    
    # Get liveness verification from result
    liveness_verified = liveness_satisfied
    liveness_score = quality_score  # Use quality score as liveness score for compatibility
    
    logger.debug(f"Enrollment frame {frames_processed}/{target_samples} - "
                f"Quality: {quality_score:.3f}, "
                f"Liveness: {liveness_score:.3f}, "
                f"Anti-spoofing: N/A, "
                f"Verified: {liveness_verified}")
    
    # Log frame acceptance
    logger.info(f"✅ Frame {frames_processed} accepted for enrollment - "
               f"Quality: {quality_score:.3f}, "
               f"Liveness: {liveness_score:.3f} (verified: {liveness_verified}), "
               f"Feedback: {result.get('feedback', 'N/A')}")

    metadata.update(
        {
            "frames_processed": frames_processed,
            "liveness_blinks": total_blinks,
            "liveness_motion_events": total_motion_events,
            "last_liveness": liveness_data,
            "last_bbox": bbox_list,
            "last_quality": quality_score,
            "last_liveness_score": liveness_score,
            "last_anti_spoofing_score": 0.0,  # Not used in session-based enrollment
            "last_updated": timezone.now().isoformat(),
            "last_error": "",
            "preview_image": snapshot_preview,
            "enrollment_progress": progress.get("progress_percentage", 0),
            "can_complete_enrollment": can_complete,
            "session_feedback": result.get("feedback", ""),
        }
    )
    if frame_number is not None:
        metadata["last_frame_number"] = frame_number
    if timestamp:
        metadata["last_timestamp"] = timestamp.isoformat()

    _append_history(
        metadata,
        "frames",
        {
            "index": frames_processed,
            "timestamp": metadata.get("last_timestamp"),
            "quality": quality_score,
            "liveness_score": liveness_score,
            "anti_spoofing_score": 0.0,  # Not used in session-based enrollment
            "blinks": blink_count,
            "motion_events": motion_events,
            "obstacles": result.get("obstacles", []),
            "session_feedback": result.get("feedback", ""),
        },
    )

    session.status = "processing"
    session.metadata = _ensure_json_serializable_metadata(metadata)
    session.save(update_fields=["status", "metadata"])

    # Note: embeddings are collected in session, final averaged embedding will be saved on completion
    enrollment.face_quality_score = quality_score
    enrollment.liveness_score = liveness_score
    enrollment.anti_spoofing_score = 0.0  # Not used in session-based enrollment
    enrollment.face_bbox = bbox_list
    
    # Save face image to storage (MinIO) if available
    face_snapshot = result.get("face_snapshot")
    if face_snapshot and not enrollment.face_image_path:
        saved_path = _save_face_image_to_storage(
            face_snapshot,
            client.client_id,
            client_user.external_user_id,
            enrollment.sample_number
        )
        if saved_path:
            enrollment.face_image_path = saved_path
            
        # Always update client user profile image with the latest/best quality frame
        if face_snapshot:
            from django.core.files.base import ContentFile
            try:
                # Delete old profile image if exists
                if client_user.profile_image:
                    try:
                        client_user.profile_image.delete(save=False)
                    except Exception as delete_error:
                        logger.warning(f"Could not delete old profile image: {delete_error}")
                
                filename = f"profile_{client_user.external_user_id}_{enrollment.sample_number}.jpg"
                client_user.profile_image.save(
                    filename,
                    ContentFile(face_snapshot),
                    save=False
                )
                client_user.save(update_fields=['profile_image'])
                logger.info(f"Updated profile image for client user {client_user.external_user_id}")
            except Exception as e:
                logger.error(f"Failed to update profile image for client user: {e}")
    
    enrollment.metadata = _ensure_json_serializable_metadata({
        **(enrollment.metadata or {}),
        "frames_processed": frames_processed,
        "last_liveness": liveness_data,
    })
    if enrollment.status == "pending":
        enrollment.status = "active"
    enrollment.save(update_fields=["embedding_vector", "face_quality_score", "liveness_score", "face_bbox", "face_image_path", "metadata", "status", "updated_at"])

    # Check if enrollment is complete based on session-based result
    enrollment_complete = can_complete
    min_frames_met = frames_processed >= target_samples
    
    # Use the liveness satisfaction from session result
    normalized_liveness = 1.0 if liveness_satisfied else 0.0
    min_liveness_threshold = 0.5  # Adjust as needed
    
    # Complete enrollment if conditions are met
    should_complete = enrollment_complete and normalized_liveness >= min_liveness_threshold
    
    if should_complete:
        # Complete enrollment using session-based averaging
        completion_result = face_engine.complete_enrollment_with_session(
            session_token=session.session_token,
            user_id=engine_user_id
        )
        
        if completion_result.get("success", False):
            logger.info(f"Session-based enrollment completed successfully: {completion_result}")
        else:
            # Check if failure is due to duplicate face
            if completion_result.get("error") == "duplicate_face":
                logger.error(f"Duplicate face detected during enrollment: {completion_result}")
                
                # Update session to failed
                session.status = "failed"
                session.is_successful = False
                session.failure_reason = "duplicate_face"
                session.completed_at = timezone.now()
                metadata["failure_reason"] = "duplicate_face"
                metadata["conflicting_user"] = completion_result.get("details", {}).get("conflicting_user_id", "unknown")
                metadata["similarity_score"] = completion_result.get("details", {}).get("similarity_score", 0.0)
                session.metadata = _ensure_json_serializable_metadata(metadata)
                session.save(update_fields=["status", "is_successful", "failure_reason", "completed_at", "metadata"])
                
                # Return error response to client
                return JsonResponse({
                    "success": False,
                    "error": "duplicate_face",
                    "error_code": "DUPLICATE_FACE_DETECTED",
                    "message": "❌ This face has already been enrolled by another user",
                    "details": completion_result.get("details", {}),
                    "session_status": "failed"
                }, status=status.HTTP_409_CONFLICT)
            
            logger.error(f"Failed to complete session-based enrollment: {completion_result}")
            # Don't fail the whole process for other errors, just continue with regular flow
    
    # More flexible completion logic - allow completion if enrollment ready even if liveness not fully satisfied
    should_complete_enrollment = (enrollment_complete and min_frames_met) or (frames_processed >= 20 and can_complete)
    
    if should_complete_enrollment:
        completion_ts = timezone.now()
        session.status = "completed"
        session.completed_at = completion_ts
        session.is_successful = True  # Set enrollment success flag
        session.confidence_score = result.get("quality_score", 0.0)  # Store quality as confidence
        metadata["final_status"] = "completed"
        metadata["liveness_verified"] = True
        session.metadata = _ensure_json_serializable_metadata(metadata)
        session.save(update_fields=["status", "completed_at", "is_successful", "confidence_score", "metadata"])

        enrollment.metadata = _ensure_json_serializable_metadata({
            **(enrollment.metadata or {}),
            "completed_at": completion_ts.isoformat()
        })
        enrollment.save(update_fields=["metadata", "updated_at"])

        client_user.is_enrolled = True
        client_user.enrollment_completed_at = completion_ts
        client_user.save(update_fields=["is_enrolled", "enrollment_completed_at"])
        
        # Log enrollment completion for the client user
        logger.info(f"Enrollment completed for client user: {client_user.external_user_id} "
                   f"(Client: {client.client_id}, User ID: {client_user.id})")

        # Embedding is already saved by the enroll_face method
        embedding_id = completion_result.get('embedding_id') if 'completion_result' in locals() else result.get('embedding_id')
        logger.info(f"Enrollment completed with embedding_id: {embedding_id} "
                   f"for client: {client.client_id}, external_user_id: {client_user.external_user_id}")
        
        # Send webhook notification for enrollment completion
        try:
            from webhooks.helpers import send_enrollment_completed_webhook
            send_enrollment_completed_webhook(client, client_user, session, enrollment)
        except Exception as e:
            logger.error(f"Failed to send enrollment webhook: {e}")
        
        # Track analytics for enrollment completion
        try:
            track_enrollment_metrics(client, enrollment, session)
        except Exception as e:
            logger.error(f"Failed to track enrollment analytics: {e}")

        response_payload = {
            "success": True,
            "session_status": session.status,
            "completed_samples": frames_processed,
            "target_samples": target_samples,
            "enrollment_progress": 100.0,
            "liveness_verified": True,
            "liveness_score": normalized_liveness,
            "quality_score": quality_score,
            "anti_spoofing_score": 0.0,  # Not used in session-based enrollment
            "preview_image": snapshot_preview,
            "requires_more_frames": False,
            "frame_accepted": True,
            "enrollment_complete": True,
            "session_feedback": result.get("feedback", ""),
            "message": "Enrollment completed successfully using averaged embeddings.",
        }
        return response_payload, status.HTTP_200_OK

    response_payload = {
        "success": True,
        "session_status": session.status,
        "completed_samples": frames_processed,
        "target_samples": target_samples,
        "enrollment_progress": progress.get("progress_percentage", 0),
        "can_complete_enrollment": can_complete,
        "liveness_verified": liveness_verified,
        "liveness_score": liveness_score,
        "quality_score": quality_score,
        "anti_spoofing_score": 0.0,  # Not used in session-based enrollment
        "preview_image": snapshot_preview,
        "requires_more_frames": not can_complete,
        "frame_accepted": True,
        "obstacles": result.get("obstacles", []),
        "session_feedback": result.get("feedback", ""),
        "message": result.get("feedback", "Frame processed successfully."),
    }
    return response_payload, status.HTTP_200_OK


def _handle_authentication_frame(
    client,
    session: AuthenticationSession,
    frame: np.ndarray,
    frame_number: Optional[int],
    timestamp: Optional[timezone.datetime],
) -> Tuple[Dict[str, Any], int]:
    metadata = session.metadata or {}

    engine_target_id = metadata.get("engine_target_id")
    if metadata.get("target_user_id") and not engine_target_id:
        engine_target_id = _engine_user_id(client.client_id, metadata["target_user_id"])
        metadata["engine_target_id"] = engine_target_id

    # Use session-based authentication with session manager
    from core.session_manager import SessionManager
    session_manager = SessionManager()
    session_id = session.session_token
    
    # Initialize session if needed
    if not session_manager.get_session_data(session_id):
        session_manager.initialize_session(session_id, 'authentication')
    
    logger.debug(f"Starting authentication for session {session_id}, target_id: {engine_target_id}")
    auth_result = face_engine.authenticate_user_with_session(
        session_id, frame, engine_target_id
    )
    
    # Log authentication result details for debugging
    logger.debug(f"Auth result - Success: {auth_result.get('success')}, "
                f"Similarity: {auth_result.get('similarity_score', 0.0)}, "
                f"Quality: {auth_result.get('quality_score', 0.0)}, "
                f"User ID: {auth_result.get('user_id')}, "
                f"Error: {auth_result.get('error')}")
    
    # Enhanced obstacle detection - reject frame if obstacles detected
    obstacles = auth_result.get("obstacles", [])
    if obstacles:
        obstacle_message = f"Authentication frame rejected due to obstacles: {', '.join(obstacles)}"
        metadata["last_error"] = obstacle_message
        metadata["last_updated"] = timezone.now().isoformat()
        metadata["rejected_frames"] = metadata.get("rejected_frames", 0) + 1
        session.metadata = _ensure_json_serializable_metadata(metadata)
        session.save(update_fields=["metadata"])
        
        return (
            {
                "success": False,
                "error": obstacle_message,
                "obstacles_detected": obstacles,
                "obstacle_confidence": auth_result.get("obstacle_confidence", {}),
                "session_status": session.status,
                "frames_processed": int(metadata.get("frames_processed", 0)),
                "requires_more_frames": True,
                "frame_rejected": True,
            },
            status.HTTP_200_OK,
        )
    
    # Get session-based liveness and frame data
    liveness_data = auth_result.get("liveness_data") or {}
    frames_processed = int(metadata.get("frames_processed", 0)) + 1
    
    # Check frame limit - stop after 3 frames if no success
    MAX_AUTH_FRAMES = 8
    if frames_processed > MAX_AUTH_FRAMES and not auth_result.get("success"):
        # Stop authentication after max frames without success
        session.status = "failed"
        session.completed_at = timezone.now()
        session.is_successful = False
        metadata["final_status"] = "failed"
        metadata["failure_reason"] = f"Authentication failed after {MAX_AUTH_FRAMES} frames - face not recognized clearly"
        metadata["frames_processed"] = frames_processed
        session.metadata = _ensure_json_serializable_metadata(metadata)
        session.save(update_fields=["status", "completed_at", "is_successful", "metadata"])
        
        return (
            {
                "success": False,
                "error": f"Wajah tidak dapat dikenali setelah {MAX_AUTH_FRAMES} percobaan",
                "message": "Pastikan wajah terlihat jelas dan pencahayaan cukup, kemudian coba lagi",
                "session_status": "failed",
                "session_finalized": True,
                "frames_processed": frames_processed,
                "max_frames_reached": True,
                "similarity_score": auth_result.get("similarity_score", 0.0),
                "quality_score": auth_result.get("quality_score", 0.0),
            },
            status.HTTP_400_BAD_REQUEST,
        )
    
    # Get cumulative session data from session manager
    session_data = session_manager.get_session_data(session_id) or {}
    liveness_detector_data = session_data.get("liveness_detector", {})
    
    blink_count = liveness_detector_data.get("total_blinks", 0)
    motion_events = liveness_detector_data.get("motion_events", 0)
    
    # Use session values as they maintain state across frames
    total_blinks = blink_count
    total_motion_events = motion_events
    
    # Enhanced liveness verification using session-based data
    overall_liveness_verified = auth_result.get("liveness_verified", False)
    liveness_reason = auth_result.get("liveness_reason", "session_based")

    # Update metadata with session-based values
    quality_score = float(auth_result.get("quality_score", 0.0))
    similarity_score = float(auth_result.get("similarity_score", 0.0))
    min_frames_required = int(metadata.get("min_frames_required", MIN_LIVENESS_FRAMES))
    
    # Debug logging for authentication flow
    logger.debug(f"Auth flow debug - Success: {auth_result.get('success')}, "
                f"Frames: {frames_processed}/{min_frames_required}, "
                f"Liveness verified: {overall_liveness_verified}, "
                f"Similarity: {similarity_score}, Quality: {quality_score}")
    
    metadata.update({
        "frames_processed": frames_processed,
        "liveness_blinks": total_blinks,
        "liveness_motion_events": total_motion_events,
        "last_liveness": liveness_data,
        "last_quality": quality_score,
        "last_similarity": similarity_score,
        "last_error": auth_result.get("error"),
        "last_updated": timezone.now().isoformat(),
        "match_fallback_used": False,  # Not used in session-based auth
        "obstacles": auth_result.get("obstacles", []),
        "session_feedback": auth_result.get("feedback", ""),
    })
    if frame_number is not None:
        metadata["last_frame_number"] = frame_number
    if timestamp:
        metadata["last_timestamp"] = timestamp.isoformat()

    _append_history(
        metadata,
        "frames",
        {
            "index": frames_processed,
            "timestamp": metadata.get("last_timestamp"),
            "similarity": similarity_score,
            "quality": quality_score,
            "blinks": blink_count,
            "motion_events": motion_events,
            "error": auth_result.get("error"),
            "session_feedback": auth_result.get("feedback", ""),
        },
    )

    # Use session-based liveness score
    liveness_verified = overall_liveness_verified
    normalized_liveness = float(auth_result.get("liveness_score", 0.0))
    metadata["liveness_verified"] = liveness_verified
    
    logger.debug(f"Auth liveness check - Blinks: {total_blinks}, "
                f"Motion: {total_motion_events}, "
                f"Verified: {overall_liveness_verified} (reason: {liveness_reason})")

    session.status = "processing"
    session.metadata = _ensure_json_serializable_metadata(metadata)
    session.save(update_fields=["status", "metadata"])

    min_frames_met = frames_processed >= min_frames_required

    # Continue if face detected but liveness not yet satisfied
    if auth_result.get("success") and not (min_frames_met and liveness_verified):
        payload = {
            "success": False,
            "requires_more_frames": True,
            "message": auth_result.get("feedback", "Liveness requirement belum terpenuhi. Lanjutkan streaming dengan kedipan atau gerakan lembut."),
            "frames_processed": frames_processed,
            "min_frames_required": min_frames_required,
            "liveness_blinks": total_blinks,
            "liveness_motion_events": total_motion_events,
            "liveness_score": normalized_liveness,
            "liveness_verified": liveness_verified,
            "session_status": session.status,
            "session_feedback": auth_result.get("feedback", ""),
        }
        return payload, status.HTTP_200_OK
    
    # Continue processing if we haven't met minimum frames yet, unless it's a critical error
    if not auth_result.get("success") and frames_processed < min_frames_required:
        error_msg = auth_result.get("error", "")
        # Only fail early for critical errors, otherwise continue
        if any(critical in error_msg.lower() for critical in ["multiple faces", "no face", "too small", "too large"]):
            # Allow a few frames before giving up on face detection
            if frames_processed < 3:
                payload = {
                    "success": False,
                    "requires_more_frames": True,
                    "message": f"Frame {frames_processed}: {error_msg}. Posisikan wajah dengan benar.",
                    "frames_processed": frames_processed,
                    "min_frames_required": min_frames_required,
                    "liveness_score": normalized_liveness,
                    "session_status": session.status,
                    "session_feedback": auth_result.get("feedback", ""),
                }
                return payload, status.HTTP_200_OK

    error_message = auth_result.get("error") or ""
    if (
        not auth_result.get("success")
        and "liveness" in error_message.lower()
        and not (min_frames_met and liveness_verified)
    ):
        payload = {
            "success": False,
            "requires_more_frames": True,
            "message": auth_result.get("feedback", "Liveness belum tervalidasi. Lanjutkan dengan kedipan atau gerakan halus."),
            "frames_processed": frames_processed,
            "min_frames_required": min_frames_required,
            "liveness_blinks": total_blinks,
            "liveness_motion_events": total_motion_events,
            "liveness_score": normalized_liveness,
            "liveness_verified": liveness_verified,
            "session_status": session.status,
            "session_feedback": auth_result.get("feedback", ""),
        }
        return payload, status.HTTP_200_OK

    # Handle case where authentication is successful but requirements not fully met
    if auth_result.get("success") and not (min_frames_met and liveness_verified):
        # If we've reached max frames and face is recognized, allow success with relaxed liveness
        if frames_processed >= MAX_AUTH_FRAMES:
            logger.info(f"Authentication success with relaxed liveness after {frames_processed} frames")
            # Proceed to final success even without full liveness verification
        else:
            # Continue asking for more frames if within limit
            payload = {
                "success": False,
                "requires_more_frames": True,
                "message": f"Wajah dikenali! Lanjutkan untuk verifikasi liveness (Frame: {frames_processed}/{min_frames_required}, Liveness: {'✓' if liveness_verified else '✗'})",
                "frames_processed": frames_processed,
                "min_frames_required": min_frames_required,
                "liveness_blinks": total_blinks,
                "liveness_motion_events": total_motion_events,
                "liveness_score": normalized_liveness,
                "liveness_verified": liveness_verified,
                "similarity_score": auth_result.get("similarity_score", 0.0),
                "quality_score": auth_result.get("quality_score", 0.0),
                "session_status": session.status,
                "session_feedback": f"Wajah dikenali! Frames: {frames_processed}/{min_frames_required}, Liveness: {'✓' if liveness_verified else '✗'}",
                "face_recognized": True,
                "preliminary_match": {
                    "user_id": auth_result.get("user_id"),
                    "similarity": auth_result.get("similarity_score", 0.0)
                }
            }
            return payload, status.HTTP_200_OK

    # Only finalize authentication if we have enough frames AND liveness is verified
    # OR if we've reached max frames and face is successfully recognized (relaxed liveness)
    relaxed_liveness = (frames_processed >= MAX_AUTH_FRAMES and auth_result.get("success"))
    if auth_result.get("success") and (min_frames_met and liveness_verified) or relaxed_liveness:
        engine_user_id = auth_result.get("user_id")
        if relaxed_liveness:
            logger.info(f"Authentication successful with relaxed liveness after {frames_processed} frames, raw engine_user_id: {engine_user_id}")
        else:
            logger.info(f"Authentication successful and requirements met, raw engine_user_id: {engine_user_id}")
        
        # Handle different engine response formats
        matched_user = None
        if engine_user_id:
            # Try direct resolution first (for proper client:external_user_id format)
            matched_user = _resolve_engine_user(client, engine_user_id)
            
            # If not found, try to construct proper format
            if not matched_user and ":" not in engine_user_id:
                # Engine returned just external_user_id, construct proper format
                constructed_engine_id = _engine_user_id(client.client_id, engine_user_id)
                logger.debug(f"Trying constructed engine_id: {constructed_engine_id}")
                matched_user = _resolve_engine_user(client, constructed_engine_id)
                
                # If still not found, try finding by external_user_id directly
                if not matched_user:
                    try:
                        matched_user = ClientUser.objects.get(
                            client=client, 
                            external_user_id=engine_user_id
                        )
                        logger.info(f"Found user by direct external_user_id lookup: {matched_user.external_user_id}")
                    except ClientUser.DoesNotExist:
                        logger.warning(f"No ClientUser found with external_user_id: {engine_user_id}")
        
        if matched_user:
            logger.info(f"Matched user resolved: {matched_user.external_user_id} (ID: {matched_user.id})")
        else:
            logger.warning(f"Could not resolve user for engine_user_id: {engine_user_id}")
            
            # In identification mode, try to find any enrolled user that might match
            if session.session_type == "identification":
                # Look for any enrolled user with active enrollment
                enrolled_users = ClientUser.objects.filter(
                    client=client,
                    is_enrolled=True,
                    enrollments__status="active"
                ).distinct()
                
                if enrolled_users.count() == 1:
                    matched_user = enrolled_users.first()
                    logger.info(f"Identification mode: Using single enrolled user {matched_user.external_user_id}")
                elif enrolled_users.count() > 1:
                    logger.warning(f"Identification mode: Multiple enrolled users found, cannot determine match")
                else:
                    logger.warning(f"Identification mode: No enrolled users found")
            
        matched_enrollment = (
            FaceEnrollment.objects.filter(client=client, client_user=matched_user, status="active")
            .order_by("-updated_at")
            .first()
            if matched_user
            else None
        )

        with transaction.atomic():
            attempt = FaceRecognitionAttempt.objects.create(
                client=client,
                session=session,
                result="success",
                matched_user=matched_user,
                matched_enrollment=matched_enrollment,
                similarity_score=float(auth_result.get("similarity_score", 0.0)),
                confidence_score=float(auth_result.get("similarity_score", 0.0)),
                face_quality_score=float(auth_result.get("quality_score", 0.0)),
                liveness_score=float(normalized_liveness),
                anti_spoofing_score=float(auth_result.get("match_data", {}).get("anti_spoofing_score", 0.0)),
                submitted_embedding="[]",
                face_bbox=_ensure_json_serializable_metadata(auth_result.get("bbox")),
                metadata=_ensure_json_serializable_metadata({
                    "match_data": auth_result.get("match_data"),
                    "obstacles": auth_result.get("obstacles"),
                    "obstacle_confidence": auth_result.get("obstacle_confidence", {}),
                    "frames_processed": frames_processed,
                }),
                ip_address=session.ip_address,
                user_agent=session.user_agent,
            )

            # Prepare liveness detection metadata with JSON serialization
            liveness_metadata = _ensure_json_serializable_metadata({
                "total_blinks": total_blinks,
                "total_motion_events": total_motion_events,
                "raw": liveness_data,
            })
            
            LivenessDetectionResult.objects.create(
                client=client,
                session=session,
                status="live",
                confidence_score=float(normalized_liveness),
                methods_used=["blink_detection", "motion_detection"],
                blink_detected=bool(liveness_data.get("blink_verified", False)),
                motion_detected=bool(liveness_data.get("motion_verified", False) or total_motion_events > 0),
                texture_score=float(liveness_data.get("texture_score", 0.0)) if liveness_data.get("texture_score") is not None else None,
                face_movements=_ensure_json_serializable_metadata(liveness_data.get("face_movements") or []),
                frames_analyzed=int(frames_processed),
                metadata=liveness_metadata,
            )

        session.status = "completed"
        session.completed_at = timezone.now()
        session.is_successful = True  # Fix: Set authentication success flag
        session.confidence_score = auth_result.get("similarity_score", 0.0)  # Store confidence in session
        
        # Update client_user field if identification was successful and we have a matched user
        if session.session_type == "identification" and matched_user and not session.client_user:
            session.client_user = matched_user
            logger.info(f"Updated session client_user to {matched_user.external_user_id} for identification session")
        
        metadata["final_status"] = "success"
        metadata["recognized_user_id"] = str(matched_user.id) if matched_user else None
        session.metadata = _ensure_json_serializable_metadata(metadata)
        session.save(update_fields=["status", "completed_at", "is_successful", "confidence_score", "client_user", "metadata"])

        # Extract variables needed for audit and risk assessment
        match_fallback_used = auth_result.get("match_fallback_used", False)
        obstacles = auth_result.get("obstacles") or []
        
        # Values already converted to float in metadata update above

        # Audit trail entry
        try:
            risk_factors = []
            if match_fallback_used:
                risk_factors.append("match_fallback")
            if obstacles:
                risk_factors.append("obstacles_detected")
            if normalized_liveness < 0.6:
                risk_factors.append("low_liveness")

            AuthenticationLog.objects.create(
                client=client,
                attempted_email=(matched_user.profile or {}).get("email") if matched_user else metadata.get("target_user_id"),
                auth_method="face",
                success=True,
                similarity_score=similarity_score,
                liveness_score=normalized_liveness,
                quality_score=quality_score,
                response_time=getattr(attempt, "processing_time_ms", None),
                ip_address=session.ip_address,
                user_agent=session.user_agent,
                device_fingerprint=session.device_fingerprint or metadata.get("device_fingerprint", ""),
                location=session.metadata.get("location", ""),
                risk_score=max(0.0, 1.0 - similarity_score),
                risk_factors=risk_factors,
                session_id=session.session_token,
            )
        except Exception:
            logger.exception("Failed to persist authentication log entry")

        # Update aggregated recognition stats
        try:
            update_face_recognition_stats(client, attempt)
        except Exception:
            logger.exception("Failed to update recognition stats for client %s", client.client_id)

        if obstacles:
            alert_payload = {
                "obstacles": obstacles,
                "session_token": session.session_token,
                "similarity_score": similarity_score,
            }
            try:
                SecurityAlert.objects.create(
                    client=client,
                    alert_type="quality_degradation",
                    severity="medium",
                    title="Obstacles detected during authentication",
                    description=f"Detected obstacles: {', '.join(obstacles)}",
                    context_data=alert_payload,
                    ip_address=session.ip_address,
                )
            except Exception:
                logger.exception("Failed to create security alert for obstacles")
            try:
                track_security_event(client, session, "obstacles_detected", alert_payload)
            except Exception:
                logger.exception("Failed to track security event for obstacles")

        # Session-based authentication - liveness data is already properly formatted
        
        payload = {
            "success": True,
            "session_status": session.status,
            "frames_processed": frames_processed,
            "similarity_score": similarity_score,
            "quality_score": quality_score,
            "liveness_score": normalized_liveness,
            "liveness_verified": liveness_verified,
            "match_fallback_used": False,  # Not used in session-based auth
            "requires_more_frames": False,
            "session_token": session.session_token,
            "session_feedback": auth_result.get("feedback", ""),
            "message": "Authentication successful",
            "authentication_metadata": {
                "algorithm_used": "session_based",
                "confidence_level": "high" if similarity_score >= 0.8 else "medium" if similarity_score >= 0.6 else "low",
                "liveness_method": "session_persistent_detector"
            }
        }
        
        if matched_user:
            payload["matched_user"] = {
                "id": str(matched_user.id),
                "external_user_id": matched_user.external_user_id,
                "display_name": matched_user.display_name,
                "is_enrolled": matched_user.is_enrolled,
            }
            # Include client user ID for client response
            payload["client_user_id"] = str(matched_user.id)
        else:
            # If no matched user but authentication succeeded, it means identification mode
            logger.warning(f"Authentication succeeded but no matched_user found for session {session.session_token}")
            payload["matched_user"] = None
            payload["client_user_id"] = None
        
        logger.info(f"Authentication success for session {session.session_token}, user: {matched_user.external_user_id if matched_user else 'Unknown'}, similarity: {similarity_score}")
        
        # Send webhook notification for successful authentication
        if matched_user:
            try:
                from webhooks.helpers import send_authentication_success_webhook
                send_authentication_success_webhook(client, matched_user, session, similarity_score, auth_result)
            except Exception as e:
                logger.error(f"Failed to send authentication success webhook: {e}")
        
        # Track analytics for successful authentication
        try:
            track_authentication_metrics(client, session, success=True, similarity_score=similarity_score)
        except Exception as e:
            logger.error(f"Failed to track authentication analytics: {e}")
        
        return payload, status.HTTP_200_OK

    # Check if we should continue with more frames instead of final failure
    if frames_processed < min_frames_required or not liveness_verified:
        # If we haven't reached minimum frames or liveness not verified, continue
        continue_message = f"Memproses frame {frames_processed}/{min_frames_required}"
        if not liveness_verified:
            continue_message += " - Kedipkan mata atau gerakkan kepala sedikit"
        
        payload = {
            "success": False,
            "requires_more_frames": True,
            "message": continue_message,
            "frames_processed": frames_processed,
            "min_frames_required": min_frames_required,
            "liveness_blinks": total_blinks,
            "liveness_motion_events": total_motion_events,
            "liveness_score": normalized_liveness,
            "liveness_verified": liveness_verified,
            "similarity_score": auth_result.get("similarity_score", 0.0),
            "quality_score": auth_result.get("quality_score", 0.0),
            "session_status": session.status,
            "session_feedback": continue_message,
            "error": auth_result.get("error", "Processing..."),
        }
        return payload, status.HTTP_200_OK

    # Final failure - only if we've tried enough frames
    failure_reason = error_message or "Authentication failed"
    attempt_result = "failed"
    lower_reason = failure_reason.lower()
    if "no matching" in lower_reason:
        attempt_result = "no_match"
    elif "quality" in lower_reason:
        attempt_result = "quality_too_low"
    elif "liveness" in lower_reason:
        attempt_result = "liveness_failed"

    matched_user = _resolve_engine_user(client, auth_result.get("user_id"))
    with transaction.atomic():
        attempt = FaceRecognitionAttempt.objects.create(
            client=client,
            session=session,
            result=attempt_result,
            matched_user=matched_user,
            matched_enrollment=None,
            similarity_score=auth_result.get("similarity_score", 0.0),
            confidence_score=auth_result.get("similarity_score", 0.0),
            face_quality_score=auth_result.get("quality_score", 0.0),
            liveness_score=normalized_liveness,
            anti_spoofing_score=auth_result.get("match_data", {}).get("anti_spoofing_score", 0.0),
            submitted_embedding="[]",
            face_bbox=auth_result.get("bbox"),
            metadata={
                "error": failure_reason,
                "obstacles": auth_result.get("obstacles"),
                "obstacle_confidence": auth_result.get("obstacle_confidence", {}),
                "frames_processed": frames_processed,
            },
            ip_address=session.ip_address,
            user_agent=session.user_agent,
        )

        LivenessDetectionResult.objects.create(
            client=client,
            session=session,
            status="uncertain" if "liveness" in lower_reason else "spoof",
            confidence_score=normalized_liveness,
            methods_used=["blink_detection", "motion_detection"],
            blink_detected=liveness_data.get("blink_verified", False),
            motion_detected=liveness_data.get("motion_verified", False) or total_motion_events > 0,
            texture_score=liveness_data.get("texture_score"),
            face_movements=liveness_data.get("face_movements") or [],
            frames_analyzed=frames_processed,
            metadata={
                "total_blinks": total_blinks,
                "total_motion_events": total_motion_events,
                "raw": liveness_data,
                "failure_reason": failure_reason,
            },
        )

    try:
        failure_similarity = auth_result.get("similarity_score", 0.0) or 0.0
        risk_factors = []
        if attempt_result == "no_match":
            risk_factors.append("face_not_recognized")
        if attempt_result == "liveness_failed":
            risk_factors.append("liveness_failed")
        if attempt_result == "quality_too_low":
            risk_factors.append("poor_quality")
        if auth_result.get("obstacles"):
            risk_factors.append("obstacles_detected")

        AuthenticationLog.objects.create(
            client=client,
            attempted_email=(matched_user.profile or {}).get("email") if matched_user else metadata.get("target_user_id"),
            auth_method="face",
            success=False,
            failure_reason=attempt_result,
            similarity_score=failure_similarity,
            liveness_score=normalized_liveness,
            quality_score=auth_result.get("quality_score", 0.0) or 0.0,
            response_time=getattr(attempt, "processing_time_ms", None),
            ip_address=session.ip_address,
            user_agent=session.user_agent,
            device_fingerprint=session.device_fingerprint or metadata.get("device_fingerprint", ""),
            location=session.metadata.get("location", ""),
            risk_score=min(1.0, 1.0 - failure_similarity if failure_similarity else 0.9),
            risk_factors=risk_factors,
            session_id=session.session_token,
        )
    except Exception:
        logger.exception("Failed to persist failed authentication log entry")

    try:
        update_face_recognition_stats(client, attempt)
    except Exception:
        logger.exception("Failed to update recognition stats after failure for client %s", client.client_id)

    alert_context = {
        "failure_reason": attempt_result,
        "session_token": session.session_token,
        "similarity_score": auth_result.get("similarity_score"),
        "obstacles": auth_result.get("obstacles"),
    }

    alert_severity = "high" if attempt_result in {"liveness_failed", "spoofing_detected"} else "medium"
    try:
        SecurityAlert.objects.create(
            client=client,
            alert_type="failed_attempts",
            severity=alert_severity,
            title="Authentication attempt failed",
            description=f"Failure reason: {attempt_result}",
            context_data=alert_context,
            ip_address=session.ip_address,
        )
    except Exception:
        logger.exception("Failed to create security alert for failed authentication")

    session.status = "failed"
    session.completed_at = timezone.now()
    session.is_successful = False  # Fix: Set authentication failure flag
    metadata["final_status"] = "failed"
    metadata["failure_reason"] = failure_reason
    session.metadata = _ensure_json_serializable_metadata(metadata)
    session.save(update_fields=["status", "completed_at", "is_successful", "metadata"])
    
    # Send webhook notification for failed authentication
    try:
        from webhooks.helpers import send_authentication_failed_webhook
        send_authentication_failed_webhook(client, session, failure_reason, auth_result)
    except Exception as e:
        logger.error(f"Failed to send authentication failed webhook: {e}")
    
    # Track analytics for failed authentication
    try:
        track_authentication_metrics(client, session, success=False, similarity_score=auth_result.get('similarity_score', 0.0))
    except Exception as e:
        logger.error(f"Failed to track authentication analytics: {e}")

    payload = {
        "success": False,
        "session_status": session.status,
        "error": failure_reason,
        "frames_processed": frames_processed,
        "liveness_score": normalized_liveness,
        "liveness_data": liveness_data,
        "requires_more_frames": False,
        "match_fallback_used": auth_result.get("match_fallback_used", False),
    }
    return payload, status.HTTP_200_OK


# ---------------------------------------------------------------------------
# Session status & analytics
# ---------------------------------------------------------------------------


@api_view(["GET"])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_session_status(request, session_token):
    session = get_object_or_404(
        AuthenticationSession,
        session_token=session_token,
        client=request.client,
    )
    payload = {
        "session_token": session.session_token,
        "status": session.status,
        "session_type": session.session_type,
        "created_at": session.created_at,
        "expires_at": session.expires_at,
        "completed_at": session.completed_at,
        "metadata": session.metadata,
    }
    if session.status in {"completed", "failed"}:
        payload["result"] = session.metadata.get("final_status")
    serializer = SessionStatusSerializer(payload)
    return Response(serializer.data)


@api_view(["GET"])
@authentication_classes([APIKeyAuthentication, JWTClientAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_client_analytics(request):
    client = request.client
    end_date = timezone.now()
    start_date = end_date - timedelta(days=30)

    if request.GET.get("start_date"):
        start_date = timezone.datetime.fromisoformat(request.GET["start_date"])
    if request.GET.get("end_date"):
        end_date = timezone.datetime.fromisoformat(request.GET["end_date"])

    enrollments_stats = client.enrollments.filter(
        created_at__range=[start_date, end_date]
    ).aggregate(
        total=Count("id"),
        active=Count("id", filter=Q(status="active")),
        pending=Count("id", filter=Q(status="pending")),
    )

    attempts_stats = client.recognition_attempts.filter(
        created_at__range=[start_date, end_date]
    ).aggregate(
        total=Count("id"),
        success=Count("id", filter=Q(result="success")),
        failed=Count("id", filter=~Q(result="success")),
        avg_similarity=Avg("similarity_score"),
    )

    sessions_stats = client.auth_sessions.filter(
        created_at__range=[start_date, end_date]
    ).aggregate(
        total=Count("id"),
        completed=Count("id", filter=Q(status="completed")),
        failed=Count("id", filter=Q(status="failed")),
    )

    payload = {
        "period": {
            "start": start_date,
            "end": end_date,
        },
        "enrollments": enrollments_stats,
        "authentication_attempts": attempts_stats,
        "sessions": sessions_stats,
    }
    return Response(payload)

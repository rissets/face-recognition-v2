"""
WebSocket consumers for real-time face recognition
"""

import asyncio
import base64
import json
import logging
import time

import cv2
import numpy as np
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from core.face_recognition_engine import FaceRecognitionEngine

from streaming.models import StreamingSession, WebRTCSignal
from django.conf import settings

logger = logging.getLogger("streaming")


class FaceRecognitionConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time face recognition"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_token = None
        self.streaming_session = None
        self.face_engine = FaceRecognitionEngine()
        self.room_group_name = None
        self._last_frame_ts = 0.0
        limits = getattr(settings, 'FACE_STREAMING_LIMITS', {})
        self._throttle_interval = 0.09  # ~11 fps max (can derive from MAX_WS_FPS)
        max_fps = limits.get('MAX_WS_FPS')
        if max_fps and max_fps > 0:
            self._throttle_interval = max(0.001, 1.0 / float(max_fps))
        self._consecutive_low_quality = 0
        self._max_low_quality = limits.get('MAX_LOW_QUALITY_CONSECUTIVE', 25)
        self._fail_low_quality_threshold = limits.get('FAIL_LOW_QUALITY_THRESHOLD', 0.30)
        self._auth_frame_budget = limits.get('AUTH_FRAME_BUDGET', 120)
        self._frames_processed_ws = 0

    async def connect(self):
        """Accept WebSocket connection"""
        self.session_token = self.scope["url_route"]["kwargs"]["session_token"]
        self.room_group_name = f"face_recognition_{self.session_token}"

        # Validate session
        try:
            self.streaming_session = await self.get_streaming_session(self.session_token)
            if not self.streaming_session:
                await self.close(code=4404)
                return
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            await self.close(code=4500)
            return

        # Security: ensure authenticated user matches session owner unless origin is public_login
        origin_flag = (self.streaming_session.session_data or {}).get('session_origin') or (self.streaming_session.session_data or {}).get('origin')
        user_mismatch = (
            self.scope.get('user')
            and self.scope['user'].is_authenticated
            and self.streaming_session.user
            and self.scope['user'] != self.streaming_session.user
        )
        anonymous_disallowed = (
            (not getattr(self.scope.get('user'), 'is_authenticated', False))
            and origin_flag not in ['public_login', 'webrtc_public_auth']
        )
        if user_mismatch or anonymous_disallowed:
            await self.close(code=4403)
            return

        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

        # Update session status
        await self.update_session_status("connected")

        # Send connection confirmation
        await self.send(
            text_data=json.dumps(
                {
                    "type": "connection_established",
                    "session_token": self.session_token,
                    "session_type": self.streaming_session.session_type,
                }
            )
        )

        logger.info(f"WebSocket connected for session {self.session_token}")

    async def disconnect(self, close_code):
        """Handle WebSocket disconnect"""
        if self.room_group_name:
            await self.channel_layer.group_discard(
                self.room_group_name, self.channel_name
            )

        if self.streaming_session:
            await self.update_session_status("disconnected")

        logger.info(f"WebSocket disconnected for session {self.session_token}")

    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get("type")

            if message_type == "frame_data":
                await self.handle_frame_data(data)
            elif message_type == "webrtc_signal":
                await self.handle_webrtc_signal(data)
            elif message_type == "session_command":
                await self.handle_session_command(data)
            else:
                await self.send_error(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            await self.send_error("Invalid JSON data")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_error(f"Processing error: {str(e)}")

    async def handle_frame_data(self, data):
        """Process frame data for face recognition"""
        try:
            now = asyncio.get_event_loop().time()
            # Throttle
            if (now - self._last_frame_ts) < self._throttle_interval:
                return  # silently drop to lower load
            self._last_frame_ts = now

            frame_data = data.get("frame_data")
            if not frame_data:
                await self.send_error("No frame data provided", code="NO_FRAME")
                return

            # Update session status
            await self.update_session_status("processing")

            # Decode frame
            frame = await self.decode_frame_data(frame_data)
            if frame is None:
                await self.send_error("Could not decode frame data")
                return

            # Process based on session type
            if self.streaming_session.session_type == "enrollment":
                result = await self.process_enrollment_frame(frame)
            elif self.streaming_session.session_type in ["authentication", "verification", "identification"]:
                result = await self.process_authentication_frame(frame)
            else:
                await self.send_error("Unknown session type", code="BAD_TYPE")
                return

            self._frames_processed_ws += 1

            # Unified quality score extraction
            quality_score = result.get('quality_score')
            if quality_score is not None:
                if quality_score < self._fail_low_quality_threshold:
                    self._consecutive_low_quality += 1
                else:
                    self._consecutive_low_quality = 0

                if self._consecutive_low_quality >= self._max_low_quality:
                    # Early abort for persistently low quality
                    await self.finalize_streaming_session('failed')
                    await self.send(json.dumps({
                        'type': 'session_final',
                        'result': result,
                        'frames_processed': self._frames_processed_ws,
                        'error': 'Terminated due to consistently low image quality',
                        'reason': 'low_quality_abort'
                    }))
                    return

            # Finalization detection for auth
            if self.streaming_session.session_type != 'enrollment':
                if result.get('success'):
                    await self.finalize_streaming_session('completed')
                    final_payload = {
                        'type': 'session_final',
                        'result': result,
                        'frames_processed': self._frames_processed_ws,
                        'reason': 'authenticated'
                    }
                    # Public login JWT issuance
                    origin_flag = (self.streaming_session.session_data or {}).get('session_origin') or (self.streaming_session.session_data or {}).get('origin')
                    if origin_flag in ['public_login', 'webrtc_public_auth'] and self.streaming_session.user:
                        try:
                            from rest_framework_simplejwt.tokens import RefreshToken
                            refresh = RefreshToken.for_user(self.streaming_session.user)
                            final_payload['access'] = str(refresh.access_token)
                            final_payload['refresh'] = str(refresh)
                        except Exception as e:
                            final_payload['jwt_issue_error'] = str(e)
                    await self.send(json.dumps(final_payload))
                    return
                # Fail fast if frame budget exceeded
                if self._frames_processed_ws > self._auth_frame_budget and not result.get('success'):
                    await self.finalize_streaming_session('failed')
                    await self.send(json.dumps({
                        'type': 'session_final',
                        'result': result,
                        'frames_processed': self._frames_processed_ws,
                        'error': 'Authentication not achieved within frame budget',
                        'reason': 'frame_budget_exceeded'
                    }))
                    return

            # Send result
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "frame_result",
                        "result": result,
                        "timestamp": data.get("timestamp"),
                        "frames_processed": self._frames_processed_ws,
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error handling frame data: {e}")
            await self.send_error(f"Frame processing error: {str(e)}")

    async def handle_webrtc_signal(self, data):
        """Handle WebRTC signaling"""
        try:
            signal_type = data.get("signal_type")
            signal_data = data.get("signal_data")

            if not signal_type or not signal_data:
                await self.send_error("Invalid WebRTC signal data")
                return

            # Save signal to database
            await self.save_webrtc_signal(signal_type, signal_data, "inbound")

            # Broadcast signal to other peers in the room
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    "type": "webrtc_signal_message",
                    "signal_type": signal_type,
                    "signal_data": signal_data,
                    "sender": self.channel_name,
                },
            )

        except Exception as e:
            logger.error(f"Error handling WebRTC signal: {e}")
            await self.send_error(f"WebRTC signaling error: {str(e)}")

    async def handle_session_command(self, data):
        """Handle session control commands"""
        command = data.get("command")

        if command == "start":
            await self.update_session_status("processing")
            await self.send(
                text_data=json.dumps(
                    {"type": "session_started", "session_token": self.session_token}
                )
            )
        elif command == "stop":
            await self.update_session_status("completed")
            await self.send(
                text_data=json.dumps(
                    {"type": "session_stopped", "session_token": self.session_token}
                )
            )
        elif command == "reset":
            self.face_engine.reset_liveness_detector()
            await self.send(
                text_data=json.dumps(
                    {"type": "session_reset", "session_token": self.session_token}
                )
            )
        else:
            await self.send_error(f"Unknown command: {command}")

    async def process_enrollment_frame(self, frame):
        """Process frame for enrollment with enhanced liveness challenges"""
        try:
            # Get user from database
            user = await self.get_session_user()
            if not user:
                return {"success": False, "error": "User not found"}

            # Process frame with face recognition engine
            result, error = await asyncio.get_event_loop().run_in_executor(
                None, self.face_engine.process_frame_for_enrollment, frame, str(user.id)
            )

            if error:
                return {"success": False, "error": error, 'stage': 'processing'}

            # Save embedding if quality is good
            embedding_saved = False
            if result["quality_score"] >= 0.7:  # configurable threshold
                embedding_saved = await self.save_enrollment_embedding(user, result)

            # Fetch updated enrollment session progress
            progress = await self.get_enrollment_progress(user)
            completed = False
            if progress:
                if progress['completed_samples'] >= progress['target_samples']:
                    completed = True
                    await self.finalize_streaming_session('completed')

            # Enhanced liveness data with challenge info
            liveness_data = result.get("liveness_data", {})
            
            # Add challenge completion status for UI
            challenges = {
                'blink': {
                    'required': 2,
                    'completed': liveness_data.get('blinks_detected', 0),
                    'done': liveness_data.get('blinks_detected', 0) >= 2
                },
                'open_mouth': {
                    'required': 1,
                    'completed': liveness_data.get('open_mouth_count', 0),
                    'done': liveness_data.get('open_mouth_count', 0) >= 1
                },
                'turn_left': {
                    'required': 1,
                    'completed': liveness_data.get('turn_left_count', 0),
                    'done': liveness_data.get('turn_left_count', 0) >= 1
                },
                'turn_right': {
                    'required': 1,
                    'completed': liveness_data.get('turn_right_count', 0),
                    'done': liveness_data.get('turn_right_count', 0) >= 1
                }
            }
            
            # Determine current feedback for user
            current_feedback = None
            if not challenges['blink']['done']:
                current_feedback = {'action': 'blink', 'message': f"Kedipkan mata Anda ({challenges['blink']['completed']}/2)", 'icon': '👁️'}
            elif not challenges['open_mouth']['done']:
                if liveness_data.get('is_mouth_currently_open', False):
                    current_feedback = {'action': 'open_mouth', 'message': 'Bagus! Tahan mulut terbuka...', 'icon': '👄', 'in_progress': True}
                else:
                    current_feedback = {'action': 'open_mouth', 'message': 'Buka mulut Anda lebar-lebar', 'icon': '👄'}
            elif not challenges['turn_left']['done']:
                current_dir = liveness_data.get('current_direction', 'center')
                if current_dir == 'left':
                    current_feedback = {'action': 'turn_left', 'message': 'Bagus! Tahan kepala ke kiri...', 'icon': '⬅️', 'in_progress': True}
                else:
                    current_feedback = {'action': 'turn_left', 'message': 'Putar kepala Anda ke KIRI', 'icon': '⬅️'}
            elif not challenges['turn_right']['done']:
                current_dir = liveness_data.get('current_direction', 'center')
                if current_dir == 'right':
                    current_feedback = {'action': 'turn_right', 'message': 'Bagus! Tahan kepala ke kanan...', 'icon': '➡️', 'in_progress': True}
                else:
                    current_feedback = {'action': 'turn_right', 'message': 'Putar kepala Anda ke KANAN', 'icon': '➡️'}
            else:
                current_feedback = {'action': 'complete', 'message': 'Semua tantangan selesai! ✓', 'icon': '✅'}

            payload = {
                "success": True,
                "quality_score": result["quality_score"],
                "liveness_data": liveness_data,
                "embedding_saved": embedding_saved,
                "progress": progress,
                "feedback": self.get_quality_feedback(result),
                "challenges": challenges,
                "current_feedback": current_feedback,
                "all_challenges_done": all(c['done'] for c in challenges.values())
            }
            if completed:
                payload['finalized'] = True
            return payload

        except Exception as e:
            logger.error(f"Error processing enrollment frame: {e}")
            return {"success": False, "error": str(e)}

    async def process_authentication_frame(self, frame):
        """Process frame for authentication"""
        try:
            target_user_id = None
            session_data = self.streaming_session.session_data or {}
            auth_type = session_data.get("auth_type") or self.streaming_session.session_type

            if auth_type == "verification":
                target_email = session_data.get("target_email")
                if target_email:
                    target_user = await self.get_user_by_email(target_email)
                    if target_user:
                        target_user_id = str(target_user.id)
            elif auth_type == "authentication":
                session_user = await self.get_session_user()
                if session_user:
                    target_user_id = str(session_user.id)

            # Authenticate with face recognition engine
            auth_result = await asyncio.get_event_loop().run_in_executor(
                None, self.face_engine.authenticate_user, frame, target_user_id
            )

            # Save authentication attempt
            await self.save_authentication_attempt(auth_result, target_user_id)

            return auth_result

        except Exception as e:
            logger.error(f"Error processing authentication frame: {e}")
            return {"success": False, "error": str(e)}

    def get_quality_feedback(self, result):
        """Generate feedback based on quality metrics and liveness challenges"""
        feedback = []

        if result["quality_score"] < 0.7:
            feedback.append("Improve image quality")

        liveness_data = result.get("liveness_data", {})
        
        # Blink feedback
        blinks = liveness_data.get("blinks_detected", 0)
        if blinks < 2:
            feedback.append(f"Please blink naturally ({blinks}/2)")
        
        # Open mouth feedback
        open_mouth_count = liveness_data.get("open_mouth_count", 0)
        if open_mouth_count < 1:
            feedback.append("Open your mouth wide")
        
        # Head turn feedback
        left_turns = liveness_data.get("turn_left_count", 0)
        right_turns = liveness_data.get("turn_right_count", 0)
        if left_turns < 1:
            feedback.append("Turn your head LEFT")
        if right_turns < 1:
            feedback.append("Turn your head RIGHT")

        if result.get("obstacles"):
            feedback.append(f"Remove obstacles: {', '.join(result['obstacles'])}")

        return feedback

    async def webrtc_signal_message(self, event):
        """Handle WebRTC signal message from group"""
        # Don't send back to sender
        if event["sender"] != self.channel_name:
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "webrtc_signal",
                        "signal_type": event["signal_type"],
                        "signal_data": event["signal_data"],
                    }
                )
            )

    async def send_error(self, message):
        """Send error message to client"""
        await self.send(text_data=json.dumps({"type": "error", "error": message}))

    async def finalize_streaming_session(self, status_value: str):
        """Mark streaming session finalized"""
        try:
            if self.streaming_session and self.streaming_session.status not in ['completed', 'failed']:
                await self.update_session_status(status_value)
        except Exception as e:
            logger.error(f"Finalize session error: {e}")

    @database_sync_to_async
    def get_enrollment_progress(self, user):
        try:
            from recognition.models import EnrollmentSession
            session = EnrollmentSession.objects.filter(user=user, status__in=['pending','in_progress','completed']).order_by('-started_at').first()
            if not session:
                return None
            return {
                'completed_samples': session.completed_samples,
                'target_samples': session.target_samples,
                'status': session.status
            }
        except Exception as e:
            logger.error(f"Progress fetch error: {e}")
            return None

    @database_sync_to_async
    def get_streaming_session(self, session_token):
        """Get streaming session from database"""
        try:
            return StreamingSession.objects.get(
                session_token=session_token,
                status__in=["initiating", "connecting", "connected", "processing"],
            )
        except StreamingSession.DoesNotExist:
            return None

    @database_sync_to_async
    def get_session_user(self):
        """Get user associated with session"""
        if self.streaming_session and self.streaming_session.user:
            return self.streaming_session.user
        return None

    @database_sync_to_async
    def get_user_by_email(self, email):
        """Get user by email"""
        from django.contrib.auth import get_user_model
        User = get_user_model()
        try:
            return User.objects.get(email=email, is_active=True)
        except User.DoesNotExist:
            return None

    @database_sync_to_async
    def update_session_status(self, status):
        """Update streaming session status"""
        if self.streaming_session:
            self.streaming_session.status = status
            if status == "connected":
                from django.utils import timezone

                self.streaming_session.connected_at = timezone.now()
            elif status in ["completed", "failed", "disconnected"]:
                from django.utils import timezone

                self.streaming_session.completed_at = timezone.now()
            self.streaming_session.save()

    @database_sync_to_async
    def save_webrtc_signal(self, signal_type, signal_data, direction):
        """Save WebRTC signal to database"""
        WebRTCSignal.objects.create(
            session=self.streaming_session,
            signal_type=signal_type,
            signal_data=signal_data,
            direction=direction,
        )

    @database_sync_to_async
    def save_enrollment_embedding(self, user, result):
        """Save enrollment embedding to database"""
        try:
            from django.utils import timezone
            from recognition.models import EnrollmentSession, FaceEmbedding

            # Get or create enrollment session
            session, created = EnrollmentSession.objects.get_or_create(
                user=user,
                status__in=["pending", "in_progress"],
                defaults={
                    "session_token": self.session_token,
                    "target_samples": 5,
                    "expires_at": timezone.now() + timezone.timedelta(minutes=30),
                },
            )

            session.completed_samples += 1
            session.status = "in_progress"
            session.save()

            # Create face embedding
            embedding = FaceEmbedding.objects.create(
                user=user,
                enrollment_session=session,
                sample_number=session.completed_samples,
                quality_score=result["quality_score"],
                confidence_score=result["confidence"],
                face_bbox=result["bbox"].tolist(),
                liveness_score=result["liveness_data"].get("blinks_detected", 0) / 5.0,
                anti_spoofing_score=result["quality_score"],
            )

            # Save embedding vector
            embedding.set_embedding_vector(result["embedding"])
            embedding.save()

            # Save to embedding store
            metadata = {
                "user_id": str(user.id),
                "sample_number": session.completed_samples,
                "quality_score": result["quality_score"],
                "enrollment_session": str(session.id),
            }

            self.face_engine.save_embedding(str(user.id), result["embedding"], metadata)

            return True

        except Exception as e:
            logger.error(f"Error saving enrollment embedding: {e}")
            return False

    @database_sync_to_async
    def save_authentication_attempt(self, auth_result, target_user_id):
        """Save authentication attempt to database"""
        try:
            from analytics.models import AuthenticationLog
            from recognition.models import AuthenticationAttempt
            from django.contrib.auth import get_user_model

            User = get_user_model()

            # Get target user
            target_user = None
            if target_user_id:
                try:
                    target_user = User.objects.get(id=target_user_id)
                except User.DoesNotExist:
                    pass

            matched_user = None
            matched_user_id = auth_result.get("user_id")
            if matched_user_id:
                try:
                    matched_user = User.objects.get(id=matched_user_id)
                except User.DoesNotExist:
                    matched_user = None

            log_user = None
            if auth_result.get("success"):
                log_user = matched_user or target_user

            # Create authentication attempt
            AuthenticationAttempt.objects.create(
                user=log_user,
                session_id=self.session_token,
                similarity_score=auth_result.get("similarity_score", 0.0),
                liveness_score=auth_result.get("liveness_data", {}).get(
                    "blinks_detected", 0
                )
                / 5.0,
                quality_score=auth_result.get("quality_score", 0.0),
                result="success" if auth_result["success"] else "failed_similarity",
                obstacles_detected=auth_result.get("obstacles", []),
                metadata=auth_result,
            )

            # Create authentication log
            AuthenticationLog.objects.create(
                user=log_user,
                auth_method="face",
                success=auth_result["success"],
                failure_reason=auth_result.get("error", "")
                if not auth_result["success"]
                else "",
                similarity_score=auth_result.get("similarity_score", 0.0),
                liveness_score=auth_result.get("liveness_data", {}).get(
                    "blinks_detected", 0
                )
                / 5.0,
                quality_score=auth_result.get("quality_score", 0.0),
                session_id=self.session_token,
            )

            session_modified = False
            session = self.streaming_session
            if session:
                session_data = session.session_data or {}
                if matched_user and not session.user:
                    session.user = matched_user
                    session_modified = True
                if matched_user:
                    session_data["recognized_user_id"] = str(matched_user.id)
                    session_data["recognized_user_email"] = matched_user.email
                    session_modified = True
                if session_modified:
                    session.session_data = session_data
                    session.save(update_fields=["user", "session_data"])

        except Exception as e:
            logger.error(f"Error saving authentication attempt: {e}")

    async def decode_frame_data(self, frame_data):
        """Decode base64 frame data to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if frame_data.startswith("data:image"):
                frame_data = frame_data.split(",")[1]

            # Decode base64
            frame_bytes = base64.b64decode(frame_data)

            # Convert to numpy array
            nparr = np.frombuffer(frame_bytes, np.uint8)

            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            return frame

        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return None

"""
WebSocket consumers for real-time face recognition
"""

import asyncio
import base64
import json
import logging

import cv2
import numpy as np
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from core.face_recognition_engine import FaceRecognitionEngine

from streaming.models import StreamingSession, WebRTCSignal

logger = logging.getLogger("streaming")


class FaceRecognitionConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time face recognition"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_token = None
        self.streaming_session = None
        self.face_engine = FaceRecognitionEngine()
        self.room_group_name = None

    async def connect(self):
        """Accept WebSocket connection"""
        self.session_token = self.scope["url_route"]["kwargs"]["session_token"]
        self.room_group_name = f"face_recognition_{self.session_token}"

        # Validate session
        try:
            self.streaming_session = await self.get_streaming_session(
                self.session_token
            )
            if not self.streaming_session:
                await self.close(code=4004)
                return
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            await self.close(code=4004)
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
            frame_data = data.get("frame_data")
            if not frame_data:
                await self.send_error("No frame data provided")
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
            elif self.streaming_session.session_type == "authentication":
                result = await self.process_authentication_frame(frame)
            else:
                await self.send_error("Unknown session type")
                return

            # Send result
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "frame_result",
                        "result": result,
                        "timestamp": data.get("timestamp"),
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
        """Process frame for enrollment"""
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
                return {"success": False, "error": error}

            # Save embedding if quality is good
            if result["quality_score"] >= 0.7:  # configurable threshold
                embedding_saved = await self.save_enrollment_embedding(user, result)
                if embedding_saved:
                    return {
                        "success": True,
                        "quality_score": result["quality_score"],
                        "liveness_data": result["liveness_data"],
                        "embedding_saved": True,
                    }

            return {
                "success": True,
                "quality_score": result["quality_score"],
                "liveness_data": result["liveness_data"],
                "embedding_saved": False,
                "feedback": self.get_quality_feedback(result),
            }

        except Exception as e:
            logger.error(f"Error processing enrollment frame: {e}")
            return {"success": False, "error": str(e)}

    async def process_authentication_frame(self, frame):
        """Process frame for authentication"""
        try:
            # Get target user if in verification mode
            target_user_id = None
            session_data = self.streaming_session.session_data or {}

            if session_data.get("auth_type") == "verification":
                target_email = session_data.get("target_email")
                if target_email:
                    target_user = await self.get_user_by_email(target_email)
                    if target_user:
                        target_user_id = str(target_user.id)

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
        """Generate feedback based on quality metrics"""
        feedback = []

        if result["quality_score"] < 0.7:
            feedback.append("Improve image quality")

        if result["liveness_data"]["blinks_detected"] < 2:
            feedback.append("Please blink naturally")

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

            # Get target user
            target_user = None
            if target_user_id:
                try:
                    from django.contrib.auth import get_user_model
                    User = get_user_model()
                    target_user = User.objects.get(id=target_user_id)
                except User.DoesNotExist:
                    pass

            # Create authentication attempt
            AuthenticationAttempt.objects.create(
                user=target_user if auth_result["success"] else None,
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
                user=target_user if auth_result["success"] else None,
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

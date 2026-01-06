"""
WebSocket consumers for face authentication and enrollment
"""

import asyncio
import base64
import json
import logging
import time
import secrets
from datetime import datetime
from typing import Optional, Dict, Any

import cv2
import numpy as np
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.utils import timezone
from django.conf import settings
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import hashlib

from .models import AuthenticationSession, FaceEnrollment
from clients.models import Client, ClientUser
from core.face_recognition_engine import FaceRecognitionEngine
from core.passive_liveness_optimal import OptimizedPassiveLivenessDetector

logger = logging.getLogger("auth_service")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def convert_numpy_types(data):
    """
    Recursively convert NumPy types to native Python types for JSON serialization
    """
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data


class AuthProcessConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for processing face images during authentication and enrollment.
    Events are created based on session token from create_enrollment_session 
    and create_authentication_session.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_token = None
        self.session = None
        self.client = None
        self.face_engine = FaceRecognitionEngine()
        self._frames_processed = 0
        self._last_frame_ts = 0.0
        self._throttle_interval = 0.1  # 10 fps max
        self._max_frames = getattr(settings, 'FACE_MAX_FRAMES_PER_SESSION', 120)
        self._is_connected = False  # Track connection state
        
        # Authentication-specific: optimal passive liveness with 3-second timeout
        self._auth_liveness_detector = None
        self._auth_start_time = None
        self._auth_timeout = 10.0  # 3 seconds for authentication

    async def connect(self):
        """Accept WebSocket connection and validate session"""
        self.session_token = self.scope["url_route"]["kwargs"].get("session_token")
        
        if not self.session_token:
            await self.close(code=4400)
            return

        # Validate session and load data
        try:
            self.session = await self.get_session(self.session_token)
            if not self.session:
                await self.send_error("Session not found", code=4404)
                await self.close(code=4404)
                return
            
            self.client = await self.get_client(self.session.client_id)
            if not self.client:
                await self.send_error("Client not found", code=4404)
                await self.close(code=4404)
                return

            # Check session status
            if self.session.status not in ["active", "processing"]:
                await self.send_error(f"Session is {self.session.status}", code=4400)
                await self.close(code=4400)
                return

            # Check expiration
            if self.session.expires_at and self.session.expires_at < timezone.now():
                await self.update_session_status("expired")
                await self.send_error("Session expired", code=4401)
                await self.close(code=4401)
                return

        except Exception as e:
            logger.error(f"Error validating session: {e}", exc_info=True)
            await self.send_error("Internal server error", code=4500)
            await self.close(code=4500)
            return

        await self.accept()
        self._is_connected = True  # Mark as connected
        
        # Initialize optimal passive liveness detector for authentication sessions
        if self.session.session_type in ["verification", "identification", "recognition"]:
            self._auth_liveness_detector = OptimizedPassiveLivenessDetector(
                debug=False, 
                max_duration=self._auth_timeout
            )
            self._auth_start_time = time.time()
            logger.info(f"Initialized optimal passive liveness detector for authentication (timeout: {self._auth_timeout}s)")
        
        # Send connection confirmation
        await self.safe_send(text_data=json.dumps({
            "type": "connection_established",
            "session_token": self.session_token,
            "session_type": self.session.session_type,
            "status": self.session.status,
            "timeout": self._auth_timeout if self.session.session_type in ["verification", "identification", "recognition"] else None,
            "message": f"{self.session.session_type.capitalize()} session ready"
        }))

        logger.info(f"WebSocket connected for {self.session.session_type} session {self.session_token}")

    async def disconnect(self, close_code):
        """Handle WebSocket disconnect"""
        self._is_connected = False  # Mark as disconnected
        if self.session and self.session.status == "active":
            await self.update_session_status("disconnected")
        
        logger.info(f"WebSocket disconnected for session {self.session_token} with code {close_code}")

    async def receive(self, text_data=None, bytes_data=None):
        """Receive and process face image frames"""
        try:
            # Throttle frame processing
            now = time.time()
            if now - self._last_frame_ts < self._throttle_interval:
                await self.send_error("Frame rate too high, please slow down", code=429)
                return
            self._last_frame_ts = now

            # Check frame budget
            if self._frames_processed >= self._max_frames:
                await self.send_error("Maximum frames exceeded for this session", code=429)
                await self.close(code=4429)
                return

            # Parse message
            if text_data:
                data = json.loads(text_data)
                message_type = data.get("type", "frame")
                
                if message_type == "frame":
                    await self.handle_frame(data)
                elif message_type == "ping":
                    await self.safe_send(text_data=json.dumps({"type": "pong"}))
                else:
                    await self.send_error(f"Unknown message type: {message_type}")
                    
        except json.JSONDecodeError as e:
            await self.send_error(f"Invalid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self.send_error(f"Processing error: {str(e)}")

    async def handle_frame(self, data: Dict[str, Any]):
        """Process a single face image frame"""
        try:
            # Get image data
            image_data = data.get("image")
            if not image_data:
                await self.send_error("No image data provided")
                return

            # Decode base64 image
            if image_data.startswith("data:image"):
                image_data = image_data.split(",", 1)[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # Convert to OpenCV frame
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await self.send_error("Unable to decode image")
                return

            self._frames_processed += 1

            # Process based on session type
            if self.session.session_type == "enrollment":
                await self.process_enrollment_frame(frame, data)
            elif self.session.session_type in ["verification", "identification", "recognition"]:
                await self.process_authentication_frame(frame, data)
            else:
                await self.send_error(f"Unsupported session type: {self.session.session_type}")

        except Exception as e:
            logger.error(f"Error handling frame: {e}", exc_info=True)
            await self.send_error(f"Frame processing error: {str(e)}")

    async def process_enrollment_frame(self, frame: np.ndarray, data: Dict[str, Any]):
        """Process enrollment frame"""
        try:
            # Get enrollment record
            enrollment = await self.get_enrollment(self.session.id)
            if not enrollment:
                await self.send_error("Enrollment not found")
                return

            # Process frame with face engine
            result = await self.run_face_detection(frame)
            
            if not result.get("success"):
                await self.safe_send(text_data=json.dumps({
                    "type": "frame_processed",
                    "success": False,
                    "error": result.get("error", "Face detection failed"),
                    "frames_processed": self._frames_processed
                }))
                return

            # Extract face data
            face_data = result.get("face_data", {})
            liveness_data = result.get("liveness_data", {})
            obstacle_data = result.get("obstacle_data", {})
            visual_data = result.get("visual_data", {})
            
            # STRICT: Reject frame if ANY obstacle is detected (blocking or non-blocking)
            detected_obstacles = obstacle_data.get("detected", [])
            if detected_obstacles:
                obstacle_names = ', '.join(detected_obstacles)
                obstacle_emoji = {
                    'glasses': '🕶️',
                    'mask': '😷',
                    'hat': '🎩',
                    'hand_covering': '✋'
                }
                emoji_list = ' '.join([obstacle_emoji.get(obs, '⚠️') for obs in detected_obstacles])
                
                await self.safe_send(text_data=json.dumps({
                    "type": "frame_rejected",
                    "success": False,
                    "error": f"Obstacles detected: {obstacle_names}",
                    "reason": f"{emoji_list} Please remove: {obstacle_names}",
                    "obstacles": detected_obstacles,
                    "obstacle_confidence": obstacle_data.get("confidence", {}),
                    "visual_data": visual_data,
                    "frame_accepted": False
                }))
                logger.warning(f"⛔ ENROLLMENT FRAME REJECTED due to obstacles: {detected_obstacles}")
                return
            
            # Update session metadata
            metadata = self.session.metadata or {}
            metadata["frames_processed"] = self._frames_processed
            metadata["liveness_blinks"] = liveness_data.get("blinks_detected", 0)
            metadata["motion_events"] = liveness_data.get("motion_events", 0)
            metadata["liveness_verified"] = liveness_data.get("is_live", False)
            
            await self.update_session_metadata(metadata)

            # Check if enrollment is complete
            target_samples = metadata.get("target_samples", 3)
            required_blinks = metadata.get("required_blinks", 1)
            required_motion = metadata.get("required_motion_events", 1)
            
            # STRICT Liveness verification: need blinks AND motion AND no obstacles
            # Explicitly convert to Python bool to avoid numpy.bool_ JSON serialization issues
            blinks_ok = bool(liveness_data.get("blinks_detected", 0) >= required_blinks)
            motion_ok = bool(liveness_data.get("motion_events", 0) >= required_motion)
            no_obstacles = bool(len(detected_obstacles) == 0)
            liveness_verified = bool(blinks_ok and motion_ok and no_obstacles)
            
            # Check if enrollment is complete
            is_complete = (
                self._frames_processed >= target_samples and 
                liveness_verified and
                face_data.get("quality", 0.0) >= self.face_engine.capture_quality_threshold
            )

            if is_complete:
                # Complete enrollment with frame data
                success, similarity_score = await self.complete_enrollment(enrollment, face_data, liveness_data, frame)
                
                if success:
                    # Generate encrypted response
                    encrypted_response = await self.generate_encrypted_response(
                        enrollment_id=str(enrollment.id),
                        session_type="enrollment"
                    )
                    
                    await self.safe_send(text_data=json.dumps({
                        "type": "enrollment_complete",
                        "success": True,
                        "enrollment_id": str(enrollment.id),
                        "frames_processed": self._frames_processed,
                        "liveness_verified": True,
                        "blinks_detected": liveness_data.get("blinks_detected", 0),
                        "motion_verified": liveness_data.get("motion_verified", False),
                        "quality_score": face_data.get("quality", 0.0),
                        "similarity_with_old_photo": float(similarity_score) if similarity_score is not None else None,
                        "encrypted_data": encrypted_response,
                        "visual_data": visual_data,
                        "message": "Enrollment completed successfully"
                    }))
                    
                    await self.update_session_status("completed")
                    
                    # Wait 1 second before closing to allow client to process response
                    await asyncio.sleep(1)
                    await self.close(code=1000)
                else:
                    # Check if failure is due to duplicate face
                    enrollment_refresh = await self.get_enrollment(self.session.id)
                    failure_reason = enrollment_refresh.metadata.get('failure_reason') if enrollment_refresh else None
                    
                    if failure_reason == 'duplicate_face':
                        conflicting_user = enrollment_refresh.metadata.get('conflicting_user_id', 'another user')
                        similarity = enrollment_refresh.metadata.get('similarity_score', 0.0)
                        
                        await self.safe_send(text_data=json.dumps({
                            "type": "enrollment_failed",
                            "success": False,
                            "error": "duplicate_face",
                            "error_code": "DUPLICATE_FACE_DETECTED",
                            "message": f"❌ This face has already been enrolled by another user",
                            "details": {
                                "reason": "Face already exists in system",
                                "similarity_score": float(similarity),
                                "conflicting_user": str(conflicting_user)
                            },
                            "visual_data": visual_data
                        }))
                        
                        await self.update_session_status("failed")
                        await asyncio.sleep(1)
                        await self.close(code=4001)  # Custom close code for duplicate
                    else:
                        await self.send_error("Failed to complete enrollment")
            else:
                # Send progress update with detailed feedback
                progress_message = []
                if self._frames_processed < target_samples:
                    progress_message.append(f"Capturing frames ({self._frames_processed}/{target_samples})")
                if not blinks_ok:
                    progress_message.append(f"Blink required ({liveness_data.get('blinks_detected', 0)}/{required_blinks})")
                if not motion_ok:
                    progress_message.append(f"Motion required ({liveness_data.get('motion_events', 0)}/{required_motion})")
                if not no_obstacles:
                    progress_message.append("⛔ Remove obstacles first!")
                if face_data.get("quality", 0.0) < self.face_engine.capture_quality_threshold:
                    progress_message.append("Improve lighting or position")
                
                # Prepare response with explicit type conversion for all numeric values
                response_data = {
                    "type": "frame_processed",
                    "success": True,
                    "frames_processed": int(self._frames_processed),
                    "target_samples": int(target_samples),
                    "liveness_verified": bool(liveness_verified),
                    "liveness_score": float(liveness_data.get("liveness_score", 0.0)),
                    "blinks_detected": int(liveness_data.get("blinks_detected", 0)),
                    "blinks_required": int(required_blinks),
                    "blinks_ok": bool(blinks_ok),
                    "motion_score": float(liveness_data.get("motion_score", 0.0)),
                    "motion_events": int(liveness_data.get("motion_events", 0)),
                    "motion_required": int(required_motion),
                    "motion_ok": bool(motion_ok),
                    "no_obstacles": bool(no_obstacles),
                    "quality_score": float(face_data.get("quality", 0.0)),
                    "quality_threshold": float(self.face_engine.capture_quality_threshold),
                    "obstacles": obstacle_data.get("detected", []),
                    "visual_data": visual_data,
                    "message": " | ".join(progress_message) if progress_message else "Blink AND move head to complete"
                }
                
                await self.safe_send(text_data=json.dumps(response_data))

        except Exception as e:
            logger.error(f"Error processing enrollment frame: {e}", exc_info=True)
            await self.send_error(f"Enrollment processing error: {str(e)}")

    async def process_authentication_frame(self, frame: np.ndarray, data: Dict[str, Any]):
        """Process authentication frame with optimal passive liveness"""
        try:
            # Check timeout first - auto-close after 3 seconds
            if self._auth_start_time and (time.time() - self._auth_start_time) >= self._auth_timeout:
                logger.warning(f"⏱️ Authentication timeout reached ({self._auth_timeout}s)")
                
                # Get final liveness result
                if self._auth_liveness_detector:
                    is_live, liveness_score, liveness_details = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._auth_liveness_detector.detect,
                        frame
                    )
                    
                    timeout_reached = liveness_details.get('timeout', False)
                    blink_count = liveness_details.get('blink_count', 0)
                    
                    # Send timeout result
                    await self.safe_send(text_data=json.dumps({
                        "type": "authentication_timeout",
                        "success": False,
                        "timeout": True,
                        "elapsed_time": self._auth_timeout,
                        "is_live": is_live,
                        "liveness_score": liveness_score,
                        "blink_count": blink_count,
                        "reason": liveness_details.get('reason', 'Authentication timeout'),
                        "message": f"Authentication timeout: {liveness_details.get('reason', 'Time limit exceeded')}"
                    }))
                else:
                    await self.safe_send(text_data=json.dumps({
                        "type": "authentication_timeout",
                        "success": False,
                        "timeout": True,
                        "elapsed_time": self._auth_timeout,
                        "message": "Authentication timeout - no response"
                    }))
                
                await self.update_session_status("timeout")
                await asyncio.sleep(0.5)
                await self.close(code=1000)
                return
            
            # Use optimal passive liveness detector for authentication
            if self._auth_liveness_detector:
                # Run optimal liveness detection
                is_live, liveness_score, liveness_details = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._auth_liveness_detector.detect,
                    frame
                )
                
                # Check if timeout occurred during detection
                if liveness_details.get('timeout', False):
                    blink_count = liveness_details.get('blink_count', 0)
                    await self.safe_send(text_data=json.dumps({
                        "type": "authentication_timeout",
                        "success": is_live,
                        "authenticated": is_live,
                        "timeout": True,
                        "elapsed_time": liveness_details.get('elapsed_time', self._auth_timeout),
                        "is_live": is_live,
                        "liveness_score": liveness_score,
                        "blink_count": blink_count,
                        "reason": liveness_details.get('reason', 'Authentication complete'),
                        "message": liveness_details.get('reason', 'Authentication timeout')
                    }))
                    
                    await self.update_session_status("timeout")
                    await asyncio.sleep(0.5)
                    await self.close(code=1000)
                    return
                
                # Extract scores from optimal detector
                scores = liveness_details.get('scores', {})
                blink_count = liveness_details.get('blink_count', 0)
                
                # Check if device detected (immediate rejection)
                if 'device_detected' in liveness_details:
                    device_name = liveness_details.get('device_detected')
                    await self.safe_send(text_data=json.dumps({
                        "type": "frame_rejected",
                        "success": False,
                        "error": f"Device detected: {device_name}",
                        "reason": f"📱 Spoofing attempt detected - device: {device_name}",
                        "device_detected": device_name,
                        "frame_accepted": False
                    }))
                    logger.warning(f"⛔ AUTHENTICATION FRAME REJECTED - Device detected: {device_name}")
                    return
                
                # Also get face detection for embedding and obstacles
                result = await self.run_face_detection(frame)
                
                if not result.get("success"):
                    await self.safe_send(text_data=json.dumps({
                        "type": "frame_processed",
                        "success": False,
                        "error": result.get("error", "Face detection failed"),
                        "frames_processed": self._frames_processed
                    }))
                    return
                
                # Merge optimal liveness data with face detection result
                face_data = result.get("face_data", {})
                obstacle_data = result.get("obstacle_data", {})
                visual_data = result.get("visual_data", {})
                
                # Construct liveness_data from optimal detector
                liveness_data = {
                    'is_live': is_live,
                    'liveness_score': liveness_score,
                    'blinks_detected': blink_count,
                    'motion_events': 0,  # Optimal detector uses different motion tracking
                    'motion_score': scores.get('movement', 0.0),
                    'motion_verified': scores.get('movement', 0.0) > 0.5,
                    'blink_score': scores.get('blink', 0.0),
                    'screen_score': scores.get('screen', 0.0),
                    'final_score': liveness_score
                }
            else:
                # Fallback to original face engine detection
                result = await self.run_face_detection(frame)
                
                if not result.get("success"):
                    await self.safe_send(text_data=json.dumps({
                        "type": "frame_processed",
                        "success": False,
                        "error": result.get("error", "Face detection failed"),
                        "frames_processed": self._frames_processed
                    }))
                    return
                
                face_data = result.get("face_data", {})
                liveness_data = result.get("liveness_data", {})
                obstacle_data = result.get("obstacle_data", {})
                visual_data = result.get("visual_data", {})
            
            # STRICT: Reject frame if ANY obstacle is detected (blocking or non-blocking)
            detected_obstacles = obstacle_data.get("detected", [])
            if detected_obstacles:
                obstacle_names = ', '.join(detected_obstacles)
                obstacle_emoji = {
                    'glasses': '🕶️',
                    'mask': '😷',
                    'hat': '🎩',
                    'hand_covering': '✋'
                }
                emoji_list = ' '.join([obstacle_emoji.get(obs, '⚠️') for obs in detected_obstacles])
                
                await self.safe_send(text_data=json.dumps({
                    "type": "frame_rejected",
                    "success": False,
                    "error": f"Obstacles detected: {obstacle_names}",
                    "reason": f"{emoji_list} Please remove: {obstacle_names}",
                    "obstacles": detected_obstacles,
                    "obstacle_confidence": obstacle_data.get("confidence", {}),
                    "visual_data": visual_data,
                    "frame_accepted": False
                }))
                logger.warning(f"⛔ AUTHENTICATION FRAME REJECTED due to obstacles: {detected_obstacles}")
                return
            
            # Update session metadata
            metadata = self.session.metadata or {}
            metadata["frames_processed"] = self._frames_processed
            metadata["liveness_score"] = float(liveness_data.get("liveness_score", 0.0))
            metadata["is_live"] = bool(liveness_data.get("is_live", False))
            metadata["blinks_detected"] = int(liveness_data.get("blinks_detected", 0))
            
            await self.update_session_metadata(metadata)

            # Check if we have enough frames for authentication
            min_frames = metadata.get("min_frames_required", 3)
            
            # For authentication: OptimizedPassiveLivenessDetector handles liveness (blink/motion)
            # We only need to check for obstacles here
            no_obstacles = bool(len(detected_obstacles) == 0)
            
            # Use is_live from OptimizedPassiveLivenessDetector
            liveness_verified = bool(liveness_data.get("is_live", False) and no_obstacles)
            
            is_ready = (
                self._frames_processed >= min_frames and 
                liveness_verified and
                face_data.get("quality", 0.0) >= self.face_engine.auth_quality_threshold
            )

            if is_ready:
                # Perform authentication
                auth_result = await self.perform_authentication(face_data, liveness_data)
                
                if auth_result.get("success"):
                    # Generate encrypted response
                    encrypted_response = await self.generate_encrypted_response(
                        user_id=auth_result.get("user_id"),
                        session_type="authentication",
                        confidence=auth_result.get("confidence", 0.0)
                    )
                    
                    await self.safe_send(text_data=json.dumps({
                        "type": "authentication_complete",
                        "success": True,
                        "authenticated": True,
                        "user_id": auth_result.get("user_id"),
                        "confidence": auth_result.get("confidence", 0.0),
                        "frames_processed": self._frames_processed,
                        "blinks_detected": liveness_data.get("blinks_detected", 0),
                        "motion_verified": liveness_data.get("motion_verified", False),
                        "encrypted_data": encrypted_response,
                        "visual_data": visual_data,
                        "message": "Authentication successful"
                    }))
                    
                    await self.update_session_status("completed")
                    
                    # Wait 1 second before closing to allow client to process response
                    await asyncio.sleep(1)
                    await self.close(code=1000)
                else:
                    await self.safe_send(text_data=json.dumps({
                        "type": "authentication_complete",
                        "success": False,
                        "authenticated": False,
                        "frames_processed": self._frames_processed,
                        "visual_data": visual_data,
                        "message": auth_result.get("error", "Authentication failed")
                    }))
                    
                    await self.update_session_status("failed")
                    
                    # Wait 1 second before closing
                    await asyncio.sleep(1)
                    await self.close(code=1000)
            else:
                # Send progress update with detailed feedback
                progress_message = []
                
                # Add elapsed time info
                if self._auth_start_time:
                    elapsed = time.time() - self._auth_start_time
                    remaining = max(0, self._auth_timeout - elapsed)
                    progress_message.append(f"⏱️ {remaining:.1f}s remaining")
                
                if self._frames_processed < min_frames:
                    progress_message.append(f"Capturing frames ({self._frames_processed}/{min_frames})")
                
                # For authentication: Only check obstacles (liveness handled by OptimizedPassiveLivenessDetector)
                if not liveness_verified:
                    if not no_obstacles:
                        progress_message.append("⛔ Remove obstacles first!")
                    elif not liveness_data.get("is_live", False):
                        progress_message.append("👁️ Looking for liveness indicators...")
                
                if face_data.get("quality", 0.0) < self.face_engine.auth_quality_threshold:
                    progress_message.append("Improve lighting or position")
                
                # Add elapsed time to response
                elapsed_time = time.time() - self._auth_start_time if self._auth_start_time else 0
                
                # Prepare response with explicit type conversion for all numeric values
                response_data = {
                    "type": "frame_processed",
                    "success": True,
                    "frames_processed": int(self._frames_processed),
                    "min_frames_required": int(min_frames),
                    "liveness_verified": bool(liveness_verified),
                    "liveness_score": float(liveness_data.get("liveness_score", 0.0)),
                    "is_live": bool(liveness_data.get("is_live", False)),
                    "blinks_detected": int(liveness_data.get("blinks_detected", 0)),
                    "motion_score": float(liveness_data.get("motion_score", 0.0)),
                    "no_obstacles": bool(no_obstacles),
                    "quality_score": float(face_data.get("quality", 0.0)),
                    "quality_threshold": float(self.face_engine.auth_quality_threshold),
                    "obstacles": obstacle_data.get("detected", []),
                    "visual_data": visual_data,
                    "elapsed_time": float(elapsed_time),
                    "timeout": float(self._auth_timeout),
                    "message": " | ".join(progress_message) if progress_message else f"Authenticating... ({elapsed_time:.1f}s/{self._auth_timeout}s)"
                }
                
                await self.safe_send(text_data=json.dumps(response_data))

        except Exception as e:
            logger.error(f"Error processing authentication frame: {e}", exc_info=True)
            await self.send_error(f"Authentication processing error: {str(e)}")

    async def generate_encrypted_response(
        self, 
        enrollment_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_type: str = "enrollment",
        confidence: float = 0.0
    ) -> Dict[str, str]:
        """
        Generate encrypted response with id and timestamp.
        Data is encrypted twice: first with client's API key, then client encrypts with secret key.
        """
        try:
            timestamp = timezone.now().isoformat()
            
            # Prepare payload
            payload = {
                "id": enrollment_id or user_id,
                "timestamp": timestamp,
                "session_type": session_type,
                "session_token": self.session_token
            }
            
            if confidence > 0:
                payload["confidence"] = confidence
            
            # Convert to JSON
            payload_json = json.dumps(payload)
            
            # Encrypt with API key (server-side encryption)
            encrypted_with_api_key = await self.encrypt_with_api_key(payload_json)
            
            # Return both versions for client
            return {
                "encrypted_payload": encrypted_with_api_key,
                "encryption_method": "AES-256-CBC",
                "instructions": "Decrypt with your secret_key to access the data"
            }
            
        except Exception as e:
            logger.error(f"Error generating encrypted response: {e}", exc_info=True)
            return {
                "error": "Failed to generate encrypted response"
            }

    async def encrypt_with_api_key(self, data: str) -> str:
        """Encrypt data with client's API key using AES-256-CBC"""
        try:
            # Get API key from client
            api_key = self.client.api_key
            
            # Derive encryption key from API key
            key = hashlib.sha256(api_key.encode()).digest()
            
            # Generate IV
            iv = secrets.token_bytes(16)
            
            # Pad data
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data.encode()) + padder.finalize()
            
            # Encrypt
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            encrypted = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine IV and encrypted data
            result = iv + encrypted
            
            # Return as base64
            return base64.b64encode(result).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error encrypting with API key: {e}", exc_info=True)
            raise

    async def run_face_detection(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run face detection and liveness check on frame"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._sync_face_detection,
            frame
        )

    def _sync_face_detection(self, frame: np.ndarray) -> Dict[str, Any]:
        """Synchronous face detection (runs in thread pool)"""
        try:
            # Detect faces
            faces = self.face_engine.detect_faces(frame)
            
            if not faces or len(faces) == 0:
                return {
                    "success": False,
                    "error": "No face detected"
                }
            
            if len(faces) > 1:
                return {
                    "success": False,
                    "error": "Multiple faces detected"
                }
            
            face = faces[0]
            bbox = face.get("bbox")
            
            # Detect obstacles
            obstacles = []
            obstacle_info = {}
            if bbox:
                obstacles, obstacle_confidence = self.face_engine.obstacle_detector.detect_obstacles(frame, bbox)
                obstacle_info = {
                    "detected": obstacles,
                    "confidence": obstacle_confidence,
                    "has_blocking": any(obs in obstacles for obs in self.face_engine.blocking_obstacles)
                }
            
            # Check liveness with session token for state tracking
            liveness_result = self.face_engine.check_liveness(frame, face, session_token=self.session_token)
            
            # Get visual feedback data for drawing polygons
            visual_data = self._extract_visual_data(frame, face, liveness_result)
            
            # Get face quality
            quality = face.get("quality", 0.0)
            
            return {
                "success": True,
                "face_data": {
                    "embedding": face.get("embedding"),
                    "quality": quality,
                    "bbox": face.get("bbox"),
                    "landmarks": face.get("landmarks"),
                },
                "liveness_data": liveness_result,
                "obstacle_data": obstacle_info,
                "visual_data": visual_data
            }
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_visual_data(self, frame: np.ndarray, face_data: Dict[str, Any], liveness_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract visual feedback data for client-side drawing"""
        try:
            visual_data = {
                "face_bbox": face_data.get("bbox"),
                "landmarks": face_data.get("landmarks"),
                "eye_regions": {},
                "face_mesh_landmarks": {},
                "liveness_indicators": {}
            }
            
            # Get session-based liveness detector for accessing internal state
            if self.session_token:
                liveness_detector = self.face_engine.session_manager.get_liveness_detector(self.session_token)
                
                if liveness_detector:
                    # Get eye landmarks for visualization
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = liveness_detector.face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        h, w = frame.shape[:2]
                        
                        # Extract left eye region (comprehensive)
                        left_eye_points = []
                        for idx in liveness_detector.LEFT_EYE_LANDMARKS:
                            if idx < len(face_landmarks.landmark):
                                point = face_landmarks.landmark[idx]
                                left_eye_points.append([int(point.x * w), int(point.y * h)])
                        
                        # Extract right eye region (comprehensive)
                        right_eye_points = []
                        for idx in liveness_detector.RIGHT_EYE_LANDMARKS:
                            if idx < len(face_landmarks.landmark):
                                point = face_landmarks.landmark[idx]
                                right_eye_points.append([int(point.x * w), int(point.y * h)])
                        
                        visual_data["eye_regions"] = {
                            "left_eye": left_eye_points,
                            "right_eye": right_eye_points
                        }
                        
                        # Extract comprehensive face mesh landmarks for better visualization
                        # Face oval/contour
                        face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                                           397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                           172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                        
                        face_oval_points = []
                        for idx in face_oval_indices:
                            if idx < len(face_landmarks.landmark):
                                point = face_landmarks.landmark[idx]
                                face_oval_points.append([int(point.x * w), int(point.y * h)])
                        
                        # Lips outer
                        lips_outer_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                                            308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
                        
                        lips_points = []
                        for idx in lips_outer_indices:
                            if idx < len(face_landmarks.landmark):
                                point = face_landmarks.landmark[idx]
                                lips_points.append([int(point.x * w), int(point.y * h)])
                        
                        # Left eyebrow
                        left_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                        left_eyebrow_points = []
                        for idx in left_eyebrow_indices:
                            if idx < len(face_landmarks.landmark):
                                point = face_landmarks.landmark[idx]
                                left_eyebrow_points.append([int(point.x * w), int(point.y * h)])
                        
                        # Right eyebrow
                        right_eyebrow_indices = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
                        right_eyebrow_points = []
                        for idx in right_eyebrow_indices:
                            if idx < len(face_landmarks.landmark):
                                point = face_landmarks.landmark[idx]
                                right_eyebrow_points.append([int(point.x * w), int(point.y * h)])
                        
                        # Nose bridge and tip
                        nose_indices = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
                        nose_points = []
                        for idx in nose_indices:
                            if idx < len(face_landmarks.landmark):
                                point = face_landmarks.landmark[idx]
                                nose_points.append([int(point.x * w), int(point.y * h)])
                        
                        visual_data["face_mesh_landmarks"] = {
                            "face_oval": face_oval_points,
                            "lips": lips_points,
                            "left_eyebrow": left_eyebrow_points,
                            "right_eyebrow": right_eyebrow_points,
                            "nose": nose_points
                        }
                    
                    # Add liveness indicators
                    visual_data["liveness_indicators"] = {
                        "blinks": liveness_result.get("blinks_detected", 0),
                        "ear": liveness_result.get("ear", 0.0),
                        "blink_verified": liveness_result.get("blink_detected", False),
                        "motion_verified": liveness_result.get("motion_verified", False),
                        "motion_score": liveness_result.get("motion_score", 0.0),
                        "motion_events": liveness_result.get("motion_events", 0),
                        "frame_counter": liveness_detector.frame_counter
                    }
            
            return visual_data
            
        except Exception as e:
            logger.error(f"Error extracting visual data: {e}", exc_info=True)
            return {}

    async def complete_enrollment(
        self, 
        enrollment: FaceEnrollment, 
        face_data: Dict[str, Any],
        liveness_data: Dict[str, Any],
        frame: np.ndarray = None
    ) -> tuple[bool, Optional[float]]:
        """Complete enrollment with final face data and return (success, similarity_score)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._sync_complete_enrollment,
            enrollment,
            face_data,
            liveness_data,
            frame
        )

    def _sync_complete_enrollment(
        self,
        enrollment: FaceEnrollment,
        face_data: Dict[str, Any],
        liveness_data: Dict[str, Any],
        frame: np.ndarray = None
    ) -> tuple[bool, Optional[float]]:
        """Synchronous enrollment completion with similarity checking and profile image saving"""
        try:
            import cv2
            from io import BytesIO
            from PIL import Image
            from django.core.files.base import ContentFile
            
            # Get embedding
            embedding = face_data.get("embedding")
            if embedding is None:
                return False, None
            
            # Convert to numpy array if needed
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # VALIDASI: Check if face already exists for another user
            engine_user_id = f"{self.client.client_id}:{self.session.client_user.external_user_id}"
            duplicate_check = self.face_engine.check_face_duplicate(
                embedding=embedding,
                current_user_id=engine_user_id,
                similarity_threshold=0.55  # 85% similarity threshold
            )
            
            if duplicate_check.get('is_duplicate', False):
                conflicting_user = duplicate_check.get('conflicting_user_id', 'Unknown')
                similarity = duplicate_check.get('similarity_score', 0.0)
                
                # Update enrollment status to failed
                enrollment.status = "failed"
                enrollment.metadata = enrollment.metadata or {}
                enrollment.metadata['failure_reason'] = 'duplicate_face'
                enrollment.metadata['conflicting_user_id'] = conflicting_user
                enrollment.metadata['similarity_score'] = similarity
                enrollment.save()
                
                logger.error(
                    f"❌ ENROLLMENT REJECTED: Face already enrolled for user '{conflicting_user}' "
                    f"(similarity: {similarity:.1%}). Current user: {engine_user_id}"
                )
                return False, None
            
            logger.info(f"✅ Face validation passed - No duplicate found for user {engine_user_id}")
            
            # Get face bbox and landmarks
            bbox = face_data.get("bbox")
            landmarks = face_data.get("landmarks")
            
            # Update enrollment with all data
            enrollment.embedding_vector = json.dumps(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
            enrollment.face_quality_score = face_data.get("quality", 0.0)
            enrollment.liveness_score = liveness_data.get("liveness_score", 0.0)
            enrollment.anti_spoofing_score = face_data.get("anti_spoofing_score", 0.0)
            enrollment.status = "completed"
            enrollment.completed_at = timezone.now()
            
            # Update bbox if available
            if bbox is not None:
                if isinstance(bbox, np.ndarray):
                    enrollment.face_bbox = bbox.tolist()
                else:
                    enrollment.face_bbox = bbox
            
            # Update landmarks if available
            if landmarks is not None:
                if isinstance(landmarks, np.ndarray):
                    enrollment.face_landmarks = landmarks.tolist()
                else:
                    enrollment.face_landmarks = landmarks
            
            enrollment.save()
            
            # Save embedding to face engine database using save_embedding method
            
            # Convert to numpy array if needed
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Save embedding with metadata
            embedding_id = self.face_engine.save_embedding(
                user_id=engine_user_id,
                embedding=embedding,
                metadata={
                    "enrollment_id": str(enrollment.id),
                    "client_id": self.client.client_id,
                    "external_user_id": self.session.client_user.external_user_id,
                    "created_at": timezone.now().timestamp()
                }
            )
            
            if not embedding_id:
                logger.error(f"Failed to save embedding for user {engine_user_id}")
                return False, None
            
            # Calculate similarity with old_profile_photo and save profile_image
            similarity_score = None
            client_user = self.session.client_user
            
            # Save the current frame as profile_image if frame is provided
            if frame is not None:
                try:
                    # Convert frame to JPEG
                    success, buffer = cv2.imencode('.jpg', frame)
                    if success:
                        image_file = ContentFile(buffer.tobytes())
                        # Save with timestamp to avoid collisions
                        filename = f"enrollment_{client_user.external_user_id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        client_user.profile_image.save(filename, image_file, save=False)
                        logger.info(f"✅ Saved profile_image for user {client_user.external_user_id}")
                except Exception as img_error:
                    logger.error(f"Failed to save profile_image: {img_error}")
            
            # Calculate similarity with old_profile_photo if it exists
            if client_user.old_profile_photo:
                try:
                    # Load old profile photo from storage (MinIO or local)
                    from PIL import Image
                    import io
                    
                    # Read file from storage (works with MinIO and local)
                    old_photo_file = client_user.old_profile_photo.open('rb')
                    old_photo_bytes = old_photo_file.read()
                    old_photo_file.close()
                    
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(old_photo_bytes, np.uint8)
                    old_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if old_image is not None:
                        logger.info(f"📸 Loaded old profile photo from storage, size: {old_image.shape}")
                        
                        # Detect face and get embedding from old photo
                        old_face_result = self.face_engine.detect_faces(old_image)
                        
                        if old_face_result and len(old_face_result) > 0:
                            # Get first face
                            old_face_data = old_face_result[0]
                            old_embedding = old_face_data.get('embedding')
                            
                            if old_embedding is not None:
                                # Convert to numpy if needed
                                if not isinstance(old_embedding, np.ndarray):
                                    old_embedding = np.array(old_embedding)
                                
                                # Calculate cosine similarity
                                similarity_score = float(np.dot(embedding, old_embedding) / 
                                                       (np.linalg.norm(embedding) * np.linalg.norm(old_embedding)))
                                
                                # Save similarity score to ClientUser
                                client_user.similarity_with_old_photo = similarity_score
                                logger.info(f"✅ Calculated similarity with old photo: {similarity_score:.3f} ({similarity_score:.1%})")
                            else:
                                logger.warning("No embedding found in old profile photo")
                        else:
                            logger.warning("No face detected in old profile photo")
                    else:
                        logger.warning(f"Failed to decode old profile photo from storage")
                        
                except Exception as sim_error:
                    logger.error(f"Error calculating similarity with old photo: {sim_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Continue enrollment even if similarity calculation fails
            
            # Update ClientUser status
            client_user.is_enrolled = True
            client_user.enrollment_completed_at = timezone.now()
            
            # Save with updated fields
            update_fields = ['is_enrolled', 'enrollment_completed_at']
            if frame is not None:
                update_fields.append('profile_image')
            if similarity_score is not None:
                update_fields.append('similarity_with_old_photo')
            
            client_user.save(update_fields=update_fields)
            
            logger.info(f"✅ Enrollment completed for user {engine_user_id} - Embedding ID: {embedding_id}")
            logger.info(f"✅ Updated ClientUser {client_user.external_user_id} - is_enrolled: True")
            logger.info(f"✅ Updated FaceEnrollment - Quality: {enrollment.face_quality_score:.3f}, Liveness: {enrollment.liveness_score:.3f}")
            if similarity_score is not None:
                logger.info(f"✅ Similarity with old photo: {similarity_score:.3f}")
            return True, similarity_score
            
        except Exception as e:
            logger.error(f"Error completing enrollment: {e}", exc_info=True)
            return False, None

    async def perform_authentication(
        self,
        face_data: Dict[str, Any],
        liveness_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform face authentication"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._sync_perform_authentication,
            face_data,
            liveness_data
        )

    def _sync_perform_authentication(
        self,
        face_data: Dict[str, Any],
        liveness_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synchronous authentication"""
        try:
            embedding = face_data.get("embedding")
            if embedding is None:
                return {"success": False, "error": "No embedding available"}
            
            # Convert to numpy array if needed
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Get target user if verification
            target_user_id = None
            if self.session.session_type == "verification" and self.session.client_user:
                target_user_id = f"{self.client.client_id}:{self.session.client_user.external_user_id}"
            
            # Search for similar faces in ChromaDB
            matches = self.face_engine.embedding_store.search_similar(
                embedding=embedding,
                top_k=5,
                threshold=0.6
            )
            
            if not matches:
                return {
                    "success": False,
                    "error": "Face not recognized"
                }
            
            # Get best match
            best_match = matches[0]
            matched_user_id = best_match.get('metadata', {}).get('user_id')
            
            if not matched_user_id:
                # Try to extract from embedding_id
                embedding_id = best_match.get('embedding_id', '')
                if '_' in embedding_id:
                    matched_user_id = embedding_id.rsplit('_', 1)[0]
            
            # If target_user_id specified, verify it matches
            if target_user_id and matched_user_id != target_user_id:
                return {
                    "success": False,
                    "error": "Face does not match specified user"
                }
            
            # Extract external user ID (remove client_id prefix if present)
            if matched_user_id and ':' in matched_user_id:
                external_user_id = matched_user_id.split(':', 1)[1]
            else:
                external_user_id = matched_user_id
            
            return {
                "success": True,
                "user_id": external_user_id,
                "confidence": best_match.get("similarity", 0.0)
            }
                
        except Exception as e:
            logger.error(f"Error performing authentication: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # Database operations
    @database_sync_to_async
    def get_session(self, session_token: str) -> Optional[AuthenticationSession]:
        """Get authentication session by token"""
        try:
            return AuthenticationSession.objects.select_related('client', 'client_user').get(
                session_token=session_token
            )
        except AuthenticationSession.DoesNotExist:
            return None

    @database_sync_to_async
    def get_client(self, client_id: int) -> Optional[Client]:
        """Get client by ID"""
        try:
            return Client.objects.get(id=client_id)
        except Client.DoesNotExist:
            return None

    @database_sync_to_async
    def get_enrollment(self, session_id) -> Optional[FaceEnrollment]:
        """Get enrollment by session"""
        try:
            return FaceEnrollment.objects.get(enrollment_session_id=session_id)
        except FaceEnrollment.DoesNotExist:
            return None

    @database_sync_to_async
    def update_session_status(self, status: str):
        """Update session status"""
        if self.session:
            self.session.status = status
            if status == "completed":
                self.session.completed_at = timezone.now()
            self.session.save(update_fields=["status", "completed_at"])

    @database_sync_to_async
    def update_session_metadata(self, metadata: Dict[str, Any]):
        """Update session metadata"""
        if self.session:
            self.session.metadata = metadata
            self.session.save(update_fields=["metadata"])

    async def safe_send(self, text_data=None, bytes_data=None):
        """Send data only if connection is still open"""
        if not self._is_connected:
            logger.debug(f"Skipping send on closed connection for session {self.session_token}")
            return False
        try:
            await super().send(text_data=text_data, bytes_data=bytes_data)
            return True
        except Exception as e:
            logger.warning(f"Failed to send data: {e}")
            self._is_connected = False
            return False

    async def send_error(self, message: str, code: int = 4000):
        """Send error message to client"""
        if not self._is_connected:
            logger.debug(f"Skipping error send on closed connection: {message}")
            return
        await self.safe_send(text_data=json.dumps({
            "type": "error",
            "error": message,
            "code": code
        }))
    
    async def send_json_safe(self, data: Dict[str, Any]):
        """
        Safely send JSON data with automatic NumPy type conversion
        """
        # Convert any NumPy types to native Python types
        safe_data = convert_numpy_types(data)
        await self.safe_send(text_data=json.dumps(safe_data))

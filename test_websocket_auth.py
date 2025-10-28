#!/usr/bin/env python3
"""
Test script for WebSocket-based face authentication and enrollment.
This script demonstrates how to use the WebSocket API for processing face images.
"""

import asyncio
import base64
import hashlib
import json
import sys

import cv2
import numpy as np
import requests
import websockets
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class FaceAuthWebSocketClient:
    """Client for WebSocket-based face authentication and enrollment"""

    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.ws = None
        self.session_token = None
        self.jwt_token = None

    def authenticate_client(self) -> dict:
        """Authenticate client and get JWT token"""
        url = f"{self.base_url}/api/core/auth/client/"
        data = {"api_key": self.api_key, "api_secret": self.secret_key}

        # headers = {
        #     'accept': '*/*',
        #     'Content-Type': 'application/json',
        #     'X-CSRFTOKEN': 'xcohaEKEMHIMA67GLL5Y1149YEFX6Nkma1EXneSs81jWXNRVELlwQk0scmvc5c48'
        # }

        # print(f"url: {url}\n data: {data}")

        print("🔑 Authenticating client...")
        response = requests.post(url, json=data)
        response.raise_for_status()

        auth_data = response.json()
        self.jwt_token = auth_data.get("access_token")
        print("✅ Client authenticated successfully!")
        print(f"   Token: {self.jwt_token}")
        return auth_data

    def create_enrollment_session(self, user_id: str, target_samples: int = 3) -> dict:
        """Create an enrollment session via REST API"""
        url = f"{self.base_url}/api/auth/enrollment/"
        headers = {
            "Authorization": f"JWT {self.jwt_token}",
            "Content-Type": "application/json",
        }
        data = {
            "user_id": user_id,
            "session_type": "webcam",
            "metadata": {
                "target_samples": target_samples,
                "device_info": {"platform": "test_script"},
            },
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def create_authentication_session(
        self, user_id: str = None, require_liveness: bool = True
    ) -> dict:
        """Create an authentication session via REST API"""
        url = f"{self.base_url}/api/auth/authentication/"
        headers = {
            "Authorization": f"JWT {self.jwt_token}",
            "Content-Type": "application/json",
        }
        data = {
            "session_type": "webcam",
            "require_liveness": require_liveness,
            "metadata": {
                "min_frames_required": 3,
                "required_blinks": 1,
                "device_info": {"platform": "test_script"},
            },
        }

        if user_id:
            data["user_id"] = user_id

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def decrypt_response(self, encrypted_payload: str) -> dict:
        """Decrypt the encrypted response from the server"""
        try:
            # Decode base64
            print(encrypted_payload)
            encrypted_data = base64.b64decode(encrypted_payload)

            # Extract IV (first 16 bytes)
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]

            # Derive key from API key (server uses api_key, not secret_key!)
            key = hashlib.sha256(self.api_key.encode()).digest()

            # Decrypt
            cipher = Cipher(
                algorithms.AES(key), modes.CBC(iv), backend=default_backend()
            )
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()

            # Unpad
            unpadder = padding.PKCS7(128).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()

            # Parse JSON
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            print(f"Error decrypting response: {e}")
            return None

    async def connect_websocket(self, websocket_url: str):
        """Connect to WebSocket"""
        # Convert http/https to ws/wss if needed
        if websocket_url.startswith("http://"):
            websocket_url = websocket_url.replace("http://", "ws://", 1)
        elif websocket_url.startswith("https://"):
            websocket_url = websocket_url.replace("https://", "wss://", 1)

        print(f"Connecting to WebSocket: {websocket_url}")
        self.ws = await websockets.connect(websocket_url)
        print("WebSocket connected!")

    async def send_frame(self, frame):
        """Send a frame to the WebSocket"""
        # Convert frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # Encode to base64
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        image_data = f"data:image/jpeg;base64,{image_base64}"

        # Send message
        message = {"type": "frame", "image": image_data}

        await self.ws.send(json.dumps(message))

    async def receive_messages(self):
        """Receive and handle messages from WebSocket"""
        async for message in self.ws:
            data = json.loads(message)
            await self.handle_message(data)

    async def handle_message(self, data: dict):
        """Handle incoming WebSocket messages"""
        msg_type = data.get("type")

        # Store latest visual data and response for overlay
        if "visual_data" in data:
            self.latest_visual_data = data.get("visual_data", {})
        self.latest_response = data

        if msg_type == "connection_established":
            print(f"✅ Connection established: {data.get('session_type')} session")

        elif msg_type == "frame_rejected":
            # HANDLE REJECTED FRAMES - display prominently
            obstacles = data.get("obstacles", [])
            obstacle_conf = data.get("obstacle_confidence", {})
            reason = data.get("reason", "Frame rejected")

            print("\n⛔ FRAME REJECTED!")
            print(f"   Reason: {reason}")
            print(f"   Obstacles detected: {', '.join(obstacles)}")
            for obs in obstacles:
                conf = obstacle_conf.get(obs, 0)
                print(f"      - {obs}: {conf:.2f} confidence")
            print("   ⚠️  Please remove obstacles and try again")

        elif msg_type == "frame_processed":
            if data.get("success"):
                visual_data = data.get("visual_data", {})
                obstacles = data.get("obstacles", [])

                # Check status flags
                blinks_ok = data.get("blinks_ok", False)
                motion_ok = data.get("motion_ok", False)
                no_obstacles = data.get("no_obstacles", True)

                print(
                    f"\n📊 Frame {data.get('frames_processed')}/{
                        data.get('target_samples', data.get('min_frames_required', '?'))
                    }"
                )
                print(f"   Liveness Score: {data.get('liveness_score', 0):.2f}")
                print(
                    f"   Blinks: {data.get('blinks_detected', 0)}/{
                        data.get('blinks_required', 1)
                    } {'✅' if blinks_ok else '❌'}"
                )
                print(
                    f"   Motion: {data.get('motion_events', 0)}/{
                        data.get('motion_required', 1)
                    } {'✅' if motion_ok else '❌'} (score: {
                        data.get('motion_score', 0):.2f})"
                )
                print(f"   No Obstacles: {'✅' if no_obstacles else '❌'}")
                print(
                    f"   Quality: {data.get('quality_score', 0):.2f} (threshold: {
                        data.get('quality_threshold', 0.65):.2f})"
                )

                if obstacles:
                    print(f"   ⚠️  Obstacles: {', '.join(obstacles)}")

                liveness_ind = visual_data.get("liveness_indicators", {})
                if liveness_ind:
                    print(
                        f"   👁️  EAR: {liveness_ind.get('ear', 0):.3f} | Blink: {
                            '✓' if liveness_ind.get('blink_verified') else '✗'
                        }"
                    )

                print(f"   💬 {data.get('message', 'Continue')}")
            else:
                error = data.get("error", "Unknown error")
                obstacles = data.get("obstacles", [])
                print(f"\n❌ Frame processing failed: {error}")
                if obstacles:
                    print(f"   ⚠️  Obstacles: {', '.join(obstacles)}")

        elif msg_type == "enrollment_complete":
            print("\n" + "=" * 60)
            print("🎉 ENROLLMENT COMPLETE!")
            print("=" * 60)
            print(f"   Enrollment ID: {data.get('enrollment_id')}")
            print(f"   Frames processed: {data.get('frames_processed')}")
            print(f"   Blinks detected: {data.get('blinks_detected', 0)}")
            print(f"   Motion verified: {'✓' if data.get('motion_verified') else '✗'}")
            print(f"   Quality score: {data.get('quality_score', 0):.2f}")

            # Decrypt the encrypted data
            encrypted_data = data.get("encrypted_data", {})
            encrypted_payload = encrypted_data.get("encrypted_payload")

            if encrypted_payload:
                print("\n🔓 Decrypting response...")
                decrypted = self.decrypt_response(encrypted_payload)
                if decrypted:
                    print("   ✅ Decrypted data:")
                    print(f"      - ID: {decrypted.get('id')}")
                    print(f"      - Timestamp: {decrypted.get('timestamp')}")
                    print(f"      - Session Type: {decrypted.get('session_type')}")
                else:
                    print("   ⚠️  Failed to decrypt response")
            print("=" * 60)

        elif msg_type == "authentication_complete":
            print("\n" + "=" * 60)
            if data.get("authenticated"):
                print("🎉 AUTHENTICATION SUCCESSFUL!")
            else:
                print("❌ AUTHENTICATION FAILED")
            print("=" * 60)
            print(f"   Authenticated: {data.get('authenticated')}")
            print(f"   User ID: {data.get('user_id', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 0):.2%}")
            print(f"   Frames processed: {data.get('frames_processed')}")
            if data.get("blinks_detected") is not None:
                print(f"   Blinks detected: {data.get('blinks_detected')}")
                print(
                    f"   Motion verified: {'✓' if data.get('motion_verified') else '✗'}"
                )

            # Decrypt the encrypted data if authenticated
            if data.get("authenticated"):
                encrypted_data = data.get("encrypted_data", {})
                encrypted_payload = encrypted_data.get("encrypted_payload")

                if encrypted_payload:
                    print("\n🔓 Decrypting response...")
                    decrypted = self.decrypt_response(encrypted_payload)
                    if decrypted:
                        print("   ✅ Decrypted data:")
                        print(f"      - ID: {decrypted.get('id')}")
                        print(f"      - Timestamp: {decrypted.get('timestamp')}")
                        print(f"      - Session Type: {decrypted.get('session_type')}")
                        print(
                            f"      - Confidence: {decrypted.get('confidence', 0):.2%}"
                        )
                    else:
                        print("   ⚠️  Failed to decrypt response")
            print("=" * 60)

        elif msg_type == "error":
            print(f"\n❌ Error: {data.get('error')} (code: {data.get('code')})")

        elif msg_type == "pong":
            print("🏓 Pong received")

        else:
            print(f"📨 Unknown message type: {msg_type}")

    async def send_camera_frames(self, camera_index: int = 0, duration: int = 10):
        """Capture and send frames from camera"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return

        print(f"📹 Starting camera capture for {duration} seconds...")
        print("   Press 'q' to quit early")
        start_time = asyncio.get_event_loop().time()
        frame_count = 0

        # Store latest visual data for overlay
        self.latest_visual_data = {}
        self.latest_response = {}

        try:
            while True:
                # Check if duration exceeded
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= duration:
                    print(f"\n⏱️  Time limit reached ({duration}s)")
                    break

                # Check if WebSocket is still open
                if self.ws.close_code is not None:
                    print("\n🔌 WebSocket closed")
                    break

                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break

                # Draw overlays from latest data
                display_frame = self._draw_overlays(frame.copy())

                # Send frame
                await self.send_frame(frame)
                frame_count += 1

                # Display frame with overlays
                cv2.imshow("Face Recognition", display_frame)

                # Wait and check for quit key
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    print("\n⏹️  Stopped by user")
                    break

                # Control frame rate (10 FPS)
                await asyncio.sleep(0.1)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"📊 Sent {frame_count} frames")

    def _draw_overlays(self, frame):
        """Draw visual overlays on frame"""
        try:
            h, w = frame.shape[:2]
            visual_data = self.latest_visual_data
            response = self.latest_response

            # Draw face bounding box
            bbox = visual_data.get("face_bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw corner markers
                corner_len = 20
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (0, 255, 0), 3)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (0, 255, 0), 3)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 0), 3)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (0, 255, 0), 3)
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (0, 255, 0), 3)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (0, 255, 0), 3)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 0), 3)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (0, 255, 0), 3)

            # Draw comprehensive face mesh landmarks
            face_mesh = visual_data.get("face_mesh_landmarks", {})

            # Draw face oval (cyan thin line)
            face_oval = face_mesh.get("face_oval", [])
            if face_oval and len(face_oval) > 2:
                pts = np.array(face_oval, np.int32)
                cv2.polylines(frame, [pts], True, (255, 255, 0), 1)

            # Draw eyebrows (green)
            left_eyebrow = face_mesh.get("left_eyebrow", [])
            if left_eyebrow and len(left_eyebrow) > 2:
                pts = np.array(left_eyebrow, np.int32)
                cv2.polylines(frame, [pts], False, (0, 255, 100), 2)

            right_eyebrow = face_mesh.get("right_eyebrow", [])
            if right_eyebrow and len(right_eyebrow) > 2:
                pts = np.array(right_eyebrow, np.int32)
                cv2.polylines(frame, [pts], False, (0, 255, 100), 2)

            # Draw nose (white)
            nose = face_mesh.get("nose", [])
            if nose and len(nose) > 2:
                pts = np.array(nose, np.int32)
                cv2.polylines(frame, [pts], False, (255, 255, 255), 2)

            # Draw lips (red)
            lips = face_mesh.get("lips", [])
            if lips and len(lips) > 2:
                pts = np.array(lips, np.int32)
                cv2.polylines(frame, [pts], True, (0, 100, 255), 2)

            # Draw eye regions for blink detection (yellow with fill)
            eye_regions = visual_data.get("eye_regions", {})
            left_eye = eye_regions.get("left_eye", [])
            right_eye = eye_regions.get("right_eye", [])

            if left_eye and len(left_eye) > 2:
                pts = np.array(left_eye, np.int32)
                # Create semi-transparent overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 255))
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

            if right_eye and len(right_eye) > 2:
                pts = np.array(right_eye, np.int32)
                # Create semi-transparent overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 255))
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

            # Draw status information panel (top-left)
            y_offset = 30
            panel_height = 200
            panel_width = 280

            # Semi-transparent background panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Liveness indicators
            liveness_ind = visual_data.get("liveness_indicators", {})
            if liveness_ind:
                blinks = liveness_ind.get("blinks", 0)
                ear = liveness_ind.get("ear", 0)
                blink_verified = liveness_ind.get("blink_verified", False)
                motion_verified = liveness_ind.get("motion_verified", False)
                motion_events = liveness_ind.get("motion_events", 0)
                frame_counter = liveness_ind.get("frame_counter", 0)

                # Frame counter
                cv2.putText(
                    frame,
                    f"Frame: {frame_counter}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                y_offset += 25

                # Blink status with icon
                blink_color = (0, 255, 0) if blink_verified else (0, 0, 255)
                blink_icon = "[OK]" if blink_verified else "[--]"
                cv2.putText(
                    frame,
                    f"{blink_icon} Blinks: {blinks}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    blink_color,
                    2,
                )
                y_offset += 25

                # EAR value with bar
                ear_color = (0, 255, 0) if ear > 0.15 else (0, 255, 255)
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    ear_color,
                    2,
                )
                # Draw EAR bar
                bar_length = int(min(ear * 500, 100))
                cv2.rectangle(
                    frame,
                    (150, y_offset - 10),
                    (150 + bar_length, y_offset + 5),
                    ear_color,
                    -1,
                )
                y_offset += 25

                # Motion status with icon
                motion_color = (0, 255, 0) if motion_verified else (0, 165, 255)
                motion_icon = "[OK]" if motion_verified else "[??]"
                motion_text = (
                    f"OK ({motion_events})" if motion_verified else "Move head"
                )
                cv2.putText(
                    frame,
                    f"{motion_icon} Motion: {motion_text}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    motion_color,
                    2,
                )
                y_offset += 25

            # Quality and liveness from response
            if response.get("quality_score") is not None:
                quality = response.get("quality_score", 0)
                quality_threshold = response.get("quality_threshold", 0.65)
                quality_color = (
                    (0, 255, 0) if quality >= quality_threshold else (0, 165, 255)
                )
                quality_icon = "[OK]" if quality >= quality_threshold else "[!!]"
                cv2.putText(
                    frame,
                    f"{quality_icon} Quality: {quality:.2f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    quality_color,
                    2,
                )
                # Draw quality bar
                bar_length = int(quality * 100)
                cv2.rectangle(
                    frame,
                    (150, y_offset - 10),
                    (150 + bar_length, y_offset + 5),
                    quality_color,
                    -1,
                )
                y_offset += 25

            if response.get("liveness_verified") is not None:
                liveness_verified = response.get("liveness_verified", False)
                liveness_color = (0, 255, 0) if liveness_verified else (0, 0, 255)
                liveness_icon = "[OK]" if liveness_verified else "[XX]"
                cv2.putText(
                    frame,
                    f"{liveness_icon} Liveness: {'VERIFIED' if liveness_verified else 'NOT OK'}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    liveness_color,
                    2,
                )
                y_offset += 25

            # Obstacles warning - PROMINENT DISPLAY
            obstacles = response.get("obstacles", [])
            if obstacles:
                # Draw BIG RED WARNING BANNER if obstacles detected
                banner_height = 60
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (0, h // 2 - banner_height // 2),
                    (w, h // 2 + banner_height // 2),
                    (0, 0, 255),
                    -1,
                )
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                # Draw border
                cv2.rectangle(
                    frame,
                    (0, h // 2 - banner_height // 2),
                    (w, h // 2 + banner_height // 2),
                    (0, 0, 255),
                    3,
                )

                # Warning text
                warning_text = "⛔ REMOVE OBSTACLES ⛔"
                (text_w, text_h), _ = cv2.getTextSize(
                    warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                )
                cv2.putText(
                    frame,
                    warning_text,
                    (w // 2 - text_w // 2, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3,
                )

                # List obstacles
                obstacle_text = ", ".join(obstacles).upper()
                (text_w2, text_h2), _ = cv2.getTextSize(
                    obstacle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.putText(
                    frame,
                    obstacle_text,
                    (w // 2 - text_w2 // 2, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Also show in panel
                y_offset += 5
                cv2.putText(
                    frame,
                    "OBSTACLES:",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                y_offset += 20
                for obs in obstacles:
                    obstacle_conf = response.get("obstacle_confidence", {})
                    conf = obstacle_conf.get(obs, 0)
                    cv2.putText(
                        frame,
                        f"  {obs}: {conf:.2f}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                    y_offset += 18

            # Progress message at bottom
            message = response.get("message", "")
            if message:
                # Word wrap message if too long
                max_width = w - 20
                # Split by pipe for multi-part messages
                words = message.split(" | ")

                # Draw semi-transparent background for message
                msg_height = len(words) * 22 + 10
                overlay = frame.copy()
                cv2.rectangle(
                    overlay, (5, h - msg_height - 5), (w - 5, h - 5), (0, 0, 0), -1
                )
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                # Draw message lines at bottom
                msg_y = h - msg_height + 5
                for line in words:
                    cv2.putText(
                        frame,
                        line,
                        (10, msg_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    msg_y += 22

            return frame

        except Exception as e:
            # Return original frame if overlay fails
            print(f"Error drawing overlays: {e}")
            return frame

    async def run_enrollment(self, user_id: str, camera_index: int = 0):
        """Run complete enrollment process"""
        print(f"🔐 Starting enrollment for user: {user_id}\n")

        # Step 1: Authenticate client
        self.authenticate_client()
        print()

        # Step 2: Create session
        session = self.create_enrollment_session(user_id)
        print(f"websocket_url: {session['websocket_url']}")
        print(f"✅ Session created: {session['session_token']}")
        print(f"   Target samples: {session['target_samples']}")
        print(f"   WebSocket URL: {session['websocket_url']}\n")

        # Step 3: Connect to WebSocket
        await self.connect_websocket(session["websocket_url"])

        # Start message receiver
        receiver_task = asyncio.create_task(self.receive_messages())

        # Send camera frames
        await self.send_camera_frames(camera_index, duration=30)

        # Close WebSocket
        await self.ws.close()

        # Wait for receiver to finish
        try:
            await asyncio.wait_for(receiver_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass

    async def run_authentication(self, user_id: str = None, camera_index: int = 0):
        """Run complete authentication process"""
        auth_type = "Identification" if user_id is None else "Verification"
        print(
            f"🔐 Starting {auth_type.lower()}"
            + (f" for user: {user_id}" if user_id else "")
            + "\n"
        )

        # Step 1: Authenticate client
        self.authenticate_client()
        print()

        # Step 2: Create session
        session = self.create_authentication_session(user_id)
        print(f"✅ Session created: {session['session_token']}")
        print(f"   Session type: {session['session_type']}")
        print(f"   WebSocket URL: {session['websocket_url']}\n")

        # Step 3: Connect to WebSocket
        await self.connect_websocket(session["websocket_url"])

        # Start message receiver
        receiver_task = asyncio.create_task(self.receive_messages())

        # Send camera frames
        await self.send_camera_frames(camera_index, duration=30)

        # Close WebSocket
        await self.ws.close()

        # Wait for receiver to finish
        try:
            await asyncio.wait_for(receiver_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass


async def main():
    """Main function"""
    if len(sys.argv) < 4:
        print(
            "Usage: python test_websocket_auth.py <API_KEY> <SECRET_KEY> <BASE_URL> [enrollment|authentication] [user_id]"
        )
        print("\nExamples:")
        print("  # Enrollment")
        print(
            "  python test_websocket_auth.py YOUR_API_KEY YOUR_SECRET_KEY http://localhost:8000 enrollment user123"
        )
        print("\n  # Verification (with user_id)")
        print(
            "  python test_websocket_auth.py YOUR_API_KEY YOUR_SECRET_KEY http://localhost:8000 authentication user123"
        )
        print("\n  # Identification (without user_id)")
        print(
            "  python test_websocket_auth.py YOUR_API_KEY YOUR_SECRET_KEY http://localhost:8000 authentication"
        )
        sys.exit(1)

    api_key = sys.argv[1]
    secret_key = sys.argv[2]
    base_url = sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "enrollment"
    user_id = sys.argv[5] if len(sys.argv) > 5 else None

    # Create client
    client = FaceAuthWebSocketClient(api_key, secret_key, base_url)

    try:
        if mode == "enrollment":
            if not user_id:
                print("❌ User ID is required for enrollment")
                sys.exit(1)
            await client.run_enrollment(user_id)
        elif mode == "authentication":
            await client.run_authentication(user_id)
        else:
            print(f"❌ Unknown mode: {mode}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

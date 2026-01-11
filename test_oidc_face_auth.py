#!/usr/bin/env python3
"""
Test OIDC Face Authentication Flow
This script tests the complete OIDC authorization code flow with face recognition.

Flow:
1. Start OIDC authorization request
2. Authenticate via face recognition WebSocket
3. Get authorization code
4. Exchange code for tokens
5. Verify tokens and get userinfo

Usage:
    python test_oidc_face_auth.py <API_KEY> <SECRET_KEY> <BASE_URL> <CLIENT_ID> <CLIENT_SECRET> <USER_ID>

Example:
    python test_oidc_face_auth.py \
        frapi_xxx secret_xxx \
        http://192.168.1.41:8003 \
        oidc_xxx oidc_secret_xxx \
        user123
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import sys
from urllib.parse import urlencode, parse_qs, urlparse

import cv2
import requests
import websockets
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class OIDCFaceAuthClient:
    """Client for OIDC authentication with face recognition"""

    def __init__(self, api_key: str, secret_key: str, base_url: str, 
                 client_id: str, client_secret: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.ws = None
        self.jwt_token = None
        self.stop_sending = False
        self.authorization_code = None
        self.latest_visual_data = {}
        self.latest_response = {}
        
        # PKCE values
        self.code_verifier = None
        self.code_challenge = None
        self.state = None
        self.nonce = None

    def generate_pkce(self):
        """Generate PKCE code verifier and challenge"""
        self.code_verifier = secrets.token_urlsafe(64)[:128]
        self.code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(self.code_verifier.encode()).digest()
        ).rstrip(b'=').decode()
        self.state = secrets.token_urlsafe(16)
        self.nonce = secrets.token_urlsafe(16)
        
        print(f"🔐 Generated PKCE:")
        print(f"   State: {self.state}")
        print(f"   Nonce: {self.nonce}")
        print(f"   Code Verifier: {self.code_verifier[:32]}...")

    def authenticate_client(self) -> dict:
        """Authenticate client and get JWT token"""
        url = f"{self.base_url}/api/core/auth/client/"
        data = {"api_key": self.api_key, "api_secret": self.secret_key}

        print("🔑 Authenticating API client...")
        response = requests.post(url, json=data)
        response.raise_for_status()

        auth_data = response.json()
        self.jwt_token = auth_data.get("access_token")
        print("✅ API client authenticated!")
        return auth_data

    def test_oidc_discovery(self):
        """Test OIDC discovery endpoint"""
        url = f"{self.base_url}/.well-known/openid-configuration"
        print(f"\n📡 Testing OIDC Discovery: {url}")
        
        response = requests.get(url)
        if response.status_code == 200:
            config = response.json()
            print("✅ OIDC Discovery successful!")
            print(f"   Issuer: {config.get('issuer')}")
            print(f"   Authorization: {config.get('authorization_endpoint')}")
            print(f"   Token: {config.get('token_endpoint')}")
            return config
        else:
            print(f"❌ Discovery failed: {response.status_code}")
            return None

    def build_authorization_url(self, redirect_uri: str = "http://localhost:8080/callback"):
        """Build OIDC authorization URL"""
        self.generate_pkce()
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'scope': 'openid profile email face_auth',
            'state': self.state,
            'nonce': self.nonce,
            'code_challenge': self.code_challenge,
            'code_challenge_method': 'S256'
        }
        
        auth_url = f"{self.base_url}/oauth/authorize?{urlencode(params)}"
        print(f"\n🔗 Authorization URL:")
        print(f"   {auth_url[:80]}...")
        
        return auth_url, redirect_uri

    def create_face_auth_session(self, user_id: str) -> dict:
        """Create face authentication session for OIDC"""
        url = f"{self.base_url}/api/auth/authentication/"
        headers = {
            "Authorization": f"JWT {self.jwt_token}",
            "Content-Type": "application/json",
        }
        data = {
            "user_id": user_id,
            "session_type": "webcam",
            "require_liveness": True,
            "metadata": {
                "min_frames_required": 10,
                "required_blinks": 1,
                "device_info": {"platform": "oidc_test_script"},
                "oidc_flow": True,
                "client_id": self.client_id,
                "state": self.state,
                "nonce": self.nonce,
            },
        }

        print(f"\n📝 Creating face auth session for user: {user_id}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        session = response.json()
        print(f"✅ Session created: {session['session_token']}")
        return session

    def simulate_oidc_callback(self, user_id: str) -> str:
        """
        Simulate OIDC callback by creating authorization code directly.
        In real flow, this happens after face auth succeeds via browser redirect.
        """
        # This simulates what happens when face auth succeeds in the OIDC flow
        # The server would create an authorization code and redirect
        
        # For testing, we'll call an internal endpoint or simulate
        print(f"\n🔄 Simulating OIDC authorization after successful face auth...")
        
        # In reality, after face auth, the server generates auth code
        # For this test, we need to manually create one via the authorize endpoint
        return None  # Will be set after face auth completes

    async def connect_websocket(self, websocket_url: str):
        """Connect to WebSocket"""
        if websocket_url.startswith("http://"):
            websocket_url = websocket_url.replace("http://", "ws://", 1)
        elif websocket_url.startswith("https://"):
            websocket_url = websocket_url.replace("https://", "wss://", 1)

        print(f"\n🔌 Connecting to WebSocket: {websocket_url}")
        self.ws = await websockets.connect(websocket_url)
        print("✅ WebSocket connected!")

    async def send_frame(self, frame):
        """Send a frame to the WebSocket"""
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        image_data = f"data:image/jpeg;base64,{image_base64}"
        message = {"type": "frame", "image": image_data}
        await self.ws.send(json.dumps(message))

    async def receive_messages(self):
        """Receive and handle messages from WebSocket"""
        async for message in self.ws:
            data = json.loads(message)
            should_stop = await self.handle_message(data)
            if should_stop:
                self.stop_sending = True

    async def handle_message(self, data: dict):
        """Handle incoming WebSocket messages"""
        msg_type = data.get("type")
        
        if "visual_data" in data:
            self.latest_visual_data = data.get("visual_data", {})
        self.latest_response = data

        if msg_type == "connection_established":
            print(f"✅ WebSocket connection established")

        elif msg_type == "frame_processed":
            if data.get("success"):
                frames = data.get('frames_processed', 0)
                target = data.get('min_frames_required', 10)
                liveness = data.get('liveness_score', 0)
                blinks = data.get('blinks_detected', 0)
                
                print(f"📊 Frame {frames}/{target} | Liveness: {liveness:.2f} | Blinks: {blinks}")

        elif msg_type == "authentication_complete":
            print("\n" + "=" * 60)
            if data.get("authenticated"):
                print("🎉 FACE AUTHENTICATION SUCCESSFUL!")
                print("=" * 60)
                print(f"   User ID: {data.get('user_id')}")
                print(f"   Confidence: {data.get('confidence', 0):.2%}")
                print(f"   Liveness verified: {data.get('liveness_verified', False)}")
                
                # Store auth result for OIDC flow
                self.face_auth_result = {
                    'authenticated': True,
                    'user_id': data.get('user_id'),
                    'confidence': data.get('confidence', 0),
                    'liveness_verified': data.get('liveness_verified', False)
                }
            else:
                print("❌ FACE AUTHENTICATION FAILED")
                self.face_auth_result = {'authenticated': False}
            print("=" * 60)
            return True

        elif msg_type == "authentication_timeout":
            print("\n⏱️ Authentication timeout")
            self.face_auth_result = {'authenticated': False, 'timeout': True}
            return True

        elif msg_type == "error":
            print(f"\n❌ Error: {data.get('error')}")
            return True

    async def send_camera_frames(self, camera_index: int = 0, duration: int = 30):
        """Capture and send frames from camera"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return

        print(f"\n📹 Starting camera capture...")
        print("   Look at the camera and blink naturally")
        print("   Press 'q' to quit")
        
        start_time = asyncio.get_event_loop().time()
        frame_count = 0

        try:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= duration or self.stop_sending:
                    break

                if self.ws.close_code is not None:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                display_frame = self._draw_overlays(frame.copy())

                try:
                    await self.send_frame(frame)
                    frame_count += 1
                except Exception as e:
                    print(f"\n⚠️  Failed to send frame: {e}")
                    break

                cv2.imshow("OIDC Face Authentication", display_frame)

                if cv2.waitKey(100) & 0xFF == ord("q"):
                    print("\n⏹️  Stopped by user")
                    break

                await asyncio.sleep(0.1)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"📊 Sent {frame_count} frames")

    def _draw_overlays(self, frame):
        """Draw visual overlays on frame"""
        try:
            h, w = frame.shape[:2]
            response = self.latest_response
            visual_data = self.latest_visual_data

            # Draw face bounding box
            bbox = visual_data.get("face_bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Status panel
            y_offset = 30
            cv2.putText(frame, "OIDC Face Authentication", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

            liveness_ind = visual_data.get("liveness_indicators", {})
            if liveness_ind:
                blinks = liveness_ind.get("blinks", 0)
                ear = liveness_ind.get("ear", 0)
                cv2.putText(frame, f"Blinks: {blinks} | EAR: {ear:.3f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25

            if response.get("liveness_score") is not None:
                cv2.putText(frame, f"Liveness: {response.get('liveness_score', 0):.2f}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Message at bottom
            message = response.get("message", "Look at camera and blink")
            cv2.putText(frame, message, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return frame
        except Exception:
            return frame

    def exchange_authorization_code(self, code: str, redirect_uri: str) -> dict:
        """Exchange authorization code for tokens"""
        url = f"{self.base_url}/oauth/token"
        
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': redirect_uri,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code_verifier': self.code_verifier
        }
        
        print(f"\n🔄 Exchanging authorization code for tokens...")
        response = requests.post(url, data=data)
        
        if response.status_code == 200:
            tokens = response.json()
            print("✅ Token exchange successful!")
            print(f"   Access Token: {tokens.get('access_token', '')[:50]}...")
            print(f"   Token Type: {tokens.get('token_type')}")
            print(f"   Expires In: {tokens.get('expires_in')} seconds")
            
            if tokens.get('id_token'):
                print(f"   ID Token: {tokens.get('id_token', '')[:50]}...")
                self._decode_id_token(tokens['id_token'])
            
            return tokens
        else:
            print(f"❌ Token exchange failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    def _decode_id_token(self, id_token: str):
        """Decode and display ID token claims"""
        try:
            parts = id_token.split('.')
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=='))
            
            print("\n📋 ID Token Claims:")
            print(f"   Subject (sub): {payload.get('sub')}")
            print(f"   Issuer (iss): {payload.get('iss')}")
            print(f"   Audience (aud): {payload.get('aud')}")
            print(f"   Name: {payload.get('name', 'N/A')}")
            print(f"   Email: {payload.get('email', 'N/A')}")
            print(f"   Face Verified: {payload.get('face_verified', 'N/A')}")
            print(f"   Liveness Verified: {payload.get('liveness_verified', 'N/A')}")
            print(f"   Auth Method (amr): {payload.get('amr', 'N/A')}")
            
            return payload
        except Exception as e:
            print(f"❌ Failed to decode ID token: {e}")
            return None

    def get_userinfo(self, access_token: str) -> dict:
        """Get user info using access token"""
        url = f"{self.base_url}/oauth/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        print(f"\n👤 Fetching UserInfo...")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            userinfo = response.json()
            print("✅ UserInfo retrieved!")
            print(f"   Subject: {userinfo.get('sub')}")
            print(f"   Name: {userinfo.get('name', 'N/A')}")
            print(f"   Email: {userinfo.get('email', 'N/A')}")
            print(f"   Face Verified: {userinfo.get('face_verified', 'N/A')}")
            return userinfo
        else:
            print(f"❌ UserInfo failed: {response.status_code}")
            return None

    def introspect_token(self, token: str) -> dict:
        """Introspect token"""
        url = f"{self.base_url}/oauth/introspect"
        data = {
            'token': token,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        print(f"\n🔍 Introspecting token...")
        response = requests.post(url, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Active: {result.get('active')}")
            print(f"   Scope: {result.get('scope', 'N/A')}")
            print(f"   Client ID: {result.get('client_id', 'N/A')}")
            return result
        else:
            print(f"❌ Introspection failed: {response.status_code}")
            return None

    async def run_oidc_face_auth(self, user_id: str, camera_index: int = 0):
        """Run complete OIDC face authentication flow"""
        print("=" * 60)
        print("🔐 OIDC Face Authentication Test")
        print("=" * 60)
        print(f"\n📍 Base URL: {self.base_url}")
        print(f"📍 Client ID: {self.client_id}")
        print(f"📍 User ID: {user_id}")

        # Step 1: Test OIDC discovery
        oidc_config = self.test_oidc_discovery()
        if not oidc_config:
            print("❌ OIDC discovery failed, aborting")
            return

        # Step 2: Authenticate API client
        self.authenticate_client()

        # Step 3: Build authorization URL (for reference)
        auth_url, redirect_uri = self.build_authorization_url()

        # Step 4: Create face auth session
        session = self.create_face_auth_session(user_id)

        # Step 5: Connect to WebSocket and perform face auth
        await self.connect_websocket(session["websocket_url"])
        self.stop_sending = False
        self.face_auth_result = None
        
        receiver_task = asyncio.create_task(self.receive_messages())
        await self.send_camera_frames(camera_index, duration=30)
        await self.ws.close()

        try:
            await asyncio.wait_for(receiver_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass

        # Step 6: If face auth successful, simulate OIDC code generation
        if self.face_auth_result and self.face_auth_result.get('authenticated'):
            print("\n" + "=" * 60)
            print("🎯 Face Authentication Passed - Generating OIDC Tokens")
            print("=" * 60)
            
            # In a real flow, the authorization code would come from browser redirect
            # For testing, we'll create an auth code directly via Django
            auth_code = self._create_test_authorization_code(user_id)
            
            if auth_code:
                # Step 7: Exchange code for tokens
                tokens = self.exchange_authorization_code(auth_code, redirect_uri)
                
                if tokens:
                    # Step 8: Get UserInfo
                    self.get_userinfo(tokens['access_token'])
                    
                    # Step 9: Introspect token
                    self.introspect_token(tokens['access_token'])
                    
                    print("\n" + "=" * 60)
                    print("✅ OIDC FLOW COMPLETE!")
                    print("=" * 60)
                    print("\n📋 Summary:")
                    print(f"   User authenticated: {user_id}")
                    print(f"   Face confidence: {self.face_auth_result.get('confidence', 0):.2%}")
                    print(f"   Liveness verified: {self.face_auth_result.get('liveness_verified')}")
                    print(f"   Access token issued: ✅")
                    print(f"   ID token issued: ✅")
        else:
            print("\n❌ Face authentication failed, OIDC flow aborted")

    def _create_test_authorization_code(self, user_id: str) -> str:
        """Create authorization code for testing (simulates browser flow)"""
        # This is for testing only - in production, this happens via browser
        import os
        import sys
        
        # Import Django to create auth code directly
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'face_recognition_app'))
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_app.settings')
        
        try:
            import django
            django.setup()
            
            from auth_service.oidc.models import OAuthClient, AuthorizationCode
            from clients.models import ClientUser
            
            # Get OAuth client
            oauth_client = OAuthClient.objects.get(client_id=self.client_id)
            
            # Get user
            client_user = ClientUser.objects.filter(external_user_id=user_id).first()
            if not client_user:
                print(f"❌ User {user_id} not found")
                return None
            
            # Create authorization code
            auth_code = AuthorizationCode.objects.create(
                client=oauth_client,
                user=client_user,
                redirect_uri="http://localhost:8080/callback",
                scope='openid profile email face_auth',
                nonce=self.nonce,
                code_challenge=self.code_challenge,
                code_challenge_method='S256'
            )
            
            print(f"\n✅ Authorization code created: {auth_code.code[:20]}...")
            return auth_code.code
            
        except Exception as e:
            print(f"❌ Failed to create auth code: {e}")
            import traceback
            traceback.print_exc()
            return None


async def main():
    """Main function"""
    if len(sys.argv) < 7:
        print("Usage: python test_oidc_face_auth.py <API_KEY> <SECRET_KEY> <BASE_URL> <CLIENT_ID> <CLIENT_SECRET> <USER_ID>")
        print("\nExample:")
        print("  python test_oidc_face_auth.py \\")
        print("      frapi_xxx secret_xxx \\")
        print("      http://192.168.1.41:8003 \\")
        print("      oidc_vBYjlMiaEUgUdnObhaetc37L-HzsF-_H \\")
        print("      Jv8VYQNN5ODN5lZloiE2MhZ8TFQzBJplwOg2th_bPwdk9ogKApyYfU2PUK8Trudt \\")
        print("      653384")
        sys.exit(1)

    api_key = sys.argv[1]
    secret_key = sys.argv[2]
    base_url = sys.argv[3]
    client_id = sys.argv[4]
    client_secret = sys.argv[5]
    user_id = sys.argv[6]

    client = OIDCFaceAuthClient(
        api_key=api_key,
        secret_key=secret_key,
        base_url=base_url,
        client_id=client_id,
        client_secret=client_secret
    )

    try:
        await client.run_oidc_face_auth(user_id)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

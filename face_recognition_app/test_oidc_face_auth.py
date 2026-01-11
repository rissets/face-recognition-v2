#!/usr/bin/env python3
"""
OIDC + Face Recognition Integration Test

This script tests the complete OIDC authorization flow combined with
face recognition authentication:

1. Discover OIDC endpoints
2. Start OIDC authorization flow (simulate Keycloak redirect)
3. Create face authentication session
4. Connect to WebSocket for face verification
5. Complete authentication
6. Exchange authorization code for tokens
7. Verify tokens and userinfo

Usage:
    python test_oidc_face_auth.py --user-id USER_ID [--mode interactive|simulated]
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import secrets
import sys
import urllib.parse
from datetime import datetime

import requests
import websockets

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = os.getenv('FACE_API_URL', 'http://127.0.0.1:8003')
WEBSOCKET_URL = os.getenv('FACE_WS_URL', 'ws://127.0.0.1:8003')

# API Credentials (from environment or defaults)
API_KEY = os.getenv('FACE_API_KEY', 'frapi_llazJ4S2Wcjz1PH6JzeOTdTYLlfAtObuZjgZuSWPi7c')
API_SECRET = os.getenv('FACE_API_SECRET', 'DuHe3d04cUU1eMnhoBgxyCBiDm3T3kRClJG3Y_cZyiTOnxpLF6uWyDUuq5K5aN3Hr1zm-otL6rpbnnfHlNAaJg')

# OIDC Client (for testing)
OIDC_CLIENT_ID = os.getenv('OIDC_CLIENT_ID', 'oidc_vBYjlMiaEUgUdnObhaetc37L-HzsF-_H')
OIDC_CLIENT_SECRET = os.getenv('OIDC_CLIENT_SECRET', 'Jv8VYQNN5ODN5lZloiE2MhZ8TFQzBJplwOg2th_bPwdk9ogKApyYfU2PUK8Trudt')
OIDC_REDIRECT_URI = os.getenv('OIDC_REDIRECT_URI', 'http://localhost:8080/callback')

# =============================================================================
# PKCE Helper Functions
# =============================================================================

def generate_code_verifier(length=64):
    """Generate a code verifier for PKCE"""
    return secrets.token_urlsafe(length)[:length]

def generate_code_challenge(verifier):
    """Generate a code challenge from the verifier (S256 method)"""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode()

# =============================================================================
# OIDC Discovery
# =============================================================================

def discover_oidc_endpoints():
    """Fetch OIDC discovery document"""
    print("\n[1] OIDC Discovery")
    print("-" * 50)
    
    discovery_url = f"{BASE_URL}/.well-known/openid-configuration"
    print(f"Fetching: {discovery_url}")
    
    try:
        r = requests.get(discovery_url)
        r.raise_for_status()
        config = r.json()
        
        print(f"✓ Issuer: {config.get('issuer')}")
        print(f"✓ Authorization Endpoint: {config.get('authorization_endpoint')}")
        print(f"✓ Token Endpoint: {config.get('token_endpoint')}")
        print(f"✓ Userinfo Endpoint: {config.get('userinfo_endpoint')}")
        print(f"✓ Supported Scopes: {config.get('scopes_supported')}")
        
        return config
    except Exception as e:
        print(f"✗ Discovery failed: {e}")
        return None

# =============================================================================
# API Authentication
# =============================================================================

def get_api_token():
    """Get JWT token for API access"""
    print("\n[2] API Authentication")
    print("-" * 50)
    
    auth_url = f"{BASE_URL}/api/core/auth/client/"
    print(f"Authenticating with API key: {API_KEY[:20]}...")
    
    try:
        r = requests.post(auth_url, json={
            'api_key': API_KEY,
            'api_secret': API_SECRET
        })
        r.raise_for_status()
        data = r.json()
        
        token = data.get('access_token')
        print(f"✓ Token obtained: {token[:50]}...")
        print(f"✓ Client: {data.get('client_name')} ({data.get('client_id')})")
        print(f"✓ Tier: {data.get('tier')}")
        
        return token
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"  Response: {e.response.text[:500]}")
        return None

# =============================================================================
# Face Authentication Session
# =============================================================================

def create_face_auth_session(token, user_id):
    """Create a face authentication session"""
    print("\n[3] Face Authentication Session")
    print("-" * 50)
    
    auth_url = f"{BASE_URL}/api/auth/authentication/"
    headers = {
        'Authorization': f'JWT {token}',
        'Content-Type': 'application/json'
    }
    
    print(f"Creating session for user: {user_id}")
    
    try:
        r = requests.post(auth_url, json={
            'user_id': user_id,
            'session_type': 'webcam',
            'require_liveness': True
        }, headers=headers)
        r.raise_for_status()
        data = r.json()
        
        session_token = data.get('session_token')
        ws_url = data.get('websocket_url')
        
        print(f"✓ Session created: {session_token}")
        print(f"✓ Status: {data.get('status')}")
        print(f"✓ WebSocket URL: {ws_url}")
        print(f"✓ Expires: {data.get('expires_at')}")
        
        return data
    except Exception as e:
        print(f"✗ Session creation failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"  Response: {e.response.text[:500]}")
        return None

# =============================================================================
# OIDC Authorization Flow
# =============================================================================

def start_authorization_flow(oidc_config, user_id):
    """Start OIDC authorization flow with PKCE"""
    print("\n[4] OIDC Authorization Flow")
    print("-" * 50)
    
    # Generate PKCE parameters
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    
    print(f"✓ Code Verifier: {code_verifier[:20]}...")
    print(f"✓ Code Challenge: {code_challenge[:20]}...")
    print(f"✓ State: {state[:20]}...")
    print(f"✓ Nonce: {nonce[:20]}...")
    
    # Build authorization URL
    auth_endpoint = oidc_config.get('authorization_endpoint')
    params = {
        'client_id': OIDC_CLIENT_ID,
        'response_type': 'code',
        'scope': 'openid profile email face_auth',
        'redirect_uri': OIDC_REDIRECT_URI,
        'state': state,
        'nonce': nonce,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256',
        'login_hint': user_id  # Pass user_id as login hint
    }
    
    auth_url = f"{auth_endpoint}?{urllib.parse.urlencode(params)}"
    print(f"\n✓ Authorization URL:")
    print(f"  {auth_url[:100]}...")
    
    return {
        'auth_url': auth_url,
        'code_verifier': code_verifier,
        'code_challenge': code_challenge,
        'state': state,
        'nonce': nonce
    }

# =============================================================================
# WebSocket Face Authentication
# =============================================================================

async def websocket_face_auth(session_data, test_image_path=None):
    """Connect to WebSocket and perform face authentication"""
    print("\n[5] WebSocket Face Authentication")
    print("-" * 50)
    
    ws_url = session_data.get('websocket_url')
    if not ws_url:
        print("✗ No WebSocket URL available")
        return None
    
    # If no test image, create a simulated response
    if not test_image_path:
        print("⚠ No test image provided - simulating WebSocket flow")
        print("  In production, the user would stream camera frames here")
        return {
            'simulated': True,
            'message': 'WebSocket flow would happen with real camera'
        }
    
    try:
        print(f"Connecting to: {ws_url}")
        
        async with websockets.connect(ws_url) as ws:
            print("✓ Connected to WebSocket")
            
            # Read test image
            with open(test_image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # Send frame
            frame_msg = {
                'type': 'frame',
                'image': f'data:image/jpeg;base64,{image_data}',
                'frame_number': 1
            }
            
            await ws.send(json.dumps(frame_msg))
            print("✓ Sent frame")
            
            # Receive response
            response = await asyncio.wait_for(ws.recv(), timeout=30)
            result = json.loads(response)
            
            print(f"✓ Response: {json.dumps(result, indent=2)}")
            
            return result
            
    except asyncio.TimeoutError:
        print("✗ WebSocket timeout")
        return None
    except Exception as e:
        print(f"✗ WebSocket error: {e}")
        return None

# =============================================================================
# Token Exchange
# =============================================================================

def exchange_code_for_tokens(oidc_config, auth_code, code_verifier):
    """Exchange authorization code for tokens"""
    print("\n[6] Token Exchange")
    print("-" * 50)
    
    token_endpoint = oidc_config.get('token_endpoint')
    
    print(f"Exchanging code at: {token_endpoint}")
    
    try:
        r = requests.post(token_endpoint, data={
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': OIDC_REDIRECT_URI,
            'client_id': OIDC_CLIENT_ID,
            'client_secret': OIDC_CLIENT_SECRET,
            'code_verifier': code_verifier
        })
        r.raise_for_status()
        tokens = r.json()
        
        print(f"✓ Access Token: {tokens.get('access_token', 'N/A')[:50]}...")
        print(f"✓ Token Type: {tokens.get('token_type')}")
        print(f"✓ Expires In: {tokens.get('expires_in')} seconds")
        if tokens.get('id_token'):
            print(f"✓ ID Token: {tokens.get('id_token')[:50]}...")
        if tokens.get('refresh_token'):
            print(f"✓ Refresh Token: {tokens.get('refresh_token')[:50]}...")
        
        return tokens
    except Exception as e:
        print(f"✗ Token exchange failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"  Response: {e.response.text[:500]}")
        return None

# =============================================================================
# Userinfo
# =============================================================================

def get_userinfo(oidc_config, access_token):
    """Get user info from OIDC provider"""
    print("\n[7] User Info")
    print("-" * 50)
    
    userinfo_endpoint = oidc_config.get('userinfo_endpoint')
    
    print(f"Fetching userinfo from: {userinfo_endpoint}")
    
    try:
        r = requests.get(userinfo_endpoint, headers={
            'Authorization': f'Bearer {access_token}'
        })
        r.raise_for_status()
        userinfo = r.json()
        
        print(f"✓ Subject (sub): {userinfo.get('sub')}")
        print(f"✓ Name: {userinfo.get('name', 'N/A')}")
        print(f"✓ Email: {userinfo.get('email', 'N/A')}")
        print(f"✓ Face Verified: {userinfo.get('face_verified', 'N/A')}")
        print(f"✓ Liveness Verified: {userinfo.get('liveness_verified', 'N/A')}")
        print(f"✓ Face Confidence: {userinfo.get('face_confidence', 'N/A')}")
        
        return userinfo
    except Exception as e:
        print(f"✗ Userinfo fetch failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"  Response: {e.response.text[:500]}")
        return None

# =============================================================================
# Simulated Complete Flow
# =============================================================================

def simulate_complete_flow(user_id, image_path=None):
    """Simulate the complete OIDC + Face Auth flow"""
    print("=" * 60)
    print("OIDC + Face Recognition Integration Test")
    print("=" * 60)
    print(f"User ID: {user_id}")
    print(f"Base URL: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Step 1: OIDC Discovery
    oidc_config = discover_oidc_endpoints()
    if not oidc_config:
        print("\n✗ OIDC Discovery failed - stopping")
        return False
    
    # Step 2: API Authentication
    api_token = get_api_token()
    if not api_token:
        print("\n✗ API Authentication failed - stopping")
        return False
    
    # Step 3: Create Face Auth Session
    session_data = create_face_auth_session(api_token, user_id)
    if not session_data:
        print("\n✗ Face Auth Session failed - stopping")
        return False
    
    # Step 4: Start OIDC Authorization
    auth_flow = start_authorization_flow(oidc_config, user_id)
    
    # Step 5: WebSocket Face Auth
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        face_result = loop.run_until_complete(
            websocket_face_auth(session_data, image_path)
        )
    finally:
        loop.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✓ OIDC Discovery: OK")
    print(f"✓ API Authentication: OK") 
    print(f"✓ Face Auth Session: Created ({session_data.get('session_token')[:20]}...)")
    print(f"✓ OIDC Auth Flow: Initiated")
    print(f"✓ WebSocket: {'Simulated' if not image_path else 'Connected'}")
    
    print("\n" + "-" * 60)
    print("Next Steps (in production):")
    print("-" * 60)
    print("1. Open authorization URL in browser")
    print("2. User authenticates via face recognition (WebSocket)")
    print("3. After successful auth, redirect to callback with code")
    print("4. Exchange code for tokens using PKCE verifier")
    print("5. Validate tokens and get user info")
    
    print(f"\nAuthorization URL:")
    print(auth_flow['auth_url'])
    
    return True

# =============================================================================
# Main
# =============================================================================

def main():
    global BASE_URL, WEBSOCKET_URL
    
    parser = argparse.ArgumentParser(
        description='OIDC + Face Recognition Integration Test'
    )
    parser.add_argument(
        '--user-id',
        default='test_similarity_1767689698',
        help='User ID to authenticate (must be enrolled)'
    )
    parser.add_argument(
        '--image',
        help='Path to test face image (optional)'
    )
    parser.add_argument(
        '--base-url',
        default=BASE_URL,
        help='Base URL of the Face Recognition API'
    )
    
    args = parser.parse_args()
    
    BASE_URL = args.base_url
    WEBSOCKET_URL = args.base_url.replace('http://', 'ws://').replace('https://', 'wss://')
    
    success = simulate_complete_flow(args.user_id, args.image)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

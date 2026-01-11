#!/usr/bin/env python3
"""
Keycloak Simulator - Simulates how Keycloak acts as OIDC Client
This script demonstrates the complete OAuth 2.0 / OIDC flow from Keycloak's perspective.

Usage:
    python keycloak_simulator.py --client-id <client_id> --client-secret <secret>
"""

import os
import sys
import json
import base64
import hashlib
import secrets
import argparse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlencode, urlparse
import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class OIDCClient:
    """OIDC Client implementation (like Keycloak would use)"""
    
    def __init__(self, issuer_url, client_id, client_secret, redirect_uri):
        self.issuer_url = issuer_url.rstrip('/')
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.config = None
        self.jwks = None
        
    def discover(self):
        """Fetch OpenID Connect configuration"""
        discovery_url = f"{self.issuer_url}/.well-known/openid-configuration"
        print(f"\n📡 Fetching OIDC Discovery from: {discovery_url}")
        
        response = requests.get(discovery_url, verify=False)
        self.config = response.json()
        
        print(f"✅ Discovery successful!")
        print(f"   Authorization Endpoint: {self.config.get('authorization_endpoint')}")
        print(f"   Token Endpoint: {self.config.get('token_endpoint')}")
        print(f"   UserInfo Endpoint: {self.config.get('userinfo_endpoint')}")
        
        return self.config
    
    def fetch_jwks(self):
        """Fetch JSON Web Key Set for token verification"""
        if not self.config:
            self.discover()
            
        jwks_uri = self.config.get('jwks_uri')
        print(f"\n🔑 Fetching JWKS from: {jwks_uri}")
        
        response = requests.get(jwks_uri, verify=False)
        self.jwks = response.json()
        
        print(f"✅ JWKS fetched! Keys: {len(self.jwks.get('keys', []))}")
        return self.jwks
    
    def generate_pkce(self):
        """Generate PKCE code verifier and challenge"""
        code_verifier = secrets.token_urlsafe(64)[:128]
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b'=').decode()
        
        return code_verifier, code_challenge
    
    def build_authorization_url(self, scopes, state=None, nonce=None, use_pkce=True):
        """Build authorization URL for redirect"""
        if not self.config:
            self.discover()
            
        state = state or secrets.token_urlsafe(16)
        nonce = nonce or secrets.token_urlsafe(16)
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': scopes,
            'state': state,
            'nonce': nonce
        }
        
        code_verifier = None
        if use_pkce:
            code_verifier, code_challenge = self.generate_pkce()
            params['code_challenge'] = code_challenge
            params['code_challenge_method'] = 'S256'
        
        auth_url = f"{self.config['authorization_endpoint']}?{urlencode(params)}"
        
        return auth_url, state, nonce, code_verifier
    
    def exchange_code(self, code, code_verifier=None):
        """Exchange authorization code for tokens"""
        if not self.config:
            self.discover()
            
        print(f"\n🔄 Exchanging authorization code for tokens...")
        
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        if code_verifier:
            data['code_verifier'] = code_verifier
        
        response = requests.post(
            self.config['token_endpoint'],
            data=data,
            verify=False
        )
        
        tokens = response.json()
        
        if 'error' in tokens:
            print(f"❌ Token exchange failed: {tokens}")
            return None
            
        print(f"✅ Token exchange successful!")
        print(f"   Access Token: {tokens.get('access_token', 'N/A')[:50]}...")
        print(f"   Token Type: {tokens.get('token_type')}")
        print(f"   Expires In: {tokens.get('expires_in')} seconds")
        
        if tokens.get('refresh_token'):
            print(f"   Refresh Token: {tokens['refresh_token'][:30]}...")
        
        return tokens
    
    def get_userinfo(self, access_token):
        """Fetch user information using access token"""
        if not self.config:
            self.discover()
            
        print(f"\n👤 Fetching UserInfo...")
        
        response = requests.get(
            self.config['userinfo_endpoint'],
            headers={'Authorization': f'Bearer {access_token}'},
            verify=False
        )
        
        userinfo = response.json()
        
        if 'error' in userinfo:
            print(f"❌ UserInfo failed: {userinfo}")
            return None
            
        print(f"✅ UserInfo fetched!")
        for key, value in userinfo.items():
            print(f"   {key}: {value}")
        
        return userinfo
    
    def refresh_tokens(self, refresh_token):
        """Refresh access token using refresh token"""
        if not self.config:
            self.discover()
            
        print(f"\n🔄 Refreshing tokens...")
        
        response = requests.post(
            self.config['token_endpoint'],
            data={
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            },
            verify=False
        )
        
        tokens = response.json()
        
        if 'error' in tokens:
            print(f"❌ Token refresh failed: {tokens}")
            return None
            
        print(f"✅ Tokens refreshed!")
        return tokens
    
    def introspect_token(self, token):
        """Introspect token to verify its validity"""
        if not self.config:
            self.discover()
            
        introspect_url = self.config.get('introspection_endpoint')
        if not introspect_url:
            print("❌ Introspection endpoint not available")
            return None
            
        print(f"\n🔍 Introspecting token...")
        
        response = requests.post(
            introspect_url,
            data={
                'token': token,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            },
            verify=False
        )
        
        result = response.json()
        print(f"✅ Introspection result:")
        print(f"   Active: {result.get('active')}")
        
        return result
    
    def revoke_token(self, token, token_type='access_token'):
        """Revoke a token"""
        if not self.config:
            self.discover()
            
        revoke_url = self.config.get('revocation_endpoint')
        if not revoke_url:
            print("❌ Revocation endpoint not available")
            return False
            
        print(f"\n🗑️ Revoking {token_type}...")
        
        response = requests.post(
            revoke_url,
            data={
                'token': token,
                'token_type_hint': token_type,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            },
            verify=False
        )
        
        if response.status_code == 200:
            print(f"✅ Token revoked successfully!")
            return True
        else:
            print(f"❌ Revocation failed: {response.text}")
            return False


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP Handler to receive OAuth callback"""
    
    client = None
    code_verifier = None
    state = None
    tokens = None
    
    def log_message(self, format, *args):
        pass  # Suppress default logging
    
    def do_GET(self):
        """Handle callback from authorization server"""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        if parsed.path == '/callback':
            code = params.get('code', [None])[0]
            returned_state = params.get('state', [None])[0]
            error = params.get('error', [None])[0]
            
            if error:
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                error_desc = params.get('error_description', ['Unknown error'])[0]
                self.wfile.write(f"""
                    <html>
                    <head><title>Authentication Failed</title></head>
                    <body style="font-family: Arial; padding: 40px; text-align: center;">
                        <h1 style="color: red;">❌ Authentication Failed</h1>
                        <p><strong>Error:</strong> {error}</p>
                        <p><strong>Description:</strong> {error_desc}</p>
                    </body>
                    </html>
                """.encode())
                return
            
            if returned_state != CallbackHandler.state:
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write("""
                    <html>
                    <head><title>Invalid State</title></head>
                    <body style="font-family: Arial; padding: 40px; text-align: center;">
                        <h1 style="color: red;">&#10060; Invalid State</h1>
                        <p>State mismatch - possible CSRF attack!</p>
                    </body>
                    </html>
                """.encode('utf-8'))
                return
            
            # Exchange code for tokens
            print(f"\n📨 Received authorization code: {code[:20]}...")
            CallbackHandler.tokens = CallbackHandler.client.exchange_code(
                code, 
                CallbackHandler.code_verifier
            )
            
            if CallbackHandler.tokens:
                # Get UserInfo
                userinfo = CallbackHandler.client.get_userinfo(
                    CallbackHandler.tokens['access_token']
                )
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                # Decode ID token
                id_token_claims = "N/A"
                if CallbackHandler.tokens.get('id_token'):
                    try:
                        parts = CallbackHandler.tokens['id_token'].split('.')
                        payload = base64.urlsafe_b64decode(parts[1] + '==')
                        id_token_claims = json.dumps(json.loads(payload), indent=2)
                    except:
                        pass
                
                html = f"""
                    <html>
                    <head>
                        <title>Authentication Successful</title>
                        <style>
                            body {{ font-family: Arial; padding: 40px; max-width: 800px; margin: 0 auto; }}
                            h1 {{ color: #22c55e; }}
                            .section {{ background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                            pre {{ background: #1f2937; color: #34d399; padding: 15px; border-radius: 8px; overflow-x: auto; }}
                            .token {{ word-break: break-all; font-size: 12px; }}
                        </style>
                    </head>
                    <body>
                        <h1>✅ Authentication Successful!</h1>
                        <p>Face Recognition OIDC authentication completed successfully.</p>
                        
                        <div class="section">
                            <h3>User Information</h3>
                            <pre>{json.dumps(userinfo, indent=2) if userinfo else 'N/A'}</pre>
                        </div>
                        
                        <div class="section">
                            <h3>ID Token Claims</h3>
                            <pre>{id_token_claims}</pre>
                        </div>
                        
                        <div class="section">
                            <h3>Access Token</h3>
                            <p class="token">{CallbackHandler.tokens.get('access_token', 'N/A')}</p>
                        </div>
                        
                        <p style="color: #6b7280; margin-top: 30px;">
                            You can close this window. Check the terminal for more details.
                        </p>
                    </body>
                    </html>
                """
                self.wfile.write(html.encode())
            else:
                self.send_response(500)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write("""
                    <html>
                    <head><title>Token Exchange Failed</title></head>
                    <body style="font-family: Arial; padding: 40px; text-align: center;">
                        <h1 style="color: red;">&#10060; Token Exchange Failed</h1>
                        <p>Failed to exchange authorization code for tokens.</p>
                    </body>
                    </html>
                """.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def run_interactive_flow(client, port=8080):
    """Run interactive OAuth flow with browser"""
    
    # Build authorization URL
    scopes = 'openid profile email face_auth'
    redirect_uri = f'http://localhost:{port}/callback'
    client.redirect_uri = redirect_uri
    
    auth_url, state, nonce, code_verifier = client.build_authorization_url(scopes)
    
    # Store for callback handler
    CallbackHandler.client = client
    CallbackHandler.code_verifier = code_verifier
    CallbackHandler.state = state
    
    print(f"\n🌐 Starting local server on port {port}...")
    print(f"\n📱 Opening browser for Face Recognition login...")
    print(f"\n🔗 Authorization URL: {auth_url[:80]}...")
    
    # Start callback server
    server = HTTPServer(('localhost', port), CallbackHandler)
    
    # Open browser
    webbrowser.open(auth_url)
    
    print(f"\n⏳ Waiting for authentication callback...")
    print(f"   (Complete face recognition in the browser)")
    
    # Handle single request (the callback)
    server.handle_request()
    server.server_close()
    
    return CallbackHandler.tokens


def run_manual_flow(client):
    """Run manual OAuth flow (for headless environments)"""
    
    # Build authorization URL
    scopes = 'openid profile email face_auth'
    auth_url, state, nonce, code_verifier = client.build_authorization_url(scopes)
    
    print(f"\n🔗 Authorization URL:")
    print(f"   {auth_url}")
    
    print(f"\n📝 Please open the URL above in a browser, complete face authentication,")
    print(f"   and then paste the authorization code below.")
    
    code = input("\n🔑 Authorization Code: ").strip()
    
    if not code:
        print("❌ No code provided")
        return None
    
    tokens = client.exchange_code(code, code_verifier)
    
    if tokens:
        client.get_userinfo(tokens['access_token'])
    
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description='Keycloak Simulator - Test Face Recognition OIDC Provider'
    )
    parser.add_argument(
        '--issuer', '-i',
        default='http://192.168.1.41:8003',
        help='OIDC Issuer URL (default: http://192.168.1.41:8003)'
    )
    parser.add_argument(
        '--client-id', '-c',
        required=True,
        help='OAuth Client ID'
    )
    parser.add_argument(
        '--client-secret', '-s',
        required=True,
        help='OAuth Client Secret'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8080,
        help='Local callback server port (default: 8080)'
    )
    parser.add_argument(
        '--manual', '-m',
        action='store_true',
        help='Use manual mode (no browser)'
    )
    parser.add_argument(
        '--test-only', '-t',
        action='store_true',
        help='Only test discovery and JWKS (no login)'
    )
    
    args = parser.parse_args()
    
    # Suppress SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    print("=" * 60)
    print("🔐 Keycloak Simulator - Face Recognition OIDC Test")
    print("=" * 60)
    print(f"\n📍 OIDC Issuer: {args.issuer}")
    print(f"🆔 Client ID: {args.client_id}")
    
    # Create client
    client = OIDCClient(
        issuer_url=args.issuer,
        client_id=args.client_id,
        client_secret=args.client_secret,
        redirect_uri=f'http://localhost:{args.port}/callback'
    )
    
    # Discover and fetch JWKS
    client.discover()
    client.fetch_jwks()
    
    if args.test_only:
        print("\n✅ Discovery and JWKS test completed!")
        return
    
    # Run OAuth flow
    if args.manual:
        tokens = run_manual_flow(client)
    else:
        tokens = run_interactive_flow(client, args.port)
    
    if tokens:
        print("\n" + "=" * 60)
        print("🎉 OIDC Flow Completed Successfully!")
        print("=" * 60)
        
        # Additional tests
        print("\n📊 Running additional tests...")
        
        # Test introspection
        client.introspect_token(tokens['access_token'])
        
        # Test refresh if available
        if tokens.get('refresh_token'):
            new_tokens = client.refresh_tokens(tokens['refresh_token'])
            if new_tokens:
                tokens = new_tokens
        
        # Prompt for revocation
        revoke = input("\n🗑️ Revoke tokens? (y/n): ").strip().lower()
        if revoke == 'y':
            client.revoke_token(tokens['access_token'])
    else:
        print("\n❌ Authentication failed!")


if __name__ == '__main__':
    main()

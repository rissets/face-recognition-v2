#!/usr/bin/env python3
"""
OIDC Integration Test Script

This script tests the complete OAuth 2.0 / OpenID Connect flow
for Face Recognition Provider integration with Keycloak.

Usage:
    python test_oidc_flow.py [BASE_URL] [CLIENT_ID] [CLIENT_SECRET]

Example:
    python test_oidc_flow.py https://face.ahu.go.id oidc_xxx secret_xxx
"""

import sys
import json
import base64
import hashlib
import secrets
import requests
from urllib.parse import urlencode, urlparse, parse_qs


class OIDCTester:
    def __init__(self, base_url: str, client_id: str, client_secret: str):
        self.base_url = base_url.rstrip('/')
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = "http://localhost:8000/oauth/callback"
        
        # PKCE values
        self.code_verifier = self._generate_code_verifier()
        self.code_challenge = self._generate_code_challenge(self.code_verifier)
        
        # State and nonce
        self.state = secrets.token_urlsafe(16)
        self.nonce = secrets.token_urlsafe(16)
        
        # Tokens
        self.access_token = None
        self.refresh_token = None
        self.id_token = None

    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier"""
        return secrets.token_urlsafe(64)[:128]

    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge from verifier"""
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip('=')

    def test_discovery(self) -> bool:
        """Test 1: OpenID Connect Discovery"""
        print("\n" + "=" * 60)
        print("Test 1: OpenID Connect Discovery")
        print("=" * 60)
        
        url = f"{self.base_url}/.well-known/openid-configuration"
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                config = response.json()
                print(f"✅ Discovery successful")
                print(f"   Issuer: {config.get('issuer')}")
                print(f"   Authorization: {config.get('authorization_endpoint')}")
                print(f"   Token: {config.get('token_endpoint')}")
                print(f"   UserInfo: {config.get('userinfo_endpoint')}")
                print(f"   JWKS: {config.get('jwks_uri')}")
                print(f"   Scopes: {config.get('scopes_supported')}")
                return True
            else:
                print(f"❌ Discovery failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return False

    def test_jwks(self) -> bool:
        """Test 2: JWKS Endpoint"""
        print("\n" + "=" * 60)
        print("Test 2: JWKS Endpoint")
        print("=" * 60)
        
        url = f"{self.base_url}/.well-known/jwks.json"
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                jwks = response.json()
                keys = jwks.get('keys', [])
                print(f"✅ JWKS successful")
                print(f"   Number of keys: {len(keys)}")
                for key in keys:
                    print(f"   - Key ID: {key.get('kid')}")
                    print(f"     Algorithm: {key.get('alg')}")
                    print(f"     Key Type: {key.get('kty')}")
                    print(f"     Use: {key.get('use')}")
                return True
            else:
                print(f"❌ JWKS failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ JWKS error: {e}")
            return False

    def get_authorization_url(self) -> str:
        """Generate authorization URL"""
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'openid profile email face_auth',
            'state': self.state,
            'nonce': self.nonce,
            'code_challenge': self.code_challenge,
            'code_challenge_method': 'S256',
        }
        return f"{self.base_url}/oauth/authorize?{urlencode(params)}"

    def test_authorization_url(self) -> bool:
        """Test 3: Authorization Endpoint (URL generation)"""
        print("\n" + "=" * 60)
        print("Test 3: Authorization Endpoint")
        print("=" * 60)
        
        auth_url = self.get_authorization_url()
        print(f"Authorization URL generated:")
        print(f"   {auth_url}")
        print(f"\n   State: {self.state}")
        print(f"   Nonce: {self.nonce}")
        print(f"   Code Verifier: {self.code_verifier[:32]}...")
        print(f"   Code Challenge: {self.code_challenge}")
        print(f"\n✅ Authorization URL ready")
        print(f"   Open this URL in browser to authenticate with face recognition")
        return True

    def test_token_exchange(self, authorization_code: str) -> bool:
        """Test 4: Token Endpoint"""
        print("\n" + "=" * 60)
        print("Test 4: Token Exchange")
        print("=" * 60)
        
        url = f"{self.base_url}/oauth/token"
        print(f"URL: {url}")
        
        data = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code_verifier': self.code_verifier,
        }
        
        try:
            response = requests.post(url, data=data, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                tokens = response.json()
                self.access_token = tokens.get('access_token')
                self.refresh_token = tokens.get('refresh_token')
                self.id_token = tokens.get('id_token')
                
                print(f"✅ Token exchange successful")
                print(f"   Access Token: {self.access_token[:50]}...")
                print(f"   Token Type: {tokens.get('token_type')}")
                print(f"   Expires In: {tokens.get('expires_in')} seconds")
                if self.refresh_token:
                    print(f"   Refresh Token: {self.refresh_token[:30]}...")
                if self.id_token:
                    print(f"   ID Token: {self.id_token[:50]}...")
                    self._decode_id_token()
                return True
            else:
                print(f"❌ Token exchange failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Token exchange error: {e}")
            return False

    def _decode_id_token(self):
        """Decode and display ID token claims"""
        if not self.id_token:
            return
        
        try:
            # Split token and decode payload (without verification for display)
            parts = self.id_token.split('.')
            if len(parts) == 3:
                # Add padding
                payload = parts[1]
                padding = 4 - len(payload) % 4
                if padding != 4:
                    payload += '=' * padding
                
                claims = json.loads(base64.urlsafe_b64decode(payload))
                print(f"\n   ID Token Claims:")
                for key, value in claims.items():
                    print(f"     {key}: {value}")
        except Exception as e:
            print(f"   (Could not decode ID token: {e})")

    def test_userinfo(self) -> bool:
        """Test 5: UserInfo Endpoint"""
        print("\n" + "=" * 60)
        print("Test 5: UserInfo Endpoint")
        print("=" * 60)
        
        if not self.access_token:
            print("❌ No access token available. Run token exchange first.")
            return False
        
        url = f"{self.base_url}/oauth/userinfo"
        print(f"URL: {url}")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                userinfo = response.json()
                print(f"✅ UserInfo successful")
                for key, value in userinfo.items():
                    print(f"   {key}: {value}")
                return True
            else:
                print(f"❌ UserInfo failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ UserInfo error: {e}")
            return False

    def test_token_refresh(self) -> bool:
        """Test 6: Token Refresh"""
        print("\n" + "=" * 60)
        print("Test 6: Token Refresh")
        print("=" * 60)
        
        if not self.refresh_token:
            print("❌ No refresh token available. Run token exchange first.")
            return False
        
        url = f"{self.base_url}/oauth/token"
        print(f"URL: {url}")
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }
        
        try:
            response = requests.post(url, data=data, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                tokens = response.json()
                self.access_token = tokens.get('access_token')
                self.refresh_token = tokens.get('refresh_token')
                
                print(f"✅ Token refresh successful")
                print(f"   New Access Token: {self.access_token[:50]}...")
                print(f"   Expires In: {tokens.get('expires_in')} seconds")
                return True
            else:
                print(f"❌ Token refresh failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Token refresh error: {e}")
            return False

    def test_introspection(self) -> bool:
        """Test 7: Token Introspection"""
        print("\n" + "=" * 60)
        print("Test 7: Token Introspection")
        print("=" * 60)
        
        if not self.access_token:
            print("❌ No access token available. Run token exchange first.")
            return False
        
        url = f"{self.base_url}/oauth/introspect"
        print(f"URL: {url}")
        
        data = {
            'token': self.access_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }
        
        try:
            response = requests.post(url, data=data, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                info = response.json()
                print(f"✅ Introspection successful")
                print(f"   Active: {info.get('active')}")
                for key, value in info.items():
                    if key != 'active':
                        print(f"   {key}: {value}")
                return True
            else:
                print(f"❌ Introspection failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Introspection error: {e}")
            return False

    def test_revocation(self) -> bool:
        """Test 8: Token Revocation"""
        print("\n" + "=" * 60)
        print("Test 8: Token Revocation")
        print("=" * 60)
        
        if not self.access_token:
            print("❌ No access token available. Run token exchange first.")
            return False
        
        url = f"{self.base_url}/oauth/revoke"
        print(f"URL: {url}")
        
        data = {
            'token': self.access_token,
            'token_type_hint': 'access_token',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }
        
        try:
            response = requests.post(url, data=data, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"✅ Token revocation successful")
                return True
            else:
                print(f"❌ Token revocation failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Token revocation error: {e}")
            return False

    def run_discovery_tests(self):
        """Run discovery and JWKS tests only"""
        print("\n" + "=" * 60)
        print("OIDC Discovery Tests")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")
        print(f"Client ID: {self.client_id}")
        
        results = []
        results.append(("Discovery", self.test_discovery()))
        results.append(("JWKS", self.test_jwks()))
        results.append(("Authorization URL", self.test_authorization_url()))
        
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {name}: {status}")
        
        return all(passed for _, passed in results)

    def run_full_flow(self, authorization_code: str):
        """Run full OAuth flow with authorization code"""
        print("\n" + "=" * 60)
        print("Full OIDC Flow Test")
        print("=" * 60)
        
        results = []
        results.append(("Token Exchange", self.test_token_exchange(authorization_code)))
        results.append(("UserInfo", self.test_userinfo()))
        results.append(("Token Refresh", self.test_token_refresh()))
        results.append(("Introspection", self.test_introspection()))
        results.append(("Revocation", self.test_revocation()))
        
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {name}: {status}")
        
        return all(passed for _, passed in results)


def main():
    if len(sys.argv) < 4:
        print("Usage: python test_oidc_flow.py <BASE_URL> <CLIENT_ID> <CLIENT_SECRET> [AUTH_CODE]")
        print("")
        print("Examples:")
        print("  # Run discovery tests only:")
        print("  python test_oidc_flow.py https://face.ahu.go.id oidc_xxx secret_xxx")
        print("")
        print("  # Run full flow with authorization code:")
        print("  python test_oidc_flow.py https://face.ahu.go.id oidc_xxx secret_xxx AUTH_CODE")
        sys.exit(1)
    
    base_url = sys.argv[1]
    client_id = sys.argv[2]
    client_secret = sys.argv[3]
    auth_code = sys.argv[4] if len(sys.argv) > 4 else None
    
    tester = OIDCTester(base_url, client_id, client_secret)
    
    if auth_code:
        # Run full flow
        success = tester.run_full_flow(auth_code)
    else:
        # Run discovery tests and show authorization URL
        success = tester.run_discovery_tests()
        
        print("\n" + "=" * 60)
        print("Next Steps")
        print("=" * 60)
        print("1. Open the Authorization URL in browser")
        print("2. Complete face authentication")
        print("3. Copy the 'code' parameter from the redirect URL")
        print("4. Run this script again with the authorization code:")
        print(f"   python test_oidc_flow.py {base_url} {client_id} {client_secret} <AUTH_CODE>")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
End-to-End OIDC Test with Face Recognition
This script performs a complete end-to-end test of the OIDC integration.

Prerequisites:
1. Django server running
2. Face enrolled user
3. OAuth client created

Usage:
    python test_oidc_e2e.py --client-id <id> --client-secret <secret> --user-id <uuid>
"""

import os
import sys
import json
import time
import base64
import hashlib
import secrets
import argparse
import asyncio
from datetime import datetime
from urllib.parse import urlencode

# Django setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'face_recognition_app'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_app.settings')

import django
django.setup()

from django.test import Client, TestCase, RequestFactory, override_settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from auth_service.oidc.models import OAuthClient, AuthorizationCode, OAuthToken
from auth_service.oidc.utils import (
    generate_access_token,
    generate_id_token,
    verify_code_challenge,
    get_oidc_config
)

User = get_user_model()


class OIDCEndToEndTest:
    """Complete OIDC End-to-End Test Suite"""
    
    def __init__(self, base_url=None):
        self.base_url = base_url or 'http://localhost:8000'
        self.client = Client()
        self.results = []
        self.oauth_client = None
        self.test_user = None
        
    def log(self, test_name, success, message, data=None):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
        print(f"{status} | {test_name}: {message}")
        if data and not success:
            print(f"       Data: {json.dumps(data, indent=2)[:200]}")
        return success
    
    def setup_test_data(self, client_id=None, client_secret=None, user_id=None):
        """Setup test OAuth client and user"""
        print("\n" + "=" * 60)
        print("🔧 Setting up test data...")
        print("=" * 60)
        
        # Get or create test user
        if user_id:
            try:
                from uuid import UUID
                self.test_user = User.objects.get(id=UUID(user_id))
                self.log("Get Test User", True, f"Found user: {self.test_user.email}")
            except User.DoesNotExist:
                self.log("Get Test User", False, f"User not found: {user_id}")
                return False
        else:
            # Create test user if not exists
            self.test_user, created = User.objects.get_or_create(
                email='oidc_test@example.com',
                defaults={
                    'username': 'oidc_test',
                    'first_name': 'OIDC',
                    'last_name': 'Test User',
                    'nik': '1234567890123456'
                }
            )
            if created:
                self.test_user.set_password('testpassword123')
                self.test_user.save()
            self.log("Create Test User", True, f"User: {self.test_user.email}")
        
        # Get or create OAuth client
        if client_id and client_secret:
            try:
                self.oauth_client = OAuthClient.objects.get(client_id=client_id)
                self.log("Get OAuth Client", True, f"Found client: {self.oauth_client.name}")
            except OAuthClient.DoesNotExist:
                self.log("Get OAuth Client", False, f"Client not found: {client_id}")
                return False
        else:
            # Create test client
            self.oauth_client, created = OAuthClient.objects.get_or_create(
                name='E2E Test Client',
                defaults={
                    'client_type': 'confidential',
                    'redirect_uris': 'http://localhost:8080/callback',
                    'allowed_scopes': 'openid profile email face_auth offline_access',
                    'token_endpoint_auth_method': 'client_secret_post'
                }
            )
            self.log("Create OAuth Client", True, f"Client: {self.oauth_client.client_id}")
            if created:
                print(f"       Client ID: {self.oauth_client.client_id}")
                print(f"       Client Secret: {self.oauth_client.client_secret}")
        
        return True
    
    def test_discovery_endpoint(self):
        """Test /.well-known/openid-configuration"""
        response = self.client.get('/.well-known/openid-configuration')
        
        if response.status_code != 200:
            return self.log("Discovery Endpoint", False, f"Status {response.status_code}")
        
        data = response.json()
        required_fields = [
            'issuer', 'authorization_endpoint', 'token_endpoint',
            'userinfo_endpoint', 'jwks_uri', 'response_types_supported'
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return self.log("Discovery Endpoint", False, f"Missing fields: {missing}")
        
        return self.log("Discovery Endpoint", True, "All required fields present", data)
    
    def test_jwks_endpoint(self):
        """Test /.well-known/jwks.json"""
        response = self.client.get('/.well-known/jwks.json')
        
        if response.status_code != 200:
            return self.log("JWKS Endpoint", False, f"Status {response.status_code}")
        
        data = response.json()
        if 'keys' not in data or len(data['keys']) == 0:
            return self.log("JWKS Endpoint", False, "No keys in JWKS")
        
        key = data['keys'][0]
        required = ['kty', 'kid', 'n', 'e']
        missing = [f for f in required if f not in key]
        
        if missing:
            return self.log("JWKS Endpoint", False, f"Missing key fields: {missing}")
        
        return self.log("JWKS Endpoint", True, f"Found {len(data['keys'])} key(s)")
    
    def test_authorization_endpoint(self):
        """Test /oauth/authorize"""
        # Generate PKCE
        code_verifier = secrets.token_urlsafe(64)[:128]
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b'=').decode()
        
        state = secrets.token_urlsafe(16)
        nonce = secrets.token_urlsafe(16)
        
        params = {
            'client_id': self.oauth_client.client_id,
            'redirect_uri': 'http://localhost:8080/callback',
            'response_type': 'code',
            'scope': 'openid profile email',
            'state': state,
            'nonce': nonce,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        }
        
        response = self.client.get(f'/oauth/authorize?{urlencode(params)}')
        
        # Should redirect to login or show face login page
        if response.status_code not in [200, 302]:
            return self.log("Authorization Endpoint", False, f"Status {response.status_code}")
        
        return self.log("Authorization Endpoint", True, 
                       "Redirects to face login" if response.status_code == 302 else "Shows face login page")
    
    def test_token_endpoint_invalid(self):
        """Test /oauth/token with invalid data"""
        response = self.client.post('/oauth/token', {
            'grant_type': 'authorization_code',
            'code': 'invalid_code',
            'client_id': self.oauth_client.client_id,
            'client_secret': self.oauth_client.client_secret
        }, content_type='application/x-www-form-urlencoded')
        
        data = response.json()
        if 'error' not in data:
            return self.log("Token Endpoint (Invalid)", False, "Should return error")
        
        return self.log("Token Endpoint (Invalid)", True, f"Returns error: {data['error']}")
    
    def test_simulated_auth_flow(self):
        """Test simulated authorization flow (without face recognition)"""
        # Generate PKCE
        code_verifier = secrets.token_urlsafe(64)[:128]
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b'=').decode()
        
        nonce = secrets.token_urlsafe(16)
        
        # Create authorization code directly (simulating successful face auth)
        auth_code = AuthorizationCode.objects.create(
            client=self.oauth_client,
            user=self.test_user,
            redirect_uri='http://localhost:8080/callback',
            scope='openid profile email face_auth',
            nonce=nonce,
            code_challenge=code_challenge,
            code_challenge_method='S256'
        )
        
        self.log("Create Auth Code", True, f"Code: {auth_code.code[:20]}...")
        
        # Exchange code for tokens
        response = self.client.post('/oauth/token', {
            'grant_type': 'authorization_code',
            'code': auth_code.code,
            'redirect_uri': 'http://localhost:8080/callback',
            'client_id': self.oauth_client.client_id,
            'client_secret': self.oauth_client.client_secret,
            'code_verifier': code_verifier
        }, content_type='application/x-www-form-urlencoded')
        
        if response.status_code != 200:
            data = response.json()
            return self.log("Token Exchange", False, f"Status {response.status_code}: {data}")
        
        tokens = response.json()
        
        if 'access_token' not in tokens:
            return self.log("Token Exchange", False, "No access_token in response")
        
        if 'id_token' not in tokens:
            return self.log("Token Exchange", False, "No id_token in response")
        
        self.log("Token Exchange", True, f"Got access_token and id_token")
        
        # Store tokens for further tests
        self.access_token = tokens['access_token']
        self.id_token = tokens['id_token']
        self.refresh_token = tokens.get('refresh_token')
        
        # Verify ID token structure
        try:
            parts = self.id_token.split('.')
            header = json.loads(base64.urlsafe_b64decode(parts[0] + '=='))
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=='))
            
            self.log("ID Token Structure", True, 
                    f"Algorithm: {header.get('alg')}, Subject: {payload.get('sub')}")
            
            # Check required claims
            required_claims = ['iss', 'sub', 'aud', 'exp', 'iat']
            missing_claims = [c for c in required_claims if c not in payload]
            
            if missing_claims:
                self.log("ID Token Claims", False, f"Missing claims: {missing_claims}")
            else:
                self.log("ID Token Claims", True, "All required claims present")
                
            # Check nonce
            if payload.get('nonce') != nonce:
                self.log("ID Token Nonce", False, "Nonce mismatch")
            else:
                self.log("ID Token Nonce", True, "Nonce matches")
                
        except Exception as e:
            self.log("ID Token Decode", False, str(e))
        
        return True
    
    def test_userinfo_endpoint(self):
        """Test /oauth/userinfo"""
        if not hasattr(self, 'access_token'):
            return self.log("UserInfo Endpoint", False, "No access token (run auth flow first)")
        
        response = self.client.get('/oauth/userinfo',
            HTTP_AUTHORIZATION=f'Bearer {self.access_token}'
        )
        
        if response.status_code != 200:
            return self.log("UserInfo Endpoint", False, f"Status {response.status_code}")
        
        data = response.json()
        
        if 'sub' not in data:
            return self.log("UserInfo Endpoint", False, "No 'sub' claim")
        
        return self.log("UserInfo Endpoint", True, 
                       f"User: {data.get('email', data.get('sub'))}")
    
    def test_userinfo_no_auth(self):
        """Test /oauth/userinfo without authorization"""
        response = self.client.get('/oauth/userinfo')
        
        if response.status_code == 401:
            return self.log("UserInfo (No Auth)", True, "Returns 401 as expected")
        
        return self.log("UserInfo (No Auth)", False, 
                       f"Expected 401, got {response.status_code}")
    
    def test_token_refresh(self):
        """Test token refresh"""
        if not hasattr(self, 'refresh_token') or not self.refresh_token:
            return self.log("Token Refresh", False, "No refresh token available")
        
        response = self.client.post('/oauth/token', {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.oauth_client.client_id,
            'client_secret': self.oauth_client.client_secret
        }, content_type='application/x-www-form-urlencoded')
        
        if response.status_code != 200:
            return self.log("Token Refresh", False, f"Status {response.status_code}")
        
        data = response.json()
        if 'access_token' not in data:
            return self.log("Token Refresh", False, "No new access_token")
        
        self.access_token = data['access_token']
        if data.get('refresh_token'):
            self.refresh_token = data['refresh_token']
        
        return self.log("Token Refresh", True, "Got new tokens")
    
    def test_token_introspection(self):
        """Test /oauth/introspect"""
        if not hasattr(self, 'access_token'):
            return self.log("Token Introspection", False, "No access token")
        
        response = self.client.post('/oauth/introspect', {
            'token': self.access_token,
            'client_id': self.oauth_client.client_id,
            'client_secret': self.oauth_client.client_secret
        }, content_type='application/x-www-form-urlencoded')
        
        if response.status_code != 200:
            return self.log("Token Introspection", False, f"Status {response.status_code}")
        
        data = response.json()
        if not data.get('active'):
            return self.log("Token Introspection", False, "Token not active")
        
        return self.log("Token Introspection", True, 
                       f"Token active, expires: {data.get('exp')}")
    
    def test_token_revocation(self):
        """Test /oauth/revoke"""
        if not hasattr(self, 'access_token'):
            return self.log("Token Revocation", False, "No access token")
        
        # Revoke access token
        response = self.client.post('/oauth/revoke', {
            'token': self.access_token,
            'token_type_hint': 'access_token',
            'client_id': self.oauth_client.client_id,
            'client_secret': self.oauth_client.client_secret
        }, content_type='application/x-www-form-urlencoded')
        
        if response.status_code != 200:
            return self.log("Token Revocation", False, f"Status {response.status_code}")
        
        self.log("Token Revocation", True, "Token revoked")
        
        # Verify token is now invalid
        response = self.client.post('/oauth/introspect', {
            'token': self.access_token,
            'client_id': self.oauth_client.client_id,
            'client_secret': self.oauth_client.client_secret
        }, content_type='application/x-www-form-urlencoded')
        
        data = response.json()
        if data.get('active'):
            return self.log("Token Revocation Verify", False, "Token still active after revocation")
        
        return self.log("Token Revocation Verify", True, "Token confirmed revoked")
    
    def test_client_credentials_flow(self):
        """Test client credentials grant"""
        response = self.client.post('/oauth/token', {
            'grant_type': 'client_credentials',
            'client_id': self.oauth_client.client_id,
            'client_secret': self.oauth_client.client_secret,
            'scope': 'openid'
        }, content_type='application/x-www-form-urlencoded')
        
        if response.status_code != 200:
            data = response.json()
            # Client credentials might not be supported
            if data.get('error') == 'unsupported_grant_type':
                return self.log("Client Credentials", True, 
                              "Grant type not supported (expected)")
            return self.log("Client Credentials", False, f"Status {response.status_code}")
        
        data = response.json()
        if 'access_token' not in data:
            return self.log("Client Credentials", False, "No access_token")
        
        return self.log("Client Credentials", True, "Got access token")
    
    def test_invalid_client(self):
        """Test with invalid client credentials"""
        response = self.client.post('/oauth/token', {
            'grant_type': 'authorization_code',
            'code': 'some_code',
            'client_id': 'invalid_client',
            'client_secret': 'invalid_secret'
        }, content_type='application/x-www-form-urlencoded')
        
        if response.status_code == 401 or 'error' in response.json():
            return self.log("Invalid Client", True, "Correctly rejected invalid client")
        
        return self.log("Invalid Client", False, "Should reject invalid client")
    
    def cleanup(self):
        """Cleanup test data"""
        print("\n🧹 Cleaning up test data...")
        
        # Delete test auth codes
        AuthorizationCode.objects.filter(client=self.oauth_client).delete()
        
        # Delete test tokens
        OAuthToken.objects.filter(client=self.oauth_client).delete()
        
        print("   Test data cleaned up")
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "=" * 60)
        print("🧪 Running OIDC End-to-End Tests")
        print("=" * 60)
        
        # Discovery tests
        self.test_discovery_endpoint()
        self.test_jwks_endpoint()
        
        # Authorization tests
        self.test_authorization_endpoint()
        self.test_token_endpoint_invalid()
        self.test_invalid_client()
        
        # Simulated auth flow
        self.test_simulated_auth_flow()
        
        # Token usage tests
        self.test_userinfo_endpoint()
        self.test_userinfo_no_auth()
        self.test_token_introspection()
        
        # Token refresh
        self.test_token_refresh()
        
        # Client credentials
        self.test_client_credentials_flow()
        
        # Token revocation (do this last as it invalidates tokens)
        self.test_token_revocation()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        passed = sum(1 for r in self.results if r['success'])
        failed = len(self.results) - passed
        
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        print(f"   Total:  {len(self.results)}")
        print(f"   Passed: {passed} ✅")
        print(f"   Failed: {failed} ❌")
        print(f"   Rate:   {(passed/len(self.results)*100):.1f}%")
        
        if failed > 0:
            print("\n   Failed Tests:")
            for r in self.results:
                if not r['success']:
                    print(f"   - {r['test']}: {r['message']}")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='OIDC End-to-End Tests')
    parser.add_argument('--client-id', help='OAuth Client ID')
    parser.add_argument('--client-secret', help='OAuth Client Secret')
    parser.add_argument('--user-id', help='Test User ID (UUID)')
    parser.add_argument('--cleanup', action='store_true', help='Cleanup test data after')
    parser.add_argument('--base-url', default='http://localhost:8000', help='Base URL')
    
    args = parser.parse_args()
    
    tester = OIDCEndToEndTest(base_url=args.base_url)
    
    if not tester.setup_test_data(args.client_id, args.client_secret, args.user_id):
        print("❌ Failed to setup test data")
        sys.exit(1)
    
    try:
        tester.run_all_tests()
    finally:
        if args.cleanup:
            tester.cleanup()


if __name__ == '__main__':
    main()

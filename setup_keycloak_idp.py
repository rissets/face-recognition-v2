#!/usr/bin/env python3
"""
Keycloak Integration Script for Face Recognition OIDC Provider

This script helps configure Face Recognition as an Identity Provider in Keycloak.

Usage:
    python setup_keycloak_idp.py --create-client
    python setup_keycloak_idp.py --test-connection
    python setup_keycloak_idp.py --configure-idp
"""

import os
import sys
import json
import argparse
import requests
from urllib.parse import urljoin

# Keycloak Configuration
KEYCLOAK_CONFIG = {
    'base_url': 'https://sso-dev.kemenkum.go.id',
    'realm': 'master',  # Change this to your realm
    'admin_username': 'admin.ahu',
    'admin_password': 'PassWordssoAHU2025#',
}

# Face Recognition OIDC Provider Configuration
FACE_OIDC_CONFIG = {
    'base_url': 'http://192.168.1.41:8003',
    'client_name': 'Face Recognition Provider',
    'alias': 'face-recognition',
    'display_name': 'Login dengan Face Recognition',
}


class KeycloakAdmin:
    """Keycloak Admin API Client"""
    
    def __init__(self, base_url, realm='master', username=None, password=None):
        self.base_url = base_url.rstrip('/')
        self.realm = realm
        self.username = username
        self.password = password
        self.access_token = None
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification for dev
        
    def authenticate(self):
        """Get admin access token"""
        token_url = f"{self.base_url}/realms/master/protocol/openid-connect/token"
        
        print(f"🔐 Authenticating to Keycloak at {self.base_url}...")
        
        try:
            response = self.session.post(token_url, data={
                'grant_type': 'password',
                'client_id': 'admin-cli',
                'username': self.username,
                'password': self.password,
            })
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data['access_token']
                print("✅ Authentication successful!")
                return True
            else:
                print(f"❌ Authentication failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def _headers(self):
        """Get authorization headers"""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    def get_realms(self):
        """List all realms"""
        url = f"{self.base_url}/admin/realms"
        response = self.session.get(url, headers=self._headers())
        return response.json() if response.status_code == 200 else []
    
    def get_identity_providers(self, realm=None):
        """List identity providers for a realm"""
        realm = realm or self.realm
        url = f"{self.base_url}/admin/realms/{realm}/identity-provider/instances"
        response = self.session.get(url, headers=self._headers())
        return response.json() if response.status_code == 200 else []
    
    def create_identity_provider(self, realm=None, config=None):
        """Create a new identity provider"""
        realm = realm or self.realm
        url = f"{self.base_url}/admin/realms/{realm}/identity-provider/instances"
        
        response = self.session.post(url, headers=self._headers(), json=config)
        
        if response.status_code == 201:
            print("✅ Identity Provider created successfully!")
            return True
        elif response.status_code == 409:
            print("⚠️ Identity Provider already exists")
            return True
        else:
            print(f"❌ Failed to create IDP: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    def update_identity_provider(self, alias, realm=None, config=None):
        """Update an existing identity provider"""
        realm = realm or self.realm
        url = f"{self.base_url}/admin/realms/{realm}/identity-provider/instances/{alias}"
        
        response = self.session.put(url, headers=self._headers(), json=config)
        
        if response.status_code in [200, 204]:
            print("✅ Identity Provider updated successfully!")
            return True
        else:
            print(f"❌ Failed to update IDP: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    def delete_identity_provider(self, alias, realm=None):
        """Delete an identity provider"""
        realm = realm or self.realm
        url = f"{self.base_url}/admin/realms/{realm}/identity-provider/instances/{alias}"
        
        response = self.session.delete(url, headers=self._headers())
        
        if response.status_code in [200, 204]:
            print("✅ Identity Provider deleted successfully!")
            return True
        else:
            print(f"❌ Failed to delete IDP: {response.status_code}")
            return False
    
    def create_idp_mapper(self, alias, mapper_config, realm=None):
        """Create an IDP attribute mapper"""
        realm = realm or self.realm
        url = f"{self.base_url}/admin/realms/{realm}/identity-provider/instances/{alias}/mappers"
        
        response = self.session.post(url, headers=self._headers(), json=mapper_config)
        
        if response.status_code == 201:
            print(f"✅ Mapper '{mapper_config.get('name')}' created successfully!")
            return True
        elif response.status_code == 409:
            print(f"⚠️ Mapper '{mapper_config.get('name')}' already exists")
            return True
        else:
            print(f"❌ Failed to create mapper: {response.status_code}")
            return False


def test_face_oidc_discovery(base_url):
    """Test Face Recognition OIDC discovery endpoint"""
    discovery_url = f"{base_url}/.well-known/openid-configuration"
    
    print(f"\n📡 Testing Face Recognition OIDC at {base_url}...")
    
    try:
        response = requests.get(discovery_url, verify=False, timeout=10)
        
        if response.status_code == 200:
            config = response.json()
            print("✅ OIDC Discovery successful!")
            print(f"   Issuer: {config.get('issuer')}")
            print(f"   Authorization: {config.get('authorization_endpoint')}")
            print(f"   Token: {config.get('token_endpoint')}")
            print(f"   UserInfo: {config.get('userinfo_endpoint')}")
            print(f"   JWKS: {config.get('jwks_uri')}")
            return config
        else:
            print(f"❌ Discovery failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None


def create_face_recognition_idp_config(face_url, client_id, client_secret):
    """Create Keycloak IDP configuration for Face Recognition"""
    
    return {
        'alias': FACE_OIDC_CONFIG['alias'],
        'displayName': FACE_OIDC_CONFIG['display_name'],
        'providerId': 'oidc',
        'enabled': True,
        'trustEmail': True,
        'storeToken': True,
        'addReadTokenRoleOnCreate': True,
        'firstBrokerLoginFlowAlias': 'first broker login',
        'config': {
            'clientId': client_id,
            'clientSecret': client_secret,
            'authorizationUrl': f'{face_url}/oauth/authorize',
            'tokenUrl': f'{face_url}/oauth/token',
            'userInfoUrl': f'{face_url}/oauth/userinfo',
            'logoutUrl': f'{face_url}/oauth/logout',
            'jwksUrl': f'{face_url}/.well-known/jwks.json',
            'issuer': face_url,
            'defaultScope': 'openid profile email face_auth',
            'validateSignature': 'true',
            'useJwksUrl': 'true',
            'pkceEnabled': 'true',
            'pkceMethod': 'S256',
            'syncMode': 'IMPORT',
            'clientAuthMethod': 'client_secret_post',
            'backchannelSupported': 'true',
        }
    }


def create_idp_mappers():
    """Create attribute mappers for Face Recognition IDP"""
    return [
        {
            'name': 'Face Verified',
            'identityProviderAlias': FACE_OIDC_CONFIG['alias'],
            'identityProviderMapper': 'oidc-user-attribute-idp-mapper',
            'config': {
                'claim': 'face_verified',
                'user.attribute': 'face_verified',
                'syncMode': 'INHERIT'
            }
        },
        {
            'name': 'Liveness Verified',
            'identityProviderAlias': FACE_OIDC_CONFIG['alias'],
            'identityProviderMapper': 'oidc-user-attribute-idp-mapper',
            'config': {
                'claim': 'liveness_verified',
                'user.attribute': 'liveness_verified',
                'syncMode': 'INHERIT'
            }
        },
        {
            'name': 'Face Confidence',
            'identityProviderAlias': FACE_OIDC_CONFIG['alias'],
            'identityProviderMapper': 'oidc-user-attribute-idp-mapper',
            'config': {
                'claim': 'face_confidence',
                'user.attribute': 'face_confidence',
                'syncMode': 'INHERIT'
            }
        },
        {
            'name': 'Email',
            'identityProviderAlias': FACE_OIDC_CONFIG['alias'],
            'identityProviderMapper': 'oidc-user-attribute-idp-mapper',
            'config': {
                'claim': 'email',
                'user.attribute': 'email',
                'syncMode': 'INHERIT'
            }
        },
        {
            'name': 'First Name',
            'identityProviderAlias': FACE_OIDC_CONFIG['alias'],
            'identityProviderMapper': 'oidc-user-attribute-idp-mapper',
            'config': {
                'claim': 'given_name',
                'user.attribute': 'firstName',
                'syncMode': 'INHERIT'
            }
        },
        {
            'name': 'Last Name',
            'identityProviderAlias': FACE_OIDC_CONFIG['alias'],
            'identityProviderMapper': 'oidc-user-attribute-idp-mapper',
            'config': {
                'claim': 'family_name',
                'user.attribute': 'lastName',
                'syncMode': 'INHERIT'
            }
        },
    ]


def main():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    parser = argparse.ArgumentParser(description='Keycloak Face Recognition IDP Setup')
    parser.add_argument('--test-connection', action='store_true', help='Test connections')
    parser.add_argument('--create-client', action='store_true', help='Create OAuth client in Face Recognition')
    parser.add_argument('--configure-idp', action='store_true', help='Configure IDP in Keycloak')
    parser.add_argument('--delete-idp', action='store_true', help='Delete IDP from Keycloak')
    parser.add_argument('--list-realms', action='store_true', help='List Keycloak realms')
    parser.add_argument('--realm', default='master', help='Target Keycloak realm')
    parser.add_argument('--client-id', help='Face Recognition OAuth client ID')
    parser.add_argument('--client-secret', help='Face Recognition OAuth client secret')
    parser.add_argument('--face-url', default=FACE_OIDC_CONFIG['base_url'], help='Face Recognition base URL')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔐 Keycloak + Face Recognition Integration Setup")
    print("=" * 60)
    print(f"\n📍 Keycloak URL: {KEYCLOAK_CONFIG['base_url']}")
    print(f"📍 Face Recognition URL: {args.face_url}")
    print(f"📍 Target Realm: {args.realm}")
    
    if args.test_connection:
        # Test Face Recognition OIDC
        face_config = test_face_oidc_discovery(args.face_url)
        
        # Test Keycloak connection
        print(f"\n📡 Testing Keycloak connection...")
        keycloak = KeycloakAdmin(
            KEYCLOAK_CONFIG['base_url'],
            args.realm,
            KEYCLOAK_CONFIG['admin_username'],
            KEYCLOAK_CONFIG['admin_password']
        )
        
        if keycloak.authenticate():
            realms = keycloak.get_realms()
            print(f"   Available realms: {[r['realm'] for r in realms]}")
            
            idps = keycloak.get_identity_providers(args.realm)
            print(f"   Existing IDPs in {args.realm}: {[i['alias'] for i in idps]}")
    
    elif args.create_client:
        print("\n📝 Creating OAuth Client in Face Recognition...")
        print("\n   Run this command in Django shell:")
        print(f"""
   cd face_recognition_app
   python manage.py create_oidc_client \\
       --name "Keycloak SSO" \\
       --redirect-uri "https://sso-dev.kemenkum.go.id/realms/{args.realm}/broker/{FACE_OIDC_CONFIG['alias']}/endpoint"
        """)
    
    elif args.list_realms:
        keycloak = KeycloakAdmin(
            KEYCLOAK_CONFIG['base_url'],
            args.realm,
            KEYCLOAK_CONFIG['admin_username'],
            KEYCLOAK_CONFIG['admin_password']
        )
        
        if keycloak.authenticate():
            realms = keycloak.get_realms()
            print("\n📋 Available Realms:")
            for realm in realms:
                print(f"   - {realm['realm']} (Enabled: {realm.get('enabled', True)})")
    
    elif args.configure_idp:
        if not args.client_id or not args.client_secret:
            print("\n❌ Error: --client-id and --client-secret required")
            print("   Run with --create-client first to get credentials")
            sys.exit(1)
        
        # Test Face Recognition first
        face_config = test_face_oidc_discovery(args.face_url)
        if not face_config:
            print("\n❌ Cannot reach Face Recognition OIDC. Make sure server is running.")
            sys.exit(1)
        
        # Connect to Keycloak
        keycloak = KeycloakAdmin(
            KEYCLOAK_CONFIG['base_url'],
            args.realm,
            KEYCLOAK_CONFIG['admin_username'],
            KEYCLOAK_CONFIG['admin_password']
        )
        
        if not keycloak.authenticate():
            sys.exit(1)
        
        # Create IDP configuration
        idp_config = create_face_recognition_idp_config(
            args.face_url,
            args.client_id,
            args.client_secret
        )
        
        print(f"\n📝 Creating Identity Provider in realm '{args.realm}'...")
        
        if keycloak.create_identity_provider(args.realm, idp_config):
            # Create attribute mappers
            print("\n📝 Creating attribute mappers...")
            for mapper in create_idp_mappers():
                keycloak.create_idp_mapper(FACE_OIDC_CONFIG['alias'], mapper, args.realm)
            
            print("\n" + "=" * 60)
            print("✅ Keycloak Integration Complete!")
            print("=" * 60)
            print(f"\n🔗 IDP Alias: {FACE_OIDC_CONFIG['alias']}")
            print(f"🔗 Callback URL for Face Recognition:")
            print(f"   https://sso-dev.kemenkum.go.id/realms/{args.realm}/broker/{FACE_OIDC_CONFIG['alias']}/endpoint")
            print(f"\n📝 Make sure to add this callback URL to your OAuth client redirect_uris!")
    
    elif args.delete_idp:
        keycloak = KeycloakAdmin(
            KEYCLOAK_CONFIG['base_url'],
            args.realm,
            KEYCLOAK_CONFIG['admin_username'],
            KEYCLOAK_CONFIG['admin_password']
        )
        
        if keycloak.authenticate():
            keycloak.delete_identity_provider(FACE_OIDC_CONFIG['alias'], args.realm)
    
    else:
        parser.print_help()
        print("\n📋 Quick Start:")
        print("   1. python setup_keycloak_idp.py --test-connection")
        print("   2. python setup_keycloak_idp.py --create-client")
        print("   3. python setup_keycloak_idp.py --configure-idp --client-id XXX --client-secret XXX")


if __name__ == '__main__':
    main()

"""
Authentication classes for Face Recognition Third-Party Service
"""
import jwt
import hashlib
import hmac
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from rest_framework import authentication, exceptions
from rest_framework.request import Request
from clients.models import Client
import logging

logger = logging.getLogger(__name__)


class APIKeyAuthentication(authentication.BaseAuthentication):
    """
    API Key authentication for third-party clients
    """
    
    def authenticate(self, request):
        api_key = self.get_api_key(request)
        if not api_key:
            return None
        
        client = Client.find_active_by_api_key(api_key)
        if not client:
            raise exceptions.AuthenticationFailed('Invalid API key')
        
        # Update client last activity
        client.update_last_activity()

        # Attach client to request for downstream usage
        request.client = client
        request.api_key = api_key
        
        # Create a user-like object for the client
        user = ClientUser(client)
        return (user, client)
    
    def get_api_key(self, request):
        """Extract API key from request headers"""
        # Try X-API-Key header first
        api_key = request.META.get('HTTP_X_API_KEY')
        if api_key:
            return api_key
        
        # Try Authorization header with Bearer
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if auth_header and auth_header.startswith('Bearer '):
            return auth_header[7:]  # Remove 'Bearer ' prefix
        
        return None


class ClientUser:
    """
    Represents a client as a user-like object for DRF compatibility
    """
    
    def __init__(self, client):
        self.client = client
        self.id = client.id
        self.pk = client.id
        self.username = client.client_id
        self.is_authenticated = True
        self.is_active = client.is_active
        self.is_anonymous = False
        self.is_staff = False
        self.is_superuser = False
    
    def __str__(self):
        return f"Client: {self.client.name}"
    
    def has_perm(self, perm, obj=None):
        """Check if client has a specific permission"""
        # Basic permissions based on client tier and features
        if perm == 'enrollment':
            return self.client.is_feature_enabled('enrollment')
        elif perm == 'recognition':
            return self.client.is_feature_enabled('recognition')
        elif perm == 'liveness_detection':
            return self.client.is_feature_enabled('liveness_detection')
        elif perm == 'analytics':
            return self.client.is_feature_enabled('analytics')
        
        return False
    
    def has_module_perms(self, app_label):
        """Check if client has permissions for an app"""
        return self.is_active
    
    def get_all_permissions(self, obj=None):
        """Get all permissions for this client"""
        permissions = set()
        features = self.client.features or {}
        
        for feature, enabled in features.items():
            if enabled and isinstance(enabled, bool):
                permissions.add(feature)
        
        return permissions


class WebhookSignatureAuthentication(authentication.BaseAuthentication):
    """
    Webhook signature authentication for incoming webhooks
    """
    
    def authenticate(self, request):
        # Only authenticate webhook endpoints
        if not request.path.startswith('/webhooks/'):
            return None
        
        signature = self.get_signature(request)
        timestamp = self.get_timestamp(request)
        
        if not signature or not timestamp:
            raise exceptions.AuthenticationFailed('Missing webhook signature or timestamp')
        
        # Get client based on some identifier in the request
        client_id = request.data.get('client_id') or request.GET.get('client_id')
        if not client_id:
            raise exceptions.AuthenticationFailed('Missing client identifier')
        
        try:
            client = Client.objects.get(client_id=client_id, status='active')
        except Client.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid client')
        
        # Verify signature
        if not self.verify_signature(request, client, signature, timestamp):
            raise exceptions.AuthenticationFailed('Invalid webhook signature')
        
        user = ClientUser(client)
        return (user, client)
    
    def get_signature(self, request):
        """Extract signature from request headers"""
        return request.META.get('HTTP_X_FR_SIGNATURE')
    
    def get_timestamp(self, request):
        """Extract timestamp from request headers"""
        return request.META.get('HTTP_X_FR_TIMESTAMP')
    
    def verify_signature(self, request, client, signature, timestamp):
        """Verify webhook signature"""
        try:
            # Reconstruct the payload
            if hasattr(request, '_body'):
                payload = request._body.decode('utf-8')
            else:
                payload = request.body.decode('utf-8')
            
            # Create signature payload
            sig_payload = f"{timestamp}.{payload}"
            
            # Generate expected signature
            expected_signature = hmac.new(
                client.webhook_secret.encode('utf-8'),
                sig_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Webhook signature verification error: {e}")
            return False


class JWTClientAuthentication(authentication.BaseAuthentication):
    """
    JWT authentication using client's secret key
    """
    
    def authenticate(self, request):
        jwt_token = self.get_jwt_token(request)
        if not jwt_token:
            return None
        
        try:
            # Decode JWT header to get client_id
            unverified_header = jwt.get_unverified_header(jwt_token)
            unverified_payload = jwt.decode(
                jwt_token, 
                options={"verify_signature": False}
            )
            
            client_id = unverified_payload.get('client_id')
            if not client_id:
                raise exceptions.AuthenticationFailed('Missing client_id in JWT')
            
            # Get client and verify JWT with their secret
            client = Client.objects.get(client_id=client_id, status='active')
            
            # Decode and verify JWT
            payload = jwt.decode(
                jwt_token,
                client.secret_key,
                algorithms=['HS256']
            )
            
            # Additional validation
            if payload.get('client_id') != client.client_id:
                raise exceptions.AuthenticationFailed('Client ID mismatch')
            
            # Update client activity
            client.update_last_activity()

            # Attach client to request for downstream usage
            request.client = client
            request.jwt_payload = payload
            
            user = ClientUser(client)
            return (user, client)
            
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed('JWT token expired')
        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed('Invalid JWT token')
        except Client.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid client')
        except Exception as e:
            logger.error(f"JWT authentication error: {e}")
            raise exceptions.AuthenticationFailed('JWT authentication failed')
    
    def get_jwt_token(self, request):
        """Extract JWT token from request"""
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if auth_header and auth_header.startswith('JWT '):
            return auth_header[4:]  # Remove 'JWT ' prefix
        
        return None

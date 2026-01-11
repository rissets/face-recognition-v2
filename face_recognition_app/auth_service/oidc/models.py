"""
OAuth 2.0 / OpenID Connect Models for Face Recognition Identity Provider

This module implements the database models needed for a complete OIDC Provider
that can be used with Keycloak as an external Identity Provider.
"""
import uuid
import secrets
import hashlib
import base64
from datetime import timedelta

from django.db import models
from django.utils import timezone
from django.conf import settings
from django.core.validators import URLValidator


def generate_client_id():
    """Generate a unique client ID"""
    return f"oidc_{secrets.token_urlsafe(24)}"


def generate_client_secret():
    """Generate a secure client secret"""
    return secrets.token_urlsafe(48)


class OAuthClient(models.Model):
    """
    OAuth 2.0 Client registration for third-party applications (like Keycloak)
    """
    
    CLIENT_TYPE_CHOICES = [
        ('confidential', 'Confidential'),  # Can securely store credentials
        ('public', 'Public'),  # Cannot securely store credentials (SPA, mobile)
    ]
    
    GRANT_TYPE_CHOICES = [
        ('authorization_code', 'Authorization Code'),
        ('client_credentials', 'Client Credentials'),
        ('refresh_token', 'Refresh Token'),
    ]
    
    # Primary identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client_id = models.CharField(max_length=100, unique=True, default=generate_client_id, db_index=True)
    client_secret = models.CharField(max_length=255, default=generate_client_secret)
    
    # Client metadata
    name = models.CharField(max_length=255, help_text="Human-readable client name")
    description = models.TextField(blank=True)
    logo_url = models.URLField(blank=True, null=True)
    
    # Client configuration
    client_type = models.CharField(max_length=20, choices=CLIENT_TYPE_CHOICES, default='confidential')
    
    # Allowed redirect URIs (one per line or comma-separated)
    redirect_uris = models.TextField(
        help_text="Allowed redirect URIs (one per line)",
        validators=[],
    )
    
    # Allowed grant types
    grant_types = models.JSONField(
        default=list,
        help_text="List of allowed grant types"
    )
    
    # Allowed response types
    response_types = models.JSONField(
        default=list,
        help_text="List of allowed response types (code, token, id_token)"
    )
    
    # Allowed scopes
    allowed_scopes = models.JSONField(
        default=list,
        help_text="List of allowed scopes"
    )
    
    # Token settings
    access_token_lifetime = models.PositiveIntegerField(
        default=3600,
        help_text="Access token lifetime in seconds"
    )
    refresh_token_lifetime = models.PositiveIntegerField(
        default=86400,
        help_text="Refresh token lifetime in seconds"
    )
    id_token_lifetime = models.PositiveIntegerField(
        default=3600,
        help_text="ID token lifetime in seconds"
    )
    
    # Security settings
    require_pkce = models.BooleanField(
        default=True,
        help_text="Require PKCE for authorization code flow"
    )
    require_consent = models.BooleanField(
        default=True,
        help_text="Require user consent before issuing tokens"
    )
    
    # Face authentication settings
    require_face_auth = models.BooleanField(
        default=True,
        help_text="Require face authentication for login"
    )
    require_liveness = models.BooleanField(
        default=True,
        help_text="Require liveness detection during face auth"
    )
    min_confidence_score = models.FloatField(
        default=0.85,
        help_text="Minimum face recognition confidence score"
    )
    
    # Associated client (optional - links to existing API client)
    api_client = models.ForeignKey(
        'clients.Client',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='oauth_clients'
    )
    
    # Status
    is_active = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "OAuth Client"
        verbose_name_plural = "OAuth Clients"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.client_id})"
    
    def get_redirect_uris_list(self):
        """Get redirect URIs as a list"""
        if not self.redirect_uris:
            return []
        return [uri.strip() for uri in self.redirect_uris.replace(',', '\n').split('\n') if uri.strip()]
    
    def is_redirect_uri_valid(self, redirect_uri):
        """Check if a redirect URI is allowed"""
        return redirect_uri in self.get_redirect_uris_list()
    
    def is_scope_allowed(self, scope):
        """Check if a scope is allowed for this client"""
        if not self.allowed_scopes:
            # Default allowed scopes
            return scope in ['openid', 'profile', 'email', 'face_auth']
        return scope in self.allowed_scopes
    
    def verify_secret(self, secret):
        """Verify the client secret"""
        return secrets.compare_digest(self.client_secret, secret)
    
    def regenerate_secret(self):
        """Regenerate the client secret"""
        self.client_secret = generate_client_secret()
        self.save(update_fields=['client_secret', 'updated_at'])
        return self.client_secret


class AuthorizationCode(models.Model):
    """
    OAuth 2.0 Authorization Code
    
    Short-lived code exchanged for access token
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    code = models.CharField(max_length=255, unique=True, db_index=True)
    
    # Client and user
    client = models.ForeignKey(OAuthClient, on_delete=models.CASCADE, related_name='authorization_codes')
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.CASCADE,
        related_name='oauth_authorization_codes'
    )
    
    # Authorization details
    redirect_uri = models.TextField()
    scope = models.CharField(max_length=500)
    state = models.CharField(max_length=255, blank=True)
    nonce = models.CharField(max_length=255, blank=True)
    
    # PKCE
    code_challenge = models.CharField(max_length=255, blank=True)
    code_challenge_method = models.CharField(max_length=10, blank=True)  # plain or S256
    
    # Face authentication session
    auth_session = models.ForeignKey(
        'auth_service.AuthenticationSession',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='oauth_codes'
    )
    
    # Status
    is_used = models.BooleanField(default=False)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    
    class Meta:
        verbose_name = "Authorization Code"
        verbose_name_plural = "Authorization Codes"
        indexes = [
            models.Index(fields=['code']),
            models.Index(fields=['client', 'expires_at']),
        ]
    
    def __str__(self):
        return f"AuthCode for {self.client_user} via {self.client.name}"
    
    def save(self, *args, **kwargs):
        if not self.code:
            self.code = secrets.token_urlsafe(32)
        if not self.expires_at:
            # Authorization codes expire in 10 minutes
            self.expires_at = timezone.now() + timedelta(minutes=10)
        super().save(*args, **kwargs)
    
    @property
    def is_expired(self):
        return timezone.now() > self.expires_at
    
    @property
    def is_valid(self):
        return not self.is_used and not self.is_expired
    
    def verify_pkce(self, code_verifier):
        """Verify PKCE code verifier"""
        if not self.code_challenge:
            return True  # PKCE not used
        
        if self.code_challenge_method == 'S256':
            # SHA256 hash
            challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).decode().rstrip('=')
        else:
            # Plain
            challenge = code_verifier
        
        return secrets.compare_digest(self.code_challenge, challenge)
    
    def mark_used(self):
        """Mark the code as used"""
        self.is_used = True
        self.save(update_fields=['is_used'])


class OAuthToken(models.Model):
    """
    OAuth 2.0 Access Token and Refresh Token
    """
    
    TOKEN_TYPE_CHOICES = [
        ('access', 'Access Token'),
        ('refresh', 'Refresh Token'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    token = models.CharField(max_length=255, unique=True, db_index=True)
    token_type = models.CharField(max_length=10, choices=TOKEN_TYPE_CHOICES)
    
    # Client and user
    client = models.ForeignKey(OAuthClient, on_delete=models.CASCADE, related_name='tokens')
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='oauth_tokens'
    )
    
    # Token details
    scope = models.CharField(max_length=500)
    
    # For refresh tokens - reference to parent access token
    parent_token = models.ForeignKey(
        'self',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='child_tokens'
    )
    
    # Status
    is_revoked = models.BooleanField(default=False)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    last_used_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "OAuth Token"
        verbose_name_plural = "OAuth Tokens"
        indexes = [
            models.Index(fields=['token']),
            models.Index(fields=['client', 'token_type', 'expires_at']),
            models.Index(fields=['client_user', 'token_type']),
        ]
    
    def __str__(self):
        return f"{self.token_type} token for {self.client_user or 'client'}"
    
    def save(self, *args, **kwargs):
        if not self.token:
            self.token = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)
    
    @property
    def is_expired(self):
        return timezone.now() > self.expires_at
    
    @property
    def is_valid(self):
        return not self.is_revoked and not self.is_expired
    
    def revoke(self):
        """Revoke this token and all child tokens"""
        self.is_revoked = True
        self.save(update_fields=['is_revoked'])
        # Revoke all child tokens
        self.child_tokens.update(is_revoked=True)
    
    def update_last_used(self):
        """Update last used timestamp"""
        self.last_used_at = timezone.now()
        self.save(update_fields=['last_used_at'])


class OIDCSession(models.Model):
    """
    OpenID Connect Session for tracking user sessions
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_id = models.CharField(max_length=255, unique=True, db_index=True)
    
    # User
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.CASCADE,
        related_name='oidc_sessions'
    )
    
    # Associated tokens
    access_token = models.ForeignKey(
        OAuthToken,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='oidc_sessions'
    )
    
    # Session info
    user_agent = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    # Face auth info
    face_auth_confidence = models.FloatField(null=True, blank=True)
    liveness_verified = models.BooleanField(default=False)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    last_activity_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "OIDC Session"
        verbose_name_plural = "OIDC Sessions"
    
    def __str__(self):
        return f"Session {self.session_id[:8]}... for {self.client_user}"
    
    def save(self, *args, **kwargs):
        if not self.session_id:
            self.session_id = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)
    
    @property
    def is_expired(self):
        return timezone.now() > self.expires_at
    
    @property
    def is_valid(self):
        return not self.is_expired


class UserConsent(models.Model):
    """
    User consent records for OAuth clients
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Client and user
    client = models.ForeignKey(OAuthClient, on_delete=models.CASCADE, related_name='user_consents')
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.CASCADE,
        related_name='oauth_consents'
    )
    
    # Consented scopes
    scopes = models.JSONField(default=list)
    
    # Timestamps
    granted_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    revoked_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "User Consent"
        verbose_name_plural = "User Consents"
        unique_together = ['client', 'client_user']
    
    def __str__(self):
        return f"Consent from {self.client_user} to {self.client.name}"
    
    @property
    def is_valid(self):
        if self.revoked_at:
            return False
        if self.expires_at and timezone.now() > self.expires_at:
            return False
        return True
    
    def has_scope(self, scope):
        """Check if user has consented to a specific scope"""
        return scope in self.scopes
    
    def revoke(self):
        """Revoke this consent"""
        self.revoked_at = timezone.now()
        self.save(update_fields=['revoked_at'])


class OIDCAuthorizationLog(models.Model):
    """
    Log for authorization attempts and decisions
    """
    
    STATUS_CHOICES = [
        ('initiated', 'Initiated'),
        ('face_auth_pending', 'Face Auth Pending'),
        ('face_auth_success', 'Face Auth Success'),
        ('face_auth_failed', 'Face Auth Failed'),
        ('consent_granted', 'Consent Granted'),
        ('consent_denied', 'Consent Denied'),
        ('code_issued', 'Code Issued'),
        ('error', 'Error'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Client and user
    client = models.ForeignKey(OAuthClient, on_delete=models.CASCADE, related_name='auth_logs')
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='oauth_auth_logs'
    )
    
    # Login identifier (email, username, external_user_id, etc)
    login_identifier = models.CharField(max_length=255, db_index=True)
    
    # Authorization details
    redirect_uri = models.TextField()
    scope = models.CharField(max_length=500)
    state = models.CharField(max_length=255, blank=True)
    nonce = models.CharField(max_length=255, blank=True)
    
    # Status tracking
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default='initiated')
    error_code = models.CharField(max_length=100, blank=True)
    error_description = models.TextField(blank=True)
    
    # Associated records
    authorization_code = models.ForeignKey(
        AuthorizationCode,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='logs'
    )
    auth_session = models.ForeignKey(
        'auth_service.AuthenticationSession',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='oauth_logs'
    )
    
    # Request metadata
    user_agent = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "OIDC Authorization Log"
        verbose_name_plural = "OIDC Authorization Logs"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['client', 'created_at']),
            models.Index(fields=['login_identifier', 'created_at']),
            models.Index(fields=['status', 'created_at']),
        ]
    
    def __str__(self):
        return f"Auth {self.status} for {self.login_identifier} via {self.client.name}"
    
    def update_status(self, status, error_code=None, error_description=None):
        """Update authorization status"""
        self.status = status
        if error_code:
            self.error_code = error_code
        if error_description:
            self.error_description = error_description
        if status in ['code_issued', 'consent_denied', 'face_auth_failed', 'error']:
            self.completed_at = timezone.now()
        self.save()


class OIDCTokenLog(models.Model):
    """
    Log for token operations (issue, refresh, revoke)
    """
    
    OPERATION_CHOICES = [
        ('issue', 'Issued'),
        ('refresh', 'Refreshed'),
        ('revoke', 'Revoked'),
        ('expire', 'Expired'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Token reference
    token = models.ForeignKey(
        OAuthToken,
        on_delete=models.CASCADE,
        related_name='logs'
    )
    
    # Operation
    operation = models.CharField(max_length=20, choices=OPERATION_CHOICES)
    
    # Client and user
    client = models.ForeignKey(OAuthClient, on_delete=models.CASCADE, related_name='token_logs')
    client_user = models.ForeignKey(
        'clients.ClientUser',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='oauth_token_logs'
    )
    
    # Grant type used
    grant_type = models.CharField(max_length=50, blank=True)
    
    # Request metadata
    user_agent = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        verbose_name = "OIDC Token Log"
        verbose_name_plural = "OIDC Token Logs"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['client', 'created_at']),
            models.Index(fields=['operation', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.operation} {self.token.token_type} for {self.client_user or 'client'}"

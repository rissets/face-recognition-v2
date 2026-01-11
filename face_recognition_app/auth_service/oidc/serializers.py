"""
OIDC Serializers for request/response validation
"""
from rest_framework import serializers
from .models import OAuthClient, AuthorizationCode, OAuthToken, UserConsent


class OAuthClientSerializer(serializers.ModelSerializer):
    """Serializer for OAuth Client registration"""
    
    class Meta:
        model = OAuthClient
        fields = [
            'id', 'client_id', 'name', 'description', 'logo_url',
            'client_type', 'redirect_uris', 'grant_types', 'response_types',
            'allowed_scopes', 'access_token_lifetime', 'refresh_token_lifetime',
            'id_token_lifetime', 'require_pkce', 'require_consent',
            'require_face_auth', 'require_liveness', 'min_confidence_score',
            'is_active', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'client_id', 'created_at', 'updated_at']
    
    def create(self, validated_data):
        # Set default grant types if not provided
        if not validated_data.get('grant_types'):
            validated_data['grant_types'] = ['authorization_code', 'refresh_token']
        
        # Set default response types if not provided
        if not validated_data.get('response_types'):
            validated_data['response_types'] = ['code']
        
        # Set default scopes if not provided
        if not validated_data.get('allowed_scopes'):
            validated_data['allowed_scopes'] = ['openid', 'profile', 'email', 'face_auth']
        
        return super().create(validated_data)


class OAuthClientPublicSerializer(serializers.ModelSerializer):
    """Public serializer for OAuth Client (no secrets)"""
    
    class Meta:
        model = OAuthClient
        fields = [
            'client_id', 'name', 'description', 'logo_url',
            'require_face_auth', 'require_liveness'
        ]


class AuthorizationRequestSerializer(serializers.Serializer):
    """Serializer for Authorization Request validation"""
    
    client_id = serializers.CharField(required=True)
    redirect_uri = serializers.URLField(required=True)
    response_type = serializers.CharField(required=True)
    scope = serializers.CharField(required=False, default='openid')
    state = serializers.CharField(required=False, allow_blank=True)
    nonce = serializers.CharField(required=False, allow_blank=True)
    code_challenge = serializers.CharField(required=False, allow_blank=True)
    code_challenge_method = serializers.ChoiceField(
        choices=['plain', 'S256'],
        required=False,
        default='S256'
    )
    prompt = serializers.CharField(required=False, allow_blank=True)
    login_hint = serializers.CharField(required=False, allow_blank=True)
    
    def validate_response_type(self, value):
        allowed = ['code', 'token', 'id_token', 'code token', 'code id_token', 
                   'token id_token', 'code token id_token']
        if value not in allowed:
            raise serializers.ValidationError(f"Unsupported response_type: {value}")
        return value


class TokenRequestSerializer(serializers.Serializer):
    """Serializer for Token Request validation"""
    
    grant_type = serializers.ChoiceField(
        choices=['authorization_code', 'refresh_token', 'client_credentials'],
        required=True
    )
    code = serializers.CharField(required=False)
    redirect_uri = serializers.URLField(required=False)
    code_verifier = serializers.CharField(required=False)
    refresh_token = serializers.CharField(required=False)
    scope = serializers.CharField(required=False)
    client_id = serializers.CharField(required=False)
    client_secret = serializers.CharField(required=False)
    
    def validate(self, attrs):
        grant_type = attrs.get('grant_type')
        
        if grant_type == 'authorization_code':
            if not attrs.get('code'):
                raise serializers.ValidationError("code is required for authorization_code grant")
            if not attrs.get('redirect_uri'):
                raise serializers.ValidationError("redirect_uri is required for authorization_code grant")
        
        elif grant_type == 'refresh_token':
            if not attrs.get('refresh_token'):
                raise serializers.ValidationError("refresh_token is required for refresh_token grant")
        
        return attrs


class TokenResponseSerializer(serializers.Serializer):
    """Serializer for Token Response"""
    
    access_token = serializers.CharField()
    token_type = serializers.CharField(default='Bearer')
    expires_in = serializers.IntegerField()
    refresh_token = serializers.CharField(required=False)
    id_token = serializers.CharField(required=False)
    scope = serializers.CharField(required=False)


class UserInfoResponseSerializer(serializers.Serializer):
    """Serializer for UserInfo Response"""
    
    sub = serializers.CharField()
    name = serializers.CharField(required=False)
    given_name = serializers.CharField(required=False)
    family_name = serializers.CharField(required=False)
    email = serializers.EmailField(required=False)
    email_verified = serializers.BooleanField(required=False)
    picture = serializers.URLField(required=False)
    updated_at = serializers.IntegerField(required=False)
    
    # Face auth claims
    face_verified = serializers.BooleanField(required=False)
    face_confidence = serializers.FloatField(required=False)
    liveness_verified = serializers.BooleanField(required=False)


class IntrospectionRequestSerializer(serializers.Serializer):
    """Serializer for Token Introspection Request"""
    
    token = serializers.CharField(required=True)
    token_type_hint = serializers.ChoiceField(
        choices=['access_token', 'refresh_token'],
        required=False
    )


class IntrospectionResponseSerializer(serializers.Serializer):
    """Serializer for Token Introspection Response"""
    
    active = serializers.BooleanField()
    scope = serializers.CharField(required=False)
    client_id = serializers.CharField(required=False)
    username = serializers.CharField(required=False)
    token_type = serializers.CharField(required=False)
    exp = serializers.IntegerField(required=False)
    iat = serializers.IntegerField(required=False)
    nbf = serializers.IntegerField(required=False)
    sub = serializers.CharField(required=False)
    aud = serializers.CharField(required=False)
    iss = serializers.CharField(required=False)
    jti = serializers.CharField(required=False)


class RevocationRequestSerializer(serializers.Serializer):
    """Serializer for Token Revocation Request"""
    
    token = serializers.CharField(required=True)
    token_type_hint = serializers.ChoiceField(
        choices=['access_token', 'refresh_token'],
        required=False
    )


class FaceAuthCallbackSerializer(serializers.Serializer):
    """Serializer for Face Auth Callback"""
    
    user_id = serializers.CharField(required=True)
    session_token = serializers.CharField(required=False)
    confidence_score = serializers.FloatField(required=False, default=0.0)
    liveness_verified = serializers.BooleanField(required=False, default=False)


class ConsentSubmitSerializer(serializers.Serializer):
    """Serializer for Consent Form Submission"""
    
    consent = serializers.ChoiceField(
        choices=['granted', 'denied'],
        required=True
    )


class OIDCErrorSerializer(serializers.Serializer):
    """Serializer for OIDC Error Response"""
    
    error = serializers.CharField()
    error_description = serializers.CharField(required=False)
    error_uri = serializers.URLField(required=False)
    state = serializers.CharField(required=False)

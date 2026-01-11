"""
OAuth 2.0 / OpenID Connect Views for Face Recognition Identity Provider

Implements standard OIDC endpoints:
- Authorization Endpoint
- Token Endpoint
- UserInfo Endpoint
- JWKS Endpoint
- Discovery Endpoint
- Revocation Endpoint
- Introspection Endpoint
"""
import json
import logging
import secrets
import time
from datetime import timedelta
from typing import Dict, Optional, Tuple, Any
from urllib.parse import urlencode, urlparse, parse_qs

from django.conf import settings
from django.db import transaction
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ObjectDoesNotExist

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from clients.models import ClientUser

from .models import (
    OAuthClient,
    AuthorizationCode,
    OAuthToken,
    OIDCSession,
    UserConsent,
)
from .utils import (
    get_oidc_config,
    get_jwks,
    get_issuer,
    generate_access_token,
    generate_id_token,
    generate_refresh_token,
    decode_access_token,
    validate_token,
    parse_scope,
    build_scope_string,
    verify_code_challenge,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discovery Endpoints
# ---------------------------------------------------------------------------

@api_view(['GET'])
@permission_classes([AllowAny])
def openid_configuration(request):
    """
    OpenID Connect Discovery endpoint
    GET /.well-known/openid-configuration
    """
    config = get_oidc_config()
    return Response(config)


@api_view(['GET'])
@permission_classes([AllowAny])
def jwks_endpoint(request):
    """
    JSON Web Key Set endpoint
    GET /.well-known/jwks.json
    """
    try:
        jwks = get_jwks()
        return Response(jwks)
    except Exception as e:
        logger.error(f"Error generating JWKS: {e}")
        return Response(
            {"error": "server_error", "error_description": "Failed to generate JWKS"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ---------------------------------------------------------------------------
# Authorization Endpoint
# ---------------------------------------------------------------------------

@require_http_methods(["GET", "POST"])
def authorize(request):
    """
    OAuth 2.0 Authorization endpoint
    GET/POST /oauth/authorize
    
    Initiates the authorization flow with face authentication
    """
    # Extract parameters
    if request.method == 'GET':
        params = request.GET
    else:
        params = request.POST
    
    client_id = params.get('client_id')
    redirect_uri = params.get('redirect_uri')
    response_type = params.get('response_type', 'code')
    scope = params.get('scope', 'openid')
    state = params.get('state', '')
    nonce = params.get('nonce', '')
    
    # PKCE parameters
    code_challenge = params.get('code_challenge', '')
    code_challenge_method = params.get('code_challenge_method', 'S256')
    
    # Prompt parameter
    prompt = params.get('prompt', '')
    
    # Login hint (user identifier)
    login_hint = params.get('login_hint', '')
    
    # Validate required parameters
    if not client_id:
        return _authorization_error(
            redirect_uri, state,
            'invalid_request', 'Missing client_id parameter'
        )
    
    if not redirect_uri:
        return _authorization_error(
            None, state,
            'invalid_request', 'Missing redirect_uri parameter'
        )
    
    # Validate client
    try:
        client = OAuthClient.objects.get(client_id=client_id, is_active=True)
    except OAuthClient.DoesNotExist:
        return _authorization_error(
            redirect_uri, state,
            'invalid_client', 'Unknown client'
        )
    
    # Validate redirect URI
    if not client.is_redirect_uri_valid(redirect_uri):
        return _authorization_error(
            None, state,
            'invalid_request', 'Invalid redirect_uri'
        )
    
    # Validate response type
    if response_type not in ['code', 'token', 'id_token', 'code token', 'code id_token']:
        return _authorization_error(
            redirect_uri, state,
            'unsupported_response_type', f'Unsupported response_type: {response_type}'
        )
    
    # Validate scopes
    scopes = parse_scope(scope)
    for s in scopes:
        if not client.is_scope_allowed(s):
            return _authorization_error(
                redirect_uri, state,
                'invalid_scope', f'Scope not allowed: {s}'
            )
    
    # Require PKCE for authorization code flow (if configured)
    if 'code' in response_type and client.require_pkce and not code_challenge:
        return _authorization_error(
            redirect_uri, state,
            'invalid_request', 'PKCE code_challenge required'
        )
    
    # Store authorization request in session
    auth_request = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'response_type': response_type,
        'scope': scope,
        'state': state,
        'nonce': nonce,
        'code_challenge': code_challenge,
        'code_challenge_method': code_challenge_method,
        'login_hint': login_hint,
        'prompt': prompt,
    }
    request.session['oauth_auth_request'] = auth_request
    
    # Redirect to face authentication page
    return HttpResponseRedirect(f'/oauth/face-login?client_id={client_id}')


def _authorization_error(redirect_uri: Optional[str], state: str, error: str, description: str):
    """Return an authorization error response"""
    if redirect_uri:
        params = {
            'error': error,
            'error_description': description,
        }
        if state:
            params['state'] = state
        
        return HttpResponseRedirect(f"{redirect_uri}?{urlencode(params)}")
    else:
        return JsonResponse({
            'error': error,
            'error_description': description,
        }, status=400)


# ---------------------------------------------------------------------------
# Face Authentication Login Page
# ---------------------------------------------------------------------------

def face_login(request):
    """
    Face authentication login page
    GET /oauth/face-login
    
    Renders the face authentication UI
    """
    auth_request = request.session.get('oauth_auth_request')
    
    if not auth_request:
        return render(request, 'oidc/error.html', {
            'error': 'No authorization request found',
            'error_description': 'Please start the login flow from the client application.'
        })
    
    try:
        client = OAuthClient.objects.get(client_id=auth_request['client_id'], is_active=True)
    except OAuthClient.DoesNotExist:
        return render(request, 'oidc/error.html', {
            'error': 'Invalid client',
            'error_description': 'The client application is not registered.'
        })
    
    scopes = parse_scope(auth_request['scope'])
    
    context = {
        'client': client,
        'scopes': scopes,
        'login_hint': auth_request.get('login_hint', ''),
        'websocket_url': f"wss://{request.get_host()}/ws/face-auth/",
        'api_base_url': f"https://{request.get_host()}/api/auth/",
    }
    
    return render(request, 'oidc/face_login.html', context)


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def face_auth_callback(request):
    """
    Callback after successful face authentication
    POST /oauth/face-auth-callback
    
    Called by the face login page after successful authentication
    """
    auth_request = request.session.get('oauth_auth_request')
    
    if not auth_request:
        return Response({
            'error': 'invalid_request',
            'error_description': 'No authorization request found'
        }, status=400)
    
    # Get face auth result from request
    user_id = request.data.get('user_id')
    auth_session_token = request.data.get('session_token')
    confidence_score = request.data.get('confidence_score', 0)
    liveness_verified = request.data.get('liveness_verified', False)
    
    if not user_id:
        return Response({
            'error': 'authentication_failed',
            'error_description': 'Face authentication failed'
        }, status=401)
    
    # Get client and user
    try:
        client = OAuthClient.objects.get(client_id=auth_request['client_id'], is_active=True)
    except OAuthClient.DoesNotExist:
        return Response({
            'error': 'invalid_client',
            'error_description': 'Unknown client'
        }, status=400)
    
    # Get or create the client user
    try:
        if client.api_client:
            client_user = ClientUser.objects.get(
                client=client.api_client,
                external_user_id=str(user_id)
            )
        else:
            client_user = ClientUser.objects.get(id=user_id)
    except ClientUser.DoesNotExist:
        return Response({
            'error': 'user_not_found',
            'error_description': 'User not found'
        }, status=404)
    
    # Validate confidence score
    if confidence_score < client.min_confidence_score:
        return Response({
            'error': 'low_confidence',
            'error_description': f'Face confidence score too low: {confidence_score}'
        }, status=401)
    
    # Validate liveness if required
    if client.require_liveness and not liveness_verified:
        return Response({
            'error': 'liveness_required',
            'error_description': 'Liveness verification required but not passed'
        }, status=401)
    
    # Check user consent
    consent = None
    if client.require_consent:
        try:
            consent = UserConsent.objects.get(client=client, client_user=client_user)
            if not consent.is_valid:
                consent = None
        except UserConsent.DoesNotExist:
            pass
        
        if not consent:
            # Need to show consent page
            request.session['pending_auth_user_id'] = str(client_user.id)
            request.session['face_auth_result'] = {
                'confidence_score': confidence_score,
                'liveness_verified': liveness_verified,
            }
            return Response({
                'redirect': '/oauth/consent',
                'requires_consent': True
            })
    
    # Generate authorization response
    return _generate_authorization_response(
        request, client, client_user, auth_request,
        confidence_score, liveness_verified
    )


def _generate_authorization_response(
    request,
    client: OAuthClient,
    client_user: ClientUser,
    auth_request: Dict,
    confidence_score: float,
    liveness_verified: bool
) -> Response:
    """Generate the authorization response (code or tokens)"""
    
    response_type = auth_request['response_type']
    redirect_uri = auth_request['redirect_uri']
    state = auth_request['state']
    nonce = auth_request['nonce']
    scopes = parse_scope(auth_request['scope'])
    
    response_params = {}
    
    if state:
        response_params['state'] = state
    
    auth_time = int(time.time())
    
    with transaction.atomic():
        # Generate authorization code if requested
        if 'code' in response_type:
            auth_code = AuthorizationCode.objects.create(
                client=client,
                client_user=client_user,
                redirect_uri=redirect_uri,
                scope=auth_request['scope'],
                state=state,
                nonce=nonce,
                code_challenge=auth_request.get('code_challenge', ''),
                code_challenge_method=auth_request.get('code_challenge_method', ''),
            )
            response_params['code'] = auth_code.code
        
        # Generate tokens if implicit flow
        if 'token' in response_type:
            # Create access token
            access_token_value = generate_access_token(
                client_id=client.client_id,
                user_id=str(client_user.id),
                scopes=scopes,
                expires_in=client.access_token_lifetime,
                extra_claims={
                    'face_verified': True,
                    'face_confidence': confidence_score,
                    'liveness_verified': liveness_verified,
                }
            )
            
            OAuthToken.objects.create(
                token=access_token_value[:64],  # Store truncated for lookup
                token_type='access',
                client=client,
                client_user=client_user,
                scope=auth_request['scope'],
                expires_at=timezone.now() + timedelta(seconds=client.access_token_lifetime)
            )
            
            response_params['access_token'] = access_token_value
            response_params['token_type'] = 'Bearer'
            response_params['expires_in'] = client.access_token_lifetime
        
        # Generate ID token if requested
        if 'id_token' in response_type or 'openid' in scopes:
            user_info = _get_user_info(client_user, scopes, confidence_score, liveness_verified)
            
            id_token = generate_id_token(
                client_id=client.client_id,
                user_id=str(client_user.id),
                user_info=user_info,
                nonce=nonce,
                auth_time=auth_time,
                expires_in=client.id_token_lifetime,
                access_token=response_params.get('access_token')
            )
            response_params['id_token'] = id_token
    
    # Clear session
    if 'oauth_auth_request' in request.session:
        del request.session['oauth_auth_request']
    
    # Return redirect URL
    redirect_url = f"{redirect_uri}?{urlencode(response_params)}"
    
    return Response({
        'redirect': redirect_url,
        'success': True
    })


# ---------------------------------------------------------------------------
# Token Endpoint
# ---------------------------------------------------------------------------

@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def token_endpoint(request):
    """
    OAuth 2.0 Token endpoint
    POST /oauth/token
    
    Exchanges authorization code for tokens or refreshes tokens
    """
    grant_type = request.data.get('grant_type')
    
    if grant_type == 'authorization_code':
        return _handle_authorization_code_grant(request)
    elif grant_type == 'refresh_token':
        return _handle_refresh_token_grant(request)
    elif grant_type == 'client_credentials':
        return _handle_client_credentials_grant(request)
    else:
        return Response({
            'error': 'unsupported_grant_type',
            'error_description': f'Grant type not supported: {grant_type}'
        }, status=400)


def _authenticate_client(request) -> Tuple[Optional[OAuthClient], Optional[str]]:
    """Authenticate the OAuth client from request"""
    # Try Basic auth first
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    if auth_header.startswith('Basic '):
        import base64
        try:
            decoded = base64.b64decode(auth_header[6:]).decode('utf-8')
            client_id, client_secret = decoded.split(':', 1)
        except Exception:
            return None, 'Invalid authorization header'
    else:
        # Try POST body
        client_id = request.data.get('client_id')
        client_secret = request.data.get('client_secret')
    
    if not client_id:
        return None, 'Missing client_id'
    
    try:
        client = OAuthClient.objects.get(client_id=client_id, is_active=True)
    except OAuthClient.DoesNotExist:
        return None, 'Unknown client'
    
    # Verify secret for confidential clients
    if client.client_type == 'confidential':
        if not client_secret or not client.verify_secret(client_secret):
            return None, 'Invalid client credentials'
    
    return client, None


def _handle_authorization_code_grant(request) -> Response:
    """Handle authorization code grant"""
    client, error = _authenticate_client(request)
    if error:
        return Response({
            'error': 'invalid_client',
            'error_description': error
        }, status=401)
    
    code = request.data.get('code')
    redirect_uri = request.data.get('redirect_uri')
    code_verifier = request.data.get('code_verifier')
    
    if not code:
        return Response({
            'error': 'invalid_request',
            'error_description': 'Missing authorization code'
        }, status=400)
    
    # Find authorization code
    try:
        auth_code = AuthorizationCode.objects.get(
            code=code,
            client=client,
            is_used=False
        )
    except AuthorizationCode.DoesNotExist:
        return Response({
            'error': 'invalid_grant',
            'error_description': 'Invalid or expired authorization code'
        }, status=400)
    
    # Validate expiration
    if auth_code.is_expired:
        return Response({
            'error': 'invalid_grant',
            'error_description': 'Authorization code has expired'
        }, status=400)
    
    # Validate redirect URI
    if auth_code.redirect_uri != redirect_uri:
        return Response({
            'error': 'invalid_grant',
            'error_description': 'Redirect URI mismatch'
        }, status=400)
    
    # Verify PKCE
    if auth_code.code_challenge:
        if not code_verifier:
            return Response({
                'error': 'invalid_request',
                'error_description': 'Missing code_verifier'
            }, status=400)
        
        if not auth_code.verify_pkce(code_verifier):
            return Response({
                'error': 'invalid_grant',
                'error_description': 'Invalid code_verifier'
            }, status=400)
    
    # Mark code as used
    auth_code.mark_used()
    
    # Generate tokens
    scopes = parse_scope(auth_code.scope)
    
    with transaction.atomic():
        # Access token
        access_token_value = generate_access_token(
            client_id=client.client_id,
            user_id=str(auth_code.client_user.id),
            scopes=scopes,
            expires_in=client.access_token_lifetime,
        )
        
        access_token = OAuthToken.objects.create(
            token=access_token_value[:64],
            token_type='access',
            client=client,
            client_user=auth_code.client_user,
            scope=auth_code.scope,
            expires_at=timezone.now() + timedelta(seconds=client.access_token_lifetime)
        )
        
        # Refresh token
        refresh_token_value = generate_refresh_token()
        
        OAuthToken.objects.create(
            token=refresh_token_value,
            token_type='refresh',
            client=client,
            client_user=auth_code.client_user,
            scope=auth_code.scope,
            parent_token=access_token,
            expires_at=timezone.now() + timedelta(seconds=client.refresh_token_lifetime)
        )
        
        # ID token
        user_info = _get_user_info(auth_code.client_user, scopes)
        
        id_token = generate_id_token(
            client_id=client.client_id,
            user_id=str(auth_code.client_user.id),
            user_info=user_info,
            nonce=auth_code.nonce,
            expires_in=client.id_token_lifetime,
            access_token=access_token_value
        )
    
    return Response({
        'access_token': access_token_value,
        'token_type': 'Bearer',
        'expires_in': client.access_token_lifetime,
        'refresh_token': refresh_token_value,
        'id_token': id_token,
        'scope': auth_code.scope,
    })


def _handle_refresh_token_grant(request) -> Response:
    """Handle refresh token grant"""
    client, error = _authenticate_client(request)
    if error:
        return Response({
            'error': 'invalid_client',
            'error_description': error
        }, status=401)
    
    refresh_token_value = request.data.get('refresh_token')
    
    if not refresh_token_value:
        return Response({
            'error': 'invalid_request',
            'error_description': 'Missing refresh_token'
        }, status=400)
    
    # Find refresh token
    try:
        refresh_token = OAuthToken.objects.get(
            token=refresh_token_value,
            token_type='refresh',
            client=client,
            is_revoked=False
        )
    except OAuthToken.DoesNotExist:
        return Response({
            'error': 'invalid_grant',
            'error_description': 'Invalid or revoked refresh token'
        }, status=400)
    
    if refresh_token.is_expired:
        return Response({
            'error': 'invalid_grant',
            'error_description': 'Refresh token has expired'
        }, status=400)
    
    scopes = parse_scope(refresh_token.scope)
    
    with transaction.atomic():
        # Generate new access token
        access_token_value = generate_access_token(
            client_id=client.client_id,
            user_id=str(refresh_token.client_user.id),
            scopes=scopes,
            expires_in=client.access_token_lifetime,
        )
        
        access_token = OAuthToken.objects.create(
            token=access_token_value[:64],
            token_type='access',
            client=client,
            client_user=refresh_token.client_user,
            scope=refresh_token.scope,
            expires_at=timezone.now() + timedelta(seconds=client.access_token_lifetime)
        )
        
        # Optionally rotate refresh token
        new_refresh_token_value = generate_refresh_token()
        
        OAuthToken.objects.create(
            token=new_refresh_token_value,
            token_type='refresh',
            client=client,
            client_user=refresh_token.client_user,
            scope=refresh_token.scope,
            parent_token=access_token,
            expires_at=timezone.now() + timedelta(seconds=client.refresh_token_lifetime)
        )
        
        # Revoke old refresh token
        refresh_token.revoke()
        
        # ID token
        user_info = _get_user_info(refresh_token.client_user, scopes)
        
        id_token = generate_id_token(
            client_id=client.client_id,
            user_id=str(refresh_token.client_user.id),
            user_info=user_info,
            expires_in=client.id_token_lifetime,
            access_token=access_token_value
        )
    
    return Response({
        'access_token': access_token_value,
        'token_type': 'Bearer',
        'expires_in': client.access_token_lifetime,
        'refresh_token': new_refresh_token_value,
        'id_token': id_token,
        'scope': refresh_token.scope,
    })


def _handle_client_credentials_grant(request) -> Response:
    """Handle client credentials grant (for machine-to-machine)"""
    client, error = _authenticate_client(request)
    if error:
        return Response({
            'error': 'invalid_client',
            'error_description': error
        }, status=401)
    
    if 'client_credentials' not in (client.grant_types or []):
        return Response({
            'error': 'unauthorized_client',
            'error_description': 'Client not authorized for client_credentials grant'
        }, status=400)
    
    scope = request.data.get('scope', 'openid')
    scopes = parse_scope(scope)
    
    # Validate scopes
    for s in scopes:
        if not client.is_scope_allowed(s):
            return Response({
                'error': 'invalid_scope',
                'error_description': f'Scope not allowed: {s}'
            }, status=400)
    
    # Generate access token (no user context)
    access_token_value = generate_access_token(
        client_id=client.client_id,
        user_id=client.client_id,  # Use client_id as subject
        scopes=scopes,
        expires_in=client.access_token_lifetime,
    )
    
    OAuthToken.objects.create(
        token=access_token_value[:64],
        token_type='access',
        client=client,
        client_user=None,
        scope=scope,
        expires_at=timezone.now() + timedelta(seconds=client.access_token_lifetime)
    )
    
    return Response({
        'access_token': access_token_value,
        'token_type': 'Bearer',
        'expires_in': client.access_token_lifetime,
        'scope': scope,
    })


# ---------------------------------------------------------------------------
# UserInfo Endpoint
# ---------------------------------------------------------------------------

@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def userinfo_endpoint(request):
    """
    OpenID Connect UserInfo endpoint
    GET/POST /oauth/userinfo
    
    Returns claims about the authenticated user
    """
    # Get access token from Authorization header
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    
    if not auth_header.startswith('Bearer '):
        return Response({
            'error': 'invalid_token',
            'error_description': 'Missing or invalid Authorization header'
        }, status=401)
    
    token = auth_header[7:]
    
    # Validate token
    is_valid, claims, error = validate_token(token)
    
    if not is_valid:
        return Response({
            'error': 'invalid_token',
            'error_description': error
        }, status=401)
    
    # Get user
    user_id = claims.get('sub')
    scopes = parse_scope(claims.get('scope', ''))
    
    try:
        client_user = ClientUser.objects.get(id=user_id)
    except (ClientUser.DoesNotExist, ValueError):
        return Response({
            'error': 'invalid_token',
            'error_description': 'User not found'
        }, status=401)
    
    # Build user info response
    user_info = _get_user_info(client_user, scopes)
    user_info['sub'] = str(client_user.id)
    
    return Response(user_info)


def _get_user_info(
    client_user: ClientUser,
    scopes: list,
    confidence_score: float = None,
    liveness_verified: bool = None
) -> Dict[str, Any]:
    """Build user info claims based on scopes"""
    user_info = {}
    
    if 'profile' in scopes:
        user_info['name'] = client_user.full_name or client_user.external_user_id
        if hasattr(client_user, 'first_name'):
            user_info['given_name'] = client_user.first_name or ''
        if hasattr(client_user, 'last_name'):
            user_info['family_name'] = client_user.last_name or ''
        if client_user.profile_photo:
            user_info['picture'] = client_user.profile_photo.url
        user_info['updated_at'] = int(client_user.updated_at.timestamp()) if hasattr(client_user, 'updated_at') else 0
    
    if 'email' in scopes:
        if hasattr(client_user, 'email') and client_user.email:
            user_info['email'] = client_user.email
            user_info['email_verified'] = getattr(client_user, 'email_verified', False)
    
    if 'face_auth' in scopes:
        user_info['face_auth'] = {
            'verified': True,
            'confidence': confidence_score or 0,
            'liveness_verified': liveness_verified or False,
        }
        # Check if user has active face enrollment
        from auth_service.models import FaceEnrollment
        has_enrollment = FaceEnrollment.objects.filter(
            client_user=client_user,
            status='active'
        ).exists()
        user_info['face_auth']['enrolled'] = has_enrollment
    
    return user_info


# ---------------------------------------------------------------------------
# Revocation Endpoint
# ---------------------------------------------------------------------------

@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def revoke_token(request):
    """
    OAuth 2.0 Token Revocation endpoint
    POST /oauth/revoke
    
    Revokes an access or refresh token
    """
    client, error = _authenticate_client(request)
    if error:
        return Response({
            'error': 'invalid_client',
            'error_description': error
        }, status=401)
    
    token = request.data.get('token')
    token_type_hint = request.data.get('token_type_hint')
    
    if not token:
        return Response({
            'error': 'invalid_request',
            'error_description': 'Missing token'
        }, status=400)
    
    # Find and revoke the token
    try:
        if token_type_hint == 'refresh_token':
            oauth_token = OAuthToken.objects.get(
                token=token,
                token_type='refresh',
                client=client
            )
        elif token_type_hint == 'access_token':
            oauth_token = OAuthToken.objects.get(
                token=token[:64],
                token_type='access',
                client=client
            )
        else:
            # Try both
            oauth_token = OAuthToken.objects.filter(
                client=client
            ).filter(
                models.Q(token=token) | models.Q(token=token[:64])
            ).first()
    except OAuthToken.DoesNotExist:
        oauth_token = None
    
    if oauth_token:
        oauth_token.revoke()
    
    # Return 200 OK even if token not found (per RFC 7009)
    return Response(status=200)


# ---------------------------------------------------------------------------
# Introspection Endpoint
# ---------------------------------------------------------------------------

@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def introspect_token(request):
    """
    OAuth 2.0 Token Introspection endpoint
    POST /oauth/introspect
    
    Returns information about a token
    """
    client, error = _authenticate_client(request)
    if error:
        return Response({
            'error': 'invalid_client',
            'error_description': error
        }, status=401)
    
    token = request.data.get('token')
    
    if not token:
        return Response({'active': False})
    
    # Try to decode as JWT first
    is_valid, claims, error = validate_token(token)
    
    if is_valid:
        return Response({
            'active': True,
            'client_id': claims.get('client_id'),
            'username': claims.get('sub'),
            'scope': claims.get('scope'),
            'sub': claims.get('sub'),
            'aud': claims.get('aud'),
            'iss': claims.get('iss'),
            'exp': claims.get('exp'),
            'iat': claims.get('iat'),
            'token_type': claims.get('token_type', 'access_token'),
        })
    
    # Try as opaque token (refresh token)
    try:
        oauth_token = OAuthToken.objects.get(token=token, is_revoked=False)
        if oauth_token.is_valid:
            return Response({
                'active': True,
                'client_id': oauth_token.client.client_id,
                'scope': oauth_token.scope,
                'sub': str(oauth_token.client_user.id) if oauth_token.client_user else None,
                'exp': int(oauth_token.expires_at.timestamp()),
                'iat': int(oauth_token.created_at.timestamp()),
                'token_type': oauth_token.token_type,
            })
    except OAuthToken.DoesNotExist:
        pass
    
    return Response({'active': False})


# ---------------------------------------------------------------------------
# Logout Endpoint
# ---------------------------------------------------------------------------

@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def end_session(request):
    """
    OpenID Connect End Session endpoint
    GET/POST /oauth/logout
    
    Ends the user's session
    """
    if request.method == 'GET':
        params = request.GET
    else:
        params = request.POST
    
    id_token_hint = params.get('id_token_hint')
    post_logout_redirect_uri = params.get('post_logout_redirect_uri')
    state = params.get('state')
    
    # If id_token_hint provided, verify and get user
    if id_token_hint:
        try:
            claims = decode_access_token(id_token_hint, verify=True)
            user_id = claims.get('sub')
            client_id = claims.get('aud')
            
            # Revoke all tokens for this user and client
            OAuthToken.objects.filter(
                client__client_id=client_id,
                client_user_id=user_id
            ).update(is_revoked=True)
            
        except ValueError:
            pass  # Invalid token, but continue logout
    
    # Clear session
    request.session.flush()
    
    # Redirect
    if post_logout_redirect_uri:
        redirect_url = post_logout_redirect_uri
        if state:
            redirect_url += f"?state={state}"
        return HttpResponseRedirect(redirect_url)
    
    return render(request, 'oidc/logged_out.html')


# ---------------------------------------------------------------------------
# Consent Page
# ---------------------------------------------------------------------------

def consent_page(request):
    """
    User consent page
    GET /oauth/consent
    """
    auth_request = request.session.get('oauth_auth_request')
    user_id = request.session.get('pending_auth_user_id')
    
    if not auth_request or not user_id:
        return render(request, 'oidc/error.html', {
            'error': 'No pending authorization',
            'error_description': 'Please start the login flow from the client application.'
        })
    
    try:
        client = OAuthClient.objects.get(client_id=auth_request['client_id'], is_active=True)
        client_user = ClientUser.objects.get(id=user_id)
    except (OAuthClient.DoesNotExist, ClientUser.DoesNotExist):
        return render(request, 'oidc/error.html', {
            'error': 'Invalid request'
        })
    
    scopes = parse_scope(auth_request['scope'])
    
    context = {
        'client': client,
        'user': client_user,
        'scopes': scopes,
        'scope_descriptions': {
            'openid': 'Access your user ID',
            'profile': 'Access your profile information (name, picture)',
            'email': 'Access your email address',
            'face_auth': 'Use face authentication for login',
            'offline_access': 'Access your data when you are not logged in',
        }
    }
    
    return render(request, 'oidc/consent.html', context)


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def consent_submit(request):
    """
    Handle consent form submission
    POST /oauth/consent
    """
    auth_request = request.session.get('oauth_auth_request')
    user_id = request.session.get('pending_auth_user_id')
    face_auth_result = request.session.get('face_auth_result', {})
    
    if not auth_request or not user_id:
        return Response({
            'error': 'invalid_request',
            'error_description': 'No pending authorization'
        }, status=400)
    
    consent_given = request.data.get('consent') == 'granted'
    
    if not consent_given:
        # User denied consent
        redirect_uri = auth_request['redirect_uri']
        state = auth_request.get('state', '')
        
        params = {
            'error': 'access_denied',
            'error_description': 'User denied consent'
        }
        if state:
            params['state'] = state
        
        # Clear session
        del request.session['oauth_auth_request']
        del request.session['pending_auth_user_id']
        
        return Response({
            'redirect': f"{redirect_uri}?{urlencode(params)}"
        })
    
    try:
        client = OAuthClient.objects.get(client_id=auth_request['client_id'], is_active=True)
        client_user = ClientUser.objects.get(id=user_id)
    except (OAuthClient.DoesNotExist, ClientUser.DoesNotExist):
        return Response({
            'error': 'invalid_request'
        }, status=400)
    
    # Save consent
    scopes = parse_scope(auth_request['scope'])
    
    UserConsent.objects.update_or_create(
        client=client,
        client_user=client_user,
        defaults={
            'scopes': scopes,
            'revoked_at': None,
        }
    )
    
    # Clear pending auth user
    del request.session['pending_auth_user_id']
    if 'face_auth_result' in request.session:
        del request.session['face_auth_result']
    
    # Generate authorization response
    return _generate_authorization_response(
        request, client, client_user, auth_request,
        face_auth_result.get('confidence_score', 0.9),
        face_auth_result.get('liveness_verified', True)
    )


# ---------------------------------------------------------------------------
# Demo Pages (for testing)
# ---------------------------------------------------------------------------

def demo_page(request):
    """
    Demo page for testing OIDC flow
    GET /oauth/demo
    """
    return render(request, 'oidc/demo.html')


def callback_page(request):
    """
    Callback page for receiving authorization codes
    GET /oauth/callback
    """
    return render(request, 'oidc/callback.html')

"""
OIDC Utilities for JWT signing, token generation, and cryptographic operations
"""
import jwt
import json
import secrets
import hashlib
import base64
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from django.conf import settings
from django.utils import timezone
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend


# ---------------------------------------------------------------------------
# Key Management
# ---------------------------------------------------------------------------

def get_keys_directory() -> Path:
    """Get the directory for storing OIDC keys"""
    keys_dir = Path(settings.BASE_DIR) / 'oidc_keys'
    keys_dir.mkdir(exist_ok=True)
    return keys_dir


def generate_rsa_key_pair() -> Tuple[bytes, bytes]:
    """Generate RSA key pair for JWT signing"""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_pem, public_pem


def get_or_create_signing_key() -> Tuple[bytes, bytes, str]:
    """
    Get or create the RSA signing key pair
    Returns: (private_key_pem, public_key_pem, key_id)
    """
    keys_dir = get_keys_directory()
    private_key_path = keys_dir / 'private_key.pem'
    public_key_path = keys_dir / 'public_key.pem'
    key_id_path = keys_dir / 'key_id.txt'
    
    if private_key_path.exists() and public_key_path.exists() and key_id_path.exists():
        private_pem = private_key_path.read_bytes()
        public_pem = public_key_path.read_bytes()
        key_id = key_id_path.read_text().strip()
    else:
        # Generate new key pair
        private_pem, public_pem = generate_rsa_key_pair()
        key_id = secrets.token_urlsafe(16)
        
        private_key_path.write_bytes(private_pem)
        public_key_path.write_bytes(public_pem)
        key_id_path.write_text(key_id)
        
        # Set restrictive permissions on private key
        private_key_path.chmod(0o600)
    
    return private_pem, public_pem, key_id


def get_jwks() -> Dict[str, Any]:
    """
    Generate JWKS (JSON Web Key Set) for public key exposure
    """
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
    
    _, public_pem, key_id = get_or_create_signing_key()
    
    public_key = serialization.load_pem_public_key(public_pem, backend=default_backend())
    
    if isinstance(public_key, RSAPublicKey):
        numbers = public_key.public_numbers()
        
        # Convert to base64url encoding
        def int_to_base64url(n: int, length: int) -> str:
            data = n.to_bytes(length, byteorder='big')
            return base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')
        
        # Get key size in bytes
        key_size = (public_key.key_size + 7) // 8
        
        jwk = {
            "kty": "RSA",
            "use": "sig",
            "alg": "RS256",
            "kid": key_id,
            "n": int_to_base64url(numbers.n, key_size),
            "e": int_to_base64url(numbers.e, 3),  # e is typically 65537 = 3 bytes
        }
        
        return {"keys": [jwk]}
    
    raise ValueError("Unsupported key type")


# ---------------------------------------------------------------------------
# Token Generation
# ---------------------------------------------------------------------------

def generate_access_token(
    client_id: str,
    user_id: str,
    scopes: List[str],
    expires_in: int = 3600,
    extra_claims: Optional[Dict] = None
) -> str:
    """Generate a JWT access token"""
    private_key, _, key_id = get_or_create_signing_key()
    
    now = datetime.utcnow()
    
    claims = {
        "iss": get_issuer(),
        "sub": str(user_id),
        "aud": client_id,
        "exp": now + timedelta(seconds=expires_in),
        "iat": now,
        "nbf": now,
        "jti": secrets.token_urlsafe(16),
        "scope": " ".join(scopes),
        "client_id": client_id,
        "token_type": "access_token",
    }
    
    if extra_claims:
        claims.update(extra_claims)
    
    return jwt.encode(
        claims,
        private_key,
        algorithm="RS256",
        headers={"kid": key_id}
    )


def generate_id_token(
    client_id: str,
    user_id: str,
    user_info: Dict[str, Any],
    nonce: Optional[str] = None,
    auth_time: Optional[int] = None,
    expires_in: int = 3600,
    access_token: Optional[str] = None,
) -> str:
    """Generate an OpenID Connect ID Token"""
    private_key, _, key_id = get_or_create_signing_key()
    
    now = datetime.utcnow()
    
    claims = {
        "iss": get_issuer(),
        "sub": str(user_id),
        "aud": client_id,
        "exp": now + timedelta(seconds=expires_in),
        "iat": now,
        "auth_time": auth_time or int(time.time()),
    }
    
    if nonce:
        claims["nonce"] = nonce
    
    # Add at_hash if access token is provided
    if access_token:
        claims["at_hash"] = generate_token_hash(access_token)
    
    # Add user info claims based on scopes
    if user_info:
        # Profile scope claims
        if 'name' in user_info:
            claims['name'] = user_info['name']
        if 'given_name' in user_info:
            claims['given_name'] = user_info['given_name']
        if 'family_name' in user_info:
            claims['family_name'] = user_info['family_name']
        if 'picture' in user_info:
            claims['picture'] = user_info['picture']
        
        # Email scope claims
        if 'email' in user_info:
            claims['email'] = user_info['email']
            claims['email_verified'] = user_info.get('email_verified', False)
        
        # Face auth claims
        if 'face_auth' in user_info:
            claims['face_verified'] = user_info['face_auth'].get('verified', False)
            claims['face_confidence'] = user_info['face_auth'].get('confidence', 0)
            claims['liveness_verified'] = user_info['face_auth'].get('liveness_verified', False)
    
    return jwt.encode(
        claims,
        private_key,
        algorithm="RS256",
        headers={"kid": key_id}
    )


def generate_refresh_token() -> str:
    """Generate an opaque refresh token"""
    return secrets.token_urlsafe(48)


def generate_token_hash(token: str) -> str:
    """
    Generate token hash for at_hash and c_hash claims
    Uses left half of SHA-256 hash
    """
    hash_digest = hashlib.sha256(token.encode('ascii')).digest()
    half_hash = hash_digest[:len(hash_digest) // 2]
    return base64.urlsafe_b64encode(half_hash).decode('ascii').rstrip('=')


# ---------------------------------------------------------------------------
# Token Validation
# ---------------------------------------------------------------------------

def decode_access_token(token: str, verify: bool = True) -> Dict[str, Any]:
    """Decode and verify an access token"""
    _, public_key, _ = get_or_create_signing_key()
    
    try:
        return jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_signature": verify}
        )
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e}")


def validate_token(token: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Validate an access token
    Returns: (is_valid, claims, error_message)
    """
    try:
        claims = decode_access_token(token)
        return True, claims, None
    except ValueError as e:
        return False, None, str(e)


# ---------------------------------------------------------------------------
# PKCE Utilities
# ---------------------------------------------------------------------------

def generate_code_verifier() -> str:
    """Generate a PKCE code verifier"""
    return secrets.token_urlsafe(64)[:128]


def generate_code_challenge(verifier: str, method: str = 'S256') -> str:
    """Generate a PKCE code challenge from verifier"""
    if method == 'S256':
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip('=')
    elif method == 'plain':
        return verifier
    else:
        raise ValueError(f"Unsupported code challenge method: {method}")


def verify_code_challenge(verifier: str, challenge: str, method: str = 'S256') -> bool:
    """Verify a PKCE code verifier against challenge"""
    computed_challenge = generate_code_challenge(verifier, method)
    return secrets.compare_digest(computed_challenge, challenge)


# ---------------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------------

def get_issuer() -> str:
    """Get the OIDC issuer URL"""
    return getattr(settings, 'OIDC_ISSUER', settings.SITE_URL if hasattr(settings, 'SITE_URL') else 'http://localhost:8000')


def get_oidc_config() -> Dict[str, Any]:
    """Get OpenID Connect Provider configuration (Discovery document)"""
    issuer = get_issuer()
    
    return {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/oauth/authorize",
        "token_endpoint": f"{issuer}/oauth/token",
        "userinfo_endpoint": f"{issuer}/oauth/userinfo",
        "jwks_uri": f"{issuer}/.well-known/jwks.json",
        "revocation_endpoint": f"{issuer}/oauth/revoke",
        "introspection_endpoint": f"{issuer}/oauth/introspect",
        "end_session_endpoint": f"{issuer}/oauth/logout",
        
        # Face authentication specific endpoints
        "face_auth_endpoint": f"{issuer}/oauth/face-auth",
        "face_enrollment_endpoint": f"{issuer}/api/auth/enrollment/",
        
        # Supported features
        "response_types_supported": [
            "code",
            "token",
            "id_token",
            "code token",
            "code id_token",
            "token id_token",
            "code token id_token"
        ],
        "response_modes_supported": ["query", "fragment", "form_post"],
        "grant_types_supported": [
            "authorization_code",
            "refresh_token",
            "client_credentials"
        ],
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["RS256"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_basic",
            "client_secret_post"
        ],
        "scopes_supported": [
            "openid",
            "profile",
            "email",
            "face_auth",
            "offline_access"
        ],
        "claims_supported": [
            "sub",
            "iss",
            "aud",
            "exp",
            "iat",
            "auth_time",
            "nonce",
            "at_hash",
            "name",
            "given_name",
            "family_name",
            "email",
            "email_verified",
            "picture",
            "face_verified",
            "face_confidence",
            "liveness_verified"
        ],
        "code_challenge_methods_supported": ["S256", "plain"],
        "request_parameter_supported": False,
        "request_uri_parameter_supported": False,
        
        # UI endpoints for face authentication
        "face_auth_ui": f"{issuer}/oauth/face-login",
    }


def parse_scope(scope_string: str) -> List[str]:
    """Parse space-separated scope string into list"""
    if not scope_string:
        return []
    return [s.strip() for s in scope_string.split() if s.strip()]


def build_scope_string(scopes: List[str]) -> str:
    """Build space-separated scope string from list"""
    return " ".join(scopes)

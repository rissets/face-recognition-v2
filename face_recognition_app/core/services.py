"""
Helper services to integrate with third-party client applications.
"""
from datetime import timedelta
from typing import Dict, Optional

import jwt
from django.utils import timezone


class ThirdPartyIntegrationService:
    """
    Utility wrapper that issues signed tokens and helper payloads for clients.
    """

    def __init__(self, client):
        self.client = client

    def _build_token_payload(self, expires_in: int) -> Dict:
        now = timezone.now()
        expiry = now + timedelta(seconds=expires_in)
        return {
            "client_id": self.client.client_id,
            "api_key": self.client.api_key,
            "tier": self.client.tier,
            "iat": int(now.timestamp()),
            "exp": int(expiry.timestamp()),
        }

    def generate_access_token(self, expires_in: int = 3600) -> str:
        """Return a short-lived JWT for client-to-server communication."""
        payload = self._build_token_payload(expires_in)
        return jwt.encode(payload, str(self.client.secret_key), algorithm="HS256")

    def build_auth_response(self, expires_in: int = 3600) -> Dict:
        """
        Produce a standard response payload containing credential metadata.
        """
        expiry = timezone.now() + timedelta(seconds=expires_in)
        token = self.generate_access_token(expires_in=expires_in)
        return {
            "client_id": self.client.client_id,
            "client_name": self.client.name,
            "tier": self.client.tier,
            "api_key": self.client.api_key,
            "token_type": "JWT",
            "access_token": token,
            "expires_at": expiry.isoformat(),
            "features": self.client.features,
            "rate_limits": {
                "per_hour": self.client.rate_limit_per_hour,
                "per_day": self.client.rate_limit_per_day,
            },
        }

    def build_authorization_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        """
        Construct headers that downstream services can reuse when calling this API.
        """
        access_token = token or self.generate_access_token()
        return {
            "Authorization": f"JWT {access_token}",
            "X-API-Key": self.client.api_key,
        }

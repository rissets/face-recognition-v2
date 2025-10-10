"""
Custom OpenAPI schema generator for Face Recognition API
"""
from drf_spectacular.openapi import AutoSchema
from drf_spectacular.extensions import OpenApiAuthenticationExtension
from rest_framework.permissions import AllowAny

from auth_service.authentication import (
    APIKeyAuthentication,
    JWTClientAuthentication,
    WebhookSignatureAuthentication,
)


class ApiKeyAuthenticationScheme(OpenApiAuthenticationExtension):
    target_class = 'auth_service.authentication.APIKeyAuthentication'
    name = 'ApiKeyAuth'

    def get_security_definition(self, auto_schema):
        return {
            'type': 'apiKey',
            'in': 'header',
            'name': 'X-API-Key',
            'description': 'API Key authentication via X-API-Key header',
        }


class JWTAuthenticationScheme(OpenApiAuthenticationExtension):
    target_class = 'auth_service.authentication.JWTClientAuthentication'
    name = 'BearerAuth'

    def get_security_definition(self, auto_schema):
        return {
            'type': 'http',
            'scheme': 'bearer',
            'bearerFormat': 'JWT',
            'description': 'JWT token obtained from /api/v1/auth/token/ endpoint',
        }


class WebhookAuthenticationScheme(OpenApiAuthenticationExtension):
    target_class = 'auth_service.authentication.WebhookSignatureAuthentication'
    name = 'WebhookAuth'

    def get_security_definition(self, auto_schema):
        return {
            'type': 'apiKey',
            'in': 'header',
            'name': 'X-FR-Signature',
            'description': 'Webhook signature authentication via X-FR-Signature header',
        }


class FaceRecognitionAutoSchema(AutoSchema):
    """
    Custom AutoSchema that automatically adds security requirements
    based on the view's authentication_classes
    """
    
    def get_security(self):
        """
        Determine security requirements based on authentication classes
        """
        if not hasattr(self.view, 'authentication_classes'):
            return []

        permission_classes = getattr(self.view, 'permission_classes', [])
        for permission_class in permission_classes:
            if isinstance(permission_class, type) and issubclass(permission_class, AllowAny):
                return []
        
        security = []
        
        for auth_class in self.view.authentication_classes:
            # authentication classes are provided as classes (not instances)
            if not isinstance(auth_class, type):
                continue

            if issubclass(auth_class, APIKeyAuthentication):
                security.append({'ApiKeyAuth': []})
            elif issubclass(auth_class, JWTClientAuthentication):
                security.append({'BearerAuth': []})
            elif issubclass(auth_class, WebhookSignatureAuthentication):
                security.append({'WebhookAuth': []})
        
        # Remove duplicates while preserving order
        unique_security = []
        for item in security:
            if item not in unique_security:
                unique_security.append(item)

        resolved_security = unique_security or super().get_security()

        cleaned_security = []
        for item in resolved_security:
            if not item:
                continue
            if item not in cleaned_security:
                cleaned_security.append(item)

        return cleaned_security

"""
URL Configuration for OIDC Provider

Standard OpenID Connect endpoints for integration with Keycloak and other identity brokers.
"""
from django.urls import path
from . import views

app_name = 'oidc'

urlpatterns = [
    # Discovery endpoints (must be at root level)
    # These will be mounted at /.well-known/ in the main urls.py
    
    # OAuth 2.0 / OIDC endpoints
    path('authorize', views.authorize, name='authorize'),
    path('token', views.token_endpoint, name='token'),
    path('userinfo', views.userinfo_endpoint, name='userinfo'),
    path('revoke', views.revoke_token, name='revoke'),
    path('introspect', views.introspect_token, name='introspect'),
    path('logout', views.end_session, name='logout'),
    
    # Face authentication flow
    path('face-login', views.face_login, name='face_login'),
    path('face-auth-callback', views.face_auth_callback, name='face_auth_callback'),
    
    # Consent
    path('consent', views.consent_page, name='consent'),
    path('consent/submit', views.consent_submit, name='consent_submit'),
    
    # Demo pages for testing
    path('demo', views.demo_page, name='demo'),
    path('callback', views.callback_page, name='callback'),
]

# Discovery endpoints (to be included separately)
discovery_urlpatterns = [
    path('openid-configuration', views.openid_configuration, name='openid_configuration'),
    path('jwks.json', views.jwks_endpoint, name='jwks'),
]

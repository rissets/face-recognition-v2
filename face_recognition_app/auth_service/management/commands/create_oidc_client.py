"""
Management command to create sample OAuth clients for Keycloak integration
"""
from django.core.management.base import BaseCommand
from auth_service.oidc.models import OAuthClient
from clients.models import Client


class Command(BaseCommand):
    help = 'Create sample OAuth clients for Keycloak integration testing'

    def add_arguments(self, parser):
        parser.add_argument(
            '--keycloak-url',
            type=str,
            default='http://localhost:8080',
            help='Keycloak base URL'
        )
        parser.add_argument(
            '--realm',
            type=str,
            default='master',
            help='Keycloak realm name'
        )
        parser.add_argument(
            '--name',
            type=str,
            default='Keycloak Development',
            help='OAuth client name'
        )

    def handle(self, *args, **options):
        keycloak_url = options['keycloak_url'].rstrip('/')
        realm = options['realm']
        name = options['name']
        
        # Construct Keycloak redirect URIs
        redirect_uris = [
            f"{keycloak_url}/realms/{realm}/broker/face-recognition/endpoint",
            f"{keycloak_url}/realms/{realm}/broker/face-recognition/endpoint/*",
            f"{keycloak_url}/auth/realms/{realm}/broker/face-recognition/endpoint",
            # For local testing
            "http://localhost:3000/callback",
            "http://localhost:8000/oauth/callback",
            "http://127.0.0.1:8000/oauth/callback",
        ]
        
        # Check if client already exists
        existing = OAuthClient.objects.filter(name=name).first()
        if existing:
            self.stdout.write(
                self.style.WARNING(f'OAuth client "{name}" already exists')
            )
            self._print_client_info(existing)
            return
        
        # Get or create API client
        api_client = Client.objects.first()
        
        # Create OAuth client
        oauth_client = OAuthClient.objects.create(
            name=name,
            description=f'OAuth client for Keycloak realm: {realm}',
            client_type='confidential',
            redirect_uris='\n'.join(redirect_uris),
            grant_types=['authorization_code', 'refresh_token'],
            response_types=['code'],
            allowed_scopes=['openid', 'profile', 'email', 'face_auth', 'offline_access'],
            access_token_lifetime=3600,
            refresh_token_lifetime=86400,
            id_token_lifetime=3600,
            require_pkce=True,
            require_consent=True,
            require_face_auth=True,
            require_liveness=True,
            min_confidence_score=0.85,
            api_client=api_client,
            is_active=True,
        )
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully created OAuth client: {name}')
        )
        self._print_client_info(oauth_client)
        
        # Print Keycloak configuration
        self._print_keycloak_config(oauth_client, keycloak_url, realm)

    def _print_client_info(self, client):
        self.stdout.write('')
        self.stdout.write('=' * 60)
        self.stdout.write(self.style.SUCCESS('OAuth Client Credentials'))
        self.stdout.write('=' * 60)
        self.stdout.write(f'Client ID:     {client.client_id}')
        self.stdout.write(f'Client Secret: {client.client_secret}')
        self.stdout.write(f'Redirect URIs:')
        for uri in client.get_redirect_uris_list():
            self.stdout.write(f'  - {uri}')
        self.stdout.write('=' * 60)
        self.stdout.write('')

    def _print_keycloak_config(self, client, keycloak_url, realm):
        from django.conf import settings
        
        issuer = getattr(settings, 'OIDC_ISSUER', 'https://face.ahu.go.id')
        
        self.stdout.write('')
        self.stdout.write('=' * 60)
        self.stdout.write(self.style.SUCCESS('Keycloak Identity Provider Configuration'))
        self.stdout.write('=' * 60)
        self.stdout.write('')
        self.stdout.write('1. Go to Keycloak Admin Console')
        self.stdout.write(f'   URL: {keycloak_url}/admin/')
        self.stdout.write('')
        self.stdout.write(f'2. Select Realm: {realm}')
        self.stdout.write('')
        self.stdout.write('3. Navigate to: Identity Providers → Add provider → OpenID Connect v1.0')
        self.stdout.write('')
        self.stdout.write('4. Fill in the form:')
        self.stdout.write('')
        self.stdout.write('   [General Settings]')
        self.stdout.write('   Alias:        face-recognition')
        self.stdout.write('   Display Name: Face Recognition Login')
        self.stdout.write('   Enabled:      ON')
        self.stdout.write('')
        self.stdout.write('   [OpenID Connect Config]')
        self.stdout.write(f'   Discovery URL: {issuer}/.well-known/openid-configuration')
        self.stdout.write('')
        self.stdout.write('   OR manually configure:')
        self.stdout.write(f'   Authorization URL: {issuer}/oauth/authorize')
        self.stdout.write(f'   Token URL:         {issuer}/oauth/token')
        self.stdout.write(f'   Logout URL:        {issuer}/oauth/logout')
        self.stdout.write(f'   User Info URL:     {issuer}/oauth/userinfo')
        self.stdout.write(f'   JWKS URL:          {issuer}/.well-known/jwks.json')
        self.stdout.write('')
        self.stdout.write('   [Client Authentication]')
        self.stdout.write(f'   Client ID:     {client.client_id}')
        self.stdout.write(f'   Client Secret: {client.client_secret}')
        self.stdout.write('   Client Auth:   Client secret sent as post')
        self.stdout.write('')
        self.stdout.write('   [Advanced Settings]')
        self.stdout.write('   Default Scopes: openid profile email face_auth')
        self.stdout.write('   PKCE Method:    S256')
        self.stdout.write('   Validate Signatures: ON')
        self.stdout.write('   Use JWKS URL:   ON')
        self.stdout.write('')
        self.stdout.write('5. Click Save')
        self.stdout.write('')
        self.stdout.write('=' * 60)

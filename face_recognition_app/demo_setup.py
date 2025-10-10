"""
Demo setup script for third-party face authentication service
Create        # Save credentials
        credentials = {
            'client_id': str(client.client_id),
            'api_key': client.api_key,
        }ient and users to test the new multi-client architecture
"""
import os
import sys
import django
from django.conf import settings

# Add the project directory to Python path
sys.path.append('/Users/user/Dev/researchs/face_regocnition_v2/face_recognition_app')

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_app.settings')
django.setup()

from clients.models import Client, ClientUser
from webhooks.models import WebhookEndpoint
from auth_service.models import AuthenticationSession
import secrets
from django.utils import timezone
from datetime import timedelta
import json

def create_demo_client():
    """Create a demo client for testing"""
    print("🚀 Creating demo client...")
    
    # Create demo client
    client, created = Client.objects.get_or_create(
        name="Demo Company",
        defaults={
            'description': 'Demo client for testing third-party face authentication service',
            'domain': 'https://demo-company.com',
            'contact_email': 'admin@demo-company.com',
            'contact_name': 'Demo Admin',
            'webhook_url': 'https://demo-company.com/webhooks/face-auth',
            'tier': 'premium',
            'status': 'active',
            'rate_limit_per_hour': 1000,
            'rate_limit_per_day': 10000,
            'allowed_domains': ['demo-company.com', 'localhost', '127.0.0.1'],
            'features': {
                'face_enrollment': True,
                'face_authentication': True, 
                'liveness_detection': True,
                'user_management': True,
                'analytics_access': True,
                'webhook_events': [
                    'enrollment.started',
                    'enrollment.completed', 
                    'enrollment.failed',
                    'authentication.started',
                    'authentication.success',
                    'authentication.failed'
                ]
            },
            'metadata': {
                'demo': True,
                'created_by': 'demo_setup_script',
                'version': '2.0'
            }
        }
    )
    
    if created:
        print(f"✅ Demo client created successfully!")
        print(f"   - Client Name: {client.name}")
        print(f"   - Client ID: {client.client_id}")
        print(f"   - API Key: {client.api_key}")
        print(f"   - Tier: {client.tier}")
        
        # Save credentials to file
        credentials = {
            'client_name': client.name,
            'client_id': client.client_id,
            'api_key': client.api_key,
            'secret_key': client.secret_key,
            'webhook_url': client.webhook_url,
            'webhook_secret': client.webhook_secret
        }
        
        with open('demo_credentials.json', 'w') as f:
            json.dump(credentials, f, indent=2)
        print(f"   - Credentials saved to: demo_credentials.json")
    else:
        print(f"✅ Demo client already exists: {client.name}")
        
        # Load existing credentials
        try:
            with open('demo_credentials.json', 'r') as f:
                credentials = json.load(f)
            print(f"   - Client ID: {credentials['client_id']}")
            print(f"   - API Key: {credentials['api_key']}")
        except FileNotFoundError:
            print(f"   - Client ID: {client.client_id}")
            print(f"   - API Key: {client.api_key}")
            print("   - Warning: demo_credentials.json not found")
    
    return client

def create_demo_users(client):
    """Create demo users for the client"""
    print(f"\n👥 Creating demo users for {client.name}...")
    
    demo_users_data = [
        {
            'external_user_id': 'john.doe',
            'profile': {
                'name': 'John Doe',
                'email': 'john.doe@demo-company.com',
                'first_name': 'John',
                'last_name': 'Doe',
                'phone_number': '+1234567890',
                'department': 'Engineering'
            },
            'metadata': {
                'source': 'demo_setup',
                'is_verified': True,
                'verification_method': 'email'
            }
        },
        {
            'external_user_id': 'jane.smith',
            'profile': {
                'name': 'Jane Smith',
                'email': 'jane.smith@demo-company.com',
                'first_name': 'Jane',
                'last_name': 'Smith',
                'phone_number': '+1234567891',
                'department': 'Marketing'
            },
            'metadata': {
                'source': 'demo_setup',
                'is_verified': True,
                'verification_method': 'email'
            }
        },
        {
            'external_user_id': 'mike.johnson',
            'profile': {
                'name': 'Mike Johnson',
                'email': 'mike.johnson@demo-company.com',
                'first_name': 'Mike',
                'last_name': 'Johnson',
                'phone_number': '+1234567892',
                'department': 'Sales'
            },
            'metadata': {
                'source': 'demo_setup',
                'is_verified': False,
                'verification_method': 'pending'
            }
        }
    ]
    
    created_users = []
    for user_data in demo_users_data:
        user, created = ClientUser.objects.get_or_create(
            client=client,
            external_user_id=user_data['external_user_id'],
            defaults=user_data
        )
        
        if created:
            print(f"   ✅ Created user: {user.external_user_id} ({user.profile.get('name', 'Unknown')})")
        else:
            print(f"   ✅ User already exists: {user.external_user_id}")
            
        created_users.append(user)
    
    return created_users

def create_webhook_endpoints(client):
    """Create webhook endpoints for the client"""
    print(f"\n🔗 Setting up webhook endpoints for {client.name}...")
    
    endpoints_data = [
        {
            'name': 'enrollment_endpoint',
            'url': f'{client.webhook_url}/enrollment',
            'subscribed_events': ['enrollment.started', 'enrollment.completed', 'enrollment.failed'],
            'status': 'active',
            'metadata': {'description': 'Face enrollment events'}
        },
        {
            'name': 'authentication_endpoint',
            'url': f'{client.webhook_url}/authentication', 
            'subscribed_events': ['authentication.started', 'authentication.success', 'authentication.failed'],
            'status': 'active',
            'metadata': {'description': 'Face authentication events'}
        },
        {
            'name': 'security_endpoint',
            'url': f'{client.webhook_url}/security',
            'subscribed_events': ['security.suspicious_activity', 'security.multiple_faces', 'security.liveness_failed'],
            'status': 'active',
            'metadata': {'description': 'Security events'}
        }
    ]
    
    for endpoint_data in endpoints_data:
        endpoint, created = WebhookEndpoint.objects.get_or_create(
            client=client,
            name=endpoint_data['name'],
            defaults={
                'url': endpoint_data['url'],
                'subscribed_events': endpoint_data['subscribed_events'],
                'status': endpoint_data['status'],
                'secret_token': client.webhook_secret,
                'max_retries': 3,
                'retry_delay_seconds': 60,
                'metadata': endpoint_data['metadata']
            }
        )
        
        if created:
            print(f"   ✅ Created webhook: {endpoint.name} - {endpoint.url}")
            print(f"      Events: {', '.join(endpoint.subscribed_events)}")
        else:
            print(f"   ✅ Webhook already exists: {endpoint.name} - {endpoint.url}")

def print_api_examples(client):
    """Print API usage examples"""
    print(f"\n📖 API Usage Examples for {client.name}")
    print("=" * 60)
    
    with open('demo_credentials.json', 'r') as f:
        credentials = json.load(f)
    
    print(f"""
🔐 Authentication:
POST /api/core/auth/client/
{{
  "api_key": "{credentials['api_key']}",
  "secret_key": "{credentials['secret_key']}"
}}

👤 Create User:
POST /api/clients/users/
Headers: Authorization: ApiKey {credentials['api_key']}:SECRET_KEY
{{
  "username": "new.user",
  "email": "new.user@demo-company.com",
  "first_name": "New",
  "last_name": "User"
}}

📝 Create Enrollment Session:
POST /api/auth/enrollment/create/
Headers: Authorization: ApiKey {credentials['api_key']}:SECRET_KEY
{{
  "user_id": "john.doe",
  "session_type": "webcam",
  "metadata": {{"source": "demo"}}
}}

🔍 Create Authentication Session:
POST /api/auth/authentication/create/
Headers: Authorization: ApiKey {credentials['api_key']}:SECRET_KEY
{{
  "session_type": "webcam", 
  "require_liveness": true
}}

📊 Get Analytics:
GET /api/auth/analytics/
Headers: Authorization: ApiKey {credentials['api_key']}:SECRET_KEY

🔄 Webhook Endpoints:
- Enrollment: {credentials['webhook_url']}/enrollment
- Authentication: {credentials['webhook_url']}/authentication  
- Security: {credentials['webhook_url']}/security

🌐 API Documentation:
- Swagger UI: http://127.0.0.1:8000/api/docs/
- ReDoc: http://127.0.0.1:8000/api/redoc/
- Schema: http://127.0.0.1:8000/api/schema/
""")

def print_admin_info():
    """Print admin interface information"""
    print(f"\n🔧 Admin Interface")
    print("=" * 30)
    print(f"""
Django Unfold Admin: http://127.0.0.1:8000/admin/

Available Models:
- Clients (Client management)
- Client Users (User management per client)
- Authentication Sessions (Face auth sessions)
- Face Enrollments (Enrollment records)
- Webhook Endpoints (Webhook configuration)
- Webhook Deliveries (Delivery logs)
- Audit Logs (System activities)
- Security Events (Security incidents)
- System Configuration (System settings)

Note: Create a superuser with: python manage.py createsuperuser
""")

def main():
    """Main setup function"""
    print("🎯 Third-Party Face Authentication Service - Demo Setup")
    print("=" * 60)
    
    try:
        # Create demo client
        client = create_demo_client()
        
        # Create demo users
        users = create_demo_users(client)
        
        # Create webhook endpoints 
        create_webhook_endpoints(client)
        
        # Print usage examples
        print_api_examples(client)
        
        # Print admin info
        print_admin_info()
        
        print(f"\n🎉 Demo setup completed successfully!")
        print(f"   - Client: {client.name}")
        print(f"   - Users: {len(users)} created")
        print(f"   - Webhooks: Configured")
        print(f"   - Credentials: Saved to demo_credentials.json")
        print(f"\n🚀 Your third-party face authentication service is ready!")
        print(f"   - Server: http://127.0.0.1:8000/")
        print(f"   - API Docs: http://127.0.0.1:8000/api/docs/")
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

"""
Demo setup script for the third-party face authentication service.
Creates a demo client, sample users, webhook endpoints, and seed analytics data.
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

from clients.models import Client, ClientUser, ClientAPIUsage, ClientWebhookLog
from webhooks.models import WebhookEndpoint
from auth_service.models import AuthenticationSession, FaceEnrollment, FaceRecognitionAttempt
from analytics.models import AuthenticationLog, SecurityAlert, SystemMetrics
from analytics.helpers import (
    track_enrollment_metrics,
    track_authentication_metrics,
    update_face_recognition_stats,
)
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


def create_sample_activity(client, users):
    """Generate sample analytics, usage, and webhook data for demos."""
    if client.recognition_attempts.exists():
        print("\n📊 Sample analytics already exist. Skipping generation.")
        return

    if not users:
        print("\n⚠️ Cannot generate sample activity without client users.")
        return

    primary_user = users[0]
    now = timezone.now()

    print("\n📈 Generating sample analytics data...")

    # Enrollment session & record
    enrollment_session = AuthenticationSession.objects.create(
        client=client,
        client_user=primary_user,
        session_type="enrollment",
        status="completed",
        ip_address="127.0.0.1",
        user_agent="demo-setup/1.0",
        metadata={
            "target_samples": 5,
            "frames_processed": 5,
            "session_origin": "demo_setup",
        },
    )

    enrollment = FaceEnrollment.objects.create(
        client=client,
        client_user=primary_user,
        enrollment_session=enrollment_session,
        status="active",
        embedding_vector="[]",
        embedding_dimension=512,
        face_quality_score=0.92,
        liveness_score=0.88,
        anti_spoofing_score=0.91,
        face_landmarks={},
        face_bbox=[0.1, 0.1, 0.8, 0.8],
        sample_number=1,
        total_samples=5,
        face_image_path="enrollments/demo/sample.jpg",
        metadata={"generated_by": "demo_setup"},
        expires_at=now + timedelta(days=90),
    )
    track_enrollment_metrics(client, enrollment, enrollment_session)

    # Authentication session & attempt
    auth_session = AuthenticationSession.objects.create(
        client=client,
        client_user=primary_user,
        session_type="verification",
        status="completed",
        ip_address="127.0.0.1",
        user_agent="demo-setup/1.0",
        is_successful=True,
        confidence_score=0.94,
        metadata={
            "frames_processed": 3,
            "target_user_id": primary_user.external_user_id,
            "session_origin": "demo_setup",
        },
    )

    attempt = FaceRecognitionAttempt.objects.create(
        client=client,
        session=auth_session,
        result="success",
        matched_user=primary_user,
        matched_enrollment=enrollment,
        similarity_score=0.94,
        confidence_score=0.94,
        face_quality_score=0.9,
        liveness_score=0.87,
        anti_spoofing_score=0.9,
        submitted_embedding="[]",
        face_landmarks={},
        face_bbox=[0.12, 0.12, 0.82, 0.82],
        processing_time_ms=125,
        ip_address="127.0.0.1",
        user_agent="demo-setup/1.0",
        metadata={"source": "demo_setup"},
    )

    update_face_recognition_stats(client, attempt)
    track_authentication_metrics(client, auth_session, success=True, similarity_score=0.94)

    AuthenticationLog.objects.create(
        client=client,
        attempted_email=primary_user.profile.get("email"),
        auth_method="face",
        success=True,
        similarity_score=0.94,
        liveness_score=0.87,
        quality_score=0.9,
        response_time=125,
        ip_address="127.0.0.1",
        user_agent="demo-setup/1.0",
        risk_score=0.06,
        risk_factors=[],
        session_id=auth_session.session_token,
    )

    SecurityAlert.objects.create(
        client=client,
        alert_type="new_device",
        severity="low",
        title="New trusted device detected",
        description="Demo setup registered a trusted device for analytics preview.",
        context_data={
            "device": "demo-setup/1.0",
            "session_token": auth_session.session_token,
        },
        ip_address="127.0.0.1",
    )

    SystemMetrics.objects.create(
        client=client,
        metric_name="system.cpu_usage",
        metric_type="gauge",
        value=32.5,
        unit="percent",
        tags={"source": "demo_setup"},
    )

    ClientAPIUsage.objects.bulk_create(
        [
            ClientAPIUsage(
                client=client,
                endpoint="enrollment",
                method="POST",
                status_code=201,
                ip_address="127.0.0.1",
                user_agent="demo-setup/1.0",
                response_time_ms=342,
                metadata={"source": "demo_setup"},
            ),
            ClientAPIUsage(
                client=client,
                endpoint="recognition",
                method="POST",
                status_code=200,
                ip_address="127.0.0.1",
                user_agent="demo-setup/1.0",
                response_time_ms=128,
                metadata={"source": "demo_setup"},
            ),
            ClientAPIUsage(
                client=client,
                endpoint="analytics",
                method="GET",
                status_code=200,
                ip_address="127.0.0.1",
                user_agent="demo-setup/1.0",
                response_time_ms=64,
                metadata={"source": "demo_setup"},
            ),
        ]
    )

    ClientWebhookLog.objects.create(
        client=client,
        event_type="enrollment.completed",
        payload={
            "user_id": primary_user.external_user_id,
            "enrollment_id": str(enrollment.id),
            "demo": True,
        },
        status="success",
        response_status_code=200,
        response_body="OK",
        attempt_count=1,
        max_attempts=3,
        delivered_at=now,
    )

    print("   ✅ Sample analytics, usage, and webhook logs created.")

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

        # Generate sample analytics/activity data
        create_sample_activity(client, users)
        
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

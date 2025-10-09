from rest_framework.test import APIRequestFactory
from core.views import PublicWebRTCAuthenticationCreateView

factory = APIRequestFactory()
payload = {
    "session_type": "identification",
    "email": None,
    "device_info": {
        "device_id": "face-login-inline",
        "device_name": "Inline Face Login",
        "device_type": "web"
    }
}
request = factory.post('/api/auth/face/webrtc/public/create/', payload, format='json')
response = PublicWebRTCAuthenticationCreateView.as_view()(request)
print("status", response.status_code)
print(response.data)

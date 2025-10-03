"""
Comprehensive test suite for face recognition application
"""
import base64
import json
import tempfile
from io import BytesIO
from unittest.mock import patch, MagicMock

from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test.utils import override_settings
from channels.testing import WebsocketCommunicator
from channels.db import database_sync_to_async
from rest_framework.test import APITestCase
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from PIL import Image
import numpy as np

from core.models import EnrollmentSession, FaceEmbedding, AuthenticationAttempt
from users.models import CustomUser, UserProfile, UserDevice
from recognition.models import FaceTemplate
from analytics.models import UserActivity, SystemMetrics, PerformanceLog, ErrorLog
from streaming.models import WebRTCSession, StreamingSession
from core.face_recognition_engine import FaceRecognitionEngine

User = get_user_model()


class UserRegistrationTestCase(APITestCase):
    """Test user registration process"""
    
    def setUp(self):
        self.registration_url = reverse('user-register')
        self.valid_user_data = {
            'email': 'test@example.com',
            'password': 'securepassword123',
            'password_confirm': 'securepassword123',
            'first_name': 'Test',
            'last_name': 'User'
        }
    
    def test_successful_registration(self):
        """Test successful user registration"""
        response = self.client.post(self.registration_url, self.valid_user_data)
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(User.objects.filter(email='test@example.com').exists())
        
        user = User.objects.get(email='test@example.com')
        self.assertEqual(user.first_name, 'Test')
        self.assertEqual(user.last_name, 'User')
        self.assertFalse(user.face_enrolled)
        self.assertTrue(user.is_active)
    
    def test_registration_with_existing_email(self):
        """Test registration with already existing email"""
        # Create user first
        User.objects.create_user(
            email='test@example.com',
            password='password123'
        )
        
        response = self.client.post(self.registration_url, self.valid_user_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('email', response.data)
    
    def test_registration_password_mismatch(self):
        """Test registration with password mismatch"""
        invalid_data = self.valid_user_data.copy()
        invalid_data['password_confirm'] = 'differentpassword'
        
        response = self.client.post(self.registration_url, invalid_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_registration_weak_password(self):
        """Test registration with weak password"""
        invalid_data = self.valid_user_data.copy()
        invalid_data['password'] = '123'
        invalid_data['password_confirm'] = '123'
        
        response = self.client.post(self.registration_url, invalid_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class UserLoginTestCase(APITestCase):
    """Test user login process"""
    
    def setUp(self):
        self.login_url = reverse('user-login')
        self.user = User.objects.create_user(
            email='test@example.com',
            password='securepassword123',
            first_name='Test',
            last_name='User'
        )
    
    def test_successful_login(self):
        """Test successful login with email and password"""
        login_data = {
            'email': 'test@example.com',
            'password': 'securepassword123'
        }
        
        response = self.client.post(self.login_url, login_data)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)
        self.assertIn('user', response.data)
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        login_data = {
            'email': 'test@example.com',
            'password': 'wrongpassword'
        }
        
        response = self.client.post(self.login_url, login_data)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_login_inactive_user(self):
        """Test login with inactive user"""
        self.user.is_active = False
        self.user.save()
        
        login_data = {
            'email': 'test@example.com',
            'password': 'securepassword123'
        }
        
        response = self.client.post(self.login_url, login_data)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class FaceEnrollmentTestCase(APITestCase):
    """Test face enrollment process"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='securepassword123'
        )
        self.client.force_authenticate(user=self.user)
        
        self.start_enrollment_url = reverse('start-enrollment')
        self.process_frame_url = reverse('process-enrollment-frame')
        self.complete_enrollment_url = reverse('complete-enrollment')
    
    def create_test_image(self):
        """Create a test image for enrollment"""
        image = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def test_start_enrollment_session(self):
        """Test starting an enrollment session"""
        response = self.client.post(self.start_enrollment_url)
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('session_id', response.data)
        
        # Check database
        session = EnrollmentSession.objects.get(user=self.user)
        self.assertEqual(session.status, 'in_progress')
        self.assertEqual(session.frames_captured, 0)
    
    def test_start_enrollment_already_enrolled(self):
        """Test starting enrollment when user is already enrolled"""
        self.user.face_enrolled = True
        self.user.save()
        
        response = self.client.post(self.start_enrollment_url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    @patch('core.face_recognition_engine.FaceRecognitionEngine.process_enrollment_frame')
    def test_process_enrollment_frame(self, mock_process):
        """Test processing an enrollment frame"""
        # Start enrollment session
        self.client.post(self.start_enrollment_url)
        session = EnrollmentSession.objects.get(user=self.user)
        
        # Mock successful processing
        mock_process.return_value = {
            'success': True,
            'quality_score': 0.85,
            'frame_count': 1,
            'total_required': 10
        }
        
        frame_data = self.create_test_image()
        response = self.client.post(self.process_frame_url, {
            'session_id': str(session.session_id),
            'frame_data': frame_data
        })
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['quality_score'], 0.85)
        
        # Check session updated
        session.refresh_from_db()
        self.assertEqual(session.frames_captured, 1)
    
    @patch('core.face_recognition_engine.FaceRecognitionEngine.complete_enrollment')
    def test_complete_enrollment(self, mock_complete):
        """Test completing an enrollment session"""
        # Start enrollment and simulate processing
        self.client.post(self.start_enrollment_url)
        session = EnrollmentSession.objects.get(user=self.user)
        session.frames_captured = 10
        session.save()
        
        # Mock successful completion
        mock_complete.return_value = {
            'success': True,
            'embedding_id': 'test-embedding-id',
            'quality_score': 0.92
        }
        
        response = self.client.post(self.complete_enrollment_url, {
            'session_id': str(session.session_id)
        })
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        
        # Check user updated
        self.user.refresh_from_db()
        self.assertTrue(self.user.face_enrolled)
        
        # Check session completed
        session.refresh_from_db()
        self.assertEqual(session.status, 'completed')


class FaceAuthenticationTestCase(APITestCase):
    """Test face authentication process"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='securepassword123'
        )
        self.user.face_enrolled = True
        self.user.save()
        
        # Create face embedding
        FaceEmbedding.objects.create(
            user=self.user,
            embedding_data={'embedding': [0.1] * 512},
            quality_score=0.9,
            confidence_score=0.85
        )
        
        self.authenticate_url = reverse('authenticate-frame')
    
    def create_test_image(self):
        """Create a test image for authentication"""
        image = Image.new('RGB', (640, 480), color='blue')
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
    
    @patch('core.face_recognition_engine.FaceRecognitionEngine.authenticate_face')
    def test_successful_authentication(self, mock_auth):
        """Test successful face authentication"""
        mock_auth.return_value = {
            'success': True,
            'confidence': 0.92,
            'user_id': str(self.user.id),
            'liveness_passed': True,
            'obstacle_detected': False
        }
        
        frame_data = self.create_test_image()
        response = self.client.post(self.authenticate_url, {
            'frame_data': frame_data
        })
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        self.assertEqual(response.data['confidence'], 0.92)
        
        # Check authentication attempt logged
        attempt = AuthenticationAttempt.objects.get(user=self.user)
        self.assertEqual(attempt.status, 'success')
        self.assertTrue(attempt.liveness_passed)
    
    @patch('core.face_recognition_engine.FaceRecognitionEngine.authenticate_face')
    def test_failed_authentication(self, mock_auth):
        """Test failed face authentication"""
        mock_auth.return_value = {
            'success': False,
            'confidence': 0.3,
            'user_id': None,
            'liveness_passed': True,
            'obstacle_detected': False,
            'error': 'No matching face found'
        }
        
        frame_data = self.create_test_image()
        response = self.client.post(self.authenticate_url, {
            'frame_data': frame_data
        })
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertFalse(response.data['success'])
    
    @patch('core.face_recognition_engine.FaceRecognitionEngine.authenticate_face')
    def test_liveness_detection_failed(self, mock_auth):
        """Test authentication with failed liveness detection"""
        mock_auth.return_value = {
            'success': False,
            'confidence': 0.95,
            'user_id': str(self.user.id),
            'liveness_passed': False,
            'obstacle_detected': False,
            'error': 'Liveness detection failed'
        }
        
        frame_data = self.create_test_image()
        response = self.client.post(self.authenticate_url, {
            'frame_data': frame_data
        })
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(response.data['error'], 'Liveness detection failed')


class WebSocketTestCase(TransactionTestCase):
    """Test WebSocket functionality"""
    
    async def test_face_recognition_websocket_connection(self):
        """Test WebSocket connection for face recognition"""
        from streaming.consumers import FaceRecognitionConsumer
        
        # Create user
        user = await database_sync_to_async(User.objects.create_user)(
            email='test@example.com',
            password='securepassword123'
        )
        
        # Create JWT token
        refresh = RefreshToken.for_user(user)
        access_token = str(refresh.access_token)
        
        communicator = WebsocketCommunicator(
            FaceRecognitionConsumer.as_asgi(),
            f"/ws/face-recognition/?token={access_token}"
        )
        
        connected, subprotocol = await communicator.connect()
        self.assertTrue(connected)
        
        # Test sending a frame
        await communicator.send_json_to({
            'type': 'process_frame',
            'frame_data': 'base64_encoded_frame_data',
            'session_type': 'enrollment'
        })
        
        # Test receiving response
        response = await communicator.receive_json_from()
        self.assertIn('type', response)
        
        await communicator.disconnect()


class FaceRecognitionEngineTestCase(TestCase):
    """Test face recognition engine functionality"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='securepassword123'
        )
    
    @patch('core.face_recognition_engine.ChromaEmbeddingStore')
    @patch('core.face_recognition_engine.insightface.app.FaceAnalysis')
    def test_engine_initialization(self, mock_face_analysis, mock_chroma):
        """Test face recognition engine initialization"""
        mock_face_analysis.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        
        engine = FaceRecognitionEngine()
        self.assertIsNotNone(engine.face_app)
        self.assertIsNotNone(engine.embedding_store)
    
    def create_test_frame(self):
        """Create test frame data"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @patch('core.face_recognition_engine.FaceRecognitionEngine._detect_faces')
    @patch('core.face_recognition_engine.FaceRecognitionEngine._check_liveness')
    @patch('core.face_recognition_engine.FaceRecognitionEngine._detect_obstacles')
    def test_process_enrollment_frame(self, mock_obstacles, mock_liveness, mock_detect):
        """Test processing enrollment frame"""
        engine = FaceRecognitionEngine()
        
        # Mock successful detection
        mock_detect.return_value = [{'embedding': np.random.rand(512)}]
        mock_liveness.return_value = True
        mock_obstacles.return_value = False
        
        frame = self.create_test_frame()
        result = engine.process_enrollment_frame(self.user.id, frame)
        
        self.assertTrue(result['success'])
        self.assertIn('quality_score', result)


class AnalyticsTestCase(APITestCase):
    """Test analytics functionality"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='securepassword123'
        )
        self.client.force_authenticate(user=self.user)
    
    def test_user_activity_logging(self):
        """Test user activity is logged correctly"""
        # Perform some action that should log activity
        self.client.post(reverse('user-login'), {
            'email': 'test@example.com',
            'password': 'securepassword123'
        })
        
        # Check activity was logged
        activity = UserActivity.objects.filter(
            user=self.user,
            activity_type='login'
        ).first()
        
        self.assertIsNotNone(activity)
        self.assertTrue(activity.success)
    
    def test_performance_logging(self):
        """Test performance metrics are logged"""
        # Create performance log entry
        PerformanceLog.objects.create(
            operation='face_authentication',
            duration_ms=150,
            success=True,
            user=self.user
        )
        
        log = PerformanceLog.objects.get(operation='face_authentication')
        self.assertEqual(log.duration_ms, 150)
        self.assertTrue(log.success)
    
    def test_system_metrics_collection(self):
        """Test system metrics collection"""
        SystemMetrics.objects.create(
            metric_name='cpu_usage',
            metric_type='gauge',
            metric_value=75.5,
            metric_unit='percent'
        )
        
        metric = SystemMetrics.objects.get(metric_name='cpu_usage')
        self.assertEqual(metric.metric_value, 75.5)
        self.assertEqual(metric.metric_type, 'gauge')


class SecurityTestCase(APITestCase):
    """Test security features"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='securepassword123'
        )
    
    def test_rate_limiting(self):
        """Test API rate limiting"""
        login_url = reverse('user-login')
        
        # Make multiple rapid requests
        for i in range(10):
            response = self.client.post(login_url, {
                'email': 'test@example.com',
                'password': 'wrongpassword'
            })
        
        # Should eventually hit rate limit
        self.assertEqual(response.status_code, status.HTTP_429_TOO_MANY_REQUESTS)
    
    def test_jwt_token_authentication(self):
        """Test JWT token authentication"""
        # Get token
        refresh = RefreshToken.for_user(self.user)
        access_token = str(refresh.access_token)
        
        # Test authenticated request
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {access_token}')
        response = self.client.get(reverse('user-profile'))
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_unauthorized_access(self):
        """Test unauthorized access to protected endpoints"""
        response = self.client.get(reverse('user-profile'))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class AdminInterfaceTestCase(TestCase):
    """Test admin interface functionality"""
    
    def setUp(self):
        self.admin_user = User.objects.create_superuser(
            email='admin@example.com',
            password='adminpassword123'
        )
        self.client.force_login(self.admin_user)
    
    def test_admin_dashboard_access(self):
        """Test admin dashboard is accessible"""
        response = self.client.get('/admin/')
        self.assertEqual(response.status_code, 200)
    
    def test_user_admin_interface(self):
        """Test user admin interface"""
        response = self.client.get('/admin/users/customuser/')
        self.assertEqual(response.status_code, 200)
    
    def test_face_embedding_admin(self):
        """Test face embedding admin interface"""
        response = self.client.get('/admin/recognition/faceembedding/')
        self.assertEqual(response.status_code, 200)


class IntegrationTestCase(APITestCase):
    """Integration tests for complete workflows"""
    
    def setUp(self):
        self.registration_url = reverse('user-register')
        self.login_url = reverse('user-login')
        self.start_enrollment_url = reverse('start-enrollment')
        self.authenticate_url = reverse('authenticate-frame')
        
        self.user_data = {
            'email': 'integration@example.com',
            'password': 'securepassword123',
            'password_confirm': 'securepassword123',
            'first_name': 'Integration',
            'last_name': 'Test'
        }
    
    def create_test_image(self):
        """Create a test image"""
        image = Image.new('RGB', (640, 480), color='green')
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
    
    @patch('core.face_recognition_engine.FaceRecognitionEngine')
    def test_complete_user_journey(self, mock_engine):
        """Test complete user journey from registration to authentication"""
        
        # Mock face recognition engine
        mock_engine_instance = MagicMock()
        mock_engine.return_value = mock_engine_instance
        
        # Step 1: User registration
        response = self.client.post(self.registration_url, self.user_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        user = User.objects.get(email='integration@example.com')
        
        # Step 2: User login
        login_response = self.client.post(self.login_url, {
            'email': 'integration@example.com',
            'password': 'securepassword123'
        })
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)
        
        # Authenticate client
        self.client.force_authenticate(user=user)
        
        # Step 3: Start enrollment
        enrollment_response = self.client.post(self.start_enrollment_url)
        self.assertEqual(enrollment_response.status_code, status.HTTP_201_CREATED)
        
        session_id = enrollment_response.data['session_id']
        
        # Step 4: Process enrollment frames (mock)
        mock_engine_instance.process_enrollment_frame.return_value = {
            'success': True,
            'quality_score': 0.85,
            'frame_count': 10,
            'total_required': 10
        }
        
        # Step 5: Complete enrollment (mock)
        mock_engine_instance.complete_enrollment.return_value = {
            'success': True,
            'embedding_id': 'test-embedding-id',
            'quality_score': 0.92
        }
        
        complete_response = self.client.post(
            reverse('complete-enrollment'),
            {'session_id': session_id}
        )
        self.assertEqual(complete_response.status_code, status.HTTP_200_OK)
        
        # Step 6: Face authentication (mock)
        mock_engine_instance.authenticate_face.return_value = {
            'success': True,
            'confidence': 0.95,
            'user_id': str(user.id),
            'liveness_passed': True,
            'obstacle_detected': False
        }
        
        auth_response = self.client.post(self.authenticate_url, {
            'frame_data': self.create_test_image()
        })
        self.assertEqual(auth_response.status_code, status.HTTP_200_OK)
        self.assertTrue(auth_response.data['success'])
        
        # Verify user journey completed successfully
        user.refresh_from_db()
        self.assertTrue(user.face_enrolled)
        
        # Verify logs were created
        self.assertTrue(UserActivity.objects.filter(user=user).exists())
        self.assertTrue(AuthenticationAttempt.objects.filter(user=user).exists())


if __name__ == '__main__':
    import django
    from django.conf import settings
    from django.test.utils import get_runner
    
    django.setup()
    TestRunner = get_runner(settings)
    test_runner = TestRunner()
    failures = test_runner.run_tests(['__main__'])
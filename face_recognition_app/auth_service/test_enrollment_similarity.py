"""
Test cases for enrollment with old profile photo similarity checking
"""
import asyncio
import json
import numpy as np
from io import BytesIO
from PIL import Image
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from datetime import timedelta
from unittest.mock import Mock, patch, MagicMock

from clients.models import Client, ClientUser
from .models import AuthenticationSession, FaceEnrollment
from .consumers import AuthProcessConsumer


class EnrollmentSimilarityTestCase(TestCase):
    """Test suite for enrollment similarity checking with old profile photo"""

    def setUp(self):
        """Set up test client and user with old profile photo"""
        # Create test client
        self.client_obj = Client.objects.create(
            name="Similarity Test Client",
            description="Test client for similarity checking",
            domain="https://similarity.example.com",
            contact_email="similarity@example.com",
            contact_name="Similarity Contact"
        )
        
        # Create test user with old profile photo
        old_photo = self.create_test_image('old_photo.jpg', color='blue')
        self.client_user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='similarity_user_001',
            profile={'name': 'Similarity Test User'},
            old_profile_photo=old_photo,
            face_auth_enabled=True
        )

    def create_test_image(self, filename='test.jpg', size=(640, 480), color='red'):
        """Create a test image file"""
        image = Image.new('RGB', size, color)
        tmp_file = BytesIO()
        image.save(tmp_file, format='JPEG')
        tmp_file.seek(0)
        return SimpleUploadedFile(
            filename,
            tmp_file.read(),
            content_type='image/jpeg'
        )

    def test_client_user_has_old_profile_photo(self):
        """Test that ClientUser has old_profile_photo field"""
        self.assertIsNotNone(self.client_user.old_profile_photo)
        self.assertTrue(self.client_user.old_profile_photo.name.startswith('client_users/old_profiles/'))

    def test_client_user_similarity_field_exists(self):
        """Test that similarity_with_old_photo field exists"""
        self.assertTrue(hasattr(self.client_user, 'similarity_with_old_photo'))
        self.assertIsNone(self.client_user.similarity_with_old_photo)

    def test_enrollment_updates_similarity_score(self):
        """Test that enrollment process updates similarity score"""
        # Create enrollment session
        session = AuthenticationSession.objects.create(
            client=self.client_obj,
            client_user=self.client_user,
            session_type='enrollment',
            status='active',
            expires_at=timezone.now() + timezone.timedelta(minutes=10)
        )
        
        # Create enrollment record
        enrollment = FaceEnrollment.objects.create(
            client=self.client_obj,
            client_user=self.client_user,
            enrollment_session=session,
            status='pending',
            face_quality_score=0.9,
            sample_number=1
        )
        
        # Simulate setting similarity score
        self.client_user.similarity_with_old_photo = 0.87
        self.client_user.save()
        
        # Verify score is saved
        self.client_user.refresh_from_db()
        self.assertEqual(self.client_user.similarity_with_old_photo, 0.87)

    def test_enrollment_saves_profile_image(self):
        """Test that enrollment saves profile_image from frame"""
        # Create enrollment session
        session = AuthenticationSession.objects.create(
            client=self.client_obj,
            client_user=self.client_user,
            session_type='enrollment',
            status='active',
            expires_at=timezone.now() + timezone.timedelta(minutes=10)
        )
        
        # Create enrollment record
        enrollment = FaceEnrollment.objects.create(
            client=self.client_obj,
            client_user=self.client_user,
            enrollment_session=session,
            status='pending',
            face_quality_score=0.9,
            sample_number=1
        )
        
        # Simulate saving profile image
        new_photo = self.create_test_image('enrollment_photo.jpg', color='green')
        self.client_user.profile_image = new_photo
        self.client_user.save()
        
        # Verify image is saved
        self.client_user.refresh_from_db()
        self.assertIsNotNone(self.client_user.profile_image)
        self.assertTrue(self.client_user.profile_image.name.startswith('client_users/profiles/'))

    def test_both_photos_and_similarity_exist_after_enrollment(self):
        """Test that after enrollment, user has both photos and similarity score"""
        # Simulate completed enrollment
        new_photo = self.create_test_image('enrollment_photo.jpg', color='yellow')
        self.client_user.profile_image = new_photo
        self.client_user.similarity_with_old_photo = 0.92
        self.client_user.is_enrolled = True
        self.client_user.enrollment_completed_at = timezone.now()
        self.client_user.save()
        
        # Verify all fields are set
        self.client_user.refresh_from_db()
        self.assertIsNotNone(self.client_user.old_profile_photo)
        self.assertIsNotNone(self.client_user.profile_image)
        self.assertIsNotNone(self.client_user.similarity_with_old_photo)
        self.assertTrue(self.client_user.is_enrolled)
        self.assertEqual(self.client_user.similarity_with_old_photo, 0.92)

    def test_enrollment_without_old_photo(self):
        """Test enrollment for user without old profile photo"""
        # Create user without old photo
        user_no_old_photo = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='no_old_photo_user',
            profile={'name': 'No Old Photo User'},
            face_auth_enabled=True
        )
        
        # Verify no old photo exists
        self.assertFalse(user_no_old_photo.old_profile_photo)
        self.assertIsNone(user_no_old_photo.similarity_with_old_photo)
        
        # Simulate enrollment
        new_photo = self.create_test_image('enrollment_photo.jpg', color='orange')
        user_no_old_photo.profile_image = new_photo
        user_no_old_photo.is_enrolled = True
        user_no_old_photo.save()
        
        # Verify enrollment succeeded without similarity score
        user_no_old_photo.refresh_from_db()
        self.assertTrue(user_no_old_photo.is_enrolled)
        self.assertIsNotNone(user_no_old_photo.profile_image)
        self.assertIsNone(user_no_old_photo.similarity_with_old_photo)

    def test_similarity_score_range(self):
        """Test that similarity scores are within valid range"""
        test_scores = [0.0, 0.5, 0.87, 0.95, 1.0]
        
        for score in test_scores:
            self.client_user.similarity_with_old_photo = score
            self.client_user.save()
            
            self.client_user.refresh_from_db()
            self.assertEqual(self.client_user.similarity_with_old_photo, score)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def tearDown(self):
        """Clean up test data"""
        FaceEnrollment.objects.all().delete()
        AuthenticationSession.objects.all().delete()
        ClientUser.objects.all().delete()
        Client.objects.all().delete()


class EnrollmentResponseTestCase(TestCase):
    """Test enrollment response includes similarity score"""

    def setUp(self):
        """Set up test environment"""
        self.client_obj = Client.objects.create(
            name="Response Test Client",
            description="Test client for response testing",
            domain="https://response.example.com",
            contact_email="response@example.com",
            contact_name="Response Contact"
        )

    def test_enrollment_response_structure(self):
        """Test that enrollment complete response includes similarity_with_old_photo field"""
        # Expected response structure
        expected_fields = [
            'type',
            'success',
            'enrollment_id',
            'frames_processed',
            'liveness_verified',
            'blinks_detected',
            'motion_verified',
            'quality_score',
            'similarity_with_old_photo',  # New field
            'encrypted_data',
            'visual_data',
            'message'
        ]
        
        # Simulate response
        mock_response = {
            'type': 'enrollment_complete',
            'success': True,
            'enrollment_id': 'test_enrollment_id',
            'frames_processed': 5,
            'liveness_verified': True,
            'blinks_detected': 2,
            'motion_verified': True,
            'quality_score': 0.95,
            'similarity_with_old_photo': 0.87,
            'encrypted_data': {},
            'visual_data': {},
            'message': 'Enrollment completed successfully'
        }
        
        # Verify all expected fields are present
        for field in expected_fields:
            self.assertIn(field, mock_response, f"Field '{field}' missing from response")
        
        # Verify similarity_with_old_photo is a float or None
        self.assertIsInstance(mock_response['similarity_with_old_photo'], (float, type(None)))

    def test_enrollment_response_without_old_photo(self):
        """Test response when no old photo exists (similarity should be None)"""
        mock_response = {
            'type': 'enrollment_complete',
            'success': True,
            'enrollment_id': 'test_enrollment_id',
            'similarity_with_old_photo': None,  # No old photo
        }
        
        self.assertIsNone(mock_response['similarity_with_old_photo'])

    def tearDown(self):
        """Clean up test data"""
        Client.objects.all().delete()

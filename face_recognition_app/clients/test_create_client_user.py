"""
Test cases for creating ClientUser with profile photos and similarity comparison
"""
from io import BytesIO
from PIL import Image
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile

from .models import Client, ClientUser
from .serializers import ClientUserSerializer, ClientUserWriteSerializer


class ClientUserModelTestCase(TestCase):
    """Test suite for ClientUser model with new photo fields"""

    def setUp(self):
        """Set up test client"""
        self.client_obj = Client.objects.create(
            name="Test Client",
            description="Test client for user creation",
            domain="https://test.example.com",
            contact_email="test@example.com",
            contact_name="Test Contact"
        )

    def create_test_image(self, filename='test.jpg', size=(100, 100), color='red'):
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

    def test_create_user_without_photos(self):
        """Test creating a user without any profile photos"""
        user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='user001',
            profile={'name': 'John Doe', 'email': 'john@example.com'},
            face_auth_enabled=True
        )
        
        self.assertEqual(user.external_user_id, 'user001')
        self.assertFalse(user.profile_image)
        self.assertFalse(user.old_profile_photo)
        self.assertIsNone(user.similarity_with_old_photo)

    def test_create_user_with_old_profile_photo(self):
        """Test creating a user with old profile photo"""
        old_photo = self.create_test_image('old_photo.jpg', color='blue')
        
        user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='user002',
            profile={'name': 'Jane Smith'},
            old_profile_photo=old_photo,
            face_auth_enabled=True
        )
        
        self.assertEqual(user.external_user_id, 'user002')
        self.assertTrue(user.old_profile_photo)
        self.assertTrue(user.old_profile_photo.name.startswith('client_users/old_profiles/'))
        self.assertFalse(user.profile_image)

    def test_create_user_with_both_photos(self):
        """Test creating a user with both old and current profile photos"""
        old_photo = self.create_test_image('old_photo.jpg', color='green')
        current_photo = self.create_test_image('current_photo.jpg', color='yellow')
        
        user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='user003',
            profile={'name': 'Bob Johnson'},
            old_profile_photo=old_photo,
            profile_image=current_photo,
            face_auth_enabled=True
        )
        
        self.assertEqual(user.external_user_id, 'user003')
        self.assertTrue(user.old_profile_photo)
        self.assertTrue(user.profile_image)
        self.assertTrue(user.old_profile_photo.name.startswith('client_users/old_profiles/'))
        self.assertTrue(user.profile_image.name.startswith('client_users/profiles/'))

    def test_user_with_similarity_score(self):
        """Test creating and updating user with similarity score"""
        user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='user004',
            profile={'name': 'Alice Williams'},
            similarity_with_old_photo=0.87
        )
        
        self.assertEqual(user.similarity_with_old_photo, 0.87)
        
        # Update similarity score
        user.similarity_with_old_photo = 0.95
        user.save()
        
        user.refresh_from_db()
        self.assertEqual(user.similarity_with_old_photo, 0.95)

    def test_user_model_fields_exist(self):
        """Test that new fields exist in the model"""
        user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='user005',
            profile={'name': 'Test User'}
        )
        
        # Check that new fields exist and are accessible
        self.assertTrue(hasattr(user, 'old_profile_photo'))
        self.assertTrue(hasattr(user, 'profile_image'))
        self.assertTrue(hasattr(user, 'similarity_with_old_photo'))

    def test_user_with_external_uuid(self):
        """Test creating a user with external_user_uuid"""
        import uuid
        user_uuid = uuid.uuid4()
        
        user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='user006',
            external_user_uuid=user_uuid,
            profile={'name': 'UUID User'},
            face_auth_enabled=False
        )
        
        self.assertEqual(user.external_user_uuid, user_uuid)
        self.assertEqual(user.face_auth_enabled, False)

    def test_duplicate_external_user_id(self):
        """Test that duplicate external_user_id within same client is not allowed"""
        from django.db import IntegrityError, transaction
        
        ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='duplicate_user',
            profile={'name': 'First User'}
        )
        
        # Try to create another user with same external_user_id
        # Use transaction.atomic to properly handle the IntegrityError
        with transaction.atomic():
            with self.assertRaises(IntegrityError):
                ClientUser.objects.create(
                    client=self.client_obj,
                    external_user_id='duplicate_user',
                    profile={'name': 'Second User'}
                )

    def tearDown(self):
        """Clean up test data"""
        ClientUser.objects.all().delete()
        Client.objects.all().delete()


class ClientUserSerializerTestCase(TestCase):
    """Test suite for ClientUser serializers"""

    def setUp(self):
        """Set up test client and user"""
        self.client_obj = Client.objects.create(
            name="Serializer Test Client",
            description="Test client for serializers",
            domain="https://serializer.example.com",
            contact_email="serializer@example.com",
            contact_name="Serializer Contact"
        )

    def create_test_image(self, filename='test.jpg', size=(100, 100), color='red'):
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

    def test_serializer_includes_new_fields(self):
        """Test that ClientUserSerializer includes new photo fields"""
        old_photo = self.create_test_image('old.jpg', color='red')
        current_photo = self.create_test_image('current.jpg', color='blue')
        
        user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='serial_user_001',
            profile={'name': 'Serializer Test'},
            old_profile_photo=old_photo,
            profile_image=current_photo,
            similarity_with_old_photo=0.92
        )
        
        serializer = ClientUserSerializer(user)
        data = serializer.data
        
        # Check that new fields are in the serialized data
        self.assertIn('old_profile_photo_url', data)
        self.assertIn('profile_image_url', data)
        self.assertIn('similarity_with_old_photo', data)
        self.assertEqual(data['similarity_with_old_photo'], 0.92)

    def test_write_serializer_accepts_photos(self):
        """Test that ClientUserWriteSerializer accepts photo fields"""
        old_photo = self.create_test_image('old_write.jpg', color='green')
        
        data = {
            'external_user_id': 'write_user_001',
            'profile': {'name': 'Write Test'},
            'old_profile_photo': old_photo,
        }
        
        serializer = ClientUserWriteSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        
        user = serializer.save(client=self.client_obj)
        self.assertEqual(user.external_user_id, 'write_user_001')
        self.assertTrue(user.old_profile_photo)

    def test_serializer_without_photos(self):
        """Test serializer with user that has no photos"""
        user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='serial_user_002',
            profile={'name': 'No Photos User'}
        )
        
        serializer = ClientUserSerializer(user)
        data = serializer.data
        
        self.assertIsNone(data['old_profile_photo_url'])
        self.assertIsNone(data['profile_image_url'])
        self.assertIsNone(data['similarity_with_old_photo'])

    def test_write_serializer_with_both_photos(self):
        """Test ClientUserWriteSerializer with both photo fields"""
        old_photo = self.create_test_image('old.jpg', color='purple')
        new_photo = self.create_test_image('new.jpg', color='orange')
        
        data = {
            'external_user_id': 'write_user_002',
            'profile': {'name': 'Both Photos Test'},
            'old_profile_photo': old_photo,
            'profile_image': new_photo,
            'face_auth_enabled': True
        }
        
        serializer = ClientUserWriteSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        
        user = serializer.save(client=self.client_obj)
        self.assertTrue(user.old_profile_photo)
        self.assertTrue(user.profile_image)
        self.assertEqual(user.face_auth_enabled, True)

    def test_serializer_field_readonly(self):
        """Test that similarity_with_old_photo is read-only in main serializer"""
        user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id='readonly_user',
            profile={'name': 'ReadOnly Test'},
            similarity_with_old_photo=0.75
        )
        
        serializer = ClientUserSerializer(user)
        
        # Verify field is in read_only_fields
        self.assertIn('similarity_with_old_photo', serializer.Meta.read_only_fields)
        self.assertIn('old_profile_photo_url', serializer.Meta.read_only_fields)
        self.assertIn('profile_image_url', serializer.Meta.read_only_fields)

    def tearDown(self):
        """Clean up test data"""
        ClientUser.objects.all().delete()
        Client.objects.all().delete()

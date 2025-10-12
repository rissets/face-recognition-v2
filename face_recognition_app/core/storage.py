"""
Custom storage backends for MinIO object storage
"""
from django.conf import settings
from storages.backends.s3boto3 import S3Boto3Storage
from django.core.files.storage import get_storage_class


class MinIOMediaStorage(S3Boto3Storage):
    """Custom storage class for MinIO media files"""
    bucket_name = settings.AWS_STORAGE_BUCKET_NAME
    location = settings.AWS_LOCATION
    default_acl = 'public-read'
    file_overwrite = False
    custom_domain = False
    
    def __init__(self, *args, **kwargs):
        kwargs['bucket_name'] = self.bucket_name
        kwargs['location'] = self.location
        super().__init__(*args, **kwargs)


class MinIOStaticStorage(S3Boto3Storage):
    """Custom storage class for MinIO static files"""
    bucket_name = settings.AWS_STORAGE_BUCKET_NAME
    location = 'static'
    default_acl = 'public-read'
    
    def __init__(self, *args, **kwargs):
        kwargs['bucket_name'] = self.bucket_name
        kwargs['location'] = self.location
        super().__init__(*args, **kwargs)


class LocalMediaStorage:
    """Fallback local storage for development"""
    
    def __init__(self):
        self.storage = get_storage_class('django.core.files.storage.FileSystemStorage')()
    
    def save(self, name, content, max_length=None):
        return self.storage.save(name, content, max_length)
    
    def delete(self, name):
        return self.storage.delete(name)
    
    def exists(self, name):
        return self.storage.exists(name)
    
    def url(self, name):
        return self.storage.url(name)
"""
Custom storage backends for MinIO object storage
"""
from django.conf import settings
from storages.backends.s3boto3 import S3Boto3Storage


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
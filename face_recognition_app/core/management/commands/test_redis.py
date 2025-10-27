"""
Management command to test Redis connectivity and configuration.
"""

from django.core.management.base import BaseCommand
from django.core.cache import cache
from django.conf import settings
import redis
import sys


class Command(BaseCommand):
    help = 'Test Redis connectivity and configuration'

    def add_arguments(self, parser):
        parser.add_argument(
            '--fix-auth',
            action='store_true',
            help='Attempt to fix Redis authentication issues',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Testing Redis connectivity...'))
        
        # Test cache connectivity
        try:
            # Test default cache
            cache.set('test_key', 'test_value', 30)
            result = cache.get('test_key')
            if result == 'test_value':
                self.stdout.write(self.style.SUCCESS('✓ Default cache (Redis) connection successful'))
                cache.delete('test_key')
            else:
                self.stdout.write(self.style.ERROR('✗ Default cache test failed'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Default cache connection failed: {str(e)}'))
            
        # Test sessions cache
        try:
            from django.core.cache import caches
            sessions_cache = caches['sessions']
            sessions_cache.set('test_session_key', 'test_session_value', 30)
            result = sessions_cache.get('test_session_key')
            if result == 'test_session_value':
                self.stdout.write(self.style.SUCCESS('✓ Sessions cache (Redis) connection successful'))
                sessions_cache.delete('test_session_key')
            else:
                self.stdout.write(self.style.ERROR('✗ Sessions cache test failed'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Sessions cache connection failed: {str(e)}'))

        # Test direct Redis connection
        try:
            redis_url = getattr(settings, 'REDIS_BASE_URL', 'redis://localhost:6379')
            client = redis.from_url(redis_url)
            client.ping()
            self.stdout.write(self.style.SUCCESS('✓ Direct Redis connection successful'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Direct Redis connection failed: {str(e)}'))
            
            if options['fix_auth']:
                self.stdout.write(self.style.WARNING('Attempting to fix Redis authentication...'))
                self._suggest_redis_fixes(e)

        # Display current Redis configuration
        self.stdout.write('\n' + self.style.HTTP_INFO('Current Redis Configuration:'))
        self.stdout.write(f"REDIS_HOST: {getattr(settings, 'REDIS_HOST', 'Not set')}")
        self.stdout.write(f"REDIS_PORT: {getattr(settings, 'REDIS_PORT', 'Not set')}")
        self.stdout.write(f"REDIS_PASSWORD: {'***' if getattr(settings, 'REDIS_PASSWORD', '') else 'Not set'}")
        self.stdout.write(f"REDIS_BASE_URL: {getattr(settings, 'REDIS_BASE_URL', 'Not set')}")

    def _suggest_redis_fixes(self, error):
        """Suggest potential fixes for Redis connection issues."""
        error_str = str(error).lower()
        
        if 'authentication' in error_str:
            self.stdout.write(self.style.WARNING('''
Redis Authentication Error Fixes:
1. Check if Redis password is correctly set in .env file
2. Verify REDIS_PASSWORD matches the Redis server configuration
3. Check docker-compose.yml for Redis requirepass setting
4. Try connecting without password for local development
            '''))
            
        elif 'connection' in error_str:
            self.stdout.write(self.style.WARNING('''
Redis Connection Error Fixes:
1. Ensure Redis server is running
2. Check REDIS_HOST and REDIS_PORT settings
3. Verify network connectivity to Redis server
4. Check firewall settings
            '''))
        
        self.stdout.write(self.style.HTTP_INFO('''
Quick Environment Fix Commands:
# For development (no password):
export REDIS_PASSWORD=""

# For production (with password):
export REDIS_PASSWORD="redis_password"

# Test connection:
python manage.py test_redis
        '''))
"""
Django Management Command untuk testing Telegram notifications
Usage: python manage.py test_telegram_monitoring
"""
from django.core.management.base import BaseCommand
from core.telegram_logger import telegram_logger


class Command(BaseCommand):
    help = 'Test Telegram error monitoring notifications'

    def add_arguments(self, parser):
        parser.add_argument(
            '--type',
            type=str,
            default='critical',
            choices=['critical', 'exception', 'security', 'database', 'api', 'celery', 'websocket', 'all'],
            help='Type of notification to test'
        )

    def handle(self, *args, **options):
        test_type = options['type']
        
        self.stdout.write(self.style.WARNING('Testing Telegram error monitoring...'))
        
        # Check configuration
        if not telegram_logger.is_configured():
            self.stdout.write(self.style.ERROR('❌ Telegram logger is not configured properly'))
            self.stdout.write(f'  - Enabled: {telegram_logger.enabled}')
            self.stdout.write(f'  - Bot Token: {"Set" if telegram_logger.bot_token else "Not Set"}')
            self.stdout.write(f'  - Chat ID: {"Set" if telegram_logger.chat_id else "Not Set"}')
            return
        
        self.stdout.write(self.style.SUCCESS('✅ Telegram logger is configured'))
        self.stdout.write(f'  - Environment: {telegram_logger.environment}')
        
        # Test notifications
        success_count = 0
        total_count = 0
        
        test_functions = []
        
        if test_type == 'all' or test_type == 'critical':
            test_functions.append(('Critical Error', self.test_critical_error))
        if test_type == 'all' or test_type == 'exception':
            test_functions.append(('Exception', self.test_exception))
        if test_type == 'all' or test_type == 'security':
            test_functions.append(('Security Alert', self.test_security_alert))
        if test_type == 'all' or test_type == 'database':
            test_functions.append(('Database Error', self.test_database_error))
        if test_type == 'all' or test_type == 'api':
            test_functions.append(('API Error', self.test_api_error))
        if test_type == 'all' or test_type == 'celery':
            test_functions.append(('Celery Error', self.test_celery_error))
        if test_type == 'all' or test_type == 'websocket':
            test_functions.append(('WebSocket Error', self.test_websocket_error))
        
        for test_name, test_func in test_functions:
            total_count += 1
            self.stdout.write(f'\nTesting {test_name}...')
            if test_func():
                success_count += 1
                self.stdout.write(self.style.SUCCESS(f'✅ {test_name} sent successfully'))
            else:
                self.stdout.write(self.style.ERROR(f'❌ {test_name} failed'))
        
        self.stdout.write(f'\n{self.style.SUCCESS("="*50)}')
        self.stdout.write(f'Results: {success_count}/{total_count} notifications sent successfully')
        
        if success_count == total_count:
            self.stdout.write(self.style.SUCCESS('✅ All tests passed!'))
        else:
            self.stdout.write(self.style.WARNING(f'⚠️  Some tests failed'))

    def test_critical_error(self):
        """Test critical error notification"""
        return telegram_logger.log_critical_error(
            message="[TEST] Critical error from Face Recognition System",
            additional_context={
                'test': True,
                'component': 'management_command'
            }
        )

    def test_exception(self):
        """Test exception notification"""
        try:
            # Generate a test exception
            raise ValueError("This is a test exception for Telegram monitoring")
        except Exception as e:
            return telegram_logger.log_exception(
                message="[TEST] Exception occurred during testing",
                exception=e,
                additional_context={
                    'test': True,
                    'component': 'management_command'
                }
            )

    def test_security_alert(self):
        """Test security alert notification"""
        return telegram_logger.log_security_alert(
            message="[TEST] Security alert - Suspicious activity detected",
            request_data={
                'method': 'POST',
                'path': '/api/test/',
                'ip': '192.168.1.100',
                'user_agent': 'Test User Agent'
            },
            additional_context={
                'test': True,
                'alert_type': 'sql_injection_attempt'
            }
        )

    def test_database_error(self):
        """Test database error notification"""
        return telegram_logger.log_database_error(
            message="[TEST] Database connection error",
            additional_context={
                'test': True,
                'database': 'postgresql',
                'operation': 'connection'
            }
        )

    def test_api_error(self):
        """Test API error notification"""
        return telegram_logger.log_api_error(
            message="[TEST] API endpoint returned 500 error",
            request_data={
                'method': 'GET',
                'path': '/api/face-recognition/',
                'ip': '127.0.0.1'
            },
            additional_context={
                'test': True,
                'status_code': 500,
                'endpoint': '/api/face-recognition/'
            }
        )

    def test_celery_error(self):
        """Test Celery error notification"""
        return telegram_logger.log_celery_error(
            message="[TEST] Celery task failed",
            task_name='test_background_task',
            additional_context={
                'test': True,
                'task_id': 'test-task-123',
                'retry_count': 3
            }
        )

    def test_websocket_error(self):
        """Test WebSocket error notification"""
        return telegram_logger.log_websocket_error(
            message="[TEST] WebSocket connection error",
            additional_context={
                'test': True,
                'connection_id': 'ws-test-123',
                'error_type': 'connection_refused'
            }
        )

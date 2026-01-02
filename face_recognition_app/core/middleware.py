"""
Middleware for error monitoring and logging to Telegram
"""
import logging
import traceback
from django.utils.deprecation import MiddlewareMixin
from django.http import JsonResponse
from .telegram_logger import telegram_logger

logger = logging.getLogger(__name__)


class TelegramErrorMonitoringMiddleware(MiddlewareMixin):
    """
    Middleware to catch and log exceptions to Telegram
    """
    
    def process_exception(self, request, exception):
        """
        Called when an exception is raised during request processing
        
        Args:
            request: Django request object
            exception: Exception that was raised
            
        Returns:
            None to allow other middleware to handle the exception
        """
        # Skip logging for certain exception types if needed
        skip_exceptions = (
            # Add exception types to skip here
            # e.g., Http404, PermissionDenied
        )
        
        if isinstance(exception, skip_exceptions):
            return None
        
        # Extract request information
        request_data = {
            'method': request.method,
            'path': request.path,
            'ip': self.get_client_ip(request),
            'user_agent': request.META.get('HTTP_USER_AGENT', 'N/A')
        }
        
        # Extract user information if available
        user_info = None
        if hasattr(request, 'user') and request.user.is_authenticated:
            user_info = {
                'id': request.user.id,
                'username': getattr(request.user, 'username', 'N/A'),
                'email': getattr(request.user, 'email', 'N/A')
            }
        
        # Extract client information if available
        if hasattr(request, 'client_user') and request.client_user:
            user_info = {
                'client_id': getattr(request.client_user, 'client_id', 'N/A'),
                'client_name': getattr(request.client_user, 'name', 'N/A'),
                'email': getattr(request.client_user, 'email', 'N/A')
            }
        
        # Additional context
        additional_context = {}
        
        # Add query parameters (sanitized)
        if request.GET:
            additional_context['query_params'] = dict(request.GET)
        
        # Add POST data keys (not values for security)
        if request.POST:
            additional_context['post_keys'] = list(request.POST.keys())
        
        # Log to Telegram
        try:
            telegram_logger.log_critical_error(
                message=f"Unhandled exception in {request.path}",
                exception=exception,
                request_data=request_data,
                user_info=user_info,
                additional_context=additional_context
            )
        except Exception as e:
            logger.error(f"Failed to log exception to Telegram: {str(e)}")
        
        # Return None to allow Django to handle the exception normally
        return None
    
    @staticmethod
    def get_client_ip(request) -> str:
        """Get client IP address from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR', 'N/A')
        return ip


class TelegramRequestMonitoringMiddleware(MiddlewareMixin):
    """
    Middleware to monitor suspicious requests and security events
    """
    
    def process_request(self, request):
        """
        Monitor requests for suspicious activity
        
        Args:
            request: Django request object
            
        Returns:
            None to continue processing
        """
        # Monitor for suspicious patterns
        suspicious_patterns = []
        
        # Check for SQL injection attempts
        sql_patterns = ['union', 'select', 'drop', 'insert', 'delete', '--', ';']
        query_string = request.META.get('QUERY_STRING', '').lower()
        for pattern in sql_patterns:
            if pattern in query_string:
                suspicious_patterns.append(f"SQL pattern detected: {pattern}")
        
        # Check for XSS attempts
        xss_patterns = ['<script', 'javascript:', 'onerror=', 'onload=']
        for pattern in xss_patterns:
            if pattern in query_string:
                suspicious_patterns.append(f"XSS pattern detected: {pattern}")
        
        # Check for path traversal
        if '../' in request.path or '..\\' in request.path:
            suspicious_patterns.append("Path traversal attempt detected")
        
        # If suspicious patterns detected, log to Telegram
        if suspicious_patterns:
            request_data = {
                'method': request.method,
                'path': request.path,
                'ip': self.get_client_ip(request),
                'user_agent': request.META.get('HTTP_USER_AGENT', 'N/A')
            }
            
            additional_context = {
                'suspicious_patterns': ', '.join(suspicious_patterns),
                'query_string': query_string[:200]
            }
            
            try:
                telegram_logger.log_security_alert(
                    message=f"Suspicious request detected from {request_data['ip']}",
                    request_data=request_data,
                    additional_context=additional_context
                )
            except Exception as e:
                logger.error(f"Failed to log security alert to Telegram: {str(e)}")
        
        return None
    
    @staticmethod
    def get_client_ip(request) -> str:
        """Get client IP address from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR', 'N/A')
        return ip


class TelegramResponseMonitoringMiddleware(MiddlewareMixin):
    """
    Middleware to monitor response status codes and log errors
    """
    
    def process_response(self, request, response):
        """
        Monitor response status codes
        
        Args:
            request: Django request object
            response: Django response object
            
        Returns:
            response object
        """
        # Log 5xx errors to Telegram
        if response.status_code >= 500:
            request_data = {
                'method': request.method,
                'path': request.path,
                'ip': self.get_client_ip(request),
                'user_agent': request.META.get('HTTP_USER_AGENT', 'N/A')
            }
            
            additional_context = {
                'status_code': response.status_code,
                'response_reason': getattr(response, 'reason_phrase', 'Unknown')
            }
            
            try:
                telegram_logger.log_api_error(
                    message=f"Server error {response.status_code} on {request.path}",
                    request_data=request_data,
                    additional_context=additional_context
                )
            except Exception as e:
                logger.error(f"Failed to log response error to Telegram: {str(e)}")
        
        return response
    
    @staticmethod
    def get_client_ip(request) -> str:
        """Get client IP address from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR', 'N/A')
        return ip

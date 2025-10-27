"""
Custom exception handlers for the Face Recognition API.
Handles various error conditions gracefully.
"""

import logging
from django.http import JsonResponse
from rest_framework.views import exception_handler
from rest_framework import status
from django_redis.exceptions import ConnectionInterrupted
import redis.exceptions

logger = logging.getLogger(__name__)


def custom_exception_handler(exc, context):
    """
    Custom exception handler that provides better error handling for various exceptions,
    including Redis connection issues.
    """
    
    # Handle Redis connection errors gracefully
    if isinstance(exc, (ConnectionInterrupted, redis.exceptions.AuthenticationError, redis.exceptions.ConnectionError)):
        logger.error(f"Redis connection error: {str(exc)}")
        
        # Check if this is during API docs access
        request = context.get('request')
        if request and '/api/docs/' in str(request.path):
            # For API docs, we can continue without Redis
            logger.warning("Redis unavailable for API documentation - continuing without caching")
            # Return None to let the view continue processing
            return None
            
        # For other endpoints, return a service unavailable error
        return JsonResponse({
            'error': 'Service temporarily unavailable',
            'detail': 'Backend services are being updated. Please try again in a moment.',
            'code': 'service_unavailable'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    # Call REST framework's default exception handler first to get the standard error response
    response = exception_handler(exc, context)
    
    if response is not None:
        # Log the error for monitoring
        logger.error(f"API Error: {str(exc)} - Path: {context.get('request', {}).path if context.get('request') else 'Unknown'}")
        
        # Customize the error response format
        custom_response_data = {
            'error': True,
            'message': 'An error occurred',
            'details': response.data
        }
        response.data = custom_response_data
    
    return response


def handle_redis_connection_error():
    """
    Helper function to handle Redis connection errors.
    Can be used in views or other components that need Redis.
    """
    logger.warning("Redis connection failed - falling back to memory cache or skipping cache")
    return None
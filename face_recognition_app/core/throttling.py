"""
Custom throttling classes that handle Redis connection errors gracefully.
"""

import logging
from rest_framework.throttling import AnonRateThrottle, UserRateThrottle
from django_redis.exceptions import ConnectionInterrupted
import redis.exceptions

logger = logging.getLogger(__name__)


class SafeAnonRateThrottle(AnonRateThrottle):
    """
    Anonymous rate throttle that gracefully handles Redis connection errors.
    Falls back to allowing requests if Redis is unavailable.
    """

    def allow_request(self, request, view):
        try:
            return super().allow_request(request, view)
        except (ConnectionInterrupted, redis.exceptions.AuthenticationError, redis.exceptions.ConnectionError) as e:
            logger.warning(f"Redis connection error in throttling - allowing request: {str(e)}")
            # Allow the request if Redis is unavailable
            return True
        except Exception as e:
            logger.error(f"Unexpected error in throttling - allowing request: {str(e)}")
            # Allow the request for any other throttling errors
            return True


class SafeUserRateThrottle(UserRateThrottle):
    """
    User rate throttle that gracefully handles Redis connection errors.
    Falls back to allowing requests if Redis is unavailable.
    """

    def allow_request(self, request, view):
        try:
            return super().allow_request(request, view)
        except (ConnectionInterrupted, redis.exceptions.AuthenticationError, redis.exceptions.ConnectionError) as e:
            logger.warning(f"Redis connection error in throttling - allowing request: {str(e)}")
            # Allow the request if Redis is unavailable
            return True
        except Exception as e:
            logger.error(f"Unexpected error in throttling - allowing request: {str(e)}")
            # Allow the request for any other throttling errors
            return True
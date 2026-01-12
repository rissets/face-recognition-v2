"""
Middleware utilities for client-side observability.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from django.utils.deprecation import MiddlewareMixin

from .models import ClientAPIUsage, Client

logger = logging.getLogger(__name__)


class ClientUsageLoggingMiddleware(MiddlewareMixin):
    """
    Capture API usage for authenticated third-party clients.

    The middleware records endpoint category, HTTP method, status code,
    execution time, and basic request metadata so that the `ClientAPIUsage`
    dashboard reflects live traffic.
    """

    #: Prefixes that we consider part of the public REST API surface.
    API_PREFIXES: List[str] = [
        "/api/",
    ]

    def process_view(self, request, view_func, view_args, view_kwargs):  # noqa: D401
        request._client_usage_start_ts = time.perf_counter()  # type: ignore[attr-defined]
        return None

    def process_response(self, request, response):  # noqa: D401
        try:
            is_api_request = self._is_api_request(request.path)
            
            if not is_api_request:
                return response

            # Try to get client from request (set by DRF authentication)
            client = getattr(request, "client", None)
            
            # If no client, try to authenticate using API key from headers
            if not client:
                client = self._try_authenticate_client(request)
            
            # Debug logging - show actual headers for debugging
            api_key = request.META.get('HTTP_X_API_KEY', 'None')
            auth_header = request.META.get('HTTP_AUTHORIZATION', 'None')
            logger.info(f"Middleware processing: {request.path}, client: {client.name if client else None}, is_api: {is_api_request}")
            logger.info(f"Headers - X-API-Key: {api_key[:10] if api_key != 'None' else 'None'}, Authorization: {auth_header[:20] if auth_header != 'None' else 'None'}")
            
            if not client:
                logger.warning(f"No client found for API request: {request.path}")
                return response

            endpoint = self._resolve_endpoint_category(request.path)
            usage_entry = ClientAPIUsage.objects.create(
                client=client,
                endpoint=endpoint,
                method=request.method.upper(),
                status_code=getattr(response, "status_code", 0) or 0,
                ip_address=self._get_client_ip(request),
                user_agent=request.META.get("HTTP_USER_AGENT", ""),
                response_time_ms=self._compute_duration_ms(request),
                metadata=self._build_metadata(request),
            )
            logger.info(f"Created API usage entry: {usage_entry.id} for {endpoint} - client: {client.client_id}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.exception(f"Failed to record client API usage: {e}")
        finally:
            if hasattr(request, "_client_usage_start_ts"):
                delattr(request, "_client_usage_start_ts")

        return response

    # ---------------------------------------------------------------------#
    # Helpers
    # ---------------------------------------------------------------------#

    def _compute_duration_ms(self, request) -> float:
        start = getattr(request, "_client_usage_start_ts", None)
        if not start:
            return 0.0
        return round((time.perf_counter() - start) * 1000, 3)

    @staticmethod
    def _get_client_ip(request) -> str:
        headers = request.META
        forwarded = headers.get("HTTP_X_FORWARDED_FOR")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return headers.get("REMOTE_ADDR", "0.0.0.0")

    def _is_api_request(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.API_PREFIXES)

    @staticmethod
    def _build_metadata(request) -> Dict[str, object]:
        params = {
            key: request.GET.getlist(key) if len(request.GET.getlist(key)) > 1 else request.GET.get(key)
            for key in request.GET.keys()
        }
        return {
            "path": request.path,
            "query_params": params,
            "view_name": getattr(getattr(request, "resolver_match", None), "view_name", None),
        }

    def _try_authenticate_client(self, request) -> Optional[Client]:
        """Try to authenticate client from API key in headers"""
        api_key = request.META.get('HTTP_X_API_KEY')
        if not api_key:
            auth_header = request.META.get('HTTP_AUTHORIZATION')
            if auth_header and auth_header.startswith('Bearer '):
                api_key = auth_header[7:]  # Remove 'Bearer ' prefix
        
        logger.info(f"API Key found: {bool(api_key)}")
        if api_key:
            logger.info(f"Full API Key: {api_key}")
        
        if api_key:
            try:
                client = Client.find_active_by_api_key(api_key)
                if not client:
                    logger.warning(f"Client not found for API key: {api_key}")
                    # Debug: check all active clients
                    all_clients = Client.objects.filter(status='active')
                    logger.info(f"Total active clients: {all_clients.count()}")
                    for c in all_clients:
                        logger.info(f"Client {c.name} has API key: {c.api_key}")
                else:
                    logger.info(f"Client found: {client.name}")
                return client
            except Exception as e:
                logger.error(f"Error finding client by API key: {e}")
        return None

    @staticmethod
    def _resolve_endpoint_category(path: str) -> str:
        lowered = path.lower()
        if "enroll" in lowered:
            return "enrollment"
        if "liveness" in lowered:
            return "liveness"
        if "webhook" in lowered:
            return "webhook"
        if "recognition" in lowered or "auth" in lowered:
            return "recognition"
        return "analytics"

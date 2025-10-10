"""
Middleware utilities for client-side observability.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List

from django.utils.deprecation import MiddlewareMixin

from .models import ClientAPIUsage

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
            client = getattr(request, "client", None)
            if not client or not self._is_api_request(request.path):
                return response

            endpoint = self._resolve_endpoint_category(request.path)
            ClientAPIUsage.objects.create(
                client=client,
                endpoint=endpoint,
                method=request.method.upper(),
                status_code=getattr(response, "status_code", 0) or 0,
                ip_address=self._get_client_ip(request),
                user_agent=request.META.get("HTTP_USER_AGENT", ""),
                response_time_ms=self._compute_duration_ms(request),
                metadata=self._build_metadata(request),
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to record client API usage")
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

"""
Telegram Error Logger for Face Recognition System
Sends critical errors and exceptions to Telegram chat for real-time monitoring
"""
import requests
import traceback
import json
from datetime import datetime
from typing import Optional, Dict, Any
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class TelegramLogger:
    """
    Handles sending error notifications to Telegram
    """
    
    def __init__(self):
        self.bot_token = getattr(settings, 'TELEGRAM_BOT_TOKEN', None)
        self.chat_id = getattr(settings, 'TELEGRAM_CHAT_ID', None)
        self.enabled = getattr(settings, 'TELEGRAM_ERROR_LOGGING_ENABLED', False)
        self.environment = getattr(settings, 'ENVIRONMENT', 'development')
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else None
        
    def is_configured(self) -> bool:
        """Check if Telegram logging is properly configured"""
        return all([self.bot_token, self.chat_id, self.enabled, self.api_url])
    
    def format_error_message(
        self,
        error_type: str,
        message: str,
        exception: Optional[Exception] = None,
        request_data: Optional[Dict[str, Any]] = None,
        user_info: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format error message for Telegram
        
        Args:
            error_type: Type of error (e.g., 'Critical Error', 'Exception', 'Security Alert')
            message: Main error message
            exception: Exception object if available
            request_data: HTTP request information
            user_info: User information if available
            additional_context: Any additional context information
            
        Returns:
            Formatted message string
        """
        lines = [
            f"🚨 *{error_type}*",
            f"*Environment:* `{self.environment}`",
            f"*Time:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}`",
            "",
            f"*Message:*",
            f"```",
            f"{message[:500]}" + ("..." if len(message) > 500 else ""),
            f"```",
        ]
        
        if exception:
            lines.extend([
                "",
                f"*Exception Type:* `{type(exception).__name__}`",
                f"*Exception Details:*",
                f"```",
                f"{str(exception)[:300]}" + ("..." if len(str(exception)) > 300 else ""),
                f"```",
            ])
            
            # Add traceback
            tb = traceback.format_exc()
            if tb and tb.strip() != "NoneType: None":
                lines.extend([
                    "",
                    f"*Traceback:*",
                    f"```",
                    f"{tb[:800]}" + ("..." if len(tb) > 800 else ""),
                    f"```",
                ])
        
        if request_data:
            lines.extend([
                "",
                f"*Request Info:*",
                f"• Method: `{request_data.get('method', 'N/A')}`",
                f"• Path: `{request_data.get('path', 'N/A')}`",
                f"• IP: `{request_data.get('ip', 'N/A')}`",
            ])
            
            if request_data.get('user_agent'):
                lines.append(f"• User Agent: `{request_data['user_agent'][:100]}`")
        
        if user_info:
            lines.extend([
                "",
                f"*User Info:*",
                f"• ID: `{user_info.get('id', 'N/A')}`",
                f"• Username: `{user_info.get('username', 'Anonymous')}`",
                f"• Email: `{user_info.get('email', 'N/A')}`",
            ])
        
        if additional_context:
            lines.extend([
                "",
                f"*Additional Context:*",
            ])
            for key, value in additional_context.items():
                lines.append(f"• {key}: `{str(value)[:100]}`")
        
        return "\n".join(lines)
    
    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send message to Telegram
        
        Args:
            message: Message text to send
            parse_mode: Parse mode (Markdown or HTML)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.is_configured():
            logger.warning("Telegram logger not configured properly. Skipping notification.")
            return False
        
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Successfully sent error notification to Telegram")
                return True
            else:
                logger.error(f"Failed to send Telegram notification: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Exception while sending Telegram notification: {str(e)}")
            return False
    
    def log_error(
        self,
        error_type: str,
        message: str,
        exception: Optional[Exception] = None,
        request_data: Optional[Dict[str, Any]] = None,
        user_info: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log error to Telegram
        
        Args:
            error_type: Type of error
            message: Error message
            exception: Exception object if available
            request_data: Request information
            user_info: User information
            additional_context: Additional context
            
        Returns:
            True if logged successfully, False otherwise
        """
        formatted_message = self.format_error_message(
            error_type=error_type,
            message=message,
            exception=exception,
            request_data=request_data,
            user_info=user_info,
            additional_context=additional_context
        )
        
        return self.send_message(formatted_message)
    
    def log_critical_error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs
    ) -> bool:
        """Log critical error"""
        return self.log_error("🔴 CRITICAL ERROR", message, exception, **kwargs)
    
    def log_exception(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs
    ) -> bool:
        """Log exception"""
        return self.log_error("⚠️ EXCEPTION", message, exception, **kwargs)
    
    def log_security_alert(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs
    ) -> bool:
        """Log security alert"""
        return self.log_error("🛡️ SECURITY ALERT", message, exception, **kwargs)
    
    def log_database_error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs
    ) -> bool:
        """Log database error"""
        return self.log_error("💾 DATABASE ERROR", message, exception, **kwargs)
    
    def log_api_error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs
    ) -> bool:
        """Log API error"""
        return self.log_error("🔌 API ERROR", message, exception, **kwargs)
    
    def log_celery_error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        task_name: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Log Celery task error"""
        if task_name:
            kwargs.setdefault('additional_context', {})['task_name'] = task_name
        return self.log_error("⚙️ CELERY TASK ERROR", message, exception, **kwargs)
    
    def log_websocket_error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs
    ) -> bool:
        """Log WebSocket error"""
        return self.log_error("🔗 WEBSOCKET ERROR", message, exception, **kwargs)


# Singleton instance
telegram_logger = TelegramLogger()


def log_to_telegram(
    error_type: str,
    message: str,
    exception: Optional[Exception] = None,
    request=None,
    user=None,
    **additional_context
) -> bool:
    """
    Convenience function to log errors to Telegram
    
    Args:
        error_type: Type of error
        message: Error message
        exception: Exception object
        request: Django request object
        user: User object
        **additional_context: Additional context as keyword arguments
        
    Returns:
        True if logged successfully, False otherwise
    """
    request_data = None
    if request:
        request_data = {
            'method': getattr(request, 'method', 'N/A'),
            'path': getattr(request, 'path', 'N/A'),
            'ip': get_client_ip(request),
            'user_agent': request.META.get('HTTP_USER_AGENT', 'N/A') if hasattr(request, 'META') else 'N/A'
        }
    
    user_info = None
    if user and hasattr(user, 'id'):
        user_info = {
            'id': user.id,
            'username': getattr(user, 'username', 'N/A'),
            'email': getattr(user, 'email', 'N/A')
        }
    
    return telegram_logger.log_error(
        error_type=error_type,
        message=message,
        exception=exception,
        request_data=request_data,
        user_info=user_info,
        additional_context=additional_context if additional_context else None
    )


def get_client_ip(request) -> str:
    """Get client IP address from request"""
    if not request:
        return 'N/A'
    
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR', 'N/A')
    return ip

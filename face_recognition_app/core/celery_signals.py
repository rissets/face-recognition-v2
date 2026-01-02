"""
Celery Configuration dengan Telegram Error Monitoring
"""
from celery.signals import task_failure, task_retry
from core.telegram_logger import telegram_logger


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **kw):
    """
    Handler untuk Celery task failure
    Mengirim notifikasi ke Telegram ketika task gagal
    """
    try:
        task_name = sender.name if sender else 'Unknown Task'
        
        additional_context = {
            'task_id': task_id,
            'task_args': str(args)[:200] if args else 'None',
            'task_kwargs': str(kwargs)[:200] if kwargs else 'None',
        }
        
        telegram_logger.log_celery_error(
            message=f"Celery task '{task_name}' failed",
            exception=exception,
            task_name=task_name,
            additional_context=additional_context
        )
    except Exception as e:
        # Don't let notification failure break the system
        pass


@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **kwargs):
    """
    Handler untuk Celery task retry
    Log ke Telegram hanya jika retry count tinggi
    """
    try:
        if sender and hasattr(sender, 'request'):
            retry_count = sender.request.retries
            # Only notify if retry count is high (e.g., > 3)
            if retry_count > 3:
                task_name = sender.name if sender else 'Unknown Task'
                
                telegram_logger.log_celery_error(
                    message=f"Celery task '{task_name}' retrying (attempt {retry_count})",
                    exception=None,
                    task_name=task_name,
                    additional_context={
                        'task_id': task_id,
                        'retry_count': retry_count,
                        'reason': str(reason)[:200] if reason else 'Unknown'
                    }
                )
    except Exception as e:
        pass

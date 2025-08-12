"""
Advanced error recovery and resilience utilities.
"""
import time
import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .logging_config import setup_logging

logger = setup_logging(__name__)


class RetryableError(Exception):
    """Exception that can be retried."""
    pass


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    reraise_on_failure: bool = True,
    log_attempts: bool = True
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between attempts (seconds)
        backoff_factor: Multiplier for delay between attempts
        exceptions: Tuple of exceptions to retry on
        reraise_on_failure: Whether to reraise the last exception after all attempts
        log_attempts: Whether to log retry attempts
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0 and log_attempts:
                        logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        if log_attempts:
                            logger.warning(
                                f"{func.__name__} failed on attempt {attempt + 1}/{max_attempts}: {e}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        if log_attempts:
                            logger.error(
                                f"{func.__name__} failed after {max_attempts} attempts: {e}"
                            )
            
            if reraise_on_failure and last_exception:
                raise last_exception
            
            return None
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker for handling cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == 'open':
                    if time.time() - self.last_failure_time > self.recovery_timeout:
                        self.state = 'half-open'
                        logger.info(f"Circuit breaker for {func.__name__} switching to half-open")
                    else:
                        raise CircuitBreakerError(
                            f"Circuit breaker for {func.__name__} is open"
                        )
                
                try:
                    result = func(*args, **kwargs)
                    
                    if self.state == 'half-open':
                        self.state = 'closed'
                        self.failure_count = 0
                        logger.info(f"Circuit breaker for {func.__name__} recovered")
                    
                    return result
                    
                except self.expected_exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = 'open'
                        logger.error(
                            f"Circuit breaker for {func.__name__} opened after "
                            f"{self.failure_count} failures"
                        )
                    
                    raise e
        
        return wrapper


class GracefulError:
    """Context manager for graceful error handling."""
    
    def __init__(
        self,
        fallback_value: Any = None,
        log_error: bool = True,
        reraise: bool = False,
        cleanup_func: Optional[Callable] = None
    ):
        self.fallback_value = fallback_value
        self.log_error = log_error
        self.reraise = reraise
        self.cleanup_func = cleanup_func
        self.error_occurred = False
        self.exception = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.error_occurred = True
            self.exception = exc_value
            
            if self.log_error:
                logger.error(f"Error in graceful handler: {exc_value}")
                logger.debug("Stack trace:\n" + "".join(traceback.format_exception(
                    exc_type, exc_value, traceback
                )))
            
            if self.cleanup_func:
                try:
                    self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(f"Error in cleanup function: {cleanup_error}")
            
            if not self.reraise:
                return True  # Suppress the exception
        
        return False


class BatchProcessor:
    """Resilient batch processing with error recovery."""
    
    def __init__(
        self,
        max_workers: int = 4,
        error_threshold: float = 0.5,
        retry_failed: bool = True
    ):
        self.max_workers = max_workers
        self.error_threshold = error_threshold
        self.retry_failed = retry_failed
    
    def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, List]:
        """
        Process batch of items with error recovery.
        
        Returns:
            Dictionary with 'success', 'failed', and 'results' lists
        """
        results = []
        failed_items = []
        successful_items = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_func, item): item 
                for item in items
            }
            
            completed = 0
            total = len(items)
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    successful_items.append(item)
                    
                except Exception as e:
                    logger.warning(f"Failed to process item {item}: {e}")
                    failed_items.append((item, e))
                
                if progress_callback:
                    progress_callback(completed, total)
        
        # Check error threshold
        error_rate = len(failed_items) / total
        if error_rate > self.error_threshold:
            logger.error(
                f"Batch processing error rate {error_rate:.2%} exceeds threshold "
                f"{self.error_threshold:.2%}"
            )
        
        # Retry failed items if requested
        if self.retry_failed and failed_items:
            logger.info(f"Retrying {len(failed_items)} failed items...")
            retry_results = self._retry_failed_items(failed_items, process_func)
            results.extend(retry_results['results'])
            successful_items.extend(retry_results['success'])
            failed_items = retry_results['failed']
        
        return {
            'results': results,
            'success': successful_items,
            'failed': failed_items,
            'error_rate': len(failed_items) / total
        }
    
    def _retry_failed_items(
        self,
        failed_items: List[tuple],
        process_func: Callable
    ) -> Dict[str, List]:
        """Retry failed items with simpler approach."""
        results = []
        success = []
        failed = []
        
        for item, original_error in failed_items:
            try:
                # Add small delay to avoid immediate re-failure
                time.sleep(0.1)
                result = process_func(item)
                results.append(result)
                success.append(item)
                
            except Exception as e:
                logger.debug(f"Retry failed for item {item}: {e}")
                failed.append((item, e))
        
        logger.info(
            f"Retry completed: {len(success)} recovered, {len(failed)} still failed"
        )
        
        return {
            'results': results,
            'success': success,
            'failed': failed
        }


@contextmanager
def timeout_context(seconds: float, error_message: str = "Operation timed out"):
    """Context manager with timeout protection."""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(error_message)
    
    # Set the signal handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))
    
    try:
        yield
    finally:
        # Reset the alarm and handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


class ErrorTracker:
    """Track and analyze error patterns."""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self.errors = []
        self._lock = threading.Lock()
    
    def record_error(self, error: Exception, context: str = ""):
        """Record an error with context."""
        with self._lock:
            error_info = {
                'timestamp': time.time(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context,
                'traceback': traceback.format_exc()
            }
            
            self.errors.append(error_info)
            
            # Limit memory usage
            if len(self.errors) > self.max_errors:
                self.errors.pop(0)
    
    def get_error_summary(self, hours: float = 24.0) -> Dict[str, Any]:
        """Get summary of errors in the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_errors = [
            e for e in self.errors 
            if e['timestamp'] > cutoff_time
        ]
        
        if not recent_errors:
            return {'total': 0, 'by_type': {}, 'by_context': {}}
        
        # Count by type
        by_type = {}
        for error in recent_errors:
            error_type = error['error_type']
            by_type[error_type] = by_type.get(error_type, 0) + 1
        
        # Count by context
        by_context = {}
        for error in recent_errors:
            context = error['context'] or 'unknown'
            by_context[context] = by_context.get(context, 0) + 1
        
        return {
            'total': len(recent_errors),
            'by_type': by_type,
            'by_context': by_context,
            'most_recent': recent_errors[-1] if recent_errors else None
        }


# Global error tracker
error_tracker = ErrorTracker()

def track_errors(context: str = ""):
    """Decorator to automatically track errors."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_tracker.record_error(e, context or func.__name__)
                raise
        return wrapper
    return decorator
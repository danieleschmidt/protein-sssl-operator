"""
Comprehensive error handling and recovery utilities for protein-sssl-operator
Provides structured error handling, recovery mechanisms, and user-friendly error messages
"""

import traceback
import logging
import sys
import functools
import time
import threading
from typing import Optional, Dict, Any, Callable, Type, Union, List
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better classification"""
    DATA = "data"
    MODEL = "model"
    TRAINING = "training"
    HARDWARE = "hardware"
    NETWORK = "network"
    MEMORY = "memory"
    CONFIG = "config"
    SECURITY = "security"
    UNKNOWN = "unknown"

@dataclass
class ErrorInfo:
    """Structured error information"""
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: Dict[str, Any]
    suggestions: List[str]
    recoverable: bool
    timestamp: float
    stack_trace: str
    user_message: str

class ProteinSSLError(Exception):
    """Base exception class for protein-sssl-operator"""
    
    def __init__(self,
                 message: str,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 suggestions: Optional[List[str]] = None,
                 recoverable: bool = True,
                 context: Optional[Dict[str, Any]] = None):
        
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.suggestions = suggestions or []
        self.recoverable = recoverable
        self.context = context or {}
        self.timestamp = time.time()

class DataError(ProteinSSLError):
    """Data-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA, **kwargs)

class ModelError(ProteinSSLError):
    """Model-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MODEL, **kwargs)

class TrainingError(ProteinSSLError):
    """Training-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.TRAINING, **kwargs)

class HardwareError(ProteinSSLError):
    """Hardware-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.HARDWARE, **kwargs)

class ConfigError(ProteinSSLError):
    """Configuration-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIG, **kwargs)

class SecurityError(ProteinSSLError):
    """Security-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.SECURITY, severity=ErrorSeverity.HIGH, **kwargs)

class ErrorHandler:
    """Advanced error handling and recovery system"""
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_strategies = {}
        self.error_callbacks = []
        self.lock = threading.Lock()
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies"""
        
        # CUDA out of memory recovery
        self.register_recovery_strategy(
            "CUDA out of memory",
            self._cuda_oom_recovery
        )
        
        # File not found recovery
        self.register_recovery_strategy(
            "No such file or directory",
            self._file_not_found_recovery
        )
        
        # Network timeout recovery
        self.register_recovery_strategy(
            "timeout",
            self._network_timeout_recovery
        )
        
        # Model loading errors
        self.register_recovery_strategy(
            "checkpoint",
            self._checkpoint_recovery
        )
    
    def register_recovery_strategy(self, 
                                 error_pattern: str,
                                 recovery_func: Callable[[Exception], bool]):
        """Register a recovery strategy for specific error patterns"""
        self.recovery_strategies[error_pattern.lower()] = recovery_func
    
    def add_error_callback(self, callback: Callable[[ErrorInfo], None]):
        """Add callback to be called when errors occur"""
        self.error_callbacks.append(callback)
    
    def handle_error(self, 
                    error: Exception,
                    context: Optional[Dict[str, Any]] = None,
                    raise_on_failure: bool = True) -> bool:
        """
        Handle an error with recovery attempts
        
        Args:
            error: The exception that occurred
            context: Additional context information
            raise_on_failure: Whether to re-raise if recovery fails
            
        Returns:
            True if error was handled/recovered, False otherwise
        """
        
        error_info = self._create_error_info(error, context)
        
        # Log the error
        self._log_error(error_info)
        
        # Update error counts
        with self.lock:
            error_key = f"{error_info.error_type}:{error_info.category.value}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
        
        # Attempt recovery if error is recoverable
        if error_info.recoverable:
            recovery_successful = self._attempt_recovery(error, error_info)
            if recovery_successful:
                logger.info(f"Successfully recovered from error: {error_info.error_type}")
                return True
        
        # If we reach here, recovery failed or error is not recoverable
        if raise_on_failure:
            raise error
        
        return False
    
    def _create_error_info(self, 
                          error: Exception,
                          context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """Create structured error information"""
        
        error_type = type(error).__name__
        message = str(error)
        stack_trace = traceback.format_exc()
        
        # Classify error
        category = self._classify_error(error, message)
        severity = self._determine_severity(error, category)
        suggestions = self._generate_suggestions(error, category)
        recoverable = self._is_recoverable(error, category)
        user_message = self._generate_user_message(error, category, suggestions)
        
        return ErrorInfo(
            error_type=error_type,
            message=message,
            category=category,
            severity=severity,
            context=context or {},
            suggestions=suggestions,
            recoverable=recoverable,
            timestamp=time.time(),
            stack_trace=stack_trace,
            user_message=user_message
        )
    
    def _classify_error(self, error: Exception, message: str) -> ErrorCategory:
        """Classify error into categories"""
        
        # Check if it's already a ProteinSSLError
        if isinstance(error, ProteinSSLError):
            return error.category
        
        message_lower = message.lower()
        
        # Data errors
        if any(pattern in message_lower for pattern in [
            'file not found', 'no such file', 'permission denied',
            'invalid format', 'parsing error', 'decode error'
        ]):
            return ErrorCategory.DATA
        
        # Model errors
        if any(pattern in message_lower for pattern in [
            'model', 'parameter', 'state_dict', 'checkpoint',
            'architecture', 'layer', 'forward'
        ]):
            return ErrorCategory.MODEL
        
        # Hardware errors
        if any(pattern in message_lower for pattern in [
            'cuda', 'gpu', 'device', 'out of memory', 'cudnn',
            'cublas', 'nvidia'
        ]):
            return ErrorCategory.HARDWARE
        
        # Memory errors
        if any(pattern in message_lower for pattern in [
            'memory', 'malloc', 'allocation', 'oom'
        ]):
            return ErrorCategory.MEMORY
        
        # Network errors
        if any(pattern in message_lower for pattern in [
            'connection', 'timeout', 'network', 'http', 'url',
            'download', 'upload'
        ]):
            return ErrorCategory.NETWORK
        
        # Configuration errors
        if any(pattern in message_lower for pattern in [
            'config', 'parameter', 'argument', 'invalid value',
            'missing', 'required'
        ]):
            return ErrorCategory.CONFIG
        
        # Training errors
        if any(pattern in message_lower for pattern in [
            'training', 'loss', 'gradient', 'optimizer',
            'backprop', 'nan', 'inf'
        ]):
            return ErrorCategory.TRAINING
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        
        # Critical errors
        if isinstance(error, (SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        if category == ErrorCategory.SECURITY:
            return ErrorSeverity.HIGH
        
        # High severity errors
        if category in [ErrorCategory.HARDWARE, ErrorCategory.MEMORY]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in [ErrorCategory.MODEL, ErrorCategory.TRAINING]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if category in [ErrorCategory.DATA, ErrorCategory.CONFIG]:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _generate_suggestions(self, error: Exception, category: ErrorCategory) -> List[str]:
        """Generate helpful suggestions for error resolution"""
        
        suggestions = []
        message = str(error).lower()
        
        # Data error suggestions
        if category == ErrorCategory.DATA:
            if 'file not found' in message:
                suggestions.extend([
                    "Check if the file path is correct",
                    "Ensure the file exists and is readable",
                    "Verify file permissions"
                ])
            elif 'format' in message:
                suggestions.extend([
                    "Check if the file format is correct (FASTA, PDB, etc.)",
                    "Verify the file is not corrupted",
                    "Try with a different file"
                ])
        
        # Hardware error suggestions
        elif category == ErrorCategory.HARDWARE:
            if 'cuda out of memory' in message:
                suggestions.extend([
                    "Reduce batch size",
                    "Enable gradient checkpointing",
                    "Use mixed precision training",
                    "Clear CUDA cache with torch.cuda.empty_cache()"
                ])
            elif 'cuda' in message:
                suggestions.extend([
                    "Check CUDA installation",
                    "Verify GPU compatibility",
                    "Try running on CPU instead"
                ])
        
        # Model error suggestions
        elif category == ErrorCategory.MODEL:
            if 'checkpoint' in message:
                suggestions.extend([
                    "Check if the checkpoint file exists",
                    "Verify checkpoint compatibility with current model",
                    "Try loading without strict mode"
                ])
            elif 'parameter' in message:
                suggestions.extend([
                    "Check model architecture matches expected",
                    "Verify parameter names and shapes",
                    "Check if model was saved correctly"
                ])
        
        # Training error suggestions
        elif category == ErrorCategory.TRAINING:
            if 'nan' in message or 'inf' in message:
                suggestions.extend([
                    "Reduce learning rate",
                    "Add gradient clipping",
                    "Check for numerical instabilities in loss function",
                    "Use mixed precision with care"
                ])
        
        # Configuration error suggestions
        elif category == ErrorCategory.CONFIG:
            suggestions.extend([
                "Check configuration file syntax",
                "Verify all required parameters are provided",
                "Check parameter value ranges and types"
            ])
        
        # General suggestions
        suggestions.extend([
            "Check the logs for more detailed information",
            "Try running in debug mode for more details"
        ])
        
        return suggestions
    
    def _is_recoverable(self, error: Exception, category: ErrorCategory) -> bool:
        """Determine if error is recoverable"""
        
        # Non-recoverable errors
        if isinstance(error, (SystemExit, KeyboardInterrupt, SystemError)):
            return False
        
        if category == ErrorCategory.SECURITY:
            return False
        
        # Most other errors are potentially recoverable
        return True
    
    def _generate_user_message(self, 
                             error: Exception,
                             category: ErrorCategory,
                             suggestions: List[str]) -> str:
        """Generate user-friendly error message"""
        
        error_type = type(error).__name__
        
        if category == ErrorCategory.DATA:
            user_msg = "There was a problem with your data file."
        elif category == ErrorCategory.HARDWARE:
            user_msg = "There was a hardware-related issue."
        elif category == ErrorCategory.MODEL:
            user_msg = "There was a problem with the model."
        elif category == ErrorCategory.TRAINING:
            user_msg = "There was a problem during training."
        elif category == ErrorCategory.CONFIG:
            user_msg = "There was a problem with the configuration."
        else:
            user_msg = f"An error occurred: {error_type}"
        
        if suggestions:
            user_msg += f"\n\nSuggested solutions:\n" + \
                       "\n".join(f"• {suggestion}" for suggestion in suggestions[:3])
        
        return user_msg
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error information"""
        
        log_level = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[error_info.severity]
        
        logger.log(
            log_level,
            f"{error_info.category.value.upper()} ERROR: {error_info.message}",
            extra={
                'error_type': error_info.error_type,
                'category': error_info.category.value,
                'severity': error_info.severity.value,
                'suggestions': error_info.suggestions,
                'context': error_info.context
            }
        )
        
        # Log stack trace for high/critical errors
        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.debug(f"Stack trace:\n{error_info.stack_trace}")
    
    def _attempt_recovery(self, error: Exception, error_info: ErrorInfo) -> bool:
        """Attempt to recover from error"""
        
        error_message = str(error).lower()
        
        # Try registered recovery strategies
        for pattern, recovery_func in self.recovery_strategies.items():
            if pattern in error_message:
                try:
                    logger.info(f"Attempting recovery using strategy: {pattern}")
                    success = recovery_func(error)
                    if success:
                        return True
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")
        
        return False
    
    def _cuda_oom_recovery(self, error: Exception) -> bool:
        """Recovery strategy for CUDA out of memory errors"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared CUDA cache")
                return True
        except Exception as e:
            logger.error(f"CUDA cache clearing failed: {e}")
        
        return False
    
    def _file_not_found_recovery(self, error: Exception) -> bool:
        """Recovery strategy for file not found errors"""
        # This would need specific implementation based on the use case
        # For now, just log the attempt
        logger.info("Attempting file recovery (placeholder)")
        return False
    
    def _network_timeout_recovery(self, error: Exception) -> bool:
        """Recovery strategy for network timeout errors"""
        logger.info("Network timeout - could retry with exponential backoff")
        return False
    
    def _checkpoint_recovery(self, error: Exception) -> bool:
        """Recovery strategy for checkpoint loading errors"""
        logger.info("Checkpoint recovery - could try loading without strict mode")
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self.lock:
            total_errors = sum(self.error_counts.values())
            return {
                'total_errors': total_errors,
                'error_counts': dict(self.error_counts),
                'most_common_error': max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
            }

# Global error handler instance
_global_error_handler = ErrorHandler()

def handle_error(error: Exception,
                context: Optional[Dict[str, Any]] = None,
                raise_on_failure: bool = True) -> bool:
    """Handle error using global error handler"""
    return _global_error_handler.handle_error(error, context, raise_on_failure)

def with_error_handling(raise_on_failure: bool = True,
                       context: Optional[Dict[str, Any]] = None):
    """Decorator to add error handling to functions"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                if context:
                    func_context.update(context)
                
                handled = handle_error(e, func_context, raise_on_failure)
                if not handled and not raise_on_failure:
                    return None
        
        return wrapper
    return decorator

@contextmanager
def error_context(context: Dict[str, Any]):
    """Context manager to provide error context"""
    try:
        yield
    except Exception as e:
        handle_error(e, context)

class SafeExecutor:
    """Safe execution wrapper with retry logic"""
    
    def __init__(self,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
    
    def execute(self,
               func: Callable,
               *args,
               retry_on: Optional[List[Type[Exception]]] = None,
               **kwargs) -> Any:
        """Execute function with retry logic"""
        
        retry_on = retry_on or [Exception]
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                # Check if we should retry on this error
                if not any(isinstance(e, error_type) for error_type in retry_on):
                    break
                
                # Don't retry on final attempt
                if attempt == self.max_retries:
                    break
                
                # Wait before retry
                if attempt > 0:
                    delay = self.retry_delay * (self.backoff_factor ** (attempt - 1))
                    logger.info(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(delay)
        
        # If we get here, all retries failed
        raise last_error

def safe_execute(func: Callable,
                *args,
                max_retries: int = 3,
                retry_delay: float = 1.0,
                **kwargs) -> Any:
    """Safe execution with retries"""
    executor = SafeExecutor(max_retries=max_retries, retry_delay=retry_delay)
    return executor.execute(func, *args, **kwargs)

# Convenience functions
def register_recovery_strategy(error_pattern: str, recovery_func: Callable):
    """Register recovery strategy globally"""
    _global_error_handler.register_recovery_strategy(error_pattern, recovery_func)

def add_error_callback(callback: Callable):
    """Add error callback globally"""
    _global_error_handler.add_error_callback(callback)

def get_error_statistics() -> Dict[str, Any]:
    """Get global error statistics"""
    return _global_error_handler.get_error_statistics()
"""
Advanced logging configuration for protein-sssl-operator
Provides structured logging with performance monitoring and security features
"""

import logging
import logging.handlers
import sys
import json
import time
import traceback
import os
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
import functools
import inspect

class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs"""
    
    def __init__(self):
        super().__init__()
        # Patterns that might contain sensitive information
        self.sensitive_patterns = [
            'password', 'token', 'key', 'secret', 'auth',
            'credential', 'api_key', 'access_token'
        ]
    
    def filter(self, record):
        """Filter sensitive information from log records"""
        if hasattr(record, 'msg'):
            msg_lower = str(record.msg).lower()
            for pattern in self.sensitive_patterns:
                if pattern in msg_lower:
                    record.msg = "[SENSITIVE DATA FILTERED]"
                    break
        
        # Also filter from args
        if hasattr(record, 'args') and record.args:
            filtered_args = []
            for arg in record.args:
                arg_str = str(arg).lower()
                is_sensitive = any(pattern in arg_str for pattern in self.sensitive_patterns)
                filtered_args.append("[FILTERED]" if is_sensitive else arg)
            record.args = tuple(filtered_args)
        
        return True

class PerformanceFormatter(logging.Formatter):
    """Custom formatter that includes performance metrics"""
    
    def __init__(self, include_performance: bool = True):
        super().__init__()
        self.include_performance = include_performance
        self.start_time = time.time()
    
    def format(self, record):
        """Format log record with performance information"""
        
        # Add timestamp
        record.timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Add performance metrics
        if self.include_performance:
            record.elapsed = time.time() - self.start_time
            record.thread_id = threading.get_ident()
            record.process_id = os.getpid()
        
        # Add context information
        record.module = record.module if hasattr(record, 'module') else record.name
        record.function = record.funcName
        record.line = record.lineno
        
        # Format message
        if self.include_performance:
            fmt = (
                "{timestamp} | {levelname:8} | {process_id}:{thread_id} | "
                "{elapsed:8.3f}s | {module}:{function}:{line} | {message}"
            )
        else:
            fmt = "{timestamp} | {levelname:8} | {module}:{function}:{line} | {message}"
        
        return fmt.format(**record.__dict__)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format record as JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': threading.get_ident(),
            'process_id': os.getpid()
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class ProteinSSLLogger:
    """Advanced logger for protein-sssl-operator with security and performance features"""
    
    def __init__(self, 
                 name: str = "protein_sssl",
                 level: str = "INFO",
                 log_dir: Optional[Union[str, Path]] = None,
                 format_type: str = "text",  # "text" or "json"
                 include_performance: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_security_filter: bool = True):
        
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.format_type = format_type
        self.include_performance = include_performance
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_security_filter = enable_security_filter
        
        # Create logs directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Performance tracking
        self._call_times = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with handlers and formatters"""
        
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        
        if self.format_type == "json":
            console_formatter = JSONFormatter()
        else:
            console_formatter = PerformanceFormatter(self.include_performance)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
        # Error file handler
        error_log_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(console_formatter)
        logger.addHandler(error_handler)
        
        # Add security filter if enabled
        if self.enable_security_filter:
            security_filter = SecurityFilter()
            for handler in logger.handlers:
                handler.addFilter(security_filter)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """Get logger instance for a specific module"""
        if module_name:
            return logging.getLogger(f"{self.name}.{module_name}")
        return self.logger
    
    @contextmanager
    def log_performance(self, operation: str, **kwargs):
        """Context manager for logging operation performance"""
        logger = self.get_logger()
        start_time = time.time()
        
        logger.info(f"Starting {operation}", extra=kwargs)
        
        try:
            yield
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(
                f"Completed {operation} in {duration:.3f}s",
                extra={**kwargs, 'duration': duration, 'status': 'success'}
            )
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            logger.error(
                f"Failed {operation} after {duration:.3f}s: {str(e)}",
                extra={**kwargs, 'duration': duration, 'status': 'error', 'error': str(e)},
                exc_info=True
            )
            raise
    
    def log_function_call(self, func):
        """Decorator to log function calls with performance metrics"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = self.get_logger()
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Log function entry
            logger.debug(f"Entering {func_name}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                # Track performance
                if func_name not in self._call_times:
                    self._call_times[func_name] = []
                self._call_times[func_name].append(duration)
                
                logger.debug(
                    f"Exiting {func_name} (duration: {duration:.3f}s)",
                    extra={'function': func_name, 'duration': duration}
                )
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                logger.error(
                    f"Exception in {func_name} after {duration:.3f}s: {str(e)}",
                    extra={'function': func_name, 'duration': duration, 'error': str(e)},
                    exc_info=True
                )
                raise
        
        return wrapper
    
    def log_model_info(self, model, model_name: str = "model"):
        """Log model architecture and parameter information"""
        logger = self.get_logger()
        
        try:
            import torch
            
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                model_info = {
                    'model_name': model_name,
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming fp32
                }
                
                if hasattr(model, '__class__'):
                    model_info['model_class'] = model.__class__.__name__
                
                logger.info(
                    f"Model info - {model_name}: {total_params:,} total params, "
                    f"{trainable_params:,} trainable, {model_info['model_size_mb']:.1f}MB",
                    extra=model_info
                )
            
        except Exception as e:
            logger.warning(f"Could not log model info for {model_name}: {e}")
    
    def log_dataset_info(self, dataset, dataset_name: str = "dataset"):
        """Log dataset information"""
        logger = self.get_logger()
        
        try:
            dataset_info = {
                'dataset_name': dataset_name,
                'size': len(dataset) if hasattr(dataset, '__len__') else 'unknown'
            }
            
            if hasattr(dataset, '__class__'):
                dataset_info['dataset_class'] = dataset.__class__.__name__
            
            # Try to get sample data info
            if len(dataset) > 0:
                try:
                    sample = dataset[0]
                    if isinstance(sample, dict):
                        dataset_info['sample_keys'] = list(sample.keys())
                        
                        # Log tensor shapes if present
                        for key, value in sample.items():
                            if hasattr(value, 'shape'):
                                dataset_info[f'{key}_shape'] = list(value.shape)
                                
                except Exception:
                    pass  # Skip if sample access fails
            
            logger.info(
                f"Dataset info - {dataset_name}: {dataset_info['size']} samples",
                extra=dataset_info
            )
            
        except Exception as e:
            logger.warning(f"Could not log dataset info for {dataset_name}: {e}")
    
    def log_training_metrics(self, 
                           epoch: int, 
                           step: int, 
                           metrics: Dict[str, float],
                           prefix: str = "train"):
        """Log training metrics"""
        logger = self.get_logger()
        
        metrics_info = {
            'epoch': epoch,
            'step': step,
            'prefix': prefix,
            **{f'{prefix}_{key}': value for key, value in metrics.items()}
        }
        
        metrics_str = ', '.join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        
        logger.info(
            f"Epoch {epoch}, Step {step} - {prefix} metrics: {metrics_str}",
            extra=metrics_info
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of function call performance"""
        summary = {}
        
        for func_name, times in self._call_times.items():
            if times:
                summary[func_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        return summary
    
    def log_performance_summary(self):
        """Log performance summary"""
        logger = self.get_logger()
        summary = self.get_performance_summary()
        
        if summary:
            logger.info("Performance Summary:")
            for func_name, stats in summary.items():
                logger.info(
                    f"  {func_name}: {stats['count']} calls, "
                    f"avg: {stats['avg_time']:.3f}s, "
                    f"total: {stats['total_time']:.3f}s",
                    extra={'performance_summary': stats, 'function': func_name}
                )

# Global logger instance
_global_logger = None

def get_logger(name: str = "protein_sssl", **kwargs) -> ProteinSSLLogger:
    """Get global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = ProteinSSLLogger(name, **kwargs)
    
    return _global_logger

def setup_logging(level: str = "INFO", 
                 log_dir: Optional[str] = None,
                 format_type: str = "text",
                 include_performance: bool = True) -> ProteinSSLLogger:
    """Setup global logging configuration"""
    
    global _global_logger
    _global_logger = ProteinSSLLogger(
        level=level,
        log_dir=log_dir,
        format_type=format_type,
        include_performance=include_performance
    )
    
    return _global_logger

# Convenience functions
def log_performance(operation: str, **kwargs):
    """Context manager for performance logging"""
    return get_logger().log_performance(operation, **kwargs)

def log_function_call(func):
    """Decorator for function call logging"""
    return get_logger().log_function_call(func)

def log_model_info(model, model_name: str = "model"):
    """Log model information"""
    get_logger().log_model_info(model, model_name)

def log_dataset_info(dataset, dataset_name: str = "dataset"):
    """Log dataset information"""
    get_logger().log_dataset_info(dataset, dataset_name)
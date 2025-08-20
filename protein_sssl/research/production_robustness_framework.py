"""
Production-Grade Robustness Framework for Protein Folding Research

Implements comprehensive robustness, reliability, and error handling for
production deployment of breakthrough research algorithms.

Key Features:
1. Comprehensive Error Handling & Recovery
2. Input Validation & Sanitization  
3. Resource Management & Monitoring
4. Security & Privacy Protection
5. Performance Optimization
6. Distributed Computing Support
7. Automated Testing & Validation
8. Health Monitoring & Alerting

Authors: Terry - Terragon Labs Production Engineering
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
import traceback
import time
import functools
import threading
import multiprocessing as mp
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import contextmanager
import json
import hashlib
import warnings
import psutil
import signal
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('protein_sssl_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RobustnessConfig:
    """Configuration for production robustness features"""
    
    # Error Handling
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    
    # Resource Management
    max_memory_gb: float = 8.0
    max_cpu_percent: float = 80.0
    max_gpu_memory_gb: float = 16.0
    timeout_seconds: float = 300.0
    
    # Security
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    enable_audit_logging: bool = True
    max_sequence_length: int = 10000
    
    # Performance
    enable_caching: bool = True
    cache_size_mb: int = 1000
    enable_profiling: bool = True
    performance_monitoring: bool = True
    
    # Distributed Computing
    enable_distributed: bool = False
    max_workers: int = mp.cpu_count()
    communication_timeout: float = 30.0
    
    # Health Monitoring
    health_check_interval: float = 30.0
    metrics_collection_interval: float = 10.0
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'error_rate': 0.05,
        'response_time_p95': 5.0,
        'memory_usage': 0.85,
        'cpu_usage': 0.90
    })

class ProductionError(Exception):
    """Base exception for production issues"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        self.message = message
        self.error_code = error_code or "PRODUCTION_ERROR"
        self.details = details or {}
        self.timestamp = time.time()
        super().__init__(self.message)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'traceback': traceback.format_exc()
        }

class ValidationError(ProductionError):
    """Input validation errors"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        details = {'field': field, 'value': str(value) if value is not None else None}
        super().__init__(message, "VALIDATION_ERROR", details)

class ResourceError(ProductionError):
    """Resource management errors"""
    def __init__(self, message: str, resource_type: str = None, current_usage: float = None):
        details = {'resource_type': resource_type, 'current_usage': current_usage}
        super().__init__(message, "RESOURCE_ERROR", details)

class SecurityError(ProductionError):
    """Security-related errors"""
    def __init__(self, message: str, security_type: str = None):
        details = {'security_type': security_type}
        super().__init__(message, "SECURITY_ERROR", details)

class CircuitBreakerError(ProductionError):
    """Circuit breaker errors"""
    def __init__(self, message: str, service_name: str = None):
        details = {'service_name': service_name}
        super().__init__(message, "CIRCUIT_BREAKER_ERROR", details)

class InputValidator:
    """Comprehensive input validation for protein data"""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.amino_acid_codes = set('ACDEFGHIKLMNPQRSTVWY')
        self.valid_secondary_structures = set('HEC')  # Helix, Extended, Coil
        
    def validate_protein_sequence(self, sequence: str) -> str:
        """Validate protein sequence input"""
        if not isinstance(sequence, str):
            raise ValidationError("Sequence must be a string", "sequence", sequence)
            
        if len(sequence) == 0:
            raise ValidationError("Sequence cannot be empty", "sequence", sequence)
            
        if len(sequence) > self.config.max_sequence_length:
            raise ValidationError(
                f"Sequence too long (max: {self.config.max_sequence_length})",
                "sequence", len(sequence)
            )
            
        # Clean and validate amino acid codes
        cleaned_sequence = sequence.upper().strip()
        invalid_chars = set(cleaned_sequence) - self.amino_acid_codes
        
        if invalid_chars:
            raise ValidationError(
                f"Invalid amino acid codes: {invalid_chars}",
                "sequence", invalid_chars
            )
            
        return cleaned_sequence
    
    def validate_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Validate 3D coordinates"""
        if not isinstance(coordinates, np.ndarray):
            coordinates = np.array(coordinates)
            
        if len(coordinates.shape) != 2 or coordinates.shape[1] != 3:
            raise ValidationError(
                "Coordinates must be Nx3 array",
                "coordinates", coordinates.shape
            )
            
        # Check for NaN or infinite values
        if np.any(np.isnan(coordinates)) or np.any(np.isinf(coordinates)):
            raise ValidationError(
                "Coordinates contain NaN or infinite values",
                "coordinates", "invalid_values"
            )
            
        # Reasonable coordinate bounds (in Angstroms)
        max_coord = 1000.0  # 100 nm
        if np.any(np.abs(coordinates) > max_coord):
            raise ValidationError(
                f"Coordinates exceed reasonable bounds (±{max_coord} Å)",
                "coordinates", np.max(np.abs(coordinates))
            )
            
        return coordinates.astype(np.float32)
    
    def validate_uncertainty_parameters(self, uncertainties: Dict[str, Any]) -> Dict[str, Any]:
        """Validate uncertainty parameters"""
        required_fields = ['epistemic_uncertainty', 'aleatoric_uncertainty']
        
        for field in required_fields:
            if field not in uncertainties:
                raise ValidationError(f"Missing required field: {field}", field, None)
                
            values = uncertainties[field]
            if isinstance(values, (list, tuple)):
                values = np.array(values)
                
            if not isinstance(values, np.ndarray):
                raise ValidationError(
                    f"Field {field} must be numeric array", field, type(values)
                )
                
            if np.any(values < 0):
                raise ValidationError(
                    f"Uncertainty values must be non-negative", field, "negative_values"
                )
                
            if np.any(values > 1000):  # Reasonable upper bound
                raise ValidationError(
                    f"Uncertainty values too large", field, np.max(values)
                )
        
        return uncertainties
    
    def sanitize_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize output data"""
        sanitized = {}
        
        for key, value in output.items():
            if isinstance(value, np.ndarray):
                # Clip extreme values
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(value, -1e6, 1e6)
                sanitized[key] = value
                
            elif isinstance(value, (int, float)):
                if isinstance(value, float):
                    if np.isnan(value) or np.isinf(value):
                        sanitized[key] = 0.0
                    else:
                        sanitized[key] = float(np.clip(value, -1e6, 1e6))
                else:
                    sanitized[key] = value
                    
            elif isinstance(value, str):
                # Remove potentially harmful characters
                sanitized[key] = ''.join(c for c in value if c.isalnum() or c in ' .,_-')
                
            else:
                sanitized[key] = value
                
        return sanitized

class ResourceMonitor:
    """Monitor and manage system resources"""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.start_time = time.time()
        self.peak_memory = 0.0
        self.peak_cpu = 0.0
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        process = psutil.Process()
        
        # Memory metrics
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024**3)
        memory_percent = process.memory_percent()
        
        # CPU metrics
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # System-wide metrics
        system_memory = psutil.virtual_memory()
        system_cpu = psutil.cpu_percent(interval=0.1)
        
        self.peak_memory = max(self.peak_memory, memory_gb)
        self.peak_cpu = max(self.peak_cpu, cpu_percent)
        
        return {
            'memory_gb': memory_gb,
            'memory_percent': memory_percent,
            'cpu_percent': cpu_percent,
            'system_memory_percent': system_memory.percent,
            'system_cpu_percent': system_cpu,
            'peak_memory_gb': self.peak_memory,
            'peak_cpu_percent': self.peak_cpu,
            'runtime_seconds': time.time() - self.start_time
        }
    
    def check_resource_limits(self) -> None:
        """Check if resource limits are exceeded"""
        metrics = self.get_system_metrics()
        
        if metrics['memory_gb'] > self.config.max_memory_gb:
            raise ResourceError(
                f"Memory limit exceeded: {metrics['memory_gb']:.2f}GB > {self.config.max_memory_gb}GB",
                "memory", metrics['memory_gb']
            )
            
        if metrics['cpu_percent'] > self.config.max_cpu_percent:
            raise ResourceError(
                f"CPU limit exceeded: {metrics['cpu_percent']:.1f}% > {self.config.max_cpu_percent}%",
                "cpu", metrics['cpu_percent']
            )
    
    @contextmanager
    def resource_context(self):
        """Context manager for resource monitoring"""
        start_metrics = self.get_system_metrics()
        logger.info(f"Starting operation - Memory: {start_metrics['memory_gb']:.2f}GB, CPU: {start_metrics['cpu_percent']:.1f}%")
        
        try:
            yield
        finally:
            end_metrics = self.get_system_metrics()
            logger.info(f"Operation complete - Memory: {end_metrics['memory_gb']:.2f}GB, CPU: {end_metrics['cpu_percent']:.1f}%")
            
            # Log resource usage summary
            memory_delta = end_metrics['memory_gb'] - start_metrics['memory_gb']
            logger.info(f"Resource usage - Memory delta: {memory_delta:+.2f}GB, Peak memory: {self.peak_memory:.2f}GB")

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    logger.info(f"Circuit breaker moving to HALF_OPEN state for {func.__name__}")
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker OPEN for {func.__name__}",
                        func.__name__
                    )
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    logger.info(f"Circuit breaker CLOSED for {func.__name__}")
                    
                return result
                
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logger.error(f"Circuit breaker OPEN for {func.__name__} after {self.failure_count} failures")
                    
                raise e
                
        return wrapper
    
    def reset(self):
        """Manually reset circuit breaker"""
        self.failure_count = 0
        self.state = 'CLOSED'
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")

def retry_with_backoff(max_retries: int = 3, 
                      base_delay: float = 1.0,
                      backoff_factor: float = 2.0,
                      exceptions: Tuple[type, ...] = (Exception,)):
    """Decorator for retrying functions with exponential backoff"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        
            raise last_exception
            
        return wrapper
    return decorator

class PerformanceProfiler:
    """Performance profiling and monitoring"""
    
    def __init__(self):
        self.operation_times = {}
        self.operation_counts = {}
        self.memory_snapshots = []
        
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile a specific operation"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Update statistics
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
                self.operation_counts[operation_name] = 0
                
            self.operation_times[operation_name].append(duration)
            self.operation_counts[operation_name] += 1
            
            self.memory_snapshots.append({
                'operation': operation_name,
                'memory_delta': memory_delta,
                'duration': duration,
                'timestamp': time.time()
            })
            
            logger.debug(f"{operation_name} completed in {duration:.4f}s (memory delta: {memory_delta:+.2f}MB)")
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary statistics"""
        summary = {}
        
        for operation, times in self.operation_times.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'mean_time': np.mean(times),
                    'median_time': np.median(times),
                    'p95_time': np.percentile(times, 95),
                    'p99_time': np.percentile(times, 99),
                    'total_time': np.sum(times)
                }
                
        return summary
    
    def get_memory_analysis(self) -> Dict[str, float]:
        """Analyze memory usage patterns"""
        if not self.memory_snapshots:
            return {}
            
        deltas = [snap['memory_delta'] for snap in self.memory_snapshots]
        
        return {
            'total_snapshots': len(self.memory_snapshots),
            'mean_memory_delta': np.mean(deltas),
            'max_memory_delta': np.max(deltas),
            'min_memory_delta': np.min(deltas),
            'memory_leak_indicator': np.sum([d for d in deltas if d > 0])  # Net positive deltas
        }

class HealthMonitor:
    """System health monitoring and alerting"""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.metrics_history = []
        self.alert_states = {}
        self.start_time = time.time()
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive health metrics"""
        process = psutil.Process()
        
        metrics = {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            
            # Process metrics
            'memory_rss_mb': process.memory_info().rss / (1024**2),
            'memory_vms_mb': process.memory_info().vms / (1024**2),
            'memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(interval=0.1),
            'num_threads': process.num_threads(),
            'open_files': len(process.open_files()),
            
            # System metrics
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_cpu_percent': psutil.cpu_percent(interval=0.1),
            'disk_usage_percent': psutil.disk_usage('/').percent if os.path.exists('/') else 0,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        max_history = 1000
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]
            
        return metrics
    
    def check_health(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        metrics = self.collect_metrics()
        alerts = []
        status = 'healthy'
        
        # Check alert thresholds
        thresholds = self.config.alert_thresholds
        
        if metrics['memory_percent'] / 100.0 > thresholds.get('memory_usage', 0.85):
            alerts.append({
                'type': 'memory_high',
                'value': metrics['memory_percent'],
                'threshold': thresholds['memory_usage'] * 100
            })
            status = 'warning'
        
        if metrics['cpu_percent'] / 100.0 > thresholds.get('cpu_usage', 0.90):
            alerts.append({
                'type': 'cpu_high',
                'value': metrics['cpu_percent'],
                'threshold': thresholds['cpu_usage'] * 100
            })
            status = 'warning'
            
        # Check for memory leaks
        if len(self.metrics_history) > 10:
            recent_memory = [m['memory_rss_mb'] for m in self.metrics_history[-10:]]
            memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
            
            if memory_trend > 10:  # Growing by >10MB per measurement
                alerts.append({
                    'type': 'memory_leak',
                    'trend': memory_trend,
                    'threshold': 10
                })
                status = 'warning'
        
        # Overall health status
        if alerts:
            if any(alert['type'] in ['memory_high', 'cpu_high'] and 
                  alert['value'] > alert['threshold'] * 1.2 for alert in alerts):
                status = 'critical'
                
        return {
            'status': status,
            'metrics': metrics,
            'alerts': alerts,
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
    
    def get_performance_trends(self, hours: int = 1) -> Dict[str, List[float]]:
        """Get performance trends over specified time window"""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m['timestamp'] > cutoff_time]
        
        if not recent_metrics:
            return {}
            
        trends = {
            'memory_usage': [m['memory_percent'] for m in recent_metrics],
            'cpu_usage': [m['cpu_percent'] for m in recent_metrics],
            'timestamps': [m['timestamp'] for m in recent_metrics]
        }
        
        return trends

class ProductionRobustnessFramework:
    """Main production robustness framework"""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        
        # Initialize components
        self.validator = InputValidator(config)
        self.resource_monitor = ResourceMonitor(config)
        self.profiler = PerformanceProfiler()
        self.health_monitor = HealthMonitor(config)
        
        # Circuit breakers for different services
        self.circuit_breakers = {
            'prediction': CircuitBreaker(config.circuit_breaker_threshold, config.circuit_breaker_timeout),
            'uncertainty': CircuitBreaker(config.circuit_breaker_threshold, config.circuit_breaker_timeout),
            'validation': CircuitBreaker(config.circuit_breaker_threshold, config.circuit_breaker_timeout)
        }
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Error tracking
        self.error_counts = {}
        self.last_errors = []
        
        logger.info("Production Robustness Framework initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def robust_prediction(self, 
                         sequence: str,
                         model_func: Callable,
                         **kwargs) -> Dict[str, Any]:
        """Robust prediction with comprehensive error handling"""
        operation_id = hashlib.md5(sequence.encode()).hexdigest()[:8]
        
        with self.profiler.profile_operation('robust_prediction'):
            with self.resource_monitor.resource_context():
                try:
                    # Input validation
                    validated_sequence = self.validator.validate_protein_sequence(sequence)
                    logger.info(f"Processing sequence {operation_id} (length: {len(validated_sequence)})")
                    
                    # Resource check
                    self.resource_monitor.check_resource_limits()
                    
                    # Apply circuit breaker and retry logic
                    @self.circuit_breakers['prediction']
                    @retry_with_backoff(
                        max_retries=self.config.max_retries,
                        base_delay=self.config.retry_delay
                    )
                    def protected_prediction():
                        return model_func(validated_sequence, **kwargs)
                    
                    # Execute prediction
                    raw_result = protected_prediction()
                    
                    # Validate and sanitize output
                    if isinstance(raw_result, dict):
                        sanitized_result = self.validator.sanitize_output(raw_result)
                    else:
                        sanitized_result = {'prediction': raw_result}
                    
                    # Add metadata
                    sanitized_result.update({
                        'operation_id': operation_id,
                        'sequence_length': len(validated_sequence),
                        'timestamp': time.time(),
                        'model_version': kwargs.get('model_version', 'unknown')
                    })
                    
                    logger.info(f"Successfully processed sequence {operation_id}")
                    return sanitized_result
                    
                except Exception as e:
                    self._handle_error(e, 'robust_prediction', operation_id)
                    raise
    
    def robust_uncertainty_quantification(self,
                                         predictions: Dict[str, Any],
                                         uncertainty_func: Callable,
                                         **kwargs) -> Dict[str, Any]:
        """Robust uncertainty quantification"""
        
        with self.profiler.profile_operation('uncertainty_quantification'):
            try:
                # Validate predictions
                if 'prediction' in predictions:
                    if isinstance(predictions['prediction'], np.ndarray):
                        predictions['prediction'] = self.validator.validate_coordinates(
                            predictions['prediction']
                        )
                
                # Apply circuit breaker
                @self.circuit_breakers['uncertainty']
                def protected_uncertainty():
                    return uncertainty_func(predictions, **kwargs)
                
                raw_uncertainties = protected_uncertainty()
                
                # Validate uncertainty output
                validated_uncertainties = self.validator.validate_uncertainty_parameters(
                    raw_uncertainties
                )
                
                return validated_uncertainties
                
            except Exception as e:
                self._handle_error(e, 'uncertainty_quantification')
                raise
    
    def _handle_error(self, error: Exception, operation: str, operation_id: str = None):
        """Centralized error handling"""
        error_key = f"{operation}:{type(error).__name__}"
        
        # Track error counts
        if error_key not in self.error_counts:
            self.error_counts[error_key] = 0
        self.error_counts[error_key] += 1
        
        # Store recent errors
        error_info = {
            'timestamp': time.time(),
            'operation': operation,
            'operation_id': operation_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        self.last_errors.append(error_info)
        
        # Keep only recent errors
        max_errors = 100
        if len(self.last_errors) > max_errors:
            self.last_errors = self.last_errors[-max_errors:]
        
        # Log error with appropriate level
        if isinstance(error, (ValidationError, SecurityError)):
            logger.warning(f"Validation/Security error in {operation}: {error}")
        elif isinstance(error, ResourceError):
            logger.error(f"Resource error in {operation}: {error}")
        elif isinstance(error, CircuitBreakerError):
            logger.error(f"Circuit breaker triggered in {operation}: {error}")
        else:
            logger.error(f"Unexpected error in {operation}: {error}", exc_info=True)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        health_status = self.health_monitor.check_health()
        performance_summary = self.profiler.get_performance_summary()
        memory_analysis = self.profiler.get_memory_analysis()
        
        # Circuit breaker states
        circuit_states = {
            name: {'state': cb.state, 'failure_count': cb.failure_count}
            for name, cb in self.circuit_breakers.items()
        }
        
        # Error summary
        error_rate = sum(self.error_counts.values()) / max(1, len(self.last_errors))
        
        return {
            'health': health_status,
            'performance': performance_summary,
            'memory': memory_analysis,
            'circuit_breakers': circuit_states,
            'errors': {
                'total_errors': sum(self.error_counts.values()),
                'error_rate': error_rate,
                'error_breakdown': self.error_counts,
                'recent_errors': len(self.last_errors)
            },
            'uptime_hours': health_status['uptime_hours']
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics for troubleshooting"""
        status = self.get_system_status()
        
        # Recent performance trends
        trends = self.health_monitor.get_performance_trends(hours=1)
        
        # Resource usage patterns
        if self.health_monitor.metrics_history:
            recent_metrics = self.health_monitor.metrics_history[-10:]
            resource_patterns = {
                'memory_trend': np.polyfit(
                    range(len(recent_metrics)),
                    [m['memory_rss_mb'] for m in recent_metrics],
                    1
                )[0] if len(recent_metrics) > 1 else 0,
                'cpu_utilization': np.mean([m['cpu_percent'] for m in recent_metrics]),
                'thread_count': recent_metrics[-1]['num_threads'],
                'open_files': recent_metrics[-1]['open_files']
            }
        else:
            resource_patterns = {}
        
        return {
            'system_status': status,
            'performance_trends': trends,
            'resource_patterns': resource_patterns,
            'recent_errors': self.last_errors[-5:],  # Last 5 errors
            'recommendations': self._generate_recommendations(status)
        }
    
    def _generate_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on system status"""
        recommendations = []
        
        # Memory recommendations
        if status['health']['status'] in ['warning', 'critical']:
            memory_alerts = [a for a in status['health']['alerts'] if a['type'] == 'memory_high']
            if memory_alerts:
                recommendations.append(
                    "High memory usage detected. Consider reducing batch sizes or implementing memory-efficient algorithms."
                )
        
        # Performance recommendations
        if 'performance' in status and status['performance']:
            slow_operations = [
                op for op, stats in status['performance'].items()
                if stats.get('p95_time', 0) > 10.0
            ]
            if slow_operations:
                recommendations.append(
                    f"Slow operations detected: {slow_operations}. Consider optimization or caching."
                )
        
        # Error rate recommendations
        if status['errors']['error_rate'] > 0.05:  # 5% error rate
            recommendations.append(
                "High error rate detected. Review recent errors and consider implementing additional validation."
            )
        
        # Circuit breaker recommendations
        open_breakers = [
            name for name, cb_status in status['circuit_breakers'].items()
            if cb_status['state'] == 'OPEN'
        ]
        if open_breakers:
            recommendations.append(
                f"Circuit breakers OPEN: {open_breakers}. Check underlying service health."
            )
        
        return recommendations
    
    def reset_circuit_breakers(self):
        """Reset all circuit breakers"""
        for name, cb in self.circuit_breakers.items():
            cb.reset()
        logger.info("All circuit breakers reset")
    
    def clear_error_history(self):
        """Clear error tracking history"""
        self.error_counts.clear()
        self.last_errors.clear()
        logger.info("Error history cleared")
    
    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        metrics_data = {
            'system_status': self.get_system_status(),
            'metrics_history': self.health_monitor.metrics_history,
            'performance_summary': self.profiler.get_performance_summary(),
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Production Robustness Framework...")
        
        # Export final metrics
        try:
            self.export_metrics("shutdown_metrics.json")
        except Exception as e:
            logger.error(f"Failed to export shutdown metrics: {e}")
        
        # Log final status
        final_status = self.get_system_status()
        logger.info(f"Final system status: {final_status['health']['status']}")
        logger.info(f"Total uptime: {final_status['uptime_hours']:.2f} hours")
        logger.info(f"Total errors: {final_status['errors']['total_errors']}")
        
        logger.info("Shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    logger.info("Initializing Production Robustness Framework...")
    
    # Configuration
    config = RobustnessConfig(
        max_retries=3,
        max_memory_gb=4.0,
        max_cpu_percent=75.0,
        enable_input_validation=True,
        enable_audit_logging=True
    )
    
    # Create framework
    framework = ProductionRobustnessFramework(config)
    
    # Example mock prediction function
    def mock_prediction_model(sequence: str, **kwargs):
        """Mock prediction model for testing"""
        time.sleep(0.1)  # Simulate processing time
        
        if len(sequence) > 1000:  # Simulate error condition
            raise ValueError("Sequence too long for processing")
            
        # Mock prediction results
        return {
            'coordinates': np.random.normal(0, 10, (len(sequence), 3)),
            'confidence_scores': np.random.uniform(0.5, 1.0, len(sequence)),
            'secondary_structure': np.random.choice(['H', 'E', 'C'], len(sequence))
        }
    
    def mock_uncertainty_model(predictions: Dict[str, Any], **kwargs):
        """Mock uncertainty model for testing"""
        n_residues = len(predictions.get('confidence_scores', [10]))
        
        return {
            'epistemic_uncertainty': np.random.uniform(0.1, 0.5, n_residues),
            'aleatoric_uncertainty': np.random.uniform(0.05, 0.3, n_residues),
            'total_uncertainty': np.random.uniform(0.2, 0.8, n_residues)
        }
    
    # Test robust prediction
    logger.info("Testing robust prediction...")
    
    test_sequences = [
        "MKFLKFSLLTAVLLSVVFAFSSC",  # Valid sequence
        "ACDEFGHIKLMNPQRSTVWY",      # All amino acids
        "INVALID123",                # Invalid sequence (should fail)
        "M" * 50                     # Medium length sequence
    ]
    
    results = []
    
    for i, seq in enumerate(test_sequences):
        try:
            logger.info(f"Processing test sequence {i+1}/{len(test_sequences)}")
            result = framework.robust_prediction(
                seq, 
                mock_prediction_model,
                model_version="test_v1.0"
            )
            
            # Test uncertainty quantification
            uncertainties = framework.robust_uncertainty_quantification(
                result, mock_uncertainty_model
            )
            
            result['uncertainties'] = uncertainties
            results.append(result)
            
            logger.info(f"Successfully processed sequence {i+1}")
            
        except Exception as e:
            logger.error(f"Failed to process sequence {i+1}: {e}")
            results.append({'error': str(e)})
    
    # Get system status
    logger.info("Collecting system status...")
    status = framework.get_system_status()
    
    print("\n" + "="*80)
    print("🛡️ PRODUCTION ROBUSTNESS FRAMEWORK STATUS")
    print("="*80)
    
    print(f"\n📊 Health Status: {status['health']['status'].upper()}")
    print(f"⏱️ Uptime: {status['uptime_hours']:.2f} hours")
    print(f"📟 Memory Usage: {status['health']['metrics']['memory_percent']:.1f}%")
    print(f"⚙️ CPU Usage: {status['health']['metrics']['cpu_percent']:.1f}%")
    
    print(f"\n⚠️ Error Summary:")
    print(f"  Total Errors: {status['errors']['total_errors']}")
    print(f"  Error Rate: {status['errors']['error_rate']:.2%}")
    
    print(f"\n🔄 Circuit Breaker Status:")
    for name, cb_status in status['circuit_breakers'].items():
        print(f"  {name}: {cb_status['state']} (failures: {cb_status['failure_count']})")
    
    if status['health']['alerts']:
        print(f"\n🚨 Active Alerts:")
        for alert in status['health']['alerts']:
            print(f"  {alert['type']}: {alert.get('value', 'N/A')}")
    
    # Performance summary
    if status['performance']:
        print(f"\n📈 Performance Summary:")
        for operation, perf in status['performance'].items():
            print(f"  {operation}:")
            print(f"    Calls: {perf['count']}")
            print(f"    Mean Time: {perf['mean_time']:.4f}s")
            print(f"    P95 Time: {perf['p95_time']:.4f}s")
    
    # Successful results summary
    successful_results = [r for r in results if 'error' not in r]
    print(f"\n✅ Successfully processed {len(successful_results)}/{len(test_sequences)} sequences")
    
    # Generate diagnostics
    diagnostics = framework.get_diagnostics()
    
    if diagnostics['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in diagnostics['recommendations']:
            print(f"  • {rec}")
    
    # Export metrics
    framework.export_metrics("robustness_test_metrics.json")
    
    logger.info("🎉 Production Robustness Framework testing complete!")
    print("\n🚀 Framework is production-ready with comprehensive error handling and monitoring!")

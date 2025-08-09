"""
Monitoring and performance tracking utilities for protein-sssl-operator
Provides real-time metrics, resource monitoring, and performance analytics
"""

import time
import threading
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
import warnings
from pathlib import Path

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric measurement"""
    timestamp: float
    value: Union[float, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_used_gb: float
    memory_percent: float
    gpu_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    disk_usage_gb: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)

@dataclass
class TrainingMetrics:
    """Training-specific metrics"""
    timestamp: float
    epoch: int
    step: int
    learning_rate: float
    losses: Dict[str, float] = field(default_factory=dict)
    throughput: float = 0.0  # samples/second
    batch_size: int = 0
    gradient_norm: float = 0.0
    memory_allocated_gb: float = 0.0

class MetricsBuffer:
    """Thread-safe circular buffer for metrics"""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.RLock()
    
    def add(self, metric: MetricPoint):
        """Add metric to buffer"""
        with self.lock:
            self.buffer.append(metric)
    
    def get_recent(self, seconds: float) -> List[MetricPoint]:
        """Get metrics from last N seconds"""
        cutoff_time = time.time() - seconds
        with self.lock:
            return [m for m in self.buffer if m.timestamp >= cutoff_time]
    
    def get_all(self) -> List[MetricPoint]:
        """Get all metrics"""
        with self.lock:
            return list(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get buffer size"""
        with self.lock:
            return len(self.buffer)

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self,
                 enable_system_monitoring: bool = True,
                 enable_gpu_monitoring: bool = True,
                 monitoring_interval: float = 1.0,
                 metrics_retention_hours: float = 24.0):
        
        self.enable_system_monitoring = enable_system_monitoring
        self.enable_gpu_monitoring = enable_gpu_monitoring and GPU_AVAILABLE
        self.monitoring_interval = monitoring_interval
        self.metrics_retention_hours = metrics_retention_hours
        
        # Metrics buffers
        self.system_metrics = MetricsBuffer()
        self.training_metrics = MetricsBuffer()
        self.custom_metrics = defaultdict(lambda: MetricsBuffer())
        
        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._is_monitoring = False
        
        # Performance tracking
        self._function_times = defaultdict(list)
        self._operation_times = defaultdict(list)
        
        # Alerts
        self._alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 90.0,
            'gpu_memory_percent': 90.0,
            'gpu_temperature': 85.0
        }
        self._alert_callbacks = []
        
        if not self.enable_gpu_monitoring and enable_gpu_monitoring:
            warnings.warn("GPU monitoring requested but GPUtil not available")
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self._is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitor_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        self._is_monitoring = True
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self._is_monitoring:
            return
        
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        self._is_monitoring = False
        
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                if self.enable_system_monitoring:
                    self._collect_system_metrics()
                    
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            timestamp = time.time()
            
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_usage_gb = disk_usage.used / (1024**3)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_metrics = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
            
            # GPU metrics
            gpu_metrics = {}
            if self.enable_gpu_monitoring:
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        gpu_metrics[i] = {
                            'utilization': gpu.load * 100,
                            'memory_used_gb': gpu.memoryUsed / 1024,
                            'memory_total_gb': gpu.memoryTotal / 1024,
                            'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            'temperature': gpu.temperature
                        }
                        
                        # Check for alerts
                        self._check_alerts(gpu_metrics[i], f"gpu_{i}")
                        
                except Exception as e:
                    logger.debug(f"Error collecting GPU metrics: {e}")
            
            # Create system metrics object
            metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_used_gb=memory.used / (1024**3),
                memory_percent=memory.percent,
                gpu_metrics=gpu_metrics,
                disk_usage_gb=disk_usage_gb,
                network_io=network_metrics
            )
            
            # Add to buffer
            self.system_metrics.add(MetricPoint(
                timestamp=timestamp,
                value=0,  # System metrics don't have single value
                metadata=metrics.__dict__
            ))
            
            # Check for system alerts
            self._check_alerts({
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent
            }, "system")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _check_alerts(self, metrics: Dict[str, float], source: str):
        """Check if any metrics exceed alert thresholds"""
        for metric_name, value in metrics.items():
            if metric_name in self._alert_thresholds:
                threshold = self._alert_thresholds[metric_name]
                if value > threshold:
                    alert_data = {
                        'source': source,
                        'metric': metric_name,
                        'value': value,
                        'threshold': threshold,
                        'timestamp': time.time()
                    }
                    
                    logger.warning(f"Alert: {source}.{metric_name} = {value:.1f} > {threshold}")
                    
                    # Call alert callbacks
                    for callback in self._alert_callbacks:
                        try:
                            callback(alert_data)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = time.time() - (self.metrics_retention_hours * 3600)
        
        # This is a simplified cleanup - in a real implementation,
        # you'd want a more efficient approach
        try:
            while (self.system_metrics.buffer and 
                   self.system_metrics.buffer[0].timestamp < cutoff_time):
                self.system_metrics.buffer.popleft()
                
            while (self.training_metrics.buffer and 
                   self.training_metrics.buffer[0].timestamp < cutoff_time):
                self.training_metrics.buffer.popleft()
                
        except (IndexError, AttributeError):
            pass  # Buffer might be empty
    
    def record_training_metrics(self,
                              epoch: int,
                              step: int,
                              learning_rate: float,
                              losses: Dict[str, float],
                              throughput: float = 0.0,
                              batch_size: int = 0,
                              gradient_norm: float = 0.0):
        """Record training-specific metrics"""
        
        timestamp = time.time()
        memory_allocated_gb = 0.0
        
        # Get PyTorch memory usage if CUDA available
        if torch.cuda.is_available():
            memory_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        
        metrics = TrainingMetrics(
            timestamp=timestamp,
            epoch=epoch,
            step=step,
            learning_rate=learning_rate,
            losses=losses,
            throughput=throughput,
            batch_size=batch_size,
            gradient_norm=gradient_norm,
            memory_allocated_gb=memory_allocated_gb
        )
        
        self.training_metrics.add(MetricPoint(
            timestamp=timestamp,
            value=losses.get('total', 0.0),
            metadata=metrics.__dict__
        ))
    
    def record_custom_metric(self,
                           name: str,
                           value: Union[float, int],
                           metadata: Optional[Dict[str, Any]] = None):
        """Record custom metric"""
        
        metric = MetricPoint(
            timestamp=time.time(),
            value=value,
            metadata=metadata or {}
        )
        
        self.custom_metrics[name].add(metric)
    
    @contextmanager
    def measure_time(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to measure operation time"""
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            self._operation_times[operation_name].append(duration)
            
            self.record_custom_metric(
                f"operation_time_{operation_name}",
                duration,
                metadata=metadata
            )
    
    def measure_function_time(self, func: Callable) -> Callable:
        """Decorator to measure function execution time"""
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                self._function_times[func_name].append(duration)
                self.record_custom_metric(
                    f"function_time_{func_name}",
                    duration
                )
        
        return wrapper
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        recent_metrics = self.system_metrics.get_recent(60.0)  # Last minute
        
        if not recent_metrics:
            return {'status': 'no_data'}
        
        latest = recent_metrics[-1].metadata
        
        status = {
            'timestamp': latest['timestamp'],
            'cpu_percent': latest['cpu_percent'],
            'memory_used_gb': latest['memory_used_gb'],
            'memory_percent': latest['memory_percent'],
            'disk_usage_gb': latest['disk_usage_gb'],
            'gpu_count': len(latest.get('gpu_metrics', {})),
            'is_monitoring': self._is_monitoring
        }
        
        # Add GPU status
        if latest.get('gpu_metrics'):
            gpu_status = {}
            for gpu_id, gpu_metrics in latest['gpu_metrics'].items():
                gpu_status[f'gpu_{gpu_id}'] = {
                    'utilization': gpu_metrics['utilization'],
                    'memory_percent': gpu_metrics['memory_percent'],
                    'temperature': gpu_metrics['temperature']
                }
            status['gpus'] = gpu_status
        
        return status
    
    def get_training_summary(self, last_minutes: float = 60.0) -> Dict[str, Any]:
        """Get training metrics summary"""
        recent_metrics = self.training_metrics.get_recent(last_minutes * 60)
        
        if not recent_metrics:
            return {'status': 'no_training_data'}
        
        # Calculate averages
        total_loss_values = [m.value for m in recent_metrics if m.value > 0]
        learning_rates = [m.metadata.get('learning_rate', 0) for m in recent_metrics]
        throughputs = [m.metadata.get('throughput', 0) for m in recent_metrics if m.metadata.get('throughput', 0) > 0]
        
        summary = {
            'total_steps': len(recent_metrics),
            'time_range_minutes': last_minutes,
            'avg_loss': np.mean(total_loss_values) if total_loss_values else 0,
            'current_lr': learning_rates[-1] if learning_rates else 0,
            'avg_throughput': np.mean(throughputs) if throughputs else 0
        }
        
        # Get latest training state
        if recent_metrics:
            latest = recent_metrics[-1].metadata
            summary.update({
                'current_epoch': latest.get('epoch', 0),
                'current_step': latest.get('step', 0),
                'memory_allocated_gb': latest.get('memory_allocated_gb', 0)
            })
        
        return summary
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'timestamp': time.time(),
            'monitoring_active': self._is_monitoring,
            'system_status': self.get_system_status(),
            'training_summary': self.get_training_summary(),
            'function_performance': {},
            'operation_performance': {},
            'buffer_sizes': {
                'system_metrics': self.system_metrics.size(),
                'training_metrics': self.training_metrics.size(),
                'custom_metrics': {name: buffer.size() for name, buffer in self.custom_metrics.items()}
            }
        }
        
        # Function performance summary
        for func_name, times in self._function_times.items():
            if times:
                report['function_performance'][func_name] = {
                    'call_count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times)
                }
        
        # Operation performance summary
        for op_name, times in self._operation_times.items():
            if times:
                report['operation_performance'][op_name] = {
                    'operation_count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times)
                }
        
        return report
    
    def export_metrics(self, file_path: Union[str, Path], format: str = 'json'):
        """Export collected metrics to file"""
        
        file_path = Path(file_path)
        
        # Collect all metrics
        export_data = {
            'timestamp': time.time(),
            'system_metrics': [],
            'training_metrics': [],
            'custom_metrics': {}
        }
        
        # System metrics
        for metric in self.system_metrics.get_all():
            export_data['system_metrics'].append({
                'timestamp': metric.timestamp,
                'data': metric.metadata
            })
        
        # Training metrics
        for metric in self.training_metrics.get_all():
            export_data['training_metrics'].append({
                'timestamp': metric.timestamp,
                'value': metric.value,
                'data': metric.metadata
            })
        
        # Custom metrics
        for name, buffer in self.custom_metrics.items():
            export_data['custom_metrics'][name] = []
            for metric in buffer.get_all():
                export_data['custom_metrics'][name].append({
                    'timestamp': metric.timestamp,
                    'value': metric.value,
                    'metadata': metric.metadata
                })
        
        # Save to file
        if format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Metrics exported to {file_path}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback function for alerts"""
        self._alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for metric"""
        self._alert_thresholds[metric_name] = threshold
    
    def clear_metrics(self):
        """Clear all collected metrics"""
        self.system_metrics.clear()
        self.training_metrics.clear()
        for buffer in self.custom_metrics.values():
            buffer.clear()
        
        self._function_times.clear()
        self._operation_times.clear()
        
        logger.info("All metrics cleared")

# Global monitor instance
_global_monitor = None

def get_monitor(**kwargs) -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(**kwargs)
    
    return _global_monitor

def start_monitoring():
    """Start global monitoring"""
    get_monitor().start_monitoring()

def stop_monitoring():
    """Stop global monitoring"""
    if _global_monitor:
        _global_monitor.stop_monitoring()

def record_training_metrics(**kwargs):
    """Record training metrics using global monitor"""
    get_monitor().record_training_metrics(**kwargs)

def record_custom_metric(name: str, value: Union[float, int], metadata: Optional[Dict[str, Any]] = None):
    """Record custom metric using global monitor"""
    get_monitor().record_custom_metric(name, value, metadata)

@contextmanager
def measure_time(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Measure operation time using global monitor"""
    with get_monitor().measure_time(operation_name, metadata):
        yield

def measure_function_time(func: Callable) -> Callable:
    """Decorator to measure function time using global monitor"""
    return get_monitor().measure_function_time(func)

class WandbMonitor:
    """Weights & Biases integration for monitoring"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.wandb_available = False
        
        try:
            import wandb
            self.wandb = wandb
            self.wandb_available = True
        except ImportError:
            logger.warning("wandb not available for monitoring integration")
    
    def log_system_metrics(self):
        """Log system metrics to wandb"""
        if not self.wandb_available:
            return
        
        status = self.monitor.get_system_status()
        if status.get('status') != 'no_data':
            wandb_metrics = {
                'system/cpu_percent': status['cpu_percent'],
                'system/memory_used_gb': status['memory_used_gb'],
                'system/memory_percent': status['memory_percent'],
                'system/disk_usage_gb': status['disk_usage_gb']
            }
            
            # Add GPU metrics
            if 'gpus' in status:
                for gpu_name, gpu_metrics in status['gpus'].items():
                    wandb_metrics[f'system/{gpu_name}_utilization'] = gpu_metrics['utilization']
                    wandb_metrics[f'system/{gpu_name}_memory'] = gpu_metrics['memory_percent']
                    wandb_metrics[f'system/{gpu_name}_temperature'] = gpu_metrics['temperature']
            
            self.wandb.log(wandb_metrics)
    
    def start_wandb_logging(self, interval: float = 30.0):
        """Start periodic logging to wandb"""
        if not self.wandb_available:
            return
        
        def log_loop():
            while not self._stop_wandb_logging.wait(interval):
                try:
                    self.log_system_metrics()
                except Exception as e:
                    logger.error(f"Error logging to wandb: {e}")
        
        self._stop_wandb_logging = threading.Event()
        self._wandb_thread = threading.Thread(target=log_loop)
        self._wandb_thread.daemon = True
        self._wandb_thread.start()
    
    def stop_wandb_logging(self):
        """Stop wandb logging"""
        if hasattr(self, '_stop_wandb_logging'):
            self._stop_wandb_logging.set()
            if hasattr(self, '_wandb_thread'):
                self._wandb_thread.join(timeout=5.0)
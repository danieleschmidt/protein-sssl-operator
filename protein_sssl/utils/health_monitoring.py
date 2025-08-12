"""
System health monitoring and performance tracking.
"""
import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import json

from .logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_used_percent: float
    disk_free_gb: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None


@dataclass
class ProcessMetrics:
    """Process-specific metrics."""
    timestamp: float
    pid: int
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    num_threads: int
    io_read_bytes: int
    io_write_bytes: int


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(
        self,
        collection_interval: float = 30.0,
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True
    ):
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        self.system_metrics = deque(maxlen=history_size)
        self.process_metrics = deque(maxlen=history_size)
        
        self.monitoring = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        
        # GPU monitoring setup
        self.gpu_available = False
        if enable_gpu_monitoring:
            self._setup_gpu_monitoring()
    
    def _setup_gpu_monitoring(self):
        """Setup GPU monitoring if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            logger.info("GPU monitoring enabled")
        except ImportError:
            logger.debug("pynvml not available, GPU monitoring disabled")
        except Exception as e:
            logger.debug(f"GPU monitoring setup failed: {e}")
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                process_metrics = self._collect_process_metrics()
                
                with self._lock:
                    self.system_metrics.append(system_metrics)
                    self.process_metrics.append(process_metrics)
                
                # Check for alerts
                self._check_alerts(system_metrics, process_metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-wide metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None
        
        if self.gpu_available:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_memory_used = gpu_info.used / 1024**2  # MB
                gpu_memory_total = gpu_info.total / 1024**2  # MB
                gpu_utilization = gpu_util.gpu
                
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / 1024**3,
            memory_available_gb=memory.available / 1024**3,
            disk_used_percent=disk.percent,
            disk_free_gb=disk.free / 1024**3,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_utilization_percent=gpu_utilization
        )
    
    def _collect_process_metrics(self) -> ProcessMetrics:
        """Collect current process metrics."""
        process = psutil.Process()
        
        try:
            io_counters = process.io_counters()
            io_read = io_counters.read_bytes
            io_write = io_counters.write_bytes
        except (AttributeError, psutil.AccessDenied):
            io_read = io_write = 0
        
        memory_info = process.memory_info()
        
        return ProcessMetrics(
            timestamp=time.time(),
            pid=process.pid,
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            memory_rss_mb=memory_info.rss / 1024**2,
            memory_vms_mb=memory_info.vms / 1024**2,
            num_threads=process.num_threads(),
            io_read_bytes=io_read,
            io_write_bytes=io_write
        )
    
    def _check_alerts(self, system_metrics: SystemMetrics, process_metrics: ProcessMetrics):
        """Check for alert conditions."""
        # Memory alerts
        if system_metrics.memory_percent > 90:
            logger.warning(
                f"High system memory usage: {system_metrics.memory_percent:.1f}%"
            )
        
        if process_metrics.memory_percent > 80:
            logger.warning(
                f"High process memory usage: {process_metrics.memory_percent:.1f}%"
            )
        
        # CPU alerts
        if system_metrics.cpu_percent > 95:
            logger.warning(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
        
        # Disk alerts
        if system_metrics.disk_used_percent > 90:
            logger.warning(
                f"Low disk space: {system_metrics.disk_used_percent:.1f}% used"
            )
        
        # GPU alerts
        if (system_metrics.gpu_memory_used_mb and 
            system_metrics.gpu_memory_total_mb):
            gpu_memory_percent = (
                system_metrics.gpu_memory_used_mb / 
                system_metrics.gpu_memory_total_mb * 100
            )
            if gpu_memory_percent > 90:
                logger.warning(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system and process metrics."""
        system_metrics = self._collect_system_metrics()
        process_metrics = self._collect_process_metrics()
        
        return {
            'system': asdict(system_metrics),
            'process': asdict(process_metrics)
        }
    
    def get_metrics_summary(self, hours: float = 1.0) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            # Filter recent metrics
            recent_system = [
                m for m in self.system_metrics 
                if m.timestamp > cutoff_time
            ]
            recent_process = [
                m for m in self.process_metrics
                if m.timestamp > cutoff_time
            ]
        
        if not recent_system:
            return {'error': 'No metrics available'}
        
        # Calculate statistics
        system_stats = self._calculate_stats(recent_system)
        process_stats = self._calculate_stats(recent_process)
        
        return {
            'period_hours': hours,
            'sample_count': len(recent_system),
            'system': system_stats,
            'process': process_stats
        }
    
    def _calculate_stats(self, metrics: List) -> Dict[str, Any]:
        """Calculate statistics for metrics list."""
        if not metrics:
            return {}
        
        # Get numeric fields
        numeric_fields = []
        for field_name, field_value in asdict(metrics[0]).items():
            if isinstance(field_value, (int, float)) and field_name != 'timestamp':
                numeric_fields.append(field_name)
        
        stats = {}
        for field in numeric_fields:
            values = [getattr(m, field) for m in metrics if getattr(m, field) is not None]
            if values:
                stats[field] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1]
                }
        
        return stats
    
    def export_metrics(self, filepath: str, hours: float = 24.0):
        """Export metrics to JSON file."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            recent_system = [
                asdict(m) for m in self.system_metrics 
                if m.timestamp > cutoff_time
            ]
            recent_process = [
                asdict(m) for m in self.process_metrics
                if m.timestamp > cutoff_time
            ]
        
        export_data = {
            'export_timestamp': time.time(),
            'period_hours': hours,
            'system_metrics': recent_system,
            'process_metrics': recent_process
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        current = self.get_current_metrics()
        
        health_status = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'issues': [],
            'metrics': current
        }
        
        # Check system health
        system = current['system']
        if system['memory_percent'] > 85:
            health_status['issues'].append('High memory usage')
        
        if system['cpu_percent'] > 90:
            health_status['issues'].append('High CPU usage')
        
        if system['disk_used_percent'] > 85:
            health_status['issues'].append('Low disk space')
        
        # Check GPU health if available
        if system.get('gpu_memory_used_mb') and system.get('gpu_memory_total_mb'):
            gpu_usage = system['gpu_memory_used_mb'] / system['gpu_memory_total_mb'] * 100
            if gpu_usage > 90:
                health_status['issues'].append('High GPU memory usage')
        
        # Set overall status
        if len(health_status['issues']) > 2:
            health_status['overall_status'] = 'critical'
        elif health_status['issues']:
            health_status['overall_status'] = 'warning'
        
        return health_status


class PerformanceProfiler:
    """Performance profiling for functions and code blocks."""
    
    def __init__(self):
        self.profiles = {}
        self._lock = threading.Lock()
    
    def profile(self, name: str):
        """Decorator for profiling function execution."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    self._record_profile(
                        name or func.__name__,
                        execution_time,
                        memory_delta,
                        success
                    )
                
                return result
            return wrapper
        return decorator
    
    def _record_profile(self, name: str, execution_time: float, memory_delta: int, success: bool):
        """Record profiling data."""
        with self._lock:
            if name not in self.profiles:
                self.profiles[name] = {
                    'call_count': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'total_memory_delta': 0,
                    'success_count': 0,
                    'failure_count': 0
                }
            
            profile = self.profiles[name]
            profile['call_count'] += 1
            profile['total_time'] += execution_time
            profile['min_time'] = min(profile['min_time'], execution_time)
            profile['max_time'] = max(profile['max_time'], execution_time)
            profile['total_memory_delta'] += memory_delta
            
            if success:
                profile['success_count'] += 1
            else:
                profile['failure_count'] += 1
    
    def get_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all profiling data."""
        with self._lock:
            profiles = {}
            for name, data in self.profiles.items():
                if data['call_count'] > 0:
                    profiles[name] = {
                        **data,
                        'avg_time': data['total_time'] / data['call_count'],
                        'avg_memory_delta_mb': data['total_memory_delta'] / data['call_count'] / 1024**2,
                        'success_rate': data['success_count'] / data['call_count']
                    }
            return profiles
    
    def reset_profiles(self):
        """Reset all profiling data."""
        with self._lock:
            self.profiles.clear()


# Global instances
health_monitor = HealthMonitor()
performance_profiler = PerformanceProfiler()
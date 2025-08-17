"""
Enterprise-Grade Health Monitoring and Observability for protein-sssl-operator
Provides comprehensive system monitoring, metrics collection, alerting,
and observability with OpenTelemetry integration.
"""

import time
import asyncio
import threading
import psutil
import json
import logging
import statistics
import socket
import platform
from typing import (
    Dict, List, Optional, Any, Callable, Union, 
    NamedTuple, Protocol, TypeVar, Generic
)
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import deque, defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, asynccontextmanager
from pathlib import Path
import weakref
import uuid
import hashlib
import os

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = ("info", 1)
    WARNING = ("warning", 2)
    CRITICAL = ("critical", 3)
    FATAL = ("fatal", 4)
    
    def __init__(self, name: str, level: int):
        self.level_name = name
        self.level = level
    
    def __lt__(self, other):
        return self.level < other.level

@dataclass
class MetricPoint:
    """Individual metric measurement"""
    name: str
    value: Union[float, int, str]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'metadata': self.metadata
        }

@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    timestamp: float
    
    # CPU metrics
    cpu_percent: float
    cpu_count: int
    cpu_freq_current: float
    cpu_freq_min: float
    cpu_freq_max: float
    load_avg_1m: float
    load_avg_5m: float
    load_avg_15m: float
    
    # Memory metrics
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float
    
    # Disk metrics
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    disk_percent: float
    disk_read_bytes: int
    disk_write_bytes: int
    disk_read_iops: int
    disk_write_iops: int
    
    # Network metrics
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    network_errors_in: int
    network_errors_out: int
    network_drops_in: int
    network_drops_out: int
    
    # Process metrics
    process_count: int
    process_cpu_percent: float
    process_memory_percent: float
    process_memory_rss_gb: float
    process_memory_vms_gb: float
    process_threads: int
    process_files_open: int
    
    # GPU metrics (if available)
    gpu_count: int = 0
    gpu_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # PyTorch metrics (if available)
    torch_cuda_available: bool = False
    torch_cuda_device_count: int = 0
    torch_cuda_current_device: int = -1
    torch_cuda_memory_allocated: float = 0.0
    torch_cuda_memory_reserved: float = 0.0
    torch_cuda_memory_cached: float = 0.0

@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: float
    
    # Request metrics
    request_count: int = 0
    request_rate: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    
    # Performance metrics
    avg_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Model metrics
    model_predictions: int = 0
    model_errors: int = 0
    model_inference_time: float = 0.0
    model_accuracy: float = 0.0
    
    # Training metrics
    training_epoch: int = 0
    training_step: int = 0
    training_loss: float = 0.0
    validation_loss: float = 0.0
    learning_rate: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    source: str
    metric_name: str
    metric_value: Union[float, int, str]
    threshold: Union[float, int, str]
    condition: str
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'severity': self.severity.level_name,
            'message': self.message,
            'timestamp': self.timestamp,
            'source': self.source,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'condition': self.condition,
            'labels': self.labels,
            'metadata': self.metadata,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'resolved_timestamp': self.resolved_timestamp
        }

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # e.g., ">", "<", "==", "!="
    threshold: Union[float, int, str]
    severity: AlertSeverity
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    cooldown_seconds: float = 300.0  # 5 minutes
    enabled: bool = True
    
    def evaluate(self, metric_value: Union[float, int, str]) -> bool:
        """Evaluate if alert should be triggered"""
        if not self.enabled:
            return False
        
        try:
            if self.condition == ">":
                return float(metric_value) > float(self.threshold)
            elif self.condition == "<":
                return float(metric_value) < float(self.threshold)
            elif self.condition == ">=":
                return float(metric_value) >= float(self.threshold)
            elif self.condition == "<=":
                return float(metric_value) <= float(self.threshold)
            elif self.condition == "==":
                return metric_value == self.threshold
            elif self.condition == "!=":
                return metric_value != self.threshold
            elif self.condition == "contains":
                return str(self.threshold) in str(metric_value)
            elif self.condition == "not_contains":
                return str(self.threshold) not in str(metric_value)
            else:
                logger.warning(f"Unknown condition: {self.condition}")
                return False
        except (ValueError, TypeError) as e:
            logger.error(f"Alert evaluation error: {e}")
            return False

class HealthChecker:
    """Health check interface"""
    
    def __init__(self, name: str, timeout: float = 30.0):
        self.name = name
        self.timeout = timeout
    
    async def check(self) -> Dict[str, Any]:
        """Perform health check"""
        raise NotImplementedError
    
    def is_critical(self) -> bool:
        """Return True if this check is critical for overall health"""
        return False

class SystemHealthChecker(HealthChecker):
    """System resource health checker"""
    
    def __init__(self, 
                 cpu_threshold: float = 90.0,
                 memory_threshold: float = 90.0,
                 disk_threshold: float = 90.0):
        super().__init__("system", timeout=10.0)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def check(self) -> Dict[str, Any]:
        """Check system health"""
        issues = []
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.cpu_threshold:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > self.memory_threshold:
            issues.append(f"High memory usage: {memory.percent:.1f}%")
        
        # Disk check
        disk = psutil.disk_usage('/')
        if disk.percent > self.disk_threshold:
            issues.append(f"High disk usage: {disk.percent:.1f}%")
        
        status = HealthStatus.CRITICAL if issues else HealthStatus.HEALTHY
        
        return {
            'status': status.value,
            'issues': issues,
            'metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent
            }
        }
    
    def is_critical(self) -> bool:
        return True

class DatabaseHealthChecker(HealthChecker):
    """Database connectivity health checker"""
    
    def __init__(self, connection_string: str):
        super().__init__("database", timeout=15.0)
        self.connection_string = connection_string
    
    async def check(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # This is a placeholder - implement actual database check
            # For example, with SQLAlchemy or similar
            await asyncio.sleep(0.1)  # Simulate check
            
            return {
                'status': HealthStatus.HEALTHY.value,
                'response_time': 0.1,
                'connection_pool_active': 5,
                'connection_pool_idle': 10
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'error': str(e)
            }
    
    def is_critical(self) -> bool:
        return True

class ExternalServiceHealthChecker(HealthChecker):
    """External service health checker"""
    
    def __init__(self, service_name: str, endpoint: str):
        super().__init__(f"external_{service_name}", timeout=20.0)
        self.service_name = service_name
        self.endpoint = endpoint
    
    async def check(self) -> Dict[str, Any]:
        """Check external service health"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                start_time = time.time()
                async with session.get(self.endpoint) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        return {
                            'status': HealthStatus.HEALTHY.value,
                            'response_time': response_time,
                            'status_code': response.status
                        }
                    else:
                        return {
                            'status': HealthStatus.DEGRADED.value,
                            'response_time': response_time,
                            'status_code': response.status,
                            'error': f"HTTP {response.status}"
                        }
                        
        except asyncio.TimeoutError:
            return {
                'status': HealthStatus.CRITICAL.value,
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'error': str(e)
            }
    
    def is_critical(self) -> bool:
        return False

class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self, 
                 collection_interval: float = 30.0,
                 retention_hours: float = 24.0,
                 max_metrics: int = 100000):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.max_metrics = max_metrics
        
        # Metrics storage
        self._metrics: deque = deque(maxlen=max_metrics)
        self._metric_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
        
        # Collection state
        self._collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        
        # GPU monitoring setup
        self._gpu_available = False
        if NVIDIA_ML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_available = True
                logger.info("GPU monitoring enabled")
            except Exception as e:
                logger.debug(f"GPU monitoring setup failed: {e}")
        
        # PyTorch monitoring setup
        self._torch_available = TORCH_AVAILABLE
        
        # Prometheus metrics (if available)
        self._prometheus_metrics: Dict[str, Any] = {}
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self._prometheus_metrics = {
            'system_cpu_percent': Gauge('system_cpu_percent', 'CPU usage percentage'),
            'system_memory_percent': Gauge('system_memory_percent', 'Memory usage percentage'),
            'system_disk_percent': Gauge('system_disk_percent', 'Disk usage percentage'),
            'system_load_avg': Gauge('system_load_avg', 'System load average', ['period']),
            'network_bytes_total': Counter('network_bytes_total', 'Network bytes', ['direction']),
            'process_cpu_percent': Gauge('process_cpu_percent', 'Process CPU usage'),
            'process_memory_percent': Gauge('process_memory_percent', 'Process memory usage'),
            'gpu_utilization': Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id']),
            'gpu_memory_percent': Gauge('gpu_memory_percent', 'GPU memory usage', ['gpu_id']),
            'gpu_temperature': Gauge('gpu_temperature_celsius', 'GPU temperature', ['gpu_id']),
            'model_inference_time': Histogram('model_inference_time_seconds', 'Model inference time'),
            'model_predictions_total': Counter('model_predictions_total', 'Total model predictions'),
            'model_errors_total': Counter('model_errors_total', 'Total model errors'),
        }
    
    def start_collection(self):
        """Start metrics collection"""
        if self._collecting:
            logger.warning("Metrics collection already started")
            return
        
        self._collecting = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        if not self._collecting:
            return
        
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=10.0)
        
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self._collecting:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self._store_metric(MetricPoint(
                    name="system_metrics",
                    value=0,  # System metrics don't have a single value
                    timestamp=time.time(),
                    metadata=asdict(system_metrics)
                ))
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE and self._prometheus_metrics:
                    self._update_prometheus_metrics(system_metrics)
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                self._store_metric(MetricPoint(
                    name="application_metrics",
                    value=0,
                    timestamp=time.time(),
                    metadata=asdict(app_metrics)
                ))
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        try:
            process_files = process.num_fds() if hasattr(process, 'num_fds') else len(process.open_files())
        except (psutil.AccessDenied, AttributeError):
            process_files = 0
        
        # GPU metrics
        gpu_count = 0
        gpu_metrics = []
        if self._gpu_available:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    # GPU memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # GPU temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = 0
                    
                    # GPU power
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    except:
                        power = 0
                    
                    gpu_metrics.append({
                        'gpu_id': i,
                        'utilization_percent': util.gpu,
                        'memory_utilization_percent': util.memory,
                        'memory_used_mb': mem_info.used / 1024**2,
                        'memory_total_mb': mem_info.total / 1024**2,
                        'memory_free_mb': mem_info.free / 1024**2,
                        'temperature_celsius': temp,
                        'power_watts': power
                    })
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
        
        # PyTorch metrics
        torch_cuda_available = False
        torch_cuda_device_count = 0
        torch_cuda_current_device = -1
        torch_cuda_memory_allocated = 0.0
        torch_cuda_memory_reserved = 0.0
        torch_cuda_memory_cached = 0.0
        
        if self._torch_available:
            try:
                torch_cuda_available = torch.cuda.is_available()
                if torch_cuda_available:
                    torch_cuda_device_count = torch.cuda.device_count()
                    torch_cuda_current_device = torch.cuda.current_device()
                    torch_cuda_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    torch_cuda_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    torch_cuda_memory_cached = torch.cuda.memory_cached() / 1024**3  # GB
            except Exception as e:
                logger.debug(f"PyTorch metrics collection failed: {e}")
        
        return SystemMetrics(
            timestamp=timestamp,
            # CPU
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            cpu_freq_current=cpu_freq.current if cpu_freq else 0,
            cpu_freq_min=cpu_freq.min if cpu_freq else 0,
            cpu_freq_max=cpu_freq.max if cpu_freq else 0,
            load_avg_1m=load_avg[0],
            load_avg_5m=load_avg[1],
            load_avg_15m=load_avg[2],
            # Memory
            memory_total_gb=memory.total / 1024**3,
            memory_used_gb=memory.used / 1024**3,
            memory_available_gb=memory.available / 1024**3,
            memory_percent=memory.percent,
            swap_total_gb=swap.total / 1024**3,
            swap_used_gb=swap.used / 1024**3,
            swap_percent=swap.percent,
            # Disk
            disk_total_gb=disk.total / 1024**3,
            disk_used_gb=disk.used / 1024**3,
            disk_free_gb=disk.free / 1024**3,
            disk_percent=disk.percent,
            disk_read_bytes=disk_io.read_bytes if disk_io else 0,
            disk_write_bytes=disk_io.write_bytes if disk_io else 0,
            disk_read_iops=disk_io.read_count if disk_io else 0,
            disk_write_iops=disk_io.write_count if disk_io else 0,
            # Network
            network_bytes_sent=network_io.bytes_sent,
            network_bytes_recv=network_io.bytes_recv,
            network_packets_sent=network_io.packets_sent,
            network_packets_recv=network_io.packets_recv,
            network_errors_in=network_io.errin,
            network_errors_out=network_io.errout,
            network_drops_in=network_io.dropin,
            network_drops_out=network_io.dropout,
            # Process
            process_count=len(psutil.pids()),
            process_cpu_percent=process.cpu_percent(),
            process_memory_percent=process.memory_percent(),
            process_memory_rss_gb=process_memory.rss / 1024**3,
            process_memory_vms_gb=process_memory.vms / 1024**3,
            process_threads=process.num_threads(),
            process_files_open=process_files,
            # GPU
            gpu_count=gpu_count,
            gpu_metrics=gpu_metrics,
            # PyTorch
            torch_cuda_available=torch_cuda_available,
            torch_cuda_device_count=torch_cuda_device_count,
            torch_cuda_current_device=torch_cuda_current_device,
            torch_cuda_memory_allocated=torch_cuda_memory_allocated,
            torch_cuda_memory_reserved=torch_cuda_memory_reserved,
            torch_cuda_memory_cached=torch_cuda_memory_cached
        )
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        timestamp = time.time()
        
        # These would be collected from your application
        # This is a placeholder implementation
        return ApplicationMetrics(
            timestamp=timestamp,
            request_count=0,
            request_rate=0.0,
            error_count=0,
            error_rate=0.0,
            avg_response_time=0.0,
            p50_response_time=0.0,
            p95_response_time=0.0,
            p99_response_time=0.0,
            model_predictions=0,
            model_errors=0,
            model_inference_time=0.0,
            model_accuracy=0.0,
            training_epoch=0,
            training_step=0,
            training_loss=0.0,
            validation_loss=0.0,
            learning_rate=0.0,
            custom_metrics={}
        )
    
    def _update_prometheus_metrics(self, system_metrics: SystemMetrics):
        """Update Prometheus metrics"""
        try:
            # System metrics
            self._prometheus_metrics['system_cpu_percent'].set(system_metrics.cpu_percent)
            self._prometheus_metrics['system_memory_percent'].set(system_metrics.memory_percent)
            self._prometheus_metrics['system_disk_percent'].set(system_metrics.disk_percent)
            
            # Load averages
            self._prometheus_metrics['system_load_avg'].labels(period='1m').set(system_metrics.load_avg_1m)
            self._prometheus_metrics['system_load_avg'].labels(period='5m').set(system_metrics.load_avg_5m)
            self._prometheus_metrics['system_load_avg'].labels(period='15m').set(system_metrics.load_avg_15m)
            
            # Network
            self._prometheus_metrics['network_bytes_total'].labels(direction='sent')._value._value = system_metrics.network_bytes_sent
            self._prometheus_metrics['network_bytes_total'].labels(direction='recv')._value._value = system_metrics.network_bytes_recv
            
            # Process
            self._prometheus_metrics['process_cpu_percent'].set(system_metrics.process_cpu_percent)
            self._prometheus_metrics['process_memory_percent'].set(system_metrics.process_memory_percent)
            
            # GPU metrics
            for gpu_metric in system_metrics.gpu_metrics:
                gpu_id = str(gpu_metric['gpu_id'])
                self._prometheus_metrics['gpu_utilization'].labels(gpu_id=gpu_id).set(gpu_metric['utilization_percent'])
                
                memory_percent = (gpu_metric['memory_used_mb'] / gpu_metric['memory_total_mb']) * 100
                self._prometheus_metrics['gpu_memory_percent'].labels(gpu_id=gpu_id).set(memory_percent)
                
                self._prometheus_metrics['gpu_temperature'].labels(gpu_id=gpu_id).set(gpu_metric['temperature_celsius'])
                
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    def _store_metric(self, metric: MetricPoint):
        """Store metric point"""
        with self._lock:
            self._metrics.append(metric)
            self._metric_buffers[metric.name].append(metric)
    
    def _cleanup_old_metrics(self):
        """Remove old metrics based on retention policy"""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        with self._lock:
            # Clean main metrics deque
            while self._metrics and self._metrics[0].timestamp < cutoff_time:
                self._metrics.popleft()
            
            # Clean metric buffers
            for metric_name, buffer in self._metric_buffers.items():
                while buffer and buffer[0].timestamp < cutoff_time:
                    buffer.popleft()
    
    def record_custom_metric(self, name: str, value: Union[float, int, str], 
                           labels: Optional[Dict[str, str]] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Record custom metric"""
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels or {},
            metadata=metadata or {}
        )
        self._store_metric(metric)
    
    def get_metrics(self, metric_name: Optional[str] = None, 
                   hours: float = 1.0) -> List[MetricPoint]:
        """Get metrics for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            if metric_name:
                buffer = self._metric_buffers.get(metric_name, deque())
                return [m for m in buffer if m.timestamp >= cutoff_time]
            else:
                return [m for m in self._metrics if m.timestamp >= cutoff_time]
    
    def get_metric_statistics(self, metric_name: str, hours: float = 1.0) -> Dict[str, Any]:
        """Get statistics for a specific metric"""
        metrics = self.get_metrics(metric_name, hours)
        
        if not metrics:
            return {'count': 0}
        
        numeric_values = []
        for metric in metrics:
            try:
                numeric_values.append(float(metric.value))
            except (ValueError, TypeError):
                continue
        
        if not numeric_values:
            return {'count': len(metrics), 'type': 'non_numeric'}
        
        return {
            'count': len(metrics),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': statistics.mean(numeric_values),
            'median': statistics.median(numeric_values),
            'stdev': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
            'latest': numeric_values[-1] if numeric_values else None
        }
    
    def export_metrics(self, format: str = "json", file_path: Optional[str] = None) -> Union[str, Dict]:
        """Export metrics in specified format"""
        with self._lock:
            metrics_data = [metric.to_dict() for metric in self._metrics]
        
        if format.lower() == "json":
            data = {
                'timestamp': time.time(),
                'metrics_count': len(metrics_data),
                'metrics': metrics_data
            }
            
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                return file_path
            else:
                return data
        
        elif format.lower() == "csv" and PANDAS_AVAILABLE:
            df = pd.DataFrame(metrics_data)
            if file_path:
                df.to_csv(file_path, index=False)
                return file_path
            else:
                return df.to_csv(index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

class AlertManager:
    """Advanced alert management system"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        self._last_check_times: Dict[str, float] = {}
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        with self._lock:
            self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove alert rule"""
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback"""
        self.callbacks.append(callback)
    
    def check_metrics(self, metrics: List[MetricPoint]):
        """Check metrics against alert rules"""
        current_time = time.time()
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric)
        
        with self._lock:
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                last_check = self._last_check_times.get(rule_name, 0)
                if current_time - last_check < rule.cooldown_seconds:
                    continue
                
                # Get latest metric value
                if rule.metric_name in metrics_by_name:
                    latest_metric = max(metrics_by_name[rule.metric_name], key=lambda m: m.timestamp)
                    
                    if rule.evaluate(latest_metric.value):
                        self._trigger_alert(rule, latest_metric)
                        self._last_check_times[rule_name] = current_time
                    else:
                        # Check if we should resolve an existing alert
                        self._resolve_alert(rule_name)
    
    def _trigger_alert(self, rule: AlertRule, metric: MetricPoint):
        """Trigger an alert"""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            name=rule.name,
            severity=rule.severity,
            message=f"{rule.description or rule.name}: {metric.value} {rule.condition} {rule.threshold}",
            timestamp=time.time(),
            source="metrics",
            metric_name=rule.metric_name,
            metric_value=metric.value,
            threshold=rule.threshold,
            condition=rule.condition,
            labels=rule.labels.copy(),
            metadata={'metric_timestamp': metric.timestamp}
        )
        
        # Store alert
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _resolve_alert(self, rule_name: str):
        """Resolve an active alert"""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolved_timestamp = time.time()
            
            del self.active_alerts[rule_name]
            
            logger.info(f"Alert resolved: {alert.name}")
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Acknowledge an alert"""
        with self._lock:
            for alert in self.active_alerts.values():
                if alert.id == alert_id:
                    alert.acknowledged = True
                    alert.metadata['acknowledged_by'] = acknowledged_by
                    alert.metadata['acknowledged_at'] = time.time()
                    logger.info(f"Alert acknowledged: {alert.name} by {acknowledged_by}")
                    return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts"""
        with self._lock:
            alerts = list(self.active_alerts.values())
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self, hours: float = 24.0) -> Dict[str, Any]:
        """Get alert statistics"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]
        
        if not recent_alerts:
            return {'total_alerts': 0, 'by_severity': {}, 'by_rule': {}}
        
        by_severity = defaultdict(int)
        by_rule = defaultdict(int)
        resolved_count = 0
        
        for alert in recent_alerts:
            by_severity[alert.severity.level_name] += 1
            by_rule[alert.name] += 1
            if alert.resolved:
                resolved_count += 1
        
        return {
            'total_alerts': len(recent_alerts),
            'active_alerts': len(self.active_alerts),
            'resolved_alerts': resolved_count,
            'resolution_rate': resolved_count / len(recent_alerts),
            'by_severity': dict(by_severity),
            'by_rule': dict(by_rule)
        }

class EnterpriseMonitor:
    """Main enterprise monitoring system"""
    
    def __init__(self,
                 metrics_collection_interval: float = 30.0,
                 health_check_interval: float = 60.0,
                 enable_prometheus: bool = True,
                 enable_opentelemetry: bool = False,
                 otel_endpoint: Optional[str] = None):
        
        self.metrics_collection_interval = metrics_collection_interval
        self.health_check_interval = health_check_interval
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_opentelemetry = enable_opentelemetry and OTEL_AVAILABLE
        
        # Core components
        self.metrics_collector = MetricsCollector(metrics_collection_interval)
        self.alert_manager = AlertManager()
        self.health_checkers: Dict[str, HealthChecker] = {}
        
        # Health check state
        self._health_check_running = False
        self._health_check_thread: Optional[threading.Thread] = None
        
        # OpenTelemetry setup
        if self.enable_opentelemetry:
            self._setup_opentelemetry(otel_endpoint)
        
        # Default alert rules
        self._setup_default_alert_rules()
        
        # Default health checkers
        self._setup_default_health_checkers()
        
        logger.info("Enterprise monitor initialized")
    
    def _setup_opentelemetry(self, endpoint: Optional[str]):
        """Setup OpenTelemetry tracing and metrics"""
        try:
            # Tracing
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            if endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Metrics
            if endpoint:
                metric_exporter = OTLPMetricExporter(endpoint=endpoint)
                metric_reader = PeriodicExportingMetricReader(metric_exporter)
                metric_provider = MeterProvider(metric_readers=[metric_reader])
                otel_metrics.set_meter_provider(metric_provider)
            
            logger.info("OpenTelemetry configured")
            
        except Exception as e:
            logger.error(f"OpenTelemetry setup failed: {e}")
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system_metrics",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                description="High CPU usage detected",
                cooldown_seconds=300
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system_metrics",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                description="High memory usage detected",
                cooldown_seconds=300
            ),
            AlertRule(
                name="high_disk_usage",
                metric_name="system_metrics",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.CRITICAL,
                description="High disk usage detected",
                cooldown_seconds=600
            ),
            AlertRule(
                name="gpu_temperature_high",
                metric_name="system_metrics",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                description="High GPU temperature detected",
                cooldown_seconds=300
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
    
    def _setup_default_health_checkers(self):
        """Setup default health checkers"""
        self.add_health_checker(SystemHealthChecker())
    
    def add_health_checker(self, checker: HealthChecker):
        """Add health checker"""
        self.health_checkers[checker.name] = checker
        logger.info(f"Added health checker: {checker.name}")
    
    def remove_health_checker(self, name: str):
        """Remove health checker"""
        if name in self.health_checkers:
            del self.health_checkers[name]
            logger.info(f"Removed health checker: {name}")
    
    def start(self):
        """Start monitoring"""
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start health checks
        self._start_health_checks()
        
        # Setup alert callbacks
        self.alert_manager.add_callback(self._default_alert_callback)
        
        logger.info("Enterprise monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Stop health checks
        self._stop_health_checks()
        
        logger.info("Enterprise monitoring stopped")
    
    def _start_health_checks(self):
        """Start health check loop"""
        if self._health_check_running:
            return
        
        self._health_check_running = True
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
    
    def _stop_health_checks(self):
        """Stop health check loop"""
        if not self._health_check_running:
            return
        
        self._health_check_running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=10.0)
    
    def _health_check_loop(self):
        """Health check loop"""
        while self._health_check_running:
            try:
                asyncio.run(self._run_health_checks())
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(self.health_check_interval)
    
    async def _run_health_checks(self):
        """Run all health checks"""
        tasks = []
        for checker in self.health_checkers.values():
            task = asyncio.create_task(self._run_single_health_check(checker))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_single_health_check(self, checker: HealthChecker):
        """Run a single health check"""
        try:
            result = await asyncio.wait_for(checker.check(), timeout=checker.timeout)
            
            # Record health check result as metric
            self.metrics_collector.record_custom_metric(
                f"health_check_{checker.name}",
                1 if result.get('status') == HealthStatus.HEALTHY.value else 0,
                labels={'checker': checker.name, 'status': result.get('status', 'unknown')},
                metadata=result
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout: {checker.name}")
            self.metrics_collector.record_custom_metric(
                f"health_check_{checker.name}",
                0,
                labels={'checker': checker.name, 'status': 'timeout'}
            )
        except Exception as e:
            logger.error(f"Health check error for {checker.name}: {e}")
            self.metrics_collector.record_custom_metric(
                f"health_check_{checker.name}",
                0,
                labels={'checker': checker.name, 'status': 'error'},
                metadata={'error': str(e)}
            )
    
    def _default_alert_callback(self, alert: Alert):
        """Default alert callback"""
        if alert.resolved:
            logger.info(f"[ALERT RESOLVED] {alert.name}: {alert.message}")
        else:
            level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.ERROR,
                AlertSeverity.FATAL: logging.CRITICAL
            }.get(alert.severity, logging.WARNING)
            
            logger.log(level, f"[ALERT {alert.severity.level_name.upper()}] {alert.name}: {alert.message}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Get latest metrics
        recent_metrics = self.metrics_collector.get_metrics(hours=0.1)  # Last 6 minutes
        
        # Get latest system metrics
        system_metrics = None
        for metric in reversed(recent_metrics):
            if metric.name == "system_metrics" and metric.metadata:
                system_metrics = metric.metadata
                break
        
        # Get alert status
        active_alerts = self.alert_manager.get_active_alerts()
        alert_stats = self.alert_manager.get_alert_statistics()
        
        # Determine overall health
        overall_status = HealthStatus.HEALTHY
        if any(a.severity in [AlertSeverity.CRITICAL, AlertSeverity.FATAL] for a in active_alerts):
            overall_status = HealthStatus.CRITICAL
        elif any(a.severity == AlertSeverity.WARNING for a in active_alerts):
            overall_status = HealthStatus.WARNING
        
        return {
            'timestamp': time.time(),
            'overall_status': overall_status.value,
            'system_metrics': system_metrics,
            'active_alerts_count': len(active_alerts),
            'critical_alerts_count': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'health_checkers_count': len(self.health_checkers),
            'metrics_collected': len(recent_metrics),
            'alert_statistics': alert_stats
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            'system_status': self.get_system_status(),
            'active_alerts': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
            'metrics_summary': {
                'cpu': self.metrics_collector.get_metric_statistics('system_metrics', hours=1),
                'memory': self.metrics_collector.get_metric_statistics('system_metrics', hours=1),
                'disk': self.metrics_collector.get_metric_statistics('system_metrics', hours=1)
            },
            'health_checks': {
                name: {'name': name, 'timeout': checker.timeout, 'critical': checker.is_critical()}
                for name, checker in self.health_checkers.items()
            }
        }

# Global monitor instance
_global_monitor: Optional[EnterpriseMonitor] = None

def get_global_monitor(**kwargs) -> EnterpriseMonitor:
    """Get or create global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = EnterpriseMonitor(**kwargs)
    return _global_monitor

def start_monitoring(**kwargs):
    """Start global monitoring"""
    monitor = get_global_monitor(**kwargs)
    monitor.start()
    return monitor

def stop_monitoring():
    """Stop global monitoring"""
    if _global_monitor:
        _global_monitor.stop()

# Context manager for monitoring
@contextmanager
def monitoring_context(**kwargs):
    """Context manager for monitoring"""
    monitor = start_monitoring(**kwargs)
    try:
        yield monitor
    finally:
        stop_monitoring()

# Decorator for performance monitoring
def monitor_performance(metric_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_global_monitor()
            name = metric_name or f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                monitor.metrics_collector.record_custom_metric(
                    f"function_duration_{name}",
                    duration,
                    labels={'function': name, 'success': str(success)}
                )
        
        return wrapper
    return decorator

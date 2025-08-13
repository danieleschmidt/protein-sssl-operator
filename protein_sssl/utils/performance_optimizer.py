"""
Advanced Performance Optimization for protein-sssl-operator
Provides performance monitoring, optimization strategies, and scaling capabilities
"""

import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import queue
import weakref
import gc
import psutil
import os
from pathlib import Path

# Mock GPU utilities for systems without CUDA
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False

@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    execution_time: float = 0.0
    memory_usage: float = 0.0  # MB
    cpu_usage: float = 0.0     # %
    gpu_usage: float = 0.0     # %
    gpu_memory: float = 0.0    # MB
    throughput: float = 0.0    # items/second
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_efficiency: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ResourceLimits:
    """Resource usage limits"""
    max_memory_mb: float = 8192  # 8GB default
    max_cpu_percent: float = 80.0
    max_gpu_memory_mb: float = 4096  # 4GB default
    max_threads: int = multiprocessing.cpu_count()
    timeout_seconds: float = 300.0  # 5 minutes

class PerformanceMonitor:
    """Real-time performance monitoring and profiling"""
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 history_size: int = 1000):
        self.collection_interval = collection_interval
        self.history_size = history_size
        
        # Metrics storage
        self.metrics_history = deque(maxlen=history_size)
        self.function_stats = defaultdict(list)
        self.resource_alerts = []
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Process handle
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start background performance monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    self._check_resource_alerts(metrics)
                    
            except Exception as e:
                print(f"Monitoring error: {e}")
                
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        metrics = PerformanceMetrics()
        
        # CPU and Memory
        metrics.cpu_usage = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        metrics.memory_usage = memory_info.rss / 1024 / 1024  # MB
        
        # GPU metrics (if available)
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                metrics.gpu_usage = gpu_util.gpu
                metrics.gpu_memory = gpu_mem.used / 1024 / 1024  # MB
            except Exception:
                pass  # GPU metrics not available
        
        return metrics
    
    def _check_resource_alerts(self, metrics: PerformanceMetrics):
        """Check for resource usage alerts"""
        alerts = []
        
        if metrics.memory_usage > 7000:  # 7GB warning
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}MB")
            
        if metrics.cpu_usage > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            
        if metrics.gpu_memory > 3500:  # 3.5GB warning
            alerts.append(f"High GPU memory: {metrics.gpu_memory:.1f}MB")
        
        if alerts:
            self.resource_alerts.extend(alerts)
            # Keep only recent alerts
            self.resource_alerts = self.resource_alerts[-10:]
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self._collect_system_metrics()
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics from metrics history"""
        if not self.metrics_history:
            return {}
            
        with self._lock:
            history = list(self.metrics_history)
        
        if not history:
            return {}
        
        return {
            'avg_memory_mb': sum(m.memory_usage for m in history) / len(history),
            'max_memory_mb': max(m.memory_usage for m in history),
            'avg_cpu_percent': sum(m.cpu_usage for m in history) / len(history),
            'max_cpu_percent': max(m.cpu_usage for m in history),
            'avg_gpu_memory_mb': sum(m.gpu_memory for m in history) / len(history),
            'max_gpu_memory_mb': max(m.gpu_memory for m in history),
            'sample_count': len(history)
        }

class SmartCache:
    """Intelligent caching system with memory management"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: float = 512,
                 ttl_seconds: float = 3600):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.ttl_seconds = ttl_seconds
        
        self._cache = {}
        self._access_times = {}
        self._access_counts = defaultdict(int)
        self._memory_usage = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Any:
        """Get item from cache"""
        with self._lock:
            if key in self._cache:
                entry_time, value = self._cache[key]
                
                # Check TTL
                if time.time() - entry_time > self.ttl_seconds:
                    self._remove_key(key)
                    return None
                
                # Update access tracking
                self._access_times[key] = time.time()
                self._access_counts[key] += 1
                return value
                
        return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache"""
        with self._lock:
            # Estimate memory usage
            item_size = self._estimate_size(value)
            
            # Check if we need to evict items
            while (len(self._cache) >= self.max_size or 
                   self._memory_usage + item_size > self.max_memory_mb * 1024 * 1024):
                if not self._evict_lru():
                    return False  # Cache full, can't evict
            
            # Store item
            current_time = time.time()
            self._cache[key] = (current_time, value)
            self._access_times[key] = current_time
            self._access_counts[key] = 1
            self._memory_usage += item_size
            
            return True
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj[:100])  # Sample first 100
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in list(obj.items())[:100])  # Sample first 100
            else:
                return 1024  # Default estimate
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self._cache:
            return False
            
        # Find LRU key
        lru_key = min(self._access_times.keys(), 
                     key=self._access_times.get)
        
        self._remove_key(lru_key)
        return True
    
    def _remove_key(self, key: str):
        """Remove key from cache"""
        if key in self._cache:
            _, value = self._cache.pop(key)
            self._memory_usage -= self._estimate_size(value)
            
        self._access_times.pop(key, None)
        self._access_counts.pop(key, None)
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._memory_usage = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._memory_usage / 1024 / 1024,
                'max_memory_mb': self.max_memory_mb,
                'hit_rate': self._calculate_hit_rate(),
                'most_accessed': self._get_most_accessed(5)
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_accesses = sum(self._access_counts.values())
        if total_accesses == 0:
            return 0.0
        return len(self._access_counts) / total_accesses
    
    def _get_most_accessed(self, n: int) -> List[Tuple[str, int]]:
        """Get most accessed cache keys"""
        return sorted(self._access_counts.items(), 
                     key=lambda x: x[1], reverse=True)[:n]

class ParallelProcessor:
    """Intelligent parallel processing with automatic scaling"""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_processes: bool = False,
                 chunk_size: Optional[int] = None):
        self.max_workers = max_workers or min(32, multiprocessing.cpu_count() + 4)
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        
        self.performance_monitor = PerformanceMonitor()
        self.execution_stats = defaultdict(list)
        
    def process_batch(self, 
                     func: Callable,
                     items: List[Any],
                     progress_callback: Optional[Callable] = None) -> List[Any]:
        """Process items in parallel with automatic optimization"""
        
        if not items:
            return []
        
        # Determine optimal chunk size
        optimal_chunk_size = self._calculate_optimal_chunk_size(
            len(items), self.max_workers
        )
        
        # Split into chunks
        chunks = self._create_chunks(items, optimal_chunk_size)
        
        # Choose execution strategy
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        results = []
        start_time = time.time()
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all chunks
                future_to_chunk = {
                    executor.submit(self._process_chunk, func, chunk): i 
                    for i, chunk in enumerate(chunks)
                }
                
                # Collect results
                completed_chunks = 0
                chunk_results = [None] * len(chunks)
                
                for future in as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    
                    try:
                        chunk_result = future.result()
                        chunk_results[chunk_index] = chunk_result
                        completed_chunks += 1
                        
                        # Progress callback
                        if progress_callback:
                            progress = completed_chunks / len(chunks)
                            progress_callback(progress)
                            
                    except Exception as e:
                        print(f"Chunk {chunk_index} failed: {e}")
                        chunk_results[chunk_index] = []
                
                # Flatten results maintaining order
                for chunk_result in chunk_results:
                    if chunk_result is not None:
                        results.extend(chunk_result)
        
        finally:
            self.performance_monitor.stop_monitoring()
        
        # Record performance stats
        total_time = time.time() - start_time
        throughput = len(items) / total_time if total_time > 0 else 0
        
        self.execution_stats[func.__name__].append({
            'items': len(items),
            'time': total_time,
            'throughput': throughput,
            'chunks': len(chunks),
            'workers': self.max_workers
        })
        
        return results
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a single chunk of items"""
        return [func(item) for item in chunk]
    
    def _calculate_optimal_chunk_size(self, total_items: int, num_workers: int) -> int:
        """Calculate optimal chunk size for parallel processing"""
        if self.chunk_size:
            return self.chunk_size
            
        # Heuristic: aim for 2-4 chunks per worker
        target_chunks = num_workers * 3
        optimal_size = max(1, total_items // target_chunks)
        
        # Ensure reasonable bounds
        return max(1, min(optimal_size, 1000))
    
    def _create_chunks(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split items into chunks"""
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i:i + chunk_size])
        return chunks
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for func_name, executions in self.execution_stats.items():
            if executions:
                avg_throughput = sum(e['throughput'] for e in executions) / len(executions)
                avg_time = sum(e['time'] for e in executions) / len(executions)
                total_items = sum(e['items'] for e in executions)
                
                stats[func_name] = {
                    'executions': len(executions),
                    'total_items_processed': total_items,
                    'avg_throughput_per_sec': avg_throughput,
                    'avg_execution_time_sec': avg_time,
                    'latest_execution': executions[-1]
                }
        
        return stats

class MemoryOptimizer:
    """Memory usage optimization and garbage collection management"""
    
    def __init__(self):
        self.gc_stats = []
        self.memory_snapshots = deque(maxlen=100)
        self._last_gc_time = 0
        
    def optimize_memory(self, aggressive: bool = False):
        """Optimize memory usage"""
        start_memory = self._get_memory_usage()
        
        if aggressive:
            # Aggressive optimization
            gc.collect()  # Full GC
            gc.collect()  # Second pass
            gc.collect()  # Third pass for good measure
        else:
            # Standard optimization
            if time.time() - self._last_gc_time > 60:  # At most once per minute
                gc.collect()
                self._last_gc_time = time.time()
        
        end_memory = self._get_memory_usage()
        freed_mb = start_memory - end_memory
        
        self.gc_stats.append({
            'timestamp': time.time(),
            'freed_mb': freed_mb,
            'aggressive': aggressive,
            'final_memory_mb': end_memory
        })
        
        return freed_mb
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def track_memory_usage(self, label: str = ""):
        """Take a memory usage snapshot"""
        usage = self._get_memory_usage()
        self.memory_snapshots.append({
            'timestamp': time.time(),
            'memory_mb': usage,
            'label': label
        })
        return usage
    
    def get_memory_trend(self) -> Dict[str, float]:
        """Get memory usage trend analysis"""
        if len(self.memory_snapshots) < 2:
            return {'trend': 0.0, 'current_mb': self._get_memory_usage()}
        
        snapshots = list(self.memory_snapshots)
        recent = snapshots[-min(10, len(snapshots)):]  # Last 10 snapshots
        
        if len(recent) < 2:
            return {'trend': 0.0, 'current_mb': recent[-1]['memory_mb']}
        
        # Calculate trend (MB per snapshot)
        x = list(range(len(recent)))
        y = [s['memory_mb'] for s in recent]
        
        # Simple linear regression
        n = len(recent)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        else:
            slope = 0.0
        
        return {
            'trend_mb_per_snapshot': slope,
            'current_mb': y[-1],
            'peak_mb': max(y),
            'min_mb': min(y),
            'snapshots_analyzed': n
        }

def performance_profile(func: Callable = None, *, 
                       cache_results: bool = False,
                       memory_tracking: bool = False):
    """Decorator for performance profiling"""
    
    def decorator(f):
        # Initialize cache if requested
        if cache_results:
            f._cache = SmartCache()
        
        # Initialize memory optimizer if requested
        if memory_tracking:
            f._memory_optimizer = MemoryOptimizer()
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Check cache first
            if cache_results and hasattr(f, '_cache'):
                cache_key = str(args) + str(sorted(kwargs.items()))
                cached_result = f._cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Track memory before execution
            if memory_tracking and hasattr(f, '_memory_optimizer'):
                f._memory_optimizer.track_memory_usage(f"before_{f.__name__}")
            
            # Execute with timing
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
            finally:
                execution_time = time.time() - start_time
            
            # Store in cache
            if cache_results and hasattr(f, '_cache'):
                f._cache.put(cache_key, result)
            
            # Track memory after execution
            if memory_tracking and hasattr(f, '_memory_optimizer'):
                f._memory_optimizer.track_memory_usage(f"after_{f.__name__}")
                
                # Periodic memory optimization
                if hasattr(f, '_last_memory_check'):
                    if time.time() - f._last_memory_check > 300:  # 5 minutes
                        freed = f._memory_optimizer.optimize_memory()
                        if freed > 100:  # Significant memory freed
                            print(f"Memory optimizer freed {freed:.1f}MB after {f.__name__}")
                        f._last_memory_check = time.time()
                else:
                    f._last_memory_check = time.time()
            
            return result
        
        # Add profiling attributes
        wrapper._performance_stats = []
        wrapper._original_function = f
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

# Global instances
_global_monitor = PerformanceMonitor()
_global_cache = SmartCache()
_global_parallel_processor = ParallelProcessor()
_global_memory_optimizer = MemoryOptimizer()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor"""
    return _global_monitor

def get_smart_cache() -> SmartCache:
    """Get global smart cache"""
    return _global_cache

def get_parallel_processor() -> ParallelProcessor:
    """Get global parallel processor"""
    return _global_parallel_processor

def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer"""
    return _global_memory_optimizer

def optimize_system_performance():
    """Optimize overall system performance"""
    print("🚀 Optimizing system performance...")
    
    # Memory optimization
    freed_mb = _global_memory_optimizer.optimize_memory(aggressive=True)
    print(f"   Freed {freed_mb:.1f}MB memory")
    
    # Cache cleanup
    cache_stats = _global_cache.stats()
    print(f"   Cache: {cache_stats['size']} items, {cache_stats['memory_usage_mb']:.1f}MB")
    
    # Performance monitoring
    _global_monitor.start_monitoring()
    current_metrics = _global_monitor.get_current_metrics()
    print(f"   Current: {current_metrics.memory_usage:.1f}MB RAM, {current_metrics.cpu_usage:.1f}% CPU")
    
    return {
        'memory_freed_mb': freed_mb,
        'cache_stats': cache_stats,
        'current_metrics': current_metrics
    }
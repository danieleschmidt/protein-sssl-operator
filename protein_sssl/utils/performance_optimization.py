"""
Advanced performance optimization utilities.
"""
import time
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import queue
from dataclasses import dataclass
from collections import defaultdict

from .logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization analysis."""
    execution_time: float
    memory_peak_mb: float
    cpu_utilization: float
    throughput_items_per_sec: float
    cache_hit_rate: Optional[float] = None
    parallel_efficiency: Optional[float] = None


class AdaptiveCache:
    """Adaptive LRU cache with performance monitoring."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: Optional[float] = None,
        size_tracking: bool = True
    ):
        self.max_size = max_size
        self.ttl = ttl
        self.size_tracking = size_tracking
        
        self._cache = {}
        self._access_times = {}
        self._access_counts = defaultdict(int)
        self._insertion_order = []
        self._total_requests = 0
        self._cache_hits = 0
        self._lock = threading.RLock()
        
        # Performance tracking
        self._hit_rate_history = []
        self._size_history = []
        self._last_optimization = time.time()
        self._optimization_interval = 300  # 5 minutes
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get item from cache with performance tracking."""
        with self._lock:
            self._total_requests += 1
            current_time = time.time()
            
            if key in self._cache:
                # Check TTL if enabled
                if self.ttl and current_time - self._access_times[key] > self.ttl:
                    self._remove_key(key)
                    return default
                
                # Update access tracking
                self._access_times[key] = current_time
                self._access_counts[key] += 1
                self._cache_hits += 1
                
                return self._cache[key]
            
            return default
    
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache with adaptive eviction."""
        with self._lock:
            current_time = time.time()
            
            # Add new item
            if key not in self._cache:
                if len(self._cache) >= self.max_size:
                    self._evict_items()
                
                self._insertion_order.append(key)
            
            self._cache[key] = value
            self._access_times[key] = current_time
            self._access_counts[key] += 1
            
            # Periodic optimization
            if current_time - self._last_optimization > self._optimization_interval:
                self._optimize_cache()
    
    def _evict_items(self) -> None:
        """Intelligent cache eviction based on access patterns."""
        # Calculate scores for eviction (lower = more likely to evict)
        scores = {}
        current_time = time.time()
        
        for key in self._cache:
            # Factors: recency, frequency, and age
            recency = current_time - self._access_times[key]
            frequency = self._access_counts[key]
            
            # Score combines recency (lower is better) and frequency (higher is better)
            scores[key] = frequency / (1 + recency)
        
        # Sort by score and remove lowest scoring items
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])
        num_to_remove = max(1, len(self._cache) // 10)  # Remove 10% or at least 1
        
        for key in sorted_keys[:num_to_remove]:
            self._remove_key(key)
    
    def _remove_key(self, key: Any) -> None:
        """Remove key from all cache structures."""
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]
            del self._access_counts[key]
            if key in self._insertion_order:
                self._insertion_order.remove(key)
    
    def _optimize_cache(self) -> None:
        """Optimize cache parameters based on usage patterns."""
        current_hit_rate = self.get_hit_rate()
        self._hit_rate_history.append(current_hit_rate)
        
        # Keep only recent history
        if len(self._hit_rate_history) > 100:
            self._hit_rate_history = self._hit_rate_history[-100:]
        
        # Adaptive size adjustment
        if len(self._hit_rate_history) >= 10:
            recent_hit_rate = sum(self._hit_rate_history[-10:]) / 10
            
            if recent_hit_rate < 0.5 and self.max_size > 100:
                # Low hit rate, reduce cache size
                self.max_size = max(100, int(self.max_size * 0.9))
                logger.debug(f"Reduced cache size to {self.max_size}")
            elif recent_hit_rate > 0.8 and len(self._cache) >= self.max_size * 0.9:
                # High hit rate and cache is full, increase size
                self.max_size = min(10000, int(self.max_size * 1.1))
                logger.debug(f"Increased cache size to {self.max_size}")
        
        self._last_optimization = time.time()
    
    def get_hit_rate(self) -> float:
        """Get current cache hit rate."""
        if self._total_requests == 0:
            return 0.0
        return self._cache_hits / self._total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': self.get_hit_rate(),
                'total_requests': self._total_requests,
                'cache_hits': self._cache_hits,
                'avg_hit_rate_recent': (
                    sum(self._hit_rate_history[-10:]) / len(self._hit_rate_history[-10:])
                    if self._hit_rate_history else 0.0
                ),
                'most_accessed': sorted(
                    self._access_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._insertion_order.clear()


def adaptive_cache(
    max_size: int = 1000,
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """Adaptive caching decorator with performance optimization."""
    def decorator(func: Callable) -> Callable:
        cache = AdaptiveCache(max_size=max_size, ttl=ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = (args, tuple(sorted(kwargs.items())))
            
            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        # Attach cache stats to wrapper
        wrapper.cache_stats = cache.get_stats
        wrapper.cache_clear = cache.clear
        
        return wrapper
    
    return decorator


class ParallelProcessor:
    """Advanced parallel processing with dynamic optimization."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        chunk_size: Optional[int] = None,
        adaptive_scheduling: bool = True
    ):
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self.adaptive_scheduling = adaptive_scheduling
        
        # Performance tracking
        self.execution_times = []
        self.throughput_history = []
        self.optimal_chunk_size = chunk_size or 1
        self.optimal_workers = self.max_workers
    
    def process_parallel(
        self,
        func: Callable,
        items: List[Any],
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process items in parallel with adaptive optimization."""
        if not items:
            return []
        
        start_time = time.time()
        
        # Determine optimal parameters
        if self.adaptive_scheduling:
            self._optimize_parameters(len(items))
        
        # Choose executor type
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        # Process in parallel
        results = []
        with executor_class(max_workers=self.optimal_workers) as executor:
            # Submit tasks in chunks
            futures = []
            for chunk in self._create_chunks(items, self.optimal_chunk_size):
                future = executor.submit(self._process_chunk, func, chunk)
                futures.append(future)
            
            # Collect results with progress tracking
            completed = 0
            total_chunks = len(futures)
            
            for future in as_completed(futures):
                chunk_results = future.result()
                results.extend(chunk_results)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total_chunks)
        
        # Update performance metrics
        execution_time = time.time() - start_time
        throughput = len(items) / execution_time if execution_time > 0 else 0
        
        self.execution_times.append(execution_time)
        self.throughput_history.append(throughput)
        
        # Keep only recent history
        if len(self.execution_times) > 50:
            self.execution_times = self.execution_times[-50:]
            self.throughput_history = self.throughput_history[-50:]
        
        logger.debug(
            f"Processed {len(items)} items in {execution_time:.2f}s "
            f"(throughput: {throughput:.1f} items/s)"
        )
        
        return results
    
    def _create_chunks(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        """Create chunks of optimal size."""
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i:i + chunk_size])
        return chunks
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        results = []
        for item in chunk:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process item {item}: {e}")
                results.append(None)
        return results
    
    def _optimize_parameters(self, num_items: int) -> None:
        """Optimize worker count and chunk size based on historical performance."""
        if len(self.throughput_history) < 5:
            return
        
        # Calculate recent average throughput
        recent_throughput = sum(self.throughput_history[-5:]) / 5
        
        # Adaptive chunk size optimization
        if num_items > 100:
            ideal_chunk_size = max(1, num_items // (self.optimal_workers * 4))
            self.optimal_chunk_size = min(ideal_chunk_size, 1000)
        else:
            self.optimal_chunk_size = max(1, num_items // self.optimal_workers)
        
        # Worker count optimization (simplified heuristic)
        if recent_throughput > 0:
            cpu_bound_threshold = 1000  # items/second
            if recent_throughput < cpu_bound_threshold and not self.use_processes:
                # Might benefit from more threads for I/O bound tasks
                self.optimal_workers = min(self.max_workers * 2, mp.cpu_count() * 4)
            else:
                self.optimal_workers = self.max_workers
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.execution_times:
            return {'message': 'No execution history available'}
        
        recent_times = self.execution_times[-10:]
        recent_throughput = self.throughput_history[-10:]
        
        return {
            'avg_execution_time': sum(recent_times) / len(recent_times),
            'avg_throughput': sum(recent_throughput) / len(recent_throughput),
            'optimal_workers': self.optimal_workers,
            'optimal_chunk_size': self.optimal_chunk_size,
            'total_executions': len(self.execution_times)
        }


class MemoryOptimizer:
    """Memory usage optimization utilities."""
    
    def __init__(self, gc_threshold: float = 0.8):
        self.gc_threshold = gc_threshold
        self._peak_memory = 0
        self._last_gc = time.time()
        
    def optimize_memory_usage(self, func: Callable) -> Callable:
        """Decorator for memory-optimized function execution."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import gc
            import psutil
            
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024**2
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024**2
                memory_increase = current_memory - initial_memory
                
                # Track peak memory
                if current_memory > self._peak_memory:
                    self._peak_memory = current_memory
                
                # Trigger garbage collection if memory usage is high
                if memory_increase > 100:  # >100MB increase
                    gc.collect()
                    logger.debug(f"Garbage collection triggered after {memory_increase:.1f}MB increase")
                
                return result
                
            except MemoryError:
                logger.error("Memory error encountered, forcing garbage collection")
                gc.collect()
                raise
        
        return wrapper
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'current_mb': memory_info.rss / 1024**2,
            'peak_mb': self._peak_memory,
            'virtual_mb': memory_info.vms / 1024**2,
            'memory_percent': process.memory_percent()
        }


class BatchOptimizer:
    """Optimize batch processing for different workloads."""
    
    def __init__(self):
        self.batch_performance = {}  # workload_type -> optimal_batch_size
        
    def find_optimal_batch_size(
        self,
        process_func: Callable,
        sample_data: List[Any],
        workload_type: str,
        max_batch_size: int = 512,
        min_batch_size: int = 1
    ) -> int:
        """Find optimal batch size through empirical testing."""
        if workload_type in self.batch_performance:
            return self.batch_performance[workload_type]
        
        batch_sizes_to_test = [
            min_batch_size, 2, 4, 8, 16, 32, 64, 128, 256, max_batch_size
        ]
        batch_sizes_to_test = [b for b in batch_sizes_to_test if min_batch_size <= b <= max_batch_size]
        
        performance_results = {}
        
        # Use small sample for testing
        test_sample = sample_data[:min(100, len(sample_data))]
        
        for batch_size in batch_sizes_to_test:
            try:
                start_time = time.time()
                
                # Process in batches
                for i in range(0, len(test_sample), batch_size):
                    batch = test_sample[i:i + batch_size]
                    process_func(batch)
                
                execution_time = time.time() - start_time
                throughput = len(test_sample) / execution_time if execution_time > 0 else 0
                performance_results[batch_size] = throughput
                
                logger.debug(f"Batch size {batch_size}: {throughput:.1f} items/s")
                
            except Exception as e:
                logger.warning(f"Failed to test batch size {batch_size}: {e}")
                continue
        
        # Find optimal batch size
        if performance_results:
            optimal_batch_size = max(performance_results.keys(), key=lambda k: performance_results[k])
            self.batch_performance[workload_type] = optimal_batch_size
            
            logger.info(f"Optimal batch size for {workload_type}: {optimal_batch_size}")
            return optimal_batch_size
        
        # Fallback to reasonable default
        return 32


# Global instances for shared optimization
adaptive_cache_global = AdaptiveCache(max_size=5000, ttl=3600)  # 1 hour TTL
parallel_processor = ParallelProcessor(adaptive_scheduling=True)
memory_optimizer = MemoryOptimizer()
batch_optimizer = BatchOptimizer()


def profile_performance(name: str):
    """Decorator to profile function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import psutil
            
            # Start metrics
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss
            start_cpu = process.cpu_percent()
            
            try:
                result = func(*args, **kwargs)
                
                # End metrics
                end_time = time.time()
                end_memory = process.memory_info().rss
                end_cpu = process.cpu_percent()
                
                # Calculate metrics
                execution_time = end_time - start_time
                memory_delta = (end_memory - start_memory) / 1024**2  # MB
                cpu_utilization = (start_cpu + end_cpu) / 2
                
                logger.info(
                    f"Performance [{name}]: "
                    f"time={execution_time:.3f}s, "
                    f"memory_delta={memory_delta:.1f}MB, "
                    f"cpu={cpu_utilization:.1f}%"
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Performance profiling failed for {name}: {e}")
                raise
        
        return wrapper
    return decorator
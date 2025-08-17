"""
Advanced Memory Optimization System for Protein-SSL Operator
Implements sophisticated memory management, garbage collection, and memory pooling
"""

import gc
import threading
import time
import mmap
import ctypes
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar
from dataclasses import dataclass
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import numpy as np
import torch
import psutil
import os
import pickle
from pathlib import Path
import logging
from contextlib import contextmanager
from enum import Enum
import resource
import tracemalloc

from .logging_config import setup_logging
from .monitoring import MetricsCollector

logger = setup_logging(__name__)

T = TypeVar('T')


class MemoryPressureLevel(Enum):
    """Memory pressure levels for adaptive management"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_mb: float
    available_mb: float
    used_mb: float
    used_percent: float
    swap_used_mb: float
    gc_collections: Dict[str, int]
    peak_usage_mb: float
    fragmentation_ratio: float
    pressure_level: MemoryPressureLevel


@dataclass
class AllocationMetrics:
    """Memory allocation tracking metrics"""
    total_allocations: int
    total_deallocations: int
    current_allocations: int
    peak_allocations: int
    total_allocated_bytes: int
    total_deallocated_bytes: int
    current_allocated_bytes: int
    peak_allocated_bytes: int
    allocation_rate_per_sec: float
    deallocation_rate_per_sec: float


class MemoryPool:
    """High-performance memory pool for reusing objects"""
    
    def __init__(self, 
                 object_factory: Callable[[], T],
                 initial_size: int = 10,
                 max_size: int = 1000,
                 cleanup_func: Optional[Callable[[T], None]] = None):
        
        self.object_factory = object_factory
        self.max_size = max_size
        self.cleanup_func = cleanup_func
        
        # Pool storage
        self._pool = deque(maxlen=max_size)
        self._in_use = set()
        self._lock = threading.RLock()
        
        # Metrics
        self.total_created = 0
        self.total_reused = 0
        self.current_pool_size = 0
        
        # Pre-populate pool
        self._populate_pool(initial_size)
    
    def acquire(self) -> T:
        """Acquire object from pool"""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self.total_reused += 1
            else:
                obj = self.object_factory()
                self.total_created += 1
            
            self._in_use.add(id(obj))
            self.current_pool_size = len(self._pool)
            return obj
    
    def release(self, obj: T) -> None:
        """Release object back to pool"""
        with self._lock:
            obj_id = id(obj)
            if obj_id not in self._in_use:
                logger.warning("Attempting to release object not from this pool")
                return
            
            self._in_use.discard(obj_id)
            
            # Clean up object if cleanup function provided
            if self.cleanup_func:
                try:
                    self.cleanup_func(obj)
                except Exception as e:
                    logger.warning(f"Object cleanup failed: {e}")
                    return  # Don't return to pool if cleanup failed
            
            # Return to pool if there's space
            if len(self._pool) < self.max_size:
                self._pool.append(obj)
            
            self.current_pool_size = len(self._pool)
    
    def _populate_pool(self, count: int) -> None:
        """Pre-populate pool with objects"""
        for _ in range(count):
            try:
                obj = self.object_factory()
                self._pool.append(obj)
                self.total_created += 1
            except Exception as e:
                logger.warning(f"Failed to pre-populate pool: {e}")
                break
        
        self.current_pool_size = len(self._pool)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                'current_pool_size': self.current_pool_size,
                'max_size': self.max_size,
                'in_use_count': len(self._in_use),
                'total_created': self.total_created,
                'total_reused': self.total_reused,
                'reuse_ratio': self.total_reused / max(self.total_created + self.total_reused, 1)
            }
    
    def clear(self) -> None:
        """Clear the pool"""
        with self._lock:
            self._pool.clear()
            self._in_use.clear()
            self.current_pool_size = 0


@contextmanager
def pooled_object(pool: MemoryPool[T]):
    """Context manager for using pooled objects"""
    obj = pool.acquire()
    try:
        yield obj
    finally:
        pool.release(obj)


class TensorMemoryPool:
    """Specialized memory pool for PyTorch tensors"""
    
    def __init__(self, device: str = 'cpu', max_tensors: int = 1000):
        self.device = device
        self.max_tensors = max_tensors
        
        # Pools organized by shape and dtype
        self._pools: Dict[Tuple[tuple, torch.dtype], deque] = defaultdict(lambda: deque(maxlen=100))
        self._in_use = set()
        self._lock = threading.RLock()
        
        # Statistics
        self.total_created = 0
        self.total_reused = 0
        self.memory_saved_bytes = 0
    
    def acquire_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Acquire tensor from pool or create new one"""
        with self._lock:
            pool_key = (shape, dtype)
            pool = self._pools[pool_key]
            
            if pool:
                tensor = pool.popleft()
                tensor.zero_()  # Clear the tensor
                self.total_reused += 1
                self.memory_saved_bytes += tensor.numel() * tensor.element_size()
            else:
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                self.total_created += 1
            
            self._in_use.add(id(tensor))
            return tensor
    
    def release_tensor(self, tensor: torch.Tensor) -> None:
        """Release tensor back to pool"""
        with self._lock:
            tensor_id = id(tensor)
            if tensor_id not in self._in_use:
                return
            
            self._in_use.discard(tensor_id)
            
            # Check if we should return to pool
            pool_key = (tuple(tensor.shape), tensor.dtype)
            pool = self._pools[pool_key]
            
            # Only pool tensors that aren't too large and fit our criteria
            tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            if len(pool) < 100 and tensor_size_mb < 100:  # Don't pool very large tensors
                pool.append(tensor.detach())
    
    def clear_device_cache(self) -> None:
        """Clear device cache for GPU tensors"""
        if self.device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tensor pool statistics"""
        with self._lock:
            total_pooled = sum(len(pool) for pool in self._pools.values())
            
            return {
                'total_pools': len(self._pools),
                'total_pooled_tensors': total_pooled,
                'in_use_count': len(self._in_use),
                'total_created': self.total_created,
                'total_reused': self.total_reused,
                'memory_saved_mb': self.memory_saved_bytes / (1024 * 1024),
                'reuse_ratio': self.total_reused / max(self.total_created + self.total_reused, 1)
            }


class SmartGarbageCollector:
    """Intelligent garbage collection with adaptive thresholds"""
    
    def __init__(self, 
                 memory_threshold: float = 0.8,
                 gc_interval: float = 30.0,
                 adaptive_tuning: bool = True):
        
        self.memory_threshold = memory_threshold
        self.gc_interval = gc_interval
        self.adaptive_tuning = adaptive_tuning
        
        # GC statistics
        self.gc_stats = {
            'collections': defaultdict(int),
            'objects_collected': defaultdict(int),
            'time_spent': defaultdict(float),
            'memory_freed': defaultdict(int)
        }
        
        # Adaptive parameters
        self.dynamic_threshold = memory_threshold
        self.dynamic_interval = gc_interval
        
        # Background GC thread
        self._gc_thread = None
        self._gc_active = False
        self._lock = threading.Lock()
        
        # Memory pressure monitoring
        self._memory_history = deque(maxlen=100)
        self._last_gc_time = 0
        
        # Configure GC
        self._configure_gc()
    
    def _configure_gc(self) -> None:
        """Configure garbage collection parameters"""
        # Set more aggressive GC thresholds for better memory management
        gc.set_threshold(700, 10, 10)  # More frequent collection
        
        # Enable debug stats
        if logger.isEnabledFor(logging.DEBUG):
            gc.set_debug(gc.DEBUG_STATS)
    
    def start_background_gc(self) -> None:
        """Start background garbage collection"""
        if self._gc_active:
            return
        
        self._gc_active = True
        self._gc_thread = threading.Thread(target=self._gc_loop, daemon=True)
        self._gc_thread.start()
        logger.info("Background garbage collection started")
    
    def stop_background_gc(self) -> None:
        """Stop background garbage collection"""
        self._gc_active = False
        if self._gc_thread:
            self._gc_thread.join(timeout=10.0)
        logger.info("Background garbage collection stopped")
    
    def _gc_loop(self) -> None:
        """Background garbage collection loop"""
        while self._gc_active:
            try:
                self._check_and_collect()
                time.sleep(self.dynamic_interval)
            except Exception as e:
                logger.warning(f"Background GC error: {e}")
                time.sleep(self.gc_interval)
    
    def _check_and_collect(self) -> None:
        """Check memory pressure and collect if needed"""
        memory_stats = self._get_memory_stats()
        self._memory_history.append(memory_stats.used_percent / 100.0)
        
        # Determine if collection is needed
        current_pressure = memory_stats.used_percent / 100.0
        should_collect = False
        
        if current_pressure > self.dynamic_threshold:
            should_collect = True
            reason = f"Memory pressure {current_pressure:.2f} > threshold {self.dynamic_threshold:.2f}"
        elif len(self._memory_history) >= 10:
            # Check for memory growth trend
            recent_trend = np.polyfit(range(10), list(self._memory_history)[-10:], 1)[0]
            if recent_trend > 0.01:  # 1% increase per measurement
                should_collect = True
                reason = f"Memory growth trend detected: {recent_trend:.4f}"
        
        if should_collect:
            collected = self.force_collection()
            logger.debug(f"Background GC triggered: {reason}, collected {collected} objects")
            
            # Adaptive tuning
            if self.adaptive_tuning:
                self._tune_parameters(memory_stats)
    
    def force_collection(self) -> int:
        """Force garbage collection and return number of objects collected"""
        with self._lock:
            start_time = time.perf_counter()
            start_memory = self._get_process_memory_mb()
            
            # Collect each generation
            total_collected = 0
            for generation in range(3):
                before_count = len(gc.get_objects())
                collected = gc.collect(generation)
                after_count = len(gc.get_objects())
                
                actual_collected = before_count - after_count
                total_collected += actual_collected
                
                self.gc_stats['collections'][generation] += 1
                self.gc_stats['objects_collected'][generation] += actual_collected
            
            # Calculate metrics
            end_time = time.perf_counter()
            end_memory = self._get_process_memory_mb()
            
            gc_time = end_time - start_time
            memory_freed = max(0, start_memory - end_memory)
            
            self.gc_stats['time_spent']['total'] += gc_time
            self.gc_stats['memory_freed']['total'] += memory_freed
            
            self._last_gc_time = time.time()
            
            logger.debug(f"GC completed: {total_collected} objects, {memory_freed:.1f}MB freed, {gc_time:.3f}s")
            
            return total_collected
    
    def _tune_parameters(self, memory_stats: MemoryStats) -> None:
        """Adaptively tune GC parameters based on performance"""
        # Adjust threshold based on memory pressure
        if memory_stats.pressure_level == MemoryPressureLevel.CRITICAL:
            self.dynamic_threshold = max(0.6, self.dynamic_threshold - 0.05)
            self.dynamic_interval = max(5.0, self.dynamic_interval * 0.8)
        elif memory_stats.pressure_level == MemoryPressureLevel.LOW:
            self.dynamic_threshold = min(0.9, self.dynamic_threshold + 0.02)
            self.dynamic_interval = min(60.0, self.dynamic_interval * 1.1)
        
        logger.debug(f"GC parameters tuned: threshold={self.dynamic_threshold:.2f}, interval={self.dynamic_interval:.1f}s")
    
    def _get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        used_percent = system_memory.percent
        
        # Determine pressure level
        if used_percent > 95:
            pressure_level = MemoryPressureLevel.CRITICAL
        elif used_percent > 85:
            pressure_level = MemoryPressureLevel.HIGH
        elif used_percent > 70:
            pressure_level = MemoryPressureLevel.MODERATE
        else:
            pressure_level = MemoryPressureLevel.LOW
        
        return MemoryStats(
            total_mb=system_memory.total / (1024 * 1024),
            available_mb=system_memory.available / (1024 * 1024),
            used_mb=system_memory.used / (1024 * 1024),
            used_percent=used_percent,
            swap_used_mb=psutil.swap_memory().used / (1024 * 1024),
            gc_collections=dict(self.gc_stats['collections']),
            peak_usage_mb=memory_info.rss / (1024 * 1024),
            fragmentation_ratio=memory_info.vms / max(memory_info.rss, 1),
            pressure_level=pressure_level
        )
    
    def _get_process_memory_mb(self) -> float:
        """Get current process memory usage in MB"""
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        return {
            'collections_by_generation': dict(self.gc_stats['collections']),
            'objects_collected_by_generation': dict(self.gc_stats['objects_collected']),
            'total_time_spent': self.gc_stats['time_spent']['total'],
            'total_memory_freed_mb': self.gc_stats['memory_freed']['total'],
            'current_threshold': self.dynamic_threshold,
            'current_interval': self.dynamic_interval,
            'last_gc_time': self._last_gc_time,
            'gc_active': self._gc_active
        }


class MemoryTracker:
    """Advanced memory allocation tracking and analysis"""
    
    def __init__(self, enable_tracemalloc: bool = True, track_allocations: bool = True):
        self.enable_tracemalloc = enable_tracemalloc
        self.track_allocations = track_allocations
        
        # Allocation tracking
        self._allocations: Dict[int, Dict[str, Any]] = {}
        self._allocation_history = deque(maxlen=10000)
        self._lock = threading.Lock()
        
        # Metrics
        self.metrics = AllocationMetrics(
            total_allocations=0,
            total_deallocations=0,
            current_allocations=0,
            peak_allocations=0,
            total_allocated_bytes=0,
            total_deallocated_bytes=0,
            current_allocated_bytes=0,
            peak_allocated_bytes=0,
            allocation_rate_per_sec=0.0,
            deallocation_rate_per_sec=0.0
        )
        
        # Rate calculation
        self._last_rate_calculation = time.time()
        self._last_allocations = 0
        self._last_deallocations = 0
        
        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def track_allocation(self, obj: Any, size_bytes: int, context: str = "") -> None:
        """Track memory allocation"""
        if not self.track_allocations:
            return
        
        with self._lock:
            obj_id = id(obj)
            allocation_info = {
                'size_bytes': size_bytes,
                'timestamp': time.time(),
                'context': context,
                'traceback': traceback.extract_stack(limit=5) if logger.isEnabledFor(logging.DEBUG) else None
            }
            
            self._allocations[obj_id] = allocation_info
            self._allocation_history.append(('alloc', obj_id, size_bytes, time.time()))
            
            # Update metrics
            self.metrics.total_allocations += 1
            self.metrics.current_allocations += 1
            self.metrics.total_allocated_bytes += size_bytes
            self.metrics.current_allocated_bytes += size_bytes
            
            # Update peaks
            if self.metrics.current_allocations > self.metrics.peak_allocations:
                self.metrics.peak_allocations = self.metrics.current_allocations
            
            if self.metrics.current_allocated_bytes > self.metrics.peak_allocated_bytes:
                self.metrics.peak_allocated_bytes = self.metrics.current_allocated_bytes
    
    def track_deallocation(self, obj: Any) -> None:
        """Track memory deallocation"""
        if not self.track_allocations:
            return
        
        with self._lock:
            obj_id = id(obj)
            if obj_id in self._allocations:
                allocation_info = self._allocations.pop(obj_id)
                size_bytes = allocation_info['size_bytes']
                
                self._allocation_history.append(('dealloc', obj_id, size_bytes, time.time()))
                
                # Update metrics
                self.metrics.total_deallocations += 1
                self.metrics.current_allocations -= 1
                self.metrics.total_deallocated_bytes += size_bytes
                self.metrics.current_allocated_bytes -= size_bytes
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get current memory snapshot"""
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            # Get top memory consuming lines
            top_lines = []
            for stat in top_stats[:10]:
                top_lines.append({
                    'filename': stat.traceback.format(),
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count
                })
            
            return {
                'tracemalloc_enabled': True,
                'top_memory_lines': top_lines,
                'total_traced_mb': sum(stat.size for stat in top_stats) / (1024 * 1024)
            }
        else:
            return {'tracemalloc_enabled': False}
    
    def analyze_allocation_patterns(self) -> Dict[str, Any]:
        """Analyze allocation patterns for optimization insights"""
        with self._lock:
            if not self._allocation_history:
                return {'error': 'No allocation history available'}
            
            # Convert to numpy for analysis
            recent_history = list(self._allocation_history)[-1000:]  # Last 1000 operations
            
            allocs = [entry for entry in recent_history if entry[0] == 'alloc']
            deallocs = [entry for entry in recent_history if entry[0] == 'dealloc']
            
            if not allocs:
                return {'error': 'No allocations in recent history'}
            
            # Size distribution analysis
            alloc_sizes = [entry[2] for entry in allocs]
            size_stats = {
                'mean_size_bytes': np.mean(alloc_sizes),
                'median_size_bytes': np.median(alloc_sizes),
                'std_size_bytes': np.std(alloc_sizes),
                'min_size_bytes': np.min(alloc_sizes),
                'max_size_bytes': np.max(alloc_sizes)
            }
            
            # Temporal analysis
            if len(allocs) >= 2:
                alloc_times = [entry[3] for entry in allocs]
                time_diffs = np.diff(alloc_times)
                temporal_stats = {
                    'mean_alloc_interval_ms': np.mean(time_diffs) * 1000,
                    'allocation_rate_per_sec': len(allocs) / max(alloc_times[-1] - alloc_times[0], 1)
                }
            else:
                temporal_stats = {
                    'mean_alloc_interval_ms': 0,
                    'allocation_rate_per_sec': 0
                }
            
            # Memory lifecycle analysis
            lifecycle_stats = {}
            if deallocs:
                # Find allocations that were deallocated
                dealloc_ids = {entry[1] for entry in deallocs}
                alloc_dict = {entry[1]: entry for entry in allocs}
                
                lifetimes = []
                for dealloc in deallocs:
                    obj_id = dealloc[1]
                    if obj_id in alloc_dict:
                        alloc_time = alloc_dict[obj_id][3]
                        dealloc_time = dealloc[3]
                        lifetime = dealloc_time - alloc_time
                        lifetimes.append(lifetime)
                
                if lifetimes:
                    lifecycle_stats = {
                        'mean_lifetime_sec': np.mean(lifetimes),
                        'median_lifetime_sec': np.median(lifetimes),
                        'short_lived_percent': sum(1 for t in lifetimes if t < 1.0) / len(lifetimes) * 100
                    }
            
            return {
                'allocation_count': len(allocs),
                'deallocation_count': len(deallocs),
                'size_statistics': size_stats,
                'temporal_statistics': temporal_stats,
                'lifecycle_statistics': lifecycle_stats,
                'current_metrics': self.metrics.__dict__
            }
    
    def update_rates(self) -> None:
        """Update allocation/deallocation rates"""
        current_time = time.time()
        time_delta = current_time - self._last_rate_calculation
        
        if time_delta >= 1.0:  # Update every second
            alloc_delta = self.metrics.total_allocations - self._last_allocations
            dealloc_delta = self.metrics.total_deallocations - self._last_deallocations
            
            self.metrics.allocation_rate_per_sec = alloc_delta / time_delta
            self.metrics.deallocation_rate_per_sec = dealloc_delta / time_delta
            
            self._last_rate_calculation = current_time
            self._last_allocations = self.metrics.total_allocations
            self._last_deallocations = self.metrics.total_deallocations


class MemoryMappedCache:
    """Memory-mapped file cache for large data structures"""
    
    def __init__(self, cache_dir: str = "./mmap_cache", max_size_gb: float = 5.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        # Cache metadata
        self._cache_info: Dict[str, Dict[str, Any]] = {}
        self._mapped_files: Dict[str, mmap.mmap] = {}
        self._lock = threading.RLock()
        
        # Load existing cache info
        self._load_cache_info()
    
    def put(self, key: str, data: Union[np.ndarray, torch.Tensor], compression: bool = True) -> bool:
        """Store data in memory-mapped file"""
        try:
            with self._lock:
                # Convert torch tensors to numpy
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                
                # Serialize data
                if compression:
                    import lz4.frame
                    serialized = lz4.frame.compress(pickle.dumps(data))
                else:
                    serialized = pickle.dumps(data)
                
                # Create memory-mapped file
                filename = f"{key}.mmap"
                filepath = self.cache_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(serialized)
                
                # Create memory mapping
                with open(filepath, 'r+b') as f:
                    mm = mmap.mmap(f.fileno(), 0)
                    self._mapped_files[key] = mm
                
                # Store metadata
                self._cache_info[key] = {
                    'filename': filename,
                    'size_bytes': len(serialized),
                    'compressed': compression,
                    'created_at': time.time(),
                    'shape': data.shape if hasattr(data, 'shape') else None,
                    'dtype': str(data.dtype) if hasattr(data, 'dtype') else None
                }
                
                self._save_cache_info()
                return True
                
        except Exception as e:
            logger.error(f"Failed to put {key} in memory-mapped cache: {e}")
            return False
    
    def get(self, key: str) -> Optional[Union[np.ndarray, Any]]:
        """Retrieve data from memory-mapped file"""
        try:
            with self._lock:
                if key not in self._cache_info:
                    return None
                
                # Get from existing mapping or create new one
                if key in self._mapped_files:
                    mm = self._mapped_files[key]
                    mm.seek(0)
                    data = mm.read()
                else:
                    filepath = self.cache_dir / self._cache_info[key]['filename']
                    if not filepath.exists():
                        del self._cache_info[key]
                        return None
                    
                    with open(filepath, 'r+b') as f:
                        mm = mmap.mmap(f.fileno(), 0)
                        self._mapped_files[key] = mm
                        data = mm.read()
                
                # Deserialize
                info = self._cache_info[key]
                if info['compressed']:
                    import lz4.frame
                    data = lz4.frame.decompress(data)
                
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Failed to get {key} from memory-mapped cache: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete data from cache"""
        try:
            with self._lock:
                if key not in self._cache_info:
                    return False
                
                # Close memory mapping
                if key in self._mapped_files:
                    self._mapped_files[key].close()
                    del self._mapped_files[key]
                
                # Remove file
                filepath = self.cache_dir / self._cache_info[key]['filename']
                if filepath.exists():
                    filepath.unlink()
                
                # Remove metadata
                del self._cache_info[key]
                self._save_cache_info()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete {key} from memory-mapped cache: {e}")
            return False
    
    def _load_cache_info(self) -> None:
        """Load cache metadata"""
        info_file = self.cache_dir / "cache_info.json"
        if info_file.exists():
            try:
                import json
                with open(info_file, 'r') as f:
                    self._cache_info = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache info: {e}")
                self._cache_info = {}
    
    def _save_cache_info(self) -> None:
        """Save cache metadata"""
        info_file = self.cache_dir / "cache_info.json"
        try:
            import json
            with open(info_file, 'w') as f:
                json.dump(self._cache_info, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache info: {e}")
    
    def cleanup(self) -> None:
        """Clean up memory mappings"""
        with self._lock:
            for mm in self._mapped_files.values():
                try:
                    mm.close()
                except Exception:
                    pass
            self._mapped_files.clear()


class MemoryOptimizer:
    """Central memory optimization coordinator"""
    
    def __init__(self,
                 enable_gc: bool = True,
                 enable_pooling: bool = True,
                 enable_tracking: bool = True,
                 gc_threshold: float = 0.8):
        
        # Components
        self.gc_manager = SmartGarbageCollector(
            memory_threshold=gc_threshold,
            adaptive_tuning=True
        ) if enable_gc else None
        
        self.memory_tracker = MemoryTracker(
            enable_tracemalloc=enable_tracking,
            track_allocations=enable_tracking
        ) if enable_tracking else None
        
        # Memory pools
        self.pools: Dict[str, MemoryPool] = {}
        self.tensor_pools: Dict[str, TensorMemoryPool] = {}
        
        # Memory-mapped cache
        self.mmap_cache = MemoryMappedCache()
        
        # Global settings
        self.enable_pooling = enable_pooling
        self.optimization_active = False
        
        # Performance metrics
        self.optimization_history = deque(maxlen=1000)
        
    def start_optimization(self) -> None:
        """Start memory optimization systems"""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        
        if self.gc_manager:
            self.gc_manager.start_background_gc()
        
        logger.info("Memory optimization started")
    
    def stop_optimization(self) -> None:
        """Stop memory optimization systems"""
        if not self.optimization_active:
            return
        
        self.optimization_active = False
        
        if self.gc_manager:
            self.gc_manager.stop_background_gc()
        
        # Clean up pools
        for pool in self.pools.values():
            pool.clear()
        
        for pool in self.tensor_pools.values():
            pool.clear_device_cache()
        
        self.mmap_cache.cleanup()
        
        logger.info("Memory optimization stopped")
    
    def create_object_pool(self, name: str, factory: Callable[[], T], **kwargs) -> MemoryPool[T]:
        """Create named object pool"""
        if not self.enable_pooling:
            raise RuntimeError("Pooling is disabled")
        
        pool = MemoryPool(factory, **kwargs)
        self.pools[name] = pool
        return pool
    
    def create_tensor_pool(self, name: str, device: str = 'cpu', **kwargs) -> TensorMemoryPool:
        """Create named tensor pool"""
        if not self.enable_pooling:
            raise RuntimeError("Pooling is disabled")
        
        pool = TensorMemoryPool(device=device, **kwargs)
        self.tensor_pools[name] = pool
        return pool
    
    def get_pool(self, name: str) -> Optional[MemoryPool]:
        """Get object pool by name"""
        return self.pools.get(name)
    
    def get_tensor_pool(self, name: str) -> Optional[TensorMemoryPool]:
        """Get tensor pool by name"""
        return self.tensor_pools.get(name)
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        optimization_results = {
            'gc_collections': 0,
            'cache_clears': 0,
            'pool_optimizations': 0,
            'memory_freed_mb': 0.0,
            'optimization_time_sec': 0.0
        }
        
        try:
            # Force garbage collection
            if self.gc_manager:
                collected = self.gc_manager.force_collection()
                optimization_results['gc_collections'] = collected
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                optimization_results['cache_clears'] += 1
            
            # Optimize tensor pools
            for pool in self.tensor_pools.values():
                pool.clear_device_cache()
                optimization_results['pool_optimizations'] += 1
            
            # Calculate memory freed
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_freed = max(0, start_memory - end_memory)
            optimization_results['memory_freed_mb'] = memory_freed
            
            # Calculate optimization time
            optimization_time = time.perf_counter() - start_time
            optimization_results['optimization_time_sec'] = optimization_time
            
            # Record in history
            self.optimization_history.append({
                'timestamp': time.time(),
                'results': optimization_results
            })
            
            logger.info(f"Memory optimization completed: freed {memory_freed:.1f}MB in {optimization_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {
            'system_memory': self._get_system_memory_stats(),
            'optimization_active': self.optimization_active
        }
        
        if self.gc_manager:
            stats['garbage_collection'] = self.gc_manager.get_gc_stats()
        
        if self.memory_tracker:
            stats['allocation_tracking'] = self.memory_tracker.analyze_allocation_patterns()
        
        if self.pools:
            stats['object_pools'] = {name: pool.get_stats() for name, pool in self.pools.items()}
        
        if self.tensor_pools:
            stats['tensor_pools'] = {name: pool.get_stats() for name, pool in self.tensor_pools.items()}
        
        # Recent optimization history
        if self.optimization_history:
            recent_optimizations = list(self.optimization_history)[-10:]
            stats['recent_optimizations'] = recent_optimizations
        
        return stats
    
    def _get_system_memory_stats(self) -> Dict[str, Any]:
        """Get system memory statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_memory_mb': memory_info.rss / (1024 * 1024),
            'process_virtual_memory_mb': memory_info.vms / (1024 * 1024),
            'system_total_mb': system_memory.total / (1024 * 1024),
            'system_available_mb': system_memory.available / (1024 * 1024),
            'system_used_percent': system_memory.percent,
            'swap_used_mb': psutil.swap_memory().used / (1024 * 1024)
        }


# Global memory optimizer instance
_global_memory_optimizer = None

def get_memory_optimizer(**kwargs) -> MemoryOptimizer:
    """Get global memory optimizer instance"""
    global _global_memory_optimizer
    
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizer(**kwargs)
    
    return _global_memory_optimizer

def start_memory_optimization(**kwargs) -> MemoryOptimizer:
    """Start global memory optimization"""
    optimizer = get_memory_optimizer(**kwargs)
    optimizer.start_optimization()
    return optimizer

@contextmanager
def memory_optimized_context(**kwargs):
    """Context manager for memory-optimized execution"""
    optimizer = get_memory_optimizer(**kwargs)
    optimizer.start_optimization()
    
    try:
        yield optimizer
    finally:
        # Perform final optimization
        optimizer.optimize_memory()

def memory_profile(func: Callable) -> Callable:
    """Decorator for memory profiling functions"""
    def wrapper(*args, **kwargs):
        optimizer = get_memory_optimizer()
        
        # Track allocation
        if optimizer.memory_tracker:
            start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            
            # Track memory usage
            if optimizer.memory_tracker:
                end_memory = psutil.Process().memory_info().rss
                memory_delta = end_memory - start_memory
                optimizer.memory_tracker.track_allocation(
                    result, memory_delta, f"Function: {func.__name__}"
                )
            
            return result
            
        except Exception as e:
            # Cleanup on exception
            if optimizer.optimization_active:
                optimizer.optimize_memory()
            raise
    
    return wrapper
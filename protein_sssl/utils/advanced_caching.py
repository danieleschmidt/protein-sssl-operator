"""
Advanced Multi-Tier Caching System for Protein-SSL Operator
Implements sophisticated caching strategies for optimal performance at scale
"""

import time
import threading
import pickle
import json
import hashlib
import sqlite3
import redis
import memcached
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
import numpy as np
import torch
import psutil
import gc
from pathlib import Path
import asyncio
import aioredis
import aiomcache
from concurrent.futures import ThreadPoolExecutor
import compression_utils  # Custom compression utilities
from enum import Enum

from .logging_config import setup_logging
from .monitoring import MetricsCollector

logger = setup_logging(__name__)


class CacheLevel(Enum):
    """Cache hierarchy levels"""
    MEMORY = "memory"          # In-memory cache (fastest)
    DISK = "disk"             # Local disk cache
    REDIS = "redis"           # Distributed Redis cache
    MEMCACHED = "memcached"   # Distributed Memcached
    DATABASE = "database"      # Persistent database cache


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    avg_access_time_ms: float = 0.0
    compression_ratio: float = 1.0
    memory_pressure: float = 0.0


@dataclass
class CacheItem:
    """Cache item with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float]
    size_bytes: int
    compression_type: Optional[str] = None
    tags: List[str] = None
    priority: int = 1  # 1=low, 5=high


class CacheStrategy(ABC):
    """Abstract base class for cache eviction strategies"""
    
    @abstractmethod
    def should_evict(self, item: CacheItem, cache_size: int, max_size: int) -> bool:
        """Determine if an item should be evicted"""
        pass
    
    @abstractmethod
    def get_eviction_score(self, item: CacheItem) -> float:
        """Get eviction score (higher = more likely to evict)"""
        pass


class LRUStrategy(CacheStrategy):
    """Least Recently Used eviction strategy"""
    
    def should_evict(self, item: CacheItem, cache_size: int, max_size: int) -> bool:
        return cache_size > max_size
    
    def get_eviction_score(self, item: CacheItem) -> float:
        # Older items get higher scores
        return time.time() - item.last_accessed


class LFUStrategy(CacheStrategy):
    """Least Frequently Used eviction strategy"""
    
    def should_evict(self, item: CacheItem, cache_size: int, max_size: int) -> bool:
        return cache_size > max_size
    
    def get_eviction_score(self, item: CacheItem) -> float:
        # Less frequently used items get higher scores
        return 1.0 / max(item.access_count, 1)


class AdaptiveStrategy(CacheStrategy):
    """Adaptive eviction strategy based on access patterns"""
    
    def __init__(self, lru_weight: float = 0.5, lfu_weight: float = 0.3, size_weight: float = 0.2):
        self.lru_weight = lru_weight
        self.lfu_weight = lfu_weight
        self.size_weight = size_weight
    
    def should_evict(self, item: CacheItem, cache_size: int, max_size: int) -> bool:
        return cache_size > max_size
    
    def get_eviction_score(self, item: CacheItem) -> float:
        current_time = time.time()
        
        # LRU component (0-1, higher = older)
        lru_score = (current_time - item.last_accessed) / max(current_time - item.created_at, 1)
        
        # LFU component (0-1, higher = less frequent)
        lfu_score = 1.0 / max(item.access_count, 1)
        
        # Size component (0-1, higher = larger)
        size_score = item.size_bytes / (1024 * 1024)  # Normalize to MB
        
        # Priority component (inverse)
        priority_score = 1.0 / max(item.priority, 1)
        
        return (self.lru_weight * lru_score + 
                self.lfu_weight * lfu_score + 
                self.size_weight * size_score) * priority_score


class CompressionManager:
    """Manages compression for cache items"""
    
    def __init__(self):
        self.compressors = {
            'gzip': self._gzip_compress,
            'lz4': self._lz4_compress,
            'zstd': self._zstd_compress,
            'pickle': self._pickle_compress
        }
        
        self.decompressors = {
            'gzip': self._gzip_decompress,
            'lz4': self._lz4_decompress,
            'zstd': self._zstd_decompress,
            'pickle': self._pickle_decompress
        }
    
    def compress(self, data: Any, method: str = 'auto') -> Tuple[bytes, str]:
        """Compress data using optimal method"""
        if method == 'auto':
            method = self._select_optimal_compression(data)
        
        compressor = self.compressors.get(method, self._pickle_compress)
        compressed_data = compressor(data)
        return compressed_data, method
    
    def decompress(self, data: bytes, method: str) -> Any:
        """Decompress data"""
        decompressor = self.decompressors.get(method, self._pickle_decompress)
        return decompressor(data)
    
    def _select_optimal_compression(self, data: Any) -> str:
        """Select optimal compression method based on data type and size"""
        if isinstance(data, (torch.Tensor, np.ndarray)):
            return 'lz4'  # Fast compression for numerical data
        elif isinstance(data, str) and len(data) > 1024:
            return 'zstd'  # Good compression for text
        elif isinstance(data, dict) and len(str(data)) > 1024:
            return 'gzip'  # Good for structured data
        else:
            return 'pickle'  # Default for small objects
    
    def _gzip_compress(self, data: Any) -> bytes:
        import gzip
        return gzip.compress(pickle.dumps(data))
    
    def _gzip_decompress(self, data: bytes) -> Any:
        import gzip
        return pickle.loads(gzip.decompress(data))
    
    def _lz4_compress(self, data: Any) -> bytes:
        try:
            import lz4.frame
            return lz4.frame.compress(pickle.dumps(data))
        except ImportError:
            return self._pickle_compress(data)
    
    def _lz4_decompress(self, data: bytes) -> Any:
        try:
            import lz4.frame
            return pickle.loads(lz4.frame.decompress(data))
        except ImportError:
            return self._pickle_decompress(data)
    
    def _zstd_compress(self, data: Any) -> bytes:
        try:
            import zstandard as zstd
            cctx = zstd.ZstdCompressor()
            return cctx.compress(pickle.dumps(data))
        except ImportError:
            return self._gzip_compress(data)
    
    def _zstd_decompress(self, data: bytes) -> Any:
        try:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            return pickle.loads(dctx.decompress(data))
        except ImportError:
            return self._gzip_decompress(data)
    
    def _pickle_compress(self, data: Any) -> bytes:
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _pickle_decompress(self, data: bytes) -> Any:
        return pickle.loads(data)


class MemoryCache:
    """High-performance in-memory cache with advanced features"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 1024,
                 ttl_seconds: Optional[float] = None,
                 strategy: CacheStrategy = None,
                 enable_compression: bool = True,
                 compression_threshold: int = 1024):
        
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = ttl_seconds
        self.strategy = strategy or AdaptiveStrategy()
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        # Core cache storage
        self._cache: Dict[str, CacheItem] = {}
        self._access_order = OrderedDict()  # For LRU tracking
        self._size_index = defaultdict(list)  # Size-based indexing
        self._tag_index = defaultdict(set)   # Tag-based indexing
        
        # Threading and performance
        self._lock = threading.RLock()
        self._metrics = CacheMetrics()
        self._compression_manager = CompressionManager()
        
        # Background maintenance
        self._maintenance_thread = None
        self._maintenance_active = False
        self._maintenance_interval = 60.0  # seconds
        
        # Performance optimization
        self._bloom_filter = BloomFilter(capacity=10000)  # For negative lookups
        self._size_tracker = 0
        
        self.start_maintenance()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with performance tracking"""
        start_time = time.perf_counter()
        
        with self._lock:
            # Quick bloom filter check for non-existent keys
            if not self._bloom_filter.might_contain(key):
                self._metrics.misses += 1
                return default
            
            if key not in self._cache:
                self._metrics.misses += 1
                return default
            
            item = self._cache[key]
            
            # Check TTL expiration
            if self._is_expired(item):
                self._remove_item(key)
                self._metrics.misses += 1
                return default
            
            # Update access information
            item.last_accessed = time.time()
            item.access_count += 1
            self._access_order.move_to_end(key)
            
            # Decompress if needed
            value = item.value
            if item.compression_type:
                try:
                    value = self._compression_manager.decompress(value, item.compression_type)
                except Exception as e:
                    logger.warning(f"Decompression failed for key {key}: {e}")
                    self._remove_item(key)
                    return default
            
            self._metrics.hits += 1
            self._update_access_time(time.perf_counter() - start_time)
            
            return value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, 
            tags: List[str] = None, priority: int = 1) -> bool:
        """Put item in cache with advanced options"""
        try:
            with self._lock:
                current_time = time.time()
                
                # Prepare cache item
                original_value = value
                compression_type = None
                
                # Compression handling
                if self.enable_compression:
                    value_size = self._estimate_size(value)
                    if value_size > self.compression_threshold:
                        try:
                            value, compression_type = self._compression_manager.compress(value)
                        except Exception as e:
                            logger.warning(f"Compression failed for key {key}: {e}")
                            value = original_value
                
                # Calculate actual size
                size_bytes = self._estimate_size(value)
                
                # Create cache item
                item = CacheItem(
                    key=key,
                    value=value,
                    created_at=current_time,
                    last_accessed=current_time,
                    access_count=1,
                    ttl=ttl or self.default_ttl,
                    size_bytes=size_bytes,
                    compression_type=compression_type,
                    tags=tags or [],
                    priority=priority
                )
                
                # Check if we need to evict items
                if not self._has_space_for_item(item):
                    self._evict_items(item.size_bytes)
                
                # Remove existing item if present
                if key in self._cache:
                    self._remove_item(key)
                
                # Add new item
                self._cache[key] = item
                self._access_order[key] = current_time
                self._size_tracker += size_bytes
                
                # Update indexes
                self._update_indexes(item)
                self._bloom_filter.add(key)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to put item {key} in cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False
    
    def clear(self, tags: Optional[List[str]] = None) -> int:
        """Clear cache items, optionally by tags"""
        with self._lock:
            if tags is None:
                # Clear entire cache
                count = len(self._cache)
                self._cache.clear()
                self._access_order.clear()
                self._size_index.clear()
                self._tag_index.clear()
                self._size_tracker = 0
                self._bloom_filter = BloomFilter(capacity=10000)
                return count
            else:
                # Clear by tags
                keys_to_remove = set()
                for tag in tags:
                    keys_to_remove.update(self._tag_index.get(tag, set()))
                
                for key in keys_to_remove:
                    self._remove_item(key)
                
                return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            total_requests = self._metrics.hits + self._metrics.misses
            hit_rate = self._metrics.hits / max(total_requests, 1)
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_bytes': self._size_tracker,
                'memory_usage_mb': self._size_tracker / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'hits': self._metrics.hits,
                'misses': self._metrics.misses,
                'evictions': self._metrics.evictions,
                'avg_access_time_ms': self._metrics.avg_access_time_ms,
                'compression_enabled': self.enable_compression,
                'compression_ratio': self._calculate_compression_ratio(),
                'memory_pressure': self._calculate_memory_pressure()
            }
    
    def _has_space_for_item(self, item: CacheItem) -> bool:
        """Check if cache has space for new item"""
        return (len(self._cache) < self.max_size and 
                self._size_tracker + item.size_bytes <= self.max_memory_bytes)
    
    def _evict_items(self, needed_space: int) -> None:
        """Evict items to make space"""
        if not self._cache:
            return
        
        # Calculate how much space we need
        target_size = self._size_tracker + needed_space
        
        # Sort items by eviction score
        items_to_evict = []
        for key, item in self._cache.items():
            if not self._is_expired(item):  # Don't score expired items
                score = self.strategy.get_eviction_score(item)
                items_to_evict.append((score, key, item))
        
        # Sort by score (highest first)
        items_to_evict.sort(reverse=True)
        
        # Evict items until we have enough space
        evicted_count = 0
        for score, key, item in items_to_evict:
            if (len(self._cache) - evicted_count <= self.max_size // 2 or
                self._size_tracker <= target_size):
                break
            
            self._remove_item(key)
            evicted_count += 1
            self._metrics.evictions += 1
        
        logger.debug(f"Evicted {evicted_count} items to make space")
    
    def _remove_item(self, key: str) -> None:
        """Remove item from all cache structures"""
        if key not in self._cache:
            return
        
        item = self._cache[key]
        
        # Remove from main cache
        del self._cache[key]
        
        # Remove from access order
        if key in self._access_order:
            del self._access_order[key]
        
        # Update size tracker
        self._size_tracker -= item.size_bytes
        
        # Remove from tag index
        for tag in item.tags or []:
            if tag in self._tag_index:
                self._tag_index[tag].discard(key)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]
    
    def _update_indexes(self, item: CacheItem) -> None:
        """Update secondary indexes"""
        # Tag index
        for tag in item.tags or []:
            self._tag_index[tag].add(item.key)
        
        # Size index (for size-based eviction)
        size_bucket = item.size_bytes // (1024 * 1024)  # MB buckets
        self._size_index[size_bucket].append(item.key)
    
    def _is_expired(self, item: CacheItem) -> bool:
        """Check if item has expired"""
        if item.ttl is None:
            return False
        return time.time() - item.created_at > item.ttl
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            if isinstance(obj, bytes):
                return len(obj)
            elif isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, torch.Tensor):
                return obj.element_size() * obj.nelement()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                # Fallback to pickle size estimation
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024  # Default estimate
    
    def _update_access_time(self, access_time: float) -> None:
        """Update average access time metric"""
        current_avg = self._metrics.avg_access_time_ms
        access_time_ms = access_time * 1000
        
        # Exponential moving average
        alpha = 0.1
        self._metrics.avg_access_time_ms = (1 - alpha) * current_avg + alpha * access_time_ms
    
    def _calculate_compression_ratio(self) -> float:
        """Calculate overall compression ratio"""
        if not self.enable_compression:
            return 1.0
        
        compressed_items = [item for item in self._cache.values() 
                          if item.compression_type is not None]
        
        if not compressed_items:
            return 1.0
        
        # This is a simplified calculation
        # In practice, we'd track original vs compressed sizes
        return 0.7  # Assume 30% compression on average
    
    def _calculate_memory_pressure(self) -> float:
        """Calculate current memory pressure (0-1)"""
        return min(1.0, self._size_tracker / self.max_memory_bytes)
    
    def start_maintenance(self) -> None:
        """Start background maintenance thread"""
        if self._maintenance_active:
            return
        
        self._maintenance_active = True
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True
        )
        self._maintenance_thread.start()
    
    def stop_maintenance(self) -> None:
        """Stop background maintenance"""
        self._maintenance_active = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5.0)
    
    def _maintenance_loop(self) -> None:
        """Background maintenance loop"""
        while self._maintenance_active:
            try:
                self._perform_maintenance()
                time.sleep(self._maintenance_interval)
            except Exception as e:
                logger.warning(f"Cache maintenance error: {e}")
                time.sleep(self._maintenance_interval)
    
    def _perform_maintenance(self) -> None:
        """Perform cache maintenance tasks"""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            # Find expired items
            for key, item in self._cache.items():
                if self._is_expired(item):
                    expired_keys.append(key)
            
            # Remove expired items
            for key in expired_keys:
                self._remove_item(key)
            
            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired items")
            
            # Adaptive optimization
            self._optimize_cache_parameters()
    
    def _optimize_cache_parameters(self) -> None:
        """Optimize cache parameters based on usage patterns"""
        stats = self.get_stats()
        
        # Adjust compression threshold based on hit rate and memory pressure
        if stats['hit_rate'] > 0.8 and stats['memory_pressure'] > 0.8:
            # High hit rate but memory pressure - be more aggressive with compression
            self.compression_threshold = max(512, self.compression_threshold * 0.9)
        elif stats['hit_rate'] < 0.5:
            # Low hit rate - maybe compression overhead is too high
            self.compression_threshold = min(4096, self.compression_threshold * 1.1)


class BloomFilter:
    """Simple Bloom filter for cache negative lookups"""
    
    def __init__(self, capacity: int, error_rate: float = 0.1):
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Calculate optimal parameters
        self.bit_array_size = int(-capacity * np.log(error_rate) / (np.log(2) ** 2))
        self.hash_functions = int(self.bit_array_size / capacity * np.log(2))
        
        self.bit_array = [False] * self.bit_array_size
        self.items_count = 0
    
    def add(self, item: str) -> None:
        """Add item to bloom filter"""
        for i in range(self.hash_functions):
            index = hash(item + str(i)) % self.bit_array_size
            self.bit_array[index] = True
        self.items_count += 1
    
    def might_contain(self, item: str) -> bool:
        """Check if item might be in the set"""
        for i in range(self.hash_functions):
            index = hash(item + str(i)) % self.bit_array_size
            if not self.bit_array[index]:
                return False
        return True


class MultiTierCache:
    """Multi-tier cache system with memory, disk, and distributed caching"""
    
    def __init__(self,
                 memory_cache_size: int = 1000,
                 memory_cache_mb: int = 512,
                 disk_cache_path: Optional[str] = None,
                 redis_url: Optional[str] = None,
                 memcached_servers: Optional[List[str]] = None,
                 enable_async: bool = True):
        
        # Cache tiers
        self.memory_cache = MemoryCache(
            max_size=memory_cache_size,
            max_memory_mb=memory_cache_mb
        )
        
        self.disk_cache = DiskCache(disk_cache_path) if disk_cache_path else None
        self.redis_cache = RedisCache(redis_url) if redis_url else None
        self.memcached_cache = MemcachedCache(memcached_servers) if memcached_servers else None
        
        self.enable_async = enable_async
        
        # Cache hierarchy (order matters)
        self.cache_tiers = [
            (CacheLevel.MEMORY, self.memory_cache),
            (CacheLevel.DISK, self.disk_cache),
            (CacheLevel.REDIS, self.redis_cache),
            (CacheLevel.MEMCACHED, self.memcached_cache)
        ]
        
        # Filter out None caches
        self.cache_tiers = [(level, cache) for level, cache in self.cache_tiers if cache is not None]
        
        # Performance metrics
        self.metrics = defaultdict(lambda: CacheMetrics())
        self._lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from multi-tier cache"""
        start_time = time.perf_counter()
        
        # Try each tier in order
        for level, cache in self.cache_tiers:
            try:
                value = cache.get(key)
                if value is not None:
                    # Promote to higher tiers
                    self._promote_to_higher_tiers(key, value, level)
                    
                    # Update metrics
                    self.metrics[level].hits += 1
                    access_time = (time.perf_counter() - start_time) * 1000
                    self._update_tier_metrics(level, access_time)
                    
                    return value
                else:
                    self.metrics[level].misses += 1
                    
            except Exception as e:
                logger.warning(f"Cache tier {level} error: {e}")
                self.metrics[level].misses += 1
                continue
        
        return default
    
    def put(self, key: str, value: Any, **kwargs) -> bool:
        """Put item in all appropriate cache tiers"""
        success = False
        
        # Store in all tiers (or until one fails)
        for level, cache in self.cache_tiers:
            try:
                if cache.put(key, value, **kwargs):
                    success = True
                    
                    # Update size metrics
                    size = self._estimate_size(value)
                    self.metrics[level].total_size_bytes += size
                    
            except Exception as e:
                logger.warning(f"Failed to put {key} in cache tier {level}: {e}")
                continue
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete item from all cache tiers"""
        any_success = False
        
        for level, cache in self.cache_tiers:
            try:
                if cache.delete(key):
                    any_success = True
            except Exception as e:
                logger.warning(f"Failed to delete {key} from cache tier {level}: {e}")
                continue
        
        return any_success
    
    def clear(self, tags: Optional[List[str]] = None) -> Dict[str, int]:
        """Clear all cache tiers"""
        results = {}
        
        for level, cache in self.cache_tiers:
            try:
                count = cache.clear(tags=tags)
                results[level.value] = count
            except Exception as e:
                logger.warning(f"Failed to clear cache tier {level}: {e}")
                results[level.value] = 0
        
        return results
    
    def get_multi_tier_stats(self) -> Dict[str, Any]:
        """Get comprehensive multi-tier cache statistics"""
        tier_stats = {}
        
        for level, cache in self.cache_tiers:
            try:
                tier_stats[level.value] = cache.get_stats()
                tier_stats[level.value].update(asdict(self.metrics[level]))
            except Exception as e:
                logger.warning(f"Failed to get stats for cache tier {level}: {e}")
                tier_stats[level.value] = {"error": str(e)}
        
        # Overall statistics
        total_hits = sum(self.metrics[level].hits for level, _ in self.cache_tiers)
        total_misses = sum(self.metrics[level].misses for level, _ in self.cache_tiers)
        total_requests = total_hits + total_misses
        
        overall_stats = {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate': total_hits / max(total_requests, 1),
            'active_tiers': [level.value for level, _ in self.cache_tiers],
            'tier_count': len(self.cache_tiers)
        }
        
        return {
            'tiers': tier_stats,
            'overall': overall_stats
        }
    
    def _promote_to_higher_tiers(self, key: str, value: Any, found_level: CacheLevel) -> None:
        """Promote cache item to higher (faster) tiers"""
        found_index = next(i for i, (level, _) in enumerate(self.cache_tiers) if level == found_level)
        
        # Promote to all higher tiers
        for i in range(found_index):
            level, cache = self.cache_tiers[i]
            try:
                cache.put(key, value)
                logger.debug(f"Promoted {key} to cache tier {level}")
            except Exception as e:
                logger.warning(f"Failed to promote {key} to cache tier {level}: {e}")
    
    def _update_tier_metrics(self, level: CacheLevel, access_time_ms: float) -> None:
        """Update tier-specific metrics"""
        metrics = self.metrics[level]
        
        # Update average access time with exponential moving average
        alpha = 0.1
        metrics.avg_access_time_ms = (1 - alpha) * metrics.avg_access_time_ms + alpha * access_time_ms
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size"""
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024  # Default estimate


class DiskCache:
    """Disk-based cache implementation"""
    
    def __init__(self, cache_dir: str = "./cache", max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        # SQLite database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        self._lock = threading.RLock()
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_items (
                    key TEXT PRIMARY KEY,
                    filename TEXT,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    size_bytes INTEGER,
                    ttl REAL,
                    tags TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_items(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_size ON cache_items(size_bytes)")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from disk cache"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT filename, ttl, created_at FROM cache_items WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if not row:
                        return default
                    
                    filename, ttl, created_at = row
                    
                    # Check TTL
                    if ttl and time.time() - created_at > ttl:
                        self.delete(key)
                        return default
                    
                    # Load from disk
                    file_path = self.cache_dir / filename
                    if not file_path.exists():
                        self.delete(key)
                        return default
                    
                    with open(file_path, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Update access information
                    conn.execute(
                        "UPDATE cache_items SET last_accessed = ?, access_count = access_count + 1 WHERE key = ?",
                        (time.time(), key)
                    )
                    
                    return value
                    
            except Exception as e:
                logger.warning(f"Disk cache get error for key {key}: {e}")
                return default
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> bool:
        """Put item in disk cache"""
        with self._lock:
            try:
                # Generate filename
                filename = f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
                file_path = self.cache_dir / filename
                
                # Serialize and save to disk
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Get file size
                size_bytes = file_path.stat().st_size
                current_time = time.time()
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_items 
                        (key, filename, created_at, last_accessed, access_count, size_bytes, ttl, tags)
                        VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    """, (key, filename, current_time, current_time, size_bytes, ttl, 
                          json.dumps(tags) if tags else None))
                
                # Check if we need to evict old items
                self._evict_if_needed()
                
                return True
                
            except Exception as e:
                logger.warning(f"Disk cache put error for key {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete item from disk cache"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT filename FROM cache_items WHERE key = ?", (key,))
                    row = cursor.fetchone()
                    
                    if row:
                        filename = row[0]
                        file_path = self.cache_dir / filename
                        
                        # Remove file
                        if file_path.exists():
                            file_path.unlink()
                        
                        # Remove from database
                        conn.execute("DELETE FROM cache_items WHERE key = ?", (key,))
                        return True
                        
            except Exception as e:
                logger.warning(f"Disk cache delete error for key {key}: {e}")
            
            return False
    
    def clear(self, tags: Optional[List[str]] = None) -> int:
        """Clear disk cache"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    if tags is None:
                        # Clear all
                        cursor = conn.execute("SELECT filename FROM cache_items")
                        filenames = [row[0] for row in cursor.fetchall()]
                        
                        for filename in filenames:
                            file_path = self.cache_dir / filename
                            if file_path.exists():
                                file_path.unlink()
                        
                        conn.execute("DELETE FROM cache_items")
                        return len(filenames)
                    else:
                        # Clear by tags
                        count = 0
                        cursor = conn.execute("SELECT key, filename, tags FROM cache_items")
                        
                        for key, filename, tags_json in cursor.fetchall():
                            if tags_json:
                                item_tags = json.loads(tags_json)
                                if any(tag in item_tags for tag in tags):
                                    file_path = self.cache_dir / filename
                                    if file_path.exists():
                                        file_path.unlink()
                                    conn.execute("DELETE FROM cache_items WHERE key = ?", (key,))
                                    count += 1
                        
                        return count
                        
            except Exception as e:
                logger.warning(f"Disk cache clear error: {e}")
                return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache_items")
                count, total_size = cursor.fetchone()
                
                # Get average access time (simplified)
                cursor = conn.execute("SELECT AVG(access_count) FROM cache_items")
                avg_access_count = cursor.fetchone()[0] or 0
                
                return {
                    'size': count or 0,
                    'total_size_bytes': total_size or 0,
                    'total_size_mb': (total_size or 0) / (1024 * 1024),
                    'max_size_gb': self.max_size_bytes / (1024 * 1024 * 1024),
                    'avg_access_count': avg_access_count,
                    'cache_dir': str(self.cache_dir)
                }
        except Exception as e:
            logger.warning(f"Disk cache stats error: {e}")
            return {'error': str(e)}
    
    def _evict_if_needed(self):
        """Evict old items if cache is too large"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_items")
                total_size = cursor.fetchone()[0] or 0
                
                if total_size > self.max_size_bytes:
                    # Remove oldest items until under limit
                    cursor = conn.execute("""
                        SELECT key, filename, size_bytes 
                        FROM cache_items 
                        ORDER BY last_accessed ASC
                    """)
                    
                    for key, filename, size_bytes in cursor.fetchall():
                        if total_size <= self.max_size_bytes * 0.8:  # Leave some headroom
                            break
                        
                        # Delete file and database entry
                        file_path = self.cache_dir / filename
                        if file_path.exists():
                            file_path.unlink()
                        
                        conn.execute("DELETE FROM cache_items WHERE key = ?", (key,))
                        total_size -= size_bytes
                        
        except Exception as e:
            logger.warning(f"Disk cache eviction error: {e}")


class RedisCache:
    """Redis-based distributed cache"""
    
    def __init__(self, redis_url: str, prefix: str = "protein_ssl_cache"):
        self.redis_url = redis_url
        self.prefix = prefix
        self.redis_client = redis.from_url(redis_url)
        self._compression_manager = CompressionManager()
        
        # Test connection
        try:
            self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from Redis cache"""
        try:
            prefixed_key = f"{self.prefix}:{key}"
            data = self.redis_client.get(prefixed_key)
            
            if data is None:
                return default
            
            # Deserialize
            cache_data = pickle.loads(data)
            value = cache_data['value']
            compression_type = cache_data.get('compression_type')
            
            # Decompress if needed
            if compression_type:
                value = self._compression_manager.decompress(value, compression_type)
            
            # Update access count
            self.redis_client.incr(f"{prefixed_key}:access_count")
            
            return value
            
        except Exception as e:
            logger.warning(f"Redis cache get error for key {key}: {e}")
            return default
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, **kwargs) -> bool:
        """Put item in Redis cache"""
        try:
            # Compress if beneficial
            original_value = value
            compression_type = None
            
            try:
                compressed_value, compression_type = self._compression_manager.compress(value)
                if len(pickle.dumps(compressed_value)) < len(pickle.dumps(original_value)) * 0.8:
                    value = compressed_value
                else:
                    compression_type = None
                    value = original_value
            except Exception:
                compression_type = None
                value = original_value
            
            # Prepare cache data
            cache_data = {
                'value': value,
                'compression_type': compression_type,
                'created_at': time.time()
            }
            
            serialized_data = pickle.dumps(cache_data)
            prefixed_key = f"{self.prefix}:{key}"
            
            # Store in Redis
            if ttl:
                self.redis_client.setex(prefixed_key, int(ttl), serialized_data)
            else:
                self.redis_client.set(prefixed_key, serialized_data)
            
            # Initialize access count
            self.redis_client.set(f"{prefixed_key}:access_count", 0, ex=int(ttl) if ttl else None)
            
            return True
            
        except Exception as e:
            logger.warning(f"Redis cache put error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from Redis cache"""
        try:
            prefixed_key = f"{self.prefix}:{key}"
            result = self.redis_client.delete(prefixed_key, f"{prefixed_key}:access_count")
            return result > 0
        except Exception as e:
            logger.warning(f"Redis cache delete error for key {key}: {e}")
            return False
    
    def clear(self, tags: Optional[List[str]] = None) -> int:
        """Clear Redis cache"""
        try:
            if tags is None:
                # Clear all keys with our prefix
                pattern = f"{self.prefix}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            else:
                # Tag-based clearing not implemented for Redis
                # Would require additional tag storage
                logger.warning("Tag-based clearing not implemented for Redis cache")
                return 0
        except Exception as e:
            logger.warning(f"Redis cache clear error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        try:
            info = self.redis_client.info()
            pattern = f"{self.prefix}:*"
            keys = self.redis_client.keys(pattern)
            
            return {
                'size': len(keys),
                'redis_memory_used': info.get('used_memory', 0),
                'redis_memory_used_mb': info.get('used_memory', 0) / (1024 * 1024),
                'redis_connected_clients': info.get('connected_clients', 0),
                'redis_total_commands_processed': info.get('total_commands_processed', 0),
                'prefix': self.prefix
            }
        except Exception as e:
            logger.warning(f"Redis cache stats error: {e}")
            return {'error': str(e)}


class MemcachedCache:
    """Memcached-based distributed cache"""
    
    def __init__(self, servers: List[str], prefix: str = "protein_ssl_cache"):
        self.servers = servers
        self.prefix = prefix
        try:
            import pymemcache.client.base
            self.client = pymemcache.client.base.Client(
                servers[0] if servers else ('localhost', 11211),
                serializer=pickle.PickleSerializer(),
                deserializer=pickle.PickleDeserializer()
            )
        except ImportError:
            logger.error("pymemcache not installed, Memcached cache disabled")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from Memcached"""
        try:
            prefixed_key = f"{self.prefix}:{key}"
            value = self.client.get(prefixed_key)
            return value if value is not None else default
        except Exception as e:
            logger.warning(f"Memcached cache get error for key {key}: {e}")
            return default
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, **kwargs) -> bool:
        """Put item in Memcached"""
        try:
            prefixed_key = f"{self.prefix}:{key}"
            expire = int(ttl) if ttl else 0
            return self.client.set(prefixed_key, value, expire=expire)
        except Exception as e:
            logger.warning(f"Memcached cache put error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from Memcached"""
        try:
            prefixed_key = f"{self.prefix}:{key}"
            return self.client.delete(prefixed_key)
        except Exception as e:
            logger.warning(f"Memcached cache delete error for key {key}: {e}")
            return False
    
    def clear(self, tags: Optional[List[str]] = None) -> int:
        """Clear Memcached cache"""
        try:
            if tags is None:
                self.client.flush_all()
                return 1  # Can't return exact count
            else:
                logger.warning("Tag-based clearing not supported for Memcached")
                return 0
        except Exception as e:
            logger.warning(f"Memcached cache clear error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Memcached statistics"""
        try:
            stats = self.client.stats()
            return {
                'memcached_stats': stats,
                'servers': self.servers,
                'prefix': self.prefix
            }
        except Exception as e:
            logger.warning(f"Memcached cache stats error: {e}")
            return {'error': str(e)}


# Global cache instance
_global_cache = None

def get_cache(**kwargs) -> MultiTierCache:
    """Get global multi-tier cache instance"""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = MultiTierCache(**kwargs)
    
    return _global_cache

def cache_decorator(ttl: Optional[float] = None, 
                   tags: Optional[List[str]] = None,
                   key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        cache = get_cache()
        
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl=ttl, tags=tags)
            
            return result
        
        return wrapper
    return decorator
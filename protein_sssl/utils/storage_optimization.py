"""
Advanced Storage Optimization System for Protein-SSL Operator
Implements database optimization, indexing strategies, compression, and data lifecycle management
"""

import time
import threading
import hashlib
import json
import pickle
import lz4
import gzip
import zstandard as zstd
import sqlite3
import asyncio
import aiofiles
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import psutil
import shutil
import os
import tempfile
from contextlib import contextmanager
import mmap
import struct

from .logging_config import setup_logging
from .monitoring import MetricsCollector

logger = setup_logging(__name__)


class CompressionType(Enum):
    """Compression algorithm types"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"
    SNAPPY = "snappy"


class IndexType(Enum):
    """Database index types"""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    BLOOM = "bloom"
    PARTIAL = "partial"


class StorageTier(Enum):
    """Storage tier levels"""
    HOT = "hot"          # Frequently accessed, SSD
    WARM = "warm"        # Occasionally accessed, HDD
    COLD = "cold"        # Rarely accessed, archival
    FROZEN = "frozen"    # Archive only, tape/cloud


@dataclass
class CompressionMetrics:
    """Compression performance metrics"""
    algorithm: CompressionType
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    cpu_usage: float


@dataclass
class IndexMetrics:
    """Index performance metrics"""
    index_name: str
    index_type: IndexType
    table_name: str
    size_mb: float
    selectivity: float
    usage_count: int
    avg_query_time: float
    creation_time: float


@dataclass
class StorageStats:
    """Storage system statistics"""
    total_size_gb: float
    used_size_gb: float
    free_size_gb: float
    compression_ratio: float
    index_hit_ratio: float
    cache_hit_ratio: float
    io_operations_per_sec: float
    throughput_mb_per_sec: float


class CompressionManager:
    """Advanced compression management system"""
    
    def __init__(self):
        self.compressors = {
            CompressionType.GZIP: self._gzip_compress,
            CompressionType.LZ4: self._lz4_compress,
            CompressionType.ZSTD: self._zstd_compress,
            CompressionType.BROTLI: self._brotli_compress,
            CompressionType.SNAPPY: self._snappy_compress
        }
        
        self.decompressors = {
            CompressionType.GZIP: self._gzip_decompress,
            CompressionType.LZ4: self._lz4_decompress,
            CompressionType.ZSTD: self._zstd_decompress,
            CompressionType.BROTLI: self._brotli_decompress,
            CompressionType.SNAPPY: self._snappy_decompress
        }
        
        # Performance tracking
        self.compression_metrics = []
        self.algorithm_performance = defaultdict(list)
        
    def compress(self, data: bytes, algorithm: CompressionType = None) -> Tuple[bytes, CompressionType, CompressionMetrics]:
        """Compress data using optimal algorithm"""
        if algorithm is None:
            algorithm = self._select_optimal_algorithm(data)
        
        if algorithm == CompressionType.NONE:
            metrics = CompressionMetrics(
                algorithm=algorithm,
                original_size=len(data),
                compressed_size=len(data),
                compression_ratio=1.0,
                compression_time=0.0,
                decompression_time=0.0,
                cpu_usage=0.0
            )
            return data, algorithm, metrics
        
        start_time = time.perf_counter()
        start_cpu = psutil.cpu_percent()
        
        compressor = self.compressors[algorithm]
        compressed_data = compressor(data)
        
        end_time = time.perf_counter()
        end_cpu = psutil.cpu_percent()
        
        # Calculate metrics
        compression_time = end_time - start_time
        compression_ratio = len(data) / len(compressed_data) if compressed_data else 1.0
        cpu_usage = (end_cpu - start_cpu) / 100.0
        
        # Test decompression time
        decomp_start = time.perf_counter()
        self.decompress(compressed_data, algorithm)
        decomp_time = time.perf_counter() - decomp_start
        
        metrics = CompressionMetrics(
            algorithm=algorithm,
            original_size=len(data),
            compressed_size=len(compressed_data),
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            decompression_time=decomp_time,
            cpu_usage=cpu_usage
        )
        
        # Track performance
        self.compression_metrics.append(metrics)
        self.algorithm_performance[algorithm].append(metrics)
        
        # Keep only recent metrics
        if len(self.compression_metrics) > 1000:
            self.compression_metrics = self.compression_metrics[-1000:]
        
        return compressed_data, algorithm, metrics
    
    def decompress(self, data: bytes, algorithm: CompressionType) -> bytes:
        """Decompress data"""
        if algorithm == CompressionType.NONE:
            return data
        
        decompressor = self.decompressors[algorithm]
        return decompressor(data)
    
    def _select_optimal_algorithm(self, data: bytes) -> CompressionType:
        """Select optimal compression algorithm based on data characteristics"""
        data_size = len(data)
        
        # Small data - use fast compression
        if data_size < 1024:  # 1KB
            return CompressionType.LZ4
        
        # Analyze data characteristics
        sample_size = min(1024, data_size)
        sample = data[:sample_size]
        
        # Calculate entropy (simplified)
        entropy = self._calculate_entropy(sample)
        
        # Check for patterns
        repetition_ratio = self._calculate_repetition_ratio(sample)
        
        # Selection logic
        if entropy < 3.0:  # Low entropy, highly compressible
            if data_size > 1024 * 1024:  # >1MB
                return CompressionType.ZSTD
            else:
                return CompressionType.LZ4
        elif entropy > 7.0:  # High entropy, less compressible
            return CompressionType.LZ4  # Fast compression
        else:  # Medium entropy
            if repetition_ratio > 0.3:
                return CompressionType.ZSTD
            else:
                return CompressionType.LZ4
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data sample"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = defaultdict(int)
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_repetition_ratio(self, data: bytes) -> float:
        """Calculate repetition ratio in data sample"""
        if len(data) < 4:
            return 0.0
        
        patterns = defaultdict(int)
        pattern_length = 4
        
        for i in range(len(data) - pattern_length + 1):
            pattern = data[i:i + pattern_length]
            patterns[pattern] += 1
        
        # Calculate repetition ratio
        total_patterns = len(data) - pattern_length + 1
        repeated_patterns = sum(count - 1 for count in patterns.values() if count > 1)
        
        return repeated_patterns / total_patterns if total_patterns > 0 else 0.0
    
    def _gzip_compress(self, data: bytes) -> bytes:
        """GZIP compression"""
        return gzip.compress(data, compresslevel=6)
    
    def _gzip_decompress(self, data: bytes) -> bytes:
        """GZIP decompression"""
        return gzip.decompress(data)
    
    def _lz4_compress(self, data: bytes) -> bytes:
        """LZ4 compression"""
        try:
            import lz4.frame
            return lz4.frame.compress(data)
        except ImportError:
            logger.warning("LZ4 not available, falling back to gzip")
            return self._gzip_compress(data)
    
    def _lz4_decompress(self, data: bytes) -> bytes:
        """LZ4 decompression"""
        try:
            import lz4.frame
            return lz4.frame.decompress(data)
        except ImportError:
            return self._gzip_decompress(data)
    
    def _zstd_compress(self, data: bytes) -> bytes:
        """Zstandard compression"""
        try:
            cctx = zstd.ZstdCompressor(level=3)
            return cctx.compress(data)
        except Exception:
            logger.warning("Zstandard not available, falling back to gzip")
            return self._gzip_compress(data)
    
    def _zstd_decompress(self, data: bytes) -> bytes:
        """Zstandard decompression"""
        try:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        except Exception:
            return self._gzip_decompress(data)
    
    def _brotli_compress(self, data: bytes) -> bytes:
        """Brotli compression"""
        try:
            import brotli
            return brotli.compress(data, quality=4)
        except ImportError:
            logger.warning("Brotli not available, falling back to gzip")
            return self._gzip_compress(data)
    
    def _brotli_decompress(self, data: bytes) -> bytes:
        """Brotli decompression"""
        try:
            import brotli
            return brotli.decompress(data)
        except ImportError:
            return self._gzip_decompress(data)
    
    def _snappy_compress(self, data: bytes) -> bytes:
        """Snappy compression"""
        try:
            import snappy
            return snappy.compress(data)
        except ImportError:
            logger.warning("Snappy not available, falling back to lz4")
            return self._lz4_compress(data)
    
    def _snappy_decompress(self, data: bytes) -> bytes:
        """Snappy decompression"""
        try:
            import snappy
            return snappy.decompress(data)
        except ImportError:
            return self._lz4_decompress(data)
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics"""
        if not self.compression_metrics:
            return {'message': 'No compression metrics available'}
        
        recent_metrics = self.compression_metrics[-100:]  # Last 100 operations
        
        # Overall statistics
        total_original = sum(m.original_size for m in recent_metrics)
        total_compressed = sum(m.compressed_size for m in recent_metrics)
        avg_ratio = total_original / total_compressed if total_compressed > 0 else 1.0
        avg_compression_time = np.mean([m.compression_time for m in recent_metrics])
        avg_decompression_time = np.mean([m.decompression_time for m in recent_metrics])
        
        # Algorithm performance
        algorithm_stats = {}
        for algorithm, metrics_list in self.algorithm_performance.items():
            if metrics_list:
                recent_alg_metrics = metrics_list[-50:]  # Recent metrics for this algorithm
                algorithm_stats[algorithm.value] = {
                    'count': len(recent_alg_metrics),
                    'avg_compression_ratio': np.mean([m.compression_ratio for m in recent_alg_metrics]),
                    'avg_compression_time': np.mean([m.compression_time for m in recent_alg_metrics]),
                    'avg_decompression_time': np.mean([m.decompression_time for m in recent_alg_metrics]),
                    'total_data_mb': sum(m.original_size for m in recent_alg_metrics) / (1024 * 1024)
                }
        
        return {
            'total_operations': len(recent_metrics),
            'average_compression_ratio': avg_ratio,
            'average_compression_time_ms': avg_compression_time * 1000,
            'average_decompression_time_ms': avg_decompression_time * 1000,
            'total_original_mb': total_original / (1024 * 1024),
            'total_compressed_mb': total_compressed / (1024 * 1024),
            'space_saved_mb': (total_original - total_compressed) / (1024 * 1024),
            'algorithm_performance': algorithm_stats
        }


class DatabaseOptimizer:
    """Advanced database optimization and indexing system"""
    
    def __init__(self, db_path: str = "./protein_ssl.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Index management
        self.active_indexes = {}
        self.index_usage_stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0})
        self.query_performance = deque(maxlen=1000)
        
        # Connection pool
        self.connection_pool = []
        self.max_connections = 10
        self._pool_lock = threading.Lock()
        
        # Initialize database
        self._initialize_database()
        
        # Auto-optimization
        self.auto_optimize_enabled = True
        self.last_optimization = time.time()
        self.optimization_interval = 3600  # 1 hour
    
    def _initialize_database(self):
        """Initialize database with optimized settings"""
        with self._get_connection() as conn:
            # Enable optimizations
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety and speed
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory map
            
            # Create core tables if they don't exist
            self._create_core_tables(conn)
            
            # Analyze existing tables
            conn.execute("ANALYZE")
    
    def _create_core_tables(self, conn: sqlite3.Connection):
        """Create core database tables"""
        # Protein sequences table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS protein_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_hash TEXT UNIQUE NOT NULL,
                sequence_data BLOB NOT NULL,
                length INTEGER NOT NULL,
                organism TEXT,
                family TEXT,
                created_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL
            )
        """)
        
        # Protein structures table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS protein_structures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id INTEGER NOT NULL,
                structure_data BLOB NOT NULL,
                structure_type TEXT NOT NULL,
                confidence_score REAL,
                model_version TEXT,
                computation_time REAL,
                created_at REAL NOT NULL,
                FOREIGN KEY (sequence_id) REFERENCES protein_sequences (id)
            )
        """)
        
        # Training data table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hash TEXT UNIQUE NOT NULL,
                input_data BLOB NOT NULL,
                target_data BLOB,
                data_type TEXT NOT NULL,
                split_type TEXT,  -- train, val, test
                created_at REAL NOT NULL,
                file_path TEXT
            )
        """)
        
        # Model checkpoints table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                checkpoint_data BLOB NOT NULL,
                epoch INTEGER,
                loss REAL,
                metrics BLOB,
                created_at REAL NOT NULL,
                is_best BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Performance metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp REAL NOT NULL,
                context TEXT,
                tags BLOB
            )
        """)
        
        conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection from pool"""
        with self._pool_lock:
            if self.connection_pool:
                conn = self.connection_pool.pop()
            else:
                conn = sqlite3.connect(str(self.db_path), timeout=30.0)
                conn.row_factory = sqlite3.Row
        
        try:
            yield conn
        finally:
            with self._pool_lock:
                if len(self.connection_pool) < self.max_connections:
                    self.connection_pool.append(conn)
                else:
                    conn.close()
    
    def create_index(self, table_name: str, columns: List[str], 
                    index_type: IndexType = IndexType.BTREE,
                    unique: bool = False, partial_condition: str = None) -> str:
        """Create optimized database index"""
        # Generate index name
        column_str = "_".join(columns)
        index_name = f"idx_{table_name}_{column_str}"
        
        if unique:
            index_name = f"uniq_{index_name}"
        
        # Build CREATE INDEX statement
        index_sql = f"CREATE {'UNIQUE ' if unique else ''}INDEX IF NOT EXISTS {index_name}"
        index_sql += f" ON {table_name} ({', '.join(columns)})"
        
        if partial_condition:
            index_sql += f" WHERE {partial_condition}"
        
        try:
            start_time = time.perf_counter()
            
            with self._get_connection() as conn:
                conn.execute(index_sql)
                conn.commit()
                
                # Analyze table to update statistics
                conn.execute(f"ANALYZE {table_name}")
            
            creation_time = time.perf_counter() - start_time
            
            # Record index information
            index_info = IndexMetrics(
                index_name=index_name,
                index_type=index_type,
                table_name=table_name,
                size_mb=self._get_index_size(index_name),
                selectivity=self._calculate_index_selectivity(table_name, columns),
                usage_count=0,
                avg_query_time=0.0,
                creation_time=creation_time
            )
            
            self.active_indexes[index_name] = index_info
            
            logger.info(f"Created index {index_name} on {table_name}({', '.join(columns)}) "
                       f"in {creation_time:.3f}s")
            
            return index_name
            
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            raise
    
    def drop_index(self, index_name: str) -> bool:
        """Drop database index"""
        try:
            with self._get_connection() as conn:
                conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                conn.commit()
            
            if index_name in self.active_indexes:
                del self.active_indexes[index_name]
            
            logger.info(f"Dropped index {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop index {index_name}: {e}")
            return False
    
    def optimize_query(self, query: str, parameters: Tuple = None) -> Tuple[Any, float]:
        """Execute and optimize database query"""
        start_time = time.perf_counter()
        
        try:
            with self._get_connection() as conn:
                # Enable query planner debugging if needed
                if logger.isEnabledFor(logging.DEBUG):
                    conn.execute("PRAGMA optimize")
                
                cursor = conn.cursor()
                
                if parameters:
                    result = cursor.execute(query, parameters).fetchall()
                else:
                    result = cursor.execute(query).fetchall()
                
                execution_time = time.perf_counter() - start_time
                
                # Track query performance
                self.query_performance.append({
                    'query': query[:100],  # First 100 chars
                    'execution_time': execution_time,
                    'result_count': len(result) if result else 0,
                    'timestamp': time.time()
                })
                
                # Update index usage statistics
                self._update_index_usage_stats(query, execution_time)
                
                # Auto-optimization check
                if (self.auto_optimize_enabled and 
                    time.time() - self.last_optimization > self.optimization_interval):
                    self._auto_optimize_database()
                
                return result, execution_time
                
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Query failed after {execution_time:.3f}s: {e}")
            raise
    
    def _get_index_size(self, index_name: str) -> float:
        """Get index size in MB"""
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    "SELECT SUM(pgsize) FROM dbstat WHERE name = ?",
                    (index_name,)
                ).fetchone()
                
                if result and result[0]:
                    return result[0] / (1024 * 1024)  # Convert to MB
                
        except Exception:
            pass
        
        return 0.0
    
    def _calculate_index_selectivity(self, table_name: str, columns: List[str]) -> float:
        """Calculate index selectivity (0 = not selective, 1 = very selective)"""
        try:
            with self._get_connection() as conn:
                # Get total row count
                total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                
                if total_rows == 0:
                    return 1.0
                
                # Get distinct value count for indexed columns
                column_expr = ", ".join(columns)
                distinct_values = conn.execute(
                    f"SELECT COUNT(DISTINCT ({column_expr})) FROM {table_name}"
                ).fetchone()[0]
                
                # Selectivity = distinct_values / total_rows
                selectivity = distinct_values / total_rows
                return min(selectivity, 1.0)
                
        except Exception as e:
            logger.warning(f"Failed to calculate selectivity for {table_name}.{columns}: {e}")
            return 0.5  # Default moderate selectivity
    
    def _update_index_usage_stats(self, query: str, execution_time: float):
        """Update index usage statistics based on query"""
        # Simple heuristic: if query contains table names that have indexes, count as usage
        query_lower = query.lower()
        
        for index_name, index_info in self.active_indexes.items():
            table_name = index_info.table_name.lower()
            
            if table_name in query_lower:
                stats = self.index_usage_stats[index_name]
                stats['count'] += 1
                stats['total_time'] += execution_time
                
                # Update average query time
                index_info.usage_count = stats['count']
                index_info.avg_query_time = stats['total_time'] / stats['count']
    
    def _auto_optimize_database(self):
        """Automatic database optimization"""
        try:
            start_time = time.perf_counter()
            
            with self._get_connection() as conn:
                # Update table statistics
                conn.execute("ANALYZE")
                
                # Optimize database
                conn.execute("PRAGMA optimize")
                
                # Vacuum if needed (check fragmentation)
                fragmentation = self._check_fragmentation(conn)
                if fragmentation > 0.3:  # 30% fragmentation
                    logger.info("Database fragmentation detected, running VACUUM")
                    conn.execute("VACUUM")
            
            optimization_time = time.perf_counter() - start_time
            self.last_optimization = time.time()
            
            logger.info(f"Database auto-optimization completed in {optimization_time:.3f}s")
            
            # Suggest index optimizations
            self._suggest_index_optimizations()
            
        except Exception as e:
            logger.error(f"Auto-optimization failed: {e}")
    
    def _check_fragmentation(self, conn: sqlite3.Connection) -> float:
        """Check database fragmentation level"""
        try:
            # Get page count and freelist count
            page_count = conn.execute("PRAGMA page_count").fetchone()[0]
            freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
            
            if page_count > 0:
                fragmentation = freelist_count / page_count
                return fragmentation
                
        except Exception:
            pass
        
        return 0.0
    
    def _suggest_index_optimizations(self):
        """Suggest index optimizations based on usage patterns"""
        suggestions = []
        
        # Find unused indexes
        for index_name, index_info in self.active_indexes.items():
            if index_info.usage_count == 0 and index_info.creation_time < time.time() - 86400:
                suggestions.append(f"Consider dropping unused index: {index_name}")
        
        # Find slow queries that might benefit from indexes
        if self.query_performance:
            slow_queries = [q for q in self.query_performance if q['execution_time'] > 1.0]
            if slow_queries:
                suggestions.append(f"Found {len(slow_queries)} slow queries that might benefit from indexing")
        
        if suggestions:
            logger.info("Index optimization suggestions:")
            for suggestion in suggestions:
                logger.info(f"  - {suggestion}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            with self._get_connection() as conn:
                # Basic database info
                page_count = conn.execute("PRAGMA page_count").fetchone()[0]
                page_size = conn.execute("PRAGMA page_size").fetchone()[0]
                total_size_mb = (page_count * page_size) / (1024 * 1024)
                
                freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
                fragmentation = freelist_count / page_count if page_count > 0 else 0.0
                
                # Table statistics
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                
                table_stats = {}
                for table in tables:
                    table_name = table[0]
                    if not table_name.startswith('sqlite_'):
                        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                        table_stats[table_name] = {'row_count': row_count}
        
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}
        
        # Query performance statistics
        if self.query_performance:
            recent_queries = list(self.query_performance)[-100:]
            avg_query_time = np.mean([q['execution_time'] for q in recent_queries])
            slow_query_count = len([q for q in recent_queries if q['execution_time'] > 1.0])
        else:
            avg_query_time = 0.0
            slow_query_count = 0
        
        # Index statistics
        index_stats = {}
        for index_name, index_info in self.active_indexes.items():
            index_stats[index_name] = {
                'table': index_info.table_name,
                'size_mb': index_info.size_mb,
                'usage_count': index_info.usage_count,
                'avg_query_time': index_info.avg_query_time,
                'selectivity': index_info.selectivity
            }
        
        return {
            'database_size_mb': total_size_mb,
            'page_count': page_count,
            'page_size': page_size,
            'fragmentation_ratio': fragmentation,
            'table_statistics': table_stats,
            'index_count': len(self.active_indexes),
            'index_statistics': index_stats,
            'query_performance': {
                'total_queries': len(self.query_performance),
                'average_query_time_ms': avg_query_time * 1000,
                'slow_query_count': slow_query_count
            },
            'last_optimization': self.last_optimization,
            'auto_optimize_enabled': self.auto_optimize_enabled
        }


class TieredStorage:
    """Tiered storage management with automatic data migration"""
    
    def __init__(self, storage_config: Dict[str, str] = None):
        self.storage_config = storage_config or {
            StorageTier.HOT.value: "./storage/hot",
            StorageTier.WARM.value: "./storage/warm",
            StorageTier.COLD.value: "./storage/cold",
            StorageTier.FROZEN.value: "./storage/frozen"
        }
        
        # Create storage directories
        for tier_path in self.storage_config.values():
            Path(tier_path).mkdir(parents=True, exist_ok=True)
        
        # Data tracking
        self.data_registry = {}
        self.access_patterns = defaultdict(list)
        self.migration_history = []
        
        # Tier policies
        self.tier_policies = {
            StorageTier.HOT: {
                'max_age_days': 7,
                'min_access_frequency': 10,  # accesses per week
                'compression': CompressionType.LZ4
            },
            StorageTier.WARM: {
                'max_age_days': 30,
                'min_access_frequency': 2,   # accesses per month
                'compression': CompressionType.ZSTD
            },
            StorageTier.COLD: {
                'max_age_days': 365,
                'min_access_frequency': 0.1, # accesses per year
                'compression': CompressionType.ZSTD
            },
            StorageTier.FROZEN: {
                'max_age_days': float('inf'),
                'min_access_frequency': 0,
                'compression': CompressionType.ZSTD
            }
        }
        
        # Compression manager
        self.compression_manager = CompressionManager()
        
        # Background migration
        self.migration_thread = None
        self.migration_active = False
        self.migration_interval = 3600  # 1 hour
    
    def store_data(self, data_id: str, data: bytes, tier: StorageTier = StorageTier.HOT,
                  metadata: Dict[str, Any] = None) -> bool:
        """Store data in specified tier"""
        try:
            tier_path = Path(self.storage_config[tier.value])
            file_path = tier_path / f"{data_id}.dat"
            
            # Compress data based on tier policy
            compression_type = self.tier_policies[tier]['compression']
            compressed_data, actual_compression, metrics = self.compression_manager.compress(
                data, compression_type
            )
            
            # Write to file
            with open(file_path, 'wb') as f:
                # Write header with metadata
                header = {
                    'original_size': len(data),
                    'compression': actual_compression.value,
                    'metadata': metadata or {},
                    'created_at': time.time(),
                    'tier': tier.value
                }
                header_bytes = json.dumps(header).encode('utf-8')
                header_size = len(header_bytes)
                
                # Write header size (4 bytes) + header + compressed data
                f.write(struct.pack('<I', header_size))
                f.write(header_bytes)
                f.write(compressed_data)
            
            # Register data
            self.data_registry[data_id] = {
                'tier': tier,
                'file_path': str(file_path),
                'created_at': time.time(),
                'last_accessed': time.time(),
                'access_count': 0,
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_type': actual_compression,
                'metadata': metadata or {}
            }
            
            logger.debug(f"Stored data {data_id} in {tier.value} tier "
                        f"(compression: {metrics.compression_ratio:.2f}x)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data {data_id}: {e}")
            return False
    
    def retrieve_data(self, data_id: str) -> Optional[bytes]:
        """Retrieve data from any tier"""
        if data_id not in self.data_registry:
            return None
        
        try:
            entry = self.data_registry[data_id]
            file_path = Path(entry['file_path'])
            
            if not file_path.exists():
                logger.warning(f"Data file missing for {data_id}: {file_path}")
                return None
            
            # Read file
            with open(file_path, 'rb') as f:
                # Read header
                header_size = struct.unpack('<I', f.read(4))[0]
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes.decode('utf-8'))
                
                # Read compressed data
                compressed_data = f.read()
            
            # Decompress
            compression_type = CompressionType(header['compression'])
            data = self.compression_manager.decompress(compressed_data, compression_type)
            
            # Update access tracking
            current_time = time.time()
            entry['last_accessed'] = current_time
            entry['access_count'] += 1
            
            self.access_patterns[data_id].append(current_time)
            
            # Consider promoting to higher tier if accessed frequently
            self._consider_promotion(data_id)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve data {data_id}: {e}")
            return None
    
    def delete_data(self, data_id: str) -> bool:
        """Delete data from storage"""
        if data_id not in self.data_registry:
            return False
        
        try:
            entry = self.data_registry[data_id]
            file_path = Path(entry['file_path'])
            
            if file_path.exists():
                file_path.unlink()
            
            del self.data_registry[data_id]
            
            if data_id in self.access_patterns:
                del self.access_patterns[data_id]
            
            logger.debug(f"Deleted data {data_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete data {data_id}: {e}")
            return False
    
    def start_automatic_migration(self):
        """Start automatic data migration between tiers"""
        if self.migration_active:
            return
        
        self.migration_active = True
        self.migration_thread = threading.Thread(
            target=self._migration_loop, daemon=True
        )
        self.migration_thread.start()
        
        logger.info("Automatic data migration started")
    
    def stop_automatic_migration(self):
        """Stop automatic data migration"""
        self.migration_active = False
        
        if self.migration_thread:
            self.migration_thread.join(timeout=10.0)
        
        logger.info("Automatic data migration stopped")
    
    def _migration_loop(self):
        """Background migration loop"""
        while self.migration_active:
            try:
                self._perform_migrations()
                time.sleep(self.migration_interval)
            except Exception as e:
                logger.error(f"Migration loop error: {e}")
                time.sleep(self.migration_interval)
    
    def _perform_migrations(self):
        """Perform data migrations based on access patterns"""
        current_time = time.time()
        migrations_performed = 0
        
        for data_id, entry in list(self.data_registry.items()):
            current_tier = entry['tier']
            suggested_tier = self._suggest_tier_for_data(data_id, current_time)
            
            if suggested_tier != current_tier:
                if self._migrate_data(data_id, suggested_tier):
                    migrations_performed += 1
        
        if migrations_performed > 0:
            logger.info(f"Performed {migrations_performed} data migrations")
    
    def _suggest_tier_for_data(self, data_id: str, current_time: float) -> StorageTier:
        """Suggest optimal tier for data based on access patterns"""
        entry = self.data_registry[data_id]
        age_days = (current_time - entry['created_at']) / 86400
        
        # Calculate access frequency (accesses per day)
        access_times = self.access_patterns.get(data_id, [])
        recent_accesses = [t for t in access_times if current_time - t < 86400 * 30]  # Last 30 days
        access_frequency = len(recent_accesses) / min(30, age_days + 1)
        
        # Determine tier based on age and access frequency
        if access_frequency >= 1.0:  # Daily access
            return StorageTier.HOT
        elif access_frequency >= 0.1:  # Weekly access
            return StorageTier.WARM
        elif age_days < 365 and access_frequency > 0:
            return StorageTier.COLD
        else:
            return StorageTier.FROZEN
    
    def _consider_promotion(self, data_id: str):
        """Consider promoting data to higher tier based on access"""
        entry = self.data_registry[data_id]
        current_tier = entry['tier']
        
        # Count recent accesses
        current_time = time.time()
        recent_accesses = [
            t for t in self.access_patterns[data_id] 
            if current_time - t < 3600  # Last hour
        ]
        
        # Promote if frequently accessed
        if len(recent_accesses) >= 3 and current_tier != StorageTier.HOT:
            if current_tier == StorageTier.WARM:
                self._migrate_data(data_id, StorageTier.HOT)
            elif current_tier in [StorageTier.COLD, StorageTier.FROZEN]:
                self._migrate_data(data_id, StorageTier.WARM)
    
    def _migrate_data(self, data_id: str, target_tier: StorageTier) -> bool:
        """Migrate data to target tier"""
        try:
            # Retrieve data
            data = self.retrieve_data(data_id)
            if data is None:
                return False
            
            # Get metadata
            entry = self.data_registry[data_id]
            metadata = entry['metadata']
            
            # Delete from current location
            old_tier = entry['tier']
            old_file_path = Path(entry['file_path'])
            if old_file_path.exists():
                old_file_path.unlink()
            
            # Store in new tier
            success = self.store_data(data_id, data, target_tier, metadata)
            
            if success:
                # Record migration
                self.migration_history.append({
                    'data_id': data_id,
                    'from_tier': old_tier.value,
                    'to_tier': target_tier.value,
                    'timestamp': time.time(),
                    'reason': 'automatic_migration'
                })
                
                logger.debug(f"Migrated data {data_id} from {old_tier.value} to {target_tier.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to migrate data {data_id}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        tier_stats = {}
        total_size = 0
        total_compressed_size = 0
        
        for tier in StorageTier:
            tier_data = [
                entry for entry in self.data_registry.values() 
                if entry['tier'] == tier
            ]
            
            tier_size = sum(entry['original_size'] for entry in tier_data)
            tier_compressed_size = sum(entry['compressed_size'] for entry in tier_data)
            
            tier_stats[tier.value] = {
                'item_count': len(tier_data),
                'total_size_mb': tier_size / (1024 * 1024),
                'compressed_size_mb': tier_compressed_size / (1024 * 1024),
                'compression_ratio': tier_size / tier_compressed_size if tier_compressed_size > 0 else 1.0,
                'storage_path': self.storage_config[tier.value]
            }
            
            total_size += tier_size
            total_compressed_size += tier_compressed_size
        
        # Access pattern analysis
        active_data_count = len([
            data_id for data_id, entry in self.data_registry.items()
            if time.time() - entry['last_accessed'] < 86400  # Accessed in last day
        ])
        
        return {
            'total_items': len(self.data_registry),
            'total_size_mb': total_size / (1024 * 1024),
            'total_compressed_size_mb': total_compressed_size / (1024 * 1024),
            'overall_compression_ratio': total_size / total_compressed_size if total_compressed_size > 0 else 1.0,
            'active_data_count': active_data_count,
            'tier_statistics': tier_stats,
            'migration_count': len(self.migration_history),
            'migration_active': self.migration_active,
            'compression_stats': self.compression_manager.get_compression_stats()
        }


class StorageOptimizer:
    """Central storage optimization coordinator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Components
        self.compression_manager = CompressionManager()
        self.database_optimizer = DatabaseOptimizer(
            self.config.get('database_path', './protein_ssl.db')
        )
        self.tiered_storage = TieredStorage(
            self.config.get('storage_tiers', None)
        )
        
        # Metrics
        self.metrics_collector = MetricsCollector()
        self.optimization_history = deque(maxlen=100)
        
        # Optimization state
        self.optimization_active = False
        self.optimization_thread = None
        self.optimization_interval = 3600  # 1 hour
    
    def start_optimization(self):
        """Start storage optimization system"""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        
        # Start tiered storage migration
        self.tiered_storage.start_automatic_migration()
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        self.optimization_thread.start()
        
        logger.info("Storage optimization started")
    
    def stop_optimization(self):
        """Stop storage optimization system"""
        if not self.optimization_active:
            return
        
        self.optimization_active = False
        
        # Stop tiered storage migration
        self.tiered_storage.stop_automatic_migration()
        
        # Stop optimization thread
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10.0)
        
        logger.info("Storage optimization stopped")
    
    def _optimization_loop(self):
        """Background optimization loop"""
        while self.optimization_active:
            try:
                self._perform_optimization_cycle()
                time.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(self.optimization_interval)
    
    def _perform_optimization_cycle(self):
        """Perform complete optimization cycle"""
        start_time = time.time()
        
        # Database optimization
        db_stats_before = self.database_optimizer.get_database_stats()
        
        # Storage analysis and cleanup
        storage_stats_before = self.tiered_storage.get_storage_stats()
        
        # Record optimization
        optimization_record = {
            'timestamp': start_time,
            'duration': time.time() - start_time,
            'database_stats_before': db_stats_before,
            'storage_stats_before': storage_stats_before,
            'optimizations_performed': []
        }
        
        self.optimization_history.append(optimization_record)
        
        logger.info(f"Storage optimization cycle completed in {optimization_record['duration']:.3f}s")
    
    def create_database_indexes(self, recommended_indexes: List[Dict[str, Any]] = None):
        """Create recommended database indexes"""
        if recommended_indexes is None:
            recommended_indexes = self._get_recommended_indexes()
        
        created_indexes = []
        
        for index_config in recommended_indexes:
            try:
                index_name = self.database_optimizer.create_index(
                    table_name=index_config['table'],
                    columns=index_config['columns'],
                    index_type=IndexType(index_config.get('type', 'btree')),
                    unique=index_config.get('unique', False),
                    partial_condition=index_config.get('condition')
                )
                created_indexes.append(index_name)
            except Exception as e:
                logger.warning(f"Failed to create index {index_config}: {e}")
        
        return created_indexes
    
    def _get_recommended_indexes(self) -> List[Dict[str, Any]]:
        """Get recommended database indexes for common queries"""
        return [
            # Protein sequences indexes
            {
                'table': 'protein_sequences',
                'columns': ['sequence_hash'],
                'type': 'btree',
                'unique': True
            },
            {
                'table': 'protein_sequences',
                'columns': ['organism', 'family'],
                'type': 'btree'
            },
            {
                'table': 'protein_sequences',
                'columns': ['length'],
                'type': 'btree'
            },
            {
                'table': 'protein_sequences',
                'columns': ['last_accessed'],
                'type': 'btree',
                'condition': 'last_accessed IS NOT NULL'
            },
            
            # Protein structures indexes
            {
                'table': 'protein_structures',
                'columns': ['sequence_id'],
                'type': 'btree'
            },
            {
                'table': 'protein_structures',
                'columns': ['structure_type', 'model_version'],
                'type': 'btree'
            },
            {
                'table': 'protein_structures',
                'columns': ['confidence_score'],
                'type': 'btree',
                'condition': 'confidence_score > 0.5'
            },
            
            # Training data indexes
            {
                'table': 'training_data',
                'columns': ['data_hash'],
                'type': 'btree',
                'unique': True
            },
            {
                'table': 'training_data',
                'columns': ['data_type', 'split_type'],
                'type': 'btree'
            },
            
            # Model checkpoints indexes
            {
                'table': 'model_checkpoints',
                'columns': ['model_name', 'epoch'],
                'type': 'btree'
            },
            {
                'table': 'model_checkpoints',
                'columns': ['is_best'],
                'type': 'btree',
                'condition': 'is_best = TRUE'
            },
            
            # Performance metrics indexes
            {
                'table': 'performance_metrics',
                'columns': ['metric_name', 'timestamp'],
                'type': 'btree'
            },
            {
                'table': 'performance_metrics',
                'columns': ['timestamp'],
                'type': 'btree'
            }
        ]
    
    def compress_data_batch(self, data_items: List[Tuple[str, bytes]], 
                          algorithm: CompressionType = None) -> Dict[str, CompressionMetrics]:
        """Compress batch of data items"""
        results = {}
        
        for data_id, data in data_items:
            try:
                compressed_data, used_algorithm, metrics = self.compression_manager.compress(
                    data, algorithm
                )
                results[data_id] = metrics
                
                # Store in tiered storage
                self.tiered_storage.store_data(
                    data_id, data, StorageTier.HOT,
                    metadata={'compression_metrics': asdict(metrics)}
                )
                
            except Exception as e:
                logger.error(f"Failed to compress data {data_id}: {e}")
        
        return results
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage optimization statistics"""
        return {
            'compression_stats': self.compression_manager.get_compression_stats(),
            'database_stats': self.database_optimizer.get_database_stats(),
            'storage_stats': self.tiered_storage.get_storage_stats(),
            'optimization_active': self.optimization_active,
            'optimization_cycles': len(self.optimization_history),
            'last_optimization': (
                self.optimization_history[-1]['timestamp'] 
                if self.optimization_history else None
            )
        }


# Global storage optimizer instance
_global_storage_optimizer = None

def get_storage_optimizer(**kwargs) -> StorageOptimizer:
    """Get global storage optimizer instance"""
    global _global_storage_optimizer
    
    if _global_storage_optimizer is None:
        _global_storage_optimizer = StorageOptimizer(**kwargs)
    
    return _global_storage_optimizer

def start_storage_optimization(**kwargs) -> StorageOptimizer:
    """Start global storage optimization"""
    optimizer = get_storage_optimizer(**kwargs)
    optimizer.start_optimization()
    return optimizer
"""
Advanced Network Optimization System for Protein-SSL Operator
Implements connection pooling, streaming, compression, and intelligent routing
"""

import time
import threading
import asyncio
import aiohttp
import socket
import ssl
import json
import gzip
import struct
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import psutil
import websockets
import uvloop
from concurrent.futures import ThreadPoolExecutor
import queue
import urllib3
from urllib3.util.retry import Retry
from urllib3.poolmanager import PoolManager
import requests
from requests.adapters import HTTPAdapter

from .logging_config import setup_logging
from .monitoring import MetricsCollector

logger = setup_logging(__name__)


class ProtocolType(Enum):
    """Network protocol types"""
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    TCP = "tcp"
    UDP = "udp"
    GRPC = "grpc"


class CompressionAlgorithm(Enum):
    """Network compression algorithms"""
    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "brotli"
    LZ4 = "lz4"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"


@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    protocol: ProtocolType
    endpoint: str
    request_count: int
    response_time_ms: float
    throughput_mbps: float
    error_rate: float
    connection_reuse_rate: float
    compression_ratio: float
    bytes_sent: int
    bytes_received: int
    active_connections: int


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration"""
    max_connections: int = 100
    max_connections_per_host: int = 20
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    keep_alive_timeout: float = 300.0
    retry_strategy: Dict[str, Any] = None
    enable_compression: bool = True
    ssl_verify: bool = True


@dataclass
class StreamingConfig:
    """Streaming configuration"""
    chunk_size: int = 8192
    buffer_size: int = 65536
    enable_compression: bool = True
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    max_concurrent_streams: int = 100


class ConnectionPool:
    """Advanced HTTP connection pool with intelligent management"""
    
    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        
        # HTTP pool manager
        retry_strategy = Retry(
            total=config.retry_strategy.get('total', 3) if config.retry_strategy else 3,
            backoff_factor=config.retry_strategy.get('backoff_factor', 0.3) if config.retry_strategy else 0.3,
            status_forcelist=config.retry_strategy.get('status_forcelist', [429, 500, 502, 503, 504]) if config.retry_strategy else [429, 500, 502, 503, 504]
        )
        
        self.pool_manager = PoolManager(
            num_pools=config.max_connections,
            maxsize=config.max_connections_per_host,
            timeout=urllib3.Timeout(
                connect=config.connection_timeout,
                read=config.read_timeout
            ),
            retries=retry_strategy,
            ssl_context=self._create_ssl_context() if config.ssl_verify else None
        )
        
        # Session management
        self.sessions = {}
        self.session_lock = threading.RLock()
        
        # Metrics tracking
        self.connection_metrics = defaultdict(lambda: {
            'requests': 0,
            'errors': 0,
            'total_time': 0.0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'active_connections': 0
        })
        
        # Connection health monitoring
        self.health_check_interval = 60.0  # seconds
        self.health_thread = None
        self.health_active = False
        
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure connections"""
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    
    def start_health_monitoring(self):
        """Start connection health monitoring"""
        if self.health_active:
            return
        
        self.health_active = True
        self.health_thread = threading.Thread(
            target=self._health_monitoring_loop, daemon=True
        )
        self.health_thread.start()
        logger.info("Connection health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop connection health monitoring"""
        self.health_active = False
        if self.health_thread:
            self.health_thread.join(timeout=10.0)
        logger.info("Connection health monitoring stopped")
    
    def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while self.health_active:
            try:
                self._check_connection_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.health_check_interval)
    
    def _check_connection_health(self):
        """Check health of active connections"""
        # Clean up stale sessions
        current_time = time.time()
        stale_sessions = []
        
        with self.session_lock:
            for session_id, session_info in self.sessions.items():
                if current_time - session_info['last_used'] > self.config.keep_alive_timeout:
                    stale_sessions.append(session_id)
            
            for session_id in stale_sessions:
                try:
                    session = self.sessions[session_id]['session']
                    session.close()
                    del self.sessions[session_id]
                    logger.debug(f"Cleaned up stale session {session_id}")
                except Exception as e:
                    logger.warning(f"Error cleaning up session {session_id}: {e}")
    
    def get_session(self, session_id: str = None) -> requests.Session:
        """Get or create HTTP session"""
        if session_id is None:
            session_id = f"default_{threading.current_thread().ident}"
        
        with self.session_lock:
            if session_id in self.sessions:
                session_info = self.sessions[session_id]
                session_info['last_used'] = time.time()
                return session_info['session']
            
            # Create new session
            session = requests.Session()
            
            # Configure session
            adapter = HTTPAdapter(
                pool_connections=self.config.max_connections_per_host,
                pool_maxsize=self.config.max_connections,
                max_retries=self.config.retry_strategy.get('total', 3) if self.config.retry_strategy else 3
            )
            
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            # Set timeouts
            session.timeout = (self.config.connection_timeout, self.config.read_timeout)
            
            # Enable compression
            if self.config.enable_compression:
                session.headers.update({
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive'
                })
            
            self.sessions[session_id] = {
                'session': session,
                'created_at': time.time(),
                'last_used': time.time()
            }
            
            return session
    
    def request(self, method: str, url: str, session_id: str = None, **kwargs) -> requests.Response:
        """Make HTTP request with connection pooling"""
        start_time = time.perf_counter()
        
        try:
            session = self.get_session(session_id)
            
            # Add compression headers if enabled
            if self.config.enable_compression and 'headers' in kwargs:
                headers = kwargs.get('headers', {})
                if 'Accept-Encoding' not in headers:
                    headers['Accept-Encoding'] = 'gzip, deflate, br'
                kwargs['headers'] = headers
            
            response = session.request(method, url, **kwargs)
            
            # Update metrics
            execution_time = time.perf_counter() - start_time
            self._update_metrics(url, execution_time, response, kwargs)
            
            return response
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self._update_error_metrics(url, execution_time)
            raise
    
    def _update_metrics(self, url: str, execution_time: float, 
                       response: requests.Response, request_kwargs: Dict[str, Any]):
        """Update connection metrics"""
        host = self._extract_host(url)
        metrics = self.connection_metrics[host]
        
        metrics['requests'] += 1
        metrics['total_time'] += execution_time
        
        # Estimate bytes
        request_size = self._estimate_request_size(request_kwargs)
        response_size = len(response.content) if hasattr(response, 'content') else 0
        
        metrics['bytes_sent'] += request_size
        metrics['bytes_received'] += response_size
        
        if response.status_code >= 400:
            metrics['errors'] += 1
    
    def _update_error_metrics(self, url: str, execution_time: float):
        """Update error metrics"""
        host = self._extract_host(url)
        metrics = self.connection_metrics[host]
        
        metrics['requests'] += 1
        metrics['errors'] += 1
        metrics['total_time'] += execution_time
    
    def _extract_host(self, url: str) -> str:
        """Extract host from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return "unknown"
    
    def _estimate_request_size(self, request_kwargs: Dict[str, Any]) -> int:
        """Estimate request size in bytes"""
        size = 100  # Base HTTP headers
        
        if 'data' in request_kwargs:
            data = request_kwargs['data']
            if isinstance(data, (str, bytes)):
                size += len(data)
            else:
                size += len(str(data))
        
        if 'json' in request_kwargs:
            size += len(json.dumps(request_kwargs['json']))
        
        if 'headers' in request_kwargs:
            for key, value in request_kwargs['headers'].items():
                size += len(key) + len(str(value)) + 4  # ": " + "\r\n"
        
        return size
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        stats_by_host = {}
        
        for host, metrics in self.connection_metrics.items():
            avg_response_time = (
                metrics['total_time'] / max(metrics['requests'], 1) * 1000  # ms
            )
            error_rate = metrics['errors'] / max(metrics['requests'], 1)
            
            stats_by_host[host] = {
                'requests': metrics['requests'],
                'errors': metrics['errors'],
                'error_rate': error_rate,
                'avg_response_time_ms': avg_response_time,
                'bytes_sent': metrics['bytes_sent'],
                'bytes_received': metrics['bytes_received'],
                'active_connections': metrics['active_connections']
            }
        
        return {
            'total_sessions': len(self.sessions),
            'pool_config': asdict(self.config),
            'stats_by_host': stats_by_host,
            'health_monitoring_active': self.health_active
        }


class AsyncConnectionPool:
    """Asynchronous connection pool for high-performance async operations"""
    
    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self.connector = None
        self.session = None
        self.metrics = defaultdict(lambda: {
            'requests': 0,
            'errors': 0,
            'total_time': 0.0,
            'bytes_sent': 0,
            'bytes_received': 0
        })
        
    async def initialize(self):
        """Initialize async connection pool"""
        # Configure connector
        self.connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.max_connections_per_host,
            keepalive_timeout=self.config.keep_alive_timeout,
            enable_cleanup_closed=True,
            ssl=False if not self.config.ssl_verify else None
        )
        
        # Configure timeout
        timeout = aiohttp.ClientTimeout(
            total=self.config.read_timeout,
            connect=self.config.connection_timeout
        )
        
        # Create session
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            } if self.config.enable_compression else None
        )
        
        logger.info("Async connection pool initialized")
    
    async def cleanup(self):
        """Cleanup async connection pool"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
        logger.info("Async connection pool cleaned up")
    
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make async HTTP request"""
        if not self.session:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                # Read response content for metrics
                content = await response.read()
                
                # Update metrics
                execution_time = time.perf_counter() - start_time
                self._update_async_metrics(url, execution_time, response, kwargs, content)
                
                # Create response object with content
                response._content = content
                return response
                
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self._update_async_error_metrics(url, execution_time)
            raise
    
    def _update_async_metrics(self, url: str, execution_time: float,
                            response: aiohttp.ClientResponse, 
                            request_kwargs: Dict[str, Any], content: bytes):
        """Update async connection metrics"""
        host = self._extract_host(url)
        metrics = self.metrics[host]
        
        metrics['requests'] += 1
        metrics['total_time'] += execution_time
        
        # Estimate request size
        request_size = self._estimate_async_request_size(request_kwargs)
        metrics['bytes_sent'] += request_size
        metrics['bytes_received'] += len(content)
        
        if response.status >= 400:
            metrics['errors'] += 1
    
    def _update_async_error_metrics(self, url: str, execution_time: float):
        """Update async error metrics"""
        host = self._extract_host(url)
        metrics = self.metrics[host]
        
        metrics['requests'] += 1
        metrics['errors'] += 1
        metrics['total_time'] += execution_time
    
    def _extract_host(self, url: str) -> str:
        """Extract host from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return "unknown"
    
    def _estimate_async_request_size(self, request_kwargs: Dict[str, Any]) -> int:
        """Estimate async request size"""
        size = 100  # Base headers
        
        if 'data' in request_kwargs:
            data = request_kwargs['data']
            if isinstance(data, (str, bytes)):
                size += len(data)
            else:
                size += len(str(data))
        
        if 'json' in request_kwargs:
            size += len(json.dumps(request_kwargs['json']))
        
        return size
    
    def get_async_stats(self) -> Dict[str, Any]:
        """Get async connection pool statistics"""
        stats_by_host = {}
        
        for host, metrics in self.metrics.items():
            avg_response_time = (
                metrics['total_time'] / max(metrics['requests'], 1) * 1000
            )
            error_rate = metrics['errors'] / max(metrics['requests'], 1)
            
            stats_by_host[host] = {
                'requests': metrics['requests'],
                'errors': metrics['errors'],
                'error_rate': error_rate,
                'avg_response_time_ms': avg_response_time,
                'bytes_sent': metrics['bytes_sent'],
                'bytes_received': metrics['bytes_received']
            }
        
        return {
            'connector_stats': self.connector._get_stats() if self.connector else {},
            'stats_by_host': stats_by_host,
            'session_active': self.session is not None and not self.session.closed
        }


class StreamingManager:
    """Advanced data streaming manager with compression and buffering"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.active_streams = {}
        self.stream_metrics = defaultdict(lambda: {
            'bytes_streamed': 0,
            'chunks_sent': 0,
            'compression_ratio': 1.0,
            'duration': 0.0
        })
        
        # Compression setup
        self.compressors = {
            CompressionAlgorithm.GZIP: self._create_gzip_compressor,
            CompressionAlgorithm.DEFLATE: self._create_deflate_compressor,
            CompressionAlgorithm.LZ4: self._create_lz4_compressor
        }
    
    def _create_gzip_compressor(self):
        """Create GZIP compressor"""
        return gzip.GzipFile(fileobj=None, mode='wb', compresslevel=6)
    
    def _create_deflate_compressor(self):
        """Create DEFLATE compressor"""
        import zlib
        return zlib.compressobj(level=6, wbits=-15)
    
    def _create_lz4_compressor(self):
        """Create LZ4 compressor"""
        try:
            import lz4.frame
            return lz4.frame.LZ4FrameCompressor()
        except ImportError:
            logger.warning("LZ4 not available, falling back to gzip")
            return self._create_gzip_compressor()
    
    async def create_stream(self, stream_id: str, destination: str, 
                          compression: CompressionAlgorithm = None) -> 'DataStream':
        """Create new data stream"""
        if compression is None:
            compression = self.config.compression_algorithm
        
        stream = DataStream(
            stream_id=stream_id,
            destination=destination,
            config=self.config,
            compression=compression,
            manager=self
        )
        
        self.active_streams[stream_id] = stream
        logger.info(f"Created stream {stream_id} to {destination}")
        
        return stream
    
    def close_stream(self, stream_id: str):
        """Close and cleanup stream"""
        if stream_id in self.active_streams:
            stream = self.active_streams[stream_id]
            stream.close()
            del self.active_streams[stream_id]
            logger.info(f"Closed stream {stream_id}")
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            'active_streams': len(self.active_streams),
            'max_concurrent_streams': self.config.max_concurrent_streams,
            'stream_metrics': dict(self.stream_metrics),
            'config': asdict(self.config)
        }


class DataStream:
    """Individual data stream with compression and buffering"""
    
    def __init__(self, stream_id: str, destination: str, 
                 config: StreamingConfig, compression: CompressionAlgorithm,
                 manager: StreamingManager):
        self.stream_id = stream_id
        self.destination = destination
        self.config = config
        self.compression = compression
        self.manager = manager
        
        # Stream state
        self.buffer = bytearray()
        self.total_bytes = 0
        self.chunks_sent = 0
        self.start_time = time.time()
        self.closed = False
        
        # Compression
        self.compressor = None
        if compression != CompressionAlgorithm.NONE:
            self.compressor = manager.compressors[compression]()
    
    async def write(self, data: bytes):
        """Write data to stream"""
        if self.closed:
            raise RuntimeError("Stream is closed")
        
        # Add to buffer
        self.buffer.extend(data)
        self.total_bytes += len(data)
        
        # Flush if buffer is full
        if len(self.buffer) >= self.config.buffer_size:
            await self.flush()
    
    async def write_chunk(self, data: bytes):
        """Write data chunk directly (bypassing buffer)"""
        if self.closed:
            raise RuntimeError("Stream is closed")
        
        compressed_data = self._compress_data(data)
        await self._send_chunk(compressed_data)
        
        self.total_bytes += len(data)
        self.chunks_sent += 1
    
    async def flush(self):
        """Flush buffered data"""
        if not self.buffer or self.closed:
            return
        
        # Compress buffered data
        data_to_send = bytes(self.buffer)
        compressed_data = self._compress_data(data_to_send)
        
        # Send in chunks
        for i in range(0, len(compressed_data), self.config.chunk_size):
            chunk = compressed_data[i:i + self.config.chunk_size]
            await self._send_chunk(chunk)
            self.chunks_sent += 1
        
        # Clear buffer
        self.buffer.clear()
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if compression is enabled"""
        if self.compression == CompressionAlgorithm.NONE or not self.compressor:
            return data
        
        try:
            if self.compression == CompressionAlgorithm.GZIP:
                return gzip.compress(data)
            elif self.compression == CompressionAlgorithm.DEFLATE:
                import zlib
                return zlib.compress(data)
            elif self.compression == CompressionAlgorithm.LZ4:
                import lz4.frame
                return lz4.frame.compress(data)
            else:
                return data
        except Exception as e:
            logger.warning(f"Compression failed for stream {self.stream_id}: {e}")
            return data
    
    async def _send_chunk(self, chunk: bytes):
        """Send chunk to destination"""
        # This would implement the actual network sending
        # For now, simulate the sending
        await asyncio.sleep(0.001)  # Simulate network delay
        
        # Update metrics
        metrics = self.manager.stream_metrics[self.stream_id]
        metrics['bytes_streamed'] += len(chunk)
    
    def close(self):
        """Close the stream"""
        if not self.closed:
            # Flush any remaining data
            if self.buffer:
                asyncio.create_task(self.flush())
            
            self.closed = True
            
            # Update final metrics
            metrics = self.manager.stream_metrics[self.stream_id]
            metrics['chunks_sent'] = self.chunks_sent
            metrics['duration'] = time.time() - self.start_time
            
            if self.compression != CompressionAlgorithm.NONE:
                # Estimate compression ratio
                estimated_original = metrics['bytes_streamed'] * 1.5  # Rough estimate
                metrics['compression_ratio'] = estimated_original / max(metrics['bytes_streamed'], 1)


class LoadBalancer:
    """Network load balancer with multiple strategies"""
    
    def __init__(self, endpoints: List[str], strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.endpoints = endpoints
        self.strategy = strategy
        
        # Endpoint health and metrics
        self.endpoint_health = {ep: True for ep in endpoints}
        self.endpoint_metrics = {ep: {
            'requests': 0,
            'errors': 0,
            'total_response_time': 0.0,
            'active_connections': 0,
            'weight': 1.0
        } for ep in endpoints}
        
        # Strategy state
        self.round_robin_index = 0
        self._lock = threading.Lock()
        
        # Health checking
        self.health_check_interval = 30.0  # seconds
        self.health_thread = None
        self.health_active = False
    
    def start_health_checking(self):
        """Start endpoint health checking"""
        if self.health_active:
            return
        
        self.health_active = True
        self.health_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self.health_thread.start()
        logger.info("Load balancer health checking started")
    
    def stop_health_checking(self):
        """Stop endpoint health checking"""
        self.health_active = False
        if self.health_thread:
            self.health_thread.join(timeout=10.0)
        logger.info("Load balancer health checking stopped")
    
    def _health_check_loop(self):
        """Background health check loop"""
        while self.health_active:
            try:
                self._check_endpoint_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(self.health_check_interval)
    
    def _check_endpoint_health(self):
        """Check health of all endpoints"""
        for endpoint in self.endpoints:
            try:
                # Simple HTTP health check
                response = requests.get(f"{endpoint}/health", timeout=5.0)
                self.endpoint_health[endpoint] = response.status_code == 200
            except Exception:
                self.endpoint_health[endpoint] = False
    
    def get_endpoint(self, client_ip: str = None) -> Optional[str]:
        """Get next endpoint based on load balancing strategy"""
        healthy_endpoints = [
            ep for ep in self.endpoints 
            if self.endpoint_health.get(ep, False)
        ]
        
        if not healthy_endpoints:
            logger.warning("No healthy endpoints available")
            return None
        
        with self._lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(healthy_endpoints)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(healthy_endpoints)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(healthy_endpoints)
            elif self.strategy == LoadBalancingStrategy.IP_HASH:
                return self._ip_hash_selection(healthy_endpoints, client_ip)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_selection(healthy_endpoints)
            else:
                return self._round_robin_selection(healthy_endpoints)
    
    def _round_robin_selection(self, endpoints: List[str]) -> str:
        """Round-robin endpoint selection"""
        endpoint = endpoints[self.round_robin_index % len(endpoints)]
        self.round_robin_index += 1
        return endpoint
    
    def _least_connections_selection(self, endpoints: List[str]) -> str:
        """Least connections endpoint selection"""
        return min(endpoints, 
                  key=lambda ep: self.endpoint_metrics[ep]['active_connections'])
    
    def _weighted_round_robin_selection(self, endpoints: List[str]) -> str:
        """Weighted round-robin selection"""
        # Simple implementation - could be improved with proper weighted selection
        weights = [self.endpoint_metrics[ep]['weight'] for ep in endpoints]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return self._round_robin_selection(endpoints)
        
        # Weighted selection based on normalized weights
        import random
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, endpoint in enumerate(endpoints):
            cumulative += weights[i]
            if r <= cumulative:
                return endpoint
        
        return endpoints[-1]  # Fallback
    
    def _ip_hash_selection(self, endpoints: List[str], client_ip: str) -> str:
        """IP hash-based selection for session affinity"""
        if not client_ip:
            return self._round_robin_selection(endpoints)
        
        # Simple hash-based selection
        hash_value = hash(client_ip)
        return endpoints[hash_value % len(endpoints)]
    
    def _least_response_time_selection(self, endpoints: List[str]) -> str:
        """Least response time selection"""
        def avg_response_time(ep):
            metrics = self.endpoint_metrics[ep]
            if metrics['requests'] == 0:
                return 0.0
            return metrics['total_response_time'] / metrics['requests']
        
        return min(endpoints, key=avg_response_time)
    
    def record_request(self, endpoint: str, response_time: float, success: bool):
        """Record request metrics for endpoint"""
        if endpoint in self.endpoint_metrics:
            metrics = self.endpoint_metrics[endpoint]
            metrics['requests'] += 1
            metrics['total_response_time'] += response_time
            
            if not success:
                metrics['errors'] += 1
    
    def update_active_connections(self, endpoint: str, delta: int):
        """Update active connection count"""
        if endpoint in self.endpoint_metrics:
            self.endpoint_metrics[endpoint]['active_connections'] += delta
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        stats_by_endpoint = {}
        
        for endpoint in self.endpoints:
            metrics = self.endpoint_metrics[endpoint]
            avg_response_time = (
                metrics['total_response_time'] / max(metrics['requests'], 1)
            )
            error_rate = metrics['errors'] / max(metrics['requests'], 1)
            
            stats_by_endpoint[endpoint] = {
                'healthy': self.endpoint_health.get(endpoint, False),
                'requests': metrics['requests'],
                'errors': metrics['errors'],
                'error_rate': error_rate,
                'avg_response_time_ms': avg_response_time * 1000,
                'active_connections': metrics['active_connections'],
                'weight': metrics['weight']
            }
        
        return {
            'strategy': self.strategy.value,
            'total_endpoints': len(self.endpoints),
            'healthy_endpoints': sum(self.endpoint_health.values()),
            'stats_by_endpoint': stats_by_endpoint,
            'health_checking_active': self.health_active
        }


class NetworkOptimizer:
    """Central network optimization coordinator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Connection pools
        self.sync_pool = ConnectionPool(
            ConnectionPoolConfig(**self.config.get('connection_pool', {}))
        )
        self.async_pool = AsyncConnectionPool(
            ConnectionPoolConfig(**self.config.get('async_connection_pool', {}))
        )
        
        # Streaming manager
        self.streaming_manager = StreamingManager(
            StreamingConfig(**self.config.get('streaming', {}))
        )
        
        # Load balancer
        endpoints = self.config.get('load_balancer', {}).get('endpoints', [])
        if endpoints:
            strategy = LoadBalancingStrategy(
                self.config.get('load_balancer', {}).get('strategy', 'round_robin')
            )
            self.load_balancer = LoadBalancer(endpoints, strategy)
        else:
            self.load_balancer = None
        
        # Optimization state
        self.optimization_active = False
        self.optimization_thread = None
        self.optimization_interval = 300  # 5 minutes
        
        # Metrics
        self.network_metrics = defaultdict(lambda: NetworkMetrics(
            protocol=ProtocolType.HTTP,
            endpoint="",
            request_count=0,
            response_time_ms=0.0,
            throughput_mbps=0.0,
            error_rate=0.0,
            connection_reuse_rate=0.0,
            compression_ratio=1.0,
            bytes_sent=0,
            bytes_received=0,
            active_connections=0
        ))
    
    def start_optimization(self):
        """Start network optimization"""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        
        # Start connection pool health monitoring
        self.sync_pool.start_health_monitoring()
        
        # Start load balancer health checking
        if self.load_balancer:
            self.load_balancer.start_health_checking()
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        self.optimization_thread.start()
        
        logger.info("Network optimization started")
    
    def stop_optimization(self):
        """Stop network optimization"""
        if not self.optimization_active:
            return
        
        self.optimization_active = False
        
        # Stop connection pool monitoring
        self.sync_pool.stop_health_monitoring()
        
        # Stop load balancer health checking
        if self.load_balancer:
            self.load_balancer.stop_health_checking()
        
        # Stop optimization thread
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10.0)
        
        # Cleanup async pool
        if self.async_pool.session:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.async_pool.cleanup())
            loop.close()
        
        logger.info("Network optimization stopped")
    
    def _optimization_loop(self):
        """Background network optimization loop"""
        while self.optimization_active:
            try:
                self._perform_optimization_cycle()
                time.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"Network optimization error: {e}")
                time.sleep(self.optimization_interval)
    
    def _perform_optimization_cycle(self):
        """Perform network optimization cycle"""
        # Analyze connection patterns
        self._analyze_connection_patterns()
        
        # Optimize connection pool settings
        self._optimize_connection_pools()
        
        # Update load balancer weights
        self._optimize_load_balancer()
        
        logger.debug("Network optimization cycle completed")
    
    def _analyze_connection_patterns(self):
        """Analyze network connection patterns"""
        # Get pool statistics
        pool_stats = self.sync_pool.get_pool_stats()
        
        # Analyze for optimization opportunities
        for host, stats in pool_stats.get('stats_by_host', {}).items():
            if stats['error_rate'] > 0.1:  # High error rate
                logger.warning(f"High error rate for {host}: {stats['error_rate']:.2%}")
            
            if stats['avg_response_time_ms'] > 5000:  # Slow responses
                logger.warning(f"Slow responses for {host}: {stats['avg_response_time_ms']:.1f}ms")
    
    def _optimize_connection_pools(self):
        """Optimize connection pool configurations"""
        # This could implement dynamic pool sizing based on usage patterns
        pass
    
    def _optimize_load_balancer(self):
        """Optimize load balancer weights and configuration"""
        if not self.load_balancer:
            return
        
        # Adjust weights based on performance
        lb_stats = self.load_balancer.get_load_balancer_stats()
        
        for endpoint, stats in lb_stats.get('stats_by_endpoint', {}).items():
            # Reduce weight for high error rate or slow endpoints
            if stats['error_rate'] > 0.1 or stats['avg_response_time_ms'] > 2000:
                current_weight = self.load_balancer.endpoint_metrics[endpoint]['weight']
                new_weight = max(0.1, current_weight * 0.8)
                self.load_balancer.endpoint_metrics[endpoint]['weight'] = new_weight
                logger.debug(f"Reduced weight for {endpoint}: {new_weight}")
    
    def make_request(self, method: str, url: str, use_load_balancer: bool = True, **kwargs):
        """Make optimized HTTP request"""
        # Use load balancer if available
        if use_load_balancer and self.load_balancer:
            endpoint = self.load_balancer.get_endpoint()
            if endpoint:
                # Replace hostname in URL with load-balanced endpoint
                from urllib.parse import urlparse, urlunparse
                parsed = urlparse(url)
                lb_parsed = urlparse(endpoint)
                new_url = urlunparse((
                    lb_parsed.scheme,
                    lb_parsed.netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))
                url = new_url
        
        # Make request through connection pool
        start_time = time.perf_counter()
        
        try:
            response = self.sync_pool.request(method, url, **kwargs)
            
            # Record load balancer metrics
            if use_load_balancer and self.load_balancer and endpoint:
                response_time = time.perf_counter() - start_time
                success = response.status_code < 400
                self.load_balancer.record_request(endpoint, response_time, success)
            
            return response
            
        except Exception as e:
            # Record failure
            if use_load_balancer and self.load_balancer and endpoint:
                response_time = time.perf_counter() - start_time
                self.load_balancer.record_request(endpoint, response_time, False)
            raise
    
    async def make_async_request(self, method: str, url: str, **kwargs):
        """Make optimized async HTTP request"""
        return await self.async_pool.request(method, url, **kwargs)
    
    async def create_stream(self, stream_id: str, destination: str, **kwargs):
        """Create optimized data stream"""
        return await self.streaming_manager.create_stream(stream_id, destination, **kwargs)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive network optimization statistics"""
        stats = {
            'optimization_active': self.optimization_active,
            'sync_pool_stats': self.sync_pool.get_pool_stats(),
            'async_pool_stats': self.async_pool.get_async_stats(),
            'streaming_stats': self.streaming_manager.get_streaming_stats()
        }
        
        if self.load_balancer:
            stats['load_balancer_stats'] = self.load_balancer.get_load_balancer_stats()
        
        return stats


# Global network optimizer instance
_global_network_optimizer = None

def get_network_optimizer(**kwargs) -> NetworkOptimizer:
    """Get global network optimizer instance"""
    global _global_network_optimizer
    
    if _global_network_optimizer is None:
        _global_network_optimizer = NetworkOptimizer(**kwargs)
    
    return _global_network_optimizer

def start_network_optimization(**kwargs) -> NetworkOptimizer:
    """Start global network optimization"""
    optimizer = get_network_optimizer(**kwargs)
    optimizer.start_optimization()
    return optimizer

def make_optimized_request(method: str, url: str, **kwargs):
    """Make optimized HTTP request using global optimizer"""
    optimizer = get_network_optimizer()
    return optimizer.make_request(method, url, **kwargs)

async def make_optimized_async_request(method: str, url: str, **kwargs):
    """Make optimized async HTTP request using global optimizer"""
    optimizer = get_network_optimizer()
    return await optimizer.make_async_request(method, url, **kwargs)
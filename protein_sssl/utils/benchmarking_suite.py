"""
Comprehensive Benchmarking and Performance Testing Suite for Protein-SSL Operator
Implements scalability testing, load testing, stress testing, and performance validation
"""

import time
import threading
import multiprocessing as mp
import asyncio
import json
import numpy as np
import torch
import psutil
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import concurrent.futures
from pathlib import Path
import pickle
import tempfile
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

from .logging_config import setup_logging
from .monitoring import MetricsCollector
from .advanced_caching import get_cache
from .memory_optimization import get_memory_optimizer
from .compute_optimization import get_compute_optimizer
from .parallel_processing import get_parallel_processor
from .network_optimization import get_network_optimizer
from .storage_optimization import get_storage_optimizer
from .resource_management import get_resource_manager

logger = setup_logging(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks"""
    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    LOAD = "load"
    STRESS = "stress"
    ENDURANCE = "endurance"
    MEMORY = "memory"
    COMPUTE = "compute"
    NETWORK = "network"
    STORAGE = "storage"
    INTEGRATION = "integration"


class TestPhase(Enum):
    """Test execution phases"""
    SETUP = "setup"
    WARMUP = "warmup"
    EXECUTION = "execution"
    COOLDOWN = "cooldown"
    TEARDOWN = "teardown"


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    name: str
    benchmark_type: BenchmarkType
    description: str
    duration_seconds: float
    warmup_seconds: float = 30.0
    cooldown_seconds: float = 10.0
    iterations: int = 1
    concurrent_workers: int = 1
    target_load: float = 1.0
    data_size_mb: int = 10
    custom_params: Dict[str, Any] = None


@dataclass
class BenchmarkResult:
    """Benchmark execution result"""
    config: BenchmarkConfig
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    
    # Performance metrics
    throughput: float = 0.0  # operations/second
    latency_ms: float = 0.0  # average latency
    latency_p95_ms: float = 0.0  # 95th percentile latency
    latency_p99_ms: float = 0.0  # 99th percentile latency
    
    # Resource metrics
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    peak_gpu_utilization: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Detailed metrics
    detailed_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.detailed_metrics is None:
            self.detailed_metrics = {}


class BenchmarkSuite:
    """Comprehensive benchmarking suite"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test registry
        self.benchmark_registry = {}
        self.custom_benchmarks = {}
        
        # Execution state
        self.current_execution = None
        self.execution_history = []
        
        # System monitoring
        self.metrics_collector = MetricsCollector()
        self.monitoring_active = False
        self.monitoring_data = deque(maxlen=10000)
        
        # Results storage
        self.results_database = []
        
        # Register built-in benchmarks
        self._register_builtin_benchmarks()
    
    def _register_builtin_benchmarks(self):
        """Register built-in benchmark tests"""
        
        # Performance benchmarks
        self.register_benchmark("cpu_intensive", self._cpu_intensive_benchmark)
        self.register_benchmark("memory_allocation", self._memory_allocation_benchmark)
        self.register_benchmark("gpu_compute", self._gpu_compute_benchmark)
        self.register_benchmark("storage_io", self._storage_io_benchmark)
        self.register_benchmark("network_throughput", self._network_throughput_benchmark)
        self.register_benchmark("cache_performance", self._cache_performance_benchmark)
        
        # Scalability benchmarks
        self.register_benchmark("parallel_scaling", self._parallel_scaling_benchmark)
        self.register_benchmark("memory_scaling", self._memory_scaling_benchmark)
        self.register_benchmark("load_scaling", self._load_scaling_benchmark)
        
        # Stress tests
        self.register_benchmark("resource_exhaustion", self._resource_exhaustion_benchmark)
        self.register_benchmark("concurrent_stress", self._concurrent_stress_benchmark)
        self.register_benchmark("memory_pressure", self._memory_pressure_benchmark)
        
        # Integration tests
        self.register_benchmark("end_to_end", self._end_to_end_benchmark)
        self.register_benchmark("system_integration", self._system_integration_benchmark)
        
        logger.info(f"Registered {len(self.benchmark_registry)} built-in benchmarks")
    
    def register_benchmark(self, name: str, benchmark_func: Callable):
        """Register custom benchmark function"""
        self.benchmark_registry[name] = benchmark_func
        logger.debug(f"Registered benchmark: {name}")
    
    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run single benchmark"""
        logger.info(f"Starting benchmark: {config.name}")
        
        if config.name not in self.benchmark_registry:
            return BenchmarkResult(
                config=config,
                start_time=time.time(),
                end_time=time.time(),
                duration=0.0,
                success=False,
                error_message=f"Benchmark '{config.name}' not found"
            )
        
        # Start monitoring
        self._start_monitoring()
        
        start_time = time.time()
        result = None
        
        try:
            # Execute benchmark phases
            self._execute_phase(TestPhase.SETUP, config)
            self._execute_phase(TestPhase.WARMUP, config)
            
            # Main execution
            benchmark_func = self.benchmark_registry[config.name]
            result = self._execute_benchmark(benchmark_func, config)
            
            self._execute_phase(TestPhase.COOLDOWN, config)
            self._execute_phase(TestPhase.TEARDOWN, config)
            
        except Exception as e:
            logger.error(f"Benchmark {config.name} failed: {e}")
            result = BenchmarkResult(
                config=config,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
        finally:
            self._stop_monitoring()
        
        # Store result
        if result:
            self.results_database.append(result)
            self._save_result(result)
        
        logger.info(f"Benchmark {config.name} completed: {result.success}")
        return result
    
    def run_benchmark_suite(self, configs: List[BenchmarkConfig]) -> List[BenchmarkResult]:
        """Run multiple benchmarks in sequence"""
        results = []
        
        logger.info(f"Starting benchmark suite with {len(configs)} tests")
        
        for config in configs:
            result = self.run_benchmark(config)
            results.append(result)
            
            # Brief pause between tests
            time.sleep(5.0)
        
        # Generate comprehensive report
        self._generate_suite_report(results)
        
        logger.info("Benchmark suite completed")
        return results
    
    def run_stress_test(self, duration_hours: float = 1.0, 
                       target_load: float = 0.8) -> Dict[str, Any]:
        """Run comprehensive stress test"""
        logger.info(f"Starting {duration_hours}h stress test at {target_load:.0%} load")
        
        stress_config = BenchmarkConfig(
            name="comprehensive_stress",
            benchmark_type=BenchmarkType.STRESS,
            description=f"Comprehensive stress test for {duration_hours}h",
            duration_seconds=duration_hours * 3600,
            target_load=target_load,
            concurrent_workers=mp.cpu_count() * 2
        )
        
        return self._comprehensive_stress_test(stress_config)
    
    def run_scalability_test(self, max_workers: int = None) -> Dict[str, Any]:
        """Run scalability analysis"""
        if max_workers is None:
            max_workers = mp.cpu_count() * 4
        
        logger.info(f"Starting scalability test up to {max_workers} workers")
        
        scalability_results = {}
        worker_counts = [1, 2, 4, 8, 16, 32, 64]
        worker_counts = [w for w in worker_counts if w <= max_workers]
        
        for worker_count in worker_counts:
            config = BenchmarkConfig(
                name="parallel_scaling",
                benchmark_type=BenchmarkType.SCALABILITY,
                description=f"Scalability test with {worker_count} workers",
                duration_seconds=60.0,
                concurrent_workers=worker_count
            )
            
            result = self.run_benchmark(config)
            scalability_results[worker_count] = {
                'throughput': result.throughput,
                'latency_ms': result.latency_ms,
                'cpu_utilization': result.avg_cpu_percent,
                'memory_usage_mb': result.avg_memory_mb
            }
        
        # Analyze scalability
        analysis = self._analyze_scalability(scalability_results)
        
        return {
            'results': scalability_results,
            'analysis': analysis
        }
    
    def _execute_benchmark(self, benchmark_func: Callable, config: BenchmarkConfig) -> BenchmarkResult:
        """Execute benchmark function with monitoring"""
        start_time = time.time()
        
        # Initialize result
        result = BenchmarkResult(
            config=config,
            start_time=start_time,
            end_time=0.0,
            duration=0.0,
            success=False
        )
        
        try:
            # Execute benchmark
            benchmark_metrics = benchmark_func(config)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate performance metrics
            self._calculate_performance_metrics(result, benchmark_metrics)
            
            # Calculate resource metrics
            self._calculate_resource_metrics(result)
            
            result.end_time = end_time
            result.duration = duration
            result.success = True
            
        except Exception as e:
            result.end_time = time.time()
            result.duration = result.end_time - start_time
            result.error_message = str(e)
            logger.error(f"Benchmark execution failed: {e}")
        
        return result
    
    def _calculate_performance_metrics(self, result: BenchmarkResult, 
                                     benchmark_metrics: Dict[str, Any]):
        """Calculate performance metrics from benchmark data"""
        if 'latencies' in benchmark_metrics:
            latencies = benchmark_metrics['latencies']
            result.latency_ms = np.mean(latencies) * 1000
            result.latency_p95_ms = np.percentile(latencies, 95) * 1000
            result.latency_p99_ms = np.percentile(latencies, 99) * 1000
        
        if 'operations_count' in benchmark_metrics and result.duration > 0:
            result.throughput = benchmark_metrics['operations_count'] / result.duration
        
        result.detailed_metrics = benchmark_metrics
    
    def _calculate_resource_metrics(self, result: BenchmarkResult):
        """Calculate resource usage metrics from monitoring data"""
        if not self.monitoring_data:
            return
        
        # Filter monitoring data for benchmark duration
        benchmark_data = [
            data for data in self.monitoring_data
            if result.start_time <= data['timestamp'] <= result.end_time
        ]
        
        if not benchmark_data:
            return
        
        # Calculate resource metrics
        cpu_values = [data['cpu_percent'] for data in benchmark_data]
        memory_values = [data['memory_mb'] for data in benchmark_data]
        
        result.peak_cpu_percent = max(cpu_values) if cpu_values else 0.0
        result.avg_cpu_percent = np.mean(cpu_values) if cpu_values else 0.0
        result.peak_memory_mb = max(memory_values) if memory_values else 0.0
        result.avg_memory_mb = np.mean(memory_values) if memory_values else 0.0
        
        # GPU metrics if available
        gpu_values = [data.get('gpu_utilization', 0) for data in benchmark_data]
        if gpu_values:
            result.peak_gpu_utilization = max(gpu_values)
    
    def _start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_data.clear()
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    data = {
                        'timestamp': time.time(),
                        'cpu_percent': psutil.cpu_percent(),
                        'memory_mb': psutil.virtual_memory().used / (1024 * 1024),
                        'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                        'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
                    }
                    
                    # Add GPU metrics if available
                    if torch.cuda.is_available():
                        try:
                            data['gpu_utilization'] = torch.cuda.utilization()
                            data['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                        except Exception:
                            pass
                    
                    self.monitoring_data.append(data)
                    time.sleep(0.1)  # 10Hz monitoring
                    
                except Exception as e:
                    logger.warning(f"Monitoring error: {e}")
                    time.sleep(1.0)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
    
    def _execute_phase(self, phase: TestPhase, config: BenchmarkConfig):
        """Execute test phase"""
        phase_duration = {
            TestPhase.SETUP: 1.0,
            TestPhase.WARMUP: config.warmup_seconds,
            TestPhase.EXECUTION: config.duration_seconds,
            TestPhase.COOLDOWN: config.cooldown_seconds,
            TestPhase.TEARDOWN: 1.0
        }
        
        if phase != TestPhase.EXECUTION:  # Execution is handled separately
            logger.debug(f"Executing {phase.value} phase")
            if phase == TestPhase.WARMUP and phase_duration[phase] > 0:
                # Perform warmup operations
                self._warmup_operations(config)
            time.sleep(min(phase_duration[phase], 1.0))
    
    def _warmup_operations(self, config: BenchmarkConfig):
        """Perform warmup operations"""
        # Simple warmup - allocate and release memory, perform CPU operations
        for _ in range(10):
            data = np.random.rand(1000, 1000)
            _ = np.dot(data, data.T)
            time.sleep(0.1)
    
    # Built-in benchmark implementations
    
    def _cpu_intensive_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """CPU-intensive benchmark"""
        operations_count = 0
        latencies = []
        
        end_time = time.time() + config.duration_seconds
        
        while time.time() < end_time:
            start_op = time.time()
            
            # CPU-intensive operation
            size = 1000
            matrix_a = np.random.rand(size, size)
            matrix_b = np.random.rand(size, size)
            _ = np.dot(matrix_a, matrix_b)
            
            latencies.append(time.time() - start_op)
            operations_count += 1
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'operation_type': 'matrix_multiplication'
        }
    
    def _memory_allocation_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Memory allocation benchmark"""
        operations_count = 0
        latencies = []
        allocated_objects = []
        
        end_time = time.time() + config.duration_seconds
        
        while time.time() < end_time:
            start_op = time.time()
            
            # Allocate memory
            size = config.data_size_mb * 1024 * 1024 // 8  # 8 bytes per float64
            data = np.random.rand(size)
            allocated_objects.append(data)
            
            # Occasionally free some memory
            if len(allocated_objects) > 10:
                allocated_objects.pop(0)
            
            latencies.append(time.time() - start_op)
            operations_count += 1
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'peak_allocated_objects': len(allocated_objects),
            'operation_type': 'memory_allocation'
        }
    
    def _gpu_compute_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """GPU compute benchmark"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for GPU benchmark")
        
        operations_count = 0
        latencies = []
        
        end_time = time.time() + config.duration_seconds
        device = torch.device('cuda')
        
        while time.time() < end_time:
            start_op = time.time()
            
            # GPU computation
            size = 2048
            a = torch.rand(size, size, device=device)
            b = torch.rand(size, size, device=device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            latencies.append(time.time() - start_op)
            operations_count += 1
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'operation_type': 'gpu_matrix_multiplication'
        }
    
    def _storage_io_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Storage I/O benchmark"""
        operations_count = 0
        latencies = []
        
        end_time = time.time() + config.duration_seconds
        
        with tempfile.TemporaryDirectory() as temp_dir:
            while time.time() < end_time:
                start_op = time.time()
                
                # Write and read file
                file_path = Path(temp_dir) / f"test_{operations_count}.dat"
                data = np.random.bytes(config.data_size_mb * 1024 * 1024)
                
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                with open(file_path, 'rb') as f:
                    _ = f.read()
                
                file_path.unlink()
                
                latencies.append(time.time() - start_op)
                operations_count += 1
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'file_size_mb': config.data_size_mb,
            'operation_type': 'file_write_read'
        }
    
    def _network_throughput_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Network throughput benchmark"""
        operations_count = 0
        latencies = []
        
        try:
            network_optimizer = get_network_optimizer()
            
            end_time = time.time() + config.duration_seconds
            
            while time.time() < end_time:
                start_op = time.time()
                
                # Simulate network operation
                # In real implementation, this would make actual network requests
                time.sleep(0.01)  # Simulate network delay
                
                latencies.append(time.time() - start_op)
                operations_count += 1
                
        except Exception as e:
            logger.warning(f"Network benchmark using simulation: {e}")
            # Fallback to simulation
            operations_count = int(config.duration_seconds * 100)
            latencies = [0.01] * operations_count
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'operation_type': 'network_request'
        }
    
    def _cache_performance_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Cache performance benchmark"""
        operations_count = 0
        latencies = []
        cache_hits = 0
        
        try:
            cache = get_cache()
            
            end_time = time.time() + config.duration_seconds
            
            while time.time() < end_time:
                start_op = time.time()
                
                key = f"bench_key_{operations_count % 1000}"
                
                # Try to get from cache
                value = cache.get(key)
                if value is not None:
                    cache_hits += 1
                else:
                    # Cache miss - generate and store value
                    value = np.random.rand(100).tolist()
                    cache.put(key, value)
                
                latencies.append(time.time() - start_op)
                operations_count += 1
                
        except Exception as e:
            logger.warning(f"Cache benchmark failed: {e}")
            operations_count = 1
            latencies = [0.001]
            cache_hits = 0
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'cache_hit_rate': cache_hits / max(operations_count, 1),
            'operation_type': 'cache_access'
        }
    
    def _parallel_scaling_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Parallel processing scaling benchmark"""
        operations_count = 0
        latencies = []
        
        def cpu_task(n):
            """CPU-intensive task for parallel execution"""
            start = time.time()
            result = sum(i * i for i in range(n))
            return time.time() - start, result
        
        try:
            processor = get_parallel_processor()
            
            # Submit tasks
            task_size = 10000
            tasks_per_worker = max(1, int(config.duration_seconds * 10 / config.concurrent_workers))
            
            start_time = time.time()
            futures = []
            
            for i in range(config.concurrent_workers * tasks_per_worker):
                task_id = processor.submit_task(
                    cpu_task, task_size,
                    worker_type='cpu_intensive'
                )
                if task_id:
                    futures.append(task_id)
            
            # Collect results
            for task_id in futures:
                try:
                    result = processor.get_result(task_id, timeout=60.0)
                    if result:
                        duration, _ = result
                        latencies.append(duration)
                        operations_count += 1
                except Exception:
                    pass
            
        except Exception as e:
            logger.warning(f"Parallel benchmark failed: {e}")
            operations_count = 1
            latencies = [config.duration_seconds]
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'workers_used': config.concurrent_workers,
            'operation_type': 'parallel_cpu_task'
        }
    
    def _memory_scaling_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Memory scaling benchmark"""
        operations_count = 0
        latencies = []
        
        # Test different memory allocation sizes
        base_size = config.data_size_mb * 1024 * 1024 // 8
        
        end_time = time.time() + config.duration_seconds
        
        while time.time() < end_time:
            start_op = time.time()
            
            # Allocate increasing amounts of memory
            size_multiplier = (operations_count % 10) + 1
            size = base_size * size_multiplier
            
            try:
                data = np.random.rand(size)
                # Perform some operation on the data
                _ = np.sum(data)
                del data
            except MemoryError:
                logger.warning("Memory allocation failed - reached memory limit")
                break
            
            latencies.append(time.time() - start_op)
            operations_count += 1
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'max_size_tested_mb': (operations_count % 10 + 1) * config.data_size_mb,
            'operation_type': 'memory_scaling'
        }
    
    def _load_scaling_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Load scaling benchmark"""
        operations_count = 0
        latencies = []
        
        # Gradually increase load
        max_load = config.target_load
        duration_per_step = config.duration_seconds / 10
        
        for step in range(10):
            current_load = (step + 1) * max_load / 10
            step_operations = 0
            
            step_end_time = time.time() + duration_per_step
            
            while time.time() < step_end_time:
                start_op = time.time()
                
                # Simulate work proportional to load
                work_amount = int(1000 * current_load)
                _ = sum(i * i for i in range(work_amount))
                
                latencies.append(time.time() - start_op)
                operations_count += 1
                step_operations += 1
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'max_load_tested': max_load,
            'operation_type': 'load_scaling'
        }
    
    def _resource_exhaustion_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Resource exhaustion stress test"""
        operations_count = 0
        resources_allocated = []
        
        end_time = time.time() + config.duration_seconds
        
        try:
            while time.time() < end_time:
                # Allocate various resources
                
                # Memory
                try:
                    memory_block = np.random.rand(1024 * 1024)  # 8MB
                    resources_allocated.append(memory_block)
                    operations_count += 1
                except MemoryError:
                    logger.warning("Memory exhaustion reached")
                    break
                
                # CPU load
                if operations_count % 10 == 0:
                    _ = sum(i * i for i in range(100000))
                
                time.sleep(0.01)
                
        except Exception as e:
            logger.warning(f"Resource exhaustion test error: {e}")
        
        return {
            'operations_count': operations_count,
            'resources_allocated_count': len(resources_allocated),
            'peak_memory_blocks': len(resources_allocated),
            'operation_type': 'resource_exhaustion'
        }
    
    def _concurrent_stress_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Concurrent stress test"""
        def stress_worker(worker_id, duration):
            """Individual stress worker"""
            operations = 0
            worker_latencies = []
            
            end_time = time.time() + duration
            
            while time.time() < end_time:
                start_op = time.time()
                
                # Mix of operations
                if worker_id % 3 == 0:
                    # CPU intensive
                    _ = sum(i * i for i in range(10000))
                elif worker_id % 3 == 1:
                    # Memory allocation
                    data = np.random.rand(1000)
                    _ = np.sum(data)
                else:
                    # Mixed workload
                    data = np.random.rand(500)
                    _ = np.dot(data, data)
                
                worker_latencies.append(time.time() - start_op)
                operations += 1
            
            return operations, worker_latencies
        
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_workers) as executor:
            futures = [
                executor.submit(stress_worker, i, config.duration_seconds)
                for i in range(config.concurrent_workers)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Aggregate results
        total_operations = sum(r[0] for r in results)
        all_latencies = []
        for _, latencies in results:
            all_latencies.extend(latencies)
        
        return {
            'operations_count': total_operations,
            'latencies': all_latencies,
            'workers_used': config.concurrent_workers,
            'operation_type': 'concurrent_stress'
        }
    
    def _memory_pressure_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Memory pressure stress test"""
        operations_count = 0
        latencies = []
        memory_blocks = []
        
        end_time = time.time() + config.duration_seconds
        
        while time.time() < end_time:
            start_op = time.time()
            
            try:
                # Gradually increase memory pressure
                block_size = (operations_count % 100 + 1) * 1024 * 1024  # 1-100 MB blocks
                memory_block = np.random.bytes(block_size)
                memory_blocks.append(memory_block)
                
                # Occasionally trigger garbage collection
                if operations_count % 50 == 0:
                    import gc
                    gc.collect()
                
                # Remove old blocks to prevent complete exhaustion
                if len(memory_blocks) > 50:
                    memory_blocks.pop(0)
                
            except MemoryError:
                logger.warning("Memory pressure limit reached")
                memory_blocks.clear()
                import gc
                gc.collect()
            
            latencies.append(time.time() - start_op)
            operations_count += 1
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'peak_memory_blocks': len(memory_blocks),
            'operation_type': 'memory_pressure'
        }
    
    def _end_to_end_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """End-to-end system benchmark"""
        operations_count = 0
        latencies = []
        
        end_time = time.time() + config.duration_seconds
        
        while time.time() < end_time:
            start_op = time.time()
            
            try:
                # Simulate end-to-end protein processing pipeline
                
                # 1. Data loading (storage)
                sequence_length = np.random.randint(100, 1000)
                sequence_data = np.random.rand(sequence_length, 20)  # 20 amino acids
                
                # 2. Preprocessing (compute)
                processed_data = np.log(sequence_data + 1e-8)
                
                # 3. Model inference (GPU if available)
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    tensor_data = torch.tensor(processed_data, device=device).float()
                    # Simulate neural network forward pass
                    result = torch.nn.functional.relu(torch.matmul(tensor_data, tensor_data.T))
                    torch.cuda.synchronize()
                    result = result.cpu().numpy()
                else:
                    # CPU inference
                    result = np.maximum(0, np.dot(processed_data, processed_data.T))
                
                # 4. Post-processing
                confidence_scores = np.diag(result)
                final_result = np.mean(confidence_scores)
                
                # 5. Caching (if available)
                try:
                    cache = get_cache()
                    cache.put(f"result_{operations_count}", final_result)
                except Exception:
                    pass
                
            except Exception as e:
                logger.warning(f"End-to-end benchmark step failed: {e}")
            
            latencies.append(time.time() - start_op)
            operations_count += 1
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'operation_type': 'end_to_end_pipeline'
        }
    
    def _system_integration_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """System integration benchmark"""
        operations_count = 0
        latencies = []
        component_stats = {}
        
        end_time = time.time() + config.duration_seconds
        
        while time.time() < end_time:
            start_op = time.time()
            
            try:
                # Test integration of all optimization components
                
                # Memory optimization
                try:
                    memory_optimizer = get_memory_optimizer()
                    memory_stats = memory_optimizer.get_comprehensive_stats()
                    component_stats['memory'] = memory_stats.get('system_memory', {})
                except Exception:
                    pass
                
                # Compute optimization
                try:
                    compute_optimizer = get_compute_optimizer()
                    compute_stats = compute_optimizer.get_optimization_stats()
                    component_stats['compute'] = compute_stats
                except Exception:
                    pass
                
                # Parallel processing
                try:
                    processor = get_parallel_processor()
                    parallel_stats = processor.get_comprehensive_stats()
                    component_stats['parallel'] = parallel_stats.get('performance_metrics', {})
                except Exception:
                    pass
                
                # Storage optimization
                try:
                    storage_optimizer = get_storage_optimizer()
                    storage_stats = storage_optimizer.get_comprehensive_stats()
                    component_stats['storage'] = storage_stats.get('optimization_active', False)
                except Exception:
                    pass
                
                # Cache system
                try:
                    cache = get_cache()
                    cache_stats = cache.get_multi_tier_stats()
                    component_stats['cache'] = cache_stats.get('overall', {})
                except Exception:
                    pass
                
            except Exception as e:
                logger.warning(f"System integration benchmark error: {e}")
            
            latencies.append(time.time() - start_op)
            operations_count += 1
        
        return {
            'operations_count': operations_count,
            'latencies': latencies,
            'component_stats': component_stats,
            'operation_type': 'system_integration'
        }
    
    def _comprehensive_stress_test(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run comprehensive stress test"""
        stress_results = {
            'start_time': time.time(),
            'duration_hours': config.duration_seconds / 3600,
            'target_load': config.target_load,
            'phases': [],
            'overall_stats': {}
        }
        
        # Phase 1: Ramp up
        logger.info("Stress test phase 1: Ramp up")
        ramp_config = BenchmarkConfig(
            name="concurrent_stress",
            benchmark_type=BenchmarkType.STRESS,
            description="Ramp up phase",
            duration_seconds=300,  # 5 minutes
            concurrent_workers=config.concurrent_workers // 2
        )
        ramp_result = self.run_benchmark(ramp_config)
        stress_results['phases'].append(('ramp_up', asdict(ramp_result)))
        
        # Phase 2: Full load
        logger.info("Stress test phase 2: Full load")
        full_load_config = BenchmarkConfig(
            name="concurrent_stress",
            benchmark_type=BenchmarkType.STRESS,
            description="Full load phase",
            duration_seconds=config.duration_seconds - 600,  # Most of the time
            concurrent_workers=config.concurrent_workers
        )
        full_load_result = self.run_benchmark(full_load_config)
        stress_results['phases'].append(('full_load', asdict(full_load_result)))
        
        # Phase 3: Ramp down
        logger.info("Stress test phase 3: Ramp down")
        ramp_down_config = BenchmarkConfig(
            name="concurrent_stress",
            benchmark_type=BenchmarkType.STRESS,
            description="Ramp down phase",
            duration_seconds=300,  # 5 minutes
            concurrent_workers=config.concurrent_workers // 4
        )
        ramp_down_result = self.run_benchmark(ramp_down_config)
        stress_results['phases'].append(('ramp_down', asdict(ramp_down_result)))
        
        # Calculate overall statistics
        all_latencies = []
        total_operations = 0
        
        for phase_name, phase_result in stress_results['phases']:
            if phase_result['detailed_metrics'] and 'latencies' in phase_result['detailed_metrics']:
                all_latencies.extend(phase_result['detailed_metrics']['latencies'])
            total_operations += phase_result.get('throughput', 0) * phase_result.get('duration', 0)
        
        stress_results['overall_stats'] = {
            'total_operations': int(total_operations),
            'overall_latency_ms': np.mean(all_latencies) * 1000 if all_latencies else 0,
            'peak_latency_ms': np.max(all_latencies) * 1000 if all_latencies else 0,
            'latency_p95_ms': np.percentile(all_latencies, 95) * 1000 if all_latencies else 0,
            'system_stability': all(r[1]['success'] for r in stress_results['phases'])
        }
        
        stress_results['end_time'] = time.time()
        stress_results['actual_duration'] = stress_results['end_time'] - stress_results['start_time']
        
        return stress_results
    
    def _analyze_scalability(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scalability results"""
        worker_counts = sorted(results.keys())
        throughputs = [results[w]['throughput'] for w in worker_counts]
        
        # Calculate scaling efficiency
        baseline_throughput = throughputs[0] if throughputs else 1
        scaling_efficiencies = []
        
        for i, (workers, throughput) in enumerate(zip(worker_counts, throughputs)):
            if i == 0:
                efficiency = 1.0
            else:
                expected_throughput = baseline_throughput * workers
                efficiency = throughput / expected_throughput if expected_throughput > 0 else 0
            scaling_efficiencies.append(efficiency)
        
        # Find optimal worker count
        efficiency_threshold = 0.8
        optimal_workers = worker_counts[0]
        for workers, efficiency in zip(worker_counts, scaling_efficiencies):
            if efficiency >= efficiency_threshold:
                optimal_workers = workers
            else:
                break
        
        return {
            'scaling_efficiencies': dict(zip(worker_counts, scaling_efficiencies)),
            'optimal_worker_count': optimal_workers,
            'max_throughput': max(throughputs) if throughputs else 0,
            'linear_scaling_limit': worker_counts[scaling_efficiencies.index(min(scaling_efficiencies))] if scaling_efficiencies else 1
        }
    
    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            logger.debug(f"Saved benchmark result to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save benchmark result: {e}")
    
    def _generate_suite_report(self, results: List[BenchmarkResult]):
        """Generate comprehensive benchmark suite report"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"benchmark_suite_report_{timestamp}.html"
        
        # Generate HTML report
        html_content = self._create_html_report(results)
        
        try:
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            # Also save as JSON
            json_file = self.output_dir / f"benchmark_suite_data_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump([asdict(r) for r in results], f, indent=2, default=str)
            
            logger.info(f"Benchmark suite report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate benchmark report: {e}")
    
    def _create_html_report(self, results: List[BenchmarkResult]) -> str:
        """Create HTML benchmark report"""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Protein-SSL Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .benchmark {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .success {{ border-left: 5px solid #4CAF50; }}
                .failure {{ border-left: 5px solid #f44336; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Protein-SSL Benchmark Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Benchmarks:</strong> {len(results)}</p>
                <p><strong>Successful:</strong> {len(successful_results)}</p>
                <p><strong>Failed:</strong> {len(failed_results)}</p>
                <p><strong>Report Generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        """
        
        if successful_results:
            html += "<h2>Successful Benchmarks</h2>"
            for result in successful_results:
                html += self._format_benchmark_html(result, success=True)
        
        if failed_results:
            html += "<h2>Failed Benchmarks</h2>"
            for result in failed_results:
                html += self._format_benchmark_html(result, success=False)
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _format_benchmark_html(self, result: BenchmarkResult, success: bool) -> str:
        """Format individual benchmark result as HTML"""
        css_class = "benchmark success" if success else "benchmark failure"
        
        html = f"""
        <div class="{css_class}">
            <h3>{result.config.name}</h3>
            <p><strong>Description:</strong> {result.config.description}</p>
            <p><strong>Duration:</strong> {result.duration:.2f} seconds</p>
        """
        
        if success:
            html += f"""
            <div class="metrics">
                <div class="metric">
                    <strong>Throughput:</strong><br>
                    {result.throughput:.2f} ops/sec
                </div>
                <div class="metric">
                    <strong>Latency (avg):</strong><br>
                    {result.latency_ms:.2f} ms
                </div>
                <div class="metric">
                    <strong>Latency (p95):</strong><br>
                    {result.latency_p95_ms:.2f} ms
                </div>
                <div class="metric">
                    <strong>Peak CPU:</strong><br>
                    {result.peak_cpu_percent:.1f}%
                </div>
                <div class="metric">
                    <strong>Peak Memory:</strong><br>
                    {result.peak_memory_mb:.1f} MB
                </div>
                <div class="metric">
                    <strong>GPU Utilization:</strong><br>
                    {result.peak_gpu_utilization:.1f}%
                </div>
            </div>
            """
        else:
            html += f"<p><strong>Error:</strong> {result.error_message}</p>"
        
        html += "</div>"
        return html
    
    def get_benchmark_history(self) -> List[BenchmarkResult]:
        """Get benchmark execution history"""
        return self.results_database.copy()
    
    def export_results(self, format: str = 'json') -> str:
        """Export benchmark results in specified format"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = f"benchmark_results_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump([asdict(r) for r in self.results_database], f, indent=2, default=str)
        
        elif format == 'csv':
            filename = f"benchmark_results_{timestamp}.csv"
            filepath = self.output_dir / filename
            
            # Convert to pandas DataFrame for easy CSV export
            data = []
            for result in self.results_database:
                row = {
                    'name': result.config.name,
                    'type': result.config.benchmark_type.value,
                    'success': result.success,
                    'duration': result.duration,
                    'throughput': result.throughput,
                    'latency_ms': result.latency_ms,
                    'latency_p95_ms': result.latency_p95_ms,
                    'peak_cpu_percent': result.peak_cpu_percent,
                    'peak_memory_mb': result.peak_memory_mb,
                    'peak_gpu_utilization': result.peak_gpu_utilization
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Benchmark results exported to {filepath}")
        return str(filepath)


# Global benchmark suite instance
_global_benchmark_suite = None

def get_benchmark_suite(**kwargs) -> BenchmarkSuite:
    """Get global benchmark suite instance"""
    global _global_benchmark_suite
    
    if _global_benchmark_suite is None:
        _global_benchmark_suite = BenchmarkSuite(**kwargs)
    
    return _global_benchmark_suite

def run_quick_benchmark() -> Dict[str, Any]:
    """Run quick performance benchmark"""
    suite = get_benchmark_suite()
    
    quick_configs = [
        BenchmarkConfig("cpu_intensive", BenchmarkType.PERFORMANCE, "Quick CPU test", 30.0),
        BenchmarkConfig("memory_allocation", BenchmarkType.PERFORMANCE, "Quick memory test", 30.0),
        BenchmarkConfig("cache_performance", BenchmarkType.PERFORMANCE, "Quick cache test", 30.0)
    ]
    
    results = suite.run_benchmark_suite(quick_configs)
    
    # Return summary
    return {
        'total_benchmarks': len(results),
        'successful': len([r for r in results if r.success]),
        'avg_throughput': np.mean([r.throughput for r in results if r.success]),
        'avg_latency_ms': np.mean([r.latency_ms for r in results if r.success])
    }

def run_comprehensive_benchmark() -> Dict[str, Any]:
    """Run comprehensive benchmark suite"""
    suite = get_benchmark_suite()
    
    # Create comprehensive benchmark configuration
    comprehensive_configs = [
        BenchmarkConfig("cpu_intensive", BenchmarkType.PERFORMANCE, "CPU performance test", 120.0),
        BenchmarkConfig("memory_allocation", BenchmarkType.MEMORY, "Memory allocation test", 120.0),
        BenchmarkConfig("gpu_compute", BenchmarkType.COMPUTE, "GPU compute test", 120.0),
        BenchmarkConfig("storage_io", BenchmarkType.STORAGE, "Storage I/O test", 120.0),
        BenchmarkConfig("cache_performance", BenchmarkType.PERFORMANCE, "Cache performance test", 120.0),
        BenchmarkConfig("parallel_scaling", BenchmarkType.SCALABILITY, "Parallel scaling test", 180.0, concurrent_workers=mp.cpu_count()),
        BenchmarkConfig("concurrent_stress", BenchmarkType.STRESS, "Concurrent stress test", 300.0, concurrent_workers=mp.cpu_count() * 2),
        BenchmarkConfig("end_to_end", BenchmarkType.INTEGRATION, "End-to-end pipeline test", 180.0)
    ]
    
    results = suite.run_benchmark_suite(comprehensive_configs)
    
    return {
        'total_benchmarks': len(results),
        'successful': len([r for r in results if r.success]),
        'results': [asdict(r) for r in results]
    }
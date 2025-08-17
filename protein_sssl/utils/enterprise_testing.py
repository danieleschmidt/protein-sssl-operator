"""
Enterprise Testing Framework for protein-sssl-operator
Provides property-based testing, stress testing, fuzzing,
and comprehensive test automation capabilities.
"""

import time
import random
import string
import threading
import multiprocessing
import logging
import json
import statistics
import traceback
import gc
import os
import psutil
from typing import (
    Dict, List, Optional, Any, Union, Callable, 
    TypeVar, Generic, Protocol, Tuple, Iterator
)
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import functools
import inspect
import weakref
import uuid
import hashlib

try:
    import hypothesis
    from hypothesis import strategies as st, given, settings, Verbosity
    from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, invariant
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PROPERTY = "property"
    STRESS = "stress"
    FUZZ = "fuzz"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"
    SMOKE = "smoke"
    ACCEPTANCE = "acceptance"

class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"

class TestPriority(Enum):
    """Test priority levels"""
    LOW = ("low", 1)
    MEDIUM = ("medium", 2)
    HIGH = ("high", 3)
    CRITICAL = ("critical", 4)
    
    def __init__(self, name: str, level: int):
        self.level_name = name
        self.level = level
    
    def __ge__(self, other):
        return self.level >= other.level

@dataclass
class TestCase:
    """Individual test case definition"""
    id: str
    name: str
    description: str
    test_type: TestType
    priority: TestPriority
    function: Callable
    timeout: float = 60.0
    retries: int = 0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_exceptions: List[Type[Exception]] = field(default_factory=list)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"{self.test_type.value}_{self.name}_{uuid.uuid4().hex[:8]}"

@dataclass
class TestExecution:
    """Test execution result"""
    test_case: TestCase
    result: TestResult
    start_time: float
    end_time: float
    duration: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    output: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_id': self.test_case.id,
            'test_name': self.test_case.name,
            'test_type': self.test_case.test_type.value,
            'priority': self.test_case.priority.level_name,
            'result': self.result.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'output': self.output,
            'metrics': self.metrics,
            'retry_count': self.retry_count
        }

@dataclass
class TestSuite:
    """Collection of related test cases"""
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    parallel_execution: bool = False
    max_workers: int = 4
    
    def add_test_case(self, test_case: TestCase):
        """Add test case to suite"""
        self.test_cases.append(test_case)
    
    def remove_test_case(self, test_id: str) -> bool:
        """Remove test case from suite"""
        for i, test_case in enumerate(self.test_cases):
            if test_case.id == test_id:
                del self.test_cases[i]
                return True
        return False
    
    def get_tests_by_type(self, test_type: TestType) -> List[TestCase]:
        """Get tests of specific type"""
        return [tc for tc in self.test_cases if tc.test_type == test_type]
    
    def get_tests_by_priority(self, min_priority: TestPriority) -> List[TestCase]:
        """Get tests with minimum priority"""
        return [tc for tc in self.test_cases if tc.priority >= min_priority]

class PropertyTestGenerator:
    """Property-based test generator using Hypothesis"""
    
    def __init__(self):
        self.hypothesis_available = HYPOTHESIS_AVAILABLE
        if not self.hypothesis_available:
            logger.warning("Hypothesis not available, property-based testing disabled")
    
    def protein_sequence_strategy(self, 
                                min_length: int = 1, 
                                max_length: int = 1000,
                                amino_acids: str = "ACDEFGHIKLMNPQRSTVWY") -> Any:
        """Generate protein sequence strategy"""
        if not self.hypothesis_available:
            return None
        
        return st.text(
            alphabet=amino_acids,
            min_size=min_length,
            max_size=max_length
        )
    
    def model_config_strategy(self) -> Any:
        """Generate model configuration strategy"""
        if not self.hypothesis_available:
            return None
        
        return st.fixed_dictionaries({
            'd_model': st.integers(min_value=64, max_value=2048).filter(lambda x: x % 8 == 0),
            'n_layers': st.integers(min_value=1, max_value=24),
            'n_heads': st.integers(min_value=1, max_value=32),
            'max_length': st.integers(min_value=128, max_value=4096),
            'dropout': st.floats(min_value=0.0, max_value=0.5),
            'learning_rate': st.floats(min_value=1e-6, max_value=1e-2)
        }).filter(lambda config: config['d_model'] % config['n_heads'] == 0)
    
    def coordinates_3d_strategy(self,
                              min_atoms: int = 1,
                              max_atoms: int = 1000,
                              coordinate_range: float = 100.0) -> Any:
        """Generate 3D coordinates strategy"""
        if not self.hypothesis_available or not NUMPY_AVAILABLE:
            return None
        
        return st.integers(min_value=min_atoms, max_value=max_atoms).flatmap(
            lambda n: st.lists(
                st.lists(
                    st.floats(
                        min_value=-coordinate_range,
                        max_value=coordinate_range,
                        allow_nan=False,
                        allow_infinity=False
                    ),
                    min_size=3,
                    max_size=3
                ),
                min_size=n,
                max_size=n
            )
        )
    
    def generate_property_test(self,
                             name: str,
                             strategy: Any,
                             property_function: Callable,
                             max_examples: int = 100,
                             timeout: float = 60.0) -> TestCase:
        """Generate property-based test case"""
        
        if not self.hypothesis_available:
            # Create a dummy test that passes
            def dummy_test():
                logger.warning(f"Property test '{name}' skipped - Hypothesis not available")
                return True
            
            return TestCase(
                id=f"property_{name}_{uuid.uuid4().hex[:8]}",
                name=name,
                description=f"Property test: {name} (disabled)",
                test_type=TestType.PROPERTY,
                priority=TestPriority.MEDIUM,
                function=dummy_test,
                timeout=timeout
            )
        
        @given(strategy)
        @settings(max_examples=max_examples, deadline=timeout * 1000)
        def property_test(data):
            return property_function(data)
        
        return TestCase(
            id=f"property_{name}_{uuid.uuid4().hex[:8]}",
            name=name,
            description=f"Property test: {name}",
            test_type=TestType.PROPERTY,
            priority=TestPriority.MEDIUM,
            function=property_test,
            timeout=timeout
        )

class StressTestGenerator:
    """Stress test generator for performance and reliability testing"""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_limit_gb = psutil.virtual_memory().total / (1024**3) * 0.8  # 80% of total memory
    
    def generate_load_test(self,
                          name: str,
                          target_function: Callable,
                          concurrent_users: int = 10,
                          duration_seconds: float = 60.0,
                          ramp_up_seconds: float = 10.0,
                          requests_per_second: Optional[float] = None) -> TestCase:
        """Generate load test"""
        
        def load_test():
            results = {'total_requests': 0, 'successful_requests': 0, 'failed_requests': 0,
                      'response_times': [], 'errors': []}
            
            def worker():
                worker_results = {'requests': 0, 'successes': 0, 'failures': 0, 'times': [], 'errors': []}
                start_time = time.time()
                
                while time.time() - start_time < duration_seconds:
                    request_start = time.time()
                    
                    try:
                        target_function()
                        worker_results['successes'] += 1
                        response_time = time.time() - request_start
                        worker_results['times'].append(response_time)
                    except Exception as e:
                        worker_results['failures'] += 1
                        worker_results['errors'].append(str(e))
                    
                    worker_results['requests'] += 1
                    
                    # Rate limiting
                    if requests_per_second:
                        time.sleep(max(0, 1.0 / requests_per_second - (time.time() - request_start)))
                
                return worker_results
            
            # Gradual ramp-up
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = []
                
                for i in range(concurrent_users):
                    # Stagger thread start times
                    if ramp_up_seconds > 0:
                        delay = (i / concurrent_users) * ramp_up_seconds
                        time.sleep(delay)
                    
                    future = executor.submit(worker)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        worker_result = future.result()
                        results['total_requests'] += worker_result['requests']
                        results['successful_requests'] += worker_result['successes']
                        results['failed_requests'] += worker_result['failures']
                        results['response_times'].extend(worker_result['times'])
                        results['errors'].extend(worker_result['errors'])
                    except Exception as e:
                        results['errors'].append(f"Worker error: {e}")
            
            # Calculate statistics
            if results['response_times']:
                results['avg_response_time'] = statistics.mean(results['response_times'])
                results['min_response_time'] = min(results['response_times'])
                results['max_response_time'] = max(results['response_times'])
                results['p95_response_time'] = statistics.quantiles(results['response_times'], n=20)[18]
                results['p99_response_time'] = statistics.quantiles(results['response_times'], n=100)[98]
            
            results['success_rate'] = (results['successful_requests'] / results['total_requests'] 
                                     if results['total_requests'] > 0 else 0)
            results['requests_per_second'] = results['total_requests'] / duration_seconds
            
            # Validate performance requirements
            assert results['success_rate'] >= 0.95, f"Success rate too low: {results['success_rate']:.2%}"
            
            if results['response_times']:
                assert results['p95_response_time'] < 5.0, f"P95 response time too high: {results['p95_response_time']:.2f}s"
            
            return results
        
        return TestCase(
            id=f"load_{name}_{uuid.uuid4().hex[:8]}",
            name=f"Load Test: {name}",
            description=f"Load test with {concurrent_users} concurrent users for {duration_seconds}s",
            test_type=TestType.STRESS,
            priority=TestPriority.HIGH,
            function=load_test,
            timeout=duration_seconds + ramp_up_seconds + 30.0  # Extra buffer
        )
    
    def generate_memory_stress_test(self,
                                  name: str,
                                  target_function: Callable,
                                  max_memory_gb: float = None) -> TestCase:
        """Generate memory stress test"""
        
        max_memory_gb = max_memory_gb or self.memory_limit_gb
        
        def memory_stress_test():
            initial_memory = psutil.Process().memory_info().rss / (1024**3)
            max_observed_memory = initial_memory
            
            # Run function multiple times and monitor memory
            iterations = 100
            for i in range(iterations):
                target_function()
                
                current_memory = psutil.Process().memory_info().rss / (1024**3)
                max_observed_memory = max(max_observed_memory, current_memory)
                
                # Check memory limit
                if current_memory > max_memory_gb:
                    raise MemoryError(f"Memory usage exceeded limit: {current_memory:.2f}GB > {max_memory_gb:.2f}GB")
                
                # Force garbage collection periodically
                if i % 10 == 0:
                    gc.collect()
            
            memory_growth = max_observed_memory - initial_memory
            
            return {
                'initial_memory_gb': initial_memory,
                'max_memory_gb': max_observed_memory,
                'memory_growth_gb': memory_growth,
                'iterations': iterations,
                'memory_per_iteration_mb': (memory_growth * 1024) / iterations
            }
        
        return TestCase(
            id=f"memory_{name}_{uuid.uuid4().hex[:8]}",
            name=f"Memory Stress Test: {name}",
            description=f"Memory stress test with limit {max_memory_gb:.1f}GB",
            test_type=TestType.STRESS,
            priority=TestPriority.HIGH,
            function=memory_stress_test,
            timeout=300.0
        )
    
    def generate_cpu_stress_test(self,
                               name: str,
                               target_function: Callable,
                               duration_seconds: float = 60.0,
                               cpu_count: Optional[int] = None) -> TestCase:
        """Generate CPU stress test"""
        
        cpu_count = cpu_count or self.cpu_count
        
        def cpu_stress_test():
            def worker():
                start_time = time.time()
                iterations = 0
                
                while time.time() - start_time < duration_seconds:
                    target_function()
                    iterations += 1
                
                return iterations
            
            with ProcessPoolExecutor(max_workers=cpu_count) as executor:
                futures = [executor.submit(worker) for _ in range(cpu_count)]
                results = [future.result() for future in as_completed(futures)]
            
            total_iterations = sum(results)
            iterations_per_second = total_iterations / duration_seconds
            
            return {
                'total_iterations': total_iterations,
                'iterations_per_second': iterations_per_second,
                'cpu_count': cpu_count,
                'duration_seconds': duration_seconds
            }
        
        return TestCase(
            id=f"cpu_{name}_{uuid.uuid4().hex[:8]}",
            name=f"CPU Stress Test: {name}",
            description=f"CPU stress test using {cpu_count} processes for {duration_seconds}s",
            test_type=TestType.STRESS,
            priority=TestPriority.HIGH,
            function=cpu_stress_test,
            timeout=duration_seconds + 30.0
        )

class FuzzTestGenerator:
    """Fuzzing test generator for security and robustness testing"""
    
    def __init__(self):
        self.random = random.Random()
    
    def generate_string_fuzz_test(self,
                                name: str,
                                target_function: Callable[[str], Any],
                                max_length: int = 10000,
                                iterations: int = 1000) -> TestCase:
        """Generate string fuzzing test"""
        
        def string_fuzz_test():
            issues_found = []
            
            for i in range(iterations):
                # Generate random string
                length = self.random.randint(0, max_length)
                
                # Various fuzzing strategies
                if i % 10 == 0:
                    # Null bytes
                    fuzz_string = '\x00' * length
                elif i % 10 == 1:
                    # Unicode characters
                    fuzz_string = ''.join(chr(self.random.randint(0, 0x10FFFF)) for _ in range(length))
                elif i % 10 == 2:
                    # Control characters
                    fuzz_string = ''.join(chr(self.random.randint(0, 31)) for _ in range(length))
                elif i % 10 == 3:
                    # Very long strings
                    fuzz_string = 'A' * max_length
                elif i % 10 == 4:
                    # SQL injection patterns
                    patterns = ["'; DROP TABLE --", "' OR '1'='1", "\\'; EXEC xp_cmdshell('dir') --"]
                    fuzz_string = self.random.choice(patterns) * (length // 20 + 1)
                elif i % 10 == 5:
                    # Path traversal
                    fuzz_string = '../' * (length // 3 + 1)
                elif i % 10 == 6:
                    # Format string attacks
                    fuzz_string = '%s%s%s%s%s' * (length // 10 + 1)
                elif i % 10 == 7:
                    # Buffer overflow patterns
                    fuzz_string = 'A' * length
                elif i % 10 == 8:
                    # Empty or whitespace
                    fuzz_string = ' \t\n\r' * (length // 4 + 1)
                else:
                    # Random ASCII
                    fuzz_string = ''.join(self.random.choice(string.printable) for _ in range(length))
                
                try:
                    result = target_function(fuzz_string)
                    
                    # Check for suspicious behavior
                    if isinstance(result, str) and len(result) > max_length * 10:
                        issues_found.append(f"Iteration {i}: Excessive output length: {len(result)}")
                    
                except MemoryError:
                    issues_found.append(f"Iteration {i}: Memory error with input length {len(fuzz_string)}")
                except RecursionError:
                    issues_found.append(f"Iteration {i}: Recursion error")
                except Exception as e:
                    # Some exceptions are expected, but log unusual ones
                    if not isinstance(e, (ValueError, TypeError, AttributeError)):
                        issues_found.append(f"Iteration {i}: Unexpected error {type(e).__name__}: {e}")
            
            return {
                'iterations': iterations,
                'issues_found': len(issues_found),
                'issues': issues_found[:10]  # Limit output
            }
        
        return TestCase(
            id=f"fuzz_string_{name}_{uuid.uuid4().hex[:8]}",
            name=f"String Fuzz Test: {name}",
            description=f"String fuzzing test with {iterations} iterations",
            test_type=TestType.FUZZ,
            priority=TestPriority.MEDIUM,
            function=string_fuzz_test,
            timeout=120.0
        )
    
    def generate_numeric_fuzz_test(self,
                                 name: str,
                                 target_function: Callable[[Union[int, float]], Any],
                                 iterations: int = 1000) -> TestCase:
        """Generate numeric fuzzing test"""
        
        def numeric_fuzz_test():
            issues_found = []
            
            # Special numeric values to test
            special_values = [
                0, 1, -1, 
                float('inf'), float('-inf'), float('nan'),
                2**31 - 1, -2**31, 2**63 - 1, -2**63,  # Integer limits
                1e308, -1e308, 1e-308, -1e-308,  # Float limits
                3.141592653589793, 2.718281828459045,  # Mathematical constants
            ]
            
            for i in range(iterations):
                if i < len(special_values):
                    test_value = special_values[i]
                elif i % 3 == 0:
                    # Random integer
                    test_value = self.random.randint(-2**63, 2**63 - 1)
                elif i % 3 == 1:
                    # Random float
                    test_value = self.random.uniform(-1e10, 1e10)
                else:
                    # Edge cases
                    exponent = self.random.randint(-100, 100)
                    test_value = self.random.uniform(0.1, 9.9) * (10 ** exponent)
                
                try:
                    result = target_function(test_value)
                    
                    # Check for suspicious behavior
                    if isinstance(result, (int, float)):
                        if abs(result) > 1e20:
                            issues_found.append(f"Iteration {i}: Excessive result magnitude: {result}")
                        elif result != result:  # NaN check
                            issues_found.append(f"Iteration {i}: NaN result from input {test_value}")
                    
                except OverflowError:
                    issues_found.append(f"Iteration {i}: Overflow with input {test_value}")
                except ZeroDivisionError:
                    # Expected for some inputs
                    pass
                except Exception as e:
                    if not isinstance(e, (ValueError, TypeError, ArithmeticError)):
                        issues_found.append(f"Iteration {i}: Unexpected error {type(e).__name__}: {e}")
            
            return {
                'iterations': iterations,
                'issues_found': len(issues_found),
                'issues': issues_found[:10]
            }
        
        return TestCase(
            id=f"fuzz_numeric_{name}_{uuid.uuid4().hex[:8]}",
            name=f"Numeric Fuzz Test: {name}",
            description=f"Numeric fuzzing test with {iterations} iterations",
            test_type=TestType.FUZZ,
            priority=TestPriority.MEDIUM,
            function=numeric_fuzz_test,
            timeout=60.0
        )

class PerformanceBenchmark:
    """Performance benchmarking and regression testing"""
    
    def __init__(self):
        self.baseline_results: Dict[str, Dict[str, float]] = {}
    
    def generate_performance_test(self,
                                name: str,
                                target_function: Callable,
                                iterations: int = 100,
                                warmup_iterations: int = 10,
                                max_acceptable_time: Optional[float] = None,
                                max_memory_mb: Optional[float] = None) -> TestCase:
        """Generate performance benchmark test"""
        
        def performance_test():
            # Warmup
            for _ in range(warmup_iterations):
                target_function()
            
            # Measurements
            execution_times = []
            memory_usage = []
            
            for i in range(iterations):
                # Memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss / (1024 * 1024)  # MB
                
                # Execute and time
                start_time = time.perf_counter()
                result = target_function()
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                # Memory after
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_usage.append(memory_after - memory_before)
                
                # Garbage collection periodically
                if i % 10 == 0:
                    gc.collect()
            
            # Calculate statistics
            stats = {
                'iterations': iterations,
                'avg_time': statistics.mean(execution_times),
                'min_time': min(execution_times),
                'max_time': max(execution_times),
                'median_time': statistics.median(execution_times),
                'stdev_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'p95_time': statistics.quantiles(execution_times, n=20)[18] if len(execution_times) >= 20 else max(execution_times),
                'p99_time': statistics.quantiles(execution_times, n=100)[98] if len(execution_times) >= 100 else max(execution_times),
                'avg_memory_delta_mb': statistics.mean(memory_usage),
                'max_memory_delta_mb': max(memory_usage),
                'operations_per_second': 1.0 / statistics.mean(execution_times) if statistics.mean(execution_times) > 0 else 0
            }
            
            # Performance assertions
            if max_acceptable_time and stats['p95_time'] > max_acceptable_time:
                raise AssertionError(
                    f"Performance regression: P95 time {stats['p95_time']:.4f}s > {max_acceptable_time:.4f}s"
                )
            
            if max_memory_mb and stats['max_memory_delta_mb'] > max_memory_mb:
                raise AssertionError(
                    f"Memory usage too high: {stats['max_memory_delta_mb']:.2f}MB > {max_memory_mb:.2f}MB"
                )
            
            # Check against baseline if available
            if name in self.baseline_results:
                baseline = self.baseline_results[name]
                
                # Allow 20% regression
                if stats['avg_time'] > baseline.get('avg_time', float('inf')) * 1.2:
                    raise AssertionError(
                        f"Performance regression: avg time {stats['avg_time']:.4f}s vs baseline {baseline['avg_time']:.4f}s"
                    )
            else:
                # Store as baseline
                self.baseline_results[name] = stats
            
            return stats
        
        return TestCase(
            id=f"perf_{name}_{uuid.uuid4().hex[:8]}",
            name=f"Performance Test: {name}",
            description=f"Performance benchmark with {iterations} iterations",
            test_type=TestType.PERFORMANCE,
            priority=TestPriority.HIGH,
            function=performance_test,
            timeout=max(60.0, iterations * 0.1 + 30)
        )
    
    def save_baseline(self, file_path: str):
        """Save baseline results to file"""
        with open(file_path, 'w') as f:
            json.dump(self.baseline_results, f, indent=2)
    
    def load_baseline(self, file_path: str):
        """Load baseline results from file"""
        try:
            with open(file_path, 'r') as f:
                self.baseline_results = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Baseline file not found: {file_path}")

class TestRunner:
    """Test execution engine"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 default_timeout: float = 60.0,
                 enable_coverage: bool = False):
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.enable_coverage = enable_coverage
        
        # Test results storage
        self.executions: List[TestExecution] = []
        self.suite_results: Dict[str, List[TestExecution]] = {}
        
        # Coverage tracking
        if enable_coverage:
            try:
                import coverage
                self.coverage = coverage.Coverage()
                self.coverage_available = True
            except ImportError:
                logger.warning("Coverage.py not available")
                self.coverage_available = False
        else:
            self.coverage_available = False
    
    def run_test_case(self, test_case: TestCase) -> TestExecution:
        """Run single test case"""
        start_time = time.time()
        
        # Setup
        if test_case.setup_function:
            try:
                test_case.setup_function()
            except Exception as e:
                return TestExecution(
                    test_case=test_case,
                    result=TestResult.ERROR,
                    start_time=start_time,
                    end_time=time.time(),
                    duration=time.time() - start_time,
                    error_message=f"Setup failed: {e}",
                    stack_trace=traceback.format_exc()
                )
        
        # Execute test with retries
        for attempt in range(test_case.retries + 1):
            test_start = time.time()
            
            try:
                # Start coverage if enabled
                if self.coverage_available and hasattr(self, 'coverage'):
                    self.coverage.start()
                
                # Execute test function
                result = test_case.function()
                
                # Stop coverage
                if self.coverage_available and hasattr(self, 'coverage'):
                    self.coverage.stop()
                
                test_end = time.time()
                
                # Check timeout
                if test_end - test_start > test_case.timeout:
                    return TestExecution(
                        test_case=test_case,
                        result=TestResult.TIMEOUT,
                        start_time=start_time,
                        end_time=test_end,
                        duration=test_end - start_time,
                        error_message=f"Test exceeded timeout of {test_case.timeout}s",
                        retry_count=attempt
                    )
                
                # Success
                execution = TestExecution(
                    test_case=test_case,
                    result=TestResult.PASSED,
                    start_time=start_time,
                    end_time=test_end,
                    duration=test_end - start_time,
                    output=str(result) if result is not None else None,
                    retry_count=attempt
                )
                
                break
                
            except Exception as e:
                test_end = time.time()
                
                # Check if this is an expected exception
                if test_case.expected_exceptions and any(
                    isinstance(e, exc_type) for exc_type in test_case.expected_exceptions
                ):
                    execution = TestExecution(
                        test_case=test_case,
                        result=TestResult.PASSED,
                        start_time=start_time,
                        end_time=test_end,
                        duration=test_end - start_time,
                        output=f"Expected exception: {type(e).__name__}",
                        retry_count=attempt
                    )
                    break
                
                # Retry on failure
                if attempt < test_case.retries:
                    logger.info(f"Retrying test {test_case.name} (attempt {attempt + 2}/{test_case.retries + 1})")
                    continue
                
                # Final failure
                execution = TestExecution(
                    test_case=test_case,
                    result=TestResult.FAILED,
                    start_time=start_time,
                    end_time=test_end,
                    duration=test_end - start_time,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    retry_count=attempt
                )
                break
        
        # Teardown
        if test_case.teardown_function:
            try:
                test_case.teardown_function()
            except Exception as e:
                logger.error(f"Teardown failed for {test_case.name}: {e}")
        
        return execution
    
    def run_test_suite(self, test_suite: TestSuite) -> List[TestExecution]:
        """Run test suite"""
        logger.info(f"Running test suite: {test_suite.name}")
        
        # Suite setup
        if test_suite.setup_function:
            try:
                test_suite.setup_function()
            except Exception as e:
                logger.error(f"Suite setup failed: {e}")
                return []
        
        # Sort tests by priority and dependencies
        sorted_tests = self._sort_tests_by_dependencies(test_suite.test_cases)
        
        executions = []
        
        if test_suite.parallel_execution:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(test_suite.max_workers, len(sorted_tests))) as executor:
                future_to_test = {executor.submit(self.run_test_case, test): test for test in sorted_tests}
                
                for future in as_completed(future_to_test):
                    try:
                        execution = future.result()
                        executions.append(execution)
                        self.executions.append(execution)
                    except Exception as e:
                        test = future_to_test[future]
                        logger.error(f"Error running test {test.name}: {e}")
        else:
            # Sequential execution
            for test_case in sorted_tests:
                execution = self.run_test_case(test_case)
                executions.append(execution)
                self.executions.append(execution)
                
                # Stop on critical test failure if needed
                if (execution.result == TestResult.FAILED and 
                    test_case.priority == TestPriority.CRITICAL):
                    logger.error(f"Critical test failed: {test_case.name}. Stopping suite execution.")
                    break
        
        # Suite teardown
        if test_suite.teardown_function:
            try:
                test_suite.teardown_function()
            except Exception as e:
                logger.error(f"Suite teardown failed: {e}")
        
        self.suite_results[test_suite.name] = executions
        
        # Log summary
        passed = sum(1 for e in executions if e.result == TestResult.PASSED)
        failed = sum(1 for e in executions if e.result == TestResult.FAILED)
        errors = sum(1 for e in executions if e.result == TestResult.ERROR)
        
        logger.info(f"Test suite {test_suite.name} completed: {passed} passed, {failed} failed, {errors} errors")
        
        return executions
    
    def _sort_tests_by_dependencies(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Sort tests by dependencies and priority"""
        # Simple topological sort for dependencies
        # For now, just sort by priority
        return sorted(test_cases, key=lambda tc: (tc.priority.level, tc.name), reverse=True)
    
    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.executions:
            return {'error': 'No test executions found'}
        
        # Calculate statistics
        total_tests = len(self.executions)
        passed = sum(1 for e in self.executions if e.result == TestResult.PASSED)
        failed = sum(1 for e in self.executions if e.result == TestResult.FAILED)
        errors = sum(1 for e in self.executions if e.result == TestResult.ERROR)
        timeouts = sum(1 for e in self.executions if e.result == TestResult.TIMEOUT)
        
        total_duration = sum(e.duration for e in self.executions)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        # Group by test type
        by_type = defaultdict(lambda: {'passed': 0, 'failed': 0, 'errors': 0, 'total': 0})
        for execution in self.executions:
            test_type = execution.test_case.test_type.value
            by_type[test_type]['total'] += 1
            if execution.result == TestResult.PASSED:
                by_type[test_type]['passed'] += 1
            elif execution.result == TestResult.FAILED:
                by_type[test_type]['failed'] += 1
            elif execution.result == TestResult.ERROR:
                by_type[test_type]['errors'] += 1
        
        # Performance metrics
        performance_tests = [e for e in self.executions if e.test_case.test_type == TestType.PERFORMANCE]
        performance_metrics = {}
        
        if performance_tests:
            for execution in performance_tests:
                if execution.metrics:
                    performance_metrics[execution.test_case.name] = execution.metrics
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'timeouts': timeouts,
                'success_rate': passed / total_tests if total_tests > 0 else 0,
                'total_duration': total_duration,
                'avg_duration': avg_duration
            },
            'by_type': dict(by_type),
            'performance_metrics': performance_metrics,
            'failed_tests': [
                {
                    'name': e.test_case.name,
                    'type': e.test_case.test_type.value,
                    'error': e.error_message,
                    'duration': e.duration
                }
                for e in self.executions if e.result == TestResult.FAILED
            ],
            'slowest_tests': sorted(
                [
                    {
                        'name': e.test_case.name,
                        'type': e.test_case.test_type.value,
                        'duration': e.duration
                    }
                    for e in self.executions
                ],
                key=lambda x: x['duration'],
                reverse=True
            )[:10],
            'executions': [e.to_dict() for e in self.executions]
        }
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Test report saved to {output_file}")
        
        return report
    
    def plot_performance_trends(self, output_file: str = None):
        """Plot performance trends over time"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return
        
        performance_tests = [e for e in self.executions if e.test_case.test_type == TestType.PERFORMANCE]
        
        if not performance_tests:
            logger.warning("No performance tests found for plotting")
            return
        
        # Group by test name
        by_test = defaultdict(list)
        for execution in performance_tests:
            by_test[execution.test_case.name].append(execution)
        
        fig, axes = plt.subplots(len(by_test), 1, figsize=(12, 4 * len(by_test)))
        
        if len(by_test) == 1:
            axes = [axes]
        
        for i, (test_name, executions) in enumerate(by_test.items()):
            times = [e.start_time for e in executions]
            durations = [e.duration for e in executions]
            
            axes[i].plot(times, durations, 'o-')
            axes[i].set_title(f'Performance Trend: {test_name}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Duration (seconds)')
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Performance trends plot saved to {output_file}")
        else:
            plt.show()

# Global test runner instance
_global_test_runner: Optional[TestRunner] = None

def get_global_test_runner(**kwargs) -> TestRunner:
    """Get or create global test runner instance"""
    global _global_test_runner
    if _global_test_runner is None:
        _global_test_runner = TestRunner(**kwargs)
    return _global_test_runner

# Convenience functions
def create_property_test(name: str, property_function: Callable, **kwargs) -> TestCase:
    """Create property-based test"""
    generator = PropertyTestGenerator()
    # This would need specific strategy based on the function signature
    strategy = st.text()  # Default strategy
    return generator.generate_property_test(name, strategy, property_function, **kwargs)

def create_stress_test(name: str, target_function: Callable, test_type: str = "load", **kwargs) -> TestCase:
    """Create stress test"""
    generator = StressTestGenerator()
    
    if test_type == "load":
        return generator.generate_load_test(name, target_function, **kwargs)
    elif test_type == "memory":
        return generator.generate_memory_stress_test(name, target_function, **kwargs)
    elif test_type == "cpu":
        return generator.generate_cpu_stress_test(name, target_function, **kwargs)
    else:
        raise ValueError(f"Unknown stress test type: {test_type}")

def create_fuzz_test(name: str, target_function: Callable, input_type: str = "string", **kwargs) -> TestCase:
    """Create fuzz test"""
    generator = FuzzTestGenerator()
    
    if input_type == "string":
        return generator.generate_string_fuzz_test(name, target_function, **kwargs)
    elif input_type == "numeric":
        return generator.generate_numeric_fuzz_test(name, target_function, **kwargs)
    else:
        raise ValueError(f"Unknown fuzz test input type: {input_type}")

def create_performance_test(name: str, target_function: Callable, **kwargs) -> TestCase:
    """Create performance test"""
    benchmark = PerformanceBenchmark()
    return benchmark.generate_performance_test(name, target_function, **kwargs)

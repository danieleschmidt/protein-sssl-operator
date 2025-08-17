"""
Advanced Parallel Processing Framework for Protein-SSL Operator
Implements work stealing, load balancing, and intelligent task distribution
"""

import time
import threading
import multiprocessing as mp
import queue
import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
import numpy as np
import psutil
import os
import signal
from abc import ABC, abstractmethod

from .logging_config import setup_logging
from .monitoring import MetricsCollector

logger = setup_logging(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class WorkerType(Enum):
    """Worker types for different workloads"""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    GPU_COMPUTE = "gpu_compute"
    MIXED = "mixed"


@dataclass
class Task:
    """Task representation for parallel processing"""
    id: str
    func: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    worker_type: WorkerType = WorkerType.MIXED
    estimated_duration: float = 1.0  # seconds
    memory_requirement: int = 64  # MB
    dependencies: List[str] = None
    timeout: Optional[float] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class WorkerStats:
    """Worker performance statistics"""
    worker_id: str
    worker_type: WorkerType
    tasks_completed: int
    tasks_failed: int
    total_execution_time: float
    average_execution_time: float
    current_load: float
    memory_usage_mb: float
    cpu_utilization: float
    efficiency_score: float
    last_task_completion: float


class WorkStealingQueue:
    """Thread-safe work stealing queue implementation"""
    
    def __init__(self, worker_id: str, max_size: int = 10000):
        self.worker_id = worker_id
        self.max_size = max_size
        self._deque = deque()
        self._lock = threading.RLock()
        self.tasks_added = 0
        self.tasks_stolen = 0
        self.tasks_completed = 0
    
    def push_bottom(self, task: Task) -> bool:
        """Push task to bottom of deque (for owner worker)"""
        with self._lock:
            if len(self._deque) >= self.max_size:
                return False
            
            self._deque.append(task)
            self.tasks_added += 1
            return True
    
    def pop_bottom(self) -> Optional[Task]:
        """Pop task from bottom of deque (for owner worker)"""
        with self._lock:
            if not self._deque:
                return None
            
            task = self._deque.pop()
            self.tasks_completed += 1
            return task
    
    def steal_top(self) -> Optional[Task]:
        """Steal task from top of deque (for other workers)"""
        with self._lock:
            if not self._deque:
                return None
            
            task = self._deque.popleft()
            self.tasks_stolen += 1
            return task
    
    def size(self) -> int:
        """Get current queue size"""
        with self._lock:
            return len(self._deque)
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        with self._lock:
            return len(self._deque) == 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            return {
                'worker_id': self.worker_id,
                'current_size': len(self._deque),
                'max_size': self.max_size,
                'tasks_added': self.tasks_added,
                'tasks_stolen': self.tasks_stolen,
                'tasks_completed': self.tasks_completed,
                'steal_ratio': self.tasks_stolen / max(self.tasks_added, 1)
            }


class LoadBalancer:
    """Intelligent load balancer for task distribution"""
    
    def __init__(self, workers: List['Worker']):
        self.workers = workers
        self.worker_stats = {w.worker_id: WorkerStats(
            worker_id=w.worker_id,
            worker_type=w.worker_type,
            tasks_completed=0,
            tasks_failed=0,
            total_execution_time=0.0,
            average_execution_time=0.0,
            current_load=0.0,
            memory_usage_mb=0.0,
            cpu_utilization=0.0,
            efficiency_score=1.0,
            last_task_completion=time.time()
        ) for w in workers}
        
        # Load balancing strategies
        self.strategies = {
            'round_robin': self._round_robin_assignment,
            'least_loaded': self._least_loaded_assignment,
            'capability_based': self._capability_based_assignment,
            'efficiency_based': self._efficiency_based_assignment,
            'hybrid': self._hybrid_assignment
        }
        
        self.current_strategy = 'hybrid'
        self.round_robin_index = 0
        
        # Performance tracking
        self.assignment_history = deque(maxlen=1000)
        self.rebalance_threshold = 0.3  # 30% load imbalance triggers rebalancing
    
    def assign_task(self, task: Task) -> Optional['Worker']:
        """Assign task to optimal worker"""
        strategy = self.strategies.get(self.current_strategy, self._hybrid_assignment)
        worker = strategy(task)
        
        if worker:
            # Record assignment
            self.assignment_history.append({
                'task_id': task.id,
                'worker_id': worker.worker_id,
                'assignment_time': time.time(),
                'strategy': self.current_strategy,
                'worker_load': self.worker_stats[worker.worker_id].current_load
            })
        
        return worker
    
    def _round_robin_assignment(self, task: Task) -> Optional['Worker']:
        """Simple round-robin assignment"""
        if not self.workers:
            return None
        
        worker = self.workers[self.round_robin_index % len(self.workers)]
        self.round_robin_index += 1
        return worker
    
    def _least_loaded_assignment(self, task: Task) -> Optional['Worker']:
        """Assign to worker with least current load"""
        if not self.workers:
            return None
        
        available_workers = [w for w in self.workers if w.is_available()]
        if not available_workers:
            return None
        
        return min(available_workers, 
                  key=lambda w: self.worker_stats[w.worker_id].current_load)
    
    def _capability_based_assignment(self, task: Task) -> Optional['Worker']:
        """Assign based on worker capabilities and task requirements"""
        suitable_workers = []
        
        for worker in self.workers:
            if not worker.is_available():
                continue
            
            # Check if worker type matches task requirements
            if (worker.worker_type == task.worker_type or 
                worker.worker_type == WorkerType.MIXED or
                task.worker_type == WorkerType.MIXED):
                
                # Check memory requirements
                available_memory = worker.get_available_memory()
                if available_memory >= task.memory_requirement:
                    suitable_workers.append(worker)
        
        if not suitable_workers:
            return self._least_loaded_assignment(task)  # Fallback
        
        # Choose least loaded among suitable workers
        return min(suitable_workers,
                  key=lambda w: self.worker_stats[w.worker_id].current_load)
    
    def _efficiency_based_assignment(self, task: Task) -> Optional['Worker']:
        """Assign based on worker efficiency scores"""
        available_workers = [w for w in self.workers if w.is_available()]
        if not available_workers:
            return None
        
        # Score workers based on efficiency and current load
        def score_worker(worker):
            stats = self.worker_stats[worker.worker_id]
            # Higher efficiency, lower load = better score
            return stats.efficiency_score / max(stats.current_load + 0.1, 0.1)
        
        return max(available_workers, key=score_worker)
    
    def _hybrid_assignment(self, task: Task) -> Optional['Worker']:
        """Hybrid strategy combining multiple factors"""
        available_workers = [w for w in self.workers if w.is_available()]
        if not available_workers:
            return None
        
        best_worker = None
        best_score = float('-inf')
        
        for worker in available_workers:
            stats = self.worker_stats[worker.worker_id]
            
            # Base score from efficiency
            score = stats.efficiency_score * 0.4
            
            # Load factor (lower load = higher score)
            load_factor = max(0, 1.0 - stats.current_load)
            score += load_factor * 0.3
            
            # Capability match bonus
            if (worker.worker_type == task.worker_type or 
                worker.worker_type == WorkerType.MIXED):
                score += 0.2
            
            # Memory availability factor
            available_memory = worker.get_available_memory()
            if available_memory >= task.memory_requirement:
                memory_factor = min(1.0, available_memory / task.memory_requirement)
                score += memory_factor * 0.1
            else:
                score -= 0.5  # Penalty for insufficient memory
            
            if score > best_score:
                best_score = score
                best_worker = worker
        
        return best_worker
    
    def update_worker_stats(self, worker_id: str, task_completion_time: float, 
                          task_success: bool, memory_usage: float = 0.0) -> None:
        """Update worker statistics after task completion"""
        if worker_id not in self.worker_stats:
            return
        
        stats = self.worker_stats[worker_id]
        
        if task_success:
            stats.tasks_completed += 1
            stats.total_execution_time += task_completion_time
            stats.average_execution_time = (
                stats.total_execution_time / max(stats.tasks_completed, 1)
            )
        else:
            stats.tasks_failed += 1
        
        stats.memory_usage_mb = memory_usage
        stats.last_task_completion = time.time()
        
        # Update efficiency score
        self._update_efficiency_score(worker_id)
    
    def _update_efficiency_score(self, worker_id: str) -> None:
        """Update worker efficiency score based on performance"""
        stats = self.worker_stats[worker_id]
        
        total_tasks = stats.tasks_completed + stats.tasks_failed
        if total_tasks == 0:
            return
        
        # Success rate component
        success_rate = stats.tasks_completed / total_tasks
        
        # Speed component (inverse of average execution time, normalized)
        if stats.average_execution_time > 0:
            speed_factor = min(1.0, 10.0 / stats.average_execution_time)
        else:
            speed_factor = 1.0
        
        # Recency component (more recent performance weighted higher)
        time_since_last = time.time() - stats.last_task_completion
        recency_factor = max(0.1, 1.0 - (time_since_last / 3600))  # Decay over 1 hour
        
        # Combined efficiency score
        stats.efficiency_score = (
            success_rate * 0.5 + 
            speed_factor * 0.3 + 
            recency_factor * 0.2
        )
    
    def rebalance_if_needed(self) -> bool:
        """Check if rebalancing is needed and perform it"""
        if len(self.workers) < 2:
            return False
        
        # Calculate load distribution
        loads = [self.worker_stats[w.worker_id].current_load for w in self.workers]
        if not loads:
            return False
        
        min_load = min(loads)
        max_load = max(loads)
        
        # Check if imbalance exceeds threshold
        if max_load - min_load > self.rebalance_threshold:
            logger.info(f"Load imbalance detected: {min_load:.2f} - {max_load:.2f}, rebalancing...")
            return self._perform_rebalancing()
        
        return False
    
    def _perform_rebalancing(self) -> bool:
        """Perform work stealing to rebalance load"""
        # Find most loaded and least loaded workers
        most_loaded = max(self.workers, 
                         key=lambda w: self.worker_stats[w.worker_id].current_load)
        least_loaded = min(self.workers,
                          key=lambda w: self.worker_stats[w.worker_id].current_load)
        
        # Attempt work stealing
        if hasattr(most_loaded, 'queue') and hasattr(least_loaded, 'queue'):
            stolen_task = most_loaded.queue.steal_top()
            if stolen_task:
                success = least_loaded.queue.push_bottom(stolen_task)
                if success:
                    logger.debug(f"Rebalanced: moved task from {most_loaded.worker_id} to {least_loaded.worker_id}")
                    return True
        
        return False
    
    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        loads = [self.worker_stats[w.worker_id].current_load for w in self.workers]
        
        if loads:
            load_stats = {
                'min_load': min(loads),
                'max_load': max(loads),
                'avg_load': sum(loads) / len(loads),
                'load_variance': np.var(loads),
                'balance_score': 1.0 - (max(loads) - min(loads))  # 1.0 = perfectly balanced
            }
        else:
            load_stats = {'min_load': 0, 'max_load': 0, 'avg_load': 0, 'load_variance': 0, 'balance_score': 1.0}
        
        return {
            'current_strategy': self.current_strategy,
            'total_assignments': len(self.assignment_history),
            'load_statistics': load_stats,
            'worker_stats': {w_id: stats.__dict__ for w_id, stats in self.worker_stats.items()}
        }


class Worker(ABC):
    """Abstract base class for workers"""
    
    def __init__(self, worker_id: str, worker_type: WorkerType, max_memory_mb: int = 1024):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.max_memory_mb = max_memory_mb
        
        # Work stealing queue
        self.queue = WorkStealingQueue(worker_id)
        
        # Worker state
        self.is_running = False
        self.current_task = None
        self.last_activity = time.time()
        
        # Performance tracking
        self.tasks_processed = 0
        self.total_processing_time = 0.0
        self.memory_usage_mb = 0.0
    
    @abstractmethod
    def execute_task(self, task: Task) -> Any:
        """Execute a task (implemented by subclasses)"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if worker is available for new tasks"""
        pass
    
    @abstractmethod
    def get_available_memory(self) -> int:
        """Get available memory in MB"""
        pass
    
    def add_task(self, task: Task) -> bool:
        """Add task to worker's queue"""
        return self.queue.push_bottom(task)
    
    def get_task(self) -> Optional[Task]:
        """Get next task from queue"""
        return self.queue.pop_bottom()
    
    def steal_task_from(self, other_worker: 'Worker') -> Optional[Task]:
        """Steal task from another worker"""
        return other_worker.queue.steal_top()


class ThreadWorker(Worker):
    """Thread-based worker for I/O intensive tasks"""
    
    def __init__(self, worker_id: str, max_memory_mb: int = 512):
        super().__init__(worker_id, WorkerType.IO_INTENSIVE, max_memory_mb)
        self.thread = None
        self.stop_event = threading.Event()
    
    def start(self):
        """Start worker thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop worker thread"""
        self.is_running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def _worker_loop(self):
        """Main worker loop"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Try to get a task from own queue
                task = self.get_task()
                
                if task:
                    self._process_task(task)
                else:
                    # No local task, try work stealing
                    time.sleep(0.01)  # Brief pause to avoid busy waiting
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                time.sleep(0.1)
    
    def _process_task(self, task: Task):
        """Process a single task"""
        start_time = time.time()
        self.current_task = task
        
        try:
            result = self.execute_task(task)
            success = True
            
            # Update statistics
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.total_processing_time += processing_time
            self.last_activity = time.time()
            
            logger.debug(f"Worker {self.worker_id} completed task {task.id} in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to execute task {task.id}: {e}")
            success = False
            
        finally:
            self.current_task = None
    
    def execute_task(self, task: Task) -> Any:
        """Execute task in thread context"""
        return task.func(*task.args, **task.kwargs)
    
    def is_available(self) -> bool:
        """Check if worker is available"""
        return self.is_running and self.current_task is None
    
    def get_available_memory(self) -> int:
        """Get available memory (simplified for thread worker)"""
        return max(0, self.max_memory_mb - int(self.memory_usage_mb))


class ProcessWorker(Worker):
    """Process-based worker for CPU intensive tasks"""
    
    def __init__(self, worker_id: str, max_memory_mb: int = 2048):
        super().__init__(worker_id, WorkerType.CPU_INTENSIVE, max_memory_mb)
        self.process = None
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.control_queue = mp.Queue()
    
    def start(self):
        """Start worker process"""
        if self.is_running:
            return
        
        self.is_running = True
        self.process = mp.Process(target=self._worker_process, daemon=True)
        self.process.start()
    
    def stop(self):
        """Stop worker process"""
        self.is_running = False
        try:
            self.control_queue.put('STOP')
            if self.process:
                self.process.join(timeout=5.0)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2.0)
                    if self.process.is_alive():
                        self.process.kill()
        except Exception as e:
            logger.warning(f"Error stopping process worker {self.worker_id}: {e}")
    
    def _worker_process(self):
        """Worker process main loop"""
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore interrupt signals
        
        while True:
            try:
                # Check for control messages
                try:
                    control_msg = self.control_queue.get_nowait()
                    if control_msg == 'STOP':
                        break
                except queue.Empty:
                    pass
                
                # Process tasks
                try:
                    task = self.task_queue.get(timeout=0.1)
                    result = self._process_task_in_process(task)
                    self.result_queue.put(result)
                except queue.Empty:
                    continue
                    
            except Exception as e:
                logger.error(f"Process worker {self.worker_id} error: {e}")
                break
    
    def _process_task_in_process(self, task: Task):
        """Process task in separate process"""
        start_time = time.time()
        
        try:
            result = task.func(*task.args, **task.kwargs)
            processing_time = time.time() - start_time
            
            return {
                'task_id': task.id,
                'result': result,
                'success': True,
                'processing_time': processing_time,
                'worker_id': self.worker_id
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'task_id': task.id,
                'error': str(e),
                'success': False,
                'processing_time': processing_time,
                'worker_id': self.worker_id
            }
    
    def execute_task(self, task: Task) -> Any:
        """Submit task to process (non-blocking)"""
        self.task_queue.put(task)
        return None  # Process workers return results asynchronously
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get completed task result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def is_available(self) -> bool:
        """Check if worker process is available"""
        return (self.is_running and 
                self.process and 
                self.process.is_alive() and
                self.task_queue.qsize() < 10)  # Limit queue size
    
    def get_available_memory(self) -> int:
        """Get available memory for process worker"""
        if self.process and self.process.is_alive():
            try:
                process = psutil.Process(self.process.pid)
                memory_info = process.memory_info()
                used_mb = memory_info.rss / (1024 * 1024)
                return max(0, self.max_memory_mb - int(used_mb))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return self.max_memory_mb


class ParallelProcessor:
    """Main parallel processing coordinator"""
    
    def __init__(self, 
                 num_thread_workers: int = None,
                 num_process_workers: int = None,
                 enable_work_stealing: bool = True,
                 load_balance_strategy: str = 'hybrid'):
        
        # Default worker counts
        if num_thread_workers is None:
            num_thread_workers = min(32, (os.cpu_count() or 1) * 4)
        if num_process_workers is None:
            num_process_workers = os.cpu_count() or 1
        
        self.enable_work_stealing = enable_work_stealing
        
        # Create workers
        self.workers = []
        
        # Thread workers for I/O intensive tasks
        for i in range(num_thread_workers):
            worker = ThreadWorker(f"thread_{i}")
            self.workers.append(worker)
        
        # Process workers for CPU intensive tasks
        for i in range(num_process_workers):
            worker = ProcessWorker(f"process_{i}")
            self.workers.append(worker)
        
        # Load balancer
        self.load_balancer = LoadBalancer(self.workers)
        self.load_balancer.current_strategy = load_balance_strategy
        
        # Task management
        self.pending_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Processing state
        self.is_running = False
        self.manager_thread = None
        self.stats_update_interval = 5.0  # seconds
        
        # Performance metrics
        self.metrics_collector = MetricsCollector()
        self.start_time = None
        
    def start(self):
        """Start parallel processing system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start all workers
        for worker in self.workers:
            worker.start()
        
        # Start management thread
        self.manager_thread = threading.Thread(target=self._management_loop, daemon=True)
        self.manager_thread.start()
        
        logger.info(f"Parallel processor started with {len(self.workers)} workers")
    
    def stop(self):
        """Stop parallel processing system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all workers
        for worker in self.workers:
            worker.stop()
        
        # Stop management thread
        if self.manager_thread:
            self.manager_thread.join(timeout=10.0)
        
        logger.info("Parallel processor stopped")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit task for parallel execution"""
        task_id = f"task_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"
        
        # Extract task configuration from kwargs
        priority = kwargs.pop('priority', TaskPriority.NORMAL)
        worker_type = kwargs.pop('worker_type', WorkerType.MIXED)
        estimated_duration = kwargs.pop('estimated_duration', 1.0)
        memory_requirement = kwargs.pop('memory_requirement', 64)
        timeout = kwargs.pop('timeout', None)
        
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            worker_type=worker_type,
            estimated_duration=estimated_duration,
            memory_requirement=memory_requirement,
            timeout=timeout
        )
        
        # Assign to worker
        worker = self.load_balancer.assign_task(task)
        if worker:
            success = worker.add_task(task)
            if success:
                self.pending_tasks[task_id] = task
                logger.debug(f"Task {task_id} assigned to worker {worker.worker_id}")
            else:
                logger.warning(f"Failed to add task {task_id} to worker {worker.worker_id}")
                return None
        else:
            logger.warning(f"No available worker for task {task_id}")
            return None
        
        return task_id
    
    def get_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """Get result for completed task"""
        start_time = time.time()
        
        while self.is_running:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            elif task_id in self.failed_tasks:
                raise Exception(f"Task {task_id} failed: {self.failed_tasks[task_id]}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            time.sleep(0.01)  # Brief pause
        
        return None
    
    def submit_and_wait(self, func: Callable, *args, timeout: float = None, **kwargs) -> Any:
        """Submit task and wait for result"""
        task_id = self.submit_task(func, *args, **kwargs)
        if task_id is None:
            raise RuntimeError("Failed to submit task")
        
        return self.get_result(task_id, timeout)
    
    def map(self, func: Callable, iterable: Iterator, 
            chunk_size: int = None, timeout: float = None, **task_kwargs) -> List[Any]:
        """Parallel map function"""
        items = list(iterable)
        if not items:
            return []
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (len(self.workers) * 4))
        
        # Submit tasks in chunks
        task_ids = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            task_id = self.submit_task(
                self._map_chunk, func, chunk, 
                timeout=timeout, **task_kwargs
            )
            if task_id:
                task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            try:
                chunk_results = self.get_result(task_id, timeout)
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Map task {task_id} failed: {e}")
                raise
        
        return results
    
    def _map_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items for parallel map"""
        return [func(item) for item in chunk]
    
    def _management_loop(self):
        """Background management loop"""
        last_stats_update = time.time()
        last_rebalance = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Collect results from process workers
                self._collect_process_results()
                
                # Update worker statistics
                if current_time - last_stats_update >= self.stats_update_interval:
                    self._update_worker_statistics()
                    last_stats_update = current_time
                
                # Perform load rebalancing
                if (self.enable_work_stealing and 
                    current_time - last_rebalance >= 10.0):  # Every 10 seconds
                    self.load_balancer.rebalance_if_needed()
                    last_rebalance = current_time
                
                time.sleep(0.1)  # Management loop frequency
                
            except Exception as e:
                logger.error(f"Management loop error: {e}")
                time.sleep(1.0)
    
    def _collect_process_results(self):
        """Collect results from process workers"""
        for worker in self.workers:
            if isinstance(worker, ProcessWorker):
                result = worker.get_result()
                if result:
                    task_id = result['task_id']
                    
                    if result['success']:
                        self.completed_tasks[task_id] = result['result']
                    else:
                        self.failed_tasks[task_id] = result['error']
                    
                    # Remove from pending
                    if task_id in self.pending_tasks:
                        del self.pending_tasks[task_id]
                    
                    # Update load balancer stats
                    self.load_balancer.update_worker_stats(
                        result['worker_id'],
                        result['processing_time'],
                        result['success']
                    )
    
    def _update_worker_statistics(self):
        """Update worker statistics"""
        for worker in self.workers:
            stats = self.load_balancer.worker_stats[worker.worker_id]
            
            # Update current load (simplified)
            queue_size = worker.queue.size()
            max_queue_size = 100  # Assumed max
            stats.current_load = min(1.0, queue_size / max_queue_size)
            
            # Update memory usage
            stats.memory_usage_mb = worker.memory_usage_mb
            
            # Update CPU utilization (simplified)
            if hasattr(worker, 'current_task') and worker.current_task:
                stats.cpu_utilization = 80.0  # Assumed high when processing
            else:
                stats.cpu_utilization = 10.0  # Assumed low when idle
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive parallel processing statistics"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        # Worker statistics
        worker_stats = []
        for worker in self.workers:
            worker_stat = {
                'worker_id': worker.worker_id,
                'worker_type': worker.worker_type.value,
                'is_available': worker.is_available(),
                'queue_size': worker.queue.size(),
                'tasks_processed': worker.tasks_processed,
                'total_processing_time': worker.total_processing_time,
                'avg_processing_time': (
                    worker.total_processing_time / max(worker.tasks_processed, 1)
                ),
                'queue_stats': worker.queue.get_stats()
            }
            worker_stats.append(worker_stat)
        
        # Overall statistics
        total_pending = len(self.pending_tasks)
        total_completed = len(self.completed_tasks)
        total_failed = len(self.failed_tasks)
        total_tasks = total_pending + total_completed + total_failed
        
        return {
            'uptime_seconds': uptime,
            'is_running': self.is_running,
            'total_workers': len(self.workers),
            'task_statistics': {
                'total_submitted': total_tasks,
                'pending': total_pending,
                'completed': total_completed,
                'failed': total_failed,
                'success_rate': total_completed / max(total_tasks, 1)
            },
            'worker_statistics': worker_stats,
            'load_balancer_stats': self.load_balancer.get_load_balance_stats(),
            'performance_metrics': {
                'total_processing_time': sum(w.total_processing_time for w in self.workers),
                'avg_task_completion_time': (
                    sum(w.total_processing_time for w in self.workers) / 
                    max(sum(w.tasks_processed for w in self.workers), 1)
                ),
                'throughput_tasks_per_second': (
                    total_completed / max(uptime, 1)
                )
            }
        }


# Global processor instance
_global_processor = None

def get_parallel_processor(**kwargs) -> ParallelProcessor:
    """Get global parallel processor instance"""
    global _global_processor
    
    if _global_processor is None:
        _global_processor = ParallelProcessor(**kwargs)
    
    return _global_processor

def start_parallel_processing(**kwargs) -> ParallelProcessor:
    """Start global parallel processing"""
    processor = get_parallel_processor(**kwargs)
    processor.start()
    return processor

def submit_parallel_task(func: Callable, *args, **kwargs) -> str:
    """Submit task to global processor"""
    processor = get_parallel_processor()
    return processor.submit_task(func, *args, **kwargs)

def parallel_map(func: Callable, iterable: Iterator, **kwargs) -> List[Any]:
    """Parallel map using global processor"""
    processor = get_parallel_processor()
    return processor.map(func, iterable, **kwargs)
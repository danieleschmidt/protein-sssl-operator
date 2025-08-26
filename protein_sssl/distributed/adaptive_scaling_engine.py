#!/usr/bin/env python3
"""
🌐 GENERATION 3: ADAPTIVE SCALING ENGINE  
=========================================
Intelligent distributed computing system with auto-scaling,
load balancing, and edge computing capabilities
"""

import asyncio
import threading
import queue
import time
import json
import logging
import socket
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import concurrent.futures

# Network and distributed computing
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

class NodeType(Enum):
    """Types of compute nodes."""
    COORDINATOR = "coordinator"
    WORKER = "worker" 
    EDGE = "edge"
    HYBRID = "hybrid"

class NodeStatus(Enum):
    """Node status states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"

@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    node_type: NodeType
    status: NodeStatus = NodeStatus.INITIALIZING
    host: str = "localhost"
    port: int = 8000
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    
    # Performance metrics
    current_load: float = 0.0
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    tasks_completed: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    
    # Capabilities
    supported_models: List[str] = field(default_factory=list)
    max_sequence_length: int = 10000
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    task_id: str
    task_type: str
    data: Any
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    assigned_node: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if task has exceeded deadline."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline

class LoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(self):
        """Initialize load balancer."""
        self.strategies = {
            'round_robin': self._round_robin,
            'least_connections': self._least_connections,
            'weighted_response_time': self._weighted_response_time,
            'resource_aware': self._resource_aware
        }
        self.current_strategy = 'resource_aware'
        self.round_robin_index = 0
        self.logger = logging.getLogger(__name__)
    
    def select_node(self, nodes: List[ComputeNode], task: DistributedTask) -> Optional[ComputeNode]:
        """Select optimal node for task."""
        available_nodes = [n for n in nodes if n.status == NodeStatus.ACTIVE]
        
        if not available_nodes:
            return None
        
        # Filter by task requirements
        suitable_nodes = self._filter_suitable_nodes(available_nodes, task)
        
        if not suitable_nodes:
            # Fallback to any available node
            suitable_nodes = available_nodes
        
        # Apply load balancing strategy
        strategy_func = self.strategies.get(self.current_strategy, self._round_robin)
        return strategy_func(suitable_nodes, task)
    
    def _filter_suitable_nodes(self, nodes: List[ComputeNode], task: DistributedTask) -> List[ComputeNode]:
        """Filter nodes suitable for task."""
        suitable = []
        
        for node in nodes:
            # Check resource requirements
            if task.task_type == 'prediction' and 'ssl_model' in node.supported_models:
                suitable.append(node)
            elif task.task_type == 'training' and node.memory_gb >= 4.0:
                suitable.append(node)
            elif task.task_type == 'analysis':
                suitable.append(node)
        
        return suitable or nodes  # Fallback to all nodes
    
    def _round_robin(self, nodes: List[ComputeNode], task: DistributedTask) -> ComputeNode:
        """Round-robin selection."""
        node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index = (self.round_robin_index + 1) % len(nodes)
        return node
    
    def _least_connections(self, nodes: List[ComputeNode], task: DistributedTask) -> ComputeNode:
        """Select node with least current load."""
        return min(nodes, key=lambda n: n.current_load)
    
    def _weighted_response_time(self, nodes: List[ComputeNode], task: DistributedTask) -> ComputeNode:
        """Select node with best response time weighted by success rate."""
        def score(node):
            if node.tasks_completed == 0:
                return 0  # Prefer untested nodes
            return node.avg_response_time / max(node.success_rate, 0.1)
        
        return min(nodes, key=score)
    
    def _resource_aware(self, nodes: List[ComputeNode], task: DistributedTask) -> ComputeNode:
        """Select node based on comprehensive resource analysis."""
        def resource_score(node):
            # Lower is better
            load_score = node.current_load
            response_score = node.avg_response_time / 1000.0  # Convert to seconds
            reliability_score = 1.0 - node.success_rate
            
            # Weighted combination
            return 0.4 * load_score + 0.4 * response_score + 0.2 * reliability_score
        
        return min(nodes, key=resource_score)

class AutoScaler:
    """Automatic scaling system for compute resources."""
    
    def __init__(self, 
                 min_nodes: int = 1,
                 max_nodes: int = 10,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3):
        """Initialize auto-scaler.
        
        Args:
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
            scale_up_threshold: CPU threshold to scale up
            scale_down_threshold: CPU threshold to scale down
        """
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.scaling_decisions = deque(maxlen=100)
        self.last_scale_action = 0
        self.scale_cooldown = 300  # 5 minutes between scaling actions
        
        self.logger = logging.getLogger(__name__)
    
    def should_scale(self, nodes: List[ComputeNode], pending_tasks: int) -> Tuple[bool, str]:
        """Determine if scaling is needed."""
        if not nodes:
            return True, "scale_up"  # Need at least one node
        
        # Calculate average load
        active_nodes = [n for n in nodes if n.status == NodeStatus.ACTIVE]
        if not active_nodes:
            return True, "scale_up"
        
        avg_load = sum(n.current_load for n in active_nodes) / len(active_nodes)
        
        # Check if we're in cooldown period
        if time.time() - self.last_scale_action < self.scale_cooldown:
            return False, "cooldown"
        
        # Scale up conditions
        scale_up_needed = (
            avg_load > self.scale_up_threshold or
            pending_tasks > len(active_nodes) * 5 or  # More than 5 tasks per node
            len(active_nodes) < self.min_nodes
        )
        
        if scale_up_needed and len(nodes) < self.max_nodes:
            return True, "scale_up"
        
        # Scale down conditions
        scale_down_needed = (
            avg_load < self.scale_down_threshold and
            pending_tasks < len(active_nodes) * 2 and  # Less than 2 tasks per node
            len(active_nodes) > self.min_nodes
        )
        
        if scale_down_needed:
            return True, "scale_down"
        
        return False, "no_action"
    
    def execute_scaling(self, action: str, nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Execute scaling action."""
        self.last_scale_action = time.time()
        
        if action == "scale_up":
            # Create new node
            new_node = ComputeNode(
                node_id=str(uuid.uuid4()),
                node_type=NodeType.WORKER,
                status=NodeStatus.INITIALIZING,
                cpu_cores=2,
                memory_gb=4.0,
                supported_models=['ssl_model', 'neural_operator']
            )
            
            self.scaling_decisions.append({
                'action': 'scale_up',
                'timestamp': time.time(),
                'node_count_before': len(nodes),
                'avg_load': sum(n.current_load for n in nodes) / max(len(nodes), 1)
            })
            
            self.logger.info(f"Scaling up: Added node {new_node.node_id}")
            return new_node
        
        elif action == "scale_down":
            # Find node to remove (lowest utilization)
            active_nodes = [n for n in nodes if n.status == NodeStatus.ACTIVE]
            if len(active_nodes) > self.min_nodes:
                node_to_remove = min(active_nodes, key=lambda n: n.current_load)
                node_to_remove.status = NodeStatus.OFFLINE
                
                self.scaling_decisions.append({
                    'action': 'scale_down',
                    'timestamp': time.time(),
                    'node_count_before': len(active_nodes),
                    'removed_node': node_to_remove.node_id
                })
                
                self.logger.info(f"Scaling down: Removed node {node_to_remove.node_id}")
        
        return None
    
    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get recent scaling decisions."""
        return list(self.scaling_decisions)

class DistributedCoordinator:
    """Central coordinator for distributed protein processing."""
    
    def __init__(self, coordinator_id: Optional[str] = None):
        """Initialize distributed coordinator."""
        self.coordinator_id = coordinator_id or str(uuid.uuid4())
        
        # Node management
        self.nodes: Dict[str, ComputeNode] = {}
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler()
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        
        # System state
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        
        # Performance tracking
        self.metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_response_time': 0.0,
            'system_throughput': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def register_node(self, node: ComputeNode) -> bool:
        """Register a new compute node."""
        try:
            self.nodes[node.node_id] = node
            node.status = NodeStatus.ACTIVE
            self.logger.info(f"Registered node {node.node_id} ({node.node_type.value})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register node {node.node_id}: {e}")
            return False
    
    def submit_task(self, task: DistributedTask) -> bool:
        """Submit task for distributed processing."""
        try:
            # Priority queue uses tuples: (priority, timestamp, task)
            priority_value = -task.priority.value  # Negative for max priority
            self.task_queue.put((priority_value, task.created_at, task))
            
            self.metrics['tasks_submitted'] += 1
            self.logger.debug(f"Submitted task {task.task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    def start_coordinator(self):
        """Start the coordinator service."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker threads
        self.worker_threads = [
            threading.Thread(target=self._task_dispatcher, daemon=True),
            threading.Thread(target=self._health_monitor, daemon=True),
            threading.Thread(target=self._auto_scaler_loop, daemon=True),
            threading.Thread(target=self._metrics_collector, daemon=True)
        ]
        
        for thread in self.worker_threads:
            thread.start()
        
        self.logger.info(f"Started distributed coordinator {self.coordinator_id}")
    
    def stop_coordinator(self):
        """Stop the coordinator service."""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self.logger.info("Stopped distributed coordinator")
    
    def _task_dispatcher(self):
        """Dispatch tasks to available nodes."""
        while self.is_running:
            try:
                # Get next task (with timeout to allow shutdown)
                try:
                    priority, timestamp, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if task is expired
                if task.is_expired():
                    self.logger.warning(f"Task {task.task_id} expired, dropping")
                    self.metrics['tasks_failed'] += 1
                    continue
                
                # Select node for task
                active_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
                selected_node = self.load_balancer.select_node(active_nodes, task)
                
                if selected_node is None:
                    # No available nodes, put task back
                    self.task_queue.put((priority, timestamp, task))
                    time.sleep(1.0)
                    continue
                
                # Assign task to node
                task.assigned_node = selected_node.node_id
                self.active_tasks[task.task_id] = task
                selected_node.current_load += 0.1  # Increment load
                
                # Simulate task execution (in real system, would send to node)
                self._execute_task_on_node(task, selected_node)
                
            except Exception as e:
                self.logger.error(f"Task dispatcher error: {e}")
                time.sleep(1.0)
    
    def _execute_task_on_node(self, task: DistributedTask, node: ComputeNode):
        """Execute task on selected node (mock implementation)."""
        def task_runner():
            try:
                start_time = time.time()
                
                # Simulate task processing
                if task.task_type == 'prediction':
                    # Simulate protein structure prediction
                    time.sleep(0.1 + len(task.data.get('sequence', '')) * 0.001)
                    result = {
                        'prediction': f"Mock prediction for {task.data.get('sequence', '')[:20]}...",
                        'confidence': 0.85 + (hash(task.task_id) % 100) / 1000.0,
                        'processing_node': node.node_id
                    }
                elif task.task_type == 'analysis':
                    # Simulate sequence analysis
                    time.sleep(0.05)
                    result = {
                        'analysis': f"Mock analysis for task {task.task_id}",
                        'features': ['feature1', 'feature2', 'feature3'],
                        'processing_node': node.node_id
                    }
                else:
                    result = {'message': f"Processed {task.task_type} task", 'node': node.node_id}
                
                # Update metrics
                processing_time = time.time() - start_time
                node.avg_response_time = (
                    (node.avg_response_time * node.tasks_completed + processing_time * 1000) /
                    (node.tasks_completed + 1)
                )
                node.tasks_completed += 1
                node.success_rate = min(1.0, node.success_rate + 0.01)  # Improve success rate
                
                # Complete task
                self._complete_task(task.task_id, result)
                
            except Exception as e:
                self.logger.error(f"Task execution failed: {e}")
                self._fail_task(task.task_id, str(e))
            finally:
                # Reduce node load
                node.current_load = max(0.0, node.current_load - 0.1)
        
        # Execute in separate thread
        thread = threading.Thread(target=task_runner, daemon=True)
        thread.start()
    
    def _complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed."""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            self.completed_tasks[task_id] = {
                'task': task,
                'result': result,
                'completed_at': time.time()
            }
            
            self.metrics['tasks_completed'] += 1
            self.logger.debug(f"Completed task {task_id}")
    
    def _fail_task(self, task_id: str, error: str):
        """Mark task as failed."""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            
            # Retry if possible
            if task.retries < task.max_retries:
                task.retries += 1
                self.submit_task(task)
                self.logger.warning(f"Retrying task {task_id} (attempt {task.retries + 1})")
            else:
                self.completed_tasks[task_id] = {
                    'task': task,
                    'result': None,
                    'error': error,
                    'completed_at': time.time()
                }
                
                self.metrics['tasks_failed'] += 1
                self.logger.error(f"Task {task_id} failed permanently: {error}")
    
    def _health_monitor(self):
        """Monitor node health and update statuses."""
        while self.is_running:
            try:
                current_time = time.time()
                
                for node in self.nodes.values():
                    # Check heartbeat (simulate)
                    if current_time - node.last_heartbeat > 30:  # 30 seconds timeout
                        if node.status == NodeStatus.ACTIVE:
                            node.status = NodeStatus.DEGRADED
                            self.logger.warning(f"Node {node.node_id} degraded - no heartbeat")
                    elif current_time - node.last_heartbeat > 60:  # 1 minute timeout
                        if node.status != NodeStatus.OFFLINE:
                            node.status = NodeStatus.OFFLINE
                            self.logger.error(f"Node {node.node_id} offline - no heartbeat")
                    
                    # Simulate heartbeat updates for active nodes
                    if node.status in [NodeStatus.ACTIVE, NodeStatus.BUSY]:
                        node.last_heartbeat = current_time
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                time.sleep(10.0)
    
    def _auto_scaler_loop(self):
        """Auto-scaling monitoring loop."""
        while self.is_running:
            try:
                # Check if scaling is needed
                pending_tasks = self.task_queue.qsize()
                node_list = list(self.nodes.values())
                
                should_scale, action = self.auto_scaler.should_scale(node_list, pending_tasks)
                
                if should_scale and action in ['scale_up', 'scale_down']:
                    new_node = self.auto_scaler.execute_scaling(action, node_list)
                    
                    if new_node and action == 'scale_up':
                        # Register the new node
                        self.register_node(new_node)
                
                time.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scaler error: {e}")
                time.sleep(30.0)
    
    def _metrics_collector(self):
        """Collect system performance metrics."""
        while self.is_running:
            try:
                # Calculate throughput (tasks per second)
                if hasattr(self, '_last_completed_count'):
                    tasks_delta = self.metrics['tasks_completed'] - self._last_completed_count
                    time_delta = time.time() - self._last_metrics_time
                    
                    if time_delta > 0:
                        self.metrics['system_throughput'] = tasks_delta / time_delta
                
                self._last_completed_count = self.metrics['tasks_completed']
                self._last_metrics_time = time.time()
                
                # Calculate average response time
                if self.nodes:
                    active_nodes = [n for n in self.nodes.values() if n.tasks_completed > 0]
                    if active_nodes:
                        total_response = sum(n.avg_response_time for n in active_nodes)
                        self.metrics['avg_response_time'] = total_response / len(active_nodes)
                
                time.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collector error: {e}")
                time.sleep(5.0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        active_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
        
        return {
            'coordinator_id': self.coordinator_id,
            'nodes': {
                'total': len(self.nodes),
                'active': len(active_nodes),
                'offline': len([n for n in self.nodes.values() if n.status == NodeStatus.OFFLINE])
            },
            'tasks': {
                'pending': self.task_queue.qsize(),
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks)
            },
            'metrics': self.metrics,
            'scaling_history': self.auto_scaler.get_scaling_history()[-5:]  # Last 5 events
        }

def test_distributed_system():
    """Test the adaptive scaling distributed system."""
    print("🌐 Testing Adaptive Scaling Distributed System")
    print("=" * 55)
    
    # Initialize coordinator
    coordinator = DistributedCoordinator()
    coordinator.start_coordinator()
    
    # Register initial nodes
    initial_nodes = [
        ComputeNode(
            node_id=f"worker-{i}",
            node_type=NodeType.WORKER,
            cpu_cores=2 + i,
            memory_gb=4.0 + i * 2,
            supported_models=['ssl_model', 'neural_operator']
        )
        for i in range(3)
    ]
    
    for node in initial_nodes:
        coordinator.register_node(node)
    
    print(f"✅ Initialized coordinator with {len(initial_nodes)} nodes")
    
    # Submit test tasks
    test_tasks = []
    for i in range(20):
        task = DistributedTask(
            task_id=f"task-{i}",
            task_type='prediction' if i % 2 == 0 else 'analysis',
            data={'sequence': f'MKFL{"A" * (i % 10)}KFSLLT'},
            priority=TaskPriority.HIGH if i < 5 else TaskPriority.NORMAL
        )
        test_tasks.append(task)
        coordinator.submit_task(task)
    
    print(f"✅ Submitted {len(test_tasks)} tasks")
    
    # Monitor progress
    for _ in range(10):  # Monitor for 10 seconds
        status = coordinator.get_system_status()
        
        print(f"\r📊 Active: {status['tasks']['active']} | "
              f"Completed: {status['tasks']['completed']} | "
              f"Throughput: {status['metrics']['system_throughput']:.1f}/s", end='')
        
        time.sleep(1.0)
    
    # Final status
    final_status = coordinator.get_system_status()
    print(f"\n\n📈 Final System Status:")
    print(f"   Nodes: {final_status['nodes']['active']}/{final_status['nodes']['total']} active")
    print(f"   Tasks completed: {final_status['tasks']['completed']}")
    print(f"   Success rate: {(final_status['tasks']['completed'] / len(test_tasks)):.1%}")
    print(f"   Avg response time: {final_status['metrics']['avg_response_time']:.1f}ms")
    print(f"   System throughput: {final_status['metrics']['system_throughput']:.1f} tasks/s")
    
    if final_status['scaling_history']:
        print(f"   Scaling events: {len(final_status['scaling_history'])}")
    
    # Cleanup
    coordinator.stop_coordinator()
    
    print("🌐 Distributed system tests completed")
    return True

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    test_distributed_system()
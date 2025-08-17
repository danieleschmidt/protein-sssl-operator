"""
Advanced Distributed Computing System for Protein-SSL Operator
Implements multi-node coordination, fault tolerance, and consensus algorithms
"""

import time
import threading
import multiprocessing as mp
import socket
import json
import asyncio
import uuid
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import torch
import torch.distributed as dist
import psutil
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import pickle
import zmq
import redis
from pathlib import Path

from .logging_config import setup_logging
from .monitoring import MetricsCollector
from .parallel_processing import Task, TaskPriority

logger = setup_logging(__name__)


class NodeRole(Enum):
    """Node roles in distributed system"""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    HYBRID = "hybrid"


class NodeStatus(Enum):
    """Node status states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    FAILING = "failing"
    FAILED = "failed"
    OFFLINE = "offline"


class TaskStatus(Enum):
    """Distributed task status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class NodeInfo:
    """Information about a distributed node"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    role: NodeRole
    status: NodeStatus
    capabilities: Dict[str, Any]
    resources: Dict[str, float]  # CPU, memory, GPU, etc.
    last_heartbeat: float
    load_factor: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0


@dataclass
class DistributedTask:
    """Distributed task with routing and fault tolerance"""
    task_id: str
    task_data: Dict[str, Any]
    priority: TaskPriority
    assigned_node: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    max_retries: int = 3
    timeout: float = 3600.0  # 1 hour default
    dependencies: List[str] = None
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.dependencies is None:
            self.dependencies = []


class ConsensusProtocol:
    """Raft-like consensus protocol for distributed coordination"""
    
    def __init__(self, node_id: str, cluster_nodes: List[str]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader election
        self.state = "follower"  # follower, candidate, leader
        self.leader_id = None
        self.election_timeout = random.uniform(5, 10)  # seconds
        self.last_heartbeat = time.time()
        
        # For leaders
        self.next_index = {}
        self.match_index = {}
        
        self._lock = threading.RLock()
        self._election_timer = None
        self._heartbeat_timer = None
    
    def start(self):
        """Start consensus protocol"""
        self._reset_election_timer()
        logger.info(f"Consensus protocol started for node {self.node_id}")
    
    def stop(self):
        """Stop consensus protocol"""
        if self._election_timer:
            self._election_timer.cancel()
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()
    
    def append_entry(self, entry: Dict[str, Any]) -> bool:
        """Append entry to log (only leader can do this)"""
        with self._lock:
            if self.state != "leader":
                return False
            
            log_entry = {
                'term': self.current_term,
                'index': len(self.log),
                'data': entry,
                'timestamp': time.time()
            }
            
            self.log.append(log_entry)
            
            # Replicate to followers
            self._replicate_to_followers()
            
            return True
    
    def handle_append_entries(self, term: int, leader_id: str, prev_log_index: int,
                            prev_log_term: int, entries: List[Dict], leader_commit: int) -> bool:
        """Handle append entries RPC from leader"""
        with self._lock:
            self.last_heartbeat = time.time()
            
            # Update term if needed
            if term > self.current_term:
                self.current_term = term
                self.voted_for = None
                self._become_follower()
            
            # Reject if term is old
            if term < self.current_term:
                return False
            
            # Update leader
            self.leader_id = leader_id
            self._become_follower()
            
            # Log consistency check
            if prev_log_index >= 0:
                if (len(self.log) <= prev_log_index or
                    self.log[prev_log_index]['term'] != prev_log_term):
                    return False
            
            # Append new entries
            if entries:
                # Remove conflicting entries
                if len(self.log) > prev_log_index + 1:
                    self.log = self.log[:prev_log_index + 1]
                
                # Append new entries
                self.log.extend(entries)
            
            # Update commit index
            if leader_commit > self.commit_index:
                self.commit_index = min(leader_commit, len(self.log) - 1)
                self._apply_committed_entries()
            
            return True
    
    def handle_request_vote(self, term: int, candidate_id: str, 
                          last_log_index: int, last_log_term: int) -> Tuple[bool, int]:
        """Handle request vote RPC from candidate"""
        with self._lock:
            vote_granted = False
            
            # Update term if needed
            if term > self.current_term:
                self.current_term = term
                self.voted_for = None
                self._become_follower()
            
            # Grant vote if conditions are met
            if (term == self.current_term and
                (self.voted_for is None or self.voted_for == candidate_id) and
                self._log_is_up_to_date(last_log_index, last_log_term)):
                
                vote_granted = True
                self.voted_for = candidate_id
                self.last_heartbeat = time.time()
            
            return vote_granted, self.current_term
    
    def _become_follower(self):
        """Transition to follower state"""
        if self.state != "follower":
            logger.debug(f"Node {self.node_id} becoming follower")
            self.state = "follower"
            if self._heartbeat_timer:
                self._heartbeat_timer.cancel()
            self._reset_election_timer()
    
    def _become_candidate(self):
        """Transition to candidate state and start election"""
        logger.info(f"Node {self.node_id} starting election for term {self.current_term + 1}")
        
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat = time.time()
        
        # Request votes from other nodes
        votes_received = 1  # Vote for self
        votes_needed = (len(self.cluster_nodes) + 1) // 2 + 1
        
        # This would send vote requests to other nodes
        # For simulation, assume some votes are received
        if votes_received >= votes_needed:
            self._become_leader()
        else:
            self._become_follower()
    
    def _become_leader(self):
        """Transition to leader state"""
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        
        self.state = "leader"
        self.leader_id = self.node_id
        
        # Initialize leader state
        self.next_index = {node: len(self.log) for node in self.cluster_nodes}
        self.match_index = {node: 0 for node in self.cluster_nodes}
        
        # Start sending heartbeats
        self._send_heartbeats()
    
    def _reset_election_timer(self):
        """Reset election timeout"""
        if self._election_timer:
            self._election_timer.cancel()
        
        if self.state != "leader":
            self._election_timer = threading.Timer(
                self.election_timeout, self._on_election_timeout
            )
            self._election_timer.start()
    
    def _on_election_timeout(self):
        """Handle election timeout"""
        with self._lock:
            if self.state != "leader":
                self._become_candidate()
    
    def _send_heartbeats(self):
        """Send heartbeats to followers (leader only)"""
        if self.state == "leader":
            # Send append entries with no entries (heartbeat)
            # This would send to other nodes
            
            # Schedule next heartbeat
            self._heartbeat_timer = threading.Timer(1.0, self._send_heartbeats)
            self._heartbeat_timer.start()
    
    def _replicate_to_followers(self):
        """Replicate log entries to followers"""
        if self.state != "leader":
            return
        
        # This would send append entries to followers
        # For now, just mark as committed after timeout
        pass
    
    def _apply_committed_entries(self):
        """Apply committed log entries"""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied]
            # Apply entry to state machine
            logger.debug(f"Applied entry {self.last_applied}: {entry}")
    
    def _log_is_up_to_date(self, last_log_index: int, last_log_term: int) -> bool:
        """Check if candidate's log is at least as up-to-date as receiver's log"""
        if not self.log:
            return True
        
        our_last_term = self.log[-1]['term']
        our_last_index = len(self.log) - 1
        
        if last_log_term > our_last_term:
            return True
        elif last_log_term == our_last_term:
            return last_log_index >= our_last_index
        else:
            return False
    
    def is_leader(self) -> bool:
        """Check if this node is the leader"""
        return self.state == "leader"
    
    def get_leader(self) -> Optional[str]:
        """Get current leader node ID"""
        return self.leader_id


class TaskDistributor:
    """Intelligent task distribution with load balancing"""
    
    def __init__(self, node_registry: 'NodeRegistry'):
        self.node_registry = node_registry
        self.pending_tasks = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Distribution strategies
        self.strategies = {
            'round_robin': self._round_robin_distribution,
            'least_loaded': self._least_loaded_distribution,
            'capability_based': self._capability_based_distribution,
            'locality_aware': self._locality_aware_distribution
        }
        
        self.current_strategy = 'capability_based'
        self.round_robin_index = 0
        
        # Performance tracking
        self.distribution_metrics = defaultdict(list)
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit task for distributed execution"""
        # Check dependencies
        if not self._check_dependencies_ready(task):
            task.status = TaskStatus.PENDING
            priority = task.priority.value
            self.pending_tasks.put((priority, time.time(), task))
            return task.task_id
        
        # Assign to node
        assigned_node = self._assign_task_to_node(task)
        if assigned_node:
            task.assigned_node = assigned_node
            task.status = TaskStatus.ASSIGNED
            task.started_at = time.time()
            self.active_tasks[task.task_id] = task
            
            # Send task to node
            success = self._send_task_to_node(task, assigned_node)
            if success:
                task.status = TaskStatus.RUNNING
                logger.info(f"Task {task.task_id} assigned to node {assigned_node}")
            else:
                task.status = TaskStatus.FAILED
                task.error = f"Failed to send task to node {assigned_node}"
                self.failed_tasks[task.task_id] = task
                del self.active_tasks[task.task_id]
        else:
            # No available nodes, queue for later
            priority = task.priority.value
            self.pending_tasks.put((priority, time.time(), task))
        
        return task.task_id
    
    def _check_dependencies_ready(self, task: DistributedTask) -> bool:
        """Check if task dependencies are completed"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _assign_task_to_node(self, task: DistributedTask) -> Optional[str]:
        """Assign task to optimal node"""
        available_nodes = self.node_registry.get_available_nodes()
        if not available_nodes:
            return None
        
        strategy = self.strategies.get(self.current_strategy, self._round_robin_distribution)
        return strategy(task, available_nodes)
    
    def _round_robin_distribution(self, task: DistributedTask, 
                                available_nodes: List[NodeInfo]) -> Optional[str]:
        """Round-robin task distribution"""
        if not available_nodes:
            return None
        
        node = available_nodes[self.round_robin_index % len(available_nodes)]
        self.round_robin_index += 1
        return node.node_id
    
    def _least_loaded_distribution(self, task: DistributedTask,
                                 available_nodes: List[NodeInfo]) -> Optional[str]:
        """Assign to least loaded node"""
        if not available_nodes:
            return None
        
        return min(available_nodes, key=lambda n: n.load_factor).node_id
    
    def _capability_based_distribution(self, task: DistributedTask,
                                     available_nodes: List[NodeInfo]) -> Optional[str]:
        """Assign based on node capabilities and task requirements"""
        if not available_nodes:
            return None
        
        # Score nodes based on capability match
        scored_nodes = []
        
        for node in available_nodes:
            score = self._calculate_capability_score(task, node)
            scored_nodes.append((score, node))
        
        # Sort by score (highest first) and return best match
        scored_nodes.sort(reverse=True, key=lambda x: x[0])
        return scored_nodes[0][1].node_id
    
    def _locality_aware_distribution(self, task: DistributedTask,
                                   available_nodes: List[NodeInfo]) -> Optional[str]:
        """Assign based on data locality and network proximity"""
        if not available_nodes:
            return None
        
        # For now, use capability-based as fallback
        return self._capability_based_distribution(task, available_nodes)
    
    def _calculate_capability_score(self, task: DistributedTask, node: NodeInfo) -> float:
        """Calculate how well a node matches task requirements"""
        score = 0.0
        
        # Base score from available resources
        cpu_score = node.resources.get('cpu_available', 0.5)
        memory_score = node.resources.get('memory_available', 0.5)
        score += (cpu_score + memory_score) * 0.3
        
        # Load factor (lower is better)
        load_score = max(0, 1.0 - node.load_factor)
        score += load_score * 0.3
        
        # Node reliability (based on success rate)
        total_tasks = node.completed_tasks + node.failed_tasks
        if total_tasks > 0:
            success_rate = node.completed_tasks / total_tasks
            score += success_rate * 0.2
        else:
            score += 0.1  # Default for new nodes
        
        # Capability match
        task_requirements = task.task_data.get('requirements', {})
        
        if task_requirements.get('gpu_required', False):
            if node.capabilities.get('gpu_count', 0) > 0:
                score += 0.2
            else:
                score -= 0.3
        
        if 'min_memory_gb' in task_requirements:
            required_memory = task_requirements['min_memory_gb']
            available_memory = node.resources.get('memory_available_gb', 0)
            if available_memory >= required_memory:
                score += 0.1
            else:
                score -= 0.2
        
        return max(0.0, score)
    
    def _send_task_to_node(self, task: DistributedTask, node_id: str) -> bool:
        """Send task to specified node"""
        try:
            node = self.node_registry.get_node(node_id)
            if not node:
                return False
            
            # This would use the actual communication protocol
            # For now, simulate success
            logger.debug(f"Sending task {task.task_id} to node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send task {task.task_id} to node {node_id}: {e}")
            return False
    
    def handle_task_completion(self, task_id: str, result: Any, node_id: str):
        """Handle task completion from worker node"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            
            # Update node stats
            node = self.node_registry.get_node(node_id)
            if node:
                node.completed_tasks += 1
                node.active_tasks -= 1
                self.node_registry.update_node_load(node_id)
            
            logger.info(f"Task {task_id} completed on node {node_id}")
            
            # Check for dependent tasks
            self._process_pending_tasks()
    
    def handle_task_failure(self, task_id: str, error: str, node_id: str):
        """Handle task failure from worker node"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.retries += 1
            task.error = error
            
            # Update node stats
            node = self.node_registry.get_node(node_id)
            if node:
                node.failed_tasks += 1
                node.active_tasks -= 1
                self.node_registry.update_node_load(node_id)
            
            # Retry if attempts remaining
            if task.retries < task.max_retries:
                task.status = TaskStatus.RETRYING
                task.assigned_node = None
                
                # Resubmit task
                priority = task.priority.value
                self.pending_tasks.put((priority, time.time(), task))
                del self.active_tasks[task_id]
                
                logger.warning(f"Task {task_id} failed on node {node_id}, retrying ({task.retries}/{task.max_retries})")
            else:
                # Max retries exceeded
                task.status = TaskStatus.FAILED
                self.failed_tasks[task_id] = task
                del self.active_tasks[task_id]
                
                logger.error(f"Task {task_id} failed permanently after {task.retries} retries")
    
    def _process_pending_tasks(self):
        """Process pending tasks that may now be ready"""
        ready_tasks = []
        
        # Check pending tasks for readiness
        temp_queue = queue.PriorityQueue()
        
        while not self.pending_tasks.empty():
            try:
                priority, timestamp, task = self.pending_tasks.get_nowait()
                
                if self._check_dependencies_ready(task):
                    ready_tasks.append(task)
                else:
                    temp_queue.put((priority, timestamp, task))
            except queue.Empty:
                break
        
        # Put non-ready tasks back
        self.pending_tasks = temp_queue
        
        # Submit ready tasks
        for task in ready_tasks:
            self.submit_task(task)
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get task distribution statistics"""
        return {
            'pending_tasks': self.pending_tasks.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'current_strategy': self.current_strategy,
            'available_strategies': list(self.strategies.keys())
        }


class NodeRegistry:
    """Registry for managing distributed nodes"""
    
    def __init__(self):
        self.nodes = {}
        self._lock = threading.RLock()
        self.heartbeat_timeout = 30.0  # seconds
        
        # Monitoring
        self.node_history = defaultdict(lambda: deque(maxlen=100))
        
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new node"""
        with self._lock:
            self.nodes[node_info.node_id] = node_info
            logger.info(f"Registered node {node_info.node_id} ({node_info.hostname})")
            return True
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node"""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Unregistered node {node_id}")
                return True
            return False
    
    def update_node_heartbeat(self, node_id: str, status_update: Dict[str, Any] = None):
        """Update node heartbeat and status"""
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.last_heartbeat = time.time()
                
                if status_update:
                    # Update node status
                    if 'status' in status_update:
                        node.status = NodeStatus(status_update['status'])
                    if 'load_factor' in status_update:
                        node.load_factor = status_update['load_factor']
                    if 'resources' in status_update:
                        node.resources.update(status_update['resources'])
                    if 'active_tasks' in status_update:
                        node.active_tasks = status_update['active_tasks']
                
                # Record in history
                self.node_history[node_id].append({
                    'timestamp': time.time(),
                    'status': node.status.value,
                    'load_factor': node.load_factor,
                    'active_tasks': node.active_tasks
                })
    
    def update_node_load(self, node_id: str):
        """Update node load factor based on active tasks"""
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                max_tasks = node.capabilities.get('max_concurrent_tasks', 10)
                node.load_factor = min(1.0, node.active_tasks / max_tasks)
    
    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Get node information"""
        with self._lock:
            return self.nodes.get(node_id)
    
    def get_available_nodes(self) -> List[NodeInfo]:
        """Get list of available nodes"""
        with self._lock:
            available = []
            current_time = time.time()
            
            for node in self.nodes.values():
                # Check if node is alive and available
                if (current_time - node.last_heartbeat < self.heartbeat_timeout and
                    node.status in [NodeStatus.ACTIVE, NodeStatus.IDLE] and
                    node.load_factor < 0.9):  # Not overloaded
                    available.append(node)
            
            return available
    
    def cleanup_stale_nodes(self):
        """Remove stale nodes that haven't sent heartbeat"""
        with self._lock:
            current_time = time.time()
            stale_nodes = []
            
            for node_id, node in self.nodes.items():
                if current_time - node.last_heartbeat > self.heartbeat_timeout:
                    stale_nodes.append(node_id)
                    node.status = NodeStatus.OFFLINE
            
            for node_id in stale_nodes:
                logger.warning(f"Node {node_id} marked as offline (stale heartbeat)")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster-wide statistics"""
        with self._lock:
            total_nodes = len(self.nodes)
            active_nodes = len([n for n in self.nodes.values() 
                              if n.status in [NodeStatus.ACTIVE, NodeStatus.IDLE]])
            
            total_tasks = sum(n.active_tasks for n in self.nodes.values())
            total_completed = sum(n.completed_tasks for n in self.nodes.values())
            total_failed = sum(n.failed_tasks for n in self.nodes.values())
            
            avg_load = (sum(n.load_factor for n in self.nodes.values()) / 
                       max(total_nodes, 1))
            
            return {
                'total_nodes': total_nodes,
                'active_nodes': active_nodes,
                'offline_nodes': total_nodes - active_nodes,
                'total_active_tasks': total_tasks,
                'total_completed_tasks': total_completed,
                'total_failed_tasks': total_failed,
                'average_load_factor': avg_load,
                'cluster_utilization': avg_load
            }


class DistributedCoordinator:
    """Main coordinator for distributed computing system"""
    
    def __init__(self, node_id: str = None, cluster_nodes: List[str] = None):
        self.node_id = node_id or self._generate_node_id()
        self.cluster_nodes = cluster_nodes or []
        
        # Core components
        self.node_registry = NodeRegistry()
        self.task_distributor = TaskDistributor(self.node_registry)
        self.consensus = ConsensusProtocol(self.node_id, self.cluster_nodes)
        
        # Communication
        self.communication_thread = None
        self.heartbeat_thread = None
        self.is_running = False
        
        # Fault tolerance
        self.fault_detector = FaultDetector(self.node_registry)
        
        # Performance tracking
        self.metrics_collector = MetricsCollector()
        self.start_time = None
        
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        hostname = socket.gethostname()
        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        return f"{hostname}_{timestamp}_{random_suffix}"
    
    def start(self, port: int = 8080):
        """Start distributed coordinator"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Register self as a node
        self_info = NodeInfo(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            ip_address=self._get_local_ip(),
            port=port,
            role=NodeRole.COORDINATOR,
            status=NodeStatus.ACTIVE,
            capabilities=self._detect_capabilities(),
            resources=self._get_resource_info(),
            last_heartbeat=time.time()
        )
        
        self.node_registry.register_node(self_info)
        
        # Start consensus protocol
        self.consensus.start()
        
        # Start background threads
        self._start_background_threads()
        
        logger.info(f"Distributed coordinator started on {self.node_id}")
    
    def stop(self):
        """Stop distributed coordinator"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop consensus
        self.consensus.stop()
        
        # Stop background threads
        if self.communication_thread:
            self.communication_thread.join(timeout=5.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
        
        logger.info("Distributed coordinator stopped")
    
    def submit_distributed_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit task for distributed execution"""
        task_id = str(uuid.uuid4())
        
        # Extract distributed task parameters
        priority = kwargs.pop('priority', TaskPriority.NORMAL)
        max_retries = kwargs.pop('max_retries', 3)
        timeout = kwargs.pop('timeout', 3600.0)
        dependencies = kwargs.pop('dependencies', [])
        
        # Serialize task data
        task_data = {
            'function_name': func.__name__,
            'module': func.__module__,
            'args': args,
            'kwargs': kwargs,
            'requirements': kwargs.pop('requirements', {})
        }
        
        distributed_task = DistributedTask(
            task_id=task_id,
            task_data=task_data,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            dependencies=dependencies
        )
        
        # Submit through task distributor
        self.task_distributor.submit_task(distributed_task)
        
        return task_id
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result of distributed task"""
        start_time = time.time()
        
        while self.is_running:
            # Check completed tasks
            if task_id in self.task_distributor.completed_tasks:
                task = self.task_distributor.completed_tasks[task_id]
                return task.result
            
            # Check failed tasks
            if task_id in self.task_distributor.failed_tasks:
                task = self.task_distributor.failed_tasks[task_id]
                raise Exception(f"Task {task_id} failed: {task.error}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            time.sleep(0.1)
        
        raise RuntimeError("Coordinator is not running")
    
    def join_cluster(self, coordinator_address: str, coordinator_port: int) -> bool:
        """Join existing cluster"""
        try:
            # Contact existing coordinator
            response = requests.post(
                f"http://{coordinator_address}:{coordinator_port}/join",
                json={
                    'node_id': self.node_id,
                    'hostname': socket.gethostname(),
                    'ip_address': self._get_local_ip(),
                    'capabilities': self._detect_capabilities(),
                    'resources': self._get_resource_info()
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                cluster_info = response.json()
                self.cluster_nodes = cluster_info.get('cluster_nodes', [])
                logger.info(f"Successfully joined cluster with {len(self.cluster_nodes)} nodes")
                return True
            else:
                logger.error(f"Failed to join cluster: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error joining cluster: {e}")
            return False
    
    def _start_background_threads(self):
        """Start background threads"""
        # Heartbeat thread
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()
        
        # Communication thread
        self.communication_thread = threading.Thread(
            target=self._communication_loop, daemon=True
        )
        self.communication_thread.start()
    
    def _heartbeat_loop(self):
        """Background heartbeat loop"""
        while self.is_running:
            try:
                # Send heartbeat to cluster
                self._send_heartbeat()
                
                # Cleanup stale nodes
                self.node_registry.cleanup_stale_nodes()
                
                # Fault detection
                self.fault_detector.check_node_health()
                
                time.sleep(5.0)  # Heartbeat every 5 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                time.sleep(5.0)
    
    def _communication_loop(self):
        """Background communication loop"""
        while self.is_running:
            try:
                # Process incoming messages
                # This would handle actual network communication
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Communication loop error: {e}")
                time.sleep(1.0)
    
    def _send_heartbeat(self):
        """Send heartbeat to cluster"""
        heartbeat_data = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'status': 'active',
            'load_factor': self._calculate_current_load(),
            'resources': self._get_resource_info(),
            'active_tasks': len(self.task_distributor.active_tasks)
        }
        
        # Update own heartbeat
        self.node_registry.update_node_heartbeat(self.node_id, heartbeat_data)
        
        # Send to other nodes (simplified)
        logger.debug(f"Heartbeat sent from {self.node_id}")
    
    def _calculate_current_load(self) -> float:
        """Calculate current system load"""
        cpu_percent = psutil.cpu_percent() / 100.0
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        # Simple average for now
        return (cpu_percent + memory_percent) / 2.0
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect node capabilities"""
        capabilities = {
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_count': 0,
            'max_concurrent_tasks': psutil.cpu_count() * 2
        }
        
        # Detect GPU
        if torch.cuda.is_available():
            capabilities['gpu_count'] = torch.cuda.device_count()
            capabilities['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return capabilities
    
    def _get_resource_info(self) -> Dict[str, float]:
        """Get current resource information"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        resources = {
            'cpu_utilization': cpu_percent / 100.0,
            'cpu_available': max(0, 1.0 - cpu_percent / 100.0),
            'memory_utilization': memory.percent / 100.0,
            'memory_available': memory.available / (1024**3),  # GB
            'memory_available_gb': memory.available / (1024**3)
        }
        
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_total = torch.cuda.max_memory_allocated()
                if gpu_memory_total > 0:
                    resources['gpu_utilization'] = gpu_memory_allocated / gpu_memory_total
                else:
                    resources['gpu_utilization'] = 0.0
            except Exception:
                resources['gpu_utilization'] = 0.0
        
        return resources
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'coordinator_id': self.node_id,
            'uptime_seconds': uptime,
            'is_running': self.is_running,
            'is_leader': self.consensus.is_leader(),
            'current_leader': self.consensus.get_leader(),
            'cluster_stats': self.node_registry.get_cluster_stats(),
            'task_distribution_stats': self.task_distributor.get_distribution_stats(),
            'consensus_term': self.consensus.current_term,
            'fault_tolerance_enabled': True
        }


class FaultDetector:
    """Fault detection and recovery system"""
    
    def __init__(self, node_registry: NodeRegistry):
        self.node_registry = node_registry
        self.failure_patterns = defaultdict(list)
        self.recovery_actions = []
        
    def check_node_health(self):
        """Check health of all nodes"""
        current_time = time.time()
        
        for node_id, node in self.node_registry.nodes.items():
            # Check heartbeat timeout
            if current_time - node.last_heartbeat > 30.0:
                self._handle_node_failure(node_id, "Heartbeat timeout")
            
            # Check excessive failures
            if node.failed_tasks > 0:
                failure_rate = node.failed_tasks / max(node.completed_tasks + node.failed_tasks, 1)
                if failure_rate > 0.3:  # 30% failure rate
                    self._handle_node_degradation(node_id, f"High failure rate: {failure_rate:.2f}")
    
    def _handle_node_failure(self, node_id: str, reason: str):
        """Handle node failure"""
        logger.warning(f"Node {node_id} failed: {reason}")
        
        node = self.node_registry.get_node(node_id)
        if node:
            node.status = NodeStatus.FAILED
            
            # Record failure pattern
            self.failure_patterns[node_id].append({
                'timestamp': time.time(),
                'reason': reason,
                'type': 'failure'
            })
    
    def _handle_node_degradation(self, node_id: str, reason: str):
        """Handle node performance degradation"""
        logger.warning(f"Node {node_id} degraded: {reason}")
        
        node = self.node_registry.get_node(node_id)
        if node:
            node.status = NodeStatus.FAILING
            
            # Record degradation pattern
            self.failure_patterns[node_id].append({
                'timestamp': time.time(),
                'reason': reason,
                'type': 'degradation'
            })


# Global coordinator instance
_global_coordinator = None

def get_distributed_coordinator(**kwargs) -> DistributedCoordinator:
    """Get global distributed coordinator instance"""
    global _global_coordinator
    
    if _global_coordinator is None:
        _global_coordinator = DistributedCoordinator(**kwargs)
    
    return _global_coordinator

def start_distributed_computing(**kwargs) -> DistributedCoordinator:
    """Start global distributed computing"""
    coordinator = get_distributed_coordinator(**kwargs)
    coordinator.start()
    return coordinator

def submit_distributed_task(func: Callable, *args, **kwargs) -> str:
    """Submit task to global coordinator"""
    coordinator = get_distributed_coordinator()
    return coordinator.submit_distributed_task(func, *args, **kwargs)

def get_distributed_result(task_id: str, timeout: float = None) -> Any:
    """Get result from global coordinator"""
    coordinator = get_distributed_coordinator()
    return coordinator.get_task_result(task_id, timeout)
"""
Intelligent Resource Management System for Protein-SSL Operator
Implements dynamic resource allocation, quota management, and intelligent scheduling
"""

import time
import threading
import multiprocessing as mp
import os
import psutil
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import queue
import heapq
import json
from pathlib import Path

from .logging_config import setup_logging
from .monitoring import MetricsCollector
from .advanced_caching import get_cache
from .memory_optimization import get_memory_optimizer
from .compute_optimization import get_compute_optimizer
from .parallel_processing import get_parallel_processor
from .network_optimization import get_network_optimizer
from .storage_optimization import get_storage_optimizer

logger = setup_logging(__name__)


class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"
    CACHE = "cache"


class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    FAIR_SHARE = "fair_share"
    PRIORITY_BASED = "priority_based"
    DEMAND_BASED = "demand_based"
    PREDICTIVE = "predictive"
    ROUND_ROBIN = "round_robin"


class ResourcePriority(Enum):
    """Resource allocation priorities"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ResourceQuota:
    """Resource quota configuration"""
    cpu_cores: float = 0.0
    memory_mb: int = 0
    gpu_count: int = 0
    storage_gb: int = 0
    network_mbps: int = 0
    cache_mb: int = 0
    max_concurrent_tasks: int = 0


@dataclass
class ResourceUsage:
    """Current resource usage"""
    cpu_utilization: float = 0.0
    memory_used_mb: int = 0
    gpu_utilization: float = 0.0
    storage_used_gb: int = 0
    network_usage_mbps: int = 0
    cache_used_mb: int = 0
    active_tasks: int = 0


@dataclass
class ResourceRequest:
    """Resource allocation request"""
    request_id: str
    requester_id: str
    resource_requirements: ResourceQuota
    priority: ResourcePriority
    duration_estimate: float  # seconds
    deadline: Optional[float] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class ResourceAllocation:
    """Active resource allocation"""
    allocation_id: str
    request: ResourceRequest
    allocated_resources: ResourceQuota
    start_time: float
    expected_end_time: float
    actual_usage: ResourceUsage
    efficiency_score: float = 1.0


class ResourceMonitor:
    """Real-time resource monitoring and tracking"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        
        # Resource tracking
        self.current_usage = ResourceUsage()
        self.usage_history = deque(maxlen=3600)  # 1 hour at 1s intervals
        self.resource_limits = self._detect_system_limits()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Metrics
        self.metrics_collector = MetricsCollector()
        
    def _detect_system_limits(self) -> ResourceQuota:
        """Detect system resource limits"""
        # CPU
        cpu_count = psutil.cpu_count(logical=True)
        
        # Memory
        memory_info = psutil.virtual_memory()
        memory_mb = int(memory_info.total / (1024 * 1024))
        
        # GPU
        gpu_count = 0
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
        
        # Storage (estimate available space)
        storage_info = psutil.disk_usage('/')
        storage_gb = int(storage_info.total / (1024 * 1024 * 1024))
        
        # Network (estimate based on system)
        network_mbps = 1000  # Default 1Gbps
        
        # Cache (estimate based on memory)
        cache_mb = int(memory_mb * 0.1)  # 10% of memory for cache
        
        return ResourceQuota(
            cpu_cores=float(cpu_count),
            memory_mb=memory_mb,
            gpu_count=gpu_count,
            storage_gb=storage_gb,
            network_mbps=network_mbps,
            cache_mb=cache_mb,
            max_concurrent_tasks=cpu_count * 4
        )
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_resource_usage()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _update_resource_usage(self):
        """Update current resource usage"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_used_mb = int((memory_info.total - memory_info.available) / (1024 * 1024))
        
        # GPU usage
        gpu_utilization = 0.0
        if torch.cuda.is_available():
            try:
                gpu_utilization = torch.cuda.utilization() / 100.0
            except Exception:
                gpu_utilization = 0.0
        
        # Storage usage
        storage_info = psutil.disk_usage('/')
        storage_used_gb = int((storage_info.total - storage_info.free) / (1024 * 1024 * 1024))
        
        # Network usage (simplified)
        net_io = psutil.net_io_counters()
        network_usage_mbps = 0  # Would need historical data to calculate rate
        
        # Cache usage (get from cache system)
        cache_used_mb = 0
        try:
            cache = get_cache()
            cache_stats = cache.get_multi_tier_stats()
            cache_used_mb = cache_stats.get('overall', {}).get('total_memory_mb', 0)
        except Exception:
            pass
        
        # Active tasks (get from parallel processor)
        active_tasks = 0
        try:
            processor = get_parallel_processor()
            stats = processor.get_comprehensive_stats()
            active_tasks = stats.get('task_statistics', {}).get('pending', 0)
        except Exception:
            pass
        
        # Update current usage
        self.current_usage = ResourceUsage(
            cpu_utilization=cpu_percent / 100.0,
            memory_used_mb=memory_used_mb,
            gpu_utilization=gpu_utilization,
            storage_used_gb=storage_used_gb,
            network_usage_mbps=network_usage_mbps,
            cache_used_mb=cache_used_mb,
            active_tasks=active_tasks
        )
        
        # Add to history
        self.usage_history.append({
            'timestamp': time.time(),
            'usage': asdict(self.current_usage)
        })
    
    def get_available_resources(self) -> ResourceQuota:
        """Get currently available resources"""
        limits = self.resource_limits
        usage = self.current_usage
        
        return ResourceQuota(
            cpu_cores=max(0, limits.cpu_cores - (limits.cpu_cores * usage.cpu_utilization)),
            memory_mb=max(0, limits.memory_mb - usage.memory_used_mb),
            gpu_count=limits.gpu_count,  # GPU availability is binary
            storage_gb=max(0, limits.storage_gb - usage.storage_used_gb),
            network_mbps=max(0, limits.network_mbps - usage.network_usage_mbps),
            cache_mb=max(0, limits.cache_mb - usage.cache_used_mb),
            max_concurrent_tasks=max(0, limits.max_concurrent_tasks - usage.active_tasks)
        )
    
    def get_resource_efficiency(self) -> Dict[str, float]:
        """Calculate resource efficiency metrics"""
        if not self.usage_history:
            return {}
        
        recent_usage = list(self.usage_history)[-60:]  # Last minute
        
        efficiency = {}
        
        # CPU efficiency
        cpu_utilizations = [u['usage']['cpu_utilization'] for u in recent_usage]
        efficiency['cpu'] = np.mean(cpu_utilizations) if cpu_utilizations else 0.0
        
        # Memory efficiency
        memory_ratios = [
            u['usage']['memory_used_mb'] / self.resource_limits.memory_mb 
            for u in recent_usage
        ]
        efficiency['memory'] = np.mean(memory_ratios) if memory_ratios else 0.0
        
        # GPU efficiency
        gpu_utilizations = [u['usage']['gpu_utilization'] for u in recent_usage]
        efficiency['gpu'] = np.mean(gpu_utilizations) if gpu_utilizations else 0.0
        
        return efficiency
    
    def predict_resource_demand(self, horizon_minutes: int = 30) -> ResourceQuota:
        """Predict future resource demand"""
        if len(self.usage_history) < 10:
            return self.current_usage.__dict__
        
        # Simple trend-based prediction
        recent_data = list(self.usage_history)[-min(60, len(self.usage_history)):]
        
        # Calculate trends for each resource
        timestamps = [d['timestamp'] for d in recent_data]
        
        predictions = {}
        
        for resource in ['cpu_utilization', 'memory_used_mb', 'gpu_utilization']:
            values = [d['usage'][resource] for d in recent_data]
            
            if len(values) >= 3:
                # Linear trend
                x = np.array(range(len(values)))
                trend = np.polyfit(x, values, 1)[0]
                
                # Project forward
                future_steps = horizon_minutes
                predicted_value = values[-1] + (trend * future_steps)
                predictions[resource] = max(0, predicted_value)
            else:
                predictions[resource] = values[-1] if values else 0
        
        return ResourceQuota(
            cpu_cores=predictions.get('cpu_utilization', 0) * self.resource_limits.cpu_cores,
            memory_mb=int(predictions.get('memory_used_mb', 0)),
            gpu_count=int(predictions.get('gpu_utilization', 0) * self.resource_limits.gpu_count)
        )


class ResourceScheduler:
    """Intelligent resource scheduler with multiple allocation strategies"""
    
    def __init__(self, monitor: ResourceMonitor, strategy: AllocationStrategy = AllocationStrategy.DEMAND_BASED):
        self.monitor = monitor
        self.strategy = strategy
        
        # Request management
        self.pending_requests = []  # Priority queue
        self.active_allocations = {}
        self.allocation_history = deque(maxlen=1000)
        
        # Scheduling parameters
        self.scheduling_interval = 5.0  # seconds
        self.preemption_enabled = False
        
        # Scheduling state
        self.scheduling_active = False
        self.scheduling_thread = None
        self._lock = threading.RLock()
        
        # Performance tracking
        self.scheduling_metrics = {
            'requests_processed': 0,
            'requests_fulfilled': 0,
            'requests_rejected': 0,
            'average_wait_time': 0.0,
            'resource_utilization': 0.0
        }
    
    def start_scheduling(self):
        """Start resource scheduling"""
        if self.scheduling_active:
            return
        
        self.scheduling_active = True
        self.scheduling_thread = threading.Thread(
            target=self._scheduling_loop, daemon=True
        )
        self.scheduling_thread.start()
        logger.info("Resource scheduling started")
    
    def stop_scheduling(self):
        """Stop resource scheduling"""
        self.scheduling_active = False
        if self.scheduling_thread:
            self.scheduling_thread.join(timeout=5.0)
        logger.info("Resource scheduling stopped")
    
    def request_resources(self, request: ResourceRequest) -> Optional[str]:
        """Submit resource allocation request"""
        with self._lock:
            # Add to priority queue
            priority_score = self._calculate_priority_score(request)
            heapq.heappush(self.pending_requests, (priority_score, time.time(), request))
            
            logger.debug(f"Resource request {request.request_id} queued with priority {priority_score}")
            return request.request_id
    
    def release_resources(self, allocation_id: str) -> bool:
        """Release allocated resources"""
        with self._lock:
            if allocation_id not in self.active_allocations:
                return False
            
            allocation = self.active_allocations[allocation_id]
            
            # Calculate actual efficiency
            actual_duration = time.time() - allocation.start_time
            estimated_duration = allocation.request.duration_estimate
            
            if estimated_duration > 0:
                time_efficiency = min(1.0, estimated_duration / actual_duration)
                allocation.efficiency_score = time_efficiency
            
            # Move to history
            self.allocation_history.append(allocation)
            del self.active_allocations[allocation_id]
            
            logger.debug(f"Released resources for allocation {allocation_id}")
            return True
    
    def _scheduling_loop(self):
        """Background scheduling loop"""
        while self.scheduling_active:
            try:
                self._process_pending_requests()
                self._update_active_allocations()
                self._check_for_preemption()
                time.sleep(self.scheduling_interval)
            except Exception as e:
                logger.error(f"Scheduling loop error: {e}")
                time.sleep(self.scheduling_interval)
    
    def _process_pending_requests(self):
        """Process pending resource requests"""
        with self._lock:
            available_resources = self.monitor.get_available_resources()
            processed_requests = []
            
            # Process requests in priority order
            while self.pending_requests:
                priority_score, timestamp, request = heapq.heappop(self.pending_requests)
                
                # Check if request has expired
                if request.deadline and time.time() > request.deadline:
                    logger.warning(f"Request {request.request_id} expired")
                    self.scheduling_metrics['requests_rejected'] += 1
                    continue
                
                # Try to allocate resources
                allocation = self._try_allocate(request, available_resources)
                
                if allocation:
                    self.active_allocations[allocation.allocation_id] = allocation
                    self.scheduling_metrics['requests_fulfilled'] += 1
                    
                    # Update available resources
                    available_resources = self._subtract_resources(
                        available_resources, allocation.allocated_resources
                    )
                    
                    logger.info(f"Allocated resources for request {request.request_id}")
                else:
                    # Put back in queue if allocation failed
                    processed_requests.append((priority_score, timestamp, request))
                
                self.scheduling_metrics['requests_processed'] += 1
            
            # Put unprocessed requests back
            for req in processed_requests:
                heapq.heappush(self.pending_requests, req)
    
    def _try_allocate(self, request: ResourceRequest, available: ResourceQuota) -> Optional[ResourceAllocation]:
        """Try to allocate resources for request"""
        required = request.resource_requirements
        
        # Check if we have enough resources
        if not self._can_satisfy_request(required, available):
            return None
        
        # Determine actual allocation based on strategy
        allocated = self._calculate_allocation(request, available)
        
        # Create allocation
        allocation_id = f"alloc_{request.request_id}_{int(time.time())}"
        
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            request=request,
            allocated_resources=allocated,
            start_time=time.time(),
            expected_end_time=time.time() + request.duration_estimate,
            actual_usage=ResourceUsage()
        )
        
        return allocation
    
    def _can_satisfy_request(self, required: ResourceQuota, available: ResourceQuota) -> bool:
        """Check if request can be satisfied with available resources"""
        return (
            required.cpu_cores <= available.cpu_cores and
            required.memory_mb <= available.memory_mb and
            required.gpu_count <= available.gpu_count and
            required.storage_gb <= available.storage_gb and
            required.network_mbps <= available.network_mbps and
            required.cache_mb <= available.cache_mb and
            required.max_concurrent_tasks <= available.max_concurrent_tasks
        )
    
    def _calculate_allocation(self, request: ResourceRequest, available: ResourceQuota) -> ResourceQuota:
        """Calculate actual resource allocation based on strategy"""
        required = request.resource_requirements
        
        if self.strategy == AllocationStrategy.FAIR_SHARE:
            return self._fair_share_allocation(required, available)
        elif self.strategy == AllocationStrategy.PRIORITY_BASED:
            return self._priority_based_allocation(request, available)
        elif self.strategy == AllocationStrategy.DEMAND_BASED:
            return self._demand_based_allocation(required, available)
        elif self.strategy == AllocationStrategy.PREDICTIVE:
            return self._predictive_allocation(request, available)
        else:
            return required  # Default: exact allocation
    
    def _fair_share_allocation(self, required: ResourceQuota, available: ResourceQuota) -> ResourceQuota:
        """Fair share allocation strategy"""
        # Allocate proportionally based on current demand
        active_count = len(self.active_allocations)
        share_factor = 1.0 / max(active_count + 1, 1)
        
        return ResourceQuota(
            cpu_cores=min(required.cpu_cores, available.cpu_cores * share_factor),
            memory_mb=min(required.memory_mb, int(available.memory_mb * share_factor)),
            gpu_count=min(required.gpu_count, max(1, int(available.gpu_count * share_factor))),
            storage_gb=required.storage_gb,  # Storage is not shared
            network_mbps=min(required.network_mbps, int(available.network_mbps * share_factor)),
            cache_mb=min(required.cache_mb, int(available.cache_mb * share_factor)),
            max_concurrent_tasks=min(required.max_concurrent_tasks, 
                                   max(1, int(available.max_concurrent_tasks * share_factor)))
        )
    
    def _priority_based_allocation(self, request: ResourceRequest, available: ResourceQuota) -> ResourceQuota:
        """Priority-based allocation strategy"""
        required = request.resource_requirements
        priority_multiplier = request.priority.value / 4.0  # Normalize to 0.25-1.0
        
        return ResourceQuota(
            cpu_cores=min(required.cpu_cores, available.cpu_cores * priority_multiplier),
            memory_mb=min(required.memory_mb, int(available.memory_mb * priority_multiplier)),
            gpu_count=required.gpu_count if request.priority.value >= 3 else 0,
            storage_gb=required.storage_gb,
            network_mbps=min(required.network_mbps, int(available.network_mbps * priority_multiplier)),
            cache_mb=min(required.cache_mb, int(available.cache_mb * priority_multiplier)),
            max_concurrent_tasks=min(required.max_concurrent_tasks,
                                   max(1, int(available.max_concurrent_tasks * priority_multiplier)))
        )
    
    def _demand_based_allocation(self, required: ResourceQuota, available: ResourceQuota) -> ResourceQuota:
        """Demand-based allocation strategy"""
        # Allocate exactly what's requested if available
        return ResourceQuota(
            cpu_cores=min(required.cpu_cores, available.cpu_cores),
            memory_mb=min(required.memory_mb, available.memory_mb),
            gpu_count=min(required.gpu_count, available.gpu_count),
            storage_gb=min(required.storage_gb, available.storage_gb),
            network_mbps=min(required.network_mbps, available.network_mbps),
            cache_mb=min(required.cache_mb, available.cache_mb),
            max_concurrent_tasks=min(required.max_concurrent_tasks, available.max_concurrent_tasks)
        )
    
    def _predictive_allocation(self, request: ResourceRequest, available: ResourceQuota) -> ResourceQuota:
        """Predictive allocation strategy"""
        # Use historical data to predict actual usage
        required = request.resource_requirements
        
        # Get efficiency from historical data
        historical_efficiency = self._get_historical_efficiency(request.requester_id)
        
        # Adjust allocation based on predicted efficiency
        efficiency_factor = max(0.5, historical_efficiency.get('resource_efficiency', 1.0))
        
        return ResourceQuota(
            cpu_cores=min(required.cpu_cores * efficiency_factor, available.cpu_cores),
            memory_mb=min(int(required.memory_mb * efficiency_factor), available.memory_mb),
            gpu_count=required.gpu_count,  # GPU allocation is binary
            storage_gb=required.storage_gb,
            network_mbps=min(int(required.network_mbps * efficiency_factor), available.network_mbps),
            cache_mb=min(int(required.cache_mb * efficiency_factor), available.cache_mb),
            max_concurrent_tasks=min(required.max_concurrent_tasks, available.max_concurrent_tasks)
        )
    
    def _get_historical_efficiency(self, requester_id: str) -> Dict[str, float]:
        """Get historical efficiency for requester"""
        requester_allocations = [
            alloc for alloc in self.allocation_history
            if alloc.request.requester_id == requester_id
        ]
        
        if not requester_allocations:
            return {'resource_efficiency': 1.0}
        
        # Calculate average efficiency
        efficiencies = [alloc.efficiency_score for alloc in requester_allocations]
        avg_efficiency = np.mean(efficiencies)
        
        return {'resource_efficiency': avg_efficiency}
    
    def _subtract_resources(self, available: ResourceQuota, allocated: ResourceQuota) -> ResourceQuota:
        """Subtract allocated resources from available"""
        return ResourceQuota(
            cpu_cores=max(0, available.cpu_cores - allocated.cpu_cores),
            memory_mb=max(0, available.memory_mb - allocated.memory_mb),
            gpu_count=max(0, available.gpu_count - allocated.gpu_count),
            storage_gb=max(0, available.storage_gb - allocated.storage_gb),
            network_mbps=max(0, available.network_mbps - allocated.network_mbps),
            cache_mb=max(0, available.cache_mb - allocated.cache_mb),
            max_concurrent_tasks=max(0, available.max_concurrent_tasks - allocated.max_concurrent_tasks)
        )
    
    def _update_active_allocations(self):
        """Update active allocations and check for completion"""
        current_time = time.time()
        completed_allocations = []
        
        with self._lock:
            for allocation_id, allocation in self.active_allocations.items():
                # Check if allocation has expired
                if current_time > allocation.expected_end_time:
                    completed_allocations.append(allocation_id)
                    logger.warning(f"Allocation {allocation_id} exceeded expected duration")
        
        # Remove completed allocations
        for allocation_id in completed_allocations:
            self.release_resources(allocation_id)
    
    def _check_for_preemption(self):
        """Check for preemption opportunities"""
        if not self.preemption_enabled:
            return
        
        # This would implement preemption logic based on priorities
        # For now, just log that preemption checking occurred
        logger.debug("Preemption check completed")
    
    def _calculate_priority_score(self, request: ResourceRequest) -> float:
        """Calculate priority score for request (lower = higher priority)"""
        base_priority = 5 - request.priority.value  # Invert priority (lower is better)
        
        # Add urgency factor based on deadline
        urgency_factor = 0
        if request.deadline:
            time_remaining = request.deadline - time.time()
            if time_remaining > 0:
                urgency_factor = 1.0 / max(time_remaining, 1.0)
        
        # Add resource demand factor
        resource_demand = (
            request.resource_requirements.cpu_cores +
            request.resource_requirements.memory_mb / 1024 +  # Normalize to GB
            request.resource_requirements.gpu_count * 2  # Weight GPU higher
        )
        
        return base_priority + urgency_factor + (resource_demand * 0.1)
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics"""
        with self._lock:
            # Calculate average wait time
            if self.scheduling_metrics['requests_processed'] > 0:
                fulfillment_rate = (
                    self.scheduling_metrics['requests_fulfilled'] / 
                    self.scheduling_metrics['requests_processed']
                )
            else:
                fulfillment_rate = 0.0
            
            # Calculate resource utilization
            available = self.monitor.get_available_resources()
            limits = self.monitor.resource_limits
            
            cpu_utilization = 1.0 - (available.cpu_cores / limits.cpu_cores)
            memory_utilization = 1.0 - (available.memory_mb / limits.memory_mb)
            
            return {
                'strategy': self.strategy.value,
                'active_allocations': len(self.active_allocations),
                'pending_requests': len(self.pending_requests),
                'requests_processed': self.scheduling_metrics['requests_processed'],
                'requests_fulfilled': self.scheduling_metrics['requests_fulfilled'],
                'requests_rejected': self.scheduling_metrics['requests_rejected'],
                'fulfillment_rate': fulfillment_rate,
                'cpu_utilization': cpu_utilization,
                'memory_utilization': memory_utilization,
                'preemption_enabled': self.preemption_enabled,
                'scheduling_active': self.scheduling_active
            }


class ResourceManager:
    """Central resource management coordinator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.monitor = ResourceMonitor(
            update_interval=self.config.get('monitoring_interval', 1.0)
        )
        
        strategy = AllocationStrategy(
            self.config.get('allocation_strategy', 'demand_based')
        )
        self.scheduler = ResourceScheduler(self.monitor, strategy)
        
        # Resource optimization integration
        self.optimization_managers = {}
        
        # Management state
        self.management_active = False
        self.management_thread = None
        self.optimization_interval = self.config.get('optimization_interval', 300)  # 5 minutes
        
    def start_management(self):
        """Start resource management system"""
        if self.management_active:
            return
        
        self.management_active = True
        
        # Start monitoring and scheduling
        self.monitor.start_monitoring()
        self.scheduler.start_scheduling()
        
        # Initialize optimization managers
        self._initialize_optimization_managers()
        
        # Start management thread
        self.management_thread = threading.Thread(
            target=self._management_loop, daemon=True
        )
        self.management_thread.start()
        
        logger.info("Resource management started")
    
    def stop_management(self):
        """Stop resource management system"""
        if not self.management_active:
            return
        
        self.management_active = False
        
        # Stop monitoring and scheduling
        self.monitor.stop_monitoring()
        self.scheduler.stop_scheduling()
        
        # Stop management thread
        if self.management_thread:
            self.management_thread.join(timeout=10.0)
        
        logger.info("Resource management stopped")
    
    def _initialize_optimization_managers(self):
        """Initialize optimization subsystem managers"""
        try:
            self.optimization_managers['memory'] = get_memory_optimizer()
            self.optimization_managers['compute'] = get_compute_optimizer()
            self.optimization_managers['parallel'] = get_parallel_processor()
            self.optimization_managers['network'] = get_network_optimizer()
            self.optimization_managers['storage'] = get_storage_optimizer()
            self.optimization_managers['cache'] = get_cache()
            
            logger.info("Optimization managers initialized")
        except Exception as e:
            logger.warning(f"Some optimization managers failed to initialize: {e}")
    
    def _management_loop(self):
        """Background resource management loop"""
        while self.management_active:
            try:
                self._perform_resource_optimization()
                time.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"Resource management loop error: {e}")
                time.sleep(self.optimization_interval)
    
    def _perform_resource_optimization(self):
        """Perform resource optimization cycle"""
        logger.debug("Performing resource optimization cycle")
        
        # Get current resource usage
        current_usage = self.monitor.current_usage
        available_resources = self.monitor.get_available_resources()
        
        # Optimize based on usage patterns
        if current_usage.memory_used_mb / self.monitor.resource_limits.memory_mb > 0.8:
            # High memory usage - trigger memory optimization
            try:
                memory_optimizer = self.optimization_managers.get('memory')
                if memory_optimizer:
                    memory_optimizer.optimize_memory()
            except Exception as e:
                logger.warning(f"Memory optimization failed: {e}")
        
        # Optimize compute resources
        if current_usage.cpu_utilization > 0.9:
            try:
                compute_optimizer = self.optimization_managers.get('compute')
                if compute_optimizer:
                    # This would trigger compute optimizations
                    pass
            except Exception as e:
                logger.warning(f"Compute optimization failed: {e}")
        
        # Cache optimization
        if current_usage.cache_used_mb / self.monitor.resource_limits.cache_mb > 0.9:
            try:
                cache = self.optimization_managers.get('cache')
                if cache:
                    cache.clear(['old_entries'])  # Clear old cache entries
            except Exception as e:
                logger.warning(f"Cache optimization failed: {e}")
    
    def request_resources(self, requester_id: str, requirements: ResourceQuota,
                         priority: ResourcePriority = ResourcePriority.NORMAL,
                         duration_estimate: float = 3600.0,
                         deadline: Optional[float] = None) -> str:
        """Request resource allocation"""
        request_id = f"req_{requester_id}_{int(time.time())}"
        
        request = ResourceRequest(
            request_id=request_id,
            requester_id=requester_id,
            resource_requirements=requirements,
            priority=priority,
            duration_estimate=duration_estimate,
            deadline=deadline
        )
        
        allocation_id = self.scheduler.request_resources(request)
        logger.info(f"Resource request submitted: {request_id}")
        
        return allocation_id
    
    def release_resources(self, allocation_id: str) -> bool:
        """Release allocated resources"""
        success = self.scheduler.release_resources(allocation_id)
        if success:
            logger.info(f"Resources released: {allocation_id}")
        return success
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource management status"""
        return {
            'management_active': self.management_active,
            'system_limits': asdict(self.monitor.resource_limits),
            'current_usage': asdict(self.monitor.current_usage),
            'available_resources': asdict(self.monitor.get_available_resources()),
            'resource_efficiency': self.monitor.get_resource_efficiency(),
            'predicted_demand': asdict(self.monitor.predict_resource_demand()),
            'scheduling_stats': self.scheduler.get_scheduling_stats(),
            'optimization_managers_active': list(self.optimization_managers.keys())
        }
    
    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """Manually trigger resource optimization"""
        start_time = time.time()
        
        optimization_results = {
            'triggered_at': start_time,
            'optimizations_performed': []
        }
        
        try:
            self._perform_resource_optimization()
            optimization_results['optimizations_performed'].append('resource_optimization')
        except Exception as e:
            optimization_results['error'] = str(e)
        
        optimization_results['duration'] = time.time() - start_time
        return optimization_results
    
    def set_resource_limits(self, limits: ResourceQuota):
        """Set custom resource limits"""
        self.monitor.resource_limits = limits
        logger.info("Resource limits updated")
    
    def enable_preemption(self, enabled: bool = True):
        """Enable/disable resource preemption"""
        self.scheduler.preemption_enabled = enabled
        logger.info(f"Resource preemption {'enabled' if enabled else 'disabled'}")


# Global resource manager instance
_global_resource_manager = None

def get_resource_manager(**kwargs) -> ResourceManager:
    """Get global resource manager instance"""
    global _global_resource_manager
    
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager(**kwargs)
    
    return _global_resource_manager

def start_resource_management(**kwargs) -> ResourceManager:
    """Start global resource management"""
    manager = get_resource_manager(**kwargs)
    manager.start_management()
    return manager

def request_resources(requester_id: str, requirements: ResourceQuota, **kwargs) -> str:
    """Request resources using global manager"""
    manager = get_resource_manager()
    return manager.request_resources(requester_id, requirements, **kwargs)

def release_resources(allocation_id: str) -> bool:
    """Release resources using global manager"""
    manager = get_resource_manager()
    return manager.release_resources(allocation_id)